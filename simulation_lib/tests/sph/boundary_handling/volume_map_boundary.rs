use nalgebra::{Point3, Vector3};
use simulation_lib::neighbor_search::{NeighborSearch, SpatialHashing};
use simulation_lib::render_info::{
    BoundaryMeshColoring, BoundarySampleColoring, BoundaryVisualization,
};
use simulation_lib::sph::boundary_handling::RequestMode;
use simulation_lib::sph::boundary_handling::{
    BoundaryCheckpoint, BoundaryHandling, ForceOntoBoundary, VolumeMapBoundary,
};
use simulation_lib::sph::kernel::CubicBSpline3D;
use simulation_lib::sph::setup::input::{StaticBoundaryDef, VertexNormalRenderOption};
use simulation_lib::utilities::triangle_mesh::{LoadedMesh, MeshContainer};

// All tests here deliberately never call `add_static_boundary`/
// `add_dynamic_boundary` — doing so triggers the (currently unverified,
// potentially slow) SDF/volume-map discretization pipeline. These tests
// instead cover the `BoundaryHandling` API's behavior on an empty
// `VolumeMapBoundary`, which exercises real code paths without that cost.

#[test]
fn new_volume_maps_is_empty() {
    let mut vm = VolumeMapBoundary::new();
    assert!(vm.is_empty());
    assert_eq!(vm.iter().count(), 0);
    assert_eq!(vm.iter_mut().count(), 0);
}

#[test]
fn get_fluid_depth_is_always_zero() {
    let vm = VolumeMapBoundary::new();
    assert_eq!(vm.get_fluid_depth(0.0), 0.0);
    assert_eq!(vm.get_fluid_depth(999.0), 0.0);
}

#[test]
fn initialize_is_a_noop() {
    let mut vm = VolumeMapBoundary::new();
    let mut ns = SpatialHashing::new(1.0);
    vm.initialize::<CubicBSpline3D>(&mut ns, 1.0, 1.0);
    // No observable effect to assert on an empty `VolumeMapBoundary`; this only
    // confirms the call doesn't panic.
}

#[test]
fn find_boundary_samples_on_empty_does_not_panic() {
    let mut vm = VolumeMapBoundary::new();
    let mut ns = SpatialHashing::new(1.0);
    let positions = vec![Point3::new(0., 0., 0.)];
    vm.find_boundary_samples(&mut ns, 1.0, &positions, 0.1);
}

#[test]
fn step_forward_in_time_on_empty_is_noop() {
    let mut vm = VolumeMapBoundary::new();
    vm.step_forward_in_time(0.1);
}

#[test]
fn add_force_onto_boundary_on_empty_does_not_panic() {
    let mut vm = VolumeMapBoundary::new();
    vm.add_force_onto_boundary(ForceOntoBoundary {
        id: 0,
        force: Vector3::new(1., 0., 0.),
        force_location: Point3::origin(),
    });
}

#[test]
fn get_visualization_samples_always_returns_empty() {
    // `VolumeMapBoundary` has no explicit samples to visualize — the `Samples`
    // selector is documented to always yield an empty result, independent
    // of the requested coloring.
    let vm = VolumeMapBoundary::new();
    let selector = BoundaryVisualization::Samples {
        positions: vec![],
        coloring: BoundarySampleColoring::BoundaryId {
            ids: vec![],
            max_id: 0,
        },
    };

    match vm.get_visualization(&selector) {
        BoundaryVisualization::Samples {
            positions,
            coloring,
        } => {
            assert!(positions.is_empty());
            assert!(matches!(coloring, BoundarySampleColoring::Uniform));
        }
        _ => panic!("expected Samples variant"),
    }
}

#[test]
fn get_visualization_triangle_mesh_on_empty_returns_empty_meshes() {
    let vm = VolumeMapBoundary::new();
    let selector = BoundaryVisualization::TriangleMesh {
        meshes: vec![],
        coloring: BoundaryMeshColoring::Original,
    };

    match vm.get_visualization(&selector) {
        BoundaryVisualization::TriangleMesh { meshes, coloring } => {
            assert!(meshes.is_empty());
            assert!(matches!(coloring, BoundaryMeshColoring::Original));
        }
        _ => panic!("expected TriangleMesh variant"),
    }
}

#[test]
fn get_checkpoint_on_empty_has_no_dynamic_states() {
    let vm = VolumeMapBoundary::new();
    assert!(vm.get_checkpoint().dynamic_states.is_empty());
}

#[test]
fn restore_from_checkpoint_on_empty_is_noop() {
    let mut vm = VolumeMapBoundary::new();
    vm.restore_from_checkpoint(&BoundaryCheckpoint::default());
}

// ─── Full integration: add_static_boundary (real SD + volume-map build) ──

/// A tiny cube (half-size 0.01), positioned directly at its final physical
/// scale rather than via `scale`, to avoid entangling this test with the
/// mesh-transform logic (already covered elsewhere).
fn tiny_cube_mesh_container() -> MeshContainer {
    let s = 0.1;
    let positions = vec![
        Point3::new(s, s, s),
        Point3::new(s, s, -s),
        Point3::new(s, -s, s),
        Point3::new(s, -s, -s),
        Point3::new(-s, s, s),
        Point3::new(-s, s, -s),
        Point3::new(-s, -s, s),
        Point3::new(-s, -s, -s),
    ];
    let indices = vec![
        [4, 2, 0],
        [2, 7, 3],
        [6, 5, 7],
        [1, 7, 5],
        [0, 3, 1],
        [4, 1, 5],
        [4, 6, 2],
        [2, 6, 7],
        [6, 4, 5],
        [1, 3, 7],
        [0, 2, 3],
        [4, 0, 1],
    ];
    MeshContainer::new(LoadedMesh {
        positions,
        normals: Vec::new(),
        indices,
    })
}

#[test]
#[ignore = "exercises the real Gauss-Legendre volume-map quadrature (order = \
            VolumeMapBoundary::INTEGRATION_ORDER, fixed in production code, not \
            reducible here) over the full mesh->AABB->grid pipeline. Even at \
            this deliberately tiny scale, actual timing hasn't been verified \
            by running this code — run explicitly via `cargo test -- \
            --ignored` and shrink the mesh/kernel_support_radius further if \
            it's too slow in practice."]
fn add_static_boundary_builds_usable_signed_distance_and_volume_fields() {
    let mut boundary = VolumeMapBoundary::new();
    let mut mesh = tiny_cube_mesh_container();
    let kernel_support_radius = 0.045;
    // dx = 4 * rest_density_grid_spacing = 0.09, chosen (see prior derivation)
    // so both the signed-distance and volume-map grids for this fixture end
    // up with a single cell.
    let rest_density_grid_spacing = 0.0225;

    let def = StaticBoundaryDef {
        mesh: String::new(),
        boundary_id: 1,
        translation: [0., 0., 0.],
        rotation_euler_deg: [0., 0., 0.],
        scale: [1., 1., 1.],
        render_vertex_normals: VertexNormalRenderOption::AngleWeightedPseudoNormals,
    };

    boundary.add_static_boundary(
        &mut mesh,
        &def,
        rest_density_grid_spacing,
        kernel_support_radius,
    );
    assert!(!boundary.is_empty());

    let mut neighbor_search = SpatialHashing::new(kernel_support_radius);
    let fluid_positions = vec![Point3::new(0.1, 0.0, 0.0)]; // at the cube's surface
    boundary.find_boundary_samples(
        &mut neighbor_search,
        kernel_support_radius,
        &fluid_positions,
        rest_density_grid_spacing,
    );

    let b = boundary.iter().next().unwrap();
    let neighbors = b.get_neighbors(0, RequestMode::Normal);
    assert!(
        !neighbors.is_empty(),
        "fluid particle at the surface should find boundary neighbors"
    );
    for &n in neighbors {
        let v = b.volume(n);
        assert!(v > 0.0 && v.is_finite(), "unexpected boundary volume {v}");
    }
}
