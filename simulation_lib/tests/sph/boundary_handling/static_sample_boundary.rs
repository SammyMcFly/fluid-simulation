use nalgebra::{Point3, Vector3};
use simulation_lib::neighbor_search::NeighborSearch;
use simulation_lib::neighbor_search::SpatialHashing;
use simulation_lib::render_info::{
    BoundaryMeshColoring, BoundarySampleColoring, BoundaryVisualization,
};
use simulation_lib::sph::boundary_handling::{
    BoundaryCheckpoint, BoundaryHandling, ForceOntoBoundary, RequestMode, StaticSampleBoundary,
};
use simulation_lib::sph::kernel::CubicBSpline3D;
use simulation_lib::sph::setup::input::{
    DynamicBoundaryDef, StaticBoundaryDef, VertexNormalRenderOption,
};
use simulation_lib::utilities::sampling::sample_triangle_mesh_surface;
use simulation_lib::utilities::triangle_mesh::{LoadedMesh, MeshContainer};

const SPACING: f64 = 0.5;
const KERNEL_SUPPORT_RADIUS: f64 = 1.2;
const WEIGHTING: f64 = 1.0;

/// A cube of side length 2, centered at the origin, with outward-facing
/// triangle winding — matches the fixture used elsewhere in this project
/// (`cube_face_normals_outwards.obj`).
fn cube_mesh_container() -> MeshContainer {
    let positions = vec![
        Point3::new(1., 1., 1.),
        Point3::new(1., 1., -1.),
        Point3::new(1., -1., 1.),
        Point3::new(1., -1., -1.),
        Point3::new(-1., 1., 1.),
        Point3::new(-1., 1., -1.),
        Point3::new(-1., -1., 1.),
        Point3::new(-1., -1., -1.),
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

fn static_def(boundary_id: u32) -> StaticBoundaryDef {
    StaticBoundaryDef {
        mesh: String::new(),
        boundary_id,
        translation: [0., 0., 0.],
        rotation_euler_deg: [0., 0., 0.],
        scale: [1., 1., 1.],
        render_vertex_normals: VertexNormalRenderOption::AngleWeightedPseudoNormals,
    }
}

fn dynamic_def(boundary_id: u32, translation: [f64; 3]) -> DynamicBoundaryDef {
    DynamicBoundaryDef {
        mesh: String::new(),
        boundary_id,
        density: 1000.0,
        translation,
        rotation_euler_deg: [0., 0., 0.],
        velocity: [0., 0., 0.],
        angular_velocity: [0., 0., 0.],
        scale: [1., 1., 1.],
        render_vertex_normals: VertexNormalRenderOption::AngleWeightedPseudoNormals,
    }
}

/// Like [`dynamic_def`], but with explicit initial linear/angular velocity —
/// needed by tests that check motion resulting from a nonzero initial
/// velocity rather than from an externally applied force.
fn dynamic_def_moving(
    boundary_id: u32,
    translation: [f64; 3],
    velocity: [f64; 3],
    angular_velocity: [f64; 3],
) -> DynamicBoundaryDef {
    DynamicBoundaryDef {
        mesh: String::new(),
        boundary_id,
        density: 1000.0,
        translation,
        rotation_euler_deg: [0., 0., 0.],
        velocity,
        angular_velocity,
        scale: [1., 1., 1.],
        render_vertex_normals: VertexNormalRenderOption::AngleWeightedPseudoNormals,
    }
}

/// Number of surface samples `sample_triangle_mesh_surface` produces for a
/// fresh cube at `SPACING` — computed independently via the same public
/// function used internally, since `Boundary` deliberately exposes no
/// `len()` (production code only ever indexes it through already-bounded
/// neighbor list indices, never queries a boundary's own sample count).
fn expected_cube_sample_count() -> usize {
    let mut mesh = cube_mesh_container();
    sample_triangle_mesh_surface(mesh.trimesh(), SPACING).len()
}

// ─── construction / is_empty ────────────────────────────────────────

#[test]
fn new_sample_boundary_is_empty() {
    let boundary = StaticSampleBoundary::new();
    assert!(boundary.is_empty());
}

#[test]
fn add_static_boundary_makes_it_non_empty() {
    let mut boundary = StaticSampleBoundary::new();
    let mut mesh = cube_mesh_container();
    boundary.add_static_boundary(&mut mesh, &static_def(1), SPACING, KERNEL_SUPPORT_RADIUS);

    assert!(!boundary.is_empty());
    assert_eq!(boundary.iter().count(), 1);
}

#[test]
fn add_static_boundary_populates_and_is_not_dynamic() {
    let mut boundary = StaticSampleBoundary::new();
    let mut mesh = cube_mesh_container();
    boundary.add_static_boundary(&mut mesh, &static_def(0), SPACING, KERNEL_SUPPORT_RADIUS);

    assert!(!boundary.is_empty());
    let b = boundary.iter().next().unwrap();
    assert!(!b.is_dynamic());
    assert!(b.center_of_mass().is_none());
}

#[test]
fn add_dynamic_boundary_makes_it_non_empty() {
    let mut boundary = StaticSampleBoundary::new();
    let mut mesh = cube_mesh_container();
    boundary.add_dynamic_boundary(
        &mut mesh,
        &dynamic_def(1, [0., 0., 0.]),
        SPACING,
        KERNEL_SUPPORT_RADIUS,
    );

    assert!(!boundary.is_empty());
    assert_eq!(boundary.iter().count(), 1);
    assert!(boundary.iter().next().unwrap().is_dynamic());
}

#[test]
fn add_dynamic_boundary_is_dynamic_with_center_of_mass_near_translation() {
    let mut boundary = StaticSampleBoundary::new();
    let mut mesh = cube_mesh_container();
    let translation = [5.0, 0.0, 0.0];
    boundary.add_dynamic_boundary(
        &mut mesh,
        &dynamic_def(0, translation),
        SPACING,
        KERNEL_SUPPORT_RADIUS,
    );

    let b = boundary.iter().next().unwrap();
    assert!(b.is_dynamic());
    let com = b
        .center_of_mass()
        .expect("dynamic boundary must report a center of mass");
    // A cube centered at its own OBJ origin has local_com ≈ (0,0,0), so the
    // global center of mass should end up close to the translation alone.
    assert!((com - Point3::new(5.0, 0.0, 0.0)).norm() < 1e-6);
}

#[test]
fn dynamic_boundary_sample_positions_start_as_origin_placeholders_before_initialize() {
    // Per the struct's construction (`position: vec![Point3::origin(); len]`),
    // world-space sample positions are only filled in once `initialize` (or
    // `step_forward_in_time`) runs `update_positions_and_velocities`.
    let mut boundary = StaticSampleBoundary::new();
    let mut mesh = cube_mesh_container();
    boundary.add_dynamic_boundary(
        &mut mesh,
        &dynamic_def(0, [5.0, 0.0, 0.0]),
        SPACING,
        KERNEL_SUPPORT_RADIUS,
    );

    let b = boundary.iter().next().unwrap();
    assert_eq!(*b.position(0), Point3::origin());
}

// ─── samples lie on the cube surface ────────────────────────────────

#[test]
fn static_boundary_samples_lie_on_cube_surface() {
    let mut boundary = StaticSampleBoundary::new();
    let mut mesh = cube_mesh_container();
    boundary.add_static_boundary(&mut mesh, &static_def(1), SPACING, KERNEL_SUPPORT_RADIUS);

    let b = boundary.iter().next().unwrap();
    for id in 0..expected_cube_sample_count() {
        let p = b.position(id);
        let max_coord = p.x.abs().max(p.y.abs()).max(p.z.abs());
        assert!(
            max_coord <= 1.0 + 1e-9,
            "sample {id} = {p:?} is outside the cube"
        );
        assert!(
            max_coord >= 1.0 - 1e-9,
            "sample {id} = {p:?} is not on the cube surface"
        );
    }
}

#[test]
fn dynamic_boundary_samples_are_offset_by_translation() {
    let translation = [10., 0., 0.];
    let mut boundary = StaticSampleBoundary::new();
    let mut mesh = cube_mesh_container();
    boundary.add_dynamic_boundary(
        &mut mesh,
        &dynamic_def(1, translation),
        SPACING,
        KERNEL_SUPPORT_RADIUS,
    );
    boundary.initialize::<CubicBSpline3D>(
        &mut SpatialHashing::new(KERNEL_SUPPORT_RADIUS),
        KERNEL_SUPPORT_RADIUS,
        WEIGHTING,
    );

    let b = boundary.iter().next().unwrap();
    for id in 0..expected_cube_sample_count() {
        let p = b.position(id);
        assert!(
            p.x >= translation[0] - 1.0 - 1e-6 && p.x <= translation[0] + 1.0 + 1e-6,
            "sample {id} = {p:?} was not translated correctly"
        );
    }
}

// ─── initialize: world-space positions and pseudo-volume ────────────

#[test]
fn initialize_moves_dynamic_boundary_samples_into_world_space() {
    let mut boundary = StaticSampleBoundary::new();
    let mut mesh = cube_mesh_container();
    boundary.add_dynamic_boundary(
        &mut mesh,
        &dynamic_def(0, [5.0, 0.0, 0.0]),
        SPACING,
        KERNEL_SUPPORT_RADIUS,
    );

    boundary.initialize::<CubicBSpline3D>(
        &mut SpatialHashing::new(KERNEL_SUPPORT_RADIUS),
        KERNEL_SUPPORT_RADIUS,
        WEIGHTING,
    );

    let b = boundary.iter().next().unwrap();
    // Cube half-size is 1.0, translated by 5.0 along x -> every sample must
    // now lie far from the origin, not at the (0,0,0) placeholder.
    assert!(b.position(0).coords.norm() > 3.0);
}

#[test]
fn initialize_assigns_positive_finite_volume_to_every_sample() {
    let mut boundary = StaticSampleBoundary::new();
    let mut mesh = cube_mesh_container();
    boundary.add_static_boundary(&mut mesh, &static_def(1), SPACING, KERNEL_SUPPORT_RADIUS);

    boundary.initialize::<CubicBSpline3D>(
        &mut SpatialHashing::new(KERNEL_SUPPORT_RADIUS),
        KERNEL_SUPPORT_RADIUS,
        WEIGHTING,
    );

    let b = boundary.iter().next().unwrap();
    for id in 0..expected_cube_sample_count() {
        let v = b.volume(id);
        assert!(
            v > 0.0 && v.is_finite(),
            "sample {id} has invalid volume {v}"
        );
    }
}

// ─── find_boundary_samples ────────────────────────────────────────────

#[test]
fn find_boundary_samples_finds_nearby_fluid_particle() {
    let mut boundary = StaticSampleBoundary::new();
    let mut mesh = cube_mesh_container();
    boundary.add_static_boundary(&mut mesh, &static_def(1), SPACING, KERNEL_SUPPORT_RADIUS);
    boundary.initialize::<CubicBSpline3D>(
        &mut SpatialHashing::new(KERNEL_SUPPORT_RADIUS),
        KERNEL_SUPPORT_RADIUS,
        WEIGHTING,
    );

    let fluid_positions = vec![Point3::new(1.0, 0.0, 0.0)];
    boundary.find_boundary_samples(
        &mut SpatialHashing::new(KERNEL_SUPPORT_RADIUS),
        KERNEL_SUPPORT_RADIUS,
        &fluid_positions,
        SPACING,
    );

    let b = boundary.iter().next().unwrap();
    assert!(
        !b.get_neighbors(0, RequestMode::Normal).is_empty(),
        "fluid particle at the cube surface should have found boundary neighbors"
    );
}

#[test]
fn find_boundary_samples_finds_none_for_distant_fluid_particle() {
    let mut boundary = StaticSampleBoundary::new();
    let mut mesh = cube_mesh_container();
    boundary.add_static_boundary(&mut mesh, &static_def(1), SPACING, KERNEL_SUPPORT_RADIUS);
    boundary.initialize::<CubicBSpline3D>(
        &mut SpatialHashing::new(KERNEL_SUPPORT_RADIUS),
        KERNEL_SUPPORT_RADIUS,
        WEIGHTING,
    );

    let fluid_positions = vec![Point3::new(1000.0, 0.0, 0.0)];
    boundary.find_boundary_samples(
        &mut SpatialHashing::new(KERNEL_SUPPORT_RADIUS),
        KERNEL_SUPPORT_RADIUS,
        &fluid_positions,
        SPACING,
    );

    let b = boundary.iter().next().unwrap();
    assert!(b.get_neighbors(0, RequestMode::Normal).is_empty());
}

// ─── force / step_forward_in_time (dynamic) ──────────────────────────

#[test]
fn force_onto_boundary_moves_dynamic_boundary_center_of_mass() {
    let mut boundary = StaticSampleBoundary::new();
    let mut mesh = cube_mesh_container();
    boundary.add_dynamic_boundary(
        &mut mesh,
        &dynamic_def(1, [0., 0., 0.]),
        SPACING,
        KERNEL_SUPPORT_RADIUS,
    );
    boundary.initialize::<CubicBSpline3D>(
        &mut SpatialHashing::new(KERNEL_SUPPORT_RADIUS),
        KERNEL_SUPPORT_RADIUS,
        WEIGHTING,
    );

    let com_before = boundary.iter().next().unwrap().center_of_mass().unwrap();

    boundary.add_force_onto_boundary(ForceOntoBoundary {
        id: 0,
        force: Vector3::new(1000.0, 0.0, 0.0),
        force_location: com_before,
    });
    boundary.step_forward_in_time(0.1);

    let com_after = boundary.iter().next().unwrap().center_of_mass().unwrap();
    assert!(
        com_after.x > com_before.x,
        "boundary should have moved in +x"
    );
}

#[test]
fn add_force_onto_boundary_accelerates_a_dynamic_boundary() {
    // Complements `force_onto_boundary_moves_dynamic_boundary_center_of_mass`
    // (which checks the resulting POSITION change) by checking the
    // resulting VELOCITY change directly, with the force applied exactly at
    // the center of mass (zero torque -> pure translation, no rotation to
    // account for).
    let mut boundary = StaticSampleBoundary::new();
    let mut mesh = cube_mesh_container();
    boundary.add_dynamic_boundary(
        &mut mesh,
        &dynamic_def(1, [0., 0., 0.]),
        SPACING,
        KERNEL_SUPPORT_RADIUS,
    );
    boundary.step_forward_in_time(0.0); // sync cached position/velocity

    let com = boundary.iter().next().unwrap().center_of_mass().unwrap();
    boundary.add_force_onto_boundary(ForceOntoBoundary {
        id: 0,
        force: Vector3::new(0.0, -100.0, 0.0),
        force_location: com,
    });
    boundary.step_forward_in_time(1.0);

    let b = boundary.iter().next().unwrap();
    assert!(
        b.velocity(0).y < 0.0,
        "expected downward velocity after a downward force, got {:?}",
        b.velocity(0)
    );
}

#[test]
fn add_force_onto_boundary_ignores_out_of_range_id() {
    let mut boundary = StaticSampleBoundary::new();
    let mut mesh = cube_mesh_container();
    boundary.add_static_boundary(&mut mesh, &static_def(1), SPACING, KERNEL_SUPPORT_RADIUS);

    boundary.add_force_onto_boundary(ForceOntoBoundary {
        id: 5,
        force: Vector3::new(1., 0., 0.),
        force_location: Point3::origin(),
    });
}

#[test]
fn step_forward_in_time_is_noop_for_static_boundary() {
    let mut boundary = StaticSampleBoundary::new();
    let mut mesh = cube_mesh_container();
    boundary.add_static_boundary(&mut mesh, &static_def(1), SPACING, KERNEL_SUPPORT_RADIUS);
    boundary.initialize::<CubicBSpline3D>(
        &mut SpatialHashing::new(KERNEL_SUPPORT_RADIUS),
        KERNEL_SUPPORT_RADIUS,
        WEIGHTING,
    );

    let before: Vec<Point3<f64>> = (0..expected_cube_sample_count())
        .map(|id| *boundary.iter().next().unwrap().position(id))
        .collect();

    boundary.step_forward_in_time(1.0);

    for (id, p) in before.iter().enumerate() {
        assert_eq!(boundary.iter().next().unwrap().position(id), p);
    }
}

#[test]
fn step_forward_in_time_translates_a_moving_dynamic_boundary() {
    // Unlike the force-based tests above, this checks motion arising
    // purely from a nonzero INITIAL velocity, with no force ever applied.
    let mut boundary = StaticSampleBoundary::new();
    let mut mesh = cube_mesh_container();
    boundary.add_dynamic_boundary(
        &mut mesh,
        &dynamic_def_moving(1, [0., 0., 0.], [1., 0., 0.], [0., 0., 0.]),
        SPACING,
        KERNEL_SUPPORT_RADIUS,
    );
    boundary.step_forward_in_time(0.0); // sync cached position with initial state

    let before = *boundary.iter().next().unwrap().position(0);
    boundary.step_forward_in_time(1.0);
    let after = *boundary.iter().next().unwrap().position(0);

    assert!(
        (after - before - Vector3::new(1.0, 0.0, 0.0)).norm() < 1e-9,
        "expected the boundary to translate by (1,0,0) over 1 second at velocity (1,0,0), got a delta of {:?}",
        after - before
    );
}

// ─── visualization ────────────────────────────────────────────────────

#[test]
fn get_visualization_samples_uniform() {
    let mut boundary = StaticSampleBoundary::new();
    let mut mesh = cube_mesh_container();
    boundary.add_static_boundary(&mut mesh, &static_def(1), SPACING, KERNEL_SUPPORT_RADIUS);

    let selector = BoundaryVisualization::Samples {
        positions: vec![],
        coloring: BoundarySampleColoring::Uniform,
    };
    let result = boundary.get_visualization(&selector);

    match result {
        BoundaryVisualization::Samples {
            positions,
            coloring,
        } => {
            assert_eq!(positions.len(), expected_cube_sample_count());
            assert!(matches!(coloring, BoundarySampleColoring::Uniform));
        }
        _ => panic!("expected Samples variant"),
    }
}

#[test]
fn get_visualization_samples_boundary_id_assigns_max_id() {
    let mut boundary = StaticSampleBoundary::new();
    let mut mesh_a = cube_mesh_container();
    boundary.add_static_boundary(&mut mesh_a, &static_def(3), SPACING, KERNEL_SUPPORT_RADIUS);
    let mut mesh_b = cube_mesh_container();
    boundary.add_static_boundary(&mut mesh_b, &static_def(7), SPACING, KERNEL_SUPPORT_RADIUS);

    let selector = BoundaryVisualization::Samples {
        positions: vec![],
        coloring: BoundarySampleColoring::BoundaryId {
            ids: vec![],
            max_id: 0,
        },
    };
    let result = boundary.get_visualization(&selector);

    match result {
        BoundaryVisualization::Samples { coloring, .. } => match coloring {
            BoundarySampleColoring::BoundaryId { max_id, ids } => {
                assert_eq!(max_id, 7);
                assert!(ids.iter().all(|&id| id == 3 || id == 7));
            }
            _ => panic!("expected BoundaryId coloring"),
        },
        _ => panic!("expected Samples variant"),
    }
}

#[test]
fn get_visualization_triangle_mesh_returns_one_mesh_per_boundary() {
    let mut boundary = StaticSampleBoundary::new();
    let mut mesh_a = cube_mesh_container();
    boundary.add_static_boundary(&mut mesh_a, &static_def(1), SPACING, KERNEL_SUPPORT_RADIUS);
    let mut mesh_b = cube_mesh_container();
    boundary.add_dynamic_boundary(
        &mut mesh_b,
        &dynamic_def(2, [5., 0., 0.]),
        SPACING,
        KERNEL_SUPPORT_RADIUS,
    );

    let selector = BoundaryVisualization::TriangleMesh {
        meshes: vec![],
        coloring: BoundaryMeshColoring::Original,
    };
    let result = boundary.get_visualization(&selector);

    match result {
        BoundaryVisualization::TriangleMesh { meshes, coloring } => {
            assert_eq!(meshes.len(), 2);
            assert!(matches!(coloring, BoundaryMeshColoring::Original));
        }
        _ => panic!("expected TriangleMesh variant"),
    }
}

#[test]
fn get_visualization_triangle_mesh_boundary_id_coloring_reports_the_configured_id() {
    // Complements `get_visualization_samples_boundary_id_assigns_max_id`
    // (which covers `BoundaryId` coloring for the `Samples` visualization)
    // by covering the same coloring variant for `TriangleMesh`.
    let mut boundary = StaticSampleBoundary::new();
    let mut mesh = cube_mesh_container();
    boundary.add_static_boundary(&mut mesh, &static_def(7), SPACING, KERNEL_SUPPORT_RADIUS);

    let selector = BoundaryVisualization::TriangleMesh {
        meshes: vec![],
        coloring: BoundaryMeshColoring::BoundaryId {
            ids: vec![],
            max_id: 0,
        },
    };

    let result = boundary.get_visualization(&selector);
    match result {
        BoundaryVisualization::TriangleMesh { meshes, coloring } => {
            assert_eq!(meshes.len(), 1);
            match coloring {
                BoundaryMeshColoring::BoundaryId { ids, max_id } => {
                    // `ids` has one entry per SAMPLE of this boundary (not
                    // one per boundary/mesh) — see
                    // `StaticSampleBoundary::get_visualization`'s use of
                    // `repeat_n(b.render_mesh_id(), b.positions().len())`.
                    assert_eq!(ids.len(), expected_cube_sample_count());
                    assert!(ids.iter().all(|&id| id == 7));
                    assert_eq!(max_id, 7);
                }
                _ => panic!("expected BoundaryId coloring to be preserved"),
            }
        }
        _ => panic!("expected a TriangleMesh result"),
    }
}

// ─── checkpoint / restore ───────────────────────────────────────────

#[test]
fn checkpoint_dynamic_states_are_none_for_static_and_some_for_dynamic() {
    let mut static_boundary = StaticSampleBoundary::new();
    let mut mesh = cube_mesh_container();
    static_boundary.add_static_boundary(&mut mesh, &static_def(1), SPACING, KERNEL_SUPPORT_RADIUS);
    let checkpoint = static_boundary.get_checkpoint();
    assert_eq!(checkpoint.dynamic_states.len(), 1);
    assert!(checkpoint.dynamic_states[0].is_none());

    let mut dynamic_boundary = StaticSampleBoundary::new();
    let mut mesh2 = cube_mesh_container();
    dynamic_boundary.add_dynamic_boundary(
        &mut mesh2,
        &dynamic_def(1, [0., 0., 0.]),
        SPACING,
        KERNEL_SUPPORT_RADIUS,
    );
    let checkpoint = dynamic_boundary.get_checkpoint();
    assert_eq!(checkpoint.dynamic_states.len(), 1);
    assert!(checkpoint.dynamic_states[0].is_some());
}

#[test]
fn checkpoint_roundtrip_preserves_dynamic_boundary_state() {
    let mut boundary = StaticSampleBoundary::new();
    let mut static_mesh = cube_mesh_container();
    boundary.add_static_boundary(
        &mut static_mesh,
        &static_def(1),
        SPACING,
        KERNEL_SUPPORT_RADIUS,
    );
    let mut dynamic_mesh = cube_mesh_container();
    boundary.add_dynamic_boundary(
        &mut dynamic_mesh,
        &dynamic_def(2, [0., 0., 0.]),
        SPACING,
        KERNEL_SUPPORT_RADIUS,
    );
    boundary.initialize::<CubicBSpline3D>(
        &mut SpatialHashing::new(KERNEL_SUPPORT_RADIUS),
        KERNEL_SUPPORT_RADIUS,
        WEIGHTING,
    );

    boundary.add_force_onto_boundary(ForceOntoBoundary {
        id: 1,
        force: Vector3::new(1000., 0., 0.),
        force_location: Point3::origin(),
    });
    boundary.step_forward_in_time(0.1);
    let com_before = boundary.iter().nth(1).unwrap().center_of_mass().unwrap();

    let checkpoint = boundary.get_checkpoint();
    assert_eq!(checkpoint.dynamic_states.len(), 2);
    assert!(checkpoint.dynamic_states[0].is_none());
    assert!(checkpoint.dynamic_states[1].is_some());

    let mut fresh = StaticSampleBoundary::new();
    let mut static_mesh_2 = cube_mesh_container();
    fresh.add_static_boundary(
        &mut static_mesh_2,
        &static_def(1),
        SPACING,
        KERNEL_SUPPORT_RADIUS,
    );
    let mut dynamic_mesh_2 = cube_mesh_container();
    fresh.add_dynamic_boundary(
        &mut dynamic_mesh_2,
        &dynamic_def(2, [0., 0., 0.]),
        SPACING,
        KERNEL_SUPPORT_RADIUS,
    );
    fresh.initialize::<CubicBSpline3D>(
        &mut SpatialHashing::new(KERNEL_SUPPORT_RADIUS),
        KERNEL_SUPPORT_RADIUS,
        WEIGHTING,
    );

    fresh.restore_from_checkpoint(&checkpoint);

    let com_after = fresh.iter().nth(1).unwrap().center_of_mass().unwrap();
    assert!((com_after - com_before).norm() < 1e-9);
}

#[test]
fn restore_from_checkpoint_round_trip_reproduces_stepped_state() {
    // Stricter variant of `checkpoint_roundtrip_preserves_dynamic_boundary_state`:
    // builds two independent boundaries from the SAME definition (so their
    // sampled `position_body` is deterministic and identical), steps only
    // the first one forward under both an initial velocity AND an angular
    // velocity, then verifies restoring the second one from the first's
    // checkpoint reproduces its stepped WORLD-SPACE sample position and
    // velocity exactly — not just the center of mass.
    let def = dynamic_def_moving(1, [1., 2., 3.], [1., 0.5, 0.], [0., 0., 1.]);

    let mut boundary_a = StaticSampleBoundary::new();
    let mut mesh_a = cube_mesh_container();
    boundary_a.add_dynamic_boundary(&mut mesh_a, &def, SPACING, KERNEL_SUPPORT_RADIUS);
    boundary_a.step_forward_in_time(0.0); // sync cache
    boundary_a.step_forward_in_time(0.3);
    boundary_a.step_forward_in_time(0.3);

    let checkpoint = boundary_a.get_checkpoint();

    let mut boundary_b = StaticSampleBoundary::new();
    let mut mesh_b = cube_mesh_container();
    boundary_b.add_dynamic_boundary(&mut mesh_b, &def, SPACING, KERNEL_SUPPORT_RADIUS);
    boundary_b.restore_from_checkpoint(&checkpoint);

    let a = boundary_a.iter().next().unwrap();
    let b = boundary_b.iter().next().unwrap();
    assert_eq!(a.center_of_mass(), b.center_of_mass());
    assert!((a.position(0) - b.position(0)).norm() < 1e-9);
    assert!((a.velocity(0) - b.velocity(0)).norm() < 1e-9);
}

#[test]
fn restore_from_checkpoint_with_mismatched_boundary_count_is_noop() {
    let mut boundary = StaticSampleBoundary::new();
    let mut mesh = cube_mesh_container();
    boundary.add_static_boundary(&mut mesh, &static_def(1), SPACING, KERNEL_SUPPORT_RADIUS);

    let mismatched = BoundaryCheckpoint {
        dynamic_states: vec![None, None],
    };

    boundary.restore_from_checkpoint(&mismatched);
    assert_eq!(boundary.iter().count(), 1);
}

// ─── get_fluid_depth ─────────────────────────────────────────────────

#[test]
fn get_fluid_depth_is_always_zero() {
    let boundary = StaticSampleBoundary::new();
    assert_eq!(boundary.get_fluid_depth(123.0), 0.0);
}
