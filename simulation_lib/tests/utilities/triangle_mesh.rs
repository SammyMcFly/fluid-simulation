//! Integration tests for `triangle_mesh`, exercising only its public API —
//! mirroring how an external user of the crate would use it. Cache
//! invalidation internals (private `MeshContainer` fields) are covered
//! separately in the in-module `#[cfg(test)]` block.

use std::sync::atomic::{AtomicUsize, Ordering};

use nalgebra::{Matrix4, Point3, Rotation3, Vector3};
use parry3d_f64::math::Vec3;
use parry3d_f64::shape::TriMesh;

use simulation_lib::sph::setup::input::VertexNormalRenderOption;
use simulation_lib::utilities::triangle_mesh::{
    LoadedMesh, MeshContainer, MeshError, MeshHandle, MeshLibrary, RenderMesh, RenderVertex,
    build_transform,
};

// ─── Fixtures ─────────────────────────────────────────────────────────────

fn tiny_triangle_loaded_mesh() -> LoadedMesh {
    LoadedMesh {
        positions: vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ],
        // Deliberately NOT unit / NOT geometrically meaningful, so that
        // tests can tell apart "normals were computed" (AngleWeighted...)
        // from "normals were copied through as-is" (FaceNormals).
        normals: vec![
            Vector3::new(7.0, 0.0, 0.0),
            Vector3::new(7.0, 0.0, 0.0),
            Vector3::new(7.0, 0.0, 0.0),
        ],
        indices: vec![[0, 1, 2]],
    }
}

/// Cube of side length 2, centered at the origin, outward-facing winding
/// (same fixture/winding used elsewhere in this crate's boundary tests).
fn cube_trimesh() -> TriMesh {
    let positions = vec![
        Vec3::new(1., 1., 1.),
        Vec3::new(1., 1., -1.),
        Vec3::new(1., -1., 1.),
        Vec3::new(1., -1., -1.),
        Vec3::new(-1., 1., 1.),
        Vec3::new(-1., 1., -1.),
        Vec3::new(-1., -1., 1.),
        Vec3::new(-1., -1., -1.),
    ];
    let indices: Vec<[u32; 3]> = vec![
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
    TriMesh::new(positions, indices).unwrap()
}

static TEMP_FILE_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn unique_temp_obj_path() -> std::path::PathBuf {
    let n = TEMP_FILE_COUNTER.fetch_add(1, Ordering::SeqCst);
    std::env::temp_dir().join(format!("triangle_mesh_test_{}_{n}.obj", std::process::id()))
}

fn matrices_approx_eq(a: &Matrix4<f64>, b: &Matrix4<f64>, eps: f64) -> bool {
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < eps)
}

// ─── MeshHandle ─────────────────────────────────────────────────────────

#[test]
fn mesh_handle_equality_is_based_on_both_fields() {
    let a = MeshHandle { idx: 1, mesh_id: 5 };
    let b = MeshHandle { idx: 1, mesh_id: 5 };
    let c = MeshHandle { idx: 1, mesh_id: 6 };
    let d = MeshHandle { idx: 2, mesh_id: 5 };
    assert_eq!(a, b);
    assert_ne!(a, c);
    assert_ne!(a, d);
}

#[test]
fn mesh_handle_hash_is_consistent_with_equality() {
    use std::collections::HashSet;
    let a = MeshHandle { idx: 3, mesh_id: 9 };
    let b = MeshHandle { idx: 3, mesh_id: 9 };
    let mut set = HashSet::new();
    set.insert(a);
    set.insert(b);
    assert_eq!(set.len(), 1, "equal handles must hash to the same bucket");
}

// ─── RenderVertex ─────────────────────────────────────────────────────────

#[test]
fn render_vertex_default_is_all_zero() {
    let v = RenderVertex::default();
    assert_eq!(v.position, [0.0, 0.0, 0.0]);
    assert_eq!(v.normal, [0.0, 0.0, 0.0]);
}

#[test]
fn render_vertex_has_no_padding_as_required_by_bytemuck_pod() {
    // Two [f64; 3] fields, no implicit padding expected for `#[repr(C)]`.
    assert_eq!(std::mem::size_of::<RenderVertex>(), 6 * 8);
}

// ─── RenderMesh::extend ────────────────────────────────────────────────────

#[test]
fn extend_onto_empty_mesh_keeps_indices_unchanged() {
    let mut mesh = RenderMesh::default();
    let other = RenderMesh {
        vertices: vec![RenderVertex::default(); 3],
        indices: vec![0, 1, 2],
    };
    mesh.extend(other.clone());
    assert_eq!(mesh, other);
}

#[test]
fn extend_offsets_indices_by_existing_vertex_count() {
    let mut mesh = RenderMesh {
        vertices: vec![RenderVertex::default(); 2],
        indices: vec![0, 1, 0],
    };
    let other = RenderMesh {
        vertices: vec![RenderVertex::default(); 3],
        indices: vec![0, 1, 2],
    };
    mesh.extend(other);

    assert_eq!(mesh.vertices.len(), 5);
    // Original indices untouched, appended ones shifted by 2.
    assert_eq!(mesh.indices, vec![0, 1, 0, 2, 3, 4]);
}

#[test]
fn extend_called_twice_accumulates_offsets_correctly() {
    let mut mesh = RenderMesh::default();
    mesh.extend(RenderMesh {
        vertices: vec![RenderVertex::default(); 3],
        indices: vec![0, 1, 2],
    });
    mesh.extend(RenderMesh {
        vertices: vec![RenderVertex::default(); 3],
        indices: vec![0, 1, 2],
    });
    assert_eq!(mesh.vertices.len(), 6);
    assert_eq!(mesh.indices, vec![0, 1, 2, 3, 4, 5]);
}

// ─── RenderMesh::from_loaded_mesh ──────────────────────────────────────────

#[test]
fn from_loaded_mesh_copies_positions_and_normals_verbatim() {
    // Documents that this path performs no geometric computation at all —
    // it just casts through whatever is stored in `LoadedMesh`, even if
    // (as here) the stored normals are not unit vectors.
    let raw = tiny_triangle_loaded_mesh();
    let mesh = RenderMesh::from_loaded_mesh(&raw);

    assert_eq!(mesh.vertices.len(), 3);
    assert_eq!(mesh.vertices[0].position, [0.0, 0.0, 0.0]);
    assert_eq!(mesh.vertices[0].normal, [7.0, 0.0, 0.0]);
    assert_eq!(mesh.indices, vec![0, 1, 2]);
}

#[test]
fn from_loaded_mesh_of_empty_input_is_empty() {
    let raw = LoadedMesh {
        positions: vec![],
        normals: vec![],
        indices: vec![],
    };
    let mesh = RenderMesh::from_loaded_mesh(&raw);
    assert!(mesh.vertices.is_empty());
    assert!(mesh.indices.is_empty());
}

// ─── RenderMesh::from_trimesh ──────────────────────────────────────────────

#[test]
fn from_trimesh_single_triangle_normals_match_the_face_normal() {
    let positions = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];
    let indices = vec![[0u32, 1, 2]];
    let trimesh = TriMesh::new(positions, indices).unwrap();

    let mesh = RenderMesh::from_trimesh(&trimesh);
    assert_eq!(mesh.vertices.len(), 3);

    // With only one contributing face, every vertex's angle-weighted normal
    // must point in exactly the face-normal direction (weighting by a
    // positive angle doesn't change direction), i.e. +z here.
    for v in &mesh.vertices {
        let n = Vector3::new(v.normal[0], v.normal[1], v.normal[2]);
        assert!(
            (n.norm() - 1.0).abs() < 1e-9,
            "normal not unit length: {n:?}"
        );
        assert!(
            (n - Vector3::new(0.0, 0.0, 1.0)).norm() < 1e-9,
            "unexpected normal direction: {n:?}"
        );
    }
}

#[test]
fn from_trimesh_cube_normals_point_outward_and_are_unit_length() {
    let trimesh = cube_trimesh();
    let mesh = RenderMesh::from_trimesh(&trimesh);

    assert_eq!(mesh.vertices.len(), 8);
    assert_eq!(mesh.indices.len(), 12 * 3);

    for v in &mesh.vertices {
        let pos = Vector3::new(v.position[0], v.position[1], v.position[2]);
        let n = Vector3::new(v.normal[0], v.normal[1], v.normal[2]);
        assert!(
            (n.norm() - 1.0).abs() < 1e-9,
            "normal not unit length: {n:?}"
        );
        // For a cube centered at the origin, each corner's outward
        // direction is (up to normalization) the corner's own position —
        // so a correct outward normal must have a positive dot product
        // with it, regardless of the exact angle-weighting convention.
        assert!(
            n.dot(&pos) > 0.0,
            "normal {n:?} at corner {pos:?} does not point outward"
        );
    }
}

// ─── MeshContainer ──────────────────────────────────────────────────────

#[test]
fn mesh_container_raw_returns_the_data_it_was_built_with() {
    let raw = tiny_triangle_loaded_mesh();
    let container = MeshContainer::new(raw.clone());
    assert_eq!(container.raw().positions, raw.positions);
    assert_eq!(container.raw().normals, raw.normals);
    assert_eq!(container.raw().indices, raw.indices);
}

#[test]
fn mesh_container_trimesh_preserves_vertex_and_triangle_counts_for_a_clean_mesh() {
    let mut container = MeshContainer::new(LoadedMesh {
        positions: vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
        ],
        normals: vec![],
        indices: vec![[0, 1, 2], [1, 3, 2]],
    });
    let trimesh = container.trimesh();
    assert_eq!(trimesh.vertices().len(), 4);
    assert_eq!(trimesh.indices().len(), 2);
}

#[test]
fn mesh_container_trimesh_is_cached_across_repeated_calls() {
    let mut container = MeshContainer::new(tiny_triangle_loaded_mesh());
    let ptr_a = container.trimesh() as *const _;
    let ptr_b = container.trimesh() as *const _;
    assert_eq!(
        ptr_a, ptr_b,
        "expected the same cached instance to be returned"
    );
}

#[test]
fn mesh_container_render_mesh_face_normals_matches_from_loaded_mesh() {
    let raw = tiny_triangle_loaded_mesh();
    let mut container = MeshContainer::new(raw.clone());
    let via_container = container
        .render_mesh(VertexNormalRenderOption::FaceNormals)
        .clone();
    let via_direct = RenderMesh::from_loaded_mesh(&raw);
    assert_eq!(via_container, via_direct);
}

#[test]
fn mesh_container_render_mesh_angle_weighted_ignores_raw_normals() {
    // `raw.normals` is [7, 0, 0] (bogus, non-geometric) in the fixture; the
    // AngleWeightedPseudoNormals path must derive normals purely from
    // geometry via `trimesh()`, so it must NOT reproduce that bogus value.
    let mut container = MeshContainer::new(tiny_triangle_loaded_mesh());
    let mesh = container.render_mesh(VertexNormalRenderOption::AngleWeightedPseudoNormals);
    for v in &mesh.vertices {
        assert_ne!(v.normal, [7.0, 0.0, 0.0]);
        let n = Vector3::new(v.normal[0], v.normal[1], v.normal[2]);
        assert!((n.norm() - 1.0).abs() < 1e-9);
    }
}

#[test]
fn mesh_container_render_mesh_is_cached_across_repeated_calls() {
    let mut container = MeshContainer::new(tiny_triangle_loaded_mesh());
    let ptr_a = container.render_mesh(VertexNormalRenderOption::FaceNormals) as *const _;
    let ptr_b = container.render_mesh(VertexNormalRenderOption::FaceNormals) as *const _;
    assert_eq!(
        ptr_a, ptr_b,
        "expected the same cached instance to be returned"
    );
}

// ─── MeshContainer::transform ─────────────────────────────────────────────

#[test]
fn transform_pure_translation_shifts_all_positions() {
    let mut container = MeshContainer::new(tiny_triangle_loaded_mesh());
    container.transform(&[1.0, 2.0, 3.0], &[0., 0., 0.], &[1., 1., 1.]);
    assert_eq!(container.raw().positions[0], Point3::new(1.0, 2.0, 3.0));
    assert_eq!(container.raw().positions[1], Point3::new(2.0, 2.0, 3.0));
    assert_eq!(container.raw().positions[2], Point3::new(1.0, 3.0, 3.0));
}

#[test]
fn transform_pure_scale_scales_positions_from_the_origin() {
    let mut container = MeshContainer::new(tiny_triangle_loaded_mesh());
    container.transform(&[0., 0., 0.], &[0., 0., 0.], &[2.0, 3.0, 4.0]);
    assert_eq!(container.raw().positions[1], Point3::new(2.0, 0.0, 0.0));
    assert_eq!(container.raw().positions[2], Point3::new(0.0, 3.0, 0.0));
}

#[test]
fn transform_matches_build_transform_applied_independently() {
    // Ties `MeshContainer::transform` back to the public, pure
    // `build_transform` function: whatever `build_transform` computes for
    // a given translation/rotation/scale must be exactly what gets applied
    // to the raw positions — this is the documented contract, independent
    // of any particular Euler-angle sign convention.
    let translation = [1.0, -2.0, 0.5];
    let rotation = [15.0, 30.0, 45.0];
    let scale = [1.5, 0.5, 2.0];

    let raw = tiny_triangle_loaded_mesh();
    let expected_matrix = build_transform(&translation, &rotation, &scale);
    let expected_positions: Vec<Point3<f64>> = raw
        .positions
        .iter()
        .map(|p| expected_matrix.transform_point(p))
        .collect();

    let mut container = MeshContainer::new(raw);
    container.transform(&translation, &rotation, &scale);

    for (actual, expected) in container.raw().positions.iter().zip(&expected_positions) {
        assert!(
            (actual - expected).norm() < 1e-9,
            "actual {actual:?} != expected {expected:?}"
        );
    }
}

#[test]
fn transform_with_identity_parameters_leaves_positions_unchanged() {
    let raw = tiny_triangle_loaded_mesh();
    let original = raw.positions.clone();
    let mut container = MeshContainer::new(raw);
    container.transform(&[0., 0., 0.], &[0., 0., 0.], &[1., 1., 1.]);
    assert_eq!(container.raw().positions, original);
}

#[test]
fn trimesh_reflects_geometry_after_a_real_transform() {
    // Whether or not a cache was invalidated internally is an
    // implementation detail (covered separately by the in-module test
    // suite via direct field access); what matters externally is that
    // `trimesh()` returns geometry consistent with the *current* raw
    // positions after a real transform, not stale pre-transform geometry.
    let mut container = MeshContainer::new(tiny_triangle_loaded_mesh());
    let _ = container.trimesh(); // populate cache with pre-transform geometry

    container.transform(&[10.0, 0.0, 0.0], &[0., 0., 0.], &[1., 1., 1.]);

    let trimesh = container.trimesh();
    let vertices: Vec<Point3<f64>> = trimesh
        .vertices()
        .iter()
        .map(|v| Point3::new(v.x, v.y, v.z))
        .collect();

    assert!(
        vertices
            .iter()
            .any(|v| (v - Point3::new(10.0, 0.0, 0.0)).norm() < 1e-9),
        "expected trimesh() to reflect the translated vertex, got {vertices:?}"
    );
}

#[test]
fn transform_calls_compose_sequentially_on_current_positions() {
    // Mirrors real usage (e.g. dynamic boundaries: scale first, then
    // translate/rotate in a second call) — each `transform` call operates
    // on the mesh's *current* positions, not the original ones.
    let mut container = MeshContainer::new(tiny_triangle_loaded_mesh());
    container.transform(&[1.0, 0.0, 0.0], &[0., 0., 0.], &[1., 1., 1.]);
    container.transform(&[0.0, 2.0, 0.0], &[0., 0., 0.], &[1., 1., 1.]);
    assert_eq!(container.raw().positions[0], Point3::new(1.0, 2.0, 0.0));
}

#[test]
#[should_panic]
fn transform_panics_on_negative_scale_in_debug_builds() {
    // Relies on `debug_assert!`, so this only panics in debug builds
    // (the default for `cargo test`); running under `cargo test --release`
    // would not trigger it.
    let mut container = MeshContainer::new(tiny_triangle_loaded_mesh());
    container.transform(&[0., 0., 0.], &[0., 0., 0.], &[-1.0, 1.0, 1.0]);
}

// ─── MeshLibrary ────────────────────────────────────────────────────────

#[test]
fn load_obj_of_a_valid_file_populates_the_library() {
    let path = unique_temp_obj_path();
    std::fs::write(
        &path,
        "v 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 0.0 1.0 0.0\nf 1 2 3\n",
    )
    .expect("failed to write temp .obj file");

    let mut lib = MeshLibrary::default();
    let result = lib.load_obj(path.to_str().unwrap());
    let _ = std::fs::remove_file(&path);

    assert!(result.is_ok());
    assert_eq!(lib.meshes.len(), 1);

    let raw = lib.meshes[0].raw();
    assert_eq!(raw.positions.len(), 3);
    assert_eq!(raw.indices, vec![[0, 1, 2]]);
    // No `vn` lines in the file -> tobj reports no normals.
    assert!(raw.normals.is_empty());
}

#[test]
fn load_obj_of_a_nonexistent_file_returns_a_descriptive_error() {
    let bogus_path = "/definitely/does/not/exist/mesh_12345.obj";
    let mut lib = MeshLibrary::default();
    let result = lib.load_obj(bogus_path);

    let err = result.expect_err("expected loading a nonexistent file to fail");
    assert!(matches!(err, MeshError::Obj { .. }));
    let message = format!("{err}");
    assert!(
        message.contains(bogus_path),
        "error message should mention the offending path: {message}"
    );
    // Loading must not have partially registered a mesh.
    assert!(lib.meshes.is_empty());
}

#[test]
fn get_mesh_container_looks_up_by_index() {
    let path = unique_temp_obj_path();
    std::fs::write(
        &path,
        "v 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 0.0 1.0 0.0\nf 1 2 3\n",
    )
    .expect("failed to write temp .obj file");

    let mut lib = MeshLibrary::default();
    lib.load_obj(path.to_str().unwrap()).unwrap();
    let _ = std::fs::remove_file(&path);

    let handle = MeshHandle {
        idx: 0,
        mesh_id: 42,
    };
    let container = lib.get_mesh_container(handle);
    assert_eq!(container.raw().positions.len(), 3);
}

// ─── build_transform ────────────────────────────────────────────────────

#[test]
fn build_transform_identity_parameters_yield_identity_matrix() {
    let m = build_transform(&[0., 0., 0.], &[0., 0., 0.], &[1., 1., 1.]);
    assert!(matrices_approx_eq(&m, &Matrix4::identity(), 1e-12));
}

#[test]
fn build_transform_pure_translation_maps_origin_to_the_translation() {
    let m = build_transform(&[3.0, -1.0, 2.0], &[0., 0., 0.], &[1., 1., 1.]);
    let p = m.transform_point(&Point3::origin());
    assert!((p - Point3::new(3.0, -1.0, 2.0)).norm() < 1e-12);
}

#[test]
fn build_transform_pure_scale_scales_from_the_origin() {
    let m = build_transform(&[0., 0., 0.], &[0., 0., 0.], &[2.0, 3.0, 4.0]);
    let p = m.transform_point(&Point3::new(1.0, 1.0, 1.0));
    assert!((p - Point3::new(2.0, 3.0, 4.0)).norm() < 1e-12);
}

#[test]
fn build_transform_pure_rotation_is_a_proper_rigid_rotation() {
    // Convention-agnostic check: whatever the exact Euler-angle convention,
    // a pure rotation (scale = 1, translation = 0) must be length-preserving,
    // orthonormal, and right-handed (no reflection).
    let m = build_transform(&[0., 0., 0.], &[15.0, 30.0, 45.0], &[1., 1., 1.]);
    let o = m.transform_point(&Point3::origin());
    let vx = m.transform_point(&Point3::new(1.0, 0.0, 0.0)) - o;
    let vy = m.transform_point(&Point3::new(0.0, 1.0, 0.0)) - o;
    let vz = m.transform_point(&Point3::new(0.0, 0.0, 1.0)) - o;

    assert!((vx.norm() - 1.0).abs() < 1e-9);
    assert!((vy.norm() - 1.0).abs() < 1e-9);
    assert!((vz.norm() - 1.0).abs() < 1e-9);
    assert!(vx.dot(&vy).abs() < 1e-9);
    assert!(vy.dot(&vz).abs() < 1e-9);
    assert!(vx.dot(&vz).abs() < 1e-9);
    assert!(
        (vx.cross(&vy).dot(&vz) - 1.0).abs() < 1e-9,
        "expected a right-handed rotation (no reflection)"
    );
}

#[test]
fn build_transform_documented_composition_order_is_scale_then_rotate_then_translate() {
    // Reconstructs the exact formula documented on `build_transform`
    // ("Order: scale first, then rotate, then translate") using the same
    // public nalgebra building blocks, independently of the function body.
    // This guards the *documented contract*, not an implementation detail:
    // if the order were ever silently changed, this test would catch it.
    let translation = [1.0, 2.0, -3.0];
    let rotation: [f64; 3] = [10.0, 20.0, 30.0];
    let scale = [1.5, 0.5, 2.0];

    let expected = Matrix4::new_translation(&Vector3::new(
        translation[0],
        translation[1],
        translation[2],
    )) * Rotation3::from_euler_angles(
        rotation[0].to_radians(),
        rotation[1].to_radians(),
        rotation[2].to_radians(),
    )
    .to_homogeneous()
        * Matrix4::new_nonuniform_scaling(&Vector3::new(scale[0], scale[1], scale[2]));

    let actual = build_transform(&translation, &rotation, &scale);
    assert!(matrices_approx_eq(&actual, &expected, 1e-9));
}
