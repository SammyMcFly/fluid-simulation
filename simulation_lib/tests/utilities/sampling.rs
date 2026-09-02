//! Integration tests for `sampling`, exercising only its public API
//! (`sample_volume_shifted`, `sample_triangle_mesh_surface`) as an external
//! user of the crate would — no access to `sample_triangle_surface`,
//! `point_in_triangle`, or `cross2`.

use nalgebra::Point3;
use parry3d_f64::math::Vec3;
use parry3d_f64::query::PointQuery;
use parry3d_f64::shape::{TriMesh, TriMeshFlags};

use simulation_lib::utilities::sampling::{sample_triangle_mesh_surface, sample_volume_shifted};

/// Axis-aligned cube of side length 2, centered at the origin, outward-facing winding.
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
    TriMesh::with_flags(
        positions,
        indices,
        TriMeshFlags::ORIENTED
            | TriMeshFlags::MERGE_DUPLICATE_VERTICES
            | TriMeshFlags::FIX_INTERNAL_EDGES,
    )
    .expect("valid cube mesh")
}

// ─── sample_volume_shifted ──────────────────────────────────────────────

#[test]
fn sample_volume_shifted_returns_nonempty_for_reasonable_spacing() {
    let mesh = cube_trimesh();
    assert!(!sample_volume_shifted(&mesh, 0.5).is_empty());
}

#[test]
fn sample_volume_shifted_all_points_are_inside_the_mesh() {
    let mesh = cube_trimesh();
    let points = sample_volume_shifted(&mesh, 0.4);
    assert!(!points.is_empty());
    for p in &points {
        assert!(
            mesh.contains_local_point(Vec3::new(p.x, p.y, p.z)),
            "point {p:?} reported by sample_volume_shifted is not actually inside the mesh"
        );
    }
}

#[test]
fn sample_volume_shifted_all_points_lie_within_mesh_aabb() {
    let mesh = cube_trimesh();
    let aabb = mesh.local_aabb();
    for p in sample_volume_shifted(&mesh, 0.3) {
        assert!(p.x >= aabb.mins.x && p.x <= aabb.maxs.x);
        assert!(p.y >= aabb.mins.y && p.y <= aabb.maxs.y);
        assert!(p.z >= aabb.mins.z && p.z <= aabb.maxs.z);
    }
}

#[test]
fn sample_volume_shifted_finer_spacing_yields_more_points() {
    let mesh = cube_trimesh();
    let coarse = sample_volume_shifted(&mesh, 0.6);
    let fine = sample_volume_shifted(&mesh, 0.2);
    assert!(fine.len() > coarse.len());
}

#[test]
fn sample_volume_shifted_spacing_larger_than_mesh_yields_few_or_no_points() {
    let mesh = cube_trimesh(); // side length 2
    assert!(sample_volume_shifted(&mesh, 100.0).len() <= 1);
}

#[test]
fn sample_volume_shifted_is_deterministic() {
    let mesh = cube_trimesh();
    assert_eq!(
        sample_volume_shifted(&mesh, 0.3),
        sample_volume_shifted(&mesh, 0.3)
    );
}

// ─── sample_triangle_mesh_surface ───────────────────────────────────────

#[test]
fn sample_triangle_mesh_surface_returns_nonempty_for_reasonable_spacing() {
    let mesh = cube_trimesh();
    assert!(!sample_triangle_mesh_surface(&mesh, 0.5).is_empty());
}

#[test]
fn sample_triangle_mesh_surface_all_points_lie_approximately_on_the_surface() {
    let mesh = cube_trimesh();
    let points = sample_triangle_mesh_surface(&mesh, 0.4);
    assert!(!points.is_empty());
    for p in &points {
        let query_point = glam::DVec3::new(p.x, p.y, p.z);
        let proj = mesh.project_local_point(query_point, true);
        let dist = (proj.point - query_point).length();
        assert!(
            dist < 1e-9,
            "point {p:?} lies {dist} away from the mesh surface"
        );
    }
}

#[test]
fn sample_triangle_mesh_surface_finer_spacing_yields_more_points() {
    let mesh = cube_trimesh();
    let coarse = sample_triangle_mesh_surface(&mesh, 0.6);
    let fine = sample_triangle_mesh_surface(&mesh, 0.2);
    assert!(fine.len() > coarse.len());
}

#[test]
fn sample_triangle_mesh_surface_aggregates_points_from_multiple_faces() {
    // The cube has 12 triangles (2 per face); confirm samples span
    // several distinct faces, not just a single triangle's worth.
    let mesh = cube_trimesh();
    let points = sample_triangle_mesh_surface(&mesh, 0.5);
    let on_face = |p: &Point3<f64>, axis: usize, value: f64| (p[axis] - value).abs() < 1e-9;

    assert!(
        points.iter().any(|p| on_face(p, 0, 1.0)),
        "expected samples on the x=+1 face"
    );
    assert!(
        points.iter().any(|p| on_face(p, 0, -1.0)),
        "expected samples on the x=-1 face"
    );
    assert!(
        points.iter().any(|p| on_face(p, 1, 1.0)),
        "expected samples on the y=+1 face"
    );
}

#[test]
fn sample_triangle_mesh_surface_is_deterministic() {
    let mesh = cube_trimesh();
    assert_eq!(
        sample_triangle_mesh_surface(&mesh, 0.35),
        sample_triangle_mesh_surface(&mesh, 0.35)
    );
}
