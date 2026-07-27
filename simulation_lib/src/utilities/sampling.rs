use nalgebra::Point3;
use parry3d_f64::math::Vec3;
use parry3d_f64::query::PointQuery;
use parry3d_f64::shape::TriMesh;

// use super::Scene;

pub fn sample_volume_shifted(mesh: &TriMesh, spacing: f64) -> Vec<Point3<f64>> {
    let aabb = mesh.local_aabb();
    let mut points: Vec<nalgebra::OPoint<f64, nalgebra::Const<3>>> = Vec::new();

    let half_spacing = spacing / 2.0;

    let mut x = aabb.mins.x + half_spacing;
    while x <= aabb.maxs.x {
        let mut y = aabb.mins.y + half_spacing;
        while y <= aabb.maxs.y {
            let mut z = aabb.mins.z + half_spacing;
            let mut layer: usize = 0;
            while z <= aabb.maxs.z {
                let shift = if layer % 2 == 1 { half_spacing } else { 0.0 };
                if mesh.contains_local_point(Vec3::new(x + shift, y, z)) {
                    points.push(Point3::new(x + shift, y, z));
                }
                z += spacing;
                layer += 1;
            }
            y += spacing;
        }
        x += spacing;
    }
    points
}

fn sample_triangle_surface(
    v0: &Point3<f64>,
    v1: &Point3<f64>,
    v2: &Point3<f64>,
    sample_area_density: f64,
) -> Vec<Point3<f64>> {
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let area = edge1.cross(&edge2).norm() * 0.5;
    let n = (area * sample_area_density).sqrt().ceil() as usize;

    let mut points = Vec::new();
    for i in 0..=n {
        for j in 0..=(n - i) {
            let u = i as f64 / n as f64;
            let v = j as f64 / n as f64;
            if u + v <= 1.0 {
                points.push(v0 + edge1 * u + edge2 * v);
            }
        }
    }
    points
}

pub fn sample_triangle_mesh_surface(mesh: &TriMesh, spacing: f64) -> Vec<Point3<f64>> {
    let sample_area_density = 2. / spacing.powi(2);
    mesh.triangles()
        .flat_map(|tri| {
            let a = Point3::new(tri.a.x, tri.a.y, tri.a.z);
            let b = Point3::new(tri.b.x, tri.b.y, tri.b.z);
            let c = Point3::new(tri.c.x, tri.c.y, tri.c.z);
            sample_triangle_surface(&a, &b, &c, sample_area_density)
        })
        .collect()
}
