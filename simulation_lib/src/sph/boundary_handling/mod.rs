/// Boundary handling module
use gauss_quad::GaussLegendre;
use parry3d::shape::Triangle;
use parry3d::query::PointQuery;
// use approx::assert_abs_diff_eq;
use std::f64::consts::PI;

pub trait BoundaryRepresentation {
    fn initialize();
    fn add_viscosity_acceleration();
    fn add_pressure_acceleration();
}

/// Integrates f(x, y, z) over a sphere with radius R with Gauß-Legendre quadrature.
fn integrate_sphere_volume<F>(
    f: F,
    radius: f64,
    n: usize,
) -> f64
where
    F: Fn(f64, f64, f64) -> f64,
{
    let quad = GaussLegendre::new(n.try_into().unwrap());

    quad.integrate(0.0, radius, |r| {
        r.powi(2) * quad.integrate(0.0, PI, |theta| {
            theta.sin() * quad.integrate(0.0, 2. * PI, |phi| {
                let x = r * theta.sin() * phi.cos();
                let y = r * theta.sin() * phi.sin();
                let z = r * theta.cos();
                f(x,y,z)
            })
        })
    })
}

fn closest_point_on_triangle() {
    let triangle = Triangle::new(
        parry3d::math::Vec3::new(0.0, 0.0, 0.0),
        parry3d::math::Vec3::new(1.0, 0.0, 0.0),
        parry3d::math::Vec3::new(0.0, 1.0, 0.0),
    );

    let point = parry3d::math::Vec3::new(0.5, 0.5, 1.0);

    // Projects the point onto the triangle (closest point)
    let closest = triangle.project_local_point(point, true);

    println!("Closest point: {:?}", closest.point);
    println!("Is inside: {:?}", closest.is_inside);
}
