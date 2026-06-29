/// Utilities module
use parry3d_f64::shape::Triangle;
use parry3d_f64::query::PointQuery;
use nalgebra::{Point3, Vector3};
use gauss_quad::GaussLegendre;
use std::f64::consts::PI;

pub mod sampling;
pub mod triangle_mesh;


// /// Calculate the distance between two 3D points
// fn distance(from: &Vector3<f64>, to: &Vector3<f64>) -> f64 {
//     (to - from).norm()
// }

/// Create a vector from location 'from' towards location 'towards'
pub fn vector(from: &Point3<f64>, towards: &Point3<f64>) -> Vector3<f64> {
    towards - from
}

/// Integrates f(x, y, z) over a sphere with radius R with Gauß-Legendre quadrature.
pub fn integrate_sphere_volume<F>(
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
