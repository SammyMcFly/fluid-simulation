/// Utilities module
pub mod discretization;
pub mod sampling;
pub mod triangle_mesh;

use nalgebra::{Point3, Vector3};

// /// Calculate the distance between two 3D points
// fn distance(from: &Vector3<f64>, to: &Vector3<f64>) -> f64 {
//     (to - from).norm()
// }

/// Create a vector from location 'from' towards location 'towards'
pub fn vector(from: &Point3<f64>, towards: &Point3<f64>) -> Vector3<f64> {
    towards - from
}
