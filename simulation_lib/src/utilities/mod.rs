//! Utilities module
pub mod discretization;
pub mod sampling;
pub mod triangle_mesh;

use nalgebra::{Point3, UnitQuaternion, Vector3};

/// Create a vector from location 'from' towards location 'towards'
pub fn vector(from: &Point3<f64>, towards: &Point3<f64>) -> Vector3<f64> {
    towards - from
}

pub fn euler_deg_to_quaternion(euler_deg: [f64; 3]) -> UnitQuaternion<f64> {
    let [rx, ry, rz] = euler_deg.map(f64::to_radians);
    UnitQuaternion::from_euler_angles(rx, ry, rz)
}
