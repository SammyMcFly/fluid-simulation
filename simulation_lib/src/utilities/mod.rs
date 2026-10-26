//! Utilities module
pub mod discretization;
pub mod sampling;
pub mod triangle_mesh;

use nalgebra::{Matrix3, Point3, UnitQuaternion, Vector3};
use num_traits::Zero;

/// Create a vector from location 'from' towards location 'towards'
pub fn vector(from: &Point3<f64>, towards: &Point3<f64>) -> Vector3<f64> {
    towards - from
}

pub fn euler_deg_to_quaternion(euler_deg: [f64; 3]) -> UnitQuaternion<f64> {
    let [rx, ry, rz] = euler_deg.map(f64::to_radians);
    UnitQuaternion::from_euler_angles(rx, ry, rz)
}

pub trait Outer {
    type OuterProductType;
    fn outer(&self, other: &Self) -> Self::OuterProductType;
}

impl<N: Copy + std::ops::Mul<N, Output = N> + Zero> Outer for Vector3<N> {
    type OuterProductType = Matrix3<N>;

    fn outer(&self, other: &Self) -> Self::OuterProductType {
        Matrix3::new(
            self[0] * other[0],
            self[0] * other[1],
            self[0] * other[2],
            self[1] * other[0],
            self[1] * other[1],
            self[1] * other[2],
            self[2] * other[0],
            self[2] * other[1],
            self[2] * other[2],
        )
    }
}
