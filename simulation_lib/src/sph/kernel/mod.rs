/// Kernel functions
///
use nalgebra::Vector3;

pub mod cubic_spline;
pub use cubic_spline::CubicBSpline3D;

/// Kernel function trait according to 'Boundary Handling and Neighbor Search in Iterative Incompressible SPH' by Stefan Band
///
/// `r_vec` = r_i - r_j (from neighbor towards evaluated position)
pub trait KernelFn: Send + Sync {
    const ALPHA: f64;
    const DIMENSION: i32;

    /// q(r_vec, h)
    fn q(r_vec: &Vector3<f64>, support_radius: f64) -> f64 {
        r_vec.norm() / support_radius
    }
    /// w(q)
    fn w(q: f64) -> f64;

    /// W(r_vec, h)
    fn kernel_function(r_vec: &Vector3<f64>, support_radius: f64) -> f64 {
        Self::ALPHA / support_radius.powi(Self::DIMENSION) * Self::w(Self::q(r_vec, support_radius))
    }

    /// dw/dq(q)
    fn d_q_w(q: f64) -> f64;

    /// Full gradient: ∇_i W(|r_i - r_j|, h)
    ///
    fn kernel_gradient(r_vec: &Vector3<f64>, support_radius: f64) -> Vector3<f64> {
        let r = r_vec.norm();
        if r == 0.0 {
            Vector3::zeros()
        } else {
            Self::ALPHA / support_radius.powi(Self::DIMENSION+1) * r_vec / r * Self::d_q_w(Self::q(r_vec, support_radius))
        }
    }
}
