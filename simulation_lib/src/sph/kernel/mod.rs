/// Kernel functions
///
use nalgebra::Vector3;

pub mod cubic_spline;
pub use cubic_spline::CubicBSpline;

pub trait KernelFn: Send + Sync {
    /// W(r, h)
    fn value(r: f64, h: f64) -> f64;

    /// dW/dr (scalar derivative)
    fn derivative(r: f64, h: f64) -> f64;

    /// Full gradient: ∇_i W(|r_i - r_j|, h)
    ///
    /// `r_vec` = r_i - r_j (from neighbor towards evaluated position)
    fn gradient(r_vec: &Vector3<f64>, r: f64, h: f64) -> Vector3<f64> {
        if r == 0.0 {
            Vector3::zeros()
        } else {
            r_vec * (Self::derivative(r, h) / r)
        }
    }
}
