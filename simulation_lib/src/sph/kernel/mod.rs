/// Kernel functions
///
use nalgebra::Vector3;

pub mod cubic_spline;
pub use cubic_spline::CubicBSpline;

pub trait KernelFn: Send + Sync {
    /// W(r, h)
    fn value(distance: f64, smoothing_length: f64) -> f64;

    /// dW/dr (scalar derivative)
    fn derivative(distance: f64, smoothing_length: f64) -> f64;

    /// Full gradient: (dW/dr) * (r_vec / |r_vec|)
    ///
    /// r_vec is defined as the vector from the center of the kernel function to evaluated position.
    fn gradient(distance_vec: &Vector3<f64>, distance: f64, smoothing_length: f64) -> Vector3<f64> {
        if distance == 0.0 {
            Vector3::zeros()
        } else {
            distance_vec * (Self::derivative(distance, smoothing_length) / distance)
        }
    }
}
