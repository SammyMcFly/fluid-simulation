//! Cubic B-Spline kernel function module
use super::KernelFn;

#[derive(Clone)]
pub struct CubicBSpline3D;

impl KernelFn for CubicBSpline3D {
    const ALPHA: f64 = 16. / std::f64::consts::PI;
    const DIMENSION: i32 = 3;

    fn w(q: f64) -> f64 {
        (f64::max(1. - q, 0.)).powi(3) - 4. * (f64::max(1. / 2. - q, 0.)).powi(3)
    }

    fn d_q_w(q: f64) -> f64 {
        -3. * (f64::max(1. - q, 0.)).powi(2) + 12. * (f64::max(1. / 2. - q, 0.)).powi(2)
    }
}
