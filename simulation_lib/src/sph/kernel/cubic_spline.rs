/// Cubic B-Spline kernel function module
use super::KernelFn;

pub struct CubicBSpline;

impl KernelFn for CubicBSpline {
    fn value(r: f64, h: f64) -> f64 {
        let q = r / h;
        let prefactor = 1. / (4. * std::f64::consts::PI * h.powi(3));

        if q < 1.0 {
            prefactor * ((2. - q).powi(3) - 4. * (1. - q).powi(3))
        } else if q < 2.0 {
            prefactor * (2. - q).powi(3)
        } else {
            0.0
        }
    }

    fn derivative(r: f64, h: f64) -> f64 {
        let q = r / h;
        let prefactor = 1. / (4. * std::f64::consts::PI * h.powi(4));

        if q < 1. {
            prefactor * (-3. * (2. - q).powi(2) + 12. * (1. - q).powi(2))
        } else if q < 2.0 {
            prefactor * (-3. * (2. - q).powi(2))
        } else {
            0.0
        }
    }
}
