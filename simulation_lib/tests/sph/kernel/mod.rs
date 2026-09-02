//! Integration tests for the generic `KernelFn` trait default methods
//! (`q`, `kernel_function`, `kernel_gradient`), using a minimal dummy
//! kernel to isolate the trait's plumbing from any specific `w`/`d_q_w`
//! formula. Concrete correctness of the actual cubic B-spline formula is
//! covered separately in `tests/cubic_spline.rs`.
mod cubic_spline;

use nalgebra::Vector3;

use simulation_lib::sph::kernel::KernelFn;

/// A deliberately non-physical dummy kernel (it doesn't vanish at compact
/// support, isn't normalized, etc.) — only used to exercise the generic
/// default-method formulas (`q`, `kernel_function`, `kernel_gradient`)
/// with known, hand-computable constants.
#[derive(Clone)]
struct LinearTestKernel;

impl KernelFn for LinearTestKernel {
    const ALPHA: f64 = 2.0;
    const DIMENSION: i32 = 3;

    fn w(q: f64) -> f64 {
        1.0 - q
    }

    fn d_q_w(_q: f64) -> f64 {
        -1.0
    }
}

fn assert_vec_approx(actual: Vector3<f64>, expected: Vector3<f64>, eps: f64) {
    assert!(
        (actual - expected).norm() < eps,
        "expected {expected:?}, got {actual:?}"
    );
}

// ─── q ──────────────────────────────────────────────────────────────────

#[test]
fn q_is_norm_of_r_vec_divided_by_support_radius() {
    let r_vec = Vector3::new(3.0, 4.0, 0.0); // norm = 5
    assert!((LinearTestKernel::q(&r_vec, 2.0) - 2.5).abs() < 1e-12);
}

#[test]
fn q_is_invariant_under_simultaneous_scaling_of_r_vec_and_support_radius() {
    let r_vec = Vector3::new(1.0, 2.0, 3.0);
    let h = 1.5;
    let q1 = LinearTestKernel::q(&r_vec, h);
    let q2 = LinearTestKernel::q(&(2.0 * r_vec), 2.0 * h);
    assert!((q1 - q2).abs() < 1e-12);
}

#[test]
fn q_depends_only_on_the_norm_of_r_vec_not_its_direction() {
    let h = 1.0;
    let a = Vector3::new(1.0, 0.0, 0.0);
    let b = Vector3::new(0.0, 1.0, 0.0);
    assert!((LinearTestKernel::q(&a, h) - LinearTestKernel::q(&b, h)).abs() < 1e-12);
}

// ─── kernel_function ────────────────────────────────────────────────────

#[test]
fn kernel_function_matches_the_documented_formula() {
    let r_vec = Vector3::new(1.0, 0.0, 0.0);
    let h = 2.0;
    let q = LinearTestKernel::q(&r_vec, h); // 0.5
    let expected =
        LinearTestKernel::ALPHA / h.powi(LinearTestKernel::DIMENSION) * LinearTestKernel::w(q);
    assert!((LinearTestKernel::kernel_function(&r_vec, h) - expected).abs() < 1e-12);
}

#[test]
fn kernel_function_is_radially_symmetric() {
    // Two vectors with the same norm but different directions must yield
    // the same kernel value, since W depends only on |r_vec|.
    let h = 1.0;
    let a = Vector3::new(1.0, 0.0, 0.0);
    let b = Vector3::new(0.0, 0.0, -1.0);
    assert!(
        (LinearTestKernel::kernel_function(&a, h) - LinearTestKernel::kernel_function(&b, h)).abs()
            < 1e-12
    );
}

#[test]
fn kernel_function_scales_as_one_over_support_radius_to_the_dimension() {
    // W(k*r_vec, k*h) = (1/k^DIMENSION) * W(r_vec, h), since q is unchanged
    // by simultaneous scaling but the 1/h^DIMENSION prefactor is not.
    // Catches e.g. an accidental off-by-one in the exponent.
    let r_vec = Vector3::new(1.0, 0.0, 0.0);
    let h = 2.0;
    let k = 3.0;

    let base = LinearTestKernel::kernel_function(&r_vec, h);
    let scaled = LinearTestKernel::kernel_function(&(k * r_vec), k * h);

    assert!((scaled - base / k.powi(LinearTestKernel::DIMENSION)).abs() < 1e-9);
}

// ─── kernel_gradient ──────────────────────────────────────────────────────

#[test]
fn kernel_gradient_is_exactly_zero_at_zero_distance() {
    // r_vec / r is undefined (0/0) at r == 0; the trait's default
    // implementation must guard against this explicitly rather than
    // propagating a NaN.
    let grad = LinearTestKernel::kernel_gradient(&Vector3::zeros(), 1.0);
    assert_eq!(grad, Vector3::zeros());
}

#[test]
fn kernel_gradient_matches_the_documented_formula() {
    let r_vec = Vector3::new(1.0, 0.0, 0.0);
    let h = 2.0;
    let r = r_vec.norm();
    let q = LinearTestKernel::q(&r_vec, h);
    let expected = LinearTestKernel::ALPHA / h.powi(LinearTestKernel::DIMENSION + 1) * r_vec / r
        * LinearTestKernel::d_q_w(q);
    assert_vec_approx(
        LinearTestKernel::kernel_gradient(&r_vec, h),
        expected,
        1e-12,
    );
}

#[test]
fn kernel_gradient_is_antisymmetric_under_negating_r_vec() {
    // ∇_i W(r_i - r_j) == -∇_i W(r_j - r_i): flipping r_vec must flip the
    // gradient's sign, since q(-r_vec, h) == q(r_vec, h) (only the norm
    // matters) while r_vec/r flips sign.
    let r_vec = Vector3::new(1.0, 2.0, 3.0);
    let h = 1.5;
    let grad = LinearTestKernel::kernel_gradient(&r_vec, h);
    let grad_neg = LinearTestKernel::kernel_gradient(&(-r_vec), h);
    assert_vec_approx(grad_neg, -grad, 1e-12);
}

#[test]
fn kernel_gradient_is_parallel_to_r_vec() {
    // The gradient formula is a scalar multiple of r_vec (the
    // direction-dependence is entirely captured by `r_vec / r`), so its
    // cross product with r_vec must vanish.
    let r_vec = Vector3::new(1.0, 2.0, -3.0);
    let h = 1.0;
    let grad = LinearTestKernel::kernel_gradient(&r_vec, h);
    assert!(grad.cross(&r_vec).norm() < 1e-12);
}

#[test]
fn kernel_gradient_scales_as_one_over_support_radius_to_the_dimension_plus_one() {
    // ∇W(k*r_vec, k*h) = (1/k^(DIMENSION+1)) * ∇W(r_vec, h). Catches e.g.
    // an accidental use of `DIMENSION` instead of `DIMENSION + 1` in the
    // exponent.
    let r_vec = Vector3::new(1.0, 0.0, 0.0);
    let h = 2.0;
    let k = 2.0;

    let base = LinearTestKernel::kernel_gradient(&r_vec, h);
    let scaled = LinearTestKernel::kernel_gradient(&(k * r_vec), k * h);

    assert_vec_approx(scaled, base / k.powi(LinearTestKernel::DIMENSION + 1), 1e-9);
}
