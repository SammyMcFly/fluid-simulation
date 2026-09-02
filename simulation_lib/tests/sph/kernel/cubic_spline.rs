//! Integration tests for `CubicBSpline3D`, the concrete cubic B-spline
//! kernel used throughout the SPH pipeline. Beyond structural formula
//! checks (covered generically in `tests/kernel.rs`), this verifies the
//! actual mathematical properties a valid, normalized SPH kernel is
//! *supposed* to have: compact support, smoothness, non-negativity,
//! monotonic decay, and — most importantly — that it integrates to
//! exactly 1 over its support (the defining property of `ALPHA`).

use nalgebra::{Point3, Vector3};

use simulation_lib::sph::kernel::{CubicBSpline3D, KernelFn};
use simulation_lib::utilities::discretization::gauss_legendre_integrate;

// ─── Constants ────────────────────────────────────────────────────────────

#[test]
fn alpha_and_dimension_have_the_expected_values() {
    assert!((CubicBSpline3D::ALPHA - 16.0 / std::f64::consts::PI).abs() < 1e-12);
    assert_eq!(CubicBSpline3D::DIMENSION, 3);
}

// ─── w: known values, compact support, smoothness ─────────────────────────

#[test]
fn w_at_q_zero_has_the_expected_value() {
    // w(0) = 1^3 - 4*(1/2)^3 = 1 - 0.5 = 0.5
    assert!((CubicBSpline3D::w(0.0) - 0.5).abs() < 1e-12);
}

#[test]
fn w_at_q_one_half_has_the_expected_value() {
    // w(1/2) = (1/2)^3 - 4*0^3 = 0.125
    assert!((CubicBSpline3D::w(0.5) - 0.125).abs() < 1e-12);
}

#[test]
fn w_is_zero_at_and_beyond_the_support_radius() {
    assert_eq!(CubicBSpline3D::w(1.0), 0.0);
    assert_eq!(CubicBSpline3D::w(1.5), 0.0);
    assert_eq!(CubicBSpline3D::w(10.0), 0.0);
}

#[test]
fn w_is_nonnegative_on_the_support() {
    // A kernel weight must never go negative, or the resulting "density"
    // contributions would be unphysical.
    let mut q = 0.0;
    while q <= 1.0 {
        assert!(
            CubicBSpline3D::w(q) >= -1e-12,
            "w({q}) = {} is negative",
            CubicBSpline3D::w(q)
        );
        q += 0.01;
    }
}

#[test]
fn w_is_monotonically_decreasing_on_the_support() {
    let samples: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();
    for i in 1..samples.len() {
        assert!(
            CubicBSpline3D::w(samples[i]) <= CubicBSpline3D::w(samples[i - 1]) + 1e-12,
            "w is not monotonically decreasing at q={}",
            samples[i]
        );
    }
}

#[test]
fn w_is_continuous_at_the_piecewise_breakpoint() {
    let eps = 1e-9;
    let left = CubicBSpline3D::w(0.5 - eps);
    let right = CubicBSpline3D::w(0.5 + eps);
    assert!(
        (left - right).abs() < 1e-6,
        "discontinuity at q=1/2: {left} vs {right}"
    );
}

// ─── d_q_w: must be the actual derivative of w ─────────────────────────────

#[test]
fn d_q_w_matches_the_finite_difference_derivative_of_w() {
    let h = 1e-6;
    // Deliberately avoid sampling exactly AT the breakpoints (q=0.5, q=1.0)
    // to sidestep any finite-difference truncation artifacts right at a
    // kink; the kernel is documented/verified to be C1 there in principle,
    // but a tiny FD step straddling a kink is not a reliable way to check
    // that.
    for &q in &[0.05, 0.2, 0.35, 0.5 + 0.05, 0.7, 0.9] {
        let analytic = CubicBSpline3D::d_q_w(q);
        let numeric = (CubicBSpline3D::w(q + h) - CubicBSpline3D::w(q - h)) / (2.0 * h);
        assert!(
            (analytic - numeric).abs() < 1e-4,
            "at q={q}: analytic {analytic}, numeric {numeric}"
        );
    }
}

#[test]
fn d_q_w_is_zero_beyond_the_support_radius() {
    assert_eq!(CubicBSpline3D::d_q_w(1.0), 0.0);
    assert_eq!(CubicBSpline3D::d_q_w(2.0), 0.0);
}

// ─── kernel_function: compact support, radial symmetry ────────────────────

#[test]
fn kernel_function_is_zero_beyond_the_support_radius() {
    let h = 1.0;
    let r_vec = Vector3::new(1.0, 0.0, 0.0); // |r_vec| == h -> q == 1
    assert_eq!(CubicBSpline3D::kernel_function(&r_vec, h), 0.0);
    assert_eq!(
        CubicBSpline3D::kernel_function(&Vector3::new(2.0, 0.0, 0.0), h),
        0.0
    );
}

#[test]
fn kernel_function_is_radially_symmetric() {
    let h = 1.0;
    let a = Vector3::new(0.3, 0.0, 0.0);
    let b = Vector3::new(0.0, 0.3, 0.0);
    let c = Vector3::new(0.0, 0.0, -0.3);
    let wa = CubicBSpline3D::kernel_function(&a, h);
    let wb = CubicBSpline3D::kernel_function(&b, h);
    let wc = CubicBSpline3D::kernel_function(&c, h);
    assert!((wa - wb).abs() < 1e-12);
    assert!((wa - wc).abs() < 1e-12);
}

// ─── kernel_function: normalization (the defining property of ALPHA) ─────

#[test]
fn kernel_function_integrates_to_one_over_its_support() {
    // The whole point of `ALPHA = 16/pi` is that the kernel is a properly
    // normalized weighting function: ∫ W(x) d³x == 1 over the ball of
    // radius `support_radius`. This is the actual mathematical contract
    // an SPH kernel must satisfy (density estimates would be biased
    // otherwise) — not just an implementation detail.
    for &support_radius in &[0.5, 1.0, 2.0] {
        let center = Point3::new(1.0, -2.0, 0.5); // arbitrary, non-origin center
        let integral = gauss_legendre_integrate(
            &|p| {
                let r_vec = p - center;
                Ok(CubicBSpline3D::kernel_function(&r_vec, support_radius))
            },
            &center,
            support_radius,
            24,
        )
        .expect("kernel_function never errors");

        assert!(
            (integral - 1.0).abs() < 1e-3,
            "support_radius={support_radius}: integral = {integral}, expected ≈ 1.0"
        );
    }
}

// ─── kernel_gradient: singularity guard, consistency with kernel_function ──

#[test]
fn kernel_gradient_is_exactly_zero_at_zero_distance() {
    let grad = CubicBSpline3D::kernel_gradient(&Vector3::zeros(), 1.0);
    assert_eq!(grad, Vector3::zeros());
}

#[test]
fn kernel_gradient_is_zero_beyond_the_support_radius() {
    let h = 1.0;
    assert_eq!(
        CubicBSpline3D::kernel_gradient(&Vector3::new(1.5, 0.0, 0.0), h),
        Vector3::zeros()
    );
}

#[test]
fn kernel_gradient_matches_numerical_gradient_of_kernel_function() {
    // Central-difference check in each Cartesian direction, away from the
    // r=0 kink (where kernel_function is not differentiable in the usual
    // multivariable sense, even though the directional limit is zero —
    // see `kernel_gradient_is_exactly_zero_at_zero_distance` for that case).
    let h_support = 1.0;
    let points = [
        Vector3::new(0.3, 0.0, 0.0),
        Vector3::new(0.0, 0.2, 0.1),
        Vector3::new(-0.4, 0.1, 0.0),
    ];
    let eps = 1e-6;

    for r_vec in points {
        let analytic = CubicBSpline3D::kernel_gradient(&r_vec, h_support);

        let mut numeric = Vector3::zeros();
        for axis in 0..3 {
            let mut plus = r_vec;
            let mut minus = r_vec;
            plus[axis] += eps;
            minus[axis] -= eps;
            numeric[axis] = (CubicBSpline3D::kernel_function(&plus, h_support)
                - CubicBSpline3D::kernel_function(&minus, h_support))
                / (2.0 * eps);
        }

        assert!(
            (analytic - numeric).norm() < 1e-3,
            "at r_vec={r_vec:?}: analytic {analytic:?}, numeric {numeric:?}"
        );
    }
}

#[test]
fn kernel_gradient_is_antisymmetric_under_negating_r_vec() {
    let r_vec = Vector3::new(0.2, -0.1, 0.3);
    let h = 1.0;
    let grad = CubicBSpline3D::kernel_gradient(&r_vec, h);
    let grad_neg = CubicBSpline3D::kernel_gradient(&(-r_vec), h);
    assert!((grad_neg - (-grad)).norm() < 1e-12);
}

#[test]
fn kernel_gradient_points_toward_the_kernel_center() {
    // Since w is monotonically decreasing, d_q_w <= 0 throughout the
    // support, so the gradient (as a function of r_i, with r_vec = r_i -
    // r_j) must point from i towards j — i.e. opposite to r_vec — matching
    // the physical intuition that the kernel weight decreases as you move
    // away from the neighbor.
    let r_vec = Vector3::new(0.4, 0.0, 0.0);
    let grad = CubicBSpline3D::kernel_gradient(&r_vec, 1.0);
    assert!(
        grad.dot(&r_vec) <= 1e-12,
        "gradient {grad:?} does not point opposite to r_vec {r_vec:?}"
    );
}
