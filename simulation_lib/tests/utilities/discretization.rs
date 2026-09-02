use nalgebra::{Point3, Vector3};
use simulation_lib::utilities::discretization::{
    CubicSerendipityDiscretization, EvaluationError, gauss_legendre_integrate,
};
use std::f64::consts::PI;

// ─── EvaluationError ────────────────────────────────────────────────

#[test]
fn evaluation_error_messages_are_distinct_and_nonempty() {
    let oob = EvaluationError::OutOfBounds.to_string();
    let pruned = EvaluationError::PrunedCell.to_string();
    assert!(!oob.is_empty());
    assert!(!pruned.is_empty());
    assert_ne!(oob, pruned);
}

// ─── gauss_legendre_integrate ────────────────────────────────────────

#[test]
fn gauss_legendre_integrate_constant_equals_ball_volume() {
    // Isolates angular (theta/phi) quadrature accuracy: the r^2 factor is
    // exactly polynomial, and sin(theta)/constant integrands are smooth
    // (entire), so Gauss-Legendre converges very fast — order=20 should
    // already be highly precise. Not verified by execution; loosen if
    // this fails in practice.
    let radius = 1.3;
    let f = |_p: &Point3<f64>| -> Result<f64, EvaluationError> { Ok(1.0) };
    let result = gauss_legendre_integrate(&f, &Point3::new(0.5, -0.2, 0.7), radius, 20).unwrap();
    let expected = 4.0 / 3.0 * PI * radius.powi(3);
    assert!(
        (result - expected).abs() / expected < 1e-6,
        "expected ≈{expected}, got {result}"
    );
}

#[test]
fn gauss_legendre_integrate_radial_linear_matches_closed_form() {
    // ∫∫∫_ball |p - center| dV = ∫0^R r·r² dr · ∫0^π sinθ dθ · ∫0^2π dφ
    //                          = (R⁴/4) · 2 · 2π = π R⁴
    let center = Point3::new(1.0, 2.0, -0.5);
    let radius = 0.8;
    let f = move |p: &Point3<f64>| -> Result<f64, EvaluationError> { Ok((p - center).norm()) };
    let result = gauss_legendre_integrate(&f, &center, radius, 24).unwrap();
    let expected = PI * radius.powi(4);
    assert!(
        (result - expected).abs() / expected < 1e-4,
        "expected ≈{expected}, got {result}"
    );
}

#[test]
fn gauss_legendre_integrate_propagates_error() {
    let f =
        |_p: &Point3<f64>| -> Result<f64, EvaluationError> { Err(EvaluationError::OutOfBounds) };
    let result = gauss_legendre_integrate(&f, &Point3::origin(), 1.0, 5);
    assert!(matches!(result, Err(EvaluationError::OutOfBounds)));
}

// ─── CubicSerendipityDiscretization::function ────────────────────────

#[test]
fn function_reproduces_constant_field_everywhere_in_domain() {
    let disc = CubicSerendipityDiscretization::new(
        Point3::new(-1., -1., -1.),
        Point3::new(1., 1., 1.),
        None,
        None,
        1.0, // 2 cells per axis
        &|_p: &Point3<f64>| Ok(3.5),
    );

    for &p in &[
        Point3::new(-1., -1., -1.),
        Point3::new(0., 0., 0.),
        Point3::new(0.7, -0.3, 0.9),
        Point3::new(1., 1., 1.),
    ] {
        let v = disc.function(&p).unwrap();
        assert!((v - 3.5).abs() < 1e-9, "at {p:?}: expected 3.5, got {v}");
    }
}

#[test]
fn function_approximately_reproduces_a_linear_field() {
    // Not verified by execution whether this cubic serendipity element
    // exactly reproduces affine functions (a common but not universal FE
    // property) — tolerance is intentionally generous; tighten once
    // confirmed empirically.
    let disc = CubicSerendipityDiscretization::new(
        Point3::new(-1., -1., -1.),
        Point3::new(1., 1., 1.),
        None,
        None,
        1.0,
        &|p: &Point3<f64>| Ok(2.0 * p.x - p.y + 0.5 * p.z + 1.0),
    );

    for &p in &[
        Point3::new(0., 0., 0.),
        Point3::new(0.4, -0.6, 0.2),
        Point3::new(-0.9, 0.9, -0.1),
    ] {
        let expected = 2.0 * p.x - p.y + 0.5 * p.z + 1.0;
        let v = disc.function(&p).unwrap();
        assert!(
            (v - expected).abs() < 1e-3,
            "at {p:?}: expected ≈{expected}, got {v}"
        );
    }
}

#[test]
fn function_out_of_bounds_below_and_above_domain() {
    let disc = CubicSerendipityDiscretization::new(
        Point3::new(0., 0., 0.),
        Point3::new(1., 1., 1.),
        None,
        None,
        1.0,
        &|_p: &Point3<f64>| Ok(0.0),
    );

    assert!(matches!(
        disc.function(&Point3::new(-0.01, 0.5, 0.5)),
        Err(EvaluationError::OutOfBounds)
    ));
    assert!(matches!(
        disc.function(&Point3::new(1.01, 0.5, 0.5)),
        Err(EvaluationError::OutOfBounds)
    ));
}

#[test]
fn function_at_exact_upper_boundary_is_valid_via_clamping() {
    let disc = CubicSerendipityDiscretization::new(
        Point3::new(0., 0., 0.),
        Point3::new(1., 1., 1.),
        None,
        None,
        1.0,
        &|_p: &Point3<f64>| Ok(4.2),
    );

    let v = disc
        .function(&Point3::new(1.0, 1.0, 1.0))
        .expect("point exactly at x_max should be valid (clamped to last cell)");
    assert!((v - 4.2).abs() < 1e-9);
}

#[test]
fn function_returns_pruned_cell_when_entire_domain_is_pruned() {
    let disc = CubicSerendipityDiscretization::new(
        Point3::new(0., 0., 0.),
        Point3::new(1., 1., 1.),
        Some(10.0),
        None,
        1.0,
        &|_p: &Point3<f64>| Ok(0.0), // every node value is 0.0 < 10.0
    );

    assert!(matches!(
        disc.function(&Point3::new(0.5, 0.5, 0.5)),
        Err(EvaluationError::PrunedCell)
    ));
}

#[test]
fn function_never_prunes_when_no_bounds_given() {
    let disc = CubicSerendipityDiscretization::new(
        Point3::new(0., 0., 0.),
        Point3::new(1., 1., 1.),
        None,
        None,
        1.0,
        &|_p: &Point3<f64>| Ok(-1000.0),
    );

    let v = disc.function(&Point3::new(0.5, 0.5, 0.5)).unwrap();
    assert!((v - (-1000.0)).abs() < 1e-6);
}

// ─── CubicSerendipityDiscretization::gradient ────────────────────────

#[test]
fn gradient_matches_known_linear_field_gradient_at_interior_point() {
    let disc = CubicSerendipityDiscretization::new(
        Point3::new(-1., -1., -1.),
        Point3::new(1., 1., 1.),
        None,
        None,
        1.0,
        &|p: &Point3<f64>| Ok(2.0 * p.x - p.y + 0.5 * p.z),
    );

    let grad = disc.gradient(&Point3::new(0., 0., 0.)).unwrap();
    let expected = Vector3::new(2.0, -1.0, 0.5);
    assert!(
        (grad - expected).norm() < 1e-2,
        "expected ≈{expected:?}, got {grad:?}"
    );
}

#[test]
fn gradient_errors_near_domain_boundary_instead_of_falling_back() {
    // Public-API-level demonstration of the doc/implementation mismatch
    // flagged in `directional_derivative`'s internal tests: no graceful
    // one-sided fallback near a domain boundary, just a propagated error.
    let disc = CubicSerendipityDiscretization::new(
        Point3::new(-1., -1., -1.),
        Point3::new(1., 1., 1.),
        None,
        None,
        2.0, // single cell, h = dx/6 = 1/3
        &|_p: &Point3<f64>| Ok(0.0),
    );

    // p.x + h = 0.9 + 1/3 ≈ 1.233, outside x_max = 1.0.
    let result = disc.gradient(&Point3::new(0.9, 0., 0.));
    assert!(matches!(result, Err(EvaluationError::OutOfBounds)));
}
