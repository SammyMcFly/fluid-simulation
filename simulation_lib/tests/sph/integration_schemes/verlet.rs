//! Integration tests for `Verlet`, exercising only its public API.
//!
//! Standard (position) Störmer-Verlet integration:
//! `x_{n+1} = 2*x_n - x_{n-1} + dt^2 * a_n`, with velocity estimated
//! afterwards as `v_{n+1} = (x_{n+1} - x_n) / dt`.
//!
//! Per the internal buffer-rotation comments, at call time the scheme reads
//! `x_n` from what was `fluid.position` before the call, and `x_{n-1}` from
//! what was `fluid.position_prev` before the call — `fluid.position_pred`'s
//! pre-call value is irrelevant, since it is only used as scratch space to
//! be overwritten.

use nalgebra::{Point3, Vector3};
use parry3d_f64::math::Vec3;
use parry3d_f64::shape::TriMesh;

use simulation_lib::sph::fluid::{Fluid, Len};
use simulation_lib::sph::integration_schemes::{IntegrationScheme, Verlet};

fn cube_trimesh(side: f64) -> TriMesh {
    let h = side / 2.0;
    let positions = vec![
        Vec3::new(h, h, h),
        Vec3::new(h, h, -h),
        Vec3::new(h, -h, h),
        Vec3::new(h, -h, -h),
        Vec3::new(-h, h, h),
        Vec3::new(-h, h, -h),
        Vec3::new(-h, -h, h),
        Vec3::new(-h, -h, -h),
    ];
    let indices: Vec<[u32; 3]> = vec![
        [4, 2, 0],
        [2, 7, 3],
        [6, 5, 7],
        [1, 7, 5],
        [0, 3, 1],
        [4, 1, 5],
        [4, 6, 2],
        [2, 6, 7],
        [6, 4, 5],
        [1, 3, 7],
        [0, 2, 3],
        [4, 0, 1],
    ];
    TriMesh::new(positions, indices).expect("valid cube mesh")
}

fn fluid_with_at_least(min_n: usize) -> Fluid {
    let mesh = cube_trimesh(4.0);
    let mut fluid = Fluid::new();
    fluid.add_samples(&mesh, 0, 1000.0, 0.5);
    assert!(
        fluid.len() >= min_n,
        "expected at least {min_n} sampled particles, got {}",
        fluid.len()
    );
    fluid
}

/// Sets up particle `id` as if it were at step `n`, with `x_n = current`
/// and `x_{n-1} = previous`. The pre-call `position_pred` value is
/// deliberately left at its sampled default, since it must not influence
/// the result (see module doc comment).
fn set_particle_state(
    fluid: &mut Fluid,
    id: usize,
    current: Point3<f64>,
    previous: Point3<f64>,
    acceleration: Vector3<f64>,
) {
    fluid.position[id] = current;
    fluid.position_prev[id] = previous;
    fluid.acceleration[id] = acceleration;
}

fn assert_point_approx(actual: Point3<f64>, expected: Point3<f64>, eps: f64) {
    assert!(
        (actual - expected).norm() < eps,
        "expected {expected:?}, got {actual:?}"
    );
}

fn assert_vector_approx(actual: Vector3<f64>, expected: Vector3<f64>, eps: f64) {
    assert!(
        (actual - expected).norm() < eps,
        "expected {expected:?}, got {actual:?}"
    );
}

// ─── Basic Verlet formula ───────────────────────────────────────────────────

#[test]
fn single_particle_matches_stormer_verlet_update() {
    let mut fluid = fluid_with_at_least(1);
    let x_n = Point3::new(1.0, 0.0, 0.0);
    let x_prev = Point3::new(0.5, 0.0, 0.0);
    let acc = Vector3::new(0.0, -9.81, 0.0);
    set_particle_state(&mut fluid, 0, x_n, x_prev, acc);

    let dt = 0.1;
    let expected_pos = x_n + (x_n - x_prev) + dt * dt * acc; // 2*x_n - x_prev + dt^2*a
    let expected_vel = (expected_pos - x_n) / dt;

    let mut scheme = Verlet;
    scheme.integrate(&mut fluid, dt);

    assert_point_approx(fluid.position[0], expected_pos, 1e-9);
    assert_vector_approx(fluid.velocity[0], expected_vel, 1e-9);
}

#[test]
fn pre_call_position_pred_value_does_not_affect_the_result() {
    // `position_pred`'s pre-call value is pure scratch space that gets
    // overwritten; the Verlet formula must only depend on the pre-call
    // `position` (x_n) and `position_prev` (x_{n-1}).
    let x_n = Point3::new(2.0, 1.0, 0.0);
    let x_prev = Point3::new(1.0, 1.0, 0.0);
    let acc = Vector3::new(1.0, 0.0, 0.0);
    let dt = 0.05;

    let mut fluid_a = fluid_with_at_least(1);
    set_particle_state(&mut fluid_a, 0, x_n, x_prev, acc);
    fluid_a.position_pred[0] = Point3::new(100.0, 100.0, 100.0);

    let mut fluid_b = fluid_with_at_least(1);
    set_particle_state(&mut fluid_b, 0, x_n, x_prev, acc);
    fluid_b.position_pred[0] = Point3::new(-100.0, -100.0, -100.0);

    Verlet.integrate(&mut fluid_a, dt);
    Verlet.integrate(&mut fluid_b, dt);

    assert_point_approx(fluid_a.position[0], fluid_b.position[0], 1e-9);
    assert_vector_approx(fluid_a.velocity[0], fluid_b.velocity[0], 1e-9);
}

// ─── Degenerate cases ───────────────────────────────────────────────────────

#[test]
fn zero_acceleration_extrapolates_at_constant_step() {
    // With a = 0, standard Verlet reduces to constant-velocity
    // extrapolation: x_{n+1} = x_n + (x_n - x_{n-1}).
    let mut fluid = fluid_with_at_least(1);
    let x_n = Point3::new(3.0, 0.0, 0.0);
    let x_prev = Point3::new(1.0, 0.0, 0.0); // step of (2,0,0) per dt
    set_particle_state(&mut fluid, 0, x_n, x_prev, Vector3::zeros());

    let dt = 0.2;
    let mut scheme = Verlet;
    scheme.integrate(&mut fluid, dt);

    let expected_pos = x_n + (x_n - x_prev);
    assert_point_approx(fluid.position[0], expected_pos, 1e-9);
    // Velocity estimate should reproduce the constant step rate.
    assert_vector_approx(fluid.velocity[0], (x_n - x_prev) / dt, 1e-9);
}

#[test]
fn stationary_particle_with_zero_acceleration_stays_at_rest() {
    let mut fluid = fluid_with_at_least(1);
    let x = Point3::new(5.0, 5.0, 5.0);
    set_particle_state(&mut fluid, 0, x, x, Vector3::zeros());

    let mut scheme = Verlet;
    scheme.integrate(&mut fluid, 0.1);

    assert_point_approx(fluid.position[0], x, 1e-9);
    assert_vector_approx(fluid.velocity[0], Vector3::zeros(), 1e-9);
}

// ─── Contract: acceleration is read, never modified ────────────────────────

#[test]
fn integrate_does_not_modify_acceleration() {
    let mut fluid = fluid_with_at_least(1);
    let acc0 = Vector3::new(3.0, -4.0, 5.0);
    set_particle_state(&mut fluid, 0, Point3::origin(), Point3::origin(), acc0);

    let mut scheme = Verlet;
    scheme.integrate(&mut fluid, 0.05);

    assert_eq!(fluid.acceleration[0], acc0);
}

// ─── Observable buffer semantics ───────────────────────────────────────────

#[test]
fn position_prev_holds_the_pre_step_current_position() {
    let mut fluid = fluid_with_at_least(1);
    let x_n = Point3::new(1.0, 1.0, 1.0);
    let x_prev = Point3::new(0.0, 0.0, 0.0);
    set_particle_state(&mut fluid, 0, x_n, x_prev, Vector3::new(1.0, 0.0, 0.0));

    let mut scheme = Verlet;
    scheme.integrate(&mut fluid, 0.1);

    assert_eq!(fluid.position_prev[0], x_n);
}

// ─── Multiple particles are independent ────────────────────────────────────

#[test]
fn multiple_particles_are_updated_independently() {
    let mut fluid = fluid_with_at_least(2);

    let x_n_a = Point3::new(1.0, 0.0, 0.0);
    let x_prev_a = Point3::new(0.0, 0.0, 0.0);
    let acc_a = Vector3::new(1.0, 0.0, 0.0);
    set_particle_state(&mut fluid, 0, x_n_a, x_prev_a, acc_a);

    let x_n_b = Point3::new(0.0, 5.0, 0.0);
    let x_prev_b = Point3::new(0.0, 4.0, 0.0);
    let acc_b = Vector3::new(0.0, 0.0, 2.0);
    set_particle_state(&mut fluid, 1, x_n_b, x_prev_b, acc_b);

    let dt = 0.1;
    let mut scheme = Verlet;
    scheme.integrate(&mut fluid, dt);

    let expected_pos_a = x_n_a + (x_n_a - x_prev_a) + dt * dt * acc_a;
    let expected_pos_b = x_n_b + (x_n_b - x_prev_b) + dt * dt * acc_b;

    assert_point_approx(fluid.position[0], expected_pos_a, 1e-9);
    assert_point_approx(fluid.position[1], expected_pos_b, 1e-9);
}

// ─── Multi-step composition against a manual recurrence ───────────────────

#[test]
fn repeated_steps_under_constant_acceleration_match_the_manual_recurrence() {
    let mut fluid = fluid_with_at_least(1);
    let g = Vector3::new(0.0, 0.0, -9.81);
    let x0 = Point3::new(0.0, 0.0, 10.0);
    // Start at rest: x_{-1} == x_0 so the first step has zero initial velocity.
    set_particle_state(&mut fluid, 0, x0, x0, g);

    let dt = 0.01;
    let steps = 5;

    // Independently compute the expected trajectory via the same
    // Stormer-Verlet recurrence: x_{n+1} = 2*x_n - x_{n-1} + dt^2*g.
    let mut x_prev = x0;
    let mut x_curr = x0;
    for _ in 0..steps {
        let x_next = x_curr + (x_curr - x_prev) + dt * dt * g;
        x_prev = x_curr;
        x_curr = x_next;
    }
    let expected_vel = (x_curr - x_prev) / dt;

    let mut scheme = Verlet;
    for _ in 0..steps {
        scheme.integrate(&mut fluid, dt);
    }

    assert_point_approx(fluid.position[0], x_curr, 1e-6);
    assert_vector_approx(fluid.velocity[0], expected_vel, 1e-6);
}

// ─── Trait bounds / basic usability ─────────────────────────────────────────

fn assert_impls_integration_scheme<T: IntegrationScheme>() {}

#[test]
fn verlet_implements_integration_scheme_default_and_clone() {
    assert_impls_integration_scheme::<Verlet>();
    let scheme = Verlet;
    let _cloned = scheme.clone();
}
