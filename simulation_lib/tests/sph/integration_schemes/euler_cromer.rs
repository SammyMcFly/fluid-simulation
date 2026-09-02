//! Integration tests for `EulerCromer`, exercising only its public API.
//!
//! Focuses on the *documented contract* of the scheme (semi-implicit /
//! symplectic Euler: velocity is updated first, then position is updated
//! using the *new* velocity — not on incidental details of how the
//! position/velocity ping-pong buffers happen to be recycled internally.

use nalgebra::{Point3, Vector3};
use parry3d_f64::math::Vec3;
use parry3d_f64::shape::TriMesh;

use simulation_lib::sph::fluid::{Fluid, Len};
use simulation_lib::sph::integration_schemes::{EulerCromer, IntegrationScheme};

// ─── Fixtures ─────────────────────────────────────────────────────────────

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

/// Builds a `Fluid` with at least `min_n` sampled particles via the public
/// `add_samples` API. The exact sampled positions/velocities don't matter —
/// every test below overwrites the fields it cares about by index before
/// calling `integrate`.
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

fn set_particle_state(
    fluid: &mut Fluid,
    id: usize,
    position: Point3<f64>,
    velocity: Vector3<f64>,
    acceleration: Vector3<f64>,
) {
    fluid.position[id] = position;
    fluid.velocity[id] = velocity;
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

// ─── Basic semi-implicit-Euler formula ─────────────────────────────────────

#[test]
fn single_particle_matches_semi_implicit_euler_update() {
    let mut fluid = fluid_with_at_least(1);
    let pos0 = Point3::new(1.0, 2.0, 3.0);
    let vel0 = Vector3::new(0.5, -0.5, 1.0);
    let acc0 = Vector3::new(2.0, 0.0, -9.81);
    set_particle_state(&mut fluid, 0, pos0, vel0, acc0);

    let dt = 0.1;
    let expected_vel = vel0 + dt * acc0;
    let expected_pos = pos0 + dt * expected_vel;

    let mut scheme = EulerCromer;
    scheme.integrate(&mut fluid, dt);

    assert_vector_approx(fluid.velocity[0], expected_vel, 1e-9);
    assert_point_approx(fluid.position[0], expected_pos, 1e-9);
}

// ─── The defining "semi-implicit" property ────────────────────────────────

#[test]
fn position_update_uses_the_new_velocity_not_the_old_one() {
    // This is what distinguishes Euler-Cromer (semi-implicit / symplectic
    // Euler) from plain explicit Euler: position is advanced using the
    // *already-updated* velocity, not the velocity from the start of the
    // step. With acc != 0 the two schemes give different results, so this
    // test pins down which one `EulerCromer` actually implements.
    let mut fluid = fluid_with_at_least(1);
    let pos0 = Point3::new(0.0, 0.0, 0.0);
    let vel0 = Vector3::new(1.0, 0.0, 0.0);
    let acc0 = Vector3::new(10.0, 0.0, 0.0);
    set_particle_state(&mut fluid, 0, pos0, vel0, acc0);

    let dt = 1.0;
    let new_vel = vel0 + dt * acc0; // = (11, 0, 0)
    let semi_implicit_pos = pos0 + dt * new_vel; // = (11, 0, 0)
    let explicit_euler_pos = pos0 + dt * vel0; // = (1, 0, 0), NOT expected

    let mut scheme = EulerCromer;
    scheme.integrate(&mut fluid, dt);

    assert_point_approx(fluid.position[0], semi_implicit_pos, 1e-9);
    assert!(
        (fluid.position[0] - explicit_euler_pos).norm() > 1e-6,
        "result matches explicit Euler, not semi-implicit Euler: {:?}",
        fluid.position[0]
    );
}

// ─── Degenerate cases ───────────────────────────────────────────────────────

#[test]
fn zero_acceleration_reduces_to_straight_line_motion() {
    let mut fluid = fluid_with_at_least(1);
    let pos0 = Point3::new(0.0, 0.0, 0.0);
    let vel0 = Vector3::new(2.0, -1.0, 0.5);
    set_particle_state(&mut fluid, 0, pos0, vel0, Vector3::zeros());

    let dt = 0.3;
    let mut scheme = EulerCromer;
    scheme.integrate(&mut fluid, dt);

    assert_vector_approx(fluid.velocity[0], vel0, 1e-9);
    assert_point_approx(fluid.position[0], pos0 + dt * vel0, 1e-9);
}

#[test]
fn zero_time_step_leaves_position_and_velocity_unchanged() {
    let mut fluid = fluid_with_at_least(1);
    let pos0 = Point3::new(4.0, -2.0, 1.0);
    let vel0 = Vector3::new(1.0, 2.0, 3.0);
    let acc0 = Vector3::new(-5.0, 0.0, 9.81);
    set_particle_state(&mut fluid, 0, pos0, vel0, acc0);

    let mut scheme = EulerCromer;
    scheme.integrate(&mut fluid, 0.0);

    assert_eq!(fluid.velocity[0], vel0);
    assert_eq!(fluid.position[0], pos0);
}

// ─── Contract: acceleration is read, never modified ────────────────────────

#[test]
fn integrate_does_not_modify_acceleration() {
    // Per the `IntegrationScheme` contract, acceleration is expected to
    // already be computed by the caller before `integrate` runs; resetting
    // or overwriting it here would break that contract for callers relying
    // on it still being valid afterwards.
    let mut fluid = fluid_with_at_least(1);
    let acc0 = Vector3::new(3.0, -4.0, 5.0);
    set_particle_state(&mut fluid, 0, Point3::origin(), Vector3::zeros(), acc0);

    let mut scheme = EulerCromer;
    scheme.integrate(&mut fluid, 0.05);

    assert_eq!(fluid.acceleration[0], acc0);
}

// ─── Observable buffer semantics ───────────────────────────────────────────

#[test]
fn position_prev_and_velocity_prev_hold_the_pre_step_state() {
    let mut fluid = fluid_with_at_least(1);
    let pos0 = Point3::new(1.0, 1.0, 1.0);
    let vel0 = Vector3::new(0.2, 0.3, 0.4);
    set_particle_state(&mut fluid, 0, pos0, vel0, Vector3::new(1.0, 0.0, 0.0));

    let mut scheme = EulerCromer;
    scheme.integrate(&mut fluid, 0.1);

    assert_eq!(fluid.position_prev[0], pos0);
    assert_eq!(fluid.velocity_prev[0], vel0);
}

// ─── Multiple particles are independent ────────────────────────────────────

#[test]
fn multiple_particles_are_updated_independently() {
    let mut fluid = fluid_with_at_least(2);

    let pos_a = Point3::new(0.0, 0.0, 0.0);
    let vel_a = Vector3::new(1.0, 0.0, 0.0);
    let acc_a = Vector3::new(1.0, 0.0, 0.0);
    set_particle_state(&mut fluid, 0, pos_a, vel_a, acc_a);

    let pos_b = Point3::new(5.0, 5.0, 5.0);
    let vel_b = Vector3::new(0.0, -2.0, 0.0);
    let acc_b = Vector3::new(0.0, 0.0, 3.0);
    set_particle_state(&mut fluid, 1, pos_b, vel_b, acc_b);

    let dt = 0.2;
    let expected_vel_a = vel_a + dt * acc_a;
    let expected_pos_a = pos_a + dt * expected_vel_a;
    let expected_vel_b = vel_b + dt * acc_b;
    let expected_pos_b = pos_b + dt * expected_vel_b;

    let mut scheme = EulerCromer;
    scheme.integrate(&mut fluid, dt);

    assert_vector_approx(fluid.velocity[0], expected_vel_a, 1e-9);
    assert_point_approx(fluid.position[0], expected_pos_a, 1e-9);
    assert_vector_approx(fluid.velocity[1], expected_vel_b, 1e-9);
    assert_point_approx(fluid.position[1], expected_pos_b, 1e-9);
}

// ─── Multi-step composition (buffer-recycling regression check) ──────────

#[test]
fn repeated_steps_under_constant_acceleration_match_the_manual_recurrence() {
    // Regresses against the swap-based position/velocity buffer recycling:
    // calling `integrate` repeatedly must keep composing correctly, not
    // just work correctly for a single isolated call.
    let mut fluid = fluid_with_at_least(1);
    let g = Vector3::new(0.0, 0.0, -9.81);
    set_particle_state(&mut fluid, 0, Point3::origin(), Vector3::zeros(), g);

    let dt = 0.01;
    let steps = 5;

    // Independently compute the expected trajectory via the same
    // semi-implicit Euler recurrence: v_{n+1} = v_n + dt*g; x_{n+1} = x_n + dt*v_{n+1}.
    let mut expected_pos = Point3::origin();
    let mut expected_vel = Vector3::zeros();
    for _ in 0..steps {
        expected_vel += dt * g;
        expected_pos += dt * expected_vel;
    }

    let mut scheme = EulerCromer;
    for _ in 0..steps {
        scheme.integrate(&mut fluid, dt);
        // Acceleration is never touched by `integrate`, so no need to
        // reset it between steps for this isolated-integrator test.
    }

    assert_vector_approx(fluid.velocity[0], expected_vel, 1e-9);
    assert_point_approx(fluid.position[0], expected_pos, 1e-9);
}

// ─── Trait bounds / basic usability ─────────────────────────────────────────

fn assert_impls_integration_scheme<T: IntegrationScheme>() {}

#[test]
fn euler_cromer_implements_integration_scheme_default_and_clone() {
    assert_impls_integration_scheme::<EulerCromer>();
    let scheme = EulerCromer;
    let _cloned = scheme.clone();
}
