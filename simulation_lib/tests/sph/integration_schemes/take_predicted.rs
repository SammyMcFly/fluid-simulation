//! Integration tests for `TakePredicted`, exercising only its public API.
//!
//! `TakePredicted` does no numerical integration at all: it simply commits
//! the already-computed predicted position/velocity (staged by a paired
//! `PressureSolver`, e.g. `IISPHwOST`, in `Fluid::integrator_position_slots`/
//! `integrator_velocity_slots`) as the new current state, via a plain
//! `mem::swap` against `fluid.position`/`fluid.velocity`. `dt` is documented
//! as unused. `POSITION_SLOTS`/`VELOCITY_SLOTS` are both `1`.

use nalgebra::{Point3, Vector3};
use parry3d_f64::math::Vec3;
use parry3d_f64::shape::TriMesh;

use simulation_lib::sph::fluid::{Fluid, Len};
use simulation_lib::sph::integration_schemes::{IntegrationScheme, TakePredicted};

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

/// Mirrors what `System::new_boxed` does via `TakePredicted::POSITION_SLOTS`/
/// `VELOCITY_SLOTS` before `integrate` ever runs on a `Fluid` — required
/// since `integrate` now asserts (`debug_assert_eq!`) that these pools are
/// sized to exactly `1` each.
fn with_integrator_slots(mut fluid: Fluid) -> Fluid {
    fluid.resize_slots(1, 1, 0, 0);
    fluid
}

// ─── Core contract: predicted state becomes the new current state ─────────

#[test]
fn position_becomes_the_old_predicted_position() {
    let mut fluid = with_integrator_slots(fluid_with_at_least(1));
    let pos_pred = Point3::new(9.0, 8.0, 7.0);
    fluid.integrator_position_slots[0][0] = pos_pred;

    let mut scheme = TakePredicted;
    scheme.integrate(&mut fluid, 0.1);

    assert_eq!(fluid.position[0], pos_pred);
}

#[test]
fn velocity_becomes_the_old_predicted_velocity() {
    let mut fluid = with_integrator_slots(fluid_with_at_least(1));
    let vel_pred = Vector3::new(1.0, -2.0, 3.0);
    fluid.integrator_velocity_slots[0][0] = vel_pred;

    let mut scheme = TakePredicted;
    scheme.integrate(&mut fluid, 0.1);

    assert_eq!(fluid.velocity[0], vel_pred);
}

#[test]
fn integrator_slots_hold_the_pre_step_current_state_after_committing() {
    // Since `integrate` is a plain `mem::swap`, there's no dedicated `_prev`
    // field anymore -- after committing, `integrator_position_slots[0]`/
    // `integrator_velocity_slots[0]` simply hold whatever `position`/
    // `velocity` were BEFORE this call, ready to be overwritten with a
    // fresh prediction (by the paired `PressureSolver`) before the next call.
    let mut fluid = with_integrator_slots(fluid_with_at_least(1));
    let pos0 = Point3::new(1.0, 2.0, 3.0);
    let vel0 = Vector3::new(0.5, 0.5, 0.5);
    fluid.position[0] = pos0;
    fluid.velocity[0] = vel0;
    fluid.integrator_position_slots[0][0] = Point3::new(9.0, 9.0, 9.0);
    fluid.integrator_velocity_slots[0][0] = Vector3::new(9.0, 9.0, 9.0);

    let mut scheme = TakePredicted;
    scheme.integrate(&mut fluid, 0.1);

    assert_eq!(fluid.integrator_position_slots[0][0], pos0);
    assert_eq!(fluid.integrator_velocity_slots[0][0], vel0);
}

// ─── dt is documented as unused ────────────────────────────────────────────

#[test]
fn result_is_independent_of_dt() {
    let make_fluid = || {
        let mut fluid = with_integrator_slots(fluid_with_at_least(1));
        fluid.position[0] = Point3::new(1.0, 1.0, 1.0);
        fluid.velocity[0] = Vector3::new(2.0, 2.0, 2.0);
        fluid.integrator_position_slots[0][0] = Point3::new(3.0, 3.0, 3.0);
        fluid.integrator_velocity_slots[0][0] = Vector3::new(4.0, 4.0, 4.0);
        fluid
    };

    let mut fluid_a = make_fluid();
    let mut fluid_b = make_fluid();

    TakePredicted.integrate(&mut fluid_a, 0.001);
    TakePredicted.integrate(&mut fluid_b, 1000.0);

    assert_eq!(fluid_a.position[0], fluid_b.position[0]);
    assert_eq!(fluid_a.velocity[0], fluid_b.velocity[0]);
}

// ─── Contract: acceleration is untouched ───────────────────────────────────

#[test]
fn integrate_does_not_modify_acceleration() {
    let mut fluid = with_integrator_slots(fluid_with_at_least(1));
    let acc0 = Vector3::new(3.0, -4.0, 5.0);
    fluid.acceleration[0] = acc0;

    let mut scheme = TakePredicted;
    scheme.integrate(&mut fluid, 0.05);

    assert_eq!(fluid.acceleration[0], acc0);
}

// ─── Multiple particles are independent ────────────────────────────────────

#[test]
fn multiple_particles_are_updated_independently() {
    let mut fluid = with_integrator_slots(fluid_with_at_least(2));

    fluid.integrator_position_slots[0][0] = Point3::new(1.0, 0.0, 0.0);
    fluid.integrator_velocity_slots[0][0] = Vector3::new(1.0, 0.0, 0.0);
    fluid.integrator_position_slots[0][1] = Point3::new(0.0, 2.0, 0.0);
    fluid.integrator_velocity_slots[0][1] = Vector3::new(0.0, 2.0, 0.0);

    let mut scheme = TakePredicted;
    scheme.integrate(&mut fluid, 0.1);

    assert_eq!(fluid.position[0], Point3::new(1.0, 0.0, 0.0));
    assert_eq!(fluid.velocity[0], Vector3::new(1.0, 0.0, 0.0));
    assert_eq!(fluid.position[1], Point3::new(0.0, 2.0, 0.0));
    assert_eq!(fluid.velocity[1], Vector3::new(0.0, 2.0, 0.0));
}

// ─── Repeated application composes the way a caller would expect ──────────

#[test]
fn repeated_calls_each_commit_whatever_was_predicted_since_the_last_call() {
    // Mirrors realistic usage: some solver (e.g. `IISPHwOST`) writes into
    // `integrator_position_slots[0]`/`integrator_velocity_slots[0]` between
    // calls to `integrate`, and each call commits exactly the most recently
    // written values via `mem::swap`.
    let mut fluid = with_integrator_slots(fluid_with_at_least(1));

    fluid.integrator_position_slots[0][0] = Point3::new(1.0, 0.0, 0.0);
    fluid.integrator_velocity_slots[0][0] = Vector3::new(1.0, 0.0, 0.0);
    let mut scheme = TakePredicted;
    scheme.integrate(&mut fluid, 0.1);
    assert_eq!(fluid.position[0], Point3::new(1.0, 0.0, 0.0));

    fluid.integrator_position_slots[0][0] = Point3::new(2.0, 0.0, 0.0);
    fluid.integrator_velocity_slots[0][0] = Vector3::new(2.0, 0.0, 0.0);
    scheme.integrate(&mut fluid, 0.1);
    assert_eq!(fluid.position[0], Point3::new(2.0, 0.0, 0.0));
    // The formerly-current position (1,0,0) is now held in the slot,
    // via `mem::swap` -- the old `position_prev` field no longer exists.
    assert_eq!(
        fluid.integrator_position_slots[0][0],
        Point3::new(1.0, 0.0, 0.0)
    );
}

// ─── Trait bounds / basic usability ─────────────────────────────────────────

fn assert_impls_integration_scheme<T: IntegrationScheme>() {}

#[test]
fn take_predicted_implements_integration_scheme_default_and_clone() {
    assert_impls_integration_scheme::<TakePredicted>();
    let scheme = TakePredicted;
    let _cloned = scheme.clone();
}
