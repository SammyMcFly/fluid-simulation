use nalgebra::{Matrix3, Point3, Unit, UnitQuaternion, Vector3};
use simulation_lib::sph::boundary_handling::{
    ForceOntoBoundary, RigidBodyMotion, RigidBodyMotionCheckpoint,
};
use std::f64::consts::FRAC_PI_2;

fn assert_scalar_close(a: f64, b: f64, eps: f64) {
    assert!((a - b).abs() < eps, "expected {b:?}, got {a:?}");
}

fn assert_vec_close(a: Vector3<f64>, b: Vector3<f64>, eps: f64) {
    assert!((a - b).norm() < eps, "expected {b:?}, got {a:?}");
}

fn assert_point_close(a: Point3<f64>, b: Point3<f64>, eps: f64) {
    assert!((a - b).norm() < eps, "expected {b:?}, got {a:?}");
}

/// Isotropic-inertia builder — most tests don't care about the inertia
/// tensor's exact shape, only about linear/momentum arithmetic, so identity
/// inertia keeps `angular_velocity` numerically equal to `angular_momentum`
/// regardless of orientation.
fn isotropic(
    mass: f64,
    com: Point3<f64>,
    orientation: UnitQuaternion<f64>,
    linear_velocity: Vector3<f64>,
    angular_velocity: Vector3<f64>,
) -> RigidBodyMotion {
    RigidBodyMotion::new(
        mass,
        Matrix3::identity(),
        Matrix3::identity(),
        com,
        orientation,
        linear_velocity,
        angular_velocity,
    )
}

// ─── new() / momentum construction ─────────────────────────────────

#[test]
fn new_computes_angular_momentum_from_inertia_and_angular_velocity() {
    let motion = RigidBodyMotion::new(
        1.0,
        Matrix3::from_diagonal(&Vector3::new(2., 3., 4.)),
        Matrix3::from_diagonal(&Vector3::new(0.5, 1. / 3., 0.25)),
        Point3::origin(),
        UnitQuaternion::identity(),
        Vector3::zeros(),
        Vector3::new(1., 2., 3.),
    );

    assert_vec_close(
        motion.get_checkpoint().angular_momentum,
        Vector3::new(2., 6., 12.),
        1e-12,
    );
    // Round-trips back exactly for identity orientation.
    assert_vec_close(motion.angular_velocity(), Vector3::new(1., 2., 3.), 1e-12);
}

// ─── pose / coordinate transforms ──────────────────────────────────

#[test]
fn pose_combines_translation_and_orientation() {
    let motion = isotropic(
        1.0,
        Point3::new(1., 2., 3.),
        UnitQuaternion::from_axis_angle(&Vector3::z_axis(), FRAC_PI_2),
        Vector3::zeros(),
        Vector3::zeros(),
    );

    // Body-frame origin maps to just the translation.
    assert_point_close(
        motion.local_to_world(&Point3::origin()),
        Point3::new(1., 2., 3.),
        1e-9,
    );
    // Body-frame (1,0,0) rotates to world (0,1,0), then translates.
    assert_point_close(
        motion.local_to_world(&Point3::new(1., 0., 0.)),
        Point3::new(1., 3., 3.),
        1e-9,
    );
}

#[test]
fn world_to_local_is_inverse_of_local_to_world() {
    let motion = isotropic(
        1.0,
        Point3::new(-2., 5., 0.5),
        UnitQuaternion::from_axis_angle(&Unit::new_normalize(Vector3::new(1., 1., 0.)), 1.234),
        Vector3::zeros(),
        Vector3::zeros(),
    );

    let p = Point3::new(3., -1., 2.);
    assert_point_close(motion.world_to_local(&motion.local_to_world(&p)), p, 1e-9);
    assert_point_close(motion.local_to_world(&motion.world_to_local(&p)), p, 1e-9);
}

#[test]
fn local_to_world_vector_ignores_translation() {
    let orientation = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), FRAC_PI_2);
    let a = isotropic(
        1.0,
        Point3::new(0., 0., 0.),
        orientation,
        Vector3::zeros(),
        Vector3::zeros(),
    );
    let b = isotropic(
        1.0,
        Point3::new(100., -50., 3.),
        orientation,
        Vector3::zeros(),
        Vector3::zeros(),
    );

    let v = Vector3::new(1., 0., 0.);
    assert_vec_close(
        a.local_to_world_vector(&v),
        b.local_to_world_vector(&v),
        1e-9,
    );
}

// ─── velocity_at_point ──────────────────────────────────────────────

#[test]
fn velocity_at_cm_equals_linear_velocity() {
    let motion = isotropic(
        1.0,
        Point3::origin(),
        UnitQuaternion::identity(),
        Vector3::new(3., 4., 5.),
        Vector3::zeros(),
    );
    assert_vec_close(motion.velocity_at_cm(), Vector3::new(3., 4., 5.), 1e-12);
}

#[test]
fn velocity_at_point_matches_rigid_body_formula() {
    // v(p) = v_cm + omega x (p - com)
    let motion = isotropic(
        1.0,
        Point3::origin(),
        UnitQuaternion::identity(),
        Vector3::new(1., 0., 0.),
        Vector3::new(0., 0., 2.),
    );

    assert_vec_close(
        motion.velocity_at_point(&Point3::new(1., 0., 0.)),
        Vector3::new(1., 2., 0.),
        1e-12,
    );
}

#[test]
fn velocity_at_point_uses_com_as_pivot_not_world_origin() {
    // The regression guard for this whole group: with the COM off the world
    // origin, omega x (p - com) = (0,1,0), whereas the lever arm measured
    // against the world origin would give (0,6,0). Every other
    // `velocity_at_point` test here places the COM at the origin, where those
    // two are indistinguishable.
    let motion = isotropic(
        1.0,
        Point3::new(5., 0., 0.),
        UnitQuaternion::identity(),
        Vector3::zeros(),
        Vector3::new(0., 0., 1.),
    );

    assert_vec_close(
        motion.velocity_at_point(&Point3::new(6., 0., 0.)),
        Vector3::new(0., 1., 0.),
        1e-12,
    );
}

#[test]
fn velocity_at_point_reduces_to_linear_velocity_at_the_com() {
    // Zero lever arm must kill the rotational term exactly, even with the COM
    // displaced and a fully general omega.
    let v_lin = Vector3::new(-1.5, 0.25, 3.);
    let com = Point3::new(5., -2., 1.);
    let motion = isotropic(
        1.0,
        com,
        UnitQuaternion::identity(),
        v_lin,
        Vector3::new(0.3, -0.7, 1.1),
    );

    assert_vec_close(motion.velocity_at_point(&com), v_lin, 1e-12);
    assert_vec_close(motion.velocity_at_point(&motion.center_of_mass()), v_lin, 1e-12);
}

#[test]
fn velocity_at_point_superposes_translation_and_rotation_off_origin() {
    let motion = isotropic(
        1.0,
        Point3::new(5., 0., 0.),
        UnitQuaternion::identity(),
        Vector3::new(0., 0., 2.),
        Vector3::new(0., 0., 1.),
    );

    // (0,0,2) + (0,0,1) x (1,0,0) = (0,0,2) + (0,1,0)
    assert_vec_close(
        motion.velocity_at_point(&Point3::new(6., 0., 0.)),
        Vector3::new(0., 1., 2.),
        1e-12,
    );
}

#[test]
fn velocity_at_point_is_antisymmetric_in_the_lever_arm() {
    // Diametrically opposite material points must straddle the COM velocity;
    // catches any nonlinearity or stray constant in the lever-arm term.
    let com = Point3::new(1., -4., 2.5);
    let motion = isotropic(
        1.0,
        com,
        UnitQuaternion::identity(),
        Vector3::new(0.7, 0., -0.2),
        Vector3::new(0.4, -1.3, 0.9),
    );

    let r = Vector3::new(2., 0.5, -1.5);
    let v_plus = motion.velocity_at_point(&(com + r));
    let v_minus = motion.velocity_at_point(&(com - r));

    assert_vec_close(0.5 * (v_plus + v_minus), motion.velocity_at_cm(), 1e-12);
}

#[test]
fn velocity_at_point_is_unaffected_along_the_rotation_axis() {
    // Displacement parallel to omega has zero cross product — this fails
    // loudly if the operands of `cross` are ever swapped, because a sign flip
    // elsewhere would not be caught by the magnitude-only checks above.
    let com = Point3::new(-3., 2., 7.);
    let omega = Vector3::new(0., 0., 2.5);
    let motion = isotropic(1.0, com, UnitQuaternion::identity(), Vector3::zeros(), omega);

    assert_vec_close(
        motion.velocity_at_point(&(com + omega * 3.)),
        Vector3::zeros(),
        1e-12,
    );
}

#[test]
fn velocity_field_satisfies_the_rigidity_constraint() {
    // (v_a - v_b) . (a - b) == 0 for any rigid motion: the distance between two
    // material points cannot change. This is the defining property the entire
    // velocity field is derived from, checked against arbitrary points.
    let com = Point3::new(2., -1., 0.5);
    let motion = isotropic(
        1.0,
        com,
        UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.83),
        Vector3::new(1., -2., 0.4),
        Vector3::new(0.6, 1.4, -0.9),
    );

    let a = Point3::new(4., 0., -2.);
    let b = Point3::new(-1., 3.5, 1.);
    let relative_velocity = motion.velocity_at_point(&a) - motion.velocity_at_point(&b);

    assert_scalar_close(relative_velocity.dot(&(a - b)), 0., 1e-11);
}

#[test]
fn velocity_at_point_crosses_world_frame_angular_velocity() {
    // Anisotropic inertia makes omega orientation-dependent, so this pins down
    // that `angular_velocity()` is a *world*-frame quantity, matching the
    // world-frame lever arm it is crossed with. With L = (0,1,0) and the body
    // rotated 90° about z, the world inverse inertia maps L to omega = (0,1,0)
    // scaled by the body-x eigenvalue 1.0 (body y-axis now points along -x).
    let mut motion = RigidBodyMotion::new(
        1.0,
        Matrix3::from_diagonal(&Vector3::new(1., 2., 3.)),
        Matrix3::from_diagonal(&Vector3::new(1., 0.5, 1. / 3.)),
        Point3::new(1., 1., 1.),
        UnitQuaternion::identity(),
        Vector3::zeros(),
        Vector3::zeros(),
    );

    motion.restore_from_checkpoint(&RigidBodyMotionCheckpoint {
        center_of_mass: Point3::new(1., 1., 1.),
        orientation: UnitQuaternion::from_axis_angle(&Vector3::z_axis(), FRAC_PI_2),
        linear_velocity: Vector3::zeros(),
        angular_momentum: Vector3::new(0., 1., 0.),
        force: Vector3::zeros(),
        torque: Vector3::zeros(),
    });

    // Cross-check the field against the omega the body reports, one unit out
    // along world x from the COM.
    let omega = motion.angular_velocity();
    let probe = motion.center_of_mass() + Vector3::new(1., 0., 0.);
    assert_vec_close(
        motion.velocity_at_point(&probe),
        omega.cross(&Vector3::new(1., 0., 0.)),
        1e-12,
    );
    // And that omega itself is the world-frame value, not the body-frame one
    // (which would be (0, 0.5, 0)).
    assert_vec_close(omega, Vector3::new(0., 1., 0.), 1e-9);
}

// ─── add_force / reset_forces / checkpoint force staging ────────────

#[test]
fn add_force_accumulates_force_and_torque() {
    let mut motion = isotropic(
        1.0,
        Point3::origin(),
        UnitQuaternion::identity(),
        Vector3::zeros(),
        Vector3::zeros(),
    );

    motion.add_force(ForceOntoBoundary {
        id: 0,
        force: Vector3::new(1., 0., 0.),
        force_location: Point3::new(1., 0., 0.),
    });
    motion.add_force(ForceOntoBoundary {
        id: 0,
        force: Vector3::new(0., 1., 0.),
        force_location: Point3::new(1., 0., 0.),
    });

    let cp = motion.get_checkpoint();
    assert_vec_close(cp.force, Vector3::new(1., 1., 0.), 1e-12);
    assert_vec_close(cp.torque, Vector3::new(0., 0., 1.), 1e-12);
}

#[test]
fn reset_forces_clears_pending_force_and_torque() {
    let mut motion = isotropic(
        1.0,
        Point3::origin(),
        UnitQuaternion::identity(),
        Vector3::zeros(),
        Vector3::zeros(),
    );
    motion.add_force(ForceOntoBoundary {
        id: 0,
        force: Vector3::new(1., 0., 0.),
        force_location: Point3::new(1., 0., 0.),
    });

    motion.reset_forces();

    let cp = motion.get_checkpoint();
    assert_vec_close(cp.force, Vector3::zeros(), 1e-12);
    assert_vec_close(cp.torque, Vector3::zeros(), 1e-12);
}

#[test]
fn get_checkpoint_captures_pending_force_before_step() {
    // Protects the doc comment on `RigidBodyMotionCheckpoint::force`: a
    // checkpoint taken right after `add_force` but before the next
    // `step_forward_in_time` must retain the not-yet-applied force.
    let mut motion = isotropic(
        1.0,
        Point3::origin(),
        UnitQuaternion::identity(),
        Vector3::zeros(),
        Vector3::zeros(),
    );
    motion.add_force(ForceOntoBoundary {
        id: 0,
        force: Vector3::new(4., 0., 0.),
        force_location: Point3::origin(),
    });

    let cp = motion.get_checkpoint();
    assert_vec_close(cp.force, Vector3::new(4., 0., 0.), 1e-12);
}

// ─── step_forward_in_time ────────────────────────────────────────────

#[test]
fn step_forward_in_time_integrates_linear_motion_semi_implicitly() {
    // Euler-Cromer: velocity is updated first, then position uses the
    // *new* velocity — so position ends up at x0 + (v0 + a*dt) * dt, not
    // x0 + v0*dt.
    let mut motion = isotropic(
        2.0,
        Point3::origin(),
        UnitQuaternion::identity(),
        Vector3::zeros(),
        Vector3::zeros(),
    );
    motion.add_force(ForceOntoBoundary {
        id: 0,
        force: Vector3::new(4., 0., 0.),
        force_location: Point3::origin(),
    });

    motion.step_forward_in_time(0.1);

    assert_vec_close(motion.velocity_at_cm(), Vector3::new(0.2, 0., 0.), 1e-12);
    assert_point_close(
        motion.pose().translation.vector.into(),
        Point3::new(0.02, 0., 0.),
        1e-12,
    );
}

#[test]
fn step_forward_in_time_resets_forces() {
    let mut motion = isotropic(
        1.0,
        Point3::origin(),
        UnitQuaternion::identity(),
        Vector3::zeros(),
        Vector3::zeros(),
    );
    motion.add_force(ForceOntoBoundary {
        id: 0,
        force: Vector3::new(1., 0., 0.),
        force_location: Point3::origin(),
    });

    motion.step_forward_in_time(0.1);

    let cp = motion.get_checkpoint();
    assert_vec_close(cp.force, Vector3::zeros(), 1e-12);
    assert_vec_close(cp.torque, Vector3::zeros(), 1e-12);
}

#[test]
fn step_forward_in_time_conserves_angular_velocity_without_torque() {
    let mut motion = isotropic(
        1.0,
        Point3::origin(),
        UnitQuaternion::identity(),
        Vector3::zeros(),
        Vector3::new(0., 0., 2.),
    );

    for _ in 0..5 {
        motion.step_forward_in_time(0.01);
        assert_vec_close(motion.angular_velocity(), Vector3::new(0., 0., 2.), 1e-12);
    }
}

#[test]
fn step_forward_in_time_rotates_orientation_approximately_as_expected() {
    let mut motion = isotropic(
        1.0,
        Point3::origin(),
        UnitQuaternion::identity(),
        Vector3::zeros(),
        Vector3::new(0., 0., 2.),
    );
    let dt = 0.001;

    motion.step_forward_in_time(dt);

    let orientation = motion.pose().rotation;
    // First-order (Euler + renormalize) integration of a constant angular
    // velocity — small `dt` keeps the discretization error well below the
    // tolerance used here.
    assert!((orientation.angle() - 2. * dt).abs() < 1e-4);
    let axis = orientation.axis().expect("nonzero rotation");
    assert_vec_close(axis.into_inner(), Vector3::new(0., 0., 1.), 1e-9);
}

#[test]
fn velocity_field_pivot_tracks_the_com_across_steps() {
    // Free translation plus constant spin: since the pivot moves with the
    // body, the velocity one unit ahead of the COM must stay constant. This
    // couples `velocity_at_point` to the integrator — a pivot left behind at
    // the starting position would drift linearly.
    let mut motion = isotropic(
        1.0,
        Point3::origin(),
        UnitQuaternion::identity(),
        Vector3::new(1., 0., 0.),
        Vector3::new(0., 0., 1.),
    );

    for _ in 0..50 {
        motion.step_forward_in_time(0.01);

        let probe = motion.center_of_mass() + Vector3::new(1., 0., 0.);
        assert_vec_close(
            motion.velocity_at_point(&probe),
            Vector3::new(1., 1., 0.),
            1e-10,
        );
    }

    // Sanity: the body actually travelled, so the loop above was not
    // trivially satisfied by a COM parked at the origin.
    assert_scalar_close(motion.center_of_mass().x, 0.5, 1e-10);
}

#[test]
fn orientation_integration_preserves_the_full_rotation_angle() {
    // One coarse step about z: the exponential map turns by exactly
    // omega*dt = 0.5 rad, whereas Euler-plus-normalize turns by only
    // 2*atan(0.25) = 0.48996 rad. The gap is ~1e-2, far above tolerance.
    let mut motion = isotropic(
        1.0,
        Point3::origin(),
        UnitQuaternion::identity(),
        Vector3::zeros(),
        Vector3::new(0., 0., 1.),
    );

    motion.step_forward_in_time(0.5);

    let rotation = motion.pose().rotation;
    assert_scalar_close(rotation.angle(), 0.5, 1e-12);
    assert_vec_close(rotation.scaled_axis(), Vector3::new(0., 0., 0.5), 1e-12);
}

#[test]
fn orientation_integration_has_no_cumulative_angle_deficit() {
    // 1000 steps of a full 2*pi rotation must return exactly to identity.
    // Euler-plus-normalize falls short by ~n*theta^3/12 = 2.1e-5 rad.
    let mut motion = isotropic(
        1.0,
        Point3::origin(),
        UnitQuaternion::identity(),
        Vector3::zeros(),
        Vector3::new(0., 0., std::f64::consts::TAU),
    );

    for _ in 0..1000 {
        motion.step_forward_in_time(1e-3);
    }

    assert_scalar_close(motion.pose().rotation.angle(), 0., 1e-11);
}

// ─── checkpoint / restore ────────────────────────────────────────────

#[test]
fn checkpoint_roundtrip_preserves_full_state() {
    let mut a = isotropic(
        1.0,
        Point3::new(1., 2., 3.),
        UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.5),
        Vector3::new(0.1, 0.2, 0.3),
        Vector3::new(0.4, 0.5, 0.6),
    );
    a.add_force(ForceOntoBoundary {
        id: 0,
        force: Vector3::new(1., 0., 0.),
        force_location: Point3::new(1., 2., 3.),
    });
    let checkpoint = a.get_checkpoint();

    let mut b = isotropic(
        1.0,
        Point3::origin(),
        UnitQuaternion::identity(),
        Vector3::zeros(),
        Vector3::zeros(),
    );
    b.restore_from_checkpoint(&checkpoint);

    assert_point_close(
        b.pose().translation.vector.into(),
        a.pose().translation.vector.into(),
        1e-12,
    );
    assert_vec_close(b.velocity_at_cm(), a.velocity_at_cm(), 1e-12);
    assert_vec_close(b.angular_velocity(), a.angular_velocity(), 1e-12);
    let (cp_a, cp_b) = (a.get_checkpoint(), b.get_checkpoint());
    assert_vec_close(cp_a.force, cp_b.force, 1e-12);
    assert_vec_close(cp_a.torque, cp_b.torque, 1e-12);
}

#[test]
fn restore_from_checkpoint_refreshes_derived_state_for_new_orientation() {
    // Anisotropic inertia + a 90° rotation gives distinct, exactly
    // computable expected values (0.5, 1.0, 0.0) that a stale-cache bug
    // (still using the OLD orientation) would visibly get wrong
    // (1.0, 0.5, 0.0) — the first two components swapped.
    let mut motion = RigidBodyMotion::new(
        1.0,
        Matrix3::from_diagonal(&Vector3::new(1., 2., 3.)),
        Matrix3::from_diagonal(&Vector3::new(1., 0.5, 1. / 3.)),
        Point3::origin(),
        UnitQuaternion::identity(),
        Vector3::zeros(),
        Vector3::zeros(),
    );

    motion.restore_from_checkpoint(&RigidBodyMotionCheckpoint {
        center_of_mass: Point3::origin(),
        orientation: UnitQuaternion::from_axis_angle(&Vector3::z_axis(), FRAC_PI_2),
        linear_velocity: Vector3::zeros(),
        angular_momentum: Vector3::new(1., 1., 0.),
        force: Vector3::zeros(),
        torque: Vector3::zeros(),
    });

    assert_vec_close(motion.angular_velocity(), Vector3::new(0.5, 1.0, 0.0), 1e-9);
}

#[test]
fn restore_from_checkpoint_preserves_pending_force_for_next_step() {
    // Direct regression test for the doc comment on
    // `RigidBodyMotionCheckpoint::force`: if `restore_from_checkpoint`
    // forgot to copy `force`/`torque`, this step would produce zero
    // acceleration instead of the expected (2, 0, 0).
    let mut motion = isotropic(
        2.0,
        Point3::origin(),
        UnitQuaternion::identity(),
        Vector3::zeros(),
        Vector3::zeros(),
    );

    motion.restore_from_checkpoint(&RigidBodyMotionCheckpoint {
        center_of_mass: Point3::origin(),
        orientation: UnitQuaternion::identity(),
        linear_velocity: Vector3::zeros(),
        angular_momentum: Vector3::zeros(),
        force: Vector3::new(4., 0., 0.),
        torque: Vector3::zeros(),
    });

    motion.step_forward_in_time(0.1);

    assert_vec_close(motion.velocity_at_cm(), Vector3::new(0.2, 0., 0.), 1e-12);
}
