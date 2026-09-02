use nalgebra::{Point3, Unit, UnitQuaternion, Vector3};
use simulation_lib::sph::boundary_handling::{
    BoundaryCheckpoint, RigidBodyMotionCheckpoint, SerBoundaryCheckpoint,
    SerRigidBodyMotionCheckpoint,
};
use std::f64::consts::FRAC_PI_4;

#[test]
fn rigid_body_motion_checkpoint_roundtrip_preserves_quaternion_component_order() {
    // (i, j, k, w) vs. nalgebra's `Quaternion::new(w, i, j, k)` constructor
    // order is an easy mismatch to introduce — use a rotation with four
    // distinct-ish components to make any axis mix-up visible.
    let checkpoint = RigidBodyMotionCheckpoint {
        center_of_mass: Point3::new(1., 2., 3.),
        orientation: UnitQuaternion::from_axis_angle(
            &Unit::new_normalize(Vector3::new(1., 2., 3.)),
            FRAC_PI_4,
        ),
        linear_velocity: Vector3::new(0.1, 0.2, 0.3),
        angular_momentum: Vector3::new(0.4, 0.5, 0.6),
        force: Vector3::new(0.7, 0.8, 0.9),
        torque: Vector3::new(1.0, 1.1, 1.2),
    };

    let ser: SerRigidBodyMotionCheckpoint = checkpoint.into();
    let restored: RigidBodyMotionCheckpoint = ser.into();

    assert!((restored.orientation.angle_to(&checkpoint.orientation)).abs() < 1e-12);
    assert_eq!(restored.center_of_mass, checkpoint.center_of_mass);
    assert_eq!(restored.linear_velocity, checkpoint.linear_velocity);
    assert_eq!(restored.angular_momentum, checkpoint.angular_momentum);
    assert_eq!(restored.force, checkpoint.force);
    assert_eq!(restored.torque, checkpoint.torque);
}

#[test]
fn boundary_checkpoint_roundtrip_preserves_static_and_dynamic_entries() {
    let dynamic = RigidBodyMotionCheckpoint {
        center_of_mass: Point3::new(1., 0., 0.),
        orientation: UnitQuaternion::identity(),
        linear_velocity: Vector3::zeros(),
        angular_momentum: Vector3::zeros(),
        force: Vector3::zeros(),
        torque: Vector3::zeros(),
    };

    // Mix of `None` (static boundary) and `Some` (dynamic boundary) entries,
    // exercising the `Vec<Option<_>>` mapping in both directions.
    let checkpoint = BoundaryCheckpoint {
        dynamic_states: vec![None, Some(dynamic), None],
    };

    let ser: SerBoundaryCheckpoint = checkpoint.into();
    assert_eq!(ser.dynamic_states.len(), 3);
    assert!(ser.dynamic_states[0].is_none());
    assert!(ser.dynamic_states[1].is_some());
    assert!(ser.dynamic_states[2].is_none());

    let restored: BoundaryCheckpoint = ser.into();
    assert_eq!(restored.dynamic_states.len(), 3);
    assert!(restored.dynamic_states[0].is_none());
    assert!(restored.dynamic_states[1].is_some());
    assert!(restored.dynamic_states[2].is_none());
    assert_eq!(
        restored.dynamic_states[1].unwrap().center_of_mass,
        Point3::new(1., 0., 0.)
    );
}
