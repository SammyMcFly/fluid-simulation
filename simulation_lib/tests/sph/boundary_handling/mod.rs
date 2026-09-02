//! Integration tests for `sph::boundary_handling`'s module-level items:
//! `BoundaryHandlingVariant`, the `Boundary` trait's default `is_dynamic()`
//! method, `RequestMode`, `ForceOntoBoundary`, `BoundaryCheckpoint`,
//! `SerBoundaryCheckpoint`, and the `From` conversions between the two
//! checkpoint types.
//!
//! `boundary_handling/mod.rs` has no private fields or functions of its
//! own, so it needs no internal `#[cfg(test)]` block. The concrete
//! `BoundaryHandling` implementations (`VolumeMapBoundary`,
//! `StaticSampleBoundary`) already have their own dedicated test suites
//! and are intentionally not duplicated here.
//!
//! ASSUMPTION: `RigidBodyMotionCheckpoint`'s exact field layout wasn't
//! shown to me directly for this module — the fields used below
//! (`center_of_mass`, `orientation`, `linear_velocity`, `angular_momentum`,
//! `force`, `torque`) are inferred from their confirmed usage in
//! `static_sample_boundary`'s own (previously shared) internal test
//! module. `SerRigidBodyMotionCheckpoint`'s `Serialize`/`Deserialize`/
//! `Encode`/`Decode` derives and the existence of
//! `From<RigidBodyMotionCheckpoint> for SerRigidBodyMotionCheckpoint` are
//! inferred from the codebase's consistent `Ser*` pattern and from the
//! `d.map(Into::into)` call shown in `From<BoundaryCheckpoint> for
//! SerBoundaryCheckpoint`. If any of this differs, only the `Some(...)`-
//! based tests below need adjusting — the `None`-based tests do not rely
//! on it.
mod checkpoint;
mod rigid_body_motion;
mod static_sample_boundary;
mod volume_map_boundary;

use nalgebra::{Point3, UnitQuaternion, Vector3};
use serde::Deserialize;
use serde::de::IntoDeserializer;
use serde::de::value::{Error as ValueError, StrDeserializer};

use simulation_lib::sph::boundary_handling::{
    Boundary, BoundaryCheckpoint, BoundaryHandlingVariant, ForceOntoBoundary, RequestMode,
    RigidBodyMotionCheckpoint, SerBoundaryCheckpoint, SerRigidBodyMotionCheckpoint,
};

// ─── BoundaryHandlingVariant: Deserialize contract ─────────────────────

fn deserialize_variant(name: &str) -> Result<BoundaryHandlingVariant, ValueError> {
    let de: StrDeserializer<'_, ValueError> = name.into_deserializer();
    BoundaryHandlingVariant::deserialize(de)
}

#[test]
fn boundary_handling_variant_deserializes_every_documented_name() {
    assert!(matches!(
        deserialize_variant("StaticSampleBoundary"),
        Ok(BoundaryHandlingVariant::StaticSampleBoundary)
    ));
    assert!(matches!(
        deserialize_variant("VolumeMapBoundary"),
        Ok(BoundaryHandlingVariant::VolumeMapBoundary)
    ));
}

#[test]
fn boundary_handling_variant_rejects_an_unknown_name() {
    assert!(deserialize_variant("NotARealBoundaryHandling").is_err());
}

// ─── Boundary trait: default `is_dynamic()` ────────────────────────────

/// Minimal `Boundary` implementer used purely to exercise the trait's
/// default `is_dynamic()` method in isolation, independent of any
/// concrete implementation's (e.g. `BoundaryType`'s) own logic.
struct MockBoundary {
    center_of_mass: Option<Point3<f64>>,
}

impl Boundary for MockBoundary {
    fn get_neighbors(&self, _id: usize, _mode: RequestMode) -> &[usize] {
        &[]
    }
    fn position(&self, _id: usize) -> &Point3<f64> {
        unimplemented!("not exercised by these tests")
    }
    fn velocity(&self, _id: usize) -> &Vector3<f64> {
        unimplemented!("not exercised by these tests")
    }
    fn volume(&self, _id: usize) -> f64 {
        0.0
    }
    fn add_acceleration(&mut self, _acceleration: Vector3<f64>) {}
    fn center_of_mass(&self) -> Option<Point3<f64>> {
        self.center_of_mass
    }
}

#[test]
fn is_dynamic_default_is_true_exactly_when_center_of_mass_is_some() {
    let dynamic = MockBoundary {
        center_of_mass: Some(Point3::origin()),
    };
    let static_boundary = MockBoundary {
        center_of_mass: None,
    };
    assert!(dynamic.is_dynamic());
    assert!(!static_boundary.is_dynamic());
}

// ─── RequestMode ────────────────────────────────────────────────────────

#[test]
fn request_mode_default_is_normal() {
    assert!(matches!(RequestMode::default(), RequestMode::Normal));
}

#[test]
fn request_mode_is_copy_and_clone() {
    let mode = RequestMode::ViscosityAcceleration;
    let copied = mode; // Copy, not a move
    let cloned = mode;
    assert!(matches!(copied, RequestMode::ViscosityAcceleration));
    assert!(matches!(cloned, RequestMode::ViscosityAcceleration));
}

// ─── ForceOntoBoundary ──────────────────────────────────────────────────

#[test]
fn force_onto_boundary_stores_its_fields_verbatim() {
    let force = ForceOntoBoundary {
        id: 3,
        force: Vector3::new(1.0, 2.0, 3.0),
        force_location: Point3::new(4.0, 5.0, 6.0),
    };
    assert_eq!(force.id, 3);
    assert_eq!(force.force, Vector3::new(1.0, 2.0, 3.0));
    assert_eq!(force.force_location, Point3::new(4.0, 5.0, 6.0));
}

// ─── BoundaryCheckpoint / SerBoundaryCheckpoint: defaults ──────────────

#[test]
fn boundary_checkpoint_default_has_no_dynamic_states() {
    let checkpoint = BoundaryCheckpoint::default();
    assert!(checkpoint.dynamic_states.is_empty());
}

#[test]
fn ser_boundary_checkpoint_default_has_no_dynamic_states() {
    let checkpoint = SerBoundaryCheckpoint::default();
    assert!(checkpoint.dynamic_states.is_empty());
}

// ─── Conversions: BoundaryCheckpoint <-> SerBoundaryCheckpoint ─────────

#[test]
fn conversion_round_trip_preserves_length_and_order_of_none_entries() {
    let checkpoint = BoundaryCheckpoint {
        dynamic_states: vec![None, None, None],
    };
    let ser: SerBoundaryCheckpoint = checkpoint.into();
    assert_eq!(ser.dynamic_states.len(), 3);
    assert!(ser.dynamic_states.iter().all(Option::is_none));

    let restored: BoundaryCheckpoint = ser.into();
    assert_eq!(restored.dynamic_states.len(), 3);
    assert!(restored.dynamic_states.iter().all(Option::is_none));
}

#[test]
fn conversion_round_trip_preserves_mixed_none_and_some_entries_in_order() {
    let saved = RigidBodyMotionCheckpoint {
        center_of_mass: Point3::new(1.0, 2.0, 3.0),
        orientation: UnitQuaternion::identity(),
        linear_velocity: Vector3::new(0.1, 0.2, 0.3),
        angular_momentum: Vector3::new(0.0, 0.0, 1.0),
        force: Vector3::zeros(),
        torque: Vector3::zeros(),
    };

    let checkpoint = BoundaryCheckpoint {
        dynamic_states: vec![None, Some(saved), None],
    };

    let ser: SerBoundaryCheckpoint = checkpoint.into();
    assert_eq!(ser.dynamic_states.len(), 3);
    assert!(ser.dynamic_states[0].is_none());
    assert!(ser.dynamic_states[1].is_some());
    assert!(ser.dynamic_states[2].is_none());

    let restored: BoundaryCheckpoint = ser.into();
    assert!(
        restored.dynamic_states[0].is_none(),
        "the leading None entry must survive the round trip in place"
    );
    let restored_state = restored.dynamic_states[1]
        .as_ref()
        .expect("expected the Some entry to survive the round trip at the same index");
    assert_eq!(restored_state.center_of_mass, saved.center_of_mass);
    assert_eq!(restored_state.linear_velocity, saved.linear_velocity);
    assert!(
        restored.dynamic_states[2].is_none(),
        "the trailing None entry must survive the round trip in place"
    );
}

// ─── SerBoundaryCheckpoint: actual serialization formats ───────────────

#[test]
fn ser_boundary_checkpoint_round_trips_through_ron() {
    let ser = SerBoundaryCheckpoint {
        dynamic_states: vec![
            None,
            Some(SerRigidBodyMotionCheckpoint::from(
                RigidBodyMotionCheckpoint {
                    center_of_mass: Point3::new(7.0, 8.0, 9.0),
                    orientation: UnitQuaternion::identity(),
                    linear_velocity: Vector3::zeros(),
                    angular_momentum: Vector3::zeros(),
                    force: Vector3::zeros(),
                    torque: Vector3::zeros(),
                },
            )),
        ],
    };

    let text = ron::to_string(&ser).expect("failed to serialize to RON");
    let deserialized: SerBoundaryCheckpoint =
        ron::from_str(&text).expect("failed to deserialize from RON");

    assert_eq!(deserialized.dynamic_states.len(), 2);
    assert!(deserialized.dynamic_states[0].is_none());
    assert!(deserialized.dynamic_states[1].is_some());
}

#[test]
fn ser_boundary_checkpoint_round_trips_through_bincode() {
    // Exercises the `Encode`/`Decode` derive specifically, as a distinct
    // serialization path from the `Serialize`/`Deserialize` (RON) one
    // checked above.
    let ser = SerBoundaryCheckpoint {
        dynamic_states: vec![None],
    };

    let config = bincode::config::standard();
    let bytes = bincode::encode_to_vec(&ser, config).expect("failed to bincode-encode");
    let (decoded, _len): (SerBoundaryCheckpoint, usize) =
        bincode::decode_from_slice(&bytes, config).expect("failed to bincode-decode");

    assert_eq!(decoded.dynamic_states.len(), 1);
    assert!(decoded.dynamic_states[0].is_none());
}

#[test]
fn ser_boundary_checkpoint_of_empty_dynamic_states_round_trips() {
    let ser = SerBoundaryCheckpoint::default();
    let text = ron::to_string(&ser).expect("failed to serialize to RON");
    let deserialized: SerBoundaryCheckpoint =
        ron::from_str(&text).expect("failed to deserialize from RON");
    assert!(deserialized.dynamic_states.is_empty());
}
