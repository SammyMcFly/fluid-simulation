use core::f64;
use simulation_lib::fluid::{Fluid3D, SerFluid3D, Len, Positional};
use nalgebra::Vector3;

fn v(x: f64, y: f64, z: f64) -> Vector3<f64> {
    Vector3::new(x, y, z)
}

// ─── Fluid3D: Len trait ─────────────────────────────────────────────

#[test]
fn fluid_default_is_empty() {
    let fluid = Fluid3D::default();
    assert_eq!(fluid.len(), 0);
    assert!(fluid.is_empty());
    assert_eq!(fluid.total_len(), 0);
}

// ─── Fluid3D: Expandable trait ──────────────────────────────────────

#[test]
fn fluid_push_increases_len() {
    let mut fluid = Fluid3D::default();
    fluid.push(v(1.0, 2.0, 3.0), v(0.1, 0.2, 0.3), 1.0);

    assert_eq!(fluid.len(), 1);
    assert_eq!(fluid.total_len(), 1);
    assert!(!fluid.is_empty());
}

#[test]
fn fluid_push_stores_correct_values() {
    let mut fluid = Fluid3D::default();
    fluid.push(v(1.0, 2.0, 3.0), v(0.1, 0.2, 0.3), 5.0);

    assert_eq!(fluid.position[0], v(1.0, 2.0, 3.0));
    assert_eq!(fluid.position_prev[0], v(1.0, 2.0, 3.0));
    assert_eq!(fluid.velocity[0], v(0.1, 0.2, 0.3));
    assert_eq!(fluid.mass[0], 5.0);
    assert_eq!(fluid.volume[0], 0.0);
    assert_eq!(fluid.pressure[0], 0.0);
    assert_eq!(fluid.acceleration[0], Vector3::zeros());
}

#[test]
fn fluid_push_multiple() {
    let mut fluid = Fluid3D::default();
    fluid.push(v(1.0, 0.0, 0.0), Vector3::zeros(), 1.0);
    fluid.push(v(2.0, 0.0, 0.0), Vector3::zeros(), 2.0);
    fluid.push(v(3.0, 0.0, 0.0), Vector3::zeros(), 3.0);

    assert_eq!(fluid.len(), 3);
    assert_eq!(fluid.total_len(), 3);
    assert_eq!(fluid.mass[0], 1.0);
    assert_eq!(fluid.mass[1], 2.0);
    assert_eq!(fluid.mass[2], 3.0);
}

#[test]
fn fluid_extend() {
    let mut fluid_a = Fluid3D::default();
    fluid_a.push(v(1.0, 0.0, 0.0), Vector3::zeros(), 1.0);
    fluid_a.push(v(2.0, 0.0, 0.0), Vector3::zeros(), 2.0);

    let mut fluid_b = Fluid3D::default();
    fluid_b.push(v(3.0, 0.0, 0.0), Vector3::zeros(), 3.0);

    fluid_a.extend(fluid_b);

    assert_eq!(fluid_a.len(), 3);
    assert_eq!(fluid_a.total_len(), 3);
    assert_eq!(fluid_a.position[2], v(3.0, 0.0, 0.0));
    assert_eq!(fluid_a.mass[2], 3.0);
}

#[test]
#[should_panic]
fn fluid_extend_panics_with_inactive() {
    let mut fluid_a = Fluid3D::default();
    fluid_a.push(v(1.0, 0.0, 0.0), Vector3::zeros(), 1.0);
    fluid_a.push(v(2.0, 0.0, 0.0), Vector3::zeros(), 2.0);
    fluid_a.disable(0); // now num_active < total_len

    let mut fluid_b = Fluid3D::default();
    fluid_b.push(v(3.0, 0.0, 0.0), Vector3::zeros(), 3.0);

    fluid_a.extend(fluid_b); // should panic
}

// ─── Fluid3D: Positional trait ──────────────────────────────────────

#[test]
fn fluid_pos_now_single() {
    let mut fluid = Fluid3D::default();
    fluid.push(v(5.0, 6.0, 7.0), Vector3::zeros(), 1.0);

    assert_eq!(*fluid.pos_now(0), v(5.0, 6.0, 7.0));
}

#[test]
fn fluid_pos_now_range() {
    let mut fluid = Fluid3D::default();
    fluid.push(v(1.0, 0.0, 0.0), Vector3::zeros(), 1.0);
    fluid.push(v(2.0, 0.0, 0.0), Vector3::zeros(), 1.0);
    fluid.push(v(3.0, 0.0, 0.0), Vector3::zeros(), 1.0);

    let slice = fluid.pos_now(0..2);
    assert_eq!(slice.len(), 2);
    assert_eq!(slice[0], v(1.0, 0.0, 0.0));
    assert_eq!(slice[1], v(2.0, 0.0, 0.0));
}

// ─── Fluid3D: disable / drop_inactive ───────────────────────────────

#[test]
fn fluid_disable_decreases_active_count() {
    let mut fluid = Fluid3D::default();
    fluid.push(v(1.0, 0.0, 0.0), Vector3::zeros(), 1.0);
    fluid.push(v(2.0, 0.0, 0.0), Vector3::zeros(), 2.0);
    fluid.push(v(3.0, 0.0, 0.0), Vector3::zeros(), 3.0);

    fluid.disable(1);

    assert_eq!(fluid.len(), 2);
    assert_eq!(fluid.total_len(), 3);
}

#[test]
fn fluid_disable_swaps_with_last_active() {
    let mut fluid = Fluid3D::default();
    fluid.push(v(1.0, 0.0, 0.0), Vector3::zeros(), 1.0);
    fluid.push(v(2.0, 0.0, 0.0), Vector3::zeros(), 2.0);
    fluid.push(v(3.0, 0.0, 0.0), Vector3::zeros(), 3.0);

    fluid.disable(0);

    // Particle at index 0 should now be what was at index 2
    assert_eq!(fluid.position[0], v(3.0, 0.0, 0.0));
    assert_eq!(fluid.mass[0], 3.0);
    // Disabled particle moved to index 2
    assert_eq!(fluid.position[2], v(1.0, 0.0, 0.0));
}

#[test]
fn fluid_disable_last_active_no_swap() {
    let mut fluid = Fluid3D::default();
    fluid.push(v(1.0, 0.0, 0.0), Vector3::zeros(), 1.0);
    fluid.push(v(2.0, 0.0, 0.0), Vector3::zeros(), 2.0);

    fluid.disable(1); // disable last → swap(1,1) → no-op swap

    assert_eq!(fluid.len(), 1);
    assert_eq!(fluid.position[0], v(1.0, 0.0, 0.0));
}

#[test]
#[should_panic]
fn fluid_disable_out_of_range_panics() {
    let mut fluid = Fluid3D::default();
    fluid.push(v(1.0, 0.0, 0.0), Vector3::zeros(), 1.0);
    fluid.disable(1); // id >= num_active
}

#[test]
fn fluid_drop_inactive() {
    let mut fluid = Fluid3D::default();
    fluid.push(v(1.0, 0.0, 0.0), Vector3::zeros(), 1.0);
    fluid.push(v(2.0, 0.0, 0.0), Vector3::zeros(), 2.0);
    fluid.push(v(3.0, 0.0, 0.0), Vector3::zeros(), 3.0);

    fluid.disable(1);
    assert_eq!(fluid.total_len(), 3);

    fluid.drop_inactive();
    assert_eq!(fluid.total_len(), 2);
    assert_eq!(fluid.len(), 2);
    assert_eq!(fluid.position.len(), 2);
    assert_eq!(fluid.mass.len(), 2);
    assert_eq!(fluid.velocity.len(), 2);
}

// ─── Fluid3D: rotate_position / rotate_velocity ─────────────────────

#[test]
fn fluid_rotate_position() {
    let mut fluid = Fluid3D::default();
    fluid.push(v(1.0, 0.0, 0.0), Vector3::zeros(), 1.0);
    fluid.position_pred[0] = v(2.0, 0.0, 0.0);

    // Before: position = [1,0,0], position_prev = [1,0,0], position_pred = [2,0,0]
    fluid.rotate_position();
    // After swap1: position_prev <-> position → position_prev = old position, position = old position_prev
    //   position_prev = [1,0,0], position = [1,0,0] (they were equal)
    // After swap2: position <-> position_pred → position = old position_pred
    //   position = [2,0,0], position_pred = [1,0,0]

    assert_eq!(fluid.position[0], v(2.0, 0.0, 0.0));
    assert_eq!(fluid.position_prev[0], v(1.0, 0.0, 0.0));
}

#[test]
fn fluid_rotate_velocity() {
    let mut fluid = Fluid3D::default();
    fluid.push(v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), 1.0);
    fluid.velocity_pred[0] = v(5.0, 0.0, 0.0);

    fluid.rotate_velocity();

    assert_eq!(fluid.velocity[0], v(5.0, 0.0, 0.0));
    assert_eq!(fluid.velocity_prev[0], v(1.0, 0.0, 0.0));
}

// ─── Fluid3D: push with inactive particles ──────────────────────────

#[test]
fn fluid_push_inserts_at_active_boundary() {
    let mut fluid = Fluid3D::default();
    fluid.push(v(1.0, 0.0, 0.0), Vector3::zeros(), 1.0);
    fluid.push(v(2.0, 0.0, 0.0), Vector3::zeros(), 2.0);
    fluid.push(v(3.0, 0.0, 0.0), Vector3::zeros(), 3.0);

    fluid.disable(1); // active: [3, _], inactive: [_, 2] at positions [3,2] active=2
    // Now num_active=2, total_len=3

    // Push new particle — should go to index 2 (num_active) after swap
    fluid.push(v(9.0, 0.0, 0.0), Vector3::zeros(), 9.0);

    assert_eq!(fluid.len(), 3);
    assert_eq!(fluid.total_len(), 4);
    // New particle inserted at index 2 (was num_active before push incremented it)
    assert_eq!(fluid.position[2], v(9.0, 0.0, 0.0));
    assert_eq!(fluid.mass[2], 9.0);
}

// ─── SerFluid3D <-> Fluid3D conversions ─────────────────────────────

#[test]
fn fluid_from_ser_fluid() {
    let ser = SerFluid3D {
        position: vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        velocity: vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        mass: 7.0,
    };

    let fluid: Fluid3D = ser.into();

    assert_eq!(fluid.len(), 2);
    assert_eq!(fluid.position[0], v(1.0, 2.0, 3.0));
    assert_eq!(fluid.position[1], v(4.0, 5.0, 6.0));
    assert_eq!(fluid.velocity[0], v(0.1, 0.2, 0.3));
    assert_eq!(fluid.velocity[1], v(0.4, 0.5, 0.6));
    assert_eq!(fluid.mass[0], 7.0);
    assert_eq!(fluid.mass[1], 7.0);
}

#[test]
fn ser_fluid_from_fluid() {
    let mut fluid = Fluid3D::default();
    fluid.push(v(1.0, 2.0, 3.0), v(0.1, 0.2, 0.3), 5.0);
    fluid.push(v(4.0, 5.0, 6.0), v(0.4, 0.5, 0.6), 5.0);

    let ser: SerFluid3D = fluid.into();

    assert_eq!(ser.position, vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    assert_eq!(ser.velocity, vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
    assert_eq!(ser.mass, 5.0);
}

#[test]
fn fluid_roundtrip_conversion() {
    let mut original = Fluid3D::default();
    original.push(v(1.0, 2.0, 3.0), v(0.5, 0.5, 0.5), 2.0);
    original.push(v(4.0, 5.0, 6.0), v(1.0, 1.0, 1.0), 2.0);

    let ser: SerFluid3D = original.clone().into();
    let restored: Fluid3D = ser.into();

    assert_eq!(restored.len(), original.len());
    for i in 0..original.len() {
        assert_eq!(restored.position[i], original.position[i]);
        assert_eq!(restored.velocity[i], original.velocity[i]);
        assert_eq!(restored.mass[i], original.mass[i]);
    }
}

// ─── Edge cases ─────────────────────────────────────────────────────

#[test]
fn fluid_disable_all() {
    let mut fluid = Fluid3D::default();
    fluid.push(v(1.0, 0.0, 0.0), Vector3::zeros(), 1.0);
    fluid.push(v(2.0, 0.0, 0.0), Vector3::zeros(), 2.0);

    fluid.disable(0);
    fluid.disable(0); // the swapped particle is now at 0

    assert_eq!(fluid.len(), 0);
    assert!(fluid.is_empty());
    assert_eq!(fluid.total_len(), 2);
}

#[test]
fn fluid_disable_then_drop_then_push() {
    let mut fluid = Fluid3D::default();
    fluid.push(v(1.0, 0.0, 0.0), Vector3::zeros(), 1.0);
    fluid.push(v(2.0, 0.0, 0.0), Vector3::zeros(), 2.0);
    fluid.push(v(3.0, 0.0, 0.0), Vector3::zeros(), 3.0);

    fluid.disable(1);
    fluid.drop_inactive();

    assert_eq!(fluid.len(), 2);
    assert_eq!(fluid.total_len(), 2);

    fluid.push(v(99.0, 0.0, 0.0), Vector3::zeros(), 99.0);
    assert_eq!(fluid.len(), 3);
    assert_eq!(fluid.total_len(), 3);
    assert_eq!(fluid.position[2], v(99.0, 0.0, 0.0));
}

#[test]
fn fluid_rotate_position_multiple_particles() {
    let mut fluid = Fluid3D::default();
    fluid.push(v(1.0, 0.0, 0.0), Vector3::zeros(), 1.0);
    fluid.push(v(2.0, 0.0, 0.0), Vector3::zeros(), 1.0);

    fluid.position_pred[0] = v(10.0, 0.0, 0.0);
    fluid.position_pred[1] = v(20.0, 0.0, 0.0);

    fluid.rotate_position();

    assert_eq!(fluid.position[0], v(10.0, 0.0, 0.0));
    assert_eq!(fluid.position[1], v(20.0, 0.0, 0.0));
    assert_eq!(fluid.position_prev[0], v(1.0, 0.0, 0.0));
    assert_eq!(fluid.position_prev[1], v(2.0, 0.0, 0.0));
}
