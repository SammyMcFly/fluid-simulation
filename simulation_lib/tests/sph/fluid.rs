use nalgebra::{Point3, Vector3};
use simulation_lib::sph::fluid::{Fluid, FluidCheckpoint, Len, SerFluidCheckpoint};

fn pos(x: f64, y: f64, z: f64) -> Point3<f64> {
    Point3::new(x, y, z)
}

fn vel(x: f64, y: f64, z: f64) -> Vector3<f64> {
    Vector3::new(x, y, z)
}

/// Builds a `Fluid` with `n` active particles using only public API — the
/// `FluidCheckpoint -> Fluid` conversion is the only way to construct a
/// `Fluid` with specific particle data from outside the crate, since
/// `Fluid` has no public `push` and `add_samples` requires actual mesh
/// geometry.
fn make_fluid(
    positions: Vec<Point3<f64>>,
    velocities: Vec<Vector3<f64>>,
    masses: Vec<f64>,
) -> Fluid {
    let n = positions.len();
    FluidCheckpoint {
        fluid_id: vec![0; n],
        position: positions,
        velocity: velocities,
        mass: masses,
        ..Default::default()
    }
    .into()
}

// ─── Fluid: Len / Default ─────────────────────────────────────────

#[test]
fn fluid_default_is_empty() {
    let fluid = Fluid::default();
    assert_eq!(fluid.len(), 0);
    assert!(fluid.is_empty());
    assert_eq!(fluid.total_len(), 0);
}

// ─── Fluid: construction via FluidCheckpoint ──────────────────────

#[test]
fn fluid_from_checkpoint_has_correct_len() {
    let fluid = make_fluid(
        vec![pos(1.0, 2.0, 3.0)],
        vec![vel(0.1, 0.2, 0.3)],
        vec![5.0],
    );

    assert_eq!(fluid.len(), 1);
    assert_eq!(fluid.total_len(), 1);
    assert!(!fluid.is_empty());
}

#[test]
fn fluid_from_checkpoint_stores_correct_values() {
    let fluid = make_fluid(
        vec![pos(1.0, 2.0, 3.0)],
        vec![vel(0.1, 0.2, 0.3)],
        vec![5.0],
    );

    assert_eq!(fluid.position[0], pos(1.0, 2.0, 3.0));
    assert_eq!(fluid.velocity[0], vel(0.1, 0.2, 0.3));
    assert_eq!(fluid.mass[0], 5.0);
    assert_eq!(fluid.fluid_id[0], 0);
}

#[test]
fn fluid_from_checkpoint_resets_derived_fields() {
    // `FluidCheckpoint` only carries `fluid_id`/`position`/`velocity`/
    // `mass` — deliberately not `acceleration`/`volume`/`pressure`, which
    // are transient, step-local integration state recomputed by the solver
    // every time step, not part of the physical state that needs to survive
    // a checkpoint/resume. The same holds for the scratch slot pools
    // (`integrator_position_slots`/`integrator_velocity_slots`/
    // `solver_position_slots`/`solver_velocity_slots`), which are left
    // empty here and only sized afterward via `resize_slots` once the
    // active `IntegrationScheme`/`PressureSolver` are known — see
    // `fluid_from_checkpoint_leaves_slot_pools_empty`.
    let fluid = make_fluid(
        vec![pos(1.0, 2.0, 3.0)],
        vec![vel(1.0, 1.0, 1.0)],
        vec![5.0],
    );

    assert_eq!(fluid.acceleration[0], Vector3::zeros());
    assert_eq!(fluid.volume[0], 0.0);
    assert_eq!(fluid.pressure[0], 0.0);
}

#[test]
fn fluid_from_checkpoint_leaves_solver_slots_empty_regardless_of_input() {
    // Unlike `integrator_*_slots` (which `FluidCheckpoint` now carries and
    // restores -- see the tests below), `solver_*_slots` have no
    // corresponding field on `FluidCheckpoint` at all, so they come back
    // empty unconditionally, by construction, not just because this
    // particular checkpoint happened not to set them.
    let fluid = make_fluid(
        vec![pos(1.0, 2.0, 3.0)],
        vec![vel(1.0, 1.0, 1.0)],
        vec![5.0],
    );

    assert!(fluid.solver_position_slots.is_empty());
    assert!(fluid.solver_velocity_slots.is_empty());
}

#[test]
fn fluid_from_checkpoint_multiple_particles() {
    let fluid = make_fluid(
        vec![pos(1.0, 0.0, 0.0), pos(2.0, 0.0, 0.0), pos(3.0, 0.0, 0.0)],
        vec![Vector3::zeros(); 3],
        vec![1.0, 2.0, 3.0],
    );

    assert_eq!(fluid.len(), 3);
    assert_eq!(fluid.total_len(), 3);
    assert_eq!(fluid.mass[0], 1.0);
    assert_eq!(fluid.mass[1], 2.0);
    assert_eq!(fluid.mass[2], 3.0);
}

// ─── Fluid: disable / drop_inactive ───────────────────────────────

#[test]
fn fluid_disable_decreases_active_count() {
    let mut fluid = make_fluid(
        vec![pos(1.0, 0.0, 0.0), pos(2.0, 0.0, 0.0), pos(3.0, 0.0, 0.0)],
        vec![Vector3::zeros(); 3],
        vec![1.0, 2.0, 3.0],
    );

    fluid.disable(1);

    assert_eq!(fluid.len(), 2);
    assert_eq!(fluid.total_len(), 3);
}

#[test]
fn fluid_disable_swaps_with_last_active() {
    let mut fluid = make_fluid(
        vec![pos(1.0, 0.0, 0.0), pos(2.0, 0.0, 0.0), pos(3.0, 0.0, 0.0)],
        vec![Vector3::zeros(); 3],
        vec![1.0, 2.0, 3.0],
    );

    fluid.disable(0);

    assert_eq!(fluid.position[0], pos(3.0, 0.0, 0.0));
    assert_eq!(fluid.mass[0], 3.0);
    assert_eq!(fluid.position[2], pos(1.0, 0.0, 0.0));
}

#[test]
fn fluid_disable_swaps_fluid_id() {
    // The existing `fluid_disable_swaps_with_last_active` test only checks
    // `position`/`mass` — it never verifies `fluid_id` is swapped too. This
    // matters because `fluid_id` is the newest field and the one most
    // likely to be forgotten in a future edit to `Fluid::swap`; a missed
    // swap here would silently mix up particles between fluid phases
    // without producing an obviously wrong result (positions/velocities
    // would still integrate correctly, but `reconstruct_surfaces` would
    // group particles under the wrong fluid_id).
    let fluid: Fluid = FluidCheckpoint {
        fluid_id: vec![10, 20, 30],
        position: vec![pos(1.0, 0.0, 0.0), pos(2.0, 0.0, 0.0), pos(3.0, 0.0, 0.0)],
        velocity: vec![Vector3::zeros(); 3],
        mass: vec![1.0, 2.0, 3.0],
        ..Default::default()
    }
    .into();
    let mut fluid = fluid;

    fluid.disable(0);

    assert_eq!(fluid.fluid_id[0], 30);
    assert_eq!(fluid.fluid_id[2], 10);
}

#[test]
fn fluid_disable_last_active_no_swap() {
    let mut fluid = make_fluid(
        vec![pos(1.0, 0.0, 0.0), pos(2.0, 0.0, 0.0)],
        vec![Vector3::zeros(); 2],
        vec![1.0, 2.0],
    );

    fluid.disable(1); // disable last → swap(1, 1) → no-op swap

    assert_eq!(fluid.len(), 1);
    assert_eq!(fluid.position[0], pos(1.0, 0.0, 0.0));
}

#[test]
#[should_panic]
fn fluid_disable_out_of_range_panics() {
    let mut fluid = make_fluid(vec![pos(1.0, 0.0, 0.0)], vec![Vector3::zeros()], vec![1.0]);
    fluid.disable(1); // id >= num_active
}

#[test]
#[should_panic]
fn fluid_disable_panics_when_all_already_disabled() {
    // `disable`'s assertion is `id < self.num_active`. With `num_active`
    // already at 0, even `id = 0` fails this check (`0 < 0` is false) —
    // distinct from `fluid_disable_out_of_range_panics`, which tests an
    // out-of-range `id` while active particles still remain.
    let mut fluid = make_fluid(vec![pos(1.0, 0.0, 0.0)], vec![Vector3::zeros()], vec![1.0]);
    fluid.disable(0); // num_active becomes 0
    fluid.disable(0); // 0 < 0 → panics
}

#[test]
fn fluid_drop_inactive() {
    let mut fluid = make_fluid(
        vec![pos(1.0, 0.0, 0.0), pos(2.0, 0.0, 0.0), pos(3.0, 0.0, 0.0)],
        vec![Vector3::zeros(); 3],
        vec![1.0, 2.0, 3.0],
    );

    fluid.disable(1);
    assert_eq!(fluid.total_len(), 3);

    fluid.drop_inactive();
    assert_eq!(fluid.total_len(), 2);
    assert_eq!(fluid.len(), 2);
    assert_eq!(fluid.position.len(), 2);
    assert_eq!(fluid.mass.len(), 2);
    assert_eq!(fluid.velocity.len(), 2);
}

#[test]
fn fluid_disable_all() {
    let mut fluid = make_fluid(
        vec![pos(1.0, 0.0, 0.0), pos(2.0, 0.0, 0.0)],
        vec![Vector3::zeros(); 2],
        vec![1.0, 2.0],
    );

    fluid.disable(0);
    fluid.disable(0); // the swapped particle is now at 0

    assert_eq!(fluid.len(), 0);
    assert!(fluid.is_empty());
    assert_eq!(fluid.total_len(), 2);
}

// ─── Fluid: resize_slots ───────────────────────────────────────────

#[test]
fn resize_slots_creates_the_requested_number_of_pools_sized_to_total_len() {
    let mut fluid = make_fluid(
        vec![pos(1.0, 0.0, 0.0), pos(2.0, 0.0, 0.0)],
        vec![Vector3::zeros(); 2],
        vec![1.0, 1.0],
    );

    fluid.resize_slots(1, 2, 0, 1);

    assert_eq!(fluid.integrator_position_slots.len(), 1);
    assert_eq!(fluid.integrator_velocity_slots.len(), 2);
    assert_eq!(fluid.solver_position_slots.len(), 0);
    assert_eq!(fluid.solver_velocity_slots.len(), 1);

    for slot in &fluid.integrator_position_slots {
        assert_eq!(slot.len(), 2);
        assert!(slot.iter().all(|p| *p == Point3::origin()));
    }
    for slot in &fluid.integrator_velocity_slots {
        assert_eq!(slot.len(), 2);
        assert!(slot.iter().all(|v| *v == Vector3::zeros()));
    }
    for slot in &fluid.solver_velocity_slots {
        assert_eq!(slot.len(), 2);
    }
}

#[test]
fn resize_slots_can_be_called_again_to_change_the_pool_counts() {
    let mut fluid = make_fluid(vec![pos(1.0, 0.0, 0.0)], vec![Vector3::zeros()], vec![1.0]);

    fluid.resize_slots(2, 0, 0, 0);
    assert_eq!(fluid.integrator_position_slots.len(), 2);

    fluid.resize_slots(1, 1, 1, 1);
    assert_eq!(fluid.integrator_position_slots.len(), 1);
    assert_eq!(fluid.integrator_velocity_slots.len(), 1);
    assert_eq!(fluid.solver_position_slots.len(), 1);
    assert_eq!(fluid.solver_velocity_slots.len(), 1);
}

#[test]
fn resize_slots_with_all_zero_counts_leaves_pools_empty() {
    let mut fluid = make_fluid(vec![pos(1.0, 0.0, 0.0)], vec![Vector3::zeros()], vec![1.0]);

    fluid.resize_slots(0, 0, 0, 0);

    assert!(fluid.integrator_position_slots.is_empty());
    assert!(fluid.integrator_velocity_slots.is_empty());
    assert!(fluid.solver_position_slots.is_empty());
    assert!(fluid.solver_velocity_slots.is_empty());
}

// ─── FluidCheckpoint / SerFluidCheckpoint: empty defaults ─────────

#[test]
fn fluid_checkpoint_default_is_empty() {
    let checkpoint = FluidCheckpoint::default();
    let fluid: Fluid = checkpoint.into();
    assert_eq!(fluid.len(), 0);
    assert!(fluid.is_empty());
}

#[test]
fn ser_fluid_checkpoint_default_is_empty() {
    let ser = SerFluidCheckpoint::default();
    let checkpoint: FluidCheckpoint = ser.into();
    assert!(checkpoint.position.is_empty());
}

// ─── FluidCheckpoint <-> SerFluidCheckpoint <-> Fluid conversions ──

#[test]
fn fluid_checkpoint_from_ser_fluid_checkpoint() {
    let ser = SerFluidCheckpoint {
        fluid_id: vec![1, 2],
        position: vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        velocity: vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        mass: vec![7.0, 8.0],
        ..Default::default()
    };

    let checkpoint: FluidCheckpoint = ser.into();

    assert_eq!(checkpoint.fluid_id, vec![1, 2]);
    assert_eq!(
        checkpoint.position,
        vec![pos(1.0, 2.0, 3.0), pos(4.0, 5.0, 6.0)]
    );
    assert_eq!(
        checkpoint.velocity,
        vec![vel(0.1, 0.2, 0.3), vel(0.4, 0.5, 0.6)]
    );
    assert_eq!(checkpoint.mass, vec![7.0, 8.0]);
}

#[test]
fn ser_fluid_checkpoint_from_fluid_checkpoint() {
    let checkpoint = FluidCheckpoint {
        fluid_id: vec![3, 4],
        position: vec![pos(1.0, 2.0, 3.0), pos(4.0, 5.0, 6.0)],
        velocity: vec![vel(0.1, 0.2, 0.3), vel(0.4, 0.5, 0.6)],
        mass: vec![5.0, 6.0],
        ..Default::default()
    };

    let ser: SerFluidCheckpoint = checkpoint.into();

    assert_eq!(ser.fluid_id, vec![3, 4]);
    assert_eq!(ser.position, vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    assert_eq!(ser.velocity, vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
    assert_eq!(ser.mass, vec![5.0, 6.0]);
}

#[test]
fn fluid_checkpoint_roundtrip() {
    let checkpoint = FluidCheckpoint {
        fluid_id: vec![0, 1],
        position: vec![pos(1.0, 2.0, 3.0), pos(4.0, 5.0, 6.0)],
        velocity: vec![vel(0.5, 0.5, 0.5), vel(1.0, 1.0, 1.0)],
        mass: vec![2.0, 3.0],
        ..Default::default()
    };

    let fluid: Fluid = checkpoint.clone().into();
    let restored: FluidCheckpoint = fluid.into();

    assert_eq!(restored.fluid_id, checkpoint.fluid_id);
    assert_eq!(restored.position, checkpoint.position);
    assert_eq!(restored.velocity, checkpoint.velocity);
    assert_eq!(restored.mass, checkpoint.mass);
}

#[test]
fn fluid_full_roundtrip_via_ser_fluid_checkpoint() {
    let fluid: Fluid = FluidCheckpoint {
        fluid_id: vec![9],
        position: vec![pos(1.0, 2.0, 3.0)],
        velocity: vec![vel(0.5, 0.5, 0.5)],
        mass: vec![2.0],
        ..Default::default()
    }
    .into();

    let ser: SerFluidCheckpoint = FluidCheckpoint::from(fluid.clone()).into();
    let restored: Fluid = FluidCheckpoint::from(ser).into();

    assert_eq!(restored.len(), fluid.len());
    assert_eq!(restored.fluid_id, fluid.fluid_id);
    assert_eq!(restored.position, fluid.position);
    assert_eq!(restored.velocity, fluid.velocity);
    assert_eq!(restored.mass, fluid.mass);
}

// ─── FluidCheckpoint: integrator slot persistence ──────────────────

#[test]
fn fluid_to_checkpoint_captures_populated_integrator_slots() {
    let mut fluid = make_fluid(
        vec![pos(1.0, 0.0, 0.0), pos(2.0, 0.0, 0.0)],
        vec![Vector3::zeros(); 2],
        vec![1.0, 1.0],
    );
    fluid.resize_slots(1, 1, 0, 0);
    fluid.integrator_position_slots[0] = vec![pos(9.0, 0.0, 0.0), pos(9.0, 1.0, 0.0)];
    fluid.integrator_velocity_slots[0] = vec![vel(1.0, 2.0, 3.0), vel(4.0, 5.0, 6.0)];

    let checkpoint: FluidCheckpoint = fluid.into();

    assert_eq!(
        checkpoint.integrator_position_slots,
        vec![vec![pos(9.0, 0.0, 0.0), pos(9.0, 1.0, 0.0)]]
    );
    assert_eq!(
        checkpoint.integrator_velocity_slots,
        vec![vec![vel(1.0, 2.0, 3.0), vel(4.0, 5.0, 6.0)]]
    );
}

#[test]
fn fluid_from_checkpoint_restores_integrator_slots_directly() {
    let checkpoint = FluidCheckpoint {
        fluid_id: vec![0],
        position: vec![pos(1.0, 0.0, 0.0)],
        velocity: vec![vel(0.0, 0.0, 0.0)],
        mass: vec![1.0],
        integrator_position_slots: vec![vec![pos(5.0, 0.0, 0.0)]],
        integrator_velocity_slots: vec![vec![vel(6.0, 0.0, 0.0)]],
    };

    let fluid: Fluid = checkpoint.into();

    assert_eq!(
        fluid.integrator_position_slots,
        vec![vec![pos(5.0, 0.0, 0.0)]]
    );
    assert_eq!(
        fluid.integrator_velocity_slots,
        vec![vec![vel(6.0, 0.0, 0.0)]]
    );
    assert!(fluid.solver_position_slots.is_empty());
    assert!(fluid.solver_velocity_slots.is_empty());
}

#[test]
fn fluid_checkpoint_round_trip_drops_solver_slots_regardless_of_content() {
    // The actual "shouldn't be persisted" behavior, demonstrated
    // end-to-end: populate ALL four pools with distinct, recognizable
    // values, round-trip through `FluidCheckpoint`, and confirm
    // integrator slots survive while solver slots are unconditionally
    // reset -- not because they happened to be unset, but because
    // `FluidCheckpoint` has no field to carry them through at all.
    let mut fluid = make_fluid(vec![pos(1.0, 0.0, 0.0)], vec![Vector3::zeros()], vec![1.0]);
    fluid.resize_slots(1, 1, 1, 1);
    fluid.integrator_position_slots[0][0] = pos(9.0, 0.0, 0.0);
    fluid.integrator_velocity_slots[0][0] = vel(9.0, 0.0, 0.0);
    fluid.solver_position_slots[0][0] = pos(-1.0, -1.0, -1.0);
    fluid.solver_velocity_slots[0][0] = vel(-1.0, -1.0, -1.0);

    let checkpoint: FluidCheckpoint = fluid.into();
    let restored: Fluid = checkpoint.into();

    assert_eq!(restored.integrator_position_slots[0][0], pos(9.0, 0.0, 0.0));
    assert_eq!(restored.integrator_velocity_slots[0][0], vel(9.0, 0.0, 0.0));
    assert!(restored.solver_position_slots.is_empty());
    assert!(restored.solver_velocity_slots.is_empty());
}

#[test]
fn fluid_checkpoint_round_trip_preserves_multiple_integrator_slots_independently() {
    // No `IntegrationScheme` today declares more than 1 position/velocity
    // slot, but the conversions iterate the outer `Vec` generically --
    // this catches an off-by-one, slot-order swap, or accidental
    // zip-to-shortest-length bug that a single-slot test could hide.
    let mut fluid = make_fluid(
        vec![pos(1.0, 0.0, 0.0), pos(2.0, 0.0, 0.0)],
        vec![Vector3::zeros(); 2],
        vec![1.0, 1.0],
    );
    fluid.resize_slots(2, 2, 0, 0);
    fluid.integrator_position_slots[0] = vec![pos(1.1, 0.0, 0.0), pos(1.2, 0.0, 0.0)];
    fluid.integrator_position_slots[1] = vec![pos(2.1, 0.0, 0.0), pos(2.2, 0.0, 0.0)];
    fluid.integrator_velocity_slots[0] = vec![vel(1.0, 0.0, 0.0), vel(2.0, 0.0, 0.0)];
    fluid.integrator_velocity_slots[1] = vec![vel(3.0, 0.0, 0.0), vel(4.0, 0.0, 0.0)];

    let checkpoint: FluidCheckpoint = fluid.clone().into();
    let restored: Fluid = checkpoint.into();

    assert_eq!(
        restored.integrator_position_slots,
        fluid.integrator_position_slots
    );
    assert_eq!(
        restored.integrator_velocity_slots,
        fluid.integrator_velocity_slots
    );
}

#[test]
fn fluid_checkpoint_round_trip_with_zero_integrator_slots_does_not_panic() {
    // E.g. `EulerCromer`/`ExplicitEuler`, which declare
    // `POSITION_SLOTS == VELOCITY_SLOTS == 0` -- confirms the
    // `.iter().map(...).collect()` chains in the `Ser` conversions handle
    // an empty outer `Vec` without panicking or misbehaving.
    let mut fluid = make_fluid(vec![pos(1.0, 0.0, 0.0)], vec![Vector3::zeros()], vec![1.0]);
    fluid.resize_slots(0, 0, 0, 0);

    let checkpoint: FluidCheckpoint = fluid.into();
    let ser: SerFluidCheckpoint = checkpoint.into();
    let restored_checkpoint: FluidCheckpoint = ser.into();
    let restored: Fluid = restored_checkpoint.into();

    assert!(restored.integrator_position_slots.is_empty());
    assert!(restored.integrator_velocity_slots.is_empty());
}

#[test]
fn fluid_checkpoint_to_ser_fluid_checkpoint_converts_integrator_slots_to_arrays() {
    let checkpoint = FluidCheckpoint {
        fluid_id: vec![0],
        position: vec![pos(1.0, 0.0, 0.0)],
        velocity: vec![vel(0.0, 0.0, 0.0)],
        mass: vec![1.0],
        integrator_position_slots: vec![vec![pos(1.0, 2.0, 3.0)]],
        integrator_velocity_slots: vec![vec![vel(4.0, 5.0, 6.0)]],
    };

    let ser: SerFluidCheckpoint = checkpoint.into();

    assert_eq!(ser.integrator_position_slots, vec![vec![[1.0, 2.0, 3.0]]]);
    assert_eq!(ser.integrator_velocity_slots, vec![vec![[4.0, 5.0, 6.0]]]);
}

#[test]
fn ser_fluid_checkpoint_to_fluid_checkpoint_converts_integrator_slots_from_arrays() {
    let ser = SerFluidCheckpoint {
        fluid_id: vec![0],
        position: vec![[1.0, 0.0, 0.0]],
        velocity: vec![[0.0, 0.0, 0.0]],
        mass: vec![1.0],
        integrator_position_slots: vec![vec![[1.0, 2.0, 3.0]]],
        integrator_velocity_slots: vec![vec![[4.0, 5.0, 6.0]]],
    };

    let checkpoint: FluidCheckpoint = ser.into();

    assert_eq!(
        checkpoint.integrator_position_slots,
        vec![vec![pos(1.0, 2.0, 3.0)]]
    );
    assert_eq!(
        checkpoint.integrator_velocity_slots,
        vec![vec![vel(4.0, 5.0, 6.0)]]
    );
}

#[test]
fn fluid_checkpoint_default_has_empty_integrator_slots() {
    let checkpoint = FluidCheckpoint::default();
    assert!(checkpoint.integrator_position_slots.is_empty());
    assert!(checkpoint.integrator_velocity_slots.is_empty());
}

#[test]
fn ser_fluid_checkpoint_default_has_empty_integrator_slots() {
    let ser = SerFluidCheckpoint::default();
    assert!(ser.integrator_position_slots.is_empty());
    assert!(ser.integrator_velocity_slots.is_empty());
}

#[test]
fn fluid_full_roundtrip_via_ser_fluid_checkpoint_preserves_integrator_slots() {
    // Regression test for the actual bug motivating this feature: before
    // `integrator_position_slots`/`integrator_velocity_slots` existed on
    // `FluidCheckpoint`, an integrator like `Verlet` (which stores x(t-1)
    // in `integrator_position_slots[0]`, read on every subsequent
    // `integrate()` call) would silently lose that history across a saved
    // `--state` file, restarting as if x(t-1) == the particle's current
    // position -- i.e. as if it had zero velocity at the moment of resume.
    let mut fluid = make_fluid(
        vec![pos(1.0, 0.0, 0.0), pos(2.0, 0.0, 0.0)],
        vec![vel(0.5, 0.0, 0.0), vel(-0.5, 0.0, 0.0)],
        vec![1.0, 1.0],
    );
    fluid.resize_slots(1, 0, 0, 0); // mirrors Verlet: 1 position slot, 0 velocity slots
    fluid.integrator_position_slots[0] = vec![pos(0.9, 0.0, 0.0), pos(2.1, 0.0, 0.0)];

    let ser: SerFluidCheckpoint = FluidCheckpoint::from(fluid.clone()).into();
    let restored: Fluid = FluidCheckpoint::from(ser).into();

    assert_eq!(
        restored.integrator_position_slots, fluid.integrator_position_slots,
        "x(t-1) must survive a save/restore cycle, or Verlet silently \
         behaves as if the particle had zero velocity at resume"
    );
    assert!(restored.integrator_velocity_slots.is_empty());
}

// ─── reconstruct_surfaces ──────────────────────────────────────────

#[test]
fn reconstruct_surfaces_empty_fluid_returns_empty_vec() {
    // With zero active particles, the grouping loop never executes and
    // `splashsurf`'s reconstruction is never invoked — this only exercises
    // the empty-input control flow, not the reconstruction itself, so it's
    // fast and has no dependency on `splashsurf`'s numerical behavior.
    let fluid = Fluid::default();
    let meshes = fluid.reconstruct_surfaces(0.1, 1000.0, 0.2);
    assert!(meshes.is_empty());
}
