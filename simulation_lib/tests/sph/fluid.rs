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
    // `mass` — deliberately not `position_prev`/`position_pred`/
    // `velocity_prev`/`velocity_pred`/`acceleration`/`volume`/`pressure`,
    // which are transient, step-local integration state recomputed by the
    // solver every time step, not part of the physical state that needs to
    // survive a checkpoint/resume.
    let fluid = make_fluid(
        vec![pos(1.0, 2.0, 3.0)],
        vec![vel(1.0, 1.0, 1.0)],
        vec![5.0],
    );

    assert_eq!(fluid.position_prev[0], Point3::origin());
    assert_eq!(fluid.position_pred[0], Point3::origin());
    assert_eq!(fluid.velocity_prev[0], Vector3::zeros());
    assert_eq!(fluid.velocity_pred[0], Vector3::zeros());
    assert_eq!(fluid.acceleration[0], Vector3::zeros());
    assert_eq!(fluid.volume[0], 0.0);
    assert_eq!(fluid.pressure[0], 0.0);
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

// ─── Fluid: rotate_position / rotate_velocity ─────────────────────

#[test]
fn fluid_rotate_position() {
    let mut fluid = make_fluid(vec![pos(1.0, 0.0, 0.0)], vec![Vector3::zeros()], vec![1.0]);
    // `position_prev` defaults to `Point3::origin()` via the `FluidCheckpoint`
    // conversion (see `fluid_from_checkpoint_resets_derived_fields`); set
    // explicitly here so this test doesn't rely on that default.
    fluid.position_prev[0] = pos(0.0, 0.0, 0.0);
    fluid.position_pred[0] = pos(2.0, 0.0, 0.0);

    fluid.rotate_position();

    assert_eq!(fluid.position[0], pos(2.0, 0.0, 0.0)); // new position = old position_pred
    assert_eq!(fluid.position_prev[0], pos(1.0, 0.0, 0.0)); // new position_prev = old position
    assert_eq!(fluid.position_pred[0], pos(0.0, 0.0, 0.0)); // new position_pred = old position_prev
}

#[test]
fn fluid_rotate_velocity() {
    let mut fluid = make_fluid(
        vec![pos(0.0, 0.0, 0.0)],
        vec![vel(1.0, 0.0, 0.0)],
        vec![1.0],
    );
    fluid.velocity_pred[0] = vel(5.0, 0.0, 0.0);

    fluid.rotate_velocity();

    assert_eq!(fluid.velocity[0], vel(5.0, 0.0, 0.0));
    assert_eq!(fluid.velocity_prev[0], vel(1.0, 0.0, 0.0));
}

#[test]
fn fluid_rotate_position_multiple_particles() {
    let mut fluid = make_fluid(
        vec![pos(1.0, 0.0, 0.0), pos(2.0, 0.0, 0.0)],
        vec![Vector3::zeros(); 2],
        vec![1.0, 1.0],
    );

    fluid.position_pred[0] = pos(10.0, 0.0, 0.0);
    fluid.position_pred[1] = pos(20.0, 0.0, 0.0);

    fluid.rotate_position();

    assert_eq!(fluid.position[0], pos(10.0, 0.0, 0.0));
    assert_eq!(fluid.position[1], pos(20.0, 0.0, 0.0));
    assert_eq!(fluid.position_prev[0], pos(1.0, 0.0, 0.0));
    assert_eq!(fluid.position_prev[1], pos(2.0, 0.0, 0.0));
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
