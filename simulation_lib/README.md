# simulation_lib

Physics-based SPH (Smoothed Particle Hydrodynamics) fluid simulation backend. Provides all components needed to configure a scene, propagate an SPH system in time, take measurements, and produce renderable output (meshes or point samples) for a UI/renderer.

## Overview

This crate is generic over the numerical building blocks of an SPH simulation — kernel function, integration scheme, pressure solver, neighbor search algorithm, and boundary handling — and assembles a concrete simulation (`System`) from a scene/parameter configuration read from `.toml` files. The resulting system is exposed behind the object-safe [`SPHSystem`] trait so that the frontend/backend crates don't need to know the concrete generic instantiation.

## Module structure

| Module | Purpose |
|--------|---------|
| `fluid` | `Fluid` — collection of fluid samples (position, velocity, mass, volume, pressure, ...) and surface reconstruction via `splashsurf_lib`; `FluidCheckpoint`/`SerFluidCheckpoint` for snapshotting and (de)serialization |
| `sph` | Core simulation: `SPHSystem` trait, generic `System<K, I, P, N, B>`, `SystemCheckpoint`/`SerSystemCheckpoint`, system parameters/properties |
| `sph::kernel` | SPH kernel functions (`KernelFn` trait, `CubicBSpline3D`) |
| `sph::pressure_solver` | Pressure solvers: `SESPH`, `SESPHwSplitting`, `IISPH`, `IISPHwOST` |
| `sph::boundary_handling` | Boundary representations: `StaticSampleBoundary` (explicit samples), `VolumeMapBoundary` (implicit signed-distance/volume-map boundary); `RigidBodyMotion` for two-way coupled dynamic (rigid-body) boundaries, with `BoundaryCheckpoint`/`SerBoundaryCheckpoint` and `RigidBodyMotionState`/`SerRigidBodyMotionState` for snapshotting dynamic boundary state |
| `sph::non_pressure_accelerations` | Gravity and viscosity acceleration contributions |
| `sph::quantities` | SPH interpolation of scalar/vector quantities (volume, speed, density, pressure, kinetic energy) at arbitrary positions |
| `integration_schemes` | Time integrators: `ExplicitEuler`, `EulerCromer`, `Verlet`, `TakePredicted` |
| `neighbor_search` | `NeighborSearch` trait, `NeighborList`, and `SpatialHashing` implementation |
| `setup` | Scene/parameter loading (`input.rs`) and generic system construction (`SystemConstructor`, `new_boxed_system3d`) |
| `render_info` | `SimulationParameters`, `TimeStepInfo` and visualization types sent to the renderer/UI |
| `measurement` | `Measurement`, `MeasurementSeries`, `RecordingStatus` — recording physical quantities to CSV |
| `utilities` | `discretization` (cubic serendipity SDF/volume-map interpolation, Gauss-Legendre quadrature), `sampling` (volume/surface particle sampling), `triangle_mesh` (mesh loading, render mesh construction) |
| `iteration` | Internal `for_each!` macro abstracting over sequential (`iter_mut`) and parallel (`rayon::par_iter_mut`) execution depending on the `parallel` feature |

## Architecture

### Generic system composition

The core simulation type is generic over five abstract procedures:

```rust
pub struct System<K: KernelFn, I: IntegrationScheme, P: PressureSolver, N: NeighborSearch, B: BoundaryHandling> {
    fluid: Fluid,
    fluid_neighbor_list: NeighborList,
    boundary: B,
    integrator: I,
    pressure_solver: P,
    neighbor_search: N,
    parameters: SystemParameters,
    properties: CurrentSystemProperties,
    time_steps_propagated: u64,
    // ...
}
```

| Type parameter | Trait | Implementations |
|---|---|---|
| `K` | `KernelFn` | `CubicBSpline3D` |
| `I` | `IntegrationScheme` | `ExplicitEuler`, `EulerCromer`, `Verlet`, `TakePredicted` |
| `P` | `PressureSolver` | `SESPH`, `SESPHwSplitting`, `IISPH`, `IISPHwOST` |
| `N` | `NeighborSearch` | `SpatialHashing` |
| `B` | `BoundaryHandling` | `StaticSampleBoundary`, `VolumeMapBoundary` |

`System<K, I, P, N, B>` implements the object-safe, `dyn_clone`-able **`SPHSystem`** trait. Consumers (`sci-phi-backend`) only ever hold a `Box<dyn SPHSystem>` and never see the concrete generic instantiation.

Both `StaticSampleBoundary` and `VolumeMapBoundary` support **dynamic (rigid-body) boundaries** in addition to static ones: a `RigidBodyMotion` tracks pose, linear/angular velocity and accumulated force/torque from fluid-boundary coupling, integrated via a Euler-Cromer scheme each time step.

### Assembly from configuration

`setup::new_boxed_system3d` reads a `Procedures` enum selection (one variant per type parameter, deserialized from `.toml`) and nested macros (`with_integrator!` → `with_pressure!` → `with_neighbor!` → `with_boundary!`) expand to the matching concrete `System<...>`, wrapped in `Box<dyn SPHSystem>`:

```
Procedures (from .toml)
    └─ KernelFnVariant, IntegrationSchemeVariant, PressureSolverVariant,
       NeighborSearchVariant, BoundaryHandlingVariant
            │
            ▼  (macro expansion, cartesian product)
    SystemConstructor<K, I, P, N, B>::new(params, scene, state)
            │
            ▼
    System::<K, I, P, N, B>::new_boxed(constructor)  ->  Box<dyn SPHSystem>
```

`Parameters` and `Scene` (also from `.toml`) supply numeric constants and the scene graph (fluid/boundary mesh instances with transforms), which `SystemConstructor` uses to sample particles (`utilities::sampling`) and build boundary representations. If a saved state file is supplied, `SystemConstructor` restores fluid samples, dynamic boundary pose/velocity and the elapsed simulation time/step count from it instead of sampling fresh geometry from the scene (see Checkpointing and state persistence).

### Time step pipeline

Each call to `SPHSystem::step_forward_in_time` runs:

```
1. integrator.integrate(fluid, dt)         — advance position/velocity (integration_schemes)
   boundary.step_forward_in_time(dt)       — integrate dynamic boundary rigid-body motion
2. time_steps_propagated += 1
3. update():
   a. disable/drop out-of-bounds particles
   b. neighbor_search.find_samples(...)     — rebuild fluid-fluid NeighborList (neighbor_search)
   c. boundary.find_boundary_samples(...)   — rebuild fluid-boundary neighbors (sph::boundary_handling)
   d. quantities::get_volume(...)           — recompute per-particle volume
   e. calc_acceleration():
        reset_acceleration
        add_non_pressure_acceleration()     — gravity + viscosity (non_pressure_accelerations)
        pressure_solver.solve_and_add_acceleration(...)  — pressure_solver (also accumulates
                                                           reaction force/torque onto dynamic
                                                           boundaries via RigidBodyMotion::add_force)
   f. update CurrentSystemProperties (avg. density, [CFL time step if enabled])
```

Every stage is implemented via the `for_each!` macro (per-particle loop, sequential or `rayon`-parallel depending on the `parallel` feature) and dispatches to the concrete `K`, `N`, `B`, `P` chosen at construction time.

### Data flow to the frontend

```
SPHSystem::take_measurement()      -> Measurement        (measurement.rs)
SPHSystem::get_fluid_surface()     -> Vec<(u32, RenderMesh)>   (fluid.rs, via splashsurf_lib)
SPHSystem::get_*_visualization()   -> FluidVisualization / BoundaryVisualization (render_info.rs)
              │
              ▼
       TimeStepInfo::from_system(...)   — bundles measurement + fluid + boundary visualization
              │
              ▼  sent over crossbeam channel to UI (see sci-phi-backend)
```

### Checkpointing and state persistence

Two related but distinct mechanisms snapshot fluid and dynamic boundary state:

- **`SystemCheckpoint`** (`SystemCheckpoint::from_sph_system` / `SPHSystem::continue_from_checkpoint`) — a lightweight, `nalgebra`-typed in-memory snapshot (fluid samples, per-boundary `RigidBodyMotionState`, `time_steps_propagated`, elapsed simulation time) used to rewind an already-running system to an earlier point in time within the same process (e.g. rebuilding the visualization buffer or changing visualization settings in `sci-phi-backend`).
- **`SerSystemCheckpoint`** — the serializable counterpart (`Serialize`/`Deserialize`/`Encode`/`Decode`), obtained via `From<SystemCheckpoint>`, used to persist a full system state to a `.ron` file (`--state` / "save state") and later resume a simulation across separate program runs. `System3DConstructor::new` restores fluid, dynamic boundary state and the step/time counters from it, assuming the scene file defines the same boundaries (order, count, static/dynamic kind) as when the state was saved.

## Features

| Feature | Default | Description |
|---------|---------|--------------|
| `logging` | ✅ | Enables `tracing` log statements (`dep:tracing`) |
| `parallel` | ✅ | Enables `rayon`-based parallel iteration in `for_each!` and elsewhere (`dep:rayon`) |
| `cfl_time_step` | ❌ | Switches to an adaptive CFL-condition-based time step instead of a fixed time increment |

## Dependencies

| Crate | Purpose |
|-------|---------|
| `nalgebra` | Linear algebra (`Point3`, `Vector3`, `Matrix3`, `UnitQuaternion`, `Isometry3`) |
| `glam` | Interop type used by `parry3d-f64` point queries |
| `parry3d-f64` | Triangle mesh representation, AABB, signed-distance/point queries, mass properties |
| `splashsurf_lib` | Surface reconstruction of fluid particles into a triangle mesh |
| `num-traits` | Generic numeric traits (e.g. `Zero`) |
| `gauss-quad` | Gauss-Legendre quadrature for volume-map integration |
| `rustc-hash` | Fast hash map used by the spatial hashing neighbor search |
| `crossbeam` | (Currently unused directly here but part of the workspace's threading model) |
| `serde` | Deserializing scene/parameter `.toml` files, serializing checkpoints |
| `tobj` | Loading `.obj`/`.mtl` triangle meshes |
| `ron` | Serializing/deserializing `SerSystemCheckpoint` state files |
| `csv` | Writing measurement series to disk |
| `toml` | Parsing scene and parameter configuration files |
| `bincode` | Compact binary (de)serialization of `TimeStepInfo` / checkpoints |
| `bytemuck` | Casting `RenderVertex` data into GPU buffer bytes |
| `thiserror` | Error type definitions (`EvaluationError`) |
| `dyn-clone` | Enables cloning `Box<dyn SPHSystem>` trait objects |
| `tracing` *(optional, `logging`)* | Structured logging |
| `rayon` *(optional, `parallel`)* | Data parallelism for per-particle computations |

## Usage

```rust
use simulation_lib::setup::input::{Parameters, Procedures, Scene};
use simulation_lib::setup::new_boxed_system3d;

let procedures = Procedures::from_file("params.toml")?;
let params = Parameters::from_file("params.toml")?;
let scene = Scene::from_file("scene.toml")?;

// state_file_path: Some(path) to resume from a saved SerSystemCheckpoint (`--state`),
// restoring fluid samples, dynamic boundary state and elapsed time/step count.
let mut system = new_boxed_system3d(&procedures, &params, &scene, None)?;

// Advance the simulation and take a measurement
system.step_forward_in_time();
let measurement = system.take_measurement();
println!("time: {}, density error: {}%", measurement.time, measurement.density_error);
```

### Testing

Tests are split into two categories, matching the visibility of what they exercise:


- **Internal unit tests** (`#[cfg(test)] mod tests` inside the module itself) — used

  wherever a module has private fields, private helper functions, or invariants that
  can only be observed by reaching into its internals (e.g. `NeighborList`'s flattening
  logic, `MeshContainer`'s cache invalidation, `IISPH`'s diagonal-element computation,
  `RigidBodyMotion`'s cached derived-state consistency).

- **External integration tests** (`tests/*.rs`) — exercise only the public API, the way

  a downstream crate (e.g. `sci-phi-backend`) would use it. Used for modules whose
  entire surface is public (e.g. `boundary_handling`'s trait definitions,
  `integration_schemes`, `kernel`), and for end-to-end tests that assemble a full
  `System` via `setup::new_boxed_system3d`.

Run the full suite:

```bash
cargo test
```

Since several tests are gated on the `cfl_time_step` feature (fixed vs. adaptive time
stepping select different code paths and, in some cases, different assertions), also run:

```bash
cargo test --features cfl_time_step
cargo test --no-default-features   # exercises the sequential (non-rayon) `for_each!` path
```

A few tests are marked `#[ignore]` because they exercise expensive code paths (the real
Gauss-Legendre volume-map quadrature in `VolumeMapBoundary`, full `splashsurf_lib`
surface reconstruction) whose execution is too slow to run automatically; run them explicitly via:

```bash
cargo test -- --ignored
```

#### Coverage overview

| Area | Status |
|------|--------|
| `sph::kernel` | ✅ Kernel contract (default methods) + `CubicBSpline3D` (normalization, monotonicity, compact support) |
| `sph::boundary_handling` | ✅ Module-level types (traits, checkpoints); `VolumeMapBoundary`, `StaticSampleBoundary` individually |
| `sph::boundary_handling::rigid_body_motion` | ✅ Cached derived-state consistency (`inertia_tensor_inv_world`, `angular_velocity`) across `new`, `update_derived`, `step_forward_in_time`, `restore_from_checkpoint` |
| `sph::pressure_solver` | ✅ Shared helpers (`add_pressure_acceleration`, `set_pred_vel_by_applying_acc`) + all four concrete solvers |
| `integration_schemes` | ✅ All schemes (`ExplicitEuler`, `EulerCromer`, `Verlet`, `TakePredicted`) |
| `neighbor_search` | ✅ `NeighborList`, `SpatialHashing` (hashing, cell lookup, collision-free range) |
| `sph::non_pressure_accelerations`, `sph::quantities` | ✅ Formula-level checks against independently derived expected values |
| `sph` (`System`, `SPHSystem`, checkpoints) | ✅ End-to-end via `setup::new_boxed_system3d`; `SystemParameters`/`CurrentSystemProperties` internals |
| `setup` / `setup::input` | ✅ TOML (de)serialization, cross-reference validation, full system construction (success + failure paths) |
| `utilities::triangle_mesh`, `utilities::sampling`, `utilities::discretization` | ✅ |
| `fluid` (surface reconstruction), `measurement` | ✅ core logic; surface reconstruction covered only by an `#[ignore]`d smoke test |
| `render_info` | ✅ `SimulationParameters`/bincode round trips, `RenderPose` conversion, visualization dispatch logic (`FluidVisualization`/`FluidSampleColoring`/`BoundaryVisualization`) |
