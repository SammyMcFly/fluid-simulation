# simulation_lib

Physics-based SPH (Smoothed Particle Hydrodynamics) fluid simulation backend. Provides all components needed to configure a scene, propagate an SPH system in time, take measurements, and produce renderable output (meshes or point samples) for a UI/renderer.

## Overview

This crate is generic over the numerical building blocks of an SPH simulation — kernel function, integration scheme, pressure solver, neighbor search algorithm, and boundary handling — and assembles a concrete simulation (`System3D`) from a scene/parameter configuration read from `.toml` files. The resulting system is exposed behind the object-safe [`SPHSystem`] trait so that the frontend/backend crates don't need to know the concrete generic instantiation.

## Module structure

| Module | Purpose |
|--------|---------|
| `fluid` | `Fluid3D` — collection of fluid samples (position, velocity, mass, volume, pressure, ...) and surface reconstruction via `splashsurf_lib`; `SerFluid3D` for (de)serialization |
| `sph` | Core simulation: `SPHSystem` trait, generic `System3D<K, I, P, N, B>`, `Checkpoint`, system parameters/properties |
| `sph::kernel` | SPH kernel functions (`KernelFn` trait, `CubicBSpline3D`) |
| `sph::pressure_solver` | Pressure solvers: `SESPH`, `SESPHwSplitting`, `IISPH`, `IISPHwOST` |
| `sph::boundary_handling` | Boundary representations: `StaticSampleBoundary` (explicit samples), `VolumeMaps` (implicit signed-distance/volume-map boundary) |
| `sph::non_pressure_accelerations` | Gravity and viscosity acceleration contributions |
| `sph::quantities` | SPH interpolation of scalar/vector quantities (volume, speed, density, pressure, kinetic energy) at arbitrary positions |
| `integration_schemes` | Time integrators: `ExplicitEuler`, `EulerCromer`, `Verlet`, `TakePredicted` (`ImplicitEuler` currently disabled) |
| `neighbor_search` | `NeighborSearch` trait, `NeighborList`, and `SpatialHashing` implementation |
| `setup` | Scene/parameter loading (`input.rs`) and generic system construction (`System3DConstructor`, `new_boxed_system3d`) |
| `render_info` | `SimulationParameters`, `TimeStepInfo` and visualization types sent to the renderer/UI |
| `measurement` | `Measurement`, `MeasurementSeries`, `RecordingStatus` — recording physical quantities to CSV |
| `utilities` | `discretization` (cubic serendipity SDF/volume-map interpolation, Gauss-Legendre quadrature), `sampling` (volume/surface particle sampling), `triangle_mesh` (mesh loading, render mesh construction) |
| `iteration` | Internal `for_each!` macro abstracting over sequential (`iter_mut`) and parallel (`rayon::par_iter_mut`) execution depending on the `parallel` feature |

## Architecture

### Generic system composition

The core simulation type is generic over five abstract procedures:

```rust
pub struct System3D<K: KernelFn, I: IntegrationScheme, P: PressureSolver, N: NeighborSearch, B: BoundaryHandling> {
    fluid: Fluid3D,
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
| `B` | `BoundaryHandling` | `StaticSampleBoundary`, `VolumeMaps` |

`System3D<K, I, P, N, B>` implements the object-safe, `dyn_clone`-able **`SPHSystem`** trait. Consumers (`sci-phi-backend`) only ever hold a `Box<dyn SPHSystem>` and never see the concrete generic instantiation.

### Assembly from configuration

`setup::new_boxed_system3d` reads a `Procedures` enum selection (one variant per type parameter, deserialized from `.toml`) and nested macros (`with_integrator!` → `with_pressure!` → `with_neighbor!` → `with_boundary!`) expand to the matching concrete `System3D<...>`, wrapped in `Box<dyn SPHSystem>`:

```
Procedures (from .toml)
    └─ KernelFnVariant, IntegrationSchemeVariant, PressureSolverVariant,
       NeighborSearchVariant, BoundaryHandlingVariant
            │
            ▼  (macro expansion, cartesian product)
    System3DConstructor<K, I, P, N, B>::new(params, scene, state)
            │
            ▼
    System3D::<K, I, P, N, B>::new_boxed(constructor)  ->  Box<dyn SPHSystem>
```

`Parameters` and `Scene` (also from `.toml`) supply numeric constants and the scene graph (fluid/boundary mesh instances with transforms), which `System3DConstructor` uses to sample particles (`utilities::sampling`) and build boundary representations.

### Time step pipeline

Each call to `SPHSystem::step_forward_in_time` runs:

```
1. integrator.integrate(fluid, dt)         — advance position/velocity (integration_schemes)
2. time_steps_propagated += 1
3. update():
   a. disable/drop out-of-bounds particles
   b. neighbor_search.find_samples(...)     — rebuild fluid-fluid NeighborList (neighbor_search)
   c. boundary.find_boundary_samples(...)   — rebuild fluid-boundary neighbors (sph::boundary_handling)
   d. quantities::get_volume(...)           — recompute per-particle volume
   e. calc_acceleration():
        reset_acceleration
        add_non_pressure_acceleration()     — gravity + viscosity (non_pressure_accelerations)
        pressure_solver.solve_and_add_acceleration(...)  — pressure_solver
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

`Checkpoint::from_sph_system` / `SPHSystem::continue_from_checkpoint` allow snapshotting and rewinding a `SerFluid3D` state independently of this pipeline (used for rebuilding the visualization buffer in `sci-phi-backend`).

## Features

| Feature | Default | Description |
|---------|---------|--------------|
| `logging` | ✅ | Enables `tracing` log statements (`dep:tracing`) |
| `parallel` | ✅ | Enables `rayon`-based parallel iteration in `for_each!` and elsewhere (`dep:rayon`) |
| `cfl_time_step` | ❌ | Switches to an adaptive CFL-condition-based time step instead of a fixed time increment |

## Dependencies

| Crate | Purpose |
|-------|---------|
| `nalgebra` | Linear algebra (`Point3`, `Vector3`, `Matrix3`) |
| `glam` | Interop type used by `parry3d-f64` point queries |
| `parry3d-f64` | Triangle mesh representation, AABB, signed-distance/point queries |
| `splashsurf_lib` | Surface reconstruction of fluid particles into a triangle mesh |
| `num-traits` | Generic numeric traits (e.g. `Zero`) |
| `gauss-quad` | Gauss-Legendre quadrature for volume-map integration |
| `rustc-hash` | Fast hash map used by the spatial hashing neighbor search |
| `crossbeam` | (Currently unused directly here but part of the workspace's threading model) |
| `serde` | Deserializing scene/parameter `.toml` files, serializing checkpoints |
| `tobj` | Loading `.obj`/`.mtl` triangle meshes |
| `ron` | Serializing/deserializing `SerFluid3D` state files |
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

// state_file_path: Some(path) to continue from a saved SerFluid3D state
let mut system = new_boxed_system3d(&procedures, &params, &scene, None)?;

// Advance the simulation and take a measurement
system.step_forward_in_time();
let measurement = system.take_measurement();
println!("time: {}, density error: {}%", measurement.time, measurement.density_error);
```

<!--## Testing

Unit tests are colocated with modules that benefit most from them (`neighbor_search::mod`, `neighbor_search::spatial_hashing`) and an integration test binary lives under `tests/neighbor_search`.

```bash
cargo test -p simulation_lib
```-->
