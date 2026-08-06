# Sci-Phi

An SPH fluid simulation application with interactive 3D visualization.

## Overview

`sci-phi` is the main simulation binary of the workspace. It loads a scene configuration, runs the SPH fluid solver on a dedicated worker thread, and renders results in real time using a libcosmic-based desktop application with custom wgpu rendering.

The following image shows the app's user interface and a simulation of two fluids with different rest densities contained in a ball.

<div align="center">
  <img src="./img/ui.png" alt="ui" style="max-width: 100%; height: auto;" />
</div>

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│              cosmic::Application (Event Loop)                │
├─────────────────────────────┬────────────────────────────────┤
│      Frontend (UI Thread)   │      Backend (Worker Thread)   │
│                             │                                │
│  - COSMIC desktop UI        │  - SPH Simulation              │
│  - Shader widget (wgpu 3D)  │  - Measurement export          │
│  - Playback control         │  - Recording                   │
│  - Camera interaction       │  - State saving                │
│                             │                                │
│    WorkerCommand ──────────►│                                │
│                             │◄──── WorkerMessage             │
│    (crossbeam channel)      │      (crossbeam channel)       │
└─────────────────────────────┴────────────────────────────────┘
```

- **UI thread** — libcosmic application with a `shader::Program` widget for 3D rendering, COSMIC-styled controls, and input handling.
- **Worker thread** — runs the simulation loop, sends completed time steps back via a crossbeam channel polled by a Subscription.
- **Communication** — `crossbeam` channels in both directions, polled asynchronously through iced Subscriptions.

### Rendering Pipeline

```
FluidViewport (Program)
  └→ FluidFrame (Primitive) ─── created each frame
       └→ FluidRenderer (Pipeline) ─── persistent GPU resources
            ├── Particle impostor pipeline
            ├── Mesh opaque/transparent pipelines
            ├── Light indicator pipeline
            └── Depth texture, bind groups, uniform buffers
```

## Usage

```bash
cargo run --release -p sci-phi -- <PARAMS> --scene <SCENE> [OPTIONS]
```

### Positional Arguments

| Argument | Description |
|----------|-------------|
| `PARAMS` | Path to a `.toml` file with simulation parameters |

### Required Options

| Option | Description |
|--------|-------------|
| `--scene <FILE>` | Path to the scene definition `.toml` file |

### Optional Flags

| Option | Short | Description |
|--------|-------|-------------|
| `--state <FILE>` | | Path to a saved particle state file to resume from |
| `--measurement-file <FILE>` | `-m` | Path to a `.csv` file for measurement output |
| `--recording-file <FILE>` | | Path to a binary file for recording time step data |
| `--rendering-dir <DIR>` | | Directory to store rendered frames as `.png` files |
| `--start-time <T>` | `-s` | Simulation time at which measurement/recording/rendering begins |
| `--finish-time <T>` | `-f` | Simulation time at which measurement/recording/rendering ends (pauses simulation) |
| `--resume` | `-r` | Start playback immediately (unpaused) |
| `--exit` | `-e` | Exit the application automatically when `finish_time` is reached |
| `--log <LEVEL>` | `-l` | Log severity: `TRACE`, `DEBUG`, `INFO`, `WARN`, `ERROR`, `OFF` (default: `INFO`) |

### Examples

```bash

### Run with a scene config

cargo run --release -p sci-phi -- params.toml --scene scene.toml

### Run with recording and auto-exit

cargo run --release -p sci-phi -- params.toml --scene scene.toml \
    --recording-file sim.bin \
    --start-time 0.5 --finish-time 3.0 --exit

### Resume from a saved state with measurement export

cargo run --release -p sci-phi -- params.toml --scene scene.toml \
    --state checkpoint.ron \
    --measurement-file results.csv --resume

### Render frames to a directory

cargo run --release -p sci-phi -- params.toml --scene scene.toml \
    --rendering-dir ./frames --start-time 0.0 --finish-time 5.0 --exit
```

## Configuration Files

`sci-phi` takes two mandatory TOML files:

| File | Passed as | Top-level sections |
|------|-----------|--------------------|
| Parameter file | positional argument | `[procedures]`, `[parameters]` |
| Scene file | `--scene` | `[light]`, `[meshes]`, `[[fluid]]`, `[boundary]` |

Each file is deserialized as a whole (`ParameterFile::from_file`, `Scene::from_file`),
so both top-level sections of the parameter file are mandatory and a missing one aborts
startup with `missing field 'parameters'`.

Unknown keys and unknown sections are **rejected**: the error names the offending key
and lists the valid alternatives for its table, and TOML errors carry the line and
column of the offending entry. Typos therefore fail loudly at startup rather than
silently changing behaviour.

Which time-stepping keys are expected depends on the `cfl_time_step` feature:
`time_increment` without it, `max_time_increment` and `cfl_number` with it. Supplying a
key for the other configuration produces an error naming the required feature flag
instead of a generic unknown-field message.

---

### `[procedures]` — algorithm selection

All five keys are **required**. Values are the exact enum variant names (case-sensitive).

| Key | Options |
|-----|---------|
| `kernel_function` | `CubicBSpline` |
| `integration_scheme` | `EulerCromer`, `ExplicitEuler`, `Verlet`, `TakePredicted` |
| `pressure_solver` | `SESPH`, `SESPHwSplitting`, `IISPH`, `IISPHwOST` |
| `neighbor_search` | `SpatialHashing` |
| `boundary_handling` | `StaticSampleBoundary`, `VolumeMaps` |

`TakePredicted` is meant to be paired with `IISPHwOST`, which already integrates the
predicted state.

---

### `[parameters]` — numerical parameters

**Every key listed here is required** — there are no `serde` defaults, so omitting one is
a parse error.

#### Time stepping

The available keys depend on the `cfl_time_step` feature flag:

| Feature state | Required keys |
|---------------|---------------|
| `cfl_time_step` **off** (default) | `time_increment` |
| `cfl_time_step` **on** | `max_time_increment`, `cfl_number` |

| Key | Type | Description |
|-----|------|-------------|
| `time_increment` | float | Fixed simulation time step |
| `max_time_increment` | float | Upper bound for the adaptive step |
| `cfl_number` | float | CFL safety factor for the adaptive step |

Using the wrong pair for your build produces a missing-field error — a common pitfall when
switching feature flags.

#### Discretization

| Key | Type | Description |
|-----|------|-------------|
| `kernel_support_radius` | float | Support radius of the smoothing kernel; also drives the neighbor-search cell size |
| `rest_density_grid_spacing` | float | Grid spacing used when sampling geometry / computing rest volumes |
| `disable_particles_below` | float | Particles falling below this height are deactivated |
| `buffer_length_limit` | integer | Max number of time steps buffered between worker thread and frontend (backpressure) |

#### Fluids

`fluid` is an array of tables and is **required** (at least one entry). Each `id` is
referenced by `fluid_id` in the scene file — this is how multi-fluid scenes with different
rest densities are set up.

| Key | Type | Description |
|-----|------|-------------|
| `id` | integer | Fluid identifier, referenced from the scene |
| `rest_density` | float | Rest density of this fluid |

#### Viscosity & boundary coupling

| Key | Type | Description |
|-----|------|-------------|
| `fluid_viscosity` | float | Artificial viscosity for fluid–fluid interaction |
| `boundary_viscosity` | float | Artificial viscosity for fluid–boundary interaction |
| `boundary_pressure_acceleration_weighting` | float | Weighting of the boundary contribution to pressure acceleration |
| `boundary_rest_volume_weighting` | float | Weighting applied to computed boundary rest volumes |

#### Pressure solver parameters

| Key | Type | Relevant for |
|-----|------|--------------|
| `stiffness` | float | `SESPH`, `SESPHwSplitting` (state equation) |
| `target_density_error` | float | `IISPH`, `IISPHwOST` (convergence criterion) |
| `relaxation_factor` | float | `IISPH`, `IISPHwOST` (Jacobi relaxation ω) |
| `min_diagonal_element` | float | `IISPH`, `IISPHwOST` (guard against near-zero diagonal) |

All four must be present regardless of the selected solver; unused ones are simply ignored.

#### Complete example

```toml
[procedures]
kernel_function     = "CubicBSpline"
integration_scheme  = "EulerCromer"
pressure_solver     = "IISPH"
neighbor_search     = "SpatialHashing"
boundary_handling   = "StaticSampleBoundary"

[parameters]
buffer_length_limit         = 1000
time_increment              = 0.006      # replace with max_time_increment + cfl_number
                                         # when built with `cfl_time_step`
kernel_support_radius       = 1.6
rest_density_grid_spacing   = 0.8
disable_particles_below     = -50.0

fluid_viscosity             = 0.2
boundary_viscosity          = 0.2

boundary_pressure_acceleration_weighting = 1.0
boundary_rest_volume_weighting           = 1.0

stiffness                   = 15000000.0
target_density_error        = 0.1
relaxation_factor           = 0.5
min_diagonal_element        = 1e-15

[[parameters.fluid]]
id = 0
rest_density = 900.0
```

---

### Scene file

Unlike the parameter file, the scene file is parsed as a whole — its keys sit at the top
level.

#### `[light]` (required)

| Key | Type | Description |
|-----|------|-------------|
| `position` | `[f64; 3]` | World-space position of the point light |

#### `[meshes]` (required)

A name → file path map. All geometry entries below reference these names, so a mesh used
several times is declared once. Keep in mind that the winding order of vertices in the mesh
files determines the orientation of the surface normals in the simulation.

#### Transform keys (shared)

Fluid and boundary entries share the same optional placement keys:

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `translation` | `[f64; 3]` | `[0, 0, 0]` | |
| `rotation_euler_deg` | `[f64; 3]` | `[0, 0, 0]` | Euler angles in **degrees** |
| `scale` | number **or** `[f64; 3]` | `[1, 1, 1]` | A single number scales uniformly: `scale = 2.0` ≡ `scale = [2.0, 2.0, 2.0]` |

#### `[[fluid]]` (optional, defaults to empty)

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `mesh` | string | yes | Key from `[meshes]`; volume is sampled with particles |
| `fluid_id` | integer | yes | Must match an `id` in `[[parameters.fluid]]` |
| `translation`, `rotation_euler_deg`, `scale` | see above | no | |

#### `[boundary]` (optional, defaults to empty)

Two arrays, `static` and `dynamic` (WIP), sharing an identical field set:

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `mesh` | string | yes | Key from `[meshes]` |
| `boundary_id` | integer | yes | Boundary identifier |
| `translation`, `rotation_euler_deg`, `scale` | see above | no | |
| `render_vertex_normals` | string | no (`FaceNormals`) | `FaceNormals` or `AngleWeightedPseudoNormals` |

`render_vertex_normals` affects shading of the boundary mesh only — smooth surfaces benefit
from `AngleWeightedPseudoNormals`, hard-edged geometry from `FaceNormals`.

#### Complete example

```toml
[light]
position = [2.0, 2.0, 100.0]

[meshes]
cube_out = "./cube_face_normals_outwards.obj"
cube_in = "./cube_face_normals_inwards.obj"
sphere_out = "./sphere_face_normals_outwards.obj"
sphere_in = "./sphere_face_normals_inwards.obj"

[[fluid]]
[[fluid]]
mesh = "cube_out"
fluid_id = 0
translation = [11.0, 0.0, 0.0]
rotation_euler_deg = [0.0, 0.0, 0.0]
scale = 10.0

[[fluid]]
mesh = "cube_out"
fluid_id = 1
translation = [-11.0, 0.0, 0.0]
scale = 10.0

[[boundary.static]]
mesh = "sphere_in"
boundary_id = 0
translation = [0.0, 0.0, 0.0]
scale = 30.0
render_vertex_normals = "AngleWeightedPseudoNormals"

# [[boundary.dynamic]]
# mesh        = "obstacle"
# boundary_id = 1
# translation = [0.0, 0.5, 0.0]
# scale       = [0.5, 1.0, 0.5]
```

## UI

The application uses the [libcosmic](https://github.com/pop-os/libcosmic) toolkit.

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `W` / `↑` | Camera forward |
| `S` / `↓` | Camera backward |
| `A` / `←` | Camera left |
| `D` / `→` | Camera right |
| `Space` | Camera up |
| `Shift` | Camera down |
| Right-click + drag | Camera rotation |
| Scroll wheel | Camera zoom |

## Dependencies

| Crate | Purpose |
|-------|---------|
| `clap` | CLI argument parsing |
| `libcosmic` | COSMIC desktop UI framework (includes iced + wgpu reexports) |
| `crossbeam` | Channel-based communication between threads |
| `tracing` / `tracing-subscriber` | Structured logging |
| `rendering_lib` | wgpu-based 3D rendering via shader widget (workspace crate) |
| `simulation_lib` | SPH simulation backend (workspace crate) |
| `sci-phi-backend` | App backend logic (workspace crate) |
| `nalgebra` / `cgmath` | Linear algebra |
| `num-traits` | Generic number handling |
| `rustc-hash` | Faster hash map for uniform grid |
| `rfd` | Native file dialogs |
| `serde` | Parsing/serializing file content |
| `ron` | Storing structs in RON format |
| `csv` | Storing measurement data |
| `toml` | Loading `.toml` config files |
| `bincode` | Writing to binary files |
| `tobj` | Loading sphere mesh for visualization |
| `bytemuck` | Casting data for GPU buffers (wgpu) |
| `pollster` | Blocking on async wgpu operations |
| `i18n-embed` / `i18n-embed-fl` | Localization (libcosmic dependency) |
| `rust-embed` | Embedding assets (libcosmic dependency) |
| `open` | Opening links/files (libcosmic dependency) |
| `tokio` | Async runtime (libcosmic dependency) |
