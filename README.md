# Sci-PHi

A physics-based **Smoothed Particle Hydrodynamics (SPH)** fluid simulation framework written in Rust, featuring real-time 3D visualization, recording, and playback.

## Overview

This workspace implements a 3D SPH fluid solver with an interactive wgpu-based renderer. The simulation runs on a dedicated worker thread while the frontend handles rendering and user input. Simulations can be recorded and replayed with the included player application.

The following image shows the user interface of `sci-phi`.

<img width="2560" height="1408" alt="ui" src="./sci-phi/img/ui.png" />

## Workspace Crates

```
├── sci-phi/                # Main simulation binary (this crate)
├── sci-phi-backend/        # backend library with worker function and communication API 
├── sci-phi-player/         # Playback-only binary
├── sci-phi-player-backend/ # backend library for `sci-phi-player`
├── rendering_lib/          # 3D rendering library (FluidRenderer, FluidViewport, etc.)
└── simulation_lib/         # SPH simulation library
```
## Related Tools

| Tool | Description |
|------|-------------|
| [`rusty_measurement_runner`](https://github.com/SammyMcFly/fluid-simulation-automation) | Automates parameter sweeps by executing `sci-phi` across predefined combinations of simulation parameters |
| [`rusty_plotter`](https://github.com/SammyMcFly/fluid-simulation-plotting) | Visualizes `.csv` measurement outputs as 2D/3D plots for analysis and comparison |

### Workflow

```
     sci-phi ──► .csv measurement files ──► rusty_plotter ──► plots (.png / .svg)
        ▲
        │
rusty_measurement_runner (automates parameter sweeps)
```

The `.csv` measurement files are produced by the `--measurement-file` flag.

## Architecture

The simulation core is generic over three traits:

```text
System3D<K: KernelFn, I: IntegrationScheme, P: PressureSolver>
```

| Trait | Responsibility | Examples |
|-------|---------------|----------|
| `KernelFn` | SPH smoothing kernel (W, ∇W) | `CubicBSpline` |
| `IntegrationScheme` | Time integration of positions/velocities | `EulerCromer`, `Verlet`, `TakePredicted` |
| `PressureSolver` | Compute pressure field and apply acceleration | `SESPH`, `IISPH`, `IISPHwOST` |

## Features

### Physics


- **SPH Fluid Simulation** with pluggable kernel functions (trait: `KernelFn`)
  - Cubic B-spline kernel (default)
- **Pressure Solvers** (trait: `PressureSolver`)
  - Local state-equation solver (`SESPH`)
  - Local state-equation solver with splitting (`SESPHwSplitting`)
  - Implicit Incompressible SPH (`IISPH`)
  - IISPH with optimized source term (`IISPHwOST`)
- **Integration Schemes** (trait: `IntegrationScheme`)
  - Euler–Cromer (`EulerCromer`)
  - Explicit Euler (`ExplicitEuler`)
  - Verlet (`Verlet`)
  - Accept predicted state (`TakePredicted`) — used by `IISPHwOST`
  <!-- - Implicit Euler with conjugate gradient (WIP) -->
- **Viscosity** — artificial viscosity for fluid–fluid and fluid–boundary interactions
- **Boundary Handling**  (trait: `BoundaryHandling`)
  - static sample boundary with (rest) volume computation to allow irregular sampling
  - volume maps (WIP)
- **Adaptive Time Stepping** via CFL condition (feature: `cfl_time_step`)

### Rendering

- Real-time 3D particle visualization via **wgpu**
- libcosmic UI
- Instanced billboard-sphere rendering with Phong lighting
- Interactive camera (orbit, pan, zoom) with smooth per-frame
  updates (~60 Hz `CameraTick`) and one-click camera reset
- Configurable fluid visualization options:
  - Samples with color mapping by scalar fields (e.g. velocity/pressure/density)
  - Reconstructed surface
  - Sensor plane for scalar field visualization
- Configurable boundary visualization options:
  - Triangle mesh
  - Samples in case of static sample boundary
- Cross-section axis-aligned cut planes on all three axes for interior
  inspection
- Simulation info (time step, particle count, …)
  and playback controls (play/pause, step forward/back)
- Screenshots to `.png` at any time from the top bar
- Offline rendering mode: writes a `.png` per frame to a
  user-supplied `rendering_dir`, with optional
  `exit_when_finished` for batch/CI runs (assemble into
  video with e.g. `ffmpeg`)
<!--- Persistent view/visualization settings via the COSMIC
  config system (per `APP_ID`)-->
- Multi-threaded architecture: rendering on the UI thread,
  simulation on a worker thread communicating over
  `crossbeam` channels

### Performance

- **Neighbor Search** algorithms (trait: `NeighborSearch`) for O(n·k) neighbor search
  (k = average neighbors per particle; optimal when `cell_size ≈ search_range`)
  - Spatial Hashing with Uniform Grid with `rustc_hash::FxHashMap` for fast grid lookups
 <!--  - Octree -->
- **Parallelization** with Rayon (feature: `parallel`)
- **Structure-of-Arrays (SoA)** particle layout for cache-efficient iteration

- Dedicated worker thread keeps UI responsive

### Recording & Playback

- Full simulation state serialization via `serde` + `bincode`
- Record time step data to binary files

- Replay recordings with `sci-phi-player`: play forward/backward, make incremental time steps
- Measurement export to `.csv`

## Feature Flags

| Feature | Description |
|---------|-------------|
| `parallel` | Parallelize particle loops with Rayon |
| `cfl_time_step` | Adaptive time stepping based on CFL condition |
| `logging` | Enable `tracing`-based structured logging |

**Note:** Pressure solver and integration scheme are selected at run time via parameter file.

## Getting Started

### Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (edition 2021+)
- A GPU with Vulkan, Metal, or DX12 support (for wgpu rendering)

### Build

```bash

### Build the entire workspace

cargo build --release

### Build with specific features

cargo build --release -p sci-phi --features "parallel,cfl_time_step"
```

### Run the Simulation

```bash
cargo run --release -p sci-phi -- params.toml --scene scene.toml
```

### Run the Player

```bash
cargo run --release -p sci-phi-player -- recording.bin
```

### Testing

```bash

### Run all tests

cargo test

### Run tests for a specific crate

cargo test -p simulation_lib

### Run tests with output visible

cargo test -- --nocapture

### Run tests matching a pattern

cargo test neighbor_search

```

<!--### sci-phi-player

```bash
cargo run --release -p sci-phi-player -- [OPTIONS] [RECORDING]
```

| Option | Short | Description |
|--------|-------|-------------|
| `recording` | | Path to a simulation recording binary file |
| `--resume` | `-r` | Start playback immediately |
| `--rendering-dir <DIR>` | | Export rendered frames as `.png` |
| `--start-time <T>` | `-s` | Begin playback at time T |
| `--finish-time <T>` | `-f` | Pause playback at time T |
| `--log <LEVEL>` | `-l` | Log level (default: `INFO`) |-->


## Dependencies

| Crate | Purpose |
|-------|---------|
| `nalgebra` | Linear algebra (Vector3, Matrix3) |
| `cgmath` | Linear algebra for graphics |
| `libcosmic` | UI framework with wgpu reexport for rendering |
| `clap` | CLI argument parsing |
| `crossbeam` | Thread communication channels |
| `serde` / `bincode` / `toml` / `tobj` / `ron` / `csv` / `image` | Serialization and file I/O |
| `rayon` | Data parallelism |
| `rustc_hash` | Fast hashing for spatial grid |
| `parry3d-f64` | Triangle mesh handling |
| `splashsurf_lib` | Surface reconstruction |
| `gauss-quad` | Gaussian quadrature for numerical integration |
| `tracing` / `tracing-subscriber` | Structured logging |
| `rfd` | Native file dialogs |
| `i18n-embed` / `i18n-embed-fl` | Localization |
 

<!-- ## License

*Add your license information here.* -->
