# Rusty Fluid Solver

A physics-based **Smoothed Particle Hydrodynamics (SPH)** fluid simulation framework written in Rust, featuring real-time 3D visualization, recording, and playback.

## Overview

This workspace implements a 3D SPH fluid solver with an interactive wgpu-based renderer. The simulation runs on a dedicated worker thread while the frontend handles rendering and user input. Simulations can be recorded and replayed with the included player application.

## Workspace Crates

| Crate | Type | Description |
|-------|------|-------------|
| `meta` | lib | Synchronous feature management |
| `simulation_lib` | lib | SPH kernels, integration schemes, pressure solvers, neighbor search, scene setup |
| `rendering_lib` | lib | wgpu-based 3D renderer with camera, lighting, UI overlay, and screenshot export |
| `rusty_fluid_solver` | bin | Main application — runs SPH simulation with real-time visualization |
| `rusty_player` | bin | Playback application for recorded simulation data |

## Related Tools

| Tool | Description |
|------|-------------|
| [`rusty_measurement_runner`](https://github.com/SammyMcFly/fluid-simulation-automation) | Automates parameter sweeps by executing `rusty_fluid_solver` across predefined combinations of simulation parameters |
| [`rusty_plotter`](https://github.com/SammyMcFly/fluid-simulation-plotting) | Visualizes `.csv` measurement outputs as 2D/3D plots for analysis and comparison |

### Workflow

```
rusty_fluid_solver ──► .csv measurement files ──► rusty_plotter ──► plots (.png / .svg)
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
- **Boundary Handling** — static boundary particles with volume computation to allow irregular sampling
- **Adaptive Time Stepping** via CFL condition (feature: `cfl_time_step`)

### Rendering

- Real-time 3D particle visualization via **wgpu**
- Instanced sphere rendering with Phong lighting

- Interactive camera (orbit, pan, zoom)
- Cross-section cut planes for interior inspection

- On-screen simulation info and playback controls
- Frame export to `.png` for offline video creation

### Performance

- **Neighbor Search** algorithms (trait: `NeighborSearch`) for O(n·k) neighbor search
  (k = average neighbors per particle; optimal when `cell_size ≈ search_range`)
  - Spatial Hashing with Uniform Grid
 <!--  - Octree -->
- **Parallelization** with Rayon (feature: `parallel`)
- **Structure-of-Arrays (SoA)** particle layout for cache-efficient iteration
- `rustc_hash::FxHashMap` for fast grid lookups

- Dedicated worker thread keeps UI responsive

### Recording & Playback

- Full simulation state serialization via `serde` + `bincode`
- Record time step data to binary files

- Replay recordings with `rusty_player`: play forward/backward, make incremental time steps
- Measurement export to `.csv`

## Feature Flags

| Feature | Description |
|---------|-------------|
| `parallel` | Parallelize particle loops with Rayon |
| `cfl_time_step` | Adaptive time stepping based on CFL condition |
| `logging` | Enable `tracing`-based structured logging |

**Note:** Pressure solver and integration scheme are selected at compile time via type parameters on `System3D<K, I, P>`, not feature flags.

## Getting Started

### Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (edition 2021+)
- A GPU with Vulkan, Metal, or DX12 support (for wgpu rendering)

### Build

```bash

### Build the entire workspace

cargo build --release

### Build with specific features

cargo build --release -p rusty_fluid_solver --features "parallel,cfl_time_step"
```

### Run the Simulation

```bash
cargo run --release -p rusty_fluid_solver -- scene_config.toml
```

### Run the Player

```bash
cargo run --release -p rusty_player -- recording.bin
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

## CLI Reference

### rusty_fluid_solver

```bash
cargo run --release -p rusty_fluid_solver -- [OPTIONS] [CONFIG]
```

| Option | Short | Description |
|--------|-------|-------------|
| `config` | | Path to a `.toml` scene configuration file |
| `--state <FILE>` | | Resume from a saved particle state file |
| `--measurement-file <FILE>` | `-m` | Export measurements to `.csv` |
| `--recording-file <FILE>` | | Record time steps to binary file |
| `--rendering-dir <DIR>` | | Export rendered frames as `.png` |
| `--start-time <T>` | `-s` | Begin measurement/recording/rendering at time T |
| `--finish-time <T>` | `-f` | End measurement/recording/rendering at time T |
| `--resume` | `-r` | Start simulation immediately (unpaused) |
| `--exit` | `-e` | Exit automatically when finish time is reached |
| `--log <LEVEL>` | `-l` | Log level: `TRACE`, `DEBUG`, `INFO`, `WARN`, `ERROR`, `OFF` |

### rusty_player

```bash
cargo run --release -p rusty_player -- [OPTIONS] [RECORDING]
```

| Option | Short | Description |
|--------|-------|-------------|
| `recording` | | Path to a simulation recording binary file |
| `--resume` | `-r` | Start playback immediately |
| `--rendering-dir <DIR>` | | Export rendered frames as `.png` |
| `--start-time <T>` | `-s` | Begin playback at time T |
| `--finish-time <T>` | `-f` | Pause playback at time T |
| `--log <LEVEL>` | `-l` | Log level (default: `INFO`) |

## Examples

```bash

### Basic simulation with default scene

cargo run --release -p rusty_fluid_solver -- rusty_fluid_solver/scene_config.toml --resume

### Record a simulation segment

cargo run --release -p rusty_fluid_solver -- rusty_fluid_solver/scene_config.toml \
    --recording-file output/sim.bin \
    --start-time 0.5 --finish-time 3.0 --resume --exit

### Export measurements

cargo run --release -p rusty_fluid_solver -- rusty_fluid_solver/scene_config.toml \
    --measurement-file output/data.csv --resume

### Replay a recording and export frames

cargo run --release -p rusty_player -- output/sim.bin \
    --rendering-dir output/frames/ --resume
```

## Dependencies

| Crate | Purpose |
|-------|---------|
| `nalgebra` | Linear algebra (Vector3, Matrix3) |
| `serde` / `bincode` / `toml` | Serialization and configuration |
| `rayon` | Data parallelism |
| `rustc_hash` | Fast hashing for spatial grid |
| `crossbeam` | Thread communication channels |
| `clap` | CLI argument parsing |
| `wgpu` | GPU rendering backend |
| `winit` / `iced_winit` | Windowing, event loop, and UI |
| `tracing` / `tracing-subscriber` | Structured logging |

<!-- ## License

*Add your license information here.* -->