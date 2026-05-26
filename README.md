# Rusty Fluid Solver

A physics-based **Smoothed Particle Hydrodynamics (SPH)** fluid simulation framework written in Rust, featuring real-time 3D visualization, recording, and playback.

## Overview

This workspace implements a 3D SPH fluid solver with an interactive wgpu-based renderer. The simulation runs on a dedicated worker thread while the frontend handles rendering and user input. Simulations can be recorded and replayed with the included player application.

## Workspace Crates

| Crate | Type | Description |
|-------|------|-------------|
| `meta` | lib | Synchronous feature management |
| `simulation_lib` | lib | SPH kernels, particle dynamics, pressure solvers, neighbor search, scene setup |
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

## Features

### Physics


- **SPH Fluid Simulation** with cubic B-spline kernel (3D)
- **Pressure Solvers**

  - Local state-equation solver (feature: `local_pressure`)
  - Global implicit incompressible SPH (feature: `global_pressure`)

  - Optimized source-term approach (feature: `optimized_source_term`)
- **Integration Schemes**

  - Explicit Euler
  - Euler–Cromer

  - Verlet
  - Implicit Euler with conjugate gradient (feature: `implicit_euler`)(WIP)

- **Viscosity** — artificial viscosity for fluid–fluid and fluid–boundary interactions
- **Boundary Handling** — static boundary particles with pseudo-volume computation (feature: `pseudo_volume_boundary`)
<!-- - **Spring Forces** — elastic connections between particles (feature: `springs`) -->
- **Adaptive Time Stepping** via CFL condition (feature: `cfl_time_step`)

### Rendering


- Real-time 3D particle visualization via **wgpu**
- Instanced sphere rendering with Phong lighting

- Interactive camera (orbit, pan, zoom)
- Cross-section cut planes for interior inspection

- On-screen simulation info and playback controls
- Frame export to `.png` for offline video creation

### Performance


- **Spatial Hashing Uniform Grid** for O(1) amortized neighbor search
- **Parallelization** with Rayon (feature: `parallelized_sph`)
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
| `local_pressure` | Local state-equation pressure solver |
| `global_pressure` | Global implicit pressure solver (IISPH-style) |
| `optimized_source_term` | Two-stage pressure solve (velocity-divergence + volume-preservation) |
| `splitting` | Predicted-density splitting for local pressure |
| `springs` | Enable spring forces between particles |
| `implicit_euler` | Implicit Euler integration with conjugate gradient |
| `cfl_time_step` | Adaptive time stepping based on CFL condition |
| `parallelized_sph` | Parallelize particle loops with Rayon |
| `pseudo_volume_boundary` | Compute boundary particle volumes from kernel summation |
| `logging` | Enable `tracing`-based structured logging |

> **Note:** `local_pressure` and `global_pressure` are mutually exclusive — exactly one must be enabled.

## Getting Started

### Prerequisites


- [Rust](https://www.rust-lang.org/tools/install) (edition 2021+)
- A GPU with Vulkan, Metal, or DX12 support (for wgpu rendering)

### Build

```bash

### Build the entire workspace

cargo build --release

### Build with specific features

cargo build --release -p rusty_fluid_solver --features "global_pressure,parallelized_sph,cfl_time_step"
```

### Run the Simulation

```bash
cargo run --release -p rusty_fluid_solver -- scene_config.toml
```

### Run the Player

```bash
cargo run --release -p rusty_player -- recording.bin
```

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