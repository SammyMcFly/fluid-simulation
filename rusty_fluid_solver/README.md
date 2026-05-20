# Rusty Fluid Solver

An SPH fluid simulation application with interactive 3D visualization.

## Overview

`rusty_fluid_solver` is the main simulation binary of the workspace. It loads a scene configuration, runs the SPH fluid solver on a dedicated worker thread, and renders results in real time using a wgpu/winit-based window.

## Architecture

```
┌──────────────────────────────────────────────────┐
│                  Event Loop (winit)              │
├────────────────────────┬─────────────────────────┤
│     Frontend (UI)      │     Backend (Worker)    │
│                        │                         │
│  - Window / Rendering  │  - SPH Simulation       │
│  - Input handling      │  - Measurement export   │
│  - Playback control    │  - Recording            │
│                        │                         │
│   WorkerCommand ──────►│                         │
│                        │◄──── WorkerMessage      │
└────────────────────────┴─────────────────────────┘
```

- **Frontend thread** — handles the winit event loop, wgpu rendering, and user interaction.
- **Backend thread** — runs the simulation loop, sends completed time steps back via an event-loop proxy.
- **Communication** — `crossbeam` channels (UI → Worker) and winit user events (Worker → UI).

## Usage

```bash
cargo run --release -p rusty_fluid_solver -- [OPTIONS] [CONFIG]
```

### Positional Arguments

| Argument | Description |
|----------|-------------|
| `config` | Path to a `.toml` scene configuration file |

### Options

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
# Run with a scene config
cargo run --release -p rusty_fluid_solver -- scene_config.toml

# Run with recording and auto-exit
cargo run --release -p rusty_fluid_solver -- scene_config.toml \
    --recording-file sim.bin \
    --start-time 0.5 --finish-time 3.0 --exit

# Resume from a saved state with measurement export
cargo run --release -p rusty_fluid_solver -- scene_config.toml \
    --state checkpoint.bin \
    --measurement-file results.csv --resume
```

## Dependencies

| Crate | Purpose |
|-------|---------|
| `clap` | CLI argument parsing |
| `iced_winit` / `winit` | Windowing and event loop |
| `crossbeam` | Channel-based communication between threads |
| `tracing` / `tracing-subscriber` | Structured logging |
| `rendering_lib` | wgpu-based 3D rendering (workspace crate) |
| `simulation_lib` | SPH simulation backend (workspace crate) |
