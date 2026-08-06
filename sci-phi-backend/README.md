# sci-phi-backend

Backend worker logic for the `sci-phi` SPH fluid simulation app. Runs the simulation loop in a background thread, handles checkpointing, recording (measurements & state), and screenshot export, and communicates with the UI via `crossbeam` channels.

## Overview

This crate implements the **worker thread** that drives an SPH simulation loaded from `simulation_lib`. It receives [`WorkerCommand`]s from the UI, advances the simulation, manages checkpoints for rewinding/continuing, optionally records measurements and time step data, and reports progress back via [`WorkerMessage`]s.

## Module structure

| Module | Purpose |
|--------|---------|
| `lib.rs` | Core simulation control: `Simulation`, `SimulationController`, and the `worker_loop` function |
| `commands.rs` | `WorkerCommand` enum — messages sent from the UI to the worker |
| `messages.rs` | `WorkerMessage` enum — messages sent from the worker back to the UI |
| `recording.rs` | Saving simulation state (RON), measurement recordings (binary), and screenshots (PNG) to disk |

## Core structs/functions

- **`Simulation`** — Owns the loaded SPH system, checkpoints (taken every `N` time steps), and optional measurement/recording state. Advances time steps and tracks recording start/finish times.
- **`SimulationController`** — Wraps an optional `Simulation`, tracks computation state (`Computing`/`Paused`) and how many time steps are queued for computation.
- **`worker_loop`** — Main loop run on a background thread: processes incoming `WorkerCommand`s, advances the simulation when active, and sends `WorkerMessage`s (time step results, errors, recording status) back to the UI.

### Notable functionality

- **Checkpoints** — Snapshots of the SPH system taken every `Simulation::N` time steps, enabling rebuilding the visualization buffer (e.g. when changing the visualization).
- **Recording** — Optionally records a `MeasurementSeries` and/or appends `TimeStepInfo` to a binary file (`TSInfoAppender`) between a configurable start and finish time.

## Communication

| Direction | Type | Description |
|-----------|------|--------------|
| UI → Worker | `WorkerCommand` | `Simulate`, `AddTimeStepsToCompute`, `SaveState`, `WriteRendering`, `SaveScreenshotToFile`, `Reload`, `ContinueFromTimeStep`, `Stop` |
| Worker → UI | `WorkerMessage` | `TimeStepReady`, `SimulationLoaded`, `FinishedReloading`, `ContinuedFromCheckpoint`, `ReachedStartTime`, `ReachedFinishTime`, `SavedState`, `SavedMeasurement`, `Error` |

Channels are provided by `crossbeam::channel` and passed into `worker_loop(from_ui, to_ui)`.

## Dependencies

| Crate | Purpose |
|-------|---------|
| `simulation_lib` | SPH simulation backend (workspace crate) |
| `rendering_lib` | Provides `ScreenshotCommand` trait implemented by `WorkerCommand` (workspace crate) |
| `crossbeam` | Channel-based communication between the worker thread and the UI |
| `image` | Saving rendered frames as PNG screenshots |
| `ron` | Serializing simulation state / parameters to the RON format |
| `thiserror` | Error type definitions (`FileIoError`) |
| `tracing` / `tracing-subscriber` | Structured logging |

## Usage

```rust
use crossbeam::channel::unbounded;
use sci_phi_backend::worker_loop;

let (to_worker, from_ui) = unbounded();
let (to_ui, from_worker) = unbounded();

std::thread::spawn(move || {
    worker_loop(from_ui, to_ui);
});

// Send commands via `to_worker`, receive updates via `from_worker`.
