# sci-phi-player-backend

Backend worker logic for the `sci-phi-player` app. Loads pre-recorded SPH simulation recordings from disk and replays them, forwarding time step data to the UI and handling screenshot export.

## Overview

Unlike `sci-phi-backend`, this crate does **not** run a live simulation. Instead it reads a binary recording file (written by `sci-phi-backend`'s `TSInfoAppender`) containing a `SimulationParameters` header followed by a sequence of length-prefixed `TimeStepInfo` entries, and sends the parsed data back to the UI for playback. It also handles saving rendered frames as PNG screenshots.

## Module structure

| Module | Purpose |
|--------|---------|
| `lib.rs` | Recording file parsing (`read_recording`), screenshot saving helpers, and the `worker_loop` function |
| `commands.rs` | `WorkerCommand` enum — messages sent from the UI to the worker |
| `messages.rs` | `WorkerMessage` enum — messages sent from the worker back to the UI |

## Core functions

- **`read_recording`** — Parses a binary recording file: an 8-byte little-endian length prefix followed by a serialized `SimulationParameters` struct, then a sequence of length-prefixed `TimeStepInfo` entries until EOF.
- **`worker_loop`** — Background thread loop: waits for `WorkerCommand`s via a non-blocking `try_recv`, processes them, and sleeps briefly when idle. Terminates on `Stop` or when the UI channel disconnects.

### Notable functionality

- **Screenshot saving** — `save_screenshot_into_directory` / `save_screenshot_to_file` write RGBA pixel buffers to PNG files, creating parent directories as needed and erroring if the target file already exists. `save_to_png` is a variant for padded row data.

## Communication

| Direction | Type | Description |
|-----------|------|--------------|
| UI → Worker | `WorkerCommand` | `ReadRecording`, `WriteRendering`, `SaveScreenshotToFile`, `Stop` |
| Worker → UI | `WorkerMessage` | `FinishedReading`, `SavedScreenshot`, `SavedState`, `Error` |

Channels are provided by `crossbeam::channel` and passed into `worker_loop(from_ui, to_ui)`.

## Dependencies

| Crate | Purpose |
|-------|---------|
| `simulation_lib` | Provides `SimulationParameters` / `TimeStepInfo` types (workspace crate) |
| `rendering_lib` | Provides `ScreenshotCommand` trait implemented by `WorkerCommand` (workspace crate) |
| `crossbeam` | Channel-based communication between the worker thread and the UI |
| `image` | Decoding recording frames and saving screenshots as PNG |
| `ron` | Error type used when deserializing recording data |
| `thiserror` | Error type definitions (`FileIoError`) |
| `tracing` / `tracing-subscriber` | Structured logging |

## Usage

```rust
use crossbeam::channel::unbounded;
use sci_phi_player_backend::worker_loop;

let (to_worker, from_ui) = unbounded();
let (to_ui, from_worker) = unbounded();

std::thread::spawn(move || {
    worker_loop(from_ui, to_ui);
});

// Send WorkerCommand::ReadRecording("recording.bin".into()) to load a recording,
// then receive WorkerMessage::FinishedReading(params, time_steps) for playback.
```
