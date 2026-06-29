//! Worker commands
use rendering_lib::readback::ReadbackRequest;
use simulation_lib::render_info::FluidVisualization;

pub enum WorkerCommand {
    ReadRecording(String),
    SaveScreenshot(ReadbackRequest),
    SaveState { fluid: FluidVisualization, filepath: String },
    Stop,
}
