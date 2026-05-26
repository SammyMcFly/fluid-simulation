//! Worker commands
use rendering_lib::readback::ReadbackRequest;
use simulation_lib::sph::sample::SerFluid3D;

pub enum WorkerCommand {
    ReadRecording(String),
    SaveScreenshot(ReadbackRequest),
    SaveState { fluid: SerFluid3D, filepath: String },
    Stop,
}
