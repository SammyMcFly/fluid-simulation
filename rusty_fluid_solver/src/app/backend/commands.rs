//! Worker commands
use crate::app::backend::sample::SerFluid3D;
use rendering_lib::readback::ReadbackRequest;

pub enum WorkerCommand {
    Simulate {
        config: String,
        state: Option<String>,
        measure: Option<String>,
        start_time: Option<f64>,
        finish_time: Option<f64>,
        recording_file: Option<String>,
    },
    AddTimeStepsToCompute(usize),
    SaveScreenshot(ReadbackRequest),
    SaveState {
        fluid: SerFluid3D,
        filepath: String,
    },
    // Resume,
    // Pause,
    Reset,
    Stop,
}
