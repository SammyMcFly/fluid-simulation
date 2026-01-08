//! Worker commands
use crate::app::backend::sph::particle::SerParticle3D;
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
    SaveState { particles: Vec<SerParticle3D>, filepath: String, },
    // Resume,
    // Pause,
    Reset,
    Stop,
}
