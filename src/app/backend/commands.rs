//! Worker commands
use crate::app::backend::sph::particle::SerParticle3D;



pub enum WorkerCommand {
    Simulate {
        config: String,
        state: Option<String>,
        measure: Option<String>,
        start_time: Option<f64>,
        finish_time: Option<f64>,
    },
    AddTimeStepsToCompute(usize),
    SaveState { particles: Vec<SerParticle3D>, filepath: String, },
    // Resume,
    // Pause,
    Reset,
    Stop,
}
