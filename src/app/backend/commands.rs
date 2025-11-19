//! Worker commands
use crate::app::backend::sph::particle::SerParticle3D;



pub enum WorkerCommand {
    Simulate { config: String, state: Option<String>, measure: Option<String>, finish_time: Option<f32>},
    AddTimeStepsToCompute(usize),
    Save { particles: Vec<SerParticle3D>, filepath: String, },
    // Resume,
    // Pause,
    Reset,
    Stop,
}
