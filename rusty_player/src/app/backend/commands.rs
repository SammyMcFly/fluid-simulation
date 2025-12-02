//! Worker commands
use crate::app::backend::rusty_fluid_solver::SerParticle3D;


pub enum WorkerCommand {
    ReadRecording(String),
    SaveImage(),
    SaveState { particles: Vec<SerParticle3D>, file_path: String, },
    Stop,
}
