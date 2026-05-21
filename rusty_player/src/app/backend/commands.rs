//! Worker commands
use rendering_lib::readback::ReadbackRequest;
use simulation_lib::sph::particle::SerParticle3D;

pub enum WorkerCommand {
    ReadRecording(String),
    SaveScreenshot(ReadbackRequest),
    SaveState {
        particles: Vec<SerParticle3D>,
        filepath: String,
    },
    Stop,
}
