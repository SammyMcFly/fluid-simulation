//! Worker commands
use simulation_lib::sph::particle::SerParticle3D;
use rendering_lib::readback::ReadbackRequest;


pub enum WorkerCommand {
    ReadRecording(String),
    SaveScreenshot(ReadbackRequest),
    SaveState { particles: Vec<SerParticle3D>, filepath: String, },
    Stop,
}
