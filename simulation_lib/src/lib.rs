/// # Physics based simulation backend
///
/// Contains all necessary components to initialize a scene, simulate the trajectories
/// of its containing samples by propagating the system time and taking measurements
/// at the simulation.
///
use serde::{Serialize, Deserialize};
use bincode::{Encode, Decode};


// #[cfg(feature = "logging")]
// use tracing::{error, warn, info}; // debug, error, info, span, trace, warn,

use sph::particle::{SerParticle3D, SerBoundaryParticle3D};

pub mod measurement;
pub mod setup;
pub mod sph;


#[cfg(all(feature = "local_pressure", feature = "global_pressure"))]
compile_error!("Only one of the features `local_pressure` and `global_pressure` can be activated at the same time.");
#[cfg(all(not(feature = "local_pressure"), not(feature = "global_pressure")))]
compile_error!("One of the features `local_pressure` and `global_pressure` must be activated.");



#[derive(Debug, Copy, Clone, Default, PartialEq, Serialize, Deserialize, Encode, Decode)]
pub enum ParticleColor {
    #[default]
    VelocityGraded,
    FixedColor([f32;3]),
}


#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct SimulationParameters {
    /// Particle size
    pub particle_diameter: f32,
    /// Rest density
    pub rest_density: f32,
    /// Light position
    pub light_position: [f32; 3],
    /// Particle color
    pub particle_color: ParticleColor,
    /// Boundary particle color
    pub boundary_particle_color: ParticleColor,
    /// Integration Scheme
    pub integration_scheme: sph::PropagationMethod,
    /// maximum buffer length
    pub buffer_length_limit: usize,
    /// Flag that is true if a measurement is taken in simulation, else false
    pub is_measured: bool,
    /// Flag that is true if simulation state are stored in a file (recorded), else false
    pub is_recorded: bool,
}

impl From<&[u8]> for SimulationParameters {
    fn from(bytes: &[u8]) -> Self {
        let cfg = bincode::config::standard();
        let (decoded, _len): (Self, usize) = bincode::decode_from_slice(bytes, cfg).unwrap();
        decoded
    }
}

impl From<SimulationParameters> for Vec<u8> {
    fn from(time_step_info: SimulationParameters) -> Self {
        let cfg = bincode::config::standard();
        bincode::encode_to_vec(time_step_info, cfg).unwrap()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, Encode, Decode)]
pub struct TimeStepInfo {
    // system time
    pub time: f32,
    // time increment
    pub time_increment: f32,
    // average density
    pub average_density: f32,
    // particles
    pub fluid: Vec<SerParticle3D>,
    pub boundary: Vec<SerBoundaryParticle3D>
}

impl From<&[u8]> for TimeStepInfo {
    fn from(bytes: &[u8]) -> Self {
        let cfg = bincode::config::standard();
        let (decoded, _len): (Self, usize) = bincode::decode_from_slice(bytes, cfg).unwrap();
        decoded
    }
}

impl From<TimeStepInfo> for Vec<u8> {
    fn from(time_step_info: TimeStepInfo) -> Self {
        let cfg = bincode::config::standard();
        bincode::encode_to_vec(time_step_info, cfg).unwrap()
    }
}



// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn it_works() {
//         let result = add(2, 2);
//         assert_eq!(result, 4);
//     }
// }
