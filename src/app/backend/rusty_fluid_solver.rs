use nalgebra::Vector3;
use serde::{Serialize, Deserialize};
use bincode::{Encode, Decode};

use crate::app::rendering::ui::controls::ParticleColor;


/// Method for propagating time in a simulated physical system
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub enum PropagationMethod {
    ExplicitEuler,
    ImplicitEuler,
    EulerCromer,
    Verlet,
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
    pub integration_scheme: PropagationMethod,
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


pub trait Positional {
    fn pos_now(&self) -> Vector3<f64>;
}

/// Compressed and serializable particle in a 3-dimensional context
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct SerParticle3D {
    position: [[f64; 3]; 3],
    velocity: [[f64; 3]; 3],
    mass: f64,
    disabled: bool,
}

impl Positional for SerParticle3D {
    fn pos_now(&self) -> Vector3<f64> {
        Vector3::new(self.position[0][0], self.position[0][1], self.position[0][2])
    }
}

impl SerParticle3D {
    pub fn vel_now(&self) -> [f64; 3] {
        self.velocity[0]
    }

    pub fn is_enabled(&self) -> bool {
        !self.disabled
    }
}

/// Compressed and serializable particle in a 3-dimensional context
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct SerBoundaryParticle3D {
    position: [f64; 3],
    #[cfg(feature = "global_pressure")]
    velocity: [f64; 3],
}

impl Positional for SerBoundaryParticle3D {
    fn pos_now(&self) -> Vector3<f64> {
        Vector3::new(self.position[0], self.position[1], self.position[2])
    }
}

impl SerBoundaryParticle3D {
    pub fn vel_now(&self) -> [f64; 3] {
        self.velocity
    }
}