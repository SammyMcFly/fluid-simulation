//! Module for scene building and parameter importing
//!
//!
mod scenes;

use serde::Deserialize;

use super::SimulationInfo;

use super::sph::particle::{Particle3D, SerParticle3D, BoundaryParticle3D};
#[cfg(feature = "springs")]
use super::sph::spring::Spring;
use super::sph::{SystemProperties, PropagationMethod, cubic_b_spline_3d, cubic_b_spline_3d_gradient};
// use super::measure;

use crate::app::rendering::ui::controls::ParticleColor;


#[derive(Debug, Deserialize)]
pub struct Setup {
    pub parameters: Parameters,
    pub light: Light,
    pub scene: SceneVariant,
}

#[derive(Debug, Deserialize)]
pub struct Parameters {
    pub buffer_length_limit: usize,
    pub time_inc: f64,
    pub integration_scheme: PropagationMethod,
    pub rest_density: f64,
    pub rest_density_grid_spacing: f64,
    pub smoothing_length: f64,
    pub disable_particles_below: f64,
    pub viscosity: f64,
    #[cfg(feature = "pseudo_mass_boundary")]
    pub boundary_pressure_acceleration_weighting: f64,
    #[cfg(feature = "local_pressure")]
    pub stiffness: f64,
    #[cfg(feature = "global_pressure")]
    pub relaxation_factor: f64,
    #[cfg(feature = "global_pressure")]
    pub min_diagonal_element: f64,
}

#[derive(Debug, Deserialize)]
pub struct Light {
    pub position: [f32; 3],
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", content = "parameters")]
pub enum SceneVariant {
    NoLidCube(scenes::NoLidCube),
    Spiral(scenes::Spiral),
}

impl Scene for SceneVariant {
    fn get_boundary(&self, rest_density_grid_spacing: f64) -> Vec<BoundaryParticle3D> {
        match self {
            Self::NoLidCube(variant) => variant.get_boundary(rest_density_grid_spacing),
            Self::Spiral(variant) => variant.get_boundary(rest_density_grid_spacing),
        }
    }
    fn get_fluid(&self, rest_density: f64, rest_density_grid_spacing: f64) -> Vec<Particle3D> {
        match self {
            Self::NoLidCube(variant) => variant.get_fluid(rest_density, rest_density_grid_spacing),
            Self::Spiral(variant) => variant.get_fluid(rest_density, rest_density_grid_spacing),
        }
    }
    #[cfg(feature = "springs")]
    fn get_springs(&self) -> Vec<Spring> {
        match self {
            Self::NoLidCube(variant) => variant.get_springs(),
            Self::Spiral(variant) => variant.get_springs(),
        }
    }
    fn calc_fluid_depth(&self, rest_density_grid_spacing: f64) -> f64 {
        match self {
            Self::NoLidCube(variant) => variant.calc_fluid_depth(rest_density_grid_spacing),
            Self::Spiral(variant) => variant.calc_fluid_depth(rest_density_grid_spacing),
        }
    }
}

trait Scene {
    fn get_boundary(&self, rest_density_grid_spacing: f64) -> Vec<BoundaryParticle3D>;
    fn get_fluid(&self, rest_density: f64, rest_density_grid_spacing: f64) -> Vec<Particle3D>;
    #[cfg(feature = "springs")]
    fn get_springs(&self) -> Vec<Spring>;
    fn calc_fluid_depth(&self, rest_density_grid_spacing: f64) -> f64;
}


pub struct System3DConfig {
    pub particles: Vec<Particle3D>,
    pub boundary_particles: Vec<BoundaryParticle3D>,
    #[cfg(feature = "springs")]
    pub springs: Vec<Spring>,
    pub system_properties: SystemProperties,
    // pub controls: Arc<Mutex<mediation::IntermediateControls>>,
    // pub measurement_series: Option<Arc<Mutex<measure::MeasurementSeries>>>,
}

pub struct System3DConfigConstructor {
    config: Setup,
    build: Option<System3DConfig>,
}

impl System3DConfigConstructor {
    fn load_config(file_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Read the scene config file
        let config_file_content = std::fs::read_to_string(file_path)?;
        // Parse the content into the Config struct
        Ok(Self {
            config: toml::from_str(&config_file_content)?,
            build: None,
        })
    }

    fn load_particles(file_path: &str) -> Result<Vec<Particle3D>, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(file_path)?;
        let particles: Vec<SerParticle3D> = ron::from_str(&content).unwrap();
        Ok(particles.into_iter().map(|p| p.into()).collect())
    }

    fn get_system_properties(&self) -> SystemProperties {
        SystemProperties::new(
            self.config.parameters.time_inc,
            self.config.parameters.rest_density,
            self.config.parameters.rest_density_grid_spacing,
            self.config.parameters.smoothing_length,
            self.config.parameters.disable_particles_below,
            self.config.scene.calc_fluid_depth(self.config.parameters.rest_density_grid_spacing),
            self.config.parameters.viscosity,
            #[cfg(feature = "pseudo_mass_boundary")]
            self.config.parameters.boundary_pressure_acceleration_weighting,
            #[cfg(feature = "local_pressure")]
            self.config.parameters.stiffness,
            #[cfg(feature = "global_pressure")]
            self.config.parameters.relaxation_factor,
            #[cfg(feature = "global_pressure")]
            self.config.parameters.min_diagonal_element,
            cubic_b_spline_3d,
            cubic_b_spline_3d_gradient
        )
    }

    fn build(
        &mut self,
        particles: Vec<Particle3D>,
        boundary_particles: Vec<BoundaryParticle3D>,
        #[cfg(feature = "springs")]
        springs: Vec<Spring>,
        system_properties: SystemProperties,
        // controls: Arc<Mutex<mediation::IntermediateControls>>,
        // measurement_series: Option<Arc<Mutex<measure::MeasurementSeries>>>,
    ) {
        self.build = Some(System3DConfig {
            particles,
            boundary_particles,
            #[cfg(feature = "springs")]
            springs,
            system_properties,
            // controls,
            // measurement_series,
        });
    }

    pub fn new(
        config_file_path: &str,
        particle_state_file_path: Option<&str>,
        // controls: Arc<Mutex<mediation::IntermediateControls>>,
        // measurement_series: Option<Arc<Mutex<measure::MeasurementSeries>>>,
    ) -> Result<(Self, SimulationInfo), Box<dyn std::error::Error>> {
        // load config file
        let mut constructor = Self::load_config(config_file_path)?;

        // load particles
        let particles = if let Some(particle_state_file_path) = particle_state_file_path {
            Self::load_particles(particle_state_file_path)?
        } else {
            constructor.config.scene.get_fluid(constructor.config.parameters.rest_density, constructor.config.parameters.rest_density_grid_spacing)
        };

        // load boundary
        let boundary_particles = constructor.config.scene.get_boundary(constructor.config.parameters.rest_density_grid_spacing);
        // load springs
        #[cfg(feature = "springs")]
        let springs = constructor.config.scene.get_springs();

        let sim_info = SimulationInfo {
            particle_diameter: constructor.config.parameters.rest_density_grid_spacing as f32,
            rest_density: constructor.config.parameters.rest_density as f32,
            light_position: constructor.config.light.position,
            particle_color: ParticleColor::default(),
            boundary_particle_color: ParticleColor::default(),
            integration_scheme: constructor.config.parameters.integration_scheme.clone(),
            buffer_length_limit: constructor.config.parameters.buffer_length_limit,
        };

        // init system properties
        let system_properties = constructor.get_system_properties();

        constructor.build(
            particles,
            boundary_particles,
            #[cfg(feature = "springs")]
            springs,
            system_properties,
            // controls,
            // measurement_series,
        );
        // create simulation system
        Ok((constructor, sim_info))
    }

    pub fn finish(self) -> System3DConfig {
        self.build.unwrap()
    }
}

