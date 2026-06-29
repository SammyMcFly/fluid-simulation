/// Input module
///
/// Defines
use serde::Deserialize;
use serde::de::{self, Deserializer, SeqAccess, Visitor};
use std::collections::HashMap;

use crate::{integration_schemes::IntegrationSchemeVariant, neighbor_search::NeighborSearchVariant, sph::{boundary_handling::BoundaryHandlingVariant, kernel::KernelFnVariant, pressure_solver::PressureSolverVariant}};


#[derive(Debug, Deserialize)]
pub struct Procedures {
    pub kernel_function: KernelFnVariant,
    pub integration_scheme: IntegrationSchemeVariant,
    pub pressure_solver: PressureSolverVariant,
    pub neighbor_search: NeighborSearchVariant,
    pub boundary_handling: BoundaryHandlingVariant,
}

impl Procedures {
    pub fn from_file(file_path: &str,) -> Result<Self, Box<dyn std::error::Error + Send + Sync>>{
        // Read the config file
        let table: toml::Table = toml::from_str(&std::fs::read_to_string(file_path)?)?;
        let procedures = table
            .get("procedures")
            .ok_or("missing [procedures] section")?
            .clone();
        let config: Self = procedures.try_into()?;
        Ok(config)
    }
}

#[derive(Debug, Deserialize)]
pub struct Parameters {
    pub buffer_length_limit: usize,
    #[cfg(not(feature = "cfl_time_step"))]
    pub time_increment: f64,
    #[cfg(feature = "cfl_time_step")]
    pub max_time_increment: f64,
    #[cfg(feature = "cfl_time_step")]
    pub cfl_number: f64,
    pub fluid: Vec<Fluid>,
    pub rest_density_grid_spacing: f64,
    pub kernel_support_radius: f64,
    pub disable_particles_below: f64,
    pub fluid_viscosity: f64,
    pub boundary_viscosity: f64,
    pub boundary_pressure_acceleration_weighting: f64,
    pub boundary_rest_volume_weighting: f64,
    pub stiffness: f64,
    // solver_iterations: u32,
    pub target_density_error: f64,
    pub relaxation_factor: f64,
    pub min_diagonal_element: f64,
}

impl Parameters {
    pub fn from_file(file_path: &str,) -> Result<Self, Box<dyn std::error::Error + Send + Sync>>{
        // Read the config file
        let table: toml::Table = toml::from_str(&std::fs::read_to_string(file_path)?)?;
        let procedures = table
            .get("parameters")
            .ok_or("missing [parameters] section")?
            .clone();
        let config: Self = procedures.try_into()?;
        Ok(config)
    }
}

#[derive(Debug, Deserialize)]
pub struct Fluid {
    pub id: u32,
    pub rest_density: f64,
}

#[derive(Debug, Deserialize)]
pub struct Scene {
    pub light: Light,
    pub meshes: HashMap<String, String>,
    #[serde(default)]
    pub fluid: Vec<FluidDef>,
    #[serde(default)]
    pub boundary: BoundaryDefs,
}

impl Scene {
    pub fn from_file(file_path: &str,) -> Result<Self, Box<dyn std::error::Error + Send + Sync>>{
        // Read the scene file
        let config: Self = toml::from_str(&std::fs::read_to_string(file_path)?)?;
        Ok(config)
    }
}

#[derive(Debug, Deserialize)]
pub struct Light {
    pub position: [f64; 3],
}


#[derive(Debug, Deserialize)]
pub struct FluidDef {
    pub mesh: String,
    pub fluid_id: u32,
    #[serde(default)]
    pub position: [f64; 3],
    #[serde(default)]
    pub rotation_euler_deg: [f64; 3],
    #[serde(default = "default_scale", deserialize_with = "deserialize_scale")]
    pub scale: [f64; 3],
}

#[derive(Debug, Default, Deserialize)]
pub struct BoundaryDefs {
    #[serde(default, rename = "static")]
    pub statics: Vec<StaticBoundaryDef>,
    #[serde(default)]
    pub dynamic: Vec<DynamicBoundaryDef>,
}

#[derive(Debug, Deserialize)]
pub struct StaticBoundaryDef {
    pub mesh: String,
    pub boundary_id: u32,
    #[serde(default)]
    pub position: [f64; 3],
    #[serde(default)]
    pub rotation_euler_deg: [f64; 3],
    #[serde(default = "default_scale", deserialize_with = "deserialize_scale")]
    pub scale: [f64; 3],
}

#[derive(Debug, Deserialize)]
pub struct DynamicBoundaryDef {
    pub mesh: String,
    pub boundary_id: u32,
    #[serde(default)]
    pub position: [f64; 3],
    #[serde(default)]
    pub rotation_euler_deg: [f64; 3],
    #[serde(default = "default_scale", deserialize_with = "deserialize_scale")]
    pub scale: [f64; 3],
}

fn default_scale() -> [f64; 3] {
    [1.0, 1.0, 1.0]
}

fn deserialize_scale<'de, D>(deserializer: D) -> Result<[f64; 3], D::Error>
where
    D: Deserializer<'de>,
{
    struct ScaleVisitor;

    impl<'de> Visitor<'de> for ScaleVisitor {
        type Value = [f64; 3];

        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "a number or an array of 3 numbers")
        }

        fn visit_f64<E: de::Error>(self, v: f64) -> Result<[f64; 3], E> {
            Ok([v, v, v])
        }

        fn visit_i64<E: de::Error>(self, v: i64) -> Result<[f64; 3], E> {
            Ok([v as f64, v as f64, v as f64])
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<[f64; 3], A::Error> {
            let x = seq.next_element::<f64>()?.ok_or_else(|| de::Error::invalid_length(0, &self))?;
            let y = seq.next_element::<f64>()?.ok_or_else(|| de::Error::invalid_length(1, &self))?;
            let z = seq.next_element::<f64>()?.ok_or_else(|| de::Error::invalid_length(2, &self))?;
            Ok([x, y, z])
        }
    }

    deserializer.deserialize_any(ScaleVisitor)
}
