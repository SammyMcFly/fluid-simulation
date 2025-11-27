use std::{collections::VecDeque, path::Path};
use serde::Serialize;

#[cfg(feature = "logging")]
use tracing::{info}; // debug, error, info, span, trace, warn,



#[derive(Debug, Clone, Default, Serialize)]
pub struct Measurement {
    pub time: f64,
    // Average density relative to rest density
    pub density: f64,
    // Average kinetic energy
    pub kinetic_energy: f64,
    #[cfg(feature = "local_pressure")]
    pub stiffness: f64,
    pub fluid_viscosity: f64,
    pub boundary_viscosity: f64,
    /// Fluid depth measured in number of particles
    pub fluid_depth: f64,
    /// Grid spacing when particles are ordered in a cubic grid at rest density
    pub rest_density_grid_spacing: f64,
    /// Smoothing lenght h
    pub smoothing_length: f64,
    /// Rest density of the fluid
    pub rest_density: f64,
    pub time_step_size: f64,
}


#[derive(Debug, Clone)]
pub struct MeasurementSeries {
    /// Container for intermediate storage of measurements
    queue: VecDeque<Measurement>,
    /// File path to store measurement series to
    file_path: String,
}

impl MeasurementSeries {
    pub fn new(file: &str) -> Self {
        Self { queue: VecDeque::default(), file_path: file.to_string() }
    }
    pub fn get_path(&self) -> String {
        self.file_path.clone()
    }
    pub fn push_back(&mut self, value: Measurement) {
        self.queue.push_back(value);
    }
    // pub fn pop_front(&mut self) -> Option<Measurement> {
    //     self.queue.pop_front()
    // }
    // pub fn clear(&mut self) {
    //     self.queue.clear();
    // }
    // pub fn is_empty(&self) -> bool {
    //     self.queue.is_empty()
    // }
    // pub fn len(&self) -> usize {
    //     self.queue.len()
    // }
    pub fn save(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let file_path = std::path::Path::new(&self.file_path);
        // convert to global path
        let file_path_parent = std::fs::canonicalize(
            file_path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."))
        )?;
        // Get the parent directory
        if !file_path_parent.exists() {
            std::fs::create_dir_all(file_path_parent.clone())?;
            #[cfg(feature = "logging")]
            info!("Created directories: {}", file_path_parent.display());
        }
        let file = std::fs::File::create(file_path_parent.join(file_path.file_name().expect("No final component found.")))?;
        let mut wtr = csv::Writer::from_writer(file);
        for measurement in &self.queue {
            wtr.serialize(measurement)?;
        }
        wtr.flush()?;
        Ok(())
    }
}


#[derive(Debug, Clone, Default)]
pub enum MeasurementStatus {
    #[default]
    None,
    NotStarted,
    Measuring,
    Finished,
}

impl MeasurementStatus {
    pub fn advance_to_next_state(&mut self) {
        match self {
            Self::NotStarted => *self = Self::Measuring,
            Self::Measuring => *self = Self::Finished,
            _ => panic!("Called advance_to_next_state on MeasurementStatus::None or MeasurementStatus::Finished"),
        }
    }
    pub fn is_active(&self) -> bool {
        matches!(self, MeasurementStatus::Measuring)
    }
    pub fn is_finished(&self) -> bool {
        matches!(self, MeasurementStatus::Finished)
    }
}