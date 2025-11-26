use std::collections::VecDeque;
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
        #[cfg(feature = "logging")]
        info!("created file path: {:?}", file_path);
        // convert to global path
        let file_path_parent = std::fs::canonicalize(file_path.parent().expect("No parent file path found."))?;
        #[cfg(feature = "logging")]
        info!("canonicalized");
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