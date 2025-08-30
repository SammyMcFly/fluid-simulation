use std::collections::VecDeque;

use serde::Serialize;


#[derive(Debug, Clone, Default, Serialize)]
pub struct Measurement {
    pub time: f64,
    // Average density relative to rest density
    pub density: f64,
    // Average kinetic energy
    pub kinetic_energy: f64,
    pub stiffness: f64,
    pub viscosity: f64,
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


#[derive(Debug, Clone, Default)]
pub struct MeasurementSeries {
    length: u32,
    queue: VecDeque<Measurement>,
}

impl MeasurementSeries {
    pub fn push_back(&mut self, value: Measurement) {
        self.length += 1;
        self.queue.push_back(value);
    }
    pub fn pop_front(&mut self) -> Option<Measurement> {
        self.length -= 1;
        self.queue.pop_front()
    }
    pub fn clear(&mut self) {
        self.length = 0;
        self.queue.clear();
    }
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
    pub fn len(&self) -> u32 {
        self.length
    }
    pub fn save(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file_path = std::path::Path::new(path);

        // Get the parent directory
        if let Some(parent) = file_path.parent() {
            // Create the parent directory if it doesn't exist
            if !parent.exists() {
                std::fs::create_dir_all(parent)?;
                println!("Created directories: {}", parent.display());
            }
        }
        let file = std::fs::File::create(file_path)?;
        let mut wtr = csv::Writer::from_writer(file);

        for measurement in &self.queue {
            wtr.serialize(measurement)?;
        }

        wtr.flush()?;
        Ok(())
    }
}