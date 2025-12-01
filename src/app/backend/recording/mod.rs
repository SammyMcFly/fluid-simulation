//! Record states or measurements of the simulation system
//!
//!
use std::io::Write;
use std::path::{Path, PathBuf};
use std::collections::VecDeque;
use serde::Serialize;

#[cfg(feature = "logging")]
use tracing::{error, warn, info}; // debug, error, info, span, trace, warn,

use crate::app::backend::SimulationParameters;

use super::sph::particle::SerParticle3D;



/// Store the current state of all fluid particles to a file
pub fn save_system_state(particles: Vec<SerParticle3D>, file_path: &str) -> std::io::Result<()> {
    let file_path = Path::new(file_path);
    // convert to global path
    let file_path_parent = std::fs::canonicalize(
        file_path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."))
    )?;
    let global_file_path = file_path_parent.join(file_path.file_name().expect("No final component found."));

    if !file_path_parent.exists() { // Create the parent directory if it does not exist
        std::fs::create_dir_all(file_path_parent.clone())?;
        #[cfg(feature = "logging")]
        info!("Created directory: {}", file_path_parent.display());
    } else if global_file_path.exists() { // Throw an error if file already exist
        #[cfg(feature = "logging")]
        error!("File already exists: {}", file_path_parent.display());
        return Err(std::io::Error::from(std::io::ErrorKind::AlreadyExists));
    }

    let ron_string = ron::to_string(&particles).unwrap();
    let mut file = std::fs::File::create(global_file_path)?;
    file.write_all(ron_string.as_bytes())?;
    Ok(())
}


#[derive(Debug, Clone, Default)]
pub enum RecordingStatus {
    #[default]
    None,
    NotStarted,
    Measuring,
    Finished,
}

impl RecordingStatus {
    pub fn advance_to_next_state(&mut self) {
        match self {
            Self::NotStarted => *self = Self::Measuring,
            Self::Measuring => *self = Self::Finished,
            Self::Finished => panic!("Called advance_to_next_state on RecordingStatus::Finished"),
            _ => panic!("Called advance_to_next_state on RecordingStatus::None"),
            // _ => panic!("Called advance_to_next_state on RecordingStatus::None or RecordingStatus::Finished"),
        }
    }
    pub fn is_active(&self) -> bool {
        matches!(self, RecordingStatus::Measuring)
    }
    pub fn is_finished(&self) -> bool {
        matches!(self, RecordingStatus::Finished)
    }
}



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
    file_path: PathBuf,
}

impl MeasurementSeries {
    pub fn new(file_path: &str) -> std::io::Result<Self> {
        let file_path = std::path::Path::new(&file_path);
        // convert to global path
        let file_path_parent = std::fs::canonicalize(
            file_path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."))
        )?;
        let global_file_path = file_path_parent.join(file_path.file_name().expect("No final component found."));

        if !file_path_parent.exists() { // Create the parent directory if it does not exist
            std::fs::create_dir_all(file_path_parent.clone())?;
            #[cfg(feature = "logging")]
            info!("Created directory: {}", file_path_parent.display());
        } else if global_file_path.exists() { // Throw an error if file already exist
            #[cfg(feature = "logging")]
            error!("File already exists: {}", file_path_parent.display());
            return Err(std::io::Error::from(std::io::ErrorKind::AlreadyExists));
        }

        Ok(Self {
            queue: VecDeque::default(),
            file_path: global_file_path,
        })
    }
    pub fn get_path(&self) -> PathBuf {
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
        if self.queue.is_empty() {
            #[cfg(feature = "logging")]
            warn!("Saving empty measurement series!");
        }
        let file = std::fs::File::create(self.file_path.clone())?;
        let mut wtr = csv::Writer::from_writer(file);
        for measurement in &self.queue {
            wtr.serialize(measurement)?;
        }
        wtr.flush()?;
        Ok(())
    }
}



#[derive(Debug)]
pub struct StateAppender {
    /// File path to store measurement series to
    file_path: PathBuf,
}

impl StateAppender {
    pub fn new(file_path: &str, sim_info: &SimulationParameters) -> std::io::Result<Self> {
        let file_path = Path::new(file_path);
        // convert to global path
        let file_path_parent = std::fs::canonicalize(
            file_path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."))
        )?;
        let global_file_path = file_path_parent.join(file_path.file_name().expect("No final component found."));

        if !file_path_parent.exists() { // Create the parent directory if it does not exist
            std::fs::create_dir_all(file_path_parent.clone())?;
            #[cfg(feature = "logging")]
            info!("Created directory: {}", file_path_parent.display());
        } else if global_file_path.exists() { // Throw an error if file already exist
            #[cfg(feature = "logging")]
            error!("File already exists: {}", file_path_parent.display());
            return Err(std::io::Error::from(std::io::ErrorKind::AlreadyExists));
        }

        if global_file_path.exists() {
            #[cfg(feature = "logging")]
            error!("File already exists: {}", file_path_parent.display());
            return Err(std::io::Error::from(std::io::ErrorKind::AlreadyExists));
        }
        let appender = Self {
            file_path: global_file_path,
        };
        appender.append_time_step_info_to_file(sim_info.clone())?;
        Ok(appender)
    }

    pub fn append_time_step_info_to_file(&self, info: impl std::convert::Into<std::vec::Vec<u8>>) -> std::io::Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.file_path.clone())?;

        let bytes: Vec<u8> = info.into();
        let len = bytes.len() as u64;

        // Write length prefix
        file.write_all(&len.to_le_bytes())?;
        // Write serialized struct
        file.write_all(&bytes)?;

        Ok(())
    }
}
