//! Record states or measurements of the simulation system
//!
//!
use serde::Serialize;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};

#[cfg(feature = "logging")]
use tracing::{error, info, warn}; // debug, error, info, span, trace, warn,

#[derive(Debug, Clone, Copy, Default)]
pub enum RecordingStatus {
    #[default]
    None,
    NotStarted,
    InProgress,
    Finished,
}

impl RecordingStatus {
    pub fn advance_to_next_state(&mut self) {
        match self {
            Self::NotStarted => *self = Self::InProgress,
            Self::InProgress => *self = Self::Finished,
            Self::Finished => panic!("Called advance_to_next_state on RecordingStatus::Finished"),
            _ => panic!("Called advance_to_next_state on RecordingStatus::None"),
            // _ => panic!("Called advance_to_next_state on RecordingStatus::None or RecordingStatus::Finished"),
        }
    }
    pub fn is_active(&self) -> bool {
        matches!(self, RecordingStatus::InProgress)
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
    pub stiffness: f64,
    pub fluid_viscosity: f64,
    pub boundary_viscosity: f64,
    /// Fluid depth measured in number of particles
    pub fluid_depth: f64,
    /// Grid spacing when particles are ordered in a cubic grid at rest density
    pub rest_density_grid_spacing: f64,
    /// Smoothing lenght h
    pub smoothing_length: f64,
    /// Kernel support radius
    pub kernel_support_radius: f64,
    /// Rest density of the fluid
    pub rest_density: f64,
    pub time_step_size: f64,
    pub target_density_error: f64,
    pub solver_iterations: u32,
    pub relaxation_factor: f64,
    pub time_step_wall_clock_time: f64,
    pub predicted_density_error: f64,
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
            file_path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new(".")),
        )?;
        let mut global_file_path =
            file_path_parent.join(file_path.file_name().expect("No final component found."));

        if !file_path_parent.exists() {
            // Create the parent directory if it does not exist
            std::fs::create_dir_all(file_path_parent.clone())?;
            #[cfg(feature = "logging")]
            info!("Created directory: {}", file_path_parent.display());
        } else if global_file_path.exists() {
            let immutable_global_file_path = global_file_path.clone();
            let mut counter: u16 = 2;
            while global_file_path.exists() {
                // modify file name with added number to make it unique
                let path = Path::new(&immutable_global_file_path);
                let stem = path.file_stem().unwrap().to_string_lossy();
                let ext = path.extension().unwrap_or_default().to_string_lossy();
                if counter == u16::MAX {
                    #[cfg(feature = "logging")]
                    error!(
                        "File '{}' and all files with the following pattern already exists: {}",
                        format!("{}.{}", stem, ext),
                        format!("{}_#123.{}", stem, ext),
                    );
                    return Err(std::io::Error::from(std::io::ErrorKind::AlreadyExists));
                }
                let new_filename = format!("{}_#{}.{}", stem, counter, ext);
                global_file_path = global_file_path.with_file_name(new_filename);
                counter += 1;
            }
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
