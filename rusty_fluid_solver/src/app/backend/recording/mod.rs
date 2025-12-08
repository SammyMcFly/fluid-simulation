//! Record states or measurements of the simulation system
//!
//!
use std::io::Write;
use std::path::{Path, PathBuf};

use tracing::{error, info}; // debug, error, info, span, trace, warn,

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
        info!("Created directory: {}", file_path_parent.display());
    } else if global_file_path.exists() { // Throw an error if file already exist
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
            info!("Created directory: {}", file_path_parent.display());
        } else if global_file_path.exists() { // Throw an error if file already exist
            error!("File already exists: {}", file_path_parent.display());
            return Err(std::io::Error::from(std::io::ErrorKind::AlreadyExists));
        }

        if global_file_path.exists() {
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
