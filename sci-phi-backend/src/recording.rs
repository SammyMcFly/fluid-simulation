//! Record states or measurements of the simulation system
//!
//!
use image::{ImageBuffer, Rgba};
use simulation_lib::render_info::TimeStepInfo;
use simulation_lib::sph::SerSystemCheckpoint;
use std::collections::HashMap;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

// use rendering_lib::readback::{ReadbackBuffer, ReadbackRequest};

use crate::SimulationParameters;

#[derive(Debug, thiserror::Error)]
pub enum FileIoError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),

    #[error("RON serialization error: {0}")]
    Ron(#[from] ron::Error),

    #[error("File already exists: {0}")]
    FileAlreadyExists(std::path::PathBuf),

    #[error("No parent directory found")]
    NoParentDirectory,

    #[error("Failed to create image buffer")]
    ImageBufferCreationFailed,
}

/// Store the current state of all fluid particles to a file
pub fn save_system_state(state: SerSystemCheckpoint, file_path: &Path) -> Result<(), FileIoError> {
    // convert to global path
    let file_path_parent = std::fs::canonicalize(
        file_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new(".")),
    )?;
    let global_file_path =
        file_path_parent.join(file_path.file_name().expect("No final component found."));

    if !file_path_parent.exists() {
        // Create the parent directory if it does not exist
        std::fs::create_dir_all(file_path_parent.clone())?;
        tracing::info!("Created directory: {}", file_path_parent.display());
    } else if global_file_path.exists() {
        // Throw an error if file already exist
        tracing::error!("File already exists: {}", global_file_path.display());
        return Err(FileIoError::FileAlreadyExists(global_file_path));
    }

    let ron_string = ron::to_string(&state)?;
    let mut file = std::fs::File::create(global_file_path)?;
    file.write_all(ron_string.as_bytes())?;
    Ok(())
}

pub fn save_screenshot_into_directory(
    rgba_data: &[u8],
    width: u32,
    height: u32,
    frame_index: usize,
    output_dir: &Path,
    overwrite: bool,
) -> Result<(), FileIoError> {
    let filename = format!("frame_{:06}.png", frame_index);
    let file_path = output_dir.join(filename);
    save_screenshot_to_file(rgba_data, width, height, &file_path, overwrite)
}

pub fn save_screenshot_to_file(
    rgba_data: &[u8],
    width: u32,
    height: u32,
    file_path: &std::path::PathBuf,
    overwrite: bool,
) -> Result<(), FileIoError> {
    let output_dir = file_path.parent().ok_or(FileIoError::NoParentDirectory)?;
    if !output_dir.exists() {
        std::fs::create_dir_all(output_dir)?;
        tracing::info!("Created directory: {}", output_dir.display());
    } else if file_path.exists() && !overwrite {
        // Throw an error if file already exist
        tracing::error!("File already exists: {}", file_path.display());
        return Err(std::io::Error::from(std::io::ErrorKind::AlreadyExists).into());
    }

    let img: ImageBuffer<Rgba<u8>, _> = ImageBuffer::from_raw(width, height, rgba_data.to_vec())
        .ok_or(FileIoError::ImageBufferCreationFailed)?;

    img.save(&file_path)?;

    Ok(())
}

/// Struct that allows to save a [[TimeStepInfo]] into a binary file
#[derive(Debug)]
pub struct TSInfoAppender {
    /// File path to store measurement series to
    file_path: PathBuf,
    /// Byte offset at which the record for a given `time_step_number` begins
    /// (right after its length prefix). Lets a re-produced time step — e.g.
    /// after a visualization-triggered rewind — overwrite its stale record
    /// (and everything written after it) instead of being duplicated;
    /// subsequent steps are simply re-appended as the simulation naturally
    /// re-produces them.
    offsets: HashMap<u64, u64>,
}

impl TSInfoAppender {
    pub fn new(file_path: &Path, sim_info: &SimulationParameters) -> Result<Self, FileIoError> {
        // convert to global path
        let file_path_parent = std::fs::canonicalize(
            file_path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new(".")),
        )?;
        let global_file_path =
            file_path_parent.join(file_path.file_name().expect("No final component found."));

        if !file_path_parent.exists() {
            // Create the parent directory if it does not exist
            std::fs::create_dir_all(file_path_parent.clone())?;
            tracing::info!("Created directory: {}", file_path_parent.display());
        } else if global_file_path.exists() {
            // Throw an error if file already exist
            tracing::error!("File already exists: {}", file_path_parent.display());
            return Err(FileIoError::FileAlreadyExists(global_file_path));
        }

        let appender = Self {
            file_path: global_file_path,
            offsets: HashMap::new(),
        };
        appender.append_time_step_info_to_file(sim_info.clone())?;
        Ok(appender)
    }

    /// Appends arbitrary length-prefixed binary data to the recording file.
    ///
    /// Used only for the one-time [`SimulationParameters`] header, which has
    /// no `time_step_number` and is never rewritten. Time step records
    /// should go through [`Self::append_time_step`] instead.
    pub fn append_time_step_info_to_file(
        &self,
        info: impl std::convert::Into<std::vec::Vec<u8>>,
    ) -> Result<(), FileIoError> {
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

    /// Appends a single [`TimeStepInfo`] record, tracking the file offset it
    /// starts at.
    ///
    /// If a record for this `time_step_number` was already written (the
    /// simulation rewound and re-produced it, e.g. after a visualization
    /// change), the file is truncated back to that offset first, so this
    /// call overwrites the stale record instead of duplicating it. Offsets
    /// for any later, now-truncated-away time steps are dropped; they get
    /// re-established as the simulation naturally re-produces them.
    pub fn append_time_step(&mut self, info: TimeStepInfo) -> Result<(), FileIoError> {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            // Explicitly keep existing content: this method seeks to a
            // specific offset (either the end, to append, or a previously
            // recorded time step's offset, to overwrite it in place) rather
            // than relying on truncate-on-open or append-mode semantics.
            .truncate(false)
            .open(self.file_path.clone())?;

        let time_step_number = info.time_step_number;

        let write_offset = if let Some(&offset) = self.offsets.get(&time_step_number) {
            file.set_len(offset)?;
            self.offsets.retain(|&ts, _| ts < time_step_number);
            offset
        } else {
            file.metadata()?.len()
        };

        file.seek(SeekFrom::Start(write_offset))?;

        let bytes: Vec<u8> = info.into();
        let len = bytes.len() as u64;
        file.write_all(&len.to_le_bytes())?;
        file.write_all(&bytes)?;

        self.offsets.insert(time_step_number, write_offset);
        Ok(())
    }
}
