//! Record states or measurements of the simulation system
//!
//!
use image::{ImageBuffer, Rgba};
use simulation_lib::render_info::FluidVisualization;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use tracing::{error, info}; // debug, error, info, span, trace, warn,

// use rendering_lib::readback::{ReadbackBuffer, ReadbackRequest};

use crate::SimulationParameters;

/// Store the current state of all fluid particles to a file
pub fn save_system_state(fluid: FluidVisualization, file_path: &PathBuf) -> std::io::Result<()> {
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
        info!("Created directory: {}", file_path_parent.display());
    } else if global_file_path.exists() {
        // Throw an error if file already exist
        error!("File already exists: {}", global_file_path.display());
        return Err(std::io::Error::from(std::io::ErrorKind::AlreadyExists));
    }

    let ron_string = ron::to_string(&fluid).unwrap();
    let mut file = std::fs::File::create(global_file_path)?;
    file.write_all(ron_string.as_bytes())?;
    Ok(())
}

/// Convert raw buffer data to RGBA. The `padded_bytes` contain rows with `padded_bpr` bytes per row,
/// with actual tight row length = width * 4.
pub fn buffer_to_rgba(
    raw_data: &[u8],
    width: u32,
    height: u32,
    padded_bytes_per_row: usize,
) -> anyhow::Result<Vec<u8>> {
    // raw_data must be width * height * 4 bytes (RGBA8)
    let expected_len = padded_bytes_per_row * (height as usize);
    if raw_data.len() < expected_len {
        anyhow::bail!("Raw image buffer too small");
    }

    // Flip vertically because wgpu textures are Y-down but PNG expects Y-up.
    let mut rgba = vec![0u8; (width * height * 4) as usize];
    let row_bytes = (width * 4) as usize;

    for y in 0..height as usize {
        let src_index = y * padded_bytes_per_row;
        let dst_index = y * row_bytes;
        for x in 0..width as usize {
            let i = src_index + x * 4;
            let o = dst_index + x * 4;

            rgba[o + 0] = raw_data[i + 2]; // R = original B
            rgba[o + 1] = raw_data[i + 1]; // G stays G
            rgba[o + 2] = raw_data[i + 0]; // B = original R
            rgba[o + 3] = raw_data[i + 3]; // A unchanged
        }
    }
    Ok(rgba)
}

pub fn save_screenshot_into_directory(
    rgba_data: &[u8],
    width: u32,
    height: u32,
    frame_index: usize,
    output_dir: &std::path::PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let filename = format!("frame_{:06}.png", frame_index);
    let file_path = output_dir.join(filename);
    save_screenshot_to_file(rgba_data, width, height, &file_path)
}

pub fn save_screenshot_to_file(
    rgba_data: &[u8],
    width: u32,
    height: u32,
    file_path: &std::path::PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = file_path.parent().ok_or("Failed to get parent directory")?;
    if !output_dir.exists() {
        std::fs::create_dir_all(output_dir)?;
        info!("Created directory: {}", output_dir.display());
    } else if file_path.exists() {
        // Throw an error if file already exist
        error!("File already exists: {}", file_path.display());
        return Err(std::io::Error::from(std::io::ErrorKind::AlreadyExists).into());
    }

    let img: ImageBuffer<Rgba<u8>, _> = ImageBuffer::from_raw(width, height, rgba_data.to_vec())
        .ok_or("Failed to create image buffer")?;

    img.save(&file_path)?;

    Ok(())
}

/// Struct that allows to save a [[TimeStepInfo]] into a binary file
#[derive(Debug)]
pub struct StateAppender {
    /// File path to store measurement series to
    file_path: PathBuf,
}

impl StateAppender {
    pub fn new(file_path: &Path, sim_info: &SimulationParameters) -> std::io::Result<Self> {
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
            info!("Created directory: {}", file_path_parent.display());
        } else if global_file_path.exists() {
            // Throw an error if file already exist
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

    pub fn append_time_step_info_to_file(
        &self,
        info: impl std::convert::Into<std::vec::Vec<u8>>,
    ) -> std::io::Result<()> {
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
