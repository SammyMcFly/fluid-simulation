//! Backend module
use image::{ImageBuffer, Rgba};
use simulation_lib::render_info::{SimulationParameters, TimeStepInfo};
use std::io::Read;
use std::path::Path;
use std::time::Duration;

use crossbeam::channel::{Receiver, Sender};

pub mod commands;
pub mod messages;

use commands::WorkerCommand;
use messages::WorkerMessage;

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

    /// A path has no final component to use as a file name.
    #[error("no file name found in path '{0}'")]
    NoFileName(std::path::PathBuf),

    /// The recording data ended before a complete length-prefixed record
    /// could be read — the file is truncated (e.g. an interrupted write) or
    /// corrupted (a garbage length prefix caused an out-of-range read).
    #[error("recording file is truncated or corrupted")]
    Truncated,

    /// A length-prefixed record's payload could not be decoded as a
    /// [`SimulationParameters`](simulation_lib::render_info::SimulationParameters)
    /// or [`TimeStepInfo`](simulation_lib::render_info::TimeStepInfo).
    #[error("failed to decode recording data: {0}")]
    Decode(#[from] bincode::error::DecodeError),
}

/// Reads one length-prefixed record (an 8-byte little-endian length followed
/// by that many payload bytes) from `buf` starting at `*pos`, advancing
/// `*pos` past it.
///
/// Returns [`FileIoError::Truncated`] if `buf` doesn't contain enough bytes
/// for the length prefix or the payload it announces.
fn read_length_prefixed<'a>(buf: &'a [u8], pos: &mut usize) -> Result<&'a [u8], FileIoError> {
    let len_bytes: [u8; 8] = buf
        .get(*pos..*pos + 8)
        .and_then(|slice| slice.try_into().ok())
        .ok_or(FileIoError::Truncated)?;
    *pos += 8;

    let len = u64::from_le_bytes(len_bytes) as usize;
    let data = buf.get(*pos..*pos + len).ok_or(FileIoError::Truncated)?;
    *pos += len;

    Ok(data)
}

fn read_recording(
    file_path: &str,
) -> Result<(SimulationParameters, Vec<TimeStepInfo>), FileIoError> {
    let file_path = Path::new(file_path);
    // convert to global path
    let file_path_parent = std::fs::canonicalize(
        file_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new(".")),
    )?;
    // Create the parent directory if it does not exist
    if !file_path_parent.exists() {
        std::fs::create_dir_all(file_path_parent.clone())?;
        tracing::info!("Created directories: {}", file_path_parent.display());
    }
    let file_name = file_path
        .file_name()
        .ok_or_else(|| FileIoError::NoFileName(file_path.to_path_buf()))?;
    let global_file_path = file_path_parent.join(file_name);

    let mut f = std::fs::File::open(global_file_path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;

    let mut pos: usize = 0;

    let general_info = SimulationParameters::try_from(read_length_prefixed(&buf, &mut pos)?)?;

    let mut time_steps = Vec::new();
    while pos < buf.len() {
        let data = read_length_prefixed(&buf, &mut pos)?;
        time_steps.push(TimeStepInfo::try_from(data)?);
    }

    Ok((general_info, time_steps))
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
        return Err(FileIoError::FileAlreadyExists(file_path.clone()));
    }

    let img: ImageBuffer<Rgba<u8>, _> = ImageBuffer::from_raw(width, height, rgba_data.to_vec())
        .ok_or(FileIoError::ImageBufferCreationFailed)?;

    img.save(&file_path)?;

    Ok(())
}

/// Function that does:
/// - receives [[WorkerCommand]] from front-end
/// - passes [[WorkerCommand]] to [[SimulationController]]
/// - sends [[WorkerMessage]] back to front-end
pub fn worker_loop(from_ui: Receiver<WorkerCommand>, to_ui: Sender<WorkerMessage>) {
    'worker: loop {
        match from_ui.try_recv() {
            Ok(msg) => match msg {
                WorkerCommand::ReadRecording(file_path) => {
                    tracing::info!("Start reading recording...");
                    match read_recording(&file_path) {
                        Ok((sim_info, time_steps)) => {
                            tracing::info!("Successfully finished reading recording!");
                            let _ =
                                to_ui.send(WorkerMessage::FinishedReading(sim_info, time_steps));
                        }
                        Err(e) => {
                            tracing::info!("Failed reading recording!");
                            let _ = to_ui.send(WorkerMessage::Error(e.to_string()));
                        }
                    }
                }
                WorkerCommand::WriteRendering {
                    data,
                    width,
                    height,
                    frame_index,
                    directory,
                    overwrite,
                } => {
                    if let Err(e) = save_screenshot_into_directory(
                        &data,
                        width,
                        height,
                        frame_index,
                        &directory,
                        overwrite,
                    ) {
                        tracing::error!("Screenshot failed: {e}");
                        let _ = to_ui.send(WorkerMessage::Error(format!("Screenshot failed: {e}")));
                    } else {
                        tracing::info!("Saved screenshot frame {frame_index}");
                    }
                }
                WorkerCommand::SaveScreenshotToFile {
                    data,
                    width,
                    height,
                    file_path,
                } => {
                    if let Err(e) = save_screenshot_to_file(&data, width, height, &file_path, false)
                    {
                        tracing::error!("Screenshot failed: {e}");
                        let _ = to_ui.send(WorkerMessage::Error(format!("Screenshot failed: {e}")));
                    } else {
                        tracing::info!("Saved manual screenshot frame");
                    }
                }
                WorkerCommand::Stop => {
                    tracing::info!("Stopped backend!");
                    break 'worker;
                }
            },
            Err(crossbeam::channel::TryRecvError::Empty) => {
                std::thread::sleep(Duration::from_millis(16));
            }
            Err(crossbeam::channel::TryRecvError::Disconnected) => {
                tracing::error!("Sender was dropped!");
                break 'worker;
            }
        }
    }
}
