//! Backend module
use image::{ImageBuffer, Rgba};
use simulation_lib::render_info::{SimulationParameters, TimeStepInfo};
use std::fs::File;
use std::io::{BufWriter, Read};
use std::path::Path;
use std::time::Duration;

use crossbeam::channel::{Receiver, Sender};

use tracing::{error, info}; // debug, error, info, span, trace, warn,

pub mod commands;
pub mod messages;

use commands::WorkerCommand;
use messages::WorkerMessage;

fn read_recording(file_path: &str) -> std::io::Result<(SimulationParameters, Vec<TimeStepInfo>)> {
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
        info!("Created directories: {}", file_path_parent.display());
    }
    let global_file_path =
        file_path_parent.join(file_path.file_name().expect("No final component found."));

    let mut f = std::fs::File::open(global_file_path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;

    let mut pos: usize = 0;

    let general_info: SimulationParameters = {
        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&buf[pos..pos + 8]);
        pos += 8;

        let len = u64::from_le_bytes(len_bytes) as usize;
        let data = &buf[pos..pos + len];
        pos += len;

        data.into()
    };

    let mut time_steps = Vec::new();

    while pos < buf.len() {
        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&buf[pos..pos + 8]);
        pos += 8;

        let len = u64::from_le_bytes(len_bytes) as usize;
        let data = &buf[pos..pos + len];
        pos += len;

        let ts_info = data.into();
        time_steps.push(ts_info);
    }

    Ok((general_info, time_steps))
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

/// Save padded data as PNG. The `padded_bytes` contain rows with `padded_bpr` bytes per row,
/// with actual tight row length = width * 4.
pub fn save_to_png(
    rgba_data: &[u8],
    width: u32,
    height: u32,
    frame_index: usize,
    output_dir: &std::path::Path,
) -> anyhow::Result<()> {
    let img: ImageBuffer<Rgba<u8>, _> = ImageBuffer::from_raw(width, height, rgba_data)
        .expect("image::ImageBuffer::from_raw failed");

    let filename = format!("frame_{:06}.png", frame_index);
    let file_path = output_dir.join(filename);
    if !output_dir.exists() {
        // Create the parent directory if it does not exist
        std::fs::create_dir_all(output_dir)?;
        info!("Created directory: {}", output_dir.display());
    } else if file_path.exists() {
        // Throw an error if file already exist
        error!("File already exists: {}", file_path.display());
        return Err(std::io::Error::from(std::io::ErrorKind::AlreadyExists).into());
    }

    let file = File::create(file_path)?;
    let writer = BufWriter::new(file);
    img.write_to(&mut BufWriter::new(writer), image::ImageFormat::Png)?;

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
                    info!("Start reading recording...");
                    match read_recording(&file_path) {
                        Ok((sim_info, time_steps)) => {
                            info!("Successfully finished reading recording!");
                            let _ =
                                to_ui.send(WorkerMessage::FinishedReading(sim_info, time_steps));
                        }
                        Err(e) => {
                            info!("Failed reading recording!");
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
                } => {
                    if let Err(e) = save_screenshot_into_directory(
                        &data,
                        width,
                        height,
                        frame_index,
                        &directory,
                    ) {
                        error!("Screenshot failed: {e}");
                        let _ = to_ui.send(WorkerMessage::Error(format!("Screenshot failed: {e}")));
                    } else {
                        info!("Saved screenshot frame {frame_index}");
                    }
                }
                WorkerCommand::SaveScreenshotToFile {
                    data,
                    width,
                    height,
                    file_path,
                } => {
                    if let Err(e) = save_screenshot_to_file(&data, width, height, &file_path) {
                        error!("Screenshot failed: {e}");
                        let _ = to_ui.send(WorkerMessage::Error(format!("Screenshot failed: {e}")));
                    } else {
                        info!("Saved manual screenshot frame");
                    }
                }
                WorkerCommand::Stop => {
                    info!("Stopped backend!");
                    break 'worker;
                }
            },
            Err(crossbeam::channel::TryRecvError::Empty) => {
                std::thread::sleep(Duration::from_millis(16));
            }
            Err(crossbeam::channel::TryRecvError::Disconnected) => {
                error!("Sender was dropped!");
                break 'worker;
            }
        }
    }
}
