//! Backend module
use image::{ImageBuffer, Rgba};
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;
use std::time::Duration;

use crossbeam::channel::Receiver;
use iced_wgpu::wgpu;
use iced_winit::winit::event_loop::EventLoopProxy;

use tracing::{error, info}; // debug, error, info, span, trace, warn,

use rendering_lib::readback::{ReadbackBuffer, ReadbackRequest};
use simulation_lib::render_info::{FluidVisualization, SimulationParameters, TimeStepInfo};

pub mod commands;

use crate::app::messages::WorkerMessage;
use commands::WorkerCommand;

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

/// Store the current state of all fluid particles to a file
pub fn save_system_state(fluid: FluidVisualization, file_path: &str) -> std::io::Result<()> {
    let file_path = Path::new(file_path);
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

fn save_screenshot(
    data: &[u8],
    rbr: &ReadbackRequest,
    buffer: &ReadbackBuffer,
    path: &Path,
) -> anyhow::Result<()> {
    let rgba_data = buffer_to_rgba(
        data,
        rbr.width,
        rbr.height,
        buffer.padded_bytes_per_row as usize,
    )?;

    save_to_png(&rgba_data, rbr.width, rbr.height, rbr.frame_index, path)?;

    Ok(())
}

/// Function that does:
/// - receives [[WorkerCommand]] from front-end
/// - passes [[WorkerCommand]] to [[SimulationController]]
/// - sends [[WorkerMessage]] back to front-end
pub fn worker_loop(from_ui: Receiver<WorkerCommand>, to_ui: EventLoopProxy<WorkerMessage>) {
    'worker: loop {
        match from_ui.try_recv() {
            Ok(msg) => {
                match msg {
                    WorkerCommand::ReadRecording(file_path) => {
                        info!("Start reading recording...");
                        match read_recording(&file_path) {
                            Ok((sim_info, time_steps)) => {
                                info!("Successfully finished reading recording!");
                                let _ = to_ui.send_event(WorkerMessage::FinishedReading(
                                    sim_info, time_steps,
                                ));
                            }
                            Err(e) => {
                                info!("Failed reading recording!");
                                let _ = to_ui.send_event(WorkerMessage::Error(e.to_string().into()));
                            }
                        }
                    }
                    WorkerCommand::SaveState {
                        fluid: particles,
                        filepath,
                    } => {
                        let save_message = if save_system_state(particles, &filepath).is_ok() {
                            info!("Successfully saved state: {}", filepath);
                            WorkerMessage::SavedState
                        } else {
                            error!("Failed to save state!");
                            WorkerMessage::Error("Failed to save state!".to_string().into())
                        };
                        let _ = to_ui.send_event(save_message);
                    }
                    WorkerCommand::SaveScreenshot(rbr) => {
                        let buffer = rbr.buffer.lock().unwrap();
                        let buffer_slice = buffer.buffer.slice(..);
                        let (tx, rx) = crossbeam::channel::bounded::<()>(1);
                        // rbr.buffer.lock().unwrap().mapping_started = true;
                        buffer_slice.map_async(wgpu::MapMode::Read, move |_| {
                            tx.send(()).ok();
                        });
                        // Drive the future to completion
                        rbr.device.poll(wgpu::Maintain::Wait);

                        // Wait for callback
                        rx.recv().unwrap();

                        let data = {
                            let slice = buffer_slice.get_mapped_range();
                            slice.to_vec()
                        };

                        // Free buffer for next use
                        buffer.buffer.unmap();
                        // buffer.mapping_started = false;

                        match save_screenshot(&data, &rbr, &buffer, &rbr.output_dir) {
                            Ok(_) => {
                                let _ = to_ui.send_event(WorkerMessage::SavedScreenshot);
                            }
                            Err(e) => {
                                let _ = to_ui.send_event(WorkerMessage::Error(e.to_string().into()));
                            }
                        }
                    }
                    WorkerCommand::Stop => {
                        info!("Stopped backend!");
                        break 'worker;
                    }
                }
            }
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
