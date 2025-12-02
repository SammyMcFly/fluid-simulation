//! Backend module
use std::time::Duration;
use std::io::{Write, Read};
use std::path::{Path};

use iced_winit::winit::event_loop::EventLoopProxy;
use crossbeam::channel::Receiver;

#[cfg(feature = "logging")]
use tracing::{error, info}; // debug, error, info, span, trace, warn,

use commands::WorkerCommand;
use crate::app::messages::WorkerMessage;
use rusty_fluid_solver::{SimulationParameters, TimeStepInfo, SerParticle3D};

pub mod commands;
pub mod rusty_fluid_solver;




fn read_recording(file_path: &str) -> std::io::Result<(SimulationParameters, Vec<TimeStepInfo>)> {
    let file_path = Path::new(file_path);
    // convert to global path
    let file_path_parent = std::fs::canonicalize(
        file_path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."))
    )?;
    // Create the parent directory if it does not exist
    if !file_path_parent.exists() {
        std::fs::create_dir_all(file_path_parent.clone())?;
        #[cfg(feature = "logging")]
        info!("Created directories: {}", file_path_parent.display());
    }
    let global_file_path = file_path_parent.join(file_path.file_name().expect("No final component found."));

    let mut f = std::fs::File::open(global_file_path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;

    let mut pos: usize = 0;

    let general_info: SimulationParameters = {
        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&buf[pos..pos+8]);
        pos += 8;

        let len = u64::from_le_bytes(len_bytes) as usize;
        let data = &buf[pos..pos+len];
        pos += len;

        data.into()
    };

    let mut time_steps = Vec::new();

    while pos < buf.len() {
        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&buf[pos..pos+8]);
        pos += 8;

        let len = u64::from_le_bytes(len_bytes) as usize;
        let data = &buf[pos..pos+len];
        pos += len;

        let ts_info = data.into();
        time_steps.push(ts_info);
    }

    Ok((general_info, time_steps))
}

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

/// Store the current state of all fluid particles to a file
fn save_image(particles: Vec<f64>, file_path: &str) -> std::io::Result<()> {
    let file_path = Path::new(file_path);
    // convert to global path
    let file_path_parent = std::fs::canonicalize(
        file_path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."))
    )?;
    // Create the parent directory if it does not exist
    if !file_path_parent.exists() {
        std::fs::create_dir_all(file_path_parent.clone())?;
        #[cfg(feature = "logging")]
        info!("Created directories: {}", file_path_parent.display());
    }
    let global_file_path = file_path_parent.join(file_path.file_name().expect("No final component found."));
    let ron_string = ron::to_string(&particles).unwrap();
    let mut file = std::fs::File::create(global_file_path)?;
    file.write_all(ron_string.as_bytes())?;
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
                        #[cfg(feature = "logging")]
                        info!("Start reading recording...");
                        match read_recording(&file_path) {
                            Ok((sim_info, time_steps)) => {
                                #[cfg(feature = "logging")]
                                info!("Successfully finished reading recording!");
                                let _ = to_ui.send_event(WorkerMessage::FinishedReading(sim_info, time_steps));
                            },
                            Err(e) => {
                                #[cfg(feature = "logging")]
                                info!("Failed reading recording!");
                                let _ = to_ui.send_event(WorkerMessage::Error(e.to_string()));
                            },
                        }
                    },
                    WorkerCommand::SaveImage() => {
                        // match save_image(particles, file_path) {
                        //     Ok(_) => {
                        //         let _ = to_ui.send_event(WorkerMessage::FinishedSavingImage);
                        //     },
                        //     Err(e) => {
                        //         let _ = to_ui.send_event(WorkerMessage::Error(e.to_string()));
                        //     },
                        // }
                    },
                    WorkerCommand::SaveState { particles, file_path } => {
                        let save_message = if save_system_state(particles, &file_path).is_ok() {
                            #[cfg(feature = "logging")]
                            info!("Successfully saved state: {}", file_path);
                            WorkerMessage::SavedState
                        } else {
                            #[cfg(feature = "logging")]
                            error!("Failed to save state!");
                            WorkerMessage::Error("Failed to save state!".to_string())
                        };
                        let _ = to_ui.send_event(save_message);
                    },
                    WorkerCommand::Stop => {
                        #[cfg(feature = "logging")]
                        info!("Stopped backend!");
                        break 'worker;
                    },
                }
            }
            Err(crossbeam::channel::TryRecvError::Empty) => {
                std::thread::sleep(Duration::from_millis(16));
            }
            Err(crossbeam::channel::TryRecvError::Disconnected) => {
                #[cfg(feature = "logging")]
                error!("Sender was dropped!");
                break 'worker;
            }
        }
    }
}