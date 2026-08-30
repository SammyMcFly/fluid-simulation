use rendering_lib::pipeline::ScreenshotCommand;
use simulation_lib::render_info::TimeStepInfo;
use std::path::PathBuf;

/// Commands sent from the UI to the worker thread
#[derive(Debug, Clone)]
pub enum WorkerCommand {
    Simulate {
        params_file_path: String,
        scene_file_path: String,
        state_file_path: Option<String>,
        measurement_file_path: Option<String>,
        start_time: Option<f64>,
        finish_time: Option<f64>,
        recording_file: Option<std::path::PathBuf>,
        rendering_dir: Option<std::path::PathBuf>,
        with_info: Box<TimeStepInfo>,
    },
    AddTimeStepsToCompute(usize),
    SaveState {
        time_step_number: u64,
        file_path: std::path::PathBuf,
    },
    WriteRendering {
        data: Vec<u8>,
        width: u32,
        height: u32,
        frame_index: usize,
        overwrite: bool,
    },
    /// Save screenshot to an explicit full file path
    SaveScreenshotToFile {
        data: Vec<u8>,
        width: u32,
        height: u32,
        file_path: std::path::PathBuf,
    },
    Reload,
    ContinueFromTimeStep {
        with_info: Box<TimeStepInfo>,
    },
    Stop,
}

impl ScreenshotCommand for WorkerCommand {
    fn write_rendering(
        data: Vec<u8>,
        width: u32,
        height: u32,
        frame_index: usize,
        _directory: std::path::PathBuf,
        overwrite: bool,
    ) -> Self {
        WorkerCommand::WriteRendering {
            data,
            width,
            height,
            frame_index,
            overwrite,
        }
    }
    fn save_screenshot_to_file(data: Vec<u8>, width: u32, height: u32, file_path: PathBuf) -> Self {
        WorkerCommand::SaveScreenshotToFile {
            data,
            width,
            height,
            file_path,
        }
    }
}
