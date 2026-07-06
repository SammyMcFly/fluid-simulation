use rendering_lib::pipeline::ScreenshotCommand;
use std::path::PathBuf;

/// Commands sent from the UI to the worker thread
#[derive(Debug, Clone)]
pub enum WorkerCommand {
    ReadRecording(String),
    WriteRendering {
        data: Vec<u8>,
        width: u32,
        height: u32,
        frame_index: usize,
        directory: std::path::PathBuf,
    },
    /// Save screenshot to an explicit full file path
    SaveScreenshotToFile {
        data: Vec<u8>,
        width: u32,
        height: u32,
        file_path: std::path::PathBuf,
    },
    Stop,
}

impl ScreenshotCommand for WorkerCommand {
    fn write_rendering(
        data: Vec<u8>,
        width: u32,
        height: u32,
        frame_index: usize,
        directory: std::path::PathBuf,
    ) -> Self {
        WorkerCommand::WriteRendering {
            data,
            width,
            height,
            frame_index,
            directory,
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
