use simulation_lib::render_info::{FluidVisualization, TimeStepInfo};

/// Commands sent from the UI to the worker thread
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
        fluid: FluidVisualization,
        file_path: std::path::PathBuf,
    },
    WriteRendering {
        data: Vec<u8>,
        width: u32,
        height: u32,
        frame_index: usize,
    },
    /// Save screenshot to an explicit full file path
    SaveScreenshotToFile {
        data: Vec<u8>,
        width: u32,
        height: u32,
        file_path: std::path::PathBuf,
    },
    Reload,
    Stop,
}
