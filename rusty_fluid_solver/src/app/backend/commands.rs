//! Worker commands
use rendering_lib::readback::ReadbackRequest;
use simulation_lib::measurement::MeasurementSeries;
use simulation_lib::render_info::{FluidVisualization, TimeStepInfo};

pub enum WorkerCommand {
    Simulate {
        params_file_path: String,
        scene_file_path: String,
        state_file_path: Option<String>,
        measurement_file_path: Option<String>,
        start_time: Option<f64>,
        finish_time: Option<f64>,
        recording_file: Option<String>,
        with_info: Box<TimeStepInfo>,
    },
    AddTimeStepsToCompute(usize),
    SaveScreenshot(ReadbackRequest),
    SaveState {
        fluid: FluidVisualization,
        filepath: String,
    },
    SaveMeasurement {
        measurement_series: MeasurementSeries,
    },
    // Resume,
    // Pause,
    Reset,
    Stop,
}
