use simulation_lib::render_info::{SimulationParameters, TimeStepInfo};

/// Messages sent from the worker thread to the UI
#[derive(Debug, Clone)]
pub enum WorkerMessage {
    FinishedReading(SimulationParameters, Vec<TimeStepInfo>),
    SavedScreenshot,
    SavedState,
    Error(String),
}
