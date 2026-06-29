//! Messages to front end

use simulation_lib::render_info::{SimulationParameters, TimeStepInfo};

pub enum WorkerMessage {
    FinishedReading(SimulationParameters, Vec<TimeStepInfo>),
    SavedScreenshot,
    SavedState,
    Error(Box<dyn std::error::Error + Send + Sync>),
}
