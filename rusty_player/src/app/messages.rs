//! Messages to front end

use simulation_lib::{SimulationParameters, TimeStepInfo};



pub enum WorkerMessage {
    FinishedReading(SimulationParameters, Vec<TimeStepInfo>),
    SavedScreenshot,
    SavedState,
    Error(String),
}