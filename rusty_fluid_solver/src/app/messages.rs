//! Messages to front end
use simulation_lib::render_info::{SimulationParameters, TimeStepInfo};

pub enum WorkerMessage {
    TimeIncFinished(Box<TimeStepInfo>),
    SimulationLoaded(SimulationParameters),
    SavedScreenshot,
    SavedState,
    SavedMeasurement,
    FinishedResetting(SimulationParameters),
    ReachedStartTime,
    ReachedFinishTime,
    Error(Box<dyn std::error::Error + Send + Sync>),
}
