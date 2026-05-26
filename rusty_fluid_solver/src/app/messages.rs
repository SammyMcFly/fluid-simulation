//! Messages to front end
use simulation_lib::{SimulationParameters, TimeStepInfo};

pub enum WorkerMessage {
    TimeIncFinished(TimeStepInfo),
    SimulationLoaded(SimulationParameters),
    SavedScreenshot,
    SavedState,
    SavedMeasurement,
    FinishedResetting(SimulationParameters),
    ReachedStartTime,
    ReachedFinishTime,
    Error(String),
}
