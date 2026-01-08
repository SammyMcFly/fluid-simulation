//! Messages to front end
use simulation_lib::{TimeStepInfo, SimulationParameters};



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