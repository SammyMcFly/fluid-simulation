//! Messages to front end
use super::backend::{TimeStepInfo, SimulationParameters};



pub enum WorkerMessage {
    TimeIncFinished(TimeStepInfo),
    SimulationLoaded(SimulationParameters),
    SavedState,
    SavedMeasurement,
    FinishedResetting(SimulationParameters),
    ReachedFinishTime,
    Error(String),
}