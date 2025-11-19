//! Messages to front end
use super::backend::{TimeStepInfo, SimulationInfo};



pub enum WorkerMessage {
    TimeIncFinished(TimeStepInfo),
    SimulationLoaded(SimulationInfo),
    SavedState,
    // SavedMeasurement,
    FinishedResetting,
    // FinishedMeasurement,
    Error(String),
}