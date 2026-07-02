use simulation_lib::render_info::{SimulationParameters, TimeStepInfo};

/// Messages sent from the worker thread to the UI
#[derive(Debug, Clone)]
pub enum WorkerMessage {
    TimeStepReady(Box<TimeStepInfo>),
    SimulationLoaded(SimulationParameters),
    FinishedResetting(SimulationParameters),
    ReachedStartTime,
    ReachedFinishTime,
    SavedState,
    SavedMeasurement,
    Error(String),
}
