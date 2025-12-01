//! Messages to front end

use crate::app::backend::rusty_fluid_solver::{SimulationParameters, TimeStepInfo};



pub enum WorkerMessage {
    FinishedReading(SimulationParameters, Vec<TimeStepInfo>),
    SavedImage,
    SavedState,
    Error(String),
}