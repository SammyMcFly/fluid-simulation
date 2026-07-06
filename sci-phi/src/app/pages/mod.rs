pub mod plotting;
pub mod simulation;

/// The page to display in the application.
pub enum Page {
    Simulation,
    Measurements,
}

/// The context page to display in the context drawer.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub enum ContextPage {
    SimulationSettings,
    PlottingSettings,
    #[default]
    About,
}
