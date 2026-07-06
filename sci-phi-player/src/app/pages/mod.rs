pub mod simulation;

/// The page to display in the application.
pub enum Page {
    Simulation,
}

/// The context page to display in the context drawer.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub enum ContextPage {
    SimulationSettings,
    #[default]
    About,
}
