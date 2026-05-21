//! Settings
//!
//!

#[derive(Debug, Clone)]
pub struct Settings {
    pub wait_for_timesteps: bool,
}

impl Settings {
    pub fn new(wait_for_timesteps: bool) -> Self {
        Self { wait_for_timesteps }
    }
}
