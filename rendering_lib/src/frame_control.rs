//! Frame control
//!
//!

#[derive(Debug, Clone)]
pub enum Action {
    PlayTimeInterval(f32),
    StepInTime,
    Wait,
}

pub struct FrameControl {
    /// Time of last rendering
    pub last_update_time: std::time::Instant,
    /// Time at which a new simulation step was rendered the last time
    pub last_sim_state_render_time: std::time::Instant,
    // pub simulation_time: f32,
    pub time_increment: f32,
    /// time steps discarded from stored instances in frontend
    pub time_steps_discarded: usize,
    // steps to do
    pub steps_to_do: usize,
}

impl Default for FrameControl {
    fn default() -> Self {
        Self {
            last_update_time: std::time::Instant::now(),
            last_sim_state_render_time: std::time::Instant::now(),
            time_increment: f32::default(),
            time_steps_discarded: usize::default(),
            steps_to_do: 0,
        }
    }
}
impl FrameControl {
    pub fn set_time_increment(&mut self, time_inc: f32) {
        self.time_increment = time_inc;
    }

    pub fn updating_now(&mut self) {
        self.last_update_time = std::time::Instant::now();
    }

    pub fn rendering_new_sim_state_now(&mut self) {
        self.last_sim_state_render_time = std::time::Instant::now();
    }

    pub fn time_since_last_update(&self) -> std::time::Duration {
        self.last_update_time.elapsed()
    }

    pub fn step_forward(&mut self) {
        self.steps_to_do += 1;
    }

    pub fn step_backward(&mut self) {
        self.steps_to_do += 1;
    }

    pub fn reset_steps(&mut self) {
        self.steps_to_do = 0;
    }

    pub fn stepped_in_time(&mut self) {
        self.steps_to_do = self.steps_to_do.saturating_sub(1);
    }

    pub fn count_discarded_timesteps(&mut self, number: usize, discard_past: bool) {
        if discard_past {
            self.time_steps_discarded += number;
        }
    }

    pub fn get_time_steps_discarded(&mut self) -> usize {
        let timesteps = self.time_steps_discarded;
        self.time_steps_discarded = 0;
        timesteps
    }

    pub fn get_next_action(&mut self, is_playing: bool) -> Action {
        if is_playing {
            Action::PlayTimeInterval(self.last_sim_state_render_time.elapsed().as_secs_f32())
        } else if self.steps_to_do > 0 {
            Action::StepInTime
        } else {
            Action::Wait
        }
    }

    pub fn reset(&mut self) {
        self.last_sim_state_render_time = std::time::Instant::now();
        self.time_steps_discarded = 0;
        self.steps_to_do = 0;
    }
}
