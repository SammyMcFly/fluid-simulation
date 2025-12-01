//! Frame control
//!
//!


pub enum NextAction {
    PlayTimeInterval(f32),
    StepInTime,
    Wait,
}


pub struct FrameControl {
    /// Time of last rendering
    pub last_render_time: std::time::Instant,
    /// Time at which a new simulation step was rendered the last time
    pub last_sim_state_render_time: std::time::Instant,
    // pub simulation_time: f32,
    pub time_increment: f32,
    // // new time steps dequeued
    // pub time_steps_dequeued: usize,
    // steps to do
    pub steps_to_do: usize,
}

impl FrameControl {
    pub fn new() -> Self {
        Self {
            last_render_time: std::time::Instant::now(),
            last_sim_state_render_time: std::time::Instant::now(),
            time_increment: f32::default(),
            // time_steps_dequeued: usize::default(),
            steps_to_do: 0,
        }
    }

    pub fn set_time_increment(&mut self, time_inc: f32) {
        self.time_increment = time_inc;
    }

    pub fn rendering_now(&mut self) {
        self.last_render_time = std::time::Instant::now();
    }

    pub fn rendering_new_sim_state_now(&mut self) {
        self.last_sim_state_render_time = std::time::Instant::now();
    }

    pub fn time_since_last_render(&self) -> std::time::Duration {
        self.last_render_time.elapsed()
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

    pub fn step_done(&mut self) {
        self.steps_to_do = self.steps_to_do.saturating_sub(1);
        // self.time_steps_dequeued += num;
    }

    pub fn get_next_action(&mut self, is_playing: bool) -> NextAction {
        if is_playing {
            NextAction::PlayTimeInterval(self.last_sim_state_render_time.elapsed().as_secs_f32())
        } else if self.steps_to_do > 0 {
            NextAction::StepInTime
        } else {
            NextAction::Wait
        }
    }

    pub fn reset(&mut self) {
        self.last_sim_state_render_time = std::time::Instant::now();
        // self.time_steps_dequeued = 0;
        self.steps_to_do = 0;
    }
}
