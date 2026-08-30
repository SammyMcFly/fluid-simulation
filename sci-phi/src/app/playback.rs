//! Playback logic: buffering timesteps, stepping through time, play/pause.

use simulation_lib::render_info::TimeStepInfo;

// ─── Frame Control ────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Action {
    PlayTimeInterval(f32),
    StepInTime(PlaybackDirection),
    Wait,
}

pub struct FrameControl {
    /// Time at which a new simulation step was rendered the last time
    last_sim_state_render_time: std::time::Instant,
    /// Time increment of current timestep
    pub time_increment: f32,
    /// Time steps discarded from stored instances
    pub time_steps_discarded: usize,
    /// Steps to do (for single-stepping)
    steps_to_do: i32,
}

impl Default for FrameControl {
    fn default() -> Self {
        Self {
            last_sim_state_render_time: std::time::Instant::now(),
            time_increment: 0.0,
            time_steps_discarded: 0,
            steps_to_do: 0,
        }
    }
}

impl FrameControl {
    pub fn set_time_increment(&mut self, time_inc: f32) {
        self.time_increment = time_inc;
    }

    pub fn rendering_new_sim_state_now(&mut self) {
        self.last_sim_state_render_time = std::time::Instant::now();
    }

    pub fn step_forward(&mut self) {
        self.steps_to_do += 1;
    }

    pub fn step_backward(&mut self) {
        self.steps_to_do -= 1;
    }

    pub fn reset_steps(&mut self) {
        self.steps_to_do = 0;
    }

    pub fn stepped_in_time(&mut self, direction: PlaybackDirection) {
        match direction {
            PlaybackDirection::Forward => self.steps_to_do -= 1,
            PlaybackDirection::Backward => self.steps_to_do += 1,
        }
    }

    pub fn count_discarded_time_steps(&mut self, number: usize, discard_past: bool) {
        if discard_past {
            self.time_steps_discarded += number;
        }
    }

    pub fn get_and_reset_time_steps_discarded(&mut self) -> usize {
        let timesteps = self.time_steps_discarded;
        self.time_steps_discarded = 0;
        timesteps
    }

    pub fn get_next_action(
        &self,
        is_playing: bool,
        is_playing_forward: bool,
        is_rendering_active: bool,
    ) -> Action {
        if is_rendering_active && is_playing {
            let dir = if is_playing_forward {
                PlaybackDirection::Forward
            } else {
                PlaybackDirection::Backward
            };
            Action::StepInTime(dir)
        } else if is_playing {
            Action::PlayTimeInterval(self.last_sim_state_render_time.elapsed().as_secs_f32())
        } else if self.steps_to_do > 0 {
            Action::StepInTime(PlaybackDirection::Forward)
        } else if self.steps_to_do < 0 {
            Action::StepInTime(PlaybackDirection::Backward)
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

// ─── Staging Result ───────────────────────────────────────────

#[derive(Debug)]
pub enum StagingResult {
    Initialized,
    SteppedInTime {
        direction: PlaybackDirection,
        discarded: usize,
    },
    SomeTaken {
        discarded: usize,
    },
    StoppedAtLoopEndWithSomeTaken {
        discarded: usize,
    },
    StoppedAtLoopEndWithNoneTaken,
    NoneTaken,
    NothingToStage,
    Uninitialized,
}

// ─── Instance Store ───────────────────────────────────────────

#[derive(Default)]
pub struct InstanceStore {
    info_buffer: Vec<TimeStepInfo>,
    buffer_length_limit: usize,
    current_index: usize,
    active: bool,
    allow_looping_once: bool,
    number_min: u64,
    number_max: u64,
}

pub enum InsertionResult {
    TooOld,
    ReplacedOther,
    ReplacedCurrent,
    Pushed,
    TooNew,
}

impl InstanceStore {
    // pub fn new() -> Self {
    //     Self {
    //         info_buffer: Vec::new(),
    //         current_index: 0,
    //         active: false,
    //         allow_looping_once: false,
    //     }
    // }

    pub fn set_length_limit(&mut self, limit: usize) {
        self.buffer_length_limit = limit;
    }

    pub fn buffer_length_limit(&self) -> usize {
        self.buffer_length_limit
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Replace the entire buffer (e.g. loading from file)
    pub fn store(&mut self, time_steps: Vec<TimeStepInfo>) {
        self.info_buffer = time_steps;
        self.current_index = 0;
        self.active = false;
        if !self.info_buffer.is_empty() {
            self.number_min = self.info_buffer.first().unwrap().time_step_number;
            self.number_max = self.info_buffer.last().unwrap().time_step_number;
        }
    }

    pub fn insert(&mut self, time_step_info: TimeStepInfo) -> InsertionResult {
        if self.info_buffer.is_empty() {
            self.number_min = time_step_info.time_step_number;
            self.number_max = time_step_info.time_step_number;
            self.info_buffer.push(time_step_info);
            return InsertionResult::Pushed;
        }
        if time_step_info.time_step_number == self.number_max + 1 {
            self.number_max += 1;
            self.info_buffer.push(time_step_info);
            return InsertionResult::Pushed;
        }
        if time_step_info.time_step_number < self.number_min {
            return InsertionResult::TooOld;
        }
        if time_step_info.time_step_number >= self.number_min
            && time_step_info.time_step_number <= self.number_max
            && let Some(idx) = self
                .info_buffer
                .iter()
                .position(|info| info.time_step_number == time_step_info.time_step_number)
        {
            if idx == self.current_index {
                self.info_buffer[idx] = time_step_info;
                return InsertionResult::ReplacedCurrent;
            } else {
                self.info_buffer[idx] = time_step_info;
                return InsertionResult::ReplacedOther;
            }
        }
        // tracing::debug!(
        //     "Could not insert time step: ts_info_ts_number_inc {:?}, ts_info_ts_number {:?}, current_index {:?}, min {:?}, max {:?}",
        //     time_step_info.time_step_number,
        //     self.info_buffer[self.current_index].time_step_number,
        //     self.current_index,
        //     self.number_min,
        //     self.number_max
        // );
        InsertionResult::TooNew
    }

    pub fn get_first_time_step_info(&self) -> Option<&TimeStepInfo> {
        if !self.info_buffer.is_empty() {
            Some(&self.info_buffer[0])
        } else {
            None
        }
    }

    pub fn get_current_time_step_info(&self) -> Option<&TimeStepInfo> {
        if self.active {
            Some(&self.info_buffer[self.current_index])
        } else {
            None
        }
    }

    pub fn get_time_inc(&self) -> f32 {
        if self.active {
            self.info_buffer[self.current_index]
                .measurement
                .time_step_size as f32
        } else {
            0.0
        }
    }

    pub fn remaining_buffer_len(&self) -> usize {
        if self.info_buffer.is_empty() {
            0
        } else {
            self.info_buffer.len() - (self.current_index + 1)
        }
    }

    pub fn can_step_forward(&self) -> bool {
        self.active && self.current_index + 1 < self.info_buffer.len()
    }

    pub fn can_step_backward(&self) -> bool {
        self.active && self.current_index > 0
    }

    pub fn finished_loop(&self, forward: bool) -> bool {
        if self.info_buffer.is_empty() {
            return true;
        }
        if forward {
            self.current_index == self.info_buffer.len() - 1
        } else {
            self.current_index == 0
        }
    }

    pub fn allow_looping_once(&mut self, looped_playback: bool) {
        if !looped_playback {
            self.allow_looping_once = true;
        }
    }

    pub fn reset_allow_looping_once(&mut self) {
        self.allow_looping_once = false;
    }

    pub fn discard_past(&mut self) -> usize {
        if self.current_index == 0 {
            return 0;
        }
        let discarded = self.current_index;
        self.info_buffer.drain(0..self.current_index);
        self.current_index = 0;
        self.number_min = self
            .info_buffer
            .first()
            .expect("buffer is empty")
            .time_step_number;
        discarded
    }

    pub fn discard_future(&mut self) -> usize {
        let discarded = self.info_buffer.len() - self.current_index - 1;
        self.info_buffer.drain((self.current_index + 1)..);
        self.number_max = self.info_buffer[self.current_index].time_step_number;
        discarded
    }

    pub fn reset(&mut self, clear_buffer: bool) {
        if clear_buffer {
            self.info_buffer.clear();
        }
        self.current_index = 0;
        self.active = false;
        self.allow_looping_once = false;
        self.number_min = 0;
        self.number_max = 0;
    }

    /// Advance to next frame, returns true if loop boundary hit (unallowed)
    fn next_index(&mut self, forward: bool, looped: bool) -> bool {
        if forward {
            if self.current_index + 1 < self.info_buffer.len() {
                self.current_index += 1;
                false
            } else if looped || self.allow_looping_once {
                self.current_index = 0;
                self.allow_looping_once = false;
                false
            } else {
                true
            }
        } else if self.current_index > 0 {
            self.current_index -= 1;
            false
        } else if looped || self.allow_looping_once {
            self.current_index = self.info_buffer.len() - 1;
            self.allow_looping_once = false;
            false
        } else {
            true
        }
    }

    fn activate(&mut self, discard_past: bool) -> usize {
        self.active = true;
        if discard_past { self.discard_past() } else { 0 }
    }

    /// Determine what to stage next based on playback action.
    pub fn stage_next(
        &mut self,
        action: Action,
        forward: bool,
        looped_playback: bool,
        discard_past: bool,
    ) -> StagingResult {
        if self.info_buffer.is_empty() && !self.active {
            return StagingResult::Uninitialized;
        }
        if self.info_buffer.is_empty() {
            return StagingResult::NothingToStage;
        }
        if !self.active {
            self.activate(false);
            return StagingResult::Initialized;
        }
        match action {
            Action::PlayTimeInterval(interval) => {
                let mut taken = 0;
                let mut remaining = interval
                    - self.info_buffer[self.current_index]
                        .measurement
                        .time_step_size as f32;
                while remaining >= 0.0 {
                    if self.next_index(forward, looped_playback) {
                        if taken > 0 {
                            let discarded = self.activate(discard_past);
                            return StagingResult::StoppedAtLoopEndWithSomeTaken { discarded };
                        }
                        return StagingResult::StoppedAtLoopEndWithNoneTaken;
                    }
                    taken += 1;
                    remaining -= self.info_buffer[self.current_index]
                        .measurement
                        .time_step_size as f32;
                }
                if taken > 0 {
                    let discarded = self.activate(discard_past);
                    StagingResult::SomeTaken { discarded }
                } else {
                    StagingResult::NoneTaken
                }
            }
            Action::StepInTime(dir) => {
                self.next_index(matches!(dir, PlaybackDirection::Forward), true);
                let discarded = self.activate(discard_past);
                StagingResult::SteppedInTime {
                    direction: dir,
                    discarded,
                }
            }
            Action::Wait => StagingResult::NoneTaken,
        }
    }
}

// ─── Playback State ───────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Default)]
pub enum PlaybackState {
    Playing,
    #[default]
    Paused,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum PlaybackDirection {
    #[default]
    Forward,
    Backward,
}

#[derive(Debug, Clone, Default)]
pub struct PlaybackControls {
    state: PlaybackState,
    pub direction: PlaybackDirection,
    pub discard_past: bool,
    pub play_looped: bool,
}

impl PlaybackControls {
    pub fn new(start_resumed: bool, discard_past: bool) -> Self {
        Self {
            state: if start_resumed {
                PlaybackState::Playing
            } else {
                PlaybackState::Paused
            },
            direction: PlaybackDirection::Forward,
            discard_past,
            play_looped: false,
        }
    }

    pub fn is_playing(&self) -> bool {
        matches!(self.state, PlaybackState::Playing)
    }

    pub fn is_playing_forward(&self) -> bool {
        matches!(self.direction, PlaybackDirection::Forward)
    }

    pub fn is_looped(&self) -> bool {
        if self.discard_past {
            false
        } else {
            self.play_looped
        }
    }

    pub fn play(&mut self) {
        self.state = PlaybackState::Playing;
    }

    pub fn pause(&mut self) {
        self.state = PlaybackState::Paused;
    }
}
