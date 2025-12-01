use iced_widget::{row, button, Toggler};

use crate::app::rendering::ui::UserInput;


#[derive(Debug, Default, Clone, PartialEq)]
pub enum PlaybackState {
    Resumed,
    #[default]
    Paused,
}

impl PlaybackState {
    pub fn play(&mut self) {
        *self = Self::Resumed;
    }

    pub fn pause(&mut self) {
        *self = Self::Paused;
    }

    pub fn toggle(&mut self) {
        match *self {
            Self::Resumed => *self = Self::Paused,
            Self::Paused => *self = Self::Resumed,
        }
    }

    pub fn is_playing(&self) -> bool {
        matches!(self, Self::Resumed)
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub enum PlaybackDirection {
    #[default]
    Forward,
    Backward,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct PlaybackControls {
    state: PlaybackState,
    direction: PlaybackDirection,
}

impl PlaybackControls {
    pub fn new(start_resumed: bool,) -> Self {
        let state = if start_resumed {
            PlaybackState::Resumed
        } else {
            PlaybackState::Paused
        };
        Self { state, direction: PlaybackDirection::default() }
    }

    pub fn play(&mut self) {
        self.state.play();
    }

    pub fn pause(&mut self) {
        self.state.pause();
    }

    pub fn forward(&mut self) {
        self.direction = PlaybackDirection::Forward;
    }

    pub fn backward(&mut self) {
        self.direction = PlaybackDirection::Backward;
    }

    pub fn toggle(&mut self) {
        self.state.toggle();
    }

    pub fn is_playing(&self) -> bool {
        self.state.is_playing()
    }

    pub fn plays_forward(&self) -> bool {
        matches!(self.direction, PlaybackDirection::Forward)
    }

    pub fn view(&self) -> row::Row<'_, UserInput> {
        let play_forward = row![
            button("Play forward")
            .on_press(UserInput::PlayForward).height(28),
        ];

        let play_backward = row![
            button("Play backward")
            .on_press(UserInput::PlayBackward).height(28),
        ];

        let pause = row![
            button("Pause")
            .on_press(UserInput::Pause).height(28),
        ];

        let step_forward = row![
            button("Step forward")
            .on_press(UserInput::StepForward).height(28),
        ];

        let step_backward = row![
            button("Step backward")
            .on_press(UserInput::StepBackward).height(28),
        ];

        if self.is_playing() && self.plays_forward() {
            row![
                pause,
                play_backward,
            ].spacing(10)
        } else if self.is_playing() { // !self.plays_forward()
            row![
                pause,
                play_forward,
            ].spacing(10)
        } else {
            row![
                step_backward,
                play_forward,
                play_backward,
                step_forward,
            ].spacing(10)
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct LoopControl {
    state: bool,
}

impl LoopControl {
    pub fn toggle(&mut self) {
        self.state = !self.state;
    }

    pub fn play_looped(&self) -> bool {
        self.state
    }

    pub fn view(&self) -> Toggler<'_, UserInput> {
        Toggler::new(self.state)
            .label("Loop")
            .on_toggle(|_| UserInput::ToggleLooping)
    }
}