use iced_widget::{Toggler, button, row};

use crate::ui::UserInput;

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
    pub fn new(start_resumed: bool) -> Self {
        let state = if start_resumed {
            PlaybackState::Resumed
        } else {
            PlaybackState::Paused
        };
        Self {
            state,
            direction: PlaybackDirection::default(),
        }
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

    pub fn is_playing_forward(&self) -> bool {
        matches!(self.direction, PlaybackDirection::Forward)
    }

    pub fn view(&self, discard_past: bool) -> row::Row<'_, UserInput> {
        let play_forward = row![
            button("Play forward")
                .on_press(UserInput::PlayForward)
                .height(28),
        ];

        let play_backward = row![
            button("Play backward")
                .on_press(UserInput::PlayBackward)
                .height(28),
        ];

        let pause = row![button("Pause").on_press(UserInput::Pause).height(28),];

        let step_forward = row![
            button("Step forward")
                .on_press(UserInput::StepForward)
                .height(28),
        ];

        let step_backward = row![
            button("Step backward")
                .on_press(UserInput::StepBackward)
                .height(28),
        ];

        if self.is_playing() && self.is_playing_forward() && discard_past {
            row![pause,].spacing(10)
        } else if self.is_playing() && self.is_playing_forward() {
            // not discard_past
            row![pause, play_backward,].spacing(10)
        } else if self.is_playing() {
            // not self.plays_forward()
            row![pause, play_forward,].spacing(10)
        } else if discard_past {
            // not self.is_playing()
            row![play_forward, step_forward,].spacing(10)
        } else {
            // not self.is_playing() && not discard_past
            row![step_backward, play_forward, play_backward, step_forward,].spacing(10)
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct BufferControl {
    discard_past: bool,
    play_looped: bool,
}

impl BufferControl {
    pub fn new(discard_past: bool) -> Self {
        Self {
            discard_past,
            play_looped: false,
        }
    }

    pub fn toggle_looped(&mut self) {
        self.play_looped = !self.play_looped;
    }

    pub fn toggle_discard_past(&mut self) {
        self.discard_past = !self.discard_past;
    }

    pub fn is_playing_looped(&self) -> bool {
        if self.discard_past {
            false
        } else {
            self.play_looped
        }
    }

    pub fn is_past_discarded(&self) -> bool {
        self.discard_past
    }

    pub fn view(&self) -> row::Row<'_, UserInput> {
        let buffer_control = row![
            Toggler::new(self.discard_past)
                .label("Discard past ")
                .on_toggle(|_| UserInput::DiscardPastToggle),
        ];
        if !self.discard_past {
            let buffer_control = buffer_control.push(
                button("Dicard now")
                    .on_press(UserInput::DiscardPast)
                    .height(28),
            );
            buffer_control.push(
                Toggler::new(self.play_looped)
                    .label("Loop")
                    .on_toggle(|_| UserInput::ToggleLooping),
            )
        } else {
            buffer_control
        }
    }
}
