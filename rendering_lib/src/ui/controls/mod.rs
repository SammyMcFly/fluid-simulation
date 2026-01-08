//! Render Controls, Settings and Utilities
//!
//!
use iced_widget::{container, column, row, button, Toggler, Space}; //text_input, slider
use iced_winit::core::{Element, Theme, Length, Color};

use simulation_lib::{SimulationParameters, TimeStepInfo, ParticleColor};

use crate::ui::UserInput;
use playback::*;
use cut::*;

pub mod playback;
pub mod cut;
pub mod info;




#[derive(Debug, Clone)]
pub struct RenderControls {
    pub playback_controls: PlaybackControls,
    pub loop_control: LoopControl,
    pub particle_color: ParticleColor,
    pub boundary_particle_color: ParticleColor,
    pub background_color: iced_winit::core::Color,
    pub hide_boundary: bool,
    pub cut: cut::Cut,
    pub discard_past: bool,

    pub info: info::UIInfo,
}

impl RenderControls {
    pub fn new(
        particle_color: ParticleColor,
        boundary_particle_color: ParticleColor,
        start_resumed: bool,
        is_rendered: bool,
        discard_past: bool,
    ) -> Self {
        Self {
            playback_controls: PlaybackControls::new(start_resumed),
            loop_control: LoopControl::default(),
            particle_color,
            boundary_particle_color,
            background_color: Color::WHITE,
            hide_boundary: false,
            cut: Cut::default(),
            discard_past,

            info: info::UIInfo::new(is_rendered),
        }
    }

    pub fn background_color(&self) -> Color {
        self.background_color
    }

    pub fn is_boundary_hidden(&self) -> bool {
        self.hide_boundary
    }

    pub fn get_cut(&self) -> &Cut {
        &self.cut
    }

    pub fn update(&mut self, input: &UserInput) {
        match input {
            UserInput::PlayForward => {
                self.playback_controls.forward();
                self.playback_controls.play();
            },
            UserInput::PlayBackward => {
                self.playback_controls.backward();
                self.playback_controls.play();
            },
            UserInput::Pause => {
                self.playback_controls.pause();
            },
            UserInput::StepForward => {
                self.playback_controls.forward();
            }
            UserInput::StepBackward => {
                self.playback_controls.backward();
            }
            UserInput::ToggleLooping => {
                self.loop_control.toggle();
            },
            UserInput::ToggleHideBoundary => {
                self.hide_boundary = !self.hide_boundary
            }
            UserInput::ToggleCutX => {
                self.cut.x = !self.cut.x;
            }
            UserInput::CutXBoundChanged(bound) => {
                self.cut.x_bound += bound
            }
            UserInput::FlipCutX => {
                self.cut.x_flip();
            }
            UserInput::ToggleCutY => {
                self.cut.y = !self.cut.y;
            }
            UserInput::CutYBoundChanged(bound) => {
                self.cut.y_bound += bound;
            }
            UserInput::FlipCutY => {
                self.cut.y_flip();
            }
            UserInput::DiscardPastToggle => {
                self.discard_past = !self.discard_past;
            },
            _ => (),
        }
    }

    pub fn new_simulation(&mut self, info: SimulationParameters) {
        self.particle_color = info.particle_color;
        self.boundary_particle_color = info.boundary_particle_color;
        self.info.update_simulation_info(info);
    }

    pub fn update_time_step_info(&mut self, info: Option<&TimeStepInfo>, queue_length: usize) {
        self.info.update_time_step_info(info, queue_length);
    }

    pub fn advance_to_next_measurement_state(&mut self) {
        self.info.advance_to_next_measurement_state();
    }
    pub fn advance_to_next_recording_state(&mut self) {
        self.info.advance_to_next_recording_state();
    }

    pub fn view(&self) -> Element<'_, UserInput, Theme, iced_wgpu::Renderer> {
        let reset_cam = row![
            button("Reset Camera")
            .on_press(UserInput::RequestCameraReset).height(28),
        ];

        let reset = row![
            button("Reset")
            .on_press(UserInput::RequestReset).height(28),
        ];

        let save_state = row![
            button("Save current state")
            .on_press(UserInput::RequestSaving).height(28),
        ];

        let screenshot = row![
            button("Take screenshot")
            .on_press(UserInput::RequestScreenshot).height(28),
        ];

        let hide_boundary = row![
            Toggler::new(self.hide_boundary)
                .label("Hide boundary")
                .on_toggle(|_| UserInput::ToggleHideBoundary),
        ];

        let forget_past = row![
            Toggler::new(self.discard_past)
                .label("Discard past ")
                .on_toggle(|_| UserInput::DiscardPastToggle),
            button("Dicard now")
                .on_press(UserInput::DiscardPast).height(28)
        ];

        // build ui controls
        let ui_controls = column![
            self.playback_controls.view(),
            self.loop_control.view(),
            reset_cam,
            reset,
            save_state,
            screenshot,
            hide_boundary,
            self.cut.view(),
            forget_past,
        ]
        .spacing(10);

        // final assembly of the ui
        container(
            column![
                self.info.view(),
                Space::with_height(Length::Fill),
                ui_controls,
            ]
            .spacing(10),
        )
        .align_left(Length::Fill)
        .align_bottom(Length::Fill)
        .padding(10)
        .into()
    }
}
