//! Render Controls, Settings and Utilities
//!
//!
use iced_widget::{Space, Toggler, button, column, container, row}; //text_input, slider
use iced_winit::core::{Color, Element, Length, Theme};

use simulation_lib::render_info::{
    BoundaryVisualization, FluidVisualization, SimulationParameters, TimeStepInfo,
};

use crate::ui::UserInput;
use cut::*;
use playback::*;

pub mod cut;
pub mod info;
pub mod playback;

#[derive(Debug, Clone)]
pub struct RenderControls {
    pub playback_controls: PlaybackControls,
    pub buffer_control: BufferControl,
    pub hide_boundary: bool,
    pub cut: cut::Cut,
    pub fluid_visualization: FluidVisualization,
    pub boundary_visualization: BoundaryVisualization,
    pub background_color: iced_winit::core::Color,

    pub info: info::UIInfo,
}

impl RenderControls {
    pub fn new(
        fluid_visualization: FluidVisualization,
        boundary_visualization: BoundaryVisualization,
        start_resumed: bool,
        is_rendered: bool,
        discard_past: bool,
    ) -> Self {
        Self {
            playback_controls: PlaybackControls::new(start_resumed),
            buffer_control: BufferControl::new(discard_past),
            hide_boundary: false,
            cut: Cut::default(),
            fluid_visualization,
            boundary_visualization,
            background_color: Color::WHITE,

            info: info::UIInfo::new(is_rendered),
        }
    }

    pub fn background_color(&self) -> Color {
        self.background_color
    }

    pub fn is_playing(&self) -> bool {
        self.playback_controls.is_playing()
    }

    pub fn is_playing_forward(&self) -> bool {
        self.playback_controls.is_playing_forward()
    }

    pub fn is_playing_looped(&self) -> bool {
        self.buffer_control.is_playing_looped()
    }

    pub fn is_past_discarded(&self) -> bool {
        self.buffer_control.is_past_discarded()
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
            }
            UserInput::PlayBackward => {
                self.playback_controls.backward();
                self.playback_controls.play();
            }
            UserInput::Pause => {
                self.playback_controls.pause();
            }
            UserInput::StepForward => {
                self.playback_controls.forward();
            }
            UserInput::StepBackward => {
                self.playback_controls.backward();
            }
            UserInput::ToggleLooping => {
                self.buffer_control.toggle_looped();
            }
            UserInput::ToggleHideBoundary => self.hide_boundary = !self.hide_boundary,
            UserInput::ToggleCutX => {
                self.cut.x_active = !self.cut.x_active;
            }
            UserInput::CutXBoundChanged(bound) => self.cut.x_bound += bound,
            UserInput::FlipCutX => {
                self.cut.x_flip();
            }
            UserInput::ToggleCutY => {
                self.cut.y_active = !self.cut.y_active;
            }
            UserInput::CutYBoundChanged(bound) => {
                self.cut.y_bound += bound;
            }
            UserInput::FlipCutY => {
                self.cut.y_flip();
            }
            UserInput::DiscardPastToggle => {
                self.buffer_control.toggle_discard_past();
            }
            _ => (),
        }
    }

    pub fn new_simulation(&mut self, info: SimulationParameters) {
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
        let hide_boundary = row![
            Toggler::new(self.hide_boundary)
                .label("Hide boundary")
                .on_toggle(|_| UserInput::ToggleHideBoundary),
        ];

        let reset_cam = row![
            button("Reset Camera")
                .on_press(UserInput::RequestCameraReset)
                .height(28),
        ];

        let reset = row![button("Reset").on_press(UserInput::RequestReset).height(28),];

        let save_state = row![
            button("Save current state")
                .on_press(UserInput::RequestSaving)
                .height(28),
        ];

        let screenshot = row![
            button("Take screenshot")
                .on_press(UserInput::RequestScreenshot)
                .height(28),
        ];

        // build ui controls
        let ui_controls = column![
            self.playback_controls.view(self.is_past_discarded()),
            self.buffer_control.view(),
            hide_boundary,
            self.cut.view(),
            reset_cam,
            reset,
            save_state,
            screenshot,
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
