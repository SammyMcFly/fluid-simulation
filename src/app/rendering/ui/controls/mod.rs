//! Render Controls, Settings and Utilities
//!
//!
use iced_widget::{container, column, row, text, button, Toggler}; //text_input, slider
use iced_winit::core::{Element, Theme, Length, Color};

use crate::app::backend::sph::particle::Positional;
use super::UserInput;

pub mod info;

#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub enum ParticleColor {
    #[default]
    VelocityGraded,
    FixedColor([f32;3]),
}

#[derive(Debug, Clone, PartialEq)]
pub struct  Cut {
    pub x: bool,
    pub x_bound: f32,
    pub x_inverse: bool,
    x_inv: f32,
    pub y: bool,
    pub y_bound: f32,
    pub y_inverse: bool,
    y_inv: f32,
    // pub z: bool,
    // pub z_bound: f32,
}

impl Default for Cut {
    fn default() -> Self {
        Self {
            x: false,
            x_bound: 0.,
            x_inverse: false,
            x_inv: 1.,
            y: false,
            y_bound: 0.,
            y_inverse: false,
            y_inv: 1.,
        }
    }
}

impl Cut {
    pub fn cut(&self, particle: &impl Positional) -> bool {
        if self.x && self.y {
            self.x_inv*(particle.pos_now()[0] as f32 -self.x_bound) >= 0.
            && self.y_inv*(-particle.pos_now()[1] as f32 -self.y_bound) >= 0.
        } else if self.x {
            self.x_inv*(particle.pos_now()[0] as f32 -self.x_bound) >= 0.
        } else if self.y {
            self.y_inv*(-particle.pos_now()[1] as f32 -self.y_bound) >= 0.
        } else {
            true
        }
    }
    pub fn x_flip(&mut self) {
        self.x_inverse = !self.x_inverse;
        self.x_inv *= -1.;
    }
    pub fn y_flip(&mut self) {
        self.y_inverse = !self.y_inverse;
        self.y_inv *= -1.;
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub enum PlaybackState {
    Resumed,
    #[default]
    Paused,
}

impl PlaybackState {
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

#[derive(Debug, Clone)]
pub struct RenderControls {
    pub play_pause: PlaybackState,
    pub particle_color: ParticleColor,
    pub boundary_particle_color: ParticleColor,
    pub background_color: iced_winit::core::Color,
    pub cut: Cut,
    pub hide_boundary: bool,

    pub info: info::UIInfo,
}

// pub struct UIControls {
//     // pub display_state: PlaybackState,
//     // reset_requested: bool,
//     // saving_requested: bool,
//     // background_color: Color,
//     // hide_boundary: bool,
//     // cut: Cut,
//     // // input: String,
//     // buffer_length: u32,
//     // rest_density: f32,
//     // average_density: f32,

//     play_pause: bool,
//     reset: bool,
//     save_state: bool,

// }

impl RenderControls {
    pub fn new(
        particle_color: ParticleColor,
        boundary_particle_color: ParticleColor,
    ) -> Self {
        Self {
            play_pause: PlaybackState::Paused,
            // reset: false,
            // save_state: false,
            particle_color,
            boundary_particle_color,
            background_color: Color::WHITE,
            hide_boundary: false,
            cut: Cut::default(),
            // input: String::default(),
            // buffer_length: u32::default(),
            // rest_density: f32::default(),
            // average_density: f32::default(),
            info: info::UIInfo::new(),
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

    // pub fn is_reset_requested(&mut self) -> bool {
    //     let output = self.reset;
    //     self.reset = false;
    //     output
    // }

    // pub fn is_saving_requested(&mut self) -> bool {
    //     let output = self.save_state;
    //     self.save_state = false;
    //     output
    // }

    pub fn update(&mut self, input: &UserInput) {
        match input {
            UserInput::ToggleDisplayState => {
                self.play_pause.toggle();
            }
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
            _ => (),
        }
    }

    pub fn new_simulation(&mut self, info: crate::app::backend::SimulationInfo) {
        self.particle_color = info.particle_color;
        self.boundary_particle_color = info.boundary_particle_color;
        self.info.update_simulation_info(info);
    }

    pub fn update_time_step_info(&mut self, queue_len: usize, info: &Option<crate::app::backend::TimeStepInfo>) {
        self.info.update_time_step_info(queue_len, info);
    }

    pub fn view(&self) -> Element<'_, UserInput, Theme, iced_wgpu::Renderer> {
        let playback_state = row![
            button(if self.play_pause.is_playing() { "Pause" } else { "Resume" })
            .on_press(UserInput::ToggleDisplayState).height(28),
        ];

        let step = row![
            button("Step")
            .on_press(UserInput::StepInTime).height(28),
        ];

        let playback_state = if self.play_pause == PlaybackState::Paused {
            row![
                playback_state,
                step,
            ].spacing(10)
        } else {
            row![
                playback_state,
            ]
        };

        let reset_cam = row![
            button("Reset Camera")
            .on_press(UserInput::RequestCameraReset).height(28),
        ];

        let reset = row![
            button("Reset")
            .on_press(UserInput::RequestReset).height(28),
        ];

        let save = row![
            button("Save current state")
            .on_press(UserInput::RequestSaving).height(28),
        ];

        // let background_color = self.background_color;

        // let sliders = row![
        //     slider(0.0..=1.0, background_color.r, move |r| {
        //         Message::BackgroundColorChanged(Color {
        //             r,
        //             ..background_color
        //         })
        //     })
        //     .step(0.01),
        //     slider(0.0..=1.0, background_color.g, move |g| {
        //         Message::BackgroundColorChanged(Color {
        //             g,
        //             ..background_color
        //         })
        //     })
        //     .step(0.01),
        //     slider(0.0..=1.0, background_color.b, move |b| {
        //         Message::BackgroundColorChanged(Color {
        //             b,
        //             ..background_color
        //         })
        //     })
        //     .step(0.01),
        // ]
        // .width(500)
        // .spacing(20);

        let hide_boundary = row![
            Toggler::new(self.hide_boundary)
                .label("Hide boundary")
                .on_toggle(|_| UserInput::ToggleHideBoundary),
        ];
        let x_condition = if self.cut.x_inverse {
            "<=".to_string()
        } else {
            ">=".to_string()
        };
        let cut_x = row![
            Toggler::new(self.cut.x)
                .label("Show half-space for:")
                .on_toggle(|_| UserInput::ToggleCutX),
            text(format!(" x {x_condition} ")),
            text(self.cut.x_bound),
            text(" "),
            button("I").on_press(UserInput::FlipCutX).width(28).height(28),
            button("+").on_press(UserInput::CutXBoundChanged(1.)).width(28).height(28),
            button("-").on_press(UserInput::CutXBoundChanged(-1.)).width(28).height(28),
        ]
        .width(500);
        let y_condition = if self.cut.y_inverse {
            "<=".to_string()
        } else {
            ">=".to_string()
        };
        let cut_y = row![
            Toggler::new(self.cut.y)
                .label("Show half-space for:")
                .on_toggle(|_| UserInput::ToggleCutY),
            // slider(0.0..=10.0, self.cut.y_bound, move |bound| {
            //     Message::CutYBoundChanged(bound)
            // })
            // .step(0.1),
            text(format!(" y {y_condition} ")),
            text(self.cut.y_bound),
            text(" "),
            button("I").on_press(UserInput::FlipCutY).width(28).height(28),
            button("+").on_press(UserInput::CutYBoundChanged(1.)).width(28).height(28),
            button("-").on_press(UserInput::CutYBoundChanged(-1.)).width(28).height(28),
        ]
        .width(500);

        // Container::new(column![
        //     text("Background color").color(Color::WHITE),
        //     text!("{background_color:?}").size(14).color(Color::WHITE),
        //     sliders,
        //     text_input("Type something...", &self.input)
        //         .on_input(Message::InputChanged),
        // ]
        // .spacing(10))
        // .align_bottom(10)
        // .into()
        let ui_controls = column![
            playback_state,
            reset_cam,
            reset,
            save,
            // text("Background color").color(Color::BLACK),
            // text!("{background_color:?}").size(14).color(Color::BLACK),
            // sliders,
            hide_boundary,
            cut_x,
            cut_y,
            // text_input("Type something...", &self.input)
            //     .on_input(Message::InputChanged),
        ]
        .spacing(10);

        container(
            column![
                self.info.view(),
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
