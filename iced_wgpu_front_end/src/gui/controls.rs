use iced_wgpu::{Renderer};
use iced_widget::{bottom, column, row, slider, text, button, Toggler}; //text_input
use iced_winit::core::{Color, Element, Theme};
// use serde::de;

use crate::mediation;

#[derive(Debug, Clone)]
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
    pub fn cut(&self, particle: &mediation::Instance) -> bool {
        if self.x && self.y {
            self.x_inv*(particle.position[0]-self.x_bound) >= 0.
            && self.y_inv*(-particle.position[2]-self.y_bound) >= 0.
        } else if self.x {
            self.x_inv*(particle.position[0]-self.x_bound) >= 0.
        } else if self.y {
            self.y_inv*(-particle.position[2]-self.y_bound) >= 0.
        } else {
            true
        }
    }
    fn x_flip(&mut self) {
        self.x_inverse = !self.x_inverse;
        self.x_inv *= -1.;
    }
    fn y_flip(&mut self) {
        self.y_inverse = !self.y_inverse;
        self.y_inv *= -1.;
    }
}

#[derive(Debug)]
pub enum DisplayState {
    Resumed,
    Paused,
}

impl DisplayState {
    pub fn toggle(&mut self) {
        match *self {
            Self::Paused => *self = Self::Resumed,
            Self::Resumed => *self = Self::Paused,
        }
    }

    pub fn is_playing(&self) -> bool {
        match self {
            Self::Resumed => {
                true
            },
            Self::Paused => {
                false
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum Message {
    DisplayStateToggle,
    RequestReset,
    RequestSaving,
    BackgroundColorChanged(Color),
    HideBoundaryToggle,
    CutXToggle,
    CutXBoundChanged(f32),
    CutXFlip,
    CutYToggle,
    CutYBoundChanged(f32),
    CutYFlip,
    // InputChanged(String),
    BufferLengthChanged(u32),
    RestDensityChanged(f32),
    AverageDensityChanged(f32),
}

pub struct UIControls {
    pub display_state: DisplayState,
    reset_requested: bool,
    saving_requested: bool,
    background_color: Color,
    hide_boundary: bool,
    cut: Cut,
    // input: String,
    buffer_length: u32,
    rest_density: f32,
    average_density: f32,
}

impl UIControls {
    pub fn new() -> UIControls {
        UIControls {
            display_state: DisplayState::Paused,
            reset_requested: false,
            saving_requested: false,
            background_color: Color::WHITE,
            hide_boundary: false,
            cut: Cut::default(),
            // input: String::default(),
            buffer_length: u32::default(),
            rest_density: f32::default(),
            average_density: f32::default(),
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
    pub fn is_reset_requested(&mut self) -> bool {
        let output = self.reset_requested;
        self.reset_requested = false;
        output
    }
    pub fn is_saving_requested(&mut self) -> bool {
        let output = self.saving_requested;
        self.saving_requested = false;
        output
    }
}

impl UIControls {
    pub fn update(&mut self, message: Message) {
        match message {
            Message::DisplayStateToggle => {
                self.display_state.toggle();
            }
            Message::RequestReset => {
                self.reset_requested = true;
            }
            Message::RequestSaving => {
                self.saving_requested = true;
            }
            Message::BackgroundColorChanged(color) => {
                self.background_color = color;
            }
            Message::HideBoundaryToggle => {
                self.hide_boundary = !self.hide_boundary
            }
            Message::CutXToggle => {
                self.cut.x = !self.cut.x;
            }
            Message::CutXBoundChanged(bound) => {
                self.cut.x_bound += bound
            }
            Message::CutXFlip => {
                self.cut.x_flip();
            }
            Message::CutYToggle => {
                self.cut.y = !self.cut.y;
            }
            Message::CutYBoundChanged(bound) => {
                self.cut.y_bound += bound;
            }
            Message::CutYFlip => {
                self.cut.y_flip();
            }
            // Message::InputChanged(input) => {
            //     self.input = input;
            // }
            Message::BufferLengthChanged(length) => {
                self.buffer_length = length;
            }
            Message::RestDensityChanged(density) => {
                self.rest_density = density;
            }
            Message::AverageDensityChanged(density) => {
                self.average_density = density;
            }
        }
    }

    pub fn view(&self) -> Element<Message, Theme, Renderer> {
        let display_state = row![
            button(if self.display_state.is_playing() { "Pause" } else { "Resume" })
            .on_press(Message::DisplayStateToggle).height(28),
        ];

        let reset = row![
            button("Reset")
            .on_press(Message::RequestReset).height(28),
        ];

        let save = row![
            button("Save current state")
            .on_press(Message::RequestSaving).height(28),
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
                .on_toggle(|_| Message::HideBoundaryToggle),
        ];
        let x_condition = if self.cut.x_inverse {
            "<=".to_string()
        } else {
            ">=".to_string()
        };
        let cut_x = row![
            Toggler::new(self.cut.x)
                .label("Show half-space for:")
                .on_toggle(|_| Message::CutXToggle),
            text(format!(" x {x_condition} ")),
            text(self.cut.x_bound),
            text(" "),
            button("I").on_press(Message::CutXFlip).width(28).height(28),
            button("+").on_press(Message::CutXBoundChanged(1.)).width(28).height(28),
            button("-").on_press(Message::CutXBoundChanged(-1.)).width(28).height(28),
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
                .on_toggle(|_| Message::CutYToggle),
            // slider(0.0..=10.0, self.cut.y_bound, move |bound| {
            //     Message::CutYBoundChanged(bound)
            // })
            // .step(0.1),
            text(format!(" y {y_condition} ")),
            text(self.cut.y_bound),
            text(" "),
            button("I").on_press(Message::CutYFlip).width(28).height(28),
            button("+").on_press(Message::CutYBoundChanged(1.)).width(28).height(28),
            button("-").on_press(Message::CutYBoundChanged(-1.)).width(28).height(28),
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
        bottom(
            column![
                display_state,
                reset,
                save,
                // text("Background color").color(Color::BLACK),
                // text!("{background_color:?}").size(14).color(Color::BLACK),
                // sliders,
                hide_boundary,
                cut_x,
                cut_y,
                row![
                    text("Average density ratio: ").color(Color::BLACK),
                    text!("{}", self.average_density/self.rest_density).color(Color::BLACK), // .size(16)
                ],
                row![
                    text("Buffer length: ").color(Color::BLACK),
                    text!("{}", self.buffer_length).color(Color::BLACK),
                ],
                // text_input("Type something...", &self.input)
                //     .on_input(Message::InputChanged),
            ]
            .spacing(10),
        )
        .padding(10)
        .into()
    }
}
