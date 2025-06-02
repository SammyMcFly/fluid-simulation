use iced_wgpu::Renderer;
// use iced::widget::{Container, column, row, slider, text, text_input};
use iced_widget::{bottom, column, row, slider, text, text_input};
use iced_winit::core::{Color, Element, Theme};
use serde::de;

pub struct UIControls {
    background_color: Color,
    // input: String,
    buffer_length: u32,
    rest_density: f32,
    average_density: f32,
}

#[derive(Debug, Clone)]
pub enum Message {
    BackgroundColorChanged(Color),
    // InputChanged(String),
    BufferLengthChanged(u32),
    RestDensityChanged(f32),
    AverageDensityChanged(f32),
}

impl UIControls {
    pub fn new() -> UIControls {
        UIControls {
            background_color: Color::WHITE,
            // input: String::default(),
            buffer_length: u32::default(),
            rest_density: f32::default(),
            average_density: f32::default(),
        }
    }

    pub fn background_color(&self) -> Color {
        self.background_color
    }
}

impl UIControls {
    pub fn update(&mut self, message: Message) {
        match message {
            Message::BackgroundColorChanged(color) => {
                self.background_color = color;
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
        let background_color = self.background_color;

        let sliders = row![
            slider(0.0..=1.0, background_color.r, move |r| {
                Message::BackgroundColorChanged(Color {
                    r,
                    ..background_color
                })
            })
            .step(0.01),
            slider(0.0..=1.0, background_color.g, move |g| {
                Message::BackgroundColorChanged(Color {
                    g,
                    ..background_color
                })
            })
            .step(0.01),
            slider(0.0..=1.0, background_color.b, move |b| {
                Message::BackgroundColorChanged(Color {
                    b,
                    ..background_color
                })
            })
            .step(0.01),
        ]
        .width(500)
        .spacing(20);

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
                text("Background color").color(Color::BLACK),
                text!("{background_color:?}").size(14).color(Color::BLACK),
                sliders,
                row![
                    text("Average density: ").color(Color::BLACK),
                    text!("{}", self.average_density).size(16).color(Color::BLACK),
                ],
                row![
                    text("Rest density: ").color(Color::BLACK),
                    text!("{}", self.rest_density).size(16).color(Color::BLACK),
                ],
                row![
                    text("Buffer length: ").color(Color::BLACK),
                    text!("{}", self.buffer_length).size(16).color(Color::BLACK),
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
