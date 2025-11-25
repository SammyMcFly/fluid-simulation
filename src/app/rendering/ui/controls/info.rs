//! Info
//!
use iced_widget::{column, row, text};
use iced_winit::core::{Color, Theme};



#[derive(Debug, Clone)]
pub struct UIInfo {
    simulation_info: Option<crate::app::backend::SimulationParameters>,

    pub queue_length: usize,
    pub average_density_ratio: f32,
}

impl UIInfo {
    pub fn new() -> Self {
        Self {
            simulation_info: None,
            queue_length: usize::default(),
            average_density_ratio: f32::default(),
        }
    }

    pub fn update_simulation_info(&mut self, info: crate::app::backend::SimulationParameters) {
        self.simulation_info = Some(info);
    }

    pub fn update_time_step_info(&mut self, queue_len: usize, info: Option<&crate::app::backend::TimeStepInfo>,) {
        self.queue_length = queue_len;
        if let Some(info) = info {
            self.average_density_ratio = info.average_density/self.simulation_info.as_ref().unwrap().rest_density;
        } else {
            self.average_density_ratio = f32::default();
        }
    }

    pub fn view(
        &self,
    ) -> iced_widget::Column<'_, super::UserInput, Theme, iced_wgpu::Renderer> {

        column![
            row![
                text("Average density ratio: ").color(Color::BLACK),
                text!("{}", self.average_density_ratio).color(Color::BLACK), // .size(16)
            ],
            row![
                text("Buffer length: ").color(Color::BLACK),
                text!("{}", self.queue_length).color(Color::BLACK),
            ],
        ]
        .spacing(10)
    }
}