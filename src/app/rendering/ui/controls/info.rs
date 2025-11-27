//! Info
//!
use iced_widget::{column, row, text};
use iced_winit::core::{Color, Theme};
use crate::app::backend::measure::MeasurementStatus;



impl std::fmt::Display for MeasurementStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MeasurementStatus::None => write!(f, ""),
            MeasurementStatus::NotStarted => write!(f, "Measurement: not started"),
            MeasurementStatus::Measuring => write!(f, "Measurement: in progress"),
            MeasurementStatus::Finished => write!(f, "Measurement: finished"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct UIInfo {
    simulation_info: Option<crate::app::backend::SimulationParameters>,

    pub queue_length: usize,
    pub time: f32,
    pub time_increment: f32,
    pub density_error: f32,
    pub measurement_status: MeasurementStatus,
}

impl UIInfo {
    pub fn new() -> Self {
        Self {
            simulation_info: None,
            queue_length: usize::default(),
            time: f32::default(),
            time_increment: f32::default(),
            density_error: f32::default(),
            measurement_status: MeasurementStatus::default(),
        }
    }

    pub fn update_simulation_info(&mut self, info: crate::app::backend::SimulationParameters) {
        if info.is_measured {
            self.measurement_status = MeasurementStatus::NotStarted;
        }
        self.simulation_info = Some(info);
    }

    pub fn update_time_step_info(&mut self, queue_len: usize, info: Option<&crate::app::backend::TimeStepInfo>,) {
        self.queue_length = queue_len;
        if let Some(info) = info {
            self.time = info.time;
            self.time_increment = info.time_increment;
            self.density_error = 100.*(info.average_density/self.simulation_info.as_ref().unwrap().rest_density-1.);
        } else {
            self.density_error = f32::default();
        }
    }

    pub fn advance_to_next_measurement_state(&mut self) {
        self.measurement_status.advance_to_next_state();
    }

    pub fn view(
        &self,
    ) -> iced_widget::Column<'_, super::UserInput, Theme, iced_wgpu::Renderer> {

        column![
            row![
                text("Buffer length: ").color(Color::BLACK),
                text!("{}", self.queue_length).color(Color::BLACK),
            ],
            row![
                text("Time: ").color(Color::BLACK),
                text!("{}", self.time).color(Color::BLACK),
            ],
            row![
                text("Time increment: ").color(Color::BLACK),
                text!("{}", self.time_increment).color(Color::BLACK),
            ],
            row![
                text("Density error (%): ").color(Color::BLACK),
                text!("{}", self.density_error).color(Color::BLACK), // .size(16)
            ],
            row![
                text!("{}", self.measurement_status).color(Color::BLACK),
            ],
        ]
        .spacing(10)
    }
}