//! Info
//!
use iced_widget::{column, row, text};
use iced_winit::core::{Color, Theme};

use crate::app::backend::recording::RecordingStatus;
use simulation_lib::{SimulationParameters, TimeStepInfo};


#[derive(Debug, Clone, Default)]
struct MRStatus {
    is_measured: bool,
    is_recorded: bool,
    recording_status: RecordingStatus,
}

impl MRStatus {
    fn new(is_measured: bool, is_recorded: bool,) -> Self {
        let recording_status = if is_measured || is_recorded {
            RecordingStatus::NotStarted
        } else {
            RecordingStatus::None
        };
        Self { is_measured, is_recorded, recording_status, }
    }
    fn advance_to_next_state(&mut self) {
        self.recording_status.advance_to_next_state();
    }
}

impl std::fmt::Display for MRStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let what = if self.is_measured && self.is_recorded {
            "Measurement/Recording: ".to_string()
        } else if self.is_measured {
            "Measurement: ".to_string()
        } else if self.is_recorded {
            "Recording: ".to_string()
        } else {
            "".to_string()
        };
        let how = match self.recording_status {
            RecordingStatus::None => "".to_string(),
            RecordingStatus::NotStarted => "not started".to_string(),
            RecordingStatus::Measuring => "in progress".to_string(),
            RecordingStatus::Finished => "Recording: finished".to_string(),
        };
        write!(f, "{}{}", what, how)
    }
}

#[derive(Debug, Clone)]
pub struct UIInfo {
    simulation_info: Option<SimulationParameters>,

    queue_length: usize,
    time: f32,
    time_increment: f32,
    density_error: f32,
    recording_status: MRStatus,
}

impl UIInfo {
    pub fn new() -> Self {
        Self {
            simulation_info: None,
            queue_length: usize::default(),
            time: f32::default(),
            time_increment: f32::default(),
            density_error: f32::default(),
            recording_status: MRStatus::default(),
        }
    }

    pub fn update_simulation_info(&mut self, info: SimulationParameters) {
        self.recording_status = MRStatus::new(info.is_measured, info.is_recorded);
        self.simulation_info = Some(info);
    }

    pub fn update_time_step_info(&mut self, queue_len: usize, info: Option<&TimeStepInfo>,) {
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
        self.recording_status.advance_to_next_state();
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
                text!("{}", self.recording_status).color(Color::BLACK),
            ],
        ]
        .spacing(10)
    }
}