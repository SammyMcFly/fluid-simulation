//! Info
//!
use iced_widget::{column, row, text};
use iced_winit::core::{Color, Theme};

use simulation_lib::{SimulationParameters, TimeStepInfo, measurement::RecordingStatus};


#[derive(Debug, Clone, Copy)]
enum MRR {
    Measurement,
    Recording,
    Rendering,
}

impl std::fmt::Display for MRR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let what = match self {
            MRR::Measurement => "Measurement".to_string(),
            MRR::Recording => "Recording".to_string(),
            MRR::Rendering => "Rendering".to_string(),
        };
        write!(f, "{}", what)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MRRStatus {
    description: MRR,
    is_rec: bool,
    pub recording_status: RecordingStatus,
}

impl MRRStatus {
    fn new(description: MRR, is_rec: bool,) -> Self {
        let recording_status = if is_rec {
            RecordingStatus::NotStarted
        } else {
            RecordingStatus::None
        };
        Self { description, is_rec, recording_status, }
    }
    fn advance_to_next_state(&mut self) {
        self.recording_status.advance_to_next_state();
    }
}

impl std::fmt::Display for MRRStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_rec {
            self.description.fmt(f)?;
        }
        let how = match self.recording_status {
            RecordingStatus::None => "".to_string(),
            RecordingStatus::NotStarted => "not started".to_string(),
            RecordingStatus::InProgress => "in progress".to_string(),
            RecordingStatus::Finished => "finished".to_string(),
        };
        write!(f, ": {}", how)
    }
}

#[derive(Debug, Clone)]
pub struct UIInfo {
    simulation_info: Option<SimulationParameters>,

    queue_length: usize,
    pub time: f32,
    time_increment: f32,
    density_error: f32,
    measurement_status: MRRStatus,
    recording_status: MRRStatus,
    pub rendering_status: MRRStatus,
}

impl UIInfo {
    pub fn new(is_rendered: bool) -> Self {
        Self {
            simulation_info: None,
            queue_length: usize::default(),
            time: f32::default(),
            time_increment: f32::default(),
            density_error: f32::default(),
            measurement_status: MRRStatus::new(MRR::Measurement, false),
            recording_status: MRRStatus::new(MRR::Recording, false),
            rendering_status: MRRStatus::new(MRR::Rendering, is_rendered),
        }
    }

    pub fn update_simulation_info(&mut self, info: SimulationParameters) {
        self.measurement_status = MRRStatus::new(MRR::Measurement, info.is_measured);
        self.recording_status = MRRStatus::new(MRR::Recording, info.is_recorded);
        self.simulation_info = Some(info);
    }

    pub fn update_time_step_info(&mut self, info: Option<&TimeStepInfo>, queue_len: usize) {
        self.queue_length = queue_len;
        if let Some(info) = info {
            self.time = info.time;
            self.time_increment = info.time_increment;
            self.density_error = 100.*(info.average_density/self.simulation_info.as_ref().unwrap().rest_density - 1.);
        } else {
            self.density_error = f32::default();
        }
    }

    pub fn advance_to_next_measurement_state(&mut self) {
        if self.measurement_status.is_rec {
            self.measurement_status.advance_to_next_state();
        }
    }

    pub fn advance_to_next_recording_state(&mut self) {
        if self.recording_status.is_rec {
            self.recording_status.advance_to_next_state();
        }
    }

    pub fn view(
        &self,
    ) -> iced_widget::Column<'_, super::UserInput, Theme, iced_wgpu::Renderer> {
        let view = column![
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
        ]
        .spacing(10);

        let view = if self.measurement_status.is_rec {
            view.push(row![
                text!("{}", self.measurement_status).color(Color::BLACK),
            ],)
        } else {
            view
        };

        let view = if self.recording_status.is_rec {
            view.push(row![
                text!("{}", self.recording_status).color(Color::BLACK),
            ],)
        } else {
            view
        };

        if self.rendering_status.is_rec {
            view.push(row![
                text!("{}", self.rendering_status).color(Color::BLACK),
            ],)
        } else {
            view
        }
    }
}