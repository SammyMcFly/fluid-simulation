//! Inspector
use crate::app::Message;
use crate::fl;

use cosmic::iced::Alignment;
use cosmic::iced::core::Element;
use cosmic::prelude::*;
use cosmic::widget::{self, icon};
use cosmic::{cosmic_theme, theme};
use simulation_lib::measurement::RecordingStatus;
use simulation_lib::render_info::TimeStepInfo;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum InspectorTab {
    #[default]
    Info,
    Logs,
}

/// Data displayed in the Info tab
#[derive(Debug, Clone, Default)]
pub struct InfoData {
    pub time: f32,
    pub time_increment: f32,
    pub density_error: f32,
    pub queue_length: usize,
    // pub particle_count: usize,
    /// Flag whether
    pub is_measurement_saved: bool,
    /// Flag whether simulation is saved to binary
    pub is_recorded: bool,
    /// Screenshotting
    pub is_rendered_to_file: bool,
    /// Measurement/Recording status
    pub recording_status: RecordingStatus,
    /// Rendering status
    pub rendering_status: RecordingStatus,
}

impl InfoData {
    fn new(is_measurement_saved: bool, is_recorded: bool, is_rendered_to_file: bool) -> Self {
        Self {
            is_measurement_saved,
            is_recorded,
            is_rendered_to_file,
            recording_status: if is_measurement_saved || is_recorded {
                RecordingStatus::NotStarted
            } else {
                RecordingStatus::None
            },
            rendering_status: if is_rendered_to_file {
                RecordingStatus::NotStarted
            } else {
                RecordingStatus::None
            },
            ..Default::default()
        }
    }
}

pub struct Inspector {
    show: bool,
    tabs: widget::segmented_button::Model<widget::segmented_button::SingleSelect>,
    pub info: InfoData,
}

impl Inspector {
    pub fn new(
        show: bool,
        is_measurement_saved: bool,
        is_recorded: bool,
        is_rendered_to_file: bool,
    ) -> Self {
        Self {
            show,
            tabs: widget::segmented_button::Model::builder()
                .insert(|b| b.text(fl!("ts_info")).data(InspectorTab::Info).activate())
                .insert(|b| b.text(fl!("logs")).data(InspectorTab::Logs))
                .build(),
            info: InfoData::new(is_measurement_saved, is_recorded, is_rendered_to_file),
        }
    }

    pub fn update_info(&mut self, info: Option<&TimeStepInfo>, queue_length: usize) {
        self.info.queue_length = queue_length;
        if let Some(info) = info {
            self.info.time = info.measurement.time as f32;
            self.info.time_increment = info.measurement.time_step_size as f32;
            self.info.density_error = info.measurement.density_error as f32;
        } else {
            self.info.density_error = f32::default();
        }
    }

    pub fn toggle_show(&mut self) {
        self.show = !self.show;
    }

    pub fn activate(&mut self, entity: widget::segmented_button::Entity) {
        self.tabs.activate(entity);
    }

    pub fn view(&self) -> Option<Element<'_, super::Message, Theme, Renderer>> {
        if !self.show {
            return None;
        }

        let spacing = theme::active().cosmic().spacing;

        let active_tab = self.tabs.active_data::<InspectorTab>().unwrap();
        let tabs = widget::segmented_button::horizontal(&self.tabs)
            .padding(spacing.space_none)
            .button_alignment(cosmic::iced::Alignment::Center)
            .on_activate(Message::InspectorTabSelected);

        let active_tab: cosmic::iced::Element<'_, Message, Theme, Renderer> = match active_tab {
            InspectorTab::Info => self.info_view(),
            InspectorTab::Logs => widget::text::body("No logs yet.").into(),
        };

        let mut content = widget::column::with_children(vec![]);

        content = content.push(
            widget::row::with_children([
                tabs.into(),
                widget::column::with_children(vec![
                    widget::tooltip(
                        widget::button::icon(icon::from_name("window-close-symbolic"))
                            .on_press(Message::ToggleInspector)
                            .padding(8),
                        widget::text::body(fl!("close")),
                        widget::tooltip::Position::Top,
                    )
                    .into(),
                ])
                .align_x(Alignment::End)
                .into(),
            ])
            .padding([spacing.space_xxs, 0])
            .spacing(spacing.space_xxs)
            .align_y(Alignment::Center),
        );

        content = content.push(widget::space::vertical().height(spacing.space_none));

        content = content.push(widget::row::with_children([widget::column::with_capacity(
            1,
        )
        .push(active_tab)
        .spacing(spacing.space_xxs)
        .into()]));

        let container = widget::layer_container(content)
            .padding([spacing.space_xxs, spacing.space_xs])
            .layer(cosmic_theme::Layer::Primary);

        Some(container.into())
    }

    //     fn info_view(&self) -> Element<'_, Message, cosmic::Theme, cosmic::Renderer> {
    //         let spacing = theme::active().cosmic().spacing;

    //         let info_row = |label: &str, value: String| -> Element<'_, Message, _, _> {
    //             widget::row::with_children(vec![
    //                 widget::text::body(label.to_string())
    //                     .width(cosmic::iced::Length::Fixed(160.0))
    //                     .into(),
    //                 widget::text::body(value).into(),
    //             ])
    //             .spacing(spacing.space_xs)
    //             .align_y(Alignment::Center)
    //             .into()
    //         };

    //         widget::column::with_children(vec![
    //             info_row("Time:", format!("{:.4}", self.info.time)),
    //             info_row(
    //                 "Time increment:",
    //                 format!("{:.6}", self.info.time_increment),
    //             ),
    //             info_row(
    //                 "Density error (%):",
    //                 format!("{:.4}", self.info.density_error),
    //             ),
    //             info_row("Buffer remaining:", format!("{}", self.info.queue_length)),
    //         ])
    //         .spacing(spacing.space_xxs)
    //         .into()
    //     }
    fn info_view(&self) -> Element<'_, Message, cosmic::Theme, cosmic::Renderer> {
        let spacing = theme::active().cosmic().spacing;

        let info_row = |label: &str, value: String| -> Element<'_, Message, _, _> {
            widget::row::with_children(vec![
                widget::text::body(label.to_string())
                    .width(cosmic::iced::Length::Fixed(160.0))
                    .into(),
                widget::text::body(value).into(),
            ])
            .spacing(spacing.space_xs)
            .align_y(Alignment::Center)
            .into()
        };

        // Left column: existing info
        let left_column = widget::column::with_children(vec![
            info_row("Time:", format!("{:.4}", self.info.time)),
            info_row(
                "Time increment:",
                format!("{:.6}", self.info.time_increment),
            ),
            info_row(
                "Density error (%):",
                format!("{:.4}", self.info.density_error),
            ),
            info_row("Buffer remaining:", format!("{}", self.info.queue_length)),
        ])
        .spacing(spacing.space_xxs)
        .width(cosmic::iced::Length::Fill);

        // Right column: status indicators (only if active)
        let mut right_items: Vec<Element<'_, Message, _, _>> = Vec::new();

        if self.info.is_measurement_saved {
            right_items.push(info_row(
                "Measurement:",
                format!("{:?}", self.info.recording_status),
            ));
        }

        if self.info.is_recorded {
            right_items.push(info_row(
                "Recording:",
                format!("{:?}", self.info.recording_status),
            ));
        }

        if self.info.is_rendered_to_file {
            right_items.push(info_row(
                "Rendering:",
                format!("{:?}", self.info.rendering_status),
            ));
        }

        let right_column = widget::column::with_children(right_items)
            .spacing(spacing.space_xxs)
            .width(cosmic::iced::Length::Fill);

        widget::row::with_children(vec![left_column.into(), right_column.into()])
            .spacing(spacing.space_m)
            .into()
    }
}
