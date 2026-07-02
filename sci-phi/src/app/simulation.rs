use cosmic::iced::Alignment;
use cosmic::iced::core::Element;
use cosmic::widget;
use cosmic::{Renderer, Theme, theme};
use rendering_libcosmic::cut::Cut;

use crate::app::Message;

#[derive(Debug, Clone)]
pub struct SimulationSettings {
    /// Cut plane state
    pub cut: Cut,
    // Text input state
    pub cut_x_input: String,
    pub cut_y_input: String,
    pub cut_z_input: String,
    /// Boundary hidden state
    pub boundary_hidden: bool,
    pub particle_radius: f32,
    pub discard_past: bool,
    pub wait_for_timesteps: bool,
    pub play_looped: bool,
    pub invert_time: bool,
}

impl Default for SimulationSettings {
    fn default() -> Self {
        Self {
            cut: Cut::default(),
            cut_x_input: "0.0".to_string(),
            cut_y_input: "0.0".to_string(),
            cut_z_input: "0.0".to_string(),
            boundary_hidden: false,
            particle_radius: 1.0,
            discard_past: true,
            wait_for_timesteps: true,
            play_looped: false,
            invert_time: false,
        }
    }
}

impl SimulationSettings {
    pub fn set_radius(&mut self, radius: f32) {
        self.particle_radius = radius;
    }
}

impl<'a> Into<Element<'a, Message, Theme, Renderer>> for &'a SimulationSettings {
    fn into(self) -> Element<'a, Message, Theme, Renderer> {
        let spacing = theme::active().cosmic().spacing;

        // ─── Boundary ─────────────────────────────────────────
        let boundary_section = widget::settings::section().title("Boundary").add(
            widget::settings::item::builder("Hide boundary")
                .toggler(self.boundary_hidden, |_| Message::ToggleHideBoundary),
        );

        // ─── Cut Controls ─────────────────────────────────────
        let cut_section = widget::settings::section()
            .title("Cut Planes")
            .add(cut_row(
                "x",
                self.cut.x_active,
                self.cut.x_bound,
                self.cut.x_inverse,
                &self.cut_x_input,
                Message::ToggleCutX,
                Message::FlipCutX,
                Message::CutXBoundChanged,
                Message::CutXBoundInput,
            ))
            .add(cut_row(
                "y",
                self.cut.y_active,
                self.cut.y_bound,
                self.cut.y_inverse,
                &self.cut_y_input,
                Message::ToggleCutY,
                Message::FlipCutY,
                Message::CutYBoundChanged,
                Message::CutXBoundInput,
            ))
            .add(cut_row(
                "z",
                self.cut.z_active,
                self.cut.z_bound,
                self.cut.z_inverse,
                &self.cut_z_input,
                Message::ToggleCutZ,
                Message::FlipCutZ,
                Message::CutZBoundChanged,
                Message::CutXBoundInput,
            ));

        // ─── Playback Settings ────────────────────────────────
        let mut playback_section = widget::settings::section().title("Playback").add(
            widget::settings::item::builder("Discard past automatically")
                .toggler(self.discard_past, |_| Message::ToggleDiscardPast),
        );

        if !self.discard_past {
            playback_section = playback_section.add(
                widget::settings::item::builder("Play reversed")
                    .toggler(self.invert_time, |_| Message::ToggleInvertTime),
            );
            playback_section = playback_section.add(
                widget::settings::item::builder("Loop")
                    .toggler(self.play_looped, |_| Message::ToggleLoop),
            );
            playback_section = playback_section.add(
                widget::settings::item::builder("Discard buffered past")
                    .control(widget::button::standard("Discard now").on_press(Message::DiscardNow)),
            );
        }

        // ─── Assemble ─────────────────────────────────────────
        widget::column::with_capacity(4)
            .spacing(spacing.space_m)
            .push(boundary_section)
            .push(cut_section)
            .push(playback_section)
            .into()
    }
}

/// Build a single cut axis row as a settings item
fn cut_row<'a>(
    axis: &'a str,
    active: bool,
    bound: f32,
    inverse: bool,
    input_value: &'a str,
    toggle_msg: Message,
    flip_msg: Message,
    bound_msg: impl Fn(f32) -> Message + 'a,
    input_msg: impl Fn(String) -> Message + 'a,
) -> Element<'a, Message, Theme, Renderer> {
    let spacing = theme::active().cosmic().spacing;
    let condition = if inverse { "≤" } else { "≥" };

    let label = widget::text::body(format!("{axis} {condition}"));
    let input = widget::text_input("0.0", input_value)
        .on_input(input_msg)
        .width(cosmic::iced::Length::Fixed(spacing.space_xxxl as f32))
        .padding(4);

    let flip_btn = widget::button::icon(
        widget::icon::from_name("object-flip-horizontal-symbolic")
            .size(spacing.space_m)
            .symbolic(true),
    )
    .icon_size(spacing.space_m)
    .padding(spacing.space_xxxs)
    .on_press(flip_msg);

    let plus_btn = widget::button::icon(
        widget::icon::from_name("list-add-symbolic")
            .size(spacing.space_m)
            .symbolic(true),
    )
    .icon_size(spacing.space_m)
    .padding(spacing.space_xxxs)
    .on_press(bound_msg(1.0));

    let minus_btn = widget::button::icon(
        widget::icon::from_name("list-remove-symbolic")
            .size(spacing.space_m)
            .symbolic(true),
    )
    .icon_size(spacing.space_m)
    .padding(spacing.space_xxxs)
    .on_press(bound_msg(-1.0));

    let toggle = widget::toggler(active).on_toggle(move |_| toggle_msg.clone());

    widget::row::with_children(vec![
        label.into(),
        input.into(),
        widget::space::horizontal().into(),
        flip_btn.into(),
        plus_btn.into(),
        minus_btn.into(),
        widget::space::horizontal().into(),
        // widget::space::horizontal().width(spacing.space_xs).into(),
        toggle.into(),
    ])
    .align_y(Alignment::Center)
    .spacing(spacing.space_xxxs)
    .into()
}
