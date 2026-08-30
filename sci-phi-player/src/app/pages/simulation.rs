use cosmic::iced::core::Element;
use cosmic::iced::{Alignment, Length};
use cosmic::widget;
use cosmic::{Renderer, Theme, theme};
use rendering_lib::colormap::Colormap;
use rendering_lib::cut::{Cut, CutAxis};
use simulation_lib::render_info::{
    BoundaryMeshColoring, BoundarySampleColoring, BoundaryVisualization, FluidMeshColoring,
    FluidSampleColoring, FluidVisualization, ScalarQuantity, TimeStepInfo,
};

use crate::app::Message;
use crate::fl;

#[derive(Debug, Clone)]
pub struct SimulationSettings {
    // Fluid visualization option
    pub fluid_vis: FluidVisOption,
    pub fluid_quantity: QuantityOption,
    pub colormap: Colormap,
    pub color_mapping_max: f32,
    pub color_mapping_max_input: String,
    // Boundary visualization option
    pub boundary_hidden: bool,
    pub boundary_vis: BoundaryVisOption,
    pub boundary_alpha: f32,
    /// Sensor plane configuration
    pub sensor_plane: SensorPlaneConfig,
    /// Cut plane state
    pub cut: Cut,
    pub cut_x_input: String,
    pub cut_y_input: String,
    pub cut_z_input: String,
    pub cut_boundary: bool,
    pub particle_radius: f32,
    pub discard_past: bool,
    pub wait_for_timesteps: bool,
    pub play_looped: bool,
    pub invert_time: bool,
    // Localized label caches
    pub fluid_vis_labels: Vec<String>,
    pub quantity_labels: Vec<String>,
    pub boundary_vis_labels: Vec<String>,
}

impl Default for SimulationSettings {
    fn default() -> Self {
        Self {
            fluid_vis: FluidVisOption::NotLoaded,
            fluid_quantity: QuantityOption::Speed,
            colormap: Colormap::default(),
            color_mapping_max: 10.0,
            color_mapping_max_input: "10.0".to_string(),
            boundary_hidden: false,
            boundary_vis: BoundaryVisOption::NotLoaded,
            boundary_alpha: 1.0,
            sensor_plane: SensorPlaneConfig::default(),
            cut: Cut::default(),
            cut_x_input: "0.0".to_string(),
            cut_y_input: "0.0".to_string(),
            cut_z_input: "0.0".to_string(),
            cut_boundary: true,
            particle_radius: 1.0,
            discard_past: false,
            wait_for_timesteps: true,
            play_looped: false,
            invert_time: false,
            fluid_vis_labels: FluidVisOption::ALL.iter().map(|o| o.label()).collect(),
            quantity_labels: QuantityOption::ALL.iter().map(|o| o.label()).collect(),
            boundary_vis_labels: BoundaryVisOption::ALL.iter().map(|o| o.label()).collect(),
        }
    }
}

impl From<&TimeStepInfo> for SimulationSettings {
    fn from(info: &TimeStepInfo) -> Self {
        SimulationSettings {
            fluid_vis: FluidVisOption::from_template(&info.fluid),
            fluid_quantity: QuantityOption::from_template(&info.fluid),
            boundary_vis: BoundaryVisOption::from_template(&info.boundary),
            ..Default::default()
        }
    }
}

impl SimulationSettings {
    pub fn set_radius(&mut self, radius: f32) {
        self.particle_radius = radius;
    }

    pub fn update(&mut self, info: &TimeStepInfo) {
        self.fluid_vis = FluidVisOption::from_template(&info.fluid);
        self.fluid_quantity = QuantityOption::from_template(&info.fluid);
        self.boundary_vis = BoundaryVisOption::from_template(&info.boundary);
    }

    fn colormap_controls<'a>(&'a self) -> cosmic::Element<'a, Message> {
        let spacing = theme::active().cosmic().spacing;

        let selected = Colormap::ALL.iter().position(|c| *c == self.colormap);
        let colormap_selector = widget::settings::item::builder(fl!("settings", "colormap"))
            .control(widget::dropdown(
                &Colormap::LABELS,
                selected,
                Message::SetColormap,
            ));

        let (w, h) = (256usize, 16usize);
        let handle = widget::image::Handle::from_rgba(
            w as u32,
            h as u32,
            colorbar_bytes(self.colormap, w, h),
        );

        let colorbar = widget::image(handle)
            .width(Length::Fill)
            .height(Length::Fixed(12.0));

        let max_input =
            widget::text_input(fl!("settings", "max-label"), &self.color_mapping_max_input)
                .width(cosmic::iced::Length::Fixed(spacing.space_xxxl as f32))
                .on_input(Message::ColorMappingMaxInput)
                .on_submit(|_| Message::ApplyColorMappingMax);

        let minus_btn = widget::button::icon(
            widget::icon::from_name("list-remove-symbolic")
                .size(spacing.space_m)
                .symbolic(true),
        )
        .icon_size(spacing.space_m)
        .padding(spacing.space_xxxs)
        .on_press(Message::ColorMappingMaxStep(1.0, false));

        let plus_btn = widget::button::icon(
            widget::icon::from_name("list-add-symbolic")
                .size(spacing.space_m)
                .symbolic(true),
        )
        .icon_size(spacing.space_m)
        .padding(spacing.space_xxxs)
        .on_press(Message::ColorMappingMaxStep(1.0, true));

        let colormap_settings = widget::row::with_capacity(4)
            .push(widget::text(format!("{} ", fl!("settings", "min-label"))))
            .push(colorbar)
            .push(widget::text(fl!("settings", "max-label")))
            .push(max_input)
            .push(minus_btn)
            .push(plus_btn)
            .align_y(Alignment::Center)
            .spacing(spacing.space_xxxs);

        widget::column::with_capacity(6)
            .push(colormap_selector)
            .push(colormap_settings)
            .spacing(spacing.space_xxxs)
            .into()
    }
}

impl<'a> Into<Element<'a, Message, Theme, Renderer>> for &'a SimulationSettings {
    fn into(self) -> Element<'a, Message, Theme, Renderer> {
        let spacing = theme::active().cosmic().spacing;

        // ─── Fluid ───────────────────────────────────────────
        let fluid_selected = FluidVisOption::ALL
            .iter()
            .position(|o| *o == self.fluid_vis)
            .expect("Failed to get position of selected fluid");
        let mut fluid_section = widget::settings::section()
            .title(fl!("section", "fluid"))
            .add(
                widget::settings::item::builder(fl!("settings", "fluid")).control(widget::text(
                    self.fluid_vis_labels[fluid_selected].to_string(),
                )),
            );

        // Quantity selector.
        if self.fluid_vis.uses_quantity() {
            let quantity_selected = QuantityOption::ALL
                .iter()
                .position(|q| *q == self.fluid_quantity)
                .expect("Failed to get position of selected quantity");
            fluid_section = fluid_section
                .add(
                    widget::settings::item::builder(fl!("settings", "quantity")).control(
                        widget::text(self.quantity_labels[quantity_selected].to_string()),
                    ),
                )
                .add(self.colormap_controls());
        }

        // Bounds + dx + commit only for Sensor-Plane.
        if self.fluid_vis == FluidVisOption::SensorPlane {
            let cfg = &self.sensor_plane;
            fluid_section = fluid_section
                .add(stepper_row(fl!("settings", "min-x"), &cfg.min[0]))
                .add(stepper_row(fl!("settings", "min-y"), &cfg.min[1]))
                .add(stepper_row(fl!("settings", "min-z"), &cfg.min[2]))
                .add(stepper_row(fl!("settings", "max-x"), &cfg.max[0]))
                .add(stepper_row(fl!("settings", "max-y"), &cfg.max[1]))
                .add(stepper_row(fl!("settings", "max-z"), &cfg.max[2]))
                .add(stepper_row(fl!("settings", "dx"), &cfg.dx));
        }

        // ─── Boundary ─────────────────────────────────────────
        let boundary_selected = BoundaryVisOption::ALL
            .iter()
            .position(|o| *o == self.boundary_vis)
            .expect("Failed to get position of selected boundary");
        let boundary_section = widget::settings::section()
            .title(fl!("section", "boundary"))
            .add(
                widget::settings::item::builder(fl!("settings", "hide-boundary"))
                    .toggler(self.boundary_hidden, |_| Message::ToggleHideBoundary),
            )
            .add(
                widget::settings::item::builder(fl!("settings", "boundary")).control(widget::text(
                    self.boundary_vis_labels[boundary_selected].to_string(),
                )),
            )
            .add(
                widget::settings::item::builder(fl!("settings", "boundary-alpha")).control(
                    widget::slider(0.0..=1.0, self.boundary_alpha, |value| {
                        Message::SetBoundaryAlpha(value)
                    })
                    .step(0.01),
                ),
            );

        // ─── Cut Controls ─────────────────────────────────────
        let cut_section = widget::settings::section()
            .title(fl!("section", "cut-planes"))
            .add(cut_row(
                "x",
                self.cut.x_active,
                self.cut.x_bound,
                self.cut.x_inverse,
                &self.cut_x_input,
                Message::ToggleCut(CutAxis::X),
                Message::FlipCut(CutAxis::X),
                |delta| Message::CutBoundChanged(CutAxis::X, delta),
                |value| Message::CutBoundInput(CutAxis::X, value),
            ))
            .add(cut_row(
                "y",
                self.cut.y_active,
                self.cut.y_bound,
                self.cut.y_inverse,
                &self.cut_y_input,
                Message::ToggleCut(CutAxis::Y),
                Message::FlipCut(CutAxis::Y),
                |delta| Message::CutBoundChanged(CutAxis::Y, delta),
                |value| Message::CutBoundInput(CutAxis::Y, value),
            ))
            .add(cut_row(
                "z",
                self.cut.z_active,
                self.cut.z_bound,
                self.cut.z_inverse,
                &self.cut_z_input,
                Message::ToggleCut(CutAxis::Z),
                Message::FlipCut(CutAxis::Z),
                |delta| Message::CutBoundChanged(CutAxis::Z, delta),
                |value| Message::CutBoundInput(CutAxis::Z, value),
            ))
            .add(
                widget::settings::item::builder(fl!("settings", "cut-boundary"))
                    .toggler(self.cut_boundary, |_| Message::ToggleCutBoundary),
            );

        // ─── Playback Settings ────────────────────────────────
        let mut playback_section = widget::settings::section()
            .title(fl!("section", "playback"))
            .add(
                widget::settings::item::builder(fl!("settings", "discard-past-auto"))
                    .toggler(self.discard_past, |_| Message::ToggleDiscardPast),
            );

        if !self.discard_past {
            playback_section = playback_section
                .add(
                    widget::settings::item::builder(fl!("settings", "play-reversed"))
                        .toggler(self.invert_time, |_| Message::ToggleInvertTime),
                )
                .add(
                    widget::settings::item::builder(fl!("settings", "loop"))
                        .toggler(self.play_looped, |_| Message::ToggleLoop),
                )
                .add(
                    widget::settings::item::builder(fl!("settings", "discard-buffered-past"))
                        .control(
                            widget::button::standard(fl!("settings", "discard-now"))
                                .on_press(Message::DiscardNow),
                        ),
                );
        }

        // ─── Assemble ─────────────────────────────────────────
        widget::column::with_capacity(5)
            .spacing(spacing.space_m)
            .push(fluid_section)
            .push(boundary_section)
            .push(cut_section)
            .push(playback_section)
            .into()
    }
}

pub fn colorbar_bytes(cmap: Colormap, width: usize, height: usize) -> Vec<u8> {
    let mut buf = Vec::with_capacity(width * height * 4);
    // get one row
    let row: Vec<[u8; 4]> = (0..width)
        .map(|i| {
            let t = i as f32 / (width.max(2) - 1) as f32;
            let [r, g, b] = cmap.eval(t);
            [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, 255]
        })
        .collect();
    // repeat height times
    for _ in 0..height {
        for px in &row {
            buf.extend_from_slice(px);
        }
    }
    buf
}

/// Row with label, text input und −/+ buttons for sensor plane fields.
fn stepper_row<'a>(
    label: impl Into<std::borrow::Cow<'a, str>>,
    value: &'a str,
) -> Element<'a, Message, Theme, Renderer> {
    let spacing = theme::active().cosmic().spacing;

    let value = widget::text(value).width(cosmic::iced::Length::Fixed(spacing.space_xxxl as f32));

    widget::settings::item::builder(label)
        .control(
            widget::row::with_children(vec![value.into()])
                .align_y(Alignment::Center)
                .spacing(spacing.space_xxxs),
        )
        .into()
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

    let minus_btn = widget::button::icon(
        widget::icon::from_name("list-remove-symbolic")
            .size(spacing.space_m)
            .symbolic(true),
    )
    .icon_size(spacing.space_m)
    .padding(spacing.space_xxxs)
    .on_press(bound_msg(-1.0));

    let plus_btn = widget::button::icon(
        widget::icon::from_name("list-add-symbolic")
            .size(spacing.space_m)
            .symbolic(true),
    )
    .icon_size(spacing.space_m)
    .padding(spacing.space_xxxs)
    .on_press(bound_msg(1.0));

    let toggle = widget::toggler(active).on_toggle(move |_| toggle_msg.clone());

    widget::row::with_children(vec![
        label.into(),
        input.into(),
        // widget::space::horizontal().into(),
        minus_btn.into(),
        plus_btn.into(),
        flip_btn.into(),
        widget::space::horizontal().into(),
        // widget::space::horizontal().width(spacing.space_xs).into(),
        toggle.into(),
    ])
    .align_y(Alignment::Center)
    .spacing(spacing.space_xxxs)
    .into()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FluidVisOption {
    TriangleMeshUniform,
    TriangleMeshFluidId,
    SamplesUniform,
    SamplesFluidId,
    SamplesQuantity,
    SensorPlane,
    NotLoaded,
}

impl FluidVisOption {
    pub const ALL: [FluidVisOption; 7] = [
        Self::TriangleMeshUniform,
        Self::TriangleMeshFluidId,
        Self::SamplesUniform,
        Self::SamplesFluidId,
        Self::SamplesQuantity,
        Self::SensorPlane,
        Self::NotLoaded,
    ];

    pub fn label(self) -> String {
        match self {
            Self::TriangleMeshUniform => fl!("fluid-vis", "mesh-uniform"),
            Self::TriangleMeshFluidId => fl!("fluid-vis", "mesh-fluid-id"),
            Self::SamplesUniform => fl!("fluid-vis", "samples-uniform"),
            Self::SamplesFluidId => fl!("fluid-vis", "samples-fluid-id"),
            Self::SamplesQuantity => fl!("fluid-vis", "samples-quantity"),
            Self::SensorPlane => fl!("fluid-vis", "sensor-plane"),
            Self::NotLoaded => fl!("fluid-vis", "not-loaded"),
        }
    }

    pub fn uses_quantity(self) -> bool {
        matches!(self, Self::SamplesQuantity | Self::SensorPlane)
    }

    pub fn from_template(v: &FluidVisualization) -> Self {
        match v {
            FluidVisualization::TriangleMesh { coloring, .. } => match coloring {
                FluidMeshColoring::Uniform => Self::TriangleMeshUniform,
                FluidMeshColoring::FluidId { .. } => Self::TriangleMeshFluidId,
            },
            FluidVisualization::Samples { coloring, .. } => match coloring {
                FluidSampleColoring::Uniform => Self::SamplesUniform,
                FluidSampleColoring::FluidId { .. } => Self::SamplesFluidId,
                FluidSampleColoring::QuantityGraded { .. } => Self::SamplesQuantity,
            },
            FluidVisualization::SensorPlane { .. } => Self::SensorPlane,
        }
    }
}

/// Scalar Quantity.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum QuantityOption {
    #[default]
    Speed,
    Volume,
    Density,
    DensityError,
    Pressure,
    KineticEnergy,
}

impl QuantityOption {
    pub const ALL: [QuantityOption; 6] = [
        Self::Speed,
        Self::Volume,
        Self::Density,
        Self::DensityError,
        Self::Pressure,
        Self::KineticEnergy,
    ];

    pub fn label(self) -> String {
        match self {
            Self::Speed => fl!("quantity", "speed"),
            Self::Volume => fl!("quantity", "volume"),
            Self::Density => fl!("quantity", "density"),
            Self::DensityError => fl!("quantity", "density-error"),
            Self::Pressure => fl!("quantity", "pressure"),
            Self::KineticEnergy => fl!("quantity", "kinetic-energy"),
        }
    }

    pub fn from_template(t: &FluidVisualization) -> Self {
        match t {
            FluidVisualization::Samples {
                coloring: FluidSampleColoring::QuantityGraded { quantity, .. },
                ..
            } => Self::from_quantity(quantity),
            FluidVisualization::SensorPlane { quantity, .. } => Self::from_quantity(quantity),
            _ => Self::default(),
        }
    }

    /// Template mit leerem Vec – `from_system` füllt die Werte beim Laden.
    pub fn to_quantity(self) -> ScalarQuantity {
        match self {
            Self::Speed => ScalarQuantity::Speed,
            Self::Volume => ScalarQuantity::Volume,
            Self::Density => ScalarQuantity::Density,
            Self::DensityError => ScalarQuantity::DensityError,
            Self::Pressure => ScalarQuantity::Pressure,
            Self::KineticEnergy => ScalarQuantity::KineticEnergy,
        }
    }

    pub fn from_quantity(q: &ScalarQuantity) -> Self {
        match q {
            ScalarQuantity::Speed => Self::Speed,
            ScalarQuantity::Volume => Self::Volume,
            ScalarQuantity::Density => Self::Density,
            ScalarQuantity::DensityError => Self::DensityError,
            ScalarQuantity::Pressure => Self::Pressure,
            ScalarQuantity::KineticEnergy => Self::KineticEnergy,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensorField {
    Min(usize), // 0 = x, 1 = y, 2 = z
    Max(usize),
    Dx,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SensorPlaneConfig {
    // Als Strings, damit freies Tippen (z. B. "1.") möglich ist.
    pub min: [String; 3],
    pub max: [String; 3],
    pub dx: String,
    pub step: f32,    // Schrittweite für min/max
    pub dx_step: f32, // Schrittweite für dx
    pub min_prev: [String; 3],
    pub max_prev: [String; 3],
    pub dx_prev: String,
}

impl Default for SensorPlaneConfig {
    fn default() -> Self {
        Self {
            min: ["0.000".into(), "0.000".into(), "0.000".into()],
            max: ["10.000".into(), "10.000".into(), "10.000".into()],
            dx: "0.050".into(),
            step: 1.0,
            dx_step: 0.01,
            min_prev: ["0.000".into(), "0.000".into(), "0.000".into()],
            max_prev: ["10.000".into(), "10.000".into(), "10.000".into()],
            dx_prev: "0.050".into(),
        }
    }
}

impl SensorPlaneConfig {
    pub fn parse_min(&self) -> [f32; 3] {
        std::array::from_fn(|i| self.min[i].parse().unwrap_or(0.))
    }
    pub fn parse_max(&self) -> [f32; 3] {
        std::array::from_fn(|i| self.max[i].parse().unwrap_or(0.))
    }
    pub fn parse_dx(&self) -> f32 {
        self.dx.parse().unwrap_or(0.05 as f32).max(1e-4)
    }

    pub fn changed(&self) -> bool {
        self.min != self.min_prev || self.max != self.max_prev || self.dx != self.dx_prev
    }

    /// Ensures min ≤ max gilt.
    /// Is called when the "Apply" button is pressed.
    pub fn clamp_min_max(&mut self) {
        for i in 0..3 {
            let min: f32 = self.min[i].parse().unwrap_or(0.);
            let max: f32 = self.max[i].parse().unwrap_or(0.);
            if min > max {
                self.min[i] = format!("{max:.3}");
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryVisOption {
    MeshOriginal,
    MeshUniform,
    MeshBoundaryId,
    SamplesUniform,
    SamplesBoundaryId,
    NotLoaded,
}

impl BoundaryVisOption {
    pub const ALL: [BoundaryVisOption; 6] = [
        Self::MeshOriginal,
        Self::MeshUniform,
        Self::MeshBoundaryId,
        Self::SamplesUniform,
        Self::SamplesBoundaryId,
        Self::NotLoaded,
    ];

    pub fn label(self) -> String {
        match self {
            Self::MeshOriginal => fl!("boundary-vis", "mesh-original"),
            Self::MeshUniform => fl!("boundary-vis", "mesh-uniform"),
            Self::MeshBoundaryId => fl!("boundary-vis", "mesh-boundary-id"),
            Self::SamplesUniform => fl!("boundary-vis", "samples-uniform"),
            Self::SamplesBoundaryId => fl!("boundary-vis", "samples-boundary-id"),
            Self::NotLoaded => fl!("boundary-vis", "not-loaded"),
        }
    }

    pub fn from_template(v: &BoundaryVisualization) -> Self {
        match v {
            BoundaryVisualization::TriangleMesh { coloring, .. } => match coloring {
                BoundaryMeshColoring::Original => Self::MeshOriginal,
                BoundaryMeshColoring::Uniform => Self::MeshUniform,
                BoundaryMeshColoring::BoundaryId { .. } => Self::MeshBoundaryId,
            },
            BoundaryVisualization::Samples { coloring, .. } => match coloring {
                BoundarySampleColoring::Uniform => Self::SamplesUniform,
                BoundarySampleColoring::BoundaryId { .. } => Self::SamplesBoundaryId,
            },
        }
    }
}
