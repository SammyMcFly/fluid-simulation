//! UI
//!
//!
use std::sync::Arc;
use iced_winit::winit;
use iced_winit::winit::event::WindowEvent;
use iced_wgpu::wgpu;
use iced_winit::runtime::user_interface::UserInterface;

use simulation_lib::{SimulationParameters, TimeStepInfo, ParticleColor};

pub mod controls;



#[derive(Debug, Clone)]
pub enum UserInput {
    ToggleDisplayState,
    StepInTime,
    RequestCameraReset,
    RequestReset,
    RequestSaving,
    ToggleHideBoundary,
    ToggleCutX,
    CutXBoundChanged(f32),
    FlipCutX,
    ToggleCutY,
    CutYBoundChanged(f32),
    FlipCutY,
}

pub struct UIState {
    pub controls: controls::RenderControls,

    pub cursor: iced_winit::core::mouse::Cursor,
    pub modifiers: winit::keyboard::ModifiersState,
    pub clipboard: iced_winit::Clipboard,
    pub mouse_right_button_pressed: bool,

    pub events: Vec<iced_winit::core::Event>,

    pub cache: iced_winit::runtime::user_interface::Cache,
    pub renderer: iced_wgpu::Renderer,
}

impl UIState {
    pub fn new(
        window: Arc<winit::window::Window>,
        gpu_context: &super::gpu_context::GpuContext,
        particle_color: ParticleColor,
        boundary_particle_color: ParticleColor,
        start_resumed: bool,
        is_rendered: bool
    ) -> Self {
        // initialize GUI controls
        let controls = controls::RenderControls::new(
            particle_color,
            boundary_particle_color,
            start_resumed,
            is_rendered,
        );

        // initialize iced renderer
        let renderer = {
            let engine = iced_wgpu::Engine::new(
                &gpu_context.adapter,
                gpu_context.device.clone(),
                gpu_context.queue.clone(),
                gpu_context.surface.get_capabilities(&gpu_context.adapter).formats
                .iter()
                .copied()
                .find(wgpu::TextureFormat::is_srgb)
                .or_else(|| {
                    gpu_context.surface.get_capabilities(&gpu_context.adapter).formats.first().copied()
                })
                .expect("Get preferred format"),
                None,
            );

            iced_wgpu::Renderer::new(
                engine,
                iced_winit::core::Font::default(),
                iced_winit::core::Pixels::from(16)
            )
        };

        let clipboard = iced_winit::Clipboard::connect(window);

        Self {
            controls: controls.clone(),
            cursor: iced_winit::core::mouse::Cursor::Unavailable,
            modifiers: winit::keyboard::ModifiersState::default(),
            clipboard,
            mouse_right_button_pressed: false,
            events: Vec::new(),
            cache: iced_winit::runtime::user_interface::Cache::new(),
            renderer,
        }
    }

    /// Process window events
    pub fn process_window_event(
        &mut self,
        scale_factor: f64,
        event: &winit::event::WindowEvent,
    ) {
        match event {
            WindowEvent::MouseInput {
                button: winit::event::MouseButton::Right,
                state,
                ..
            } => {
                self.mouse_right_button_pressed = *state == winit::event::ElementState::Pressed;
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor =
                    iced_winit::core::mouse::Cursor::Available(iced_winit::conversion::cursor_position(
                        *position,
                        scale_factor,
                    ));
            }
            WindowEvent::ModifiersChanged(new_modifiers) => {
                self.modifiers = new_modifiers.state();
            }
            _ => (),
        }
    }

    pub fn process_keyboard(&mut self, key: &winit::keyboard::KeyCode, state: &winit::event::ElementState) {
        #![allow(clippy::single_match)]
        match key {
            winit::keyboard::KeyCode::KeyK => {
                if *state == winit::event::ElementState::Pressed {
                    // #[cfg(feature = "logging")]
                    // debug!("K pressed");
                    self.controls.play_pause.toggle();
                }
            }
            _ => (),
        }
    }

    /// update UI
    pub fn update(&mut self, message: UserInput ) {
        self.controls.update(&message);
    }

    pub fn new_simulation(&mut self, info: SimulationParameters) {
        self.controls.new_simulation(info);
    }

    pub fn update_time_step_info(&mut self, queue_len: usize, info: Option<&TimeStepInfo>,) {
        self.controls.update_time_step_info(queue_len, info);
    }

    pub fn advance_to_next_measurement_state(&mut self) {
        self.controls.advance_to_next_measurement_state();
    }

    pub fn advance_to_next_recording_state(&mut self) {
        self.controls.advance_to_next_recording_state();
    }

    pub fn draw(
        &mut self,
        window: &Arc<winit::window::Window>,
        viewport: &iced_wgpu::graphics::Viewport,
        frame: &wgpu::SurfaceTexture,
        view: &wgpu::TextureView,
    ) {
        let mut interface = UserInterface::build(
            self.controls.view(),
            viewport.logical_size(),
            std::mem::take(&mut self.cache),
            &mut self.renderer,
        );

        let (state, _) = interface.update(
            &[iced_winit::core::Event::Window(
                iced_winit::core::window::Event::RedrawRequested(
                    std::time::Instant::now(),
                ),
            )],
            self.cursor,
            &mut self.renderer,
            &mut self.clipboard,
            &mut Vec::new(),
        );

        // Update the mouse cursor
        if let iced_winit::runtime::user_interface::State::Updated {
            mouse_interaction,
            ..
        } = state
        {
            window.set_cursor(
                iced_winit::conversion::mouse_interaction(
                    mouse_interaction,
                ),
            );
        }

        // Draw the interface
        interface.draw(
            &mut self.renderer,
            &iced_widget::Theme::Dark,
            &iced_winit::core::renderer::Style::default(),
            self.cursor,
        );
        self.cache = interface.into_cache();

        // // Update the mouse cursor
        // {
        //     self.window.set_cursor(
        //         conversion::mouse_interaction(
        //             mouse_interaction,
        //         ),
        //     );
        // }

        // let mut encoder_2 = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        //     label: Some("Render Encoder"),
        // });

        // self.renderer.present(
        //     &mut self.engine,
        //     &self.device,
        //     &self.queue,
        //     &mut encoder_2,
        //     None,
        //     frame.texture.format(),
        //     &view,
        //     &self.viewport,
        //     &["Hi".to_string()],
        // );
        self.renderer.present(
            None,
            frame.texture.format(),
            view,
            viewport,
        );
    }
}