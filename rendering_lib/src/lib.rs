//! AppState
//!
//!
use std::sync::Arc;
use iced_winit::winit;
use iced_winit::winit::event::{WindowEvent, DeviceEvent};
use iced_winit::runtime::user_interface::UserInterface;
use iced_wgpu::wgpu;
use iced_wgpu::graphics::Viewport;

#[cfg(feature = "logging")]
use tracing::{
    debug,
}; // error, trace, warn, debug, info,

use simulation_lib::{ParticleColor};

pub mod gpu_context;
pub mod pipelines;
pub mod camera;
pub mod lighting;
pub mod model;
pub mod instances;
pub mod ui;
pub mod frame_control;
pub mod readback;
pub mod settings;

use model::VertexBufferLayout;
use model::DrawLight;
use model::DrawModel;

use ui::UserInput;

use crate::readback::{ReadbackAction, ReadbackController};



const CAMERA_POSITION: (f32, f32, f32) = (10.0, -30.0, 40.0);
const YAW: cgmath::Deg<f32> = cgmath::Deg(-90.0);
const PITCH: cgmath::Deg<f32> = cgmath::Deg(-30.0);
const SPEED: f32 = 0.25;
const SENSITIVITY: f32 = 1.25;
const SCROLL_SPEED: f32 = 5.;
const FOVY: cgmath::Deg<f32> = cgmath::Deg(45.0);
const ZNEAR: f32 = 0.1;
const ZFAR: f32 = 100.;
const LIGHT_POSITION: [f32; 3] = [2., 2., 100.];
// const LIGHT_COLOR: Option<[f32; 3]> = Some([1., 0.5, 0.5]);
const LIGHT_COLOR: Option<[f32; 3]> = Some([1.; 3]);
const LIGHT_MOVEMENT_SPEED: f32 = 5.;

const PARTICLE_COLOR: ParticleColor = ParticleColor::VelocityGraded;
const BOUNDARY_PARTICLE_COLOR: ParticleColor = ParticleColor::FixedColor([0.; 3]);


pub struct AppState {
    pub window: Arc<winit::window::Window>,
    pub viewport: Viewport,
    pub gpu: gpu_context::GpuContext,
    pub pipelines: pipelines::Pipelines,
    pub camera: camera::CameraBundle,
    pub light: lighting::LightBundle,
    pub model: model::ModelAssets,
    pub instances: instances::InstanceStore,
    pub ui: ui::UIState,
    pub messages: Vec<UserInput>,
    pub frame: frame_control::FrameControl,
    pub screenshot: readback::ReadbackController,
    pub settings: settings::Settings,
}

impl AppState {
    pub fn new(
        window: winit::window::Window,
        start_resumed: bool,
        rendering_dir: Option<String>,
        start_time: Option<f64>,
        finish_time: Option<f64>,
        discard_past: bool,
        wait_for_timesteps: bool,
    ) -> Result<Self, tobj::LoadError> {
        let window_arc = Arc::new(window);

        let size = window_arc.inner_size();
        let viewport = Viewport::with_physical_size(
            iced_winit::core::Size::new(size.width, size.height),
            window_arc.scale_factor(),
        );

        let gpu = gpu_context::GpuContext::new(window_arc.clone(), size);

        let camera = camera::CameraBundle::new(&gpu, CAMERA_POSITION, YAW, PITCH, SPEED, SENSITIVITY, SCROLL_SPEED, FOVY, ZNEAR, ZFAR);

        let light = lighting::LightBundle::new(&gpu, LIGHT_POSITION, LIGHT_COLOR, LIGHT_MOVEMENT_SPEED);

        let pipelines = pipelines::Pipelines::new(
            &gpu,
            &camera,
            &light,
            Some(gpu_context::Texture::DEPTH_FORMAT),
            &[model::ModelVertex::desc(), model::InstanceRaw::desc()],
        );

        // let sphere = model::load_model("./src/gui/model/sphere.obj", &device, sphere_size).unwrap();
        let particle_diameter = 1.0;
        let model = model::ModelAssets::new(&gpu, particle_diameter)?;

        let ui = ui::UIState::new(
            window_arc.clone(),
            &gpu,
            PARTICLE_COLOR,
            BOUNDARY_PARTICLE_COLOR,
            start_resumed,
            rendering_dir.is_some(),
            discard_past,
        );

        let instances = instances::InstanceStore::new(&gpu);

        let frame = frame_control::FrameControl::default();

        let screenshot = ReadbackController::new(&gpu, size, rendering_dir, start_time, finish_time);

        let settings = settings::Settings::new(wait_for_timesteps);

        Ok(Self {
            window: window_arc,
            viewport,
            gpu,
            pipelines,
            camera,
            light,
            model,
            ui,
            instances,
            messages: Vec::new(),
            frame,
            screenshot,
            settings,
        })
    }

    pub fn window(&self) -> &winit::window::Window {
        &self.window
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.viewport = Viewport::with_physical_size(
                iced_wgpu::core::Size::new(new_size.width, new_size.height),
                self.window.scale_factor(),
            );
            self.gpu.config.width = new_size.width;
            self.gpu.config.height = new_size.height;
            self.gpu.depth_texture = gpu_context::Texture::create_depth_texture(
                &self.gpu.device,
                &self.gpu.config,
                "depth_texture"
            );
            self.gpu.offscreen_texture = gpu_context::GpuContext::create_offscreen_texture(&self.gpu.device, new_size);
            self.gpu.surface.configure(&self.gpu.device, &self.gpu.config,);
            self.camera.projection.resize(new_size.width, new_size.height);
            self.screenshot.resize(&self.gpu, new_size);

            #[cfg(feature = "logging")]
            debug!("Resized to {:?}", new_size);
        }
    }

    pub fn process_window_event(&mut self, event: &WindowEvent,) {
        // map window event to iced events, which then trigger [[UserInput]]s
        self.get_iced_events(self.viewport.logical_size(), self.window.scale_factor(), event.clone());
        // process keyboard
        self.process_keyboard(event);
        // get UserInput messages from events
        self.ui.process_window_event(self.window.scale_factor(), event);
        //
        self.camera.process_window_event(event);
    }

    pub fn get_iced_events(
        &mut self,
        bounds: iced_winit::core::Size,
        scale_factor: f64,
        event: winit::event::WindowEvent,
    ) {
        if let Some(event) = iced_winit::conversion::window_event(
            event,
            scale_factor,
            self.ui.modifiers,
        ) {
            self.ui.events.push(event);
        }

        let mut interface = UserInterface::build(
            self.ui.controls.view(),
            bounds,
            std::mem::take(&mut self.ui.cache),
            &mut self.ui.renderer,
        );

        let _ = interface.update(
            &self.ui.events,
            self.ui.cursor,
            &mut self.ui.renderer,
            &mut self.ui.clipboard,
            &mut self.messages,
        );

        self.ui.events.clear();
        self.ui.cache = interface.into_cache();
    }

    #[allow(clippy::single_match)]
    fn process_keyboard(&mut self, event: &winit::event::WindowEvent,) {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key: winit::keyboard::PhysicalKey::Code(key),
                        state,
                        ..
                    },
                ..
            } => {
                self.ui.process_keyboard(key, state);
                match key {
                    winit::keyboard::KeyCode::KeyR => {
                        if *state == winit::event::ElementState::Pressed {
                            // #[cfg(feature = "logging")]
                            // debug!("R pressed");
                            self.messages.push(UserInput::RequestReset);
                        }
                    }
                    _ => (),
                }
            },
            _ => (),
        }
    }

    pub fn process_device_event(&mut self, event: &DeviceEvent) {
        self.camera.process_device_event(event, self.ui.mouse_right_button_pressed);
    }

    /// This function renders the one frame. This includes:
    /// - selecting next frame
    /// - filtering and stage next frame
    /// - rendering instances
    /// - rendering ui
    /// - setting time of this rendering
    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // choose frame
        let next_action = self.frame.get_next_action(self.ui.controls.is_playing());

        let staging_settings = instances::StagingSettings::new(
            self.ui.controls.get_cut().clone(),
            self.ui.controls.is_boundary_hidden(),
            self.ui.controls.particle_color,
            self.ui.controls.boundary_particle_color,
        );
        let mut frame_new = false;
        // get next rendered instances
        match self.instances.stage_next(
            &self.gpu,
            &staging_settings,
            next_action,
            self.ui.controls.is_playing_forward(),
            self.ui.controls.is_playing_looped(),
            self.ui.controls.is_past_discarded(),
        ) {
            instances::StagingResult::Initialized => {
                self.frame.rendering_new_sim_state_now();
                self.frame.set_time_increment(self.instances.get_time_inc());
                frame_new = true;
            },
            instances::StagingResult::SteppedInTime => {
                self.frame.rendering_new_sim_state_now();
                self.frame.stepped_in_time();
                self.frame.count_discarded_timesteps(1, self.ui.controls.is_past_discarded());
                self.frame.set_time_increment(self.instances.get_time_inc());
                frame_new = true;
            },
            instances::StagingResult::SomeTaken(discarded) => {
                self.frame.rendering_new_sim_state_now();
                self.frame.count_discarded_timesteps(discarded, self.ui.controls.is_past_discarded());
                self.frame.set_time_increment(self.instances.get_time_inc());
                frame_new = true;
            },
            instances::StagingResult::StoppedAtLoopEndWithSomeTaken(discarded) => {
                self.frame.rendering_new_sim_state_now();
                self.frame.count_discarded_timesteps(discarded, self.ui.controls.is_past_discarded());
                if !self.ui.controls.is_past_discarded() || !self.settings.wait_for_timesteps {
                    self.ui.controls.playback_controls.pause();
                }
                self.frame.set_time_increment(self.instances.get_time_inc());
                frame_new = true;
            },
            instances::StagingResult::StoppedAtLoopEndWithNoneTaken => {
                self.instances.update_staged(
                    &self.gpu,
                    &staging_settings,
                );
                if !self.ui.controls.is_past_discarded() || !self.settings.wait_for_timesteps {
                    self.ui.controls.playback_controls.pause();
                }
                self.frame.rendering_new_sim_state_now();
            },
            instances::StagingResult::NoneTaken | instances::StagingResult::NothingToStage => {
                self.instances.update_staged(
                    &self.gpu,
                    &staging_settings,
                );
                if !self.ui.controls.is_playing() {
                    self.frame.rendering_new_sim_state_now();
                }
            },
            instances::StagingResult::Uninitialized => (),

        }
        self.ui.update_time_step_info(self.instances.get_info(), self.instances.remaining_buffer_len());

        self.screenshot.update_rendering_status(
            self.ui.controls.info.time,
            &mut self.ui.controls.info.rendering_status.recording_status,
        );
        if let ReadbackAction::Read(path) = self.screenshot.screenshot_this(
            frame_new,
            self.ui.controls.info.rendering_status.recording_status,
        ) {
            // render to offscreen texture
            let view = self.gpu.offscreen_texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.render_scene(&view);
            self.send_readback_request_for_current_frame(path);
        }

        // render to screen
        let frame = self.gpu.surface.get_current_texture()?;
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.render_scene(&view);

        // draw iced on top
        self.ui.draw( &self.window, &self.viewport, &frame, &view);

        // present the frame
        frame.present();

        Ok(())
    }

    fn render_scene(&self, view: &wgpu::TextureView,) {
        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(
                                wgpu::Color {
                                    r: self.ui.controls.background_color().r as f64,
                                    g: self.ui.controls.background_color().g as f64,
                                    b: self.ui.controls.background_color().b as f64,
                                    a: self.ui.controls.background_color().a as f64,
                                }
                            ),
                            store: wgpu::StoreOp::Store,
                        },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.gpu.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_vertex_buffer(1, self.instances.buffer.slice(..));

            render_pass.set_pipeline(&self.pipelines.light);
            render_pass.draw_light_model(
                &self.model.sphere_mesh,
                &self.camera.bind_group,
                &self.light.bind_group,
            );

            let length = if let Some(ren_incs) = &self.instances.rendered_instances {
                ren_incs.len() as u32
            } else {
                1
            };

            render_pass.set_pipeline(&self.pipelines.object);
            render_pass.draw_model_instanced(
                &self.model.sphere_mesh,
                0..length,
                &self.camera.bind_group,
                &self.light.bind_group,
            );
        }
        // submit will accept anything that implements IntoIter
        self.gpu.queue.submit(std::iter::once(encoder.finish()));
    }

    fn send_readback_request_for_current_frame(&mut self, output_dir:std::path::PathBuf) {
        let (buffer, next_frame_index, padded_bytes_per_row) = self.screenshot.buffers.get_next_buffer_and_info();
        let size = self.window.inner_size();

        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("screenshot encoder"),
            },
        );

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.gpu.offscreen_texture,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
                mip_level: 0,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer.lock().unwrap().buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(size.height),
                },
            },
            wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
        );

        self.gpu.queue.submit(Some(encoder.finish()));

        // Send job to worker thread
        let req = readback::ReadbackRequest {
            buffer: buffer.clone(),
            width: size.width,
            height: size.height,
            frame_index: next_frame_index,
            device: self.gpu.device.clone(),
            output_dir,
        };

        self.messages.push(UserInput::RequestReadback(req));
    }
}


// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn it_works() {
//         let result = add(2, 2);
//         assert_eq!(result, 4);
//     }
// }
