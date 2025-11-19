//! AppState
//!
//!
use std::sync::Arc;
use crossbeam::channel::Sender;
use iced_winit::winit;
use iced_winit::winit::event::{WindowEvent, DeviceEvent};
use iced_winit::runtime::user_interface::UserInterface;
use iced_wgpu::wgpu;
use iced_wgpu::graphics::Viewport;

#[cfg(feature = "logging")]
use tracing::{
    info,
    debug,
}; // error, trace, warn, debug, info,

mod gpu_context;
mod pipelines;
mod camera;
mod lighting;
mod model;
mod instances;
pub mod ui;
mod frame_control;

use model::VertexBufferLayout;
use model::DrawLight;
use model::DrawModel;

use super::backend::{SimulationInfo, TimeStepInfo, commands::WorkerCommand};
use ui::controls;
use ui::UserInput;



const CAMERA_POSITION: (f32, f32, f32) = (-20.0, 30.0, -20.0);
const YAW: cgmath::Deg<f32> = cgmath::Deg(45.0);
const PITCH: cgmath::Deg<f32> = cgmath::Deg(-20.0);
const SPEED: f32 = 50.0;
const SENSITIVITY: f32 = 5.0;
const FOVY: cgmath::Deg<f32> = cgmath::Deg(45.0);
const ZNEAR: f32 = 0.1;
const ZFAR: f32 = 100.;
const LIGHT_POSITION: [f32; 3] = [100., 100., 0.];
// const LIGHT_COLOR: Option<[f32; 3]> = Some([1., 0.5, 0.5]);
const LIGHT_COLOR: Option<[f32; 3]> = Some([1.; 3]);

const PARTICLE_COLOR: controls::ParticleColor = ui::controls::ParticleColor::VelocityGraded;
const BOUNDARY_PARTICLE_COLOR: controls::ParticleColor = ui::controls::ParticleColor::FixedColor([0.; 3]);


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


    // surface: wgpu::Surface<'static>,
    // device: Device,
    // queue: Queue,
    // config: SurfaceConfiguration,
    // depth_texture: model::Texture,

    // sphere: model::Model,

    // particles: Vec<super::mediation::Instance>,
    // boundary_particles: Vec<super::mediation::Instance>,
    // rendered_instances: Vec<super::mediation::Instance>,
    // rendered_instance_buffer: wgpu::Buffer,

    // camera: camera::Camera,
    // projection: camera::Projection,
    // camera_controller: camera::CameraController,

    // camera_uniform: camera::CameraUniform,
    // camera_buffer: wgpu::Buffer,
    // camera_bind_group: wgpu::BindGroup,


    // light_uniform: model::LightUniform,
    // light_buffer: wgpu::Buffer,
    // light_bind_group: BindGroup,

    // render_pipeline: RenderPipeline,
    // light_render_pipeline: RenderPipeline,

    // renderer: Renderer,

    // uicontrols: UIControls,
    // events: Vec<Event>,
    // cache: user_interface::Cache,
    // modifiers: ModifiersState,
    // cursor: mouse::Cursor,
    // mouse_right_button_pressed: bool,
    // clipboard: Clipboard,
}

impl AppState {
    pub fn new(
        window: winit::window::Window,
    ) -> Result<Self, tobj::LoadError> {
        let window_arc = Arc::new(window);

        let size = window_arc.inner_size();
        let viewport = Viewport::with_physical_size(
            iced_winit::core::Size::new(size.width, size.height),
            window_arc.scale_factor(),
        );


        // let instance = Self::create_gpu_instance();
        // let surface = instance.create_surface(window_arc.clone()).unwrap();
        // let adapter = Self::create_adapter(instance, &surface);
        // let (device, queue) = Self::create_device(&adapter);
        // let surface_caps = surface.get_capabilities(&adapter);
        // let config = Self::create_surface_config(size, surface_caps);
        // surface.configure(&device, &config);
        // let depth_texture = Texture::create_depth_texture(&device, &config, "depth_texture");
        let gpu = gpu_context::GpuContext::new(window_arc.clone(), size);

        // let camera = camera::Camera::new((0.0, 5.0, 10.0), cgmath::Deg(-90.0), cgmath::Deg(-20.0));
        // let projection = camera::Projection::new(config.width, config.height, cgmath::Deg(45.0), 0.1, 100.0);
        // let camera_controller = camera::CameraController::new(4.0, 0.4);

        // let mut camera_uniform = camera::CameraUniform::default(); // edit?
        // camera_uniform.update_view_proj(&camera, &projection);

        // let camera_buffer = Self::create_uniform_buffer(&device, camera_uniform, "Camera Buffer");

        // let camera_bind_group_layout = Self::create_bind_group_layout(&device, "camera_bind_group_layout");
        // let camera_bind_group = Self::create_bind_group(&device, &camera_bind_group_layout, &camera_buffer, "camera_bind_group");
        let camera = camera::CameraBundle::new(&gpu, CAMERA_POSITION, YAW, PITCH, SPEED, SENSITIVITY, FOVY, ZNEAR, ZFAR);

        // let light_uniform = model::LightUniform::new(light_position); // edit?

        // let light_buffer = Self::create_uniform_buffer(&device, light_uniform, "Light Buffer");
        // let light_bind_group_layout = Self::create_bind_group_layout(&device, "light_bind_group_layout");
        // let light_bind_group = Self::create_bind_group(&device, &light_bind_group_layout, &light_buffer, "light_bind_group");
        let light = lighting::LightBundle::new(&gpu, LIGHT_POSITION, LIGHT_COLOR);

        let pipelines = pipelines::Pipelines::new(
            &gpu,
            &camera,
            &light,
            Some(gpu_context::Texture::DEPTH_FORMAT),
            &[model::ModelVertex::desc(), model::InstanceRaw::desc()],
        );
        // let render_pipeline = {
        //     let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        //         label: Some("Render Pipeline Layout"),
        //         bind_group_layouts: &[&camera_bind_group_layout, &light_bind_group_layout],
        //         push_constant_ranges: &[],
        //     });
        //     let shader = wgpu::ShaderModuleDescriptor {
        //         label: Some("Normal Shader"),
        //         source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        //     };
        //     Self::create_render_pipeline(
        //         &device,
        //         &layout,
        //         config.format,
        //         Some(Texture::DEPTH_FORMAT),
        //         &[model::ModelVertex::desc(), model::InstanceRaw::desc()],
        //         shader,
        //     )
        // };


        // let light_render_pipeline = {
        //     let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        //         label: Some("Light Pipeline Layout"),
        //         bind_group_layouts: &[&camera_bind_group_layout, &light_bind_group_layout],
        //         push_constant_ranges: &[],
        //     });
        //     let shader = wgpu::ShaderModuleDescriptor {
        //         label: Some("Light Shader"),
        //         source: wgpu::ShaderSource::Wgsl(include_str!("light.wgsl").into()),
        //     };
        //     Self::create_render_pipeline(
        //         &device,
        //         &layout,
        //         config.format,
        //         Some(Texture::DEPTH_FORMAT),
        //         &[model::ModelVertex::desc()],
        //         shader,
        //     )
        // };

        // let sphere = model::load_model("./src/gui/model/sphere.obj", &device, sphere_size).unwrap();
        let particle_diameter = 1.0;
        let model = model::ModelAssets::new(&gpu, particle_diameter)?;

        let ui = ui::UIState::new(
            window_arc.clone(),
            &gpu,
            PARTICLE_COLOR,
            BOUNDARY_PARTICLE_COLOR,
        );

        let instances = instances::InstanceStore::new(&gpu, 0);

        let frame_control = frame_control::FrameControl::new();

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
            frame: frame_control,
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
            self.camera.projection.resize(new_size.width, new_size.height);
            self.gpu.surface.configure(&self.gpu.device, &self.gpu.config,);
            #[cfg(feature = "logging")]
            info!("Resized to {:?}", new_size)
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

    // /// modify State's state depending on input
    // pub fn input(&mut self, event: &WindowEvent) -> bool {
        // // Map window event to iced event
        // if let Some(event) = iced_winit::conversion::window_event(
        //     event.clone(),
        //     self.window.scale_factor(),
        //     self.ui.modifiers,
        // ) {
        //     self.ui.events.push(event);
        // }
        // // If there are events pending
        // if !self.ui.events.is_empty() {
        //     self.process_events()
        // }
        // match event {
        //     WindowEvent::KeyboardInput {
        //         event:
        //             winit::event::KeyEvent {
        //                 physical_key: winit::keyboard::PhysicalKey::Code(key),
        //                 state,
        //                 ..
        //             },
        //         ..
        //     } => {
        //         self.camera.as_mut().unwrap().controller.process_keyboard(*key, *state) || self.process_keyboard(key, state)
        //     },
            // WindowEvent::MouseWheel { delta, .. } => {
            //     self.camera.as_mut().unwrap().controller.process_scroll(delta);
            //     true
            // }
    //         WindowEvent::MouseInput {
    //             button: winit::event::MouseButton::Right,
    //             state,
    //             ..
    //         } => {
    //             self.ui.mouse_right_button_pressed = *state == winit::event::ElementState::Pressed;
    //             true
    //         }
    //         WindowEvent::CursorMoved { position, .. } => {
    //             self.ui.cursor =
    //                 iced_winit::core::mouse::Cursor::Available(iced_winit::conversion::cursor_position(
    //                     *position,
    //                     self.viewport.scale_factor(),
    //                 ));
    //             true
    //         }
    //         WindowEvent::ModifiersChanged(new_modifiers) => {
    //             self.ui.modifiers = new_modifiers.state();
    //             true
    //         }
    //         _ => false,
    //     }
    // }
    //     self.ui.view = self.ui.view();
    //     // Process events
    //     let mut interface = UserInterface::build(
    //         self.ui.controls.view(),
    //         self.viewport.logical_size(),
    //         std::mem::take(&mut self.ui.cache),
    //         &mut self.ui.renderer,
    //     );

    //     let mut messages = Vec::new();

    //     let _ = interface.update(
    //         &self.ui.events,
    //         self.ui.cursor,
    //         &mut self.ui.renderer,
    //         &mut self.ui.clipboard,
    //         &mut messages,
    //     );

    //     self.ui.events.clear();
    //     self.ui.cache = interface.into_cache();

    //     // update our UI with any messages
    //     for message in messages {
    //         self.ui.controls.update(message);
    //     }

    //     // and request a redraw
    //     self.window.request_redraw();
    // }

    // // TODO move and unify this into/with controls
    // fn process_keyboard(&mut self, key: &winit::keyboard::KeyCode, state: &winit::event::ElementState) -> bool {
    //     match key {
    //         winit::keyboard::KeyCode::KeyK => {
    //             if *state == winit::event::ElementState::Pressed {
    //                 // if cfg!(feature = "logging") {
    //                 //     debug!("K pressed");
    //                 // }
    //                 self.ui.controls.play_pause.toggle();
    //                 true
    //             } else {
    //                 false
    //             }
    //         }
    //         winit::keyboard::KeyCode::KeyR => {
    //             if *state == winit::event::ElementState::Pressed {
    //                 // if cfg!(feature = "logging") {
    //                 //     debug!("R pressed");
    //                 // }
    //                 self.ui.controls.update(super::ui::UserInput::RequestReset);
    //                 true
    //             } else {
    //                 false
    //             }
    //         }
    //         _ => false,
    //     }
    // }

    /// Update buffer for next rendering step (consider new state of State)
    pub fn update(
        &mut self,
        to_worker: &Sender<WorkerCommand>,
    ) {
        // send new timesteps to compute
        let add_steps = self.timesteps_to_compute();
        if add_steps > 0 {
            to_worker.send(WorkerCommand::AddTimeStepsToCompute(add_steps)).unwrap();
        }

        // handle user input messages
        for message in self.messages.drain(..) {
            match message {
                // send commands to worker thread
                UserInput::RequestCameraReset => {
                    self.camera.reset(&self.gpu);
                    // self.light.reset(&self.gpu);
                },
                UserInput::StepInTime => {
                    self.frame.step();
                    #[cfg(feature = "logging")]
                    info!("Step in time");
                }
                UserInput::RequestReset => {
                    to_worker.send(WorkerCommand::Reset).unwrap();
                    to_worker.send(WorkerCommand::AddTimeStepsToCompute(
                        self.instances.length_limit-self.instances.queue_len()
                    )).unwrap();
                },
                UserInput::RequestSaving => {
                    if !self.instances.is_empty() {
                        to_worker.send(
                            WorkerCommand::Save {
                                particles: self.instances.get_info().unwrap().fluid,
                                filepath: "./state.ron".to_string()
                            }
                        ).unwrap()
                    }
                },
                UserInput::ToggleDisplayState => {
                    self.frame.reset_steps();
                    // control is update in ui.update not here
                }
                _ => (),
            }
            // also update ui
            self.ui.update(message);
        }

        // Update camera
        self.camera.update(&self.gpu, self.frame.time_since_last_render());

        // Update the light
        self.light.update(&self.gpu, self.frame.time_since_last_render());
    }

    fn timesteps_to_compute(&mut self) -> usize {
        let timesteps = self.frame.time_steps_dequeued;
        self.frame.time_steps_dequeued = 0;
        timesteps
    }




    pub fn new_simulation(&mut self, info: SimulationInfo, to_worker: &Sender<WorkerCommand>,) {
        match model::ModelAssets::new(&self.gpu, info.particle_diameter) {
            Ok(model) => self.model = model,
            Err(e) => panic!("Failed to load sphere: {}", e),
        }
        self.camera.reset(&self.gpu);
        self.light.set_light(&self.gpu, info.light_position, LIGHT_COLOR);
        self.instances = instances::InstanceStore::new(&self.gpu, info.buffer_length_limit);
        self.instances.update_length_limit(&info);
        self.ui.new_simulation(info);
        self.frame.step();

        to_worker.send(WorkerCommand::AddTimeStepsToCompute(
            self.instances.length_limit-self.instances.queue_len()
        )).unwrap();
    }

    // might panic
    pub fn received_new_time_step(&mut self, info: TimeStepInfo) {
        self.instances.store(info);
    }

    pub fn continue_after_reset(&mut self) {
        self.instances.clear(&self.gpu);
        self.frame.reset();
    }




    /// This function renders the one frame. This includes:
    /// - selecting next frame
    /// - filtering and stage next frame
    /// - rendering instances
    /// - rendering ui
    /// - setting time of this rendering
    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // select
        let take = self.frame.take_the_xth_element(&self.ui.controls.play_pause);
        #[cfg(feature = "logging")]
        debug!("take: {}", take);
        let staging_settings = instances::StagingSettings::new(
            self.ui.controls.get_cut().clone(),
            self.ui.controls.is_boundary_hidden(),
            self.ui.controls.particle_color,
            self.ui.controls.boundary_particle_color,
        );
        // get next rendered instances
        if let Ok(taken) = self.instances.stage_next(
            &self.gpu,
            &staging_settings,
            take,
        ) {
            #[cfg(feature = "logging")]
            debug!("taken: {}", taken);
            self.frame.rendering_new_sim_state_now();
            self.frame.steps_dequeued(taken);
            self.frame.set_time_increment(self.instances.get_time_inc());
        } else {
            self.instances.update_staged(
                &self.gpu,
                &staging_settings,
            );
            if !self.ui.controls.play_pause.is_playing() {
                self.frame.rendering_new_sim_state_now();
            }
        }
        self.ui.update_time_step_info(self.instances.queue_len(), &self.instances.get_info(),);

        // prepare rendering
        let frame = self.gpu.surface.get_current_texture()?;
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.render_scene(&view);

        // draw iced on top
        self.ui.draw( &self.window, &self.viewport, &frame, &view);

        // present the frame
        frame.present();

        self.frame.rendering_now();

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
}
