//! ## Simulation frontend
//!
//! Frontend is based on wgpu and winit.
//!
use iced_winit::program::Message;
use tracing::{debug, info}; // error, trace, warn

use std::sync::{Arc, Mutex};

use iced_winit::winit;
use iced_winit::winit::window::Window;
use iced_winit::winit::event::WindowEvent;
use iced_winit::Clipboard;
use iced_winit::conversion;
use iced_winit::core::mouse;
use iced_winit::core::renderer;
use iced_winit::core::time::Instant;
use iced_winit::core::window;
use iced_winit::core::{Event, Font, Pixels, Size, Theme};
// use iced_winit::futures;
use iced_winit::winit::keyboard::ModifiersState;
use iced_winit::runtime::user_interface::{self, UserInterface};

use iced_wgpu::{Engine, Renderer};
use iced_wgpu::wgpu::{
    Adapter, BindGroup, BindGroupLayout, Device, PipelineLayout,
    PresentMode, Queue, RenderPipeline, Surface, SurfaceCapabilities, SurfaceConfiguration,
};
use iced_wgpu::wgpu::util::DeviceExt;
use iced_wgpu::graphics::Viewport;
use iced_wgpu::wgpu;

use pollster::FutureExt;

use cgmath::prelude::*;

pub mod model;
use model::{DrawLight, DrawModel, VertexBufferLayout, ToRaw};
mod camera;
mod controls;
use controls::UIControls;




#[derive(Debug)]
enum SimulationDisplayState {
    Resumed,
    Paused,
}

impl SimulationDisplayState {
    fn toggle(&mut self) {
        match *self {
            Self::Paused => *self = Self::Resumed,
            Self::Resumed => *self = Self::Paused,
        }
    }
}

pub struct StateApplication {
    state: Option<State>,
    last_render_time: Option<std::time::Instant>,
    last_frame_time: Option<std::time::Instant>,
    queue: Arc<Mutex<super::mediation::IntermediateQueue>>,
    controls: Arc<Mutex<super::mediation::IntermediateControls>>,
    // time inc in s
    time_inc: f32,
    do_reset: bool,
}

impl StateApplication {
    pub fn new(queue: Arc<Mutex<super::mediation::IntermediateQueue>>, controls: Arc<Mutex<super::mediation::IntermediateControls>>) -> Self {
        let time_inc = controls.lock().unwrap().time_inc();
        Self {
            state: Option::default(),
            last_render_time: Option::default(),
            last_frame_time: Option::default(),
            queue,
            controls,
            time_inc,
            do_reset: false,
        }
    }
}

impl winit::application::ApplicationHandler for StateApplication {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop.create_window(winit::window::Window::default_attributes()
            .with_visible(true).with_title("Rusty Fluid Solver")).unwrap();
        let instances = self.queue.lock().unwrap().pop_front().unwrap();
        let sphere_size = self.controls.lock().unwrap().particle_size();
        let light_position = self.controls.lock().unwrap().light_position();
        self.state = Some(State::new(window, instances, sphere_size, light_position));
        self.last_render_time = Some(std::time::Instant::now());
        self.last_frame_time = Some(std::time::Instant::now());
        self.time_inc = self.controls.lock().unwrap().time_inc();
    }

    fn window_event(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, id: winit::window::WindowId, event: WindowEvent) {
        let window = self.state.as_ref().unwrap().window();
        if window.id() == id && !self.state.as_mut().unwrap().input(&event) {
            match event {
                WindowEvent::CloseRequested => {
                    info!("The close button was pressed; stopping");
                    self.controls.lock().unwrap().terminate_connection();
                    event_loop.exit();
                },
                WindowEvent::Resized(physical_size) => {
                    self.state.as_mut().unwrap().resize(physical_size);
                }
                WindowEvent::RedrawRequested => {
                    let mut update_messages: Vec<controls::Message> = vec![];

                    if self.state.as_mut().unwrap().reset_requested {
                        self.controls.lock().unwrap().request_reset();
                        self.do_reset = true;
                        self.state.as_mut().unwrap().reset_requested = false;
                    }

                    let time_delta_to_last_render_time = self.last_render_time.unwrap().elapsed();

                    let new_instances = {
                        if !self.queue.lock().unwrap().is_empty() && !self.controls.lock().unwrap().is_reset_requested() {
                            // as soon as the queue has refreshed (reloading of scene file has finished), update parameters
                            // and dequeue and load first frame
                            if self.do_reset {
                                self.time_inc = self.controls.lock().unwrap().time_inc();
                                self.do_reset = false;
                                // update last frametime to now
                                self.last_frame_time = Some(std::time::Instant::now());
                                // UI update messages
                                update_messages.push(controls::Message::AverageDensityChanged(self.controls.lock().unwrap().get_average_density()));
                                update_messages.push(controls::Message::RestDensityChanged(self.controls.lock().unwrap().get_rest_density()));
                                // return
                                Some(self.queue.lock().unwrap().pop_front().unwrap())
                            } else if let SimulationDisplayState::Resumed = self.state.as_ref().unwrap().sim_state {
                                let next_visualized_queue_element = (self.last_frame_time.unwrap().elapsed().as_secs_f32()
                                    /self.time_inc) as u32;
                                if next_visualized_queue_element >= 1 {
                                    for _ in 1..next_visualized_queue_element {
                                        let mut queue = self.queue.lock().unwrap();
                                        if queue.len() >= 2 {
                                            queue.pop_front();
                                        }
                                    }
                                    // update last frametime to now
                                    self.last_frame_time = Some(std::time::Instant::now());
                                    // UI update messages
                                    update_messages.push(controls::Message::AverageDensityChanged(self.controls.lock().unwrap().get_average_density()));
                                    update_messages.push(controls::Message::RestDensityChanged(self.controls.lock().unwrap().get_rest_density()));
                                    Some(self.queue.lock().unwrap().pop_front().unwrap())
                                } else {
                                    None
                                }
                            } else {
                                self.last_frame_time = Some(std::time::Instant::now());
                                None
                            }
                        } else {
                            None
                        }
                    };

                    update_messages.push(controls::Message::BufferLengthChanged(self.queue.lock().unwrap().len()));
                    self.state.as_mut().unwrap().update(time_delta_to_last_render_time, new_instances, update_messages);

                    self.last_render_time = Some(std::time::Instant::now());
                    self.state.as_mut().unwrap().render().unwrap();
                    self.state.as_mut().unwrap().window.request_redraw();
                }
                _ => (),
            }
        }
    }
    fn device_event(
            &mut self,
            _event_loop: &winit::event_loop::ActiveEventLoop,
            _device_id: winit::event::DeviceId,
            event: winit::event::DeviceEvent,
        ) {
        match event {
            winit::event::DeviceEvent::MouseMotion{ delta, } => {
                if self.state.as_mut().unwrap().mouse_pressed {
                    self.state.as_mut().unwrap().camera_controller.process_mouse(delta.0, delta.1);
                }
            }
            _ => (),
        }
    }
}

struct State {
    window: Arc<Window>,
    surface: Surface<'static>,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    depth_texture: model::Texture,

    sphere: model::Model,

    instances: Vec<super::mediation::Instance>,
    instance_buffer: wgpu::Buffer,

    camera: camera::Camera,
    camera_uniform: camera::CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    projection: camera::Projection,
    camera_controller: camera::CameraController,

    light_uniform: model::LightUniform,
    light_buffer: wgpu::Buffer,
    light_bind_group_layout: BindGroupLayout,
    light_bind_group: BindGroup,

    render_pipeline: RenderPipeline,
    light_render_pipeline: RenderPipeline,

    viewport: Viewport,
    renderer: Renderer,

    events: Vec<Event>,
    uicontrols: UIControls,
    cache: user_interface::Cache,
    modifiers: ModifiersState,
    cursor: mouse::Cursor,
    mouse_pressed: bool,
    clipboard: Clipboard,

    sim_state: SimulationDisplayState,
    reset_requested: bool,
}

impl State {
    pub fn new(window: Window, instances: Vec<super::mediation::Instance>, sphere_size: f32, light_position: [f32; 3]) -> Self {
        let window_arc = Arc::new(window);

        let size = window_arc.inner_size();
        let viewport = Viewport::with_physical_size(
            Size::new(size.width, size.height),
            window_arc.scale_factor(),
        );
        let clipboard = Clipboard::connect(window_arc.clone());

        let instance = Self::create_gpu_instance();
        let surface = instance.create_surface(window_arc.clone()).unwrap();
        let adapter = Self::create_adapter(instance, &surface);
        let (device, queue) = Self::create_device(&adapter);
        let surface_caps = surface.get_capabilities(&adapter);
        let config = Self::create_surface_config(size, surface_caps);
        surface.configure(&device, &config);

        let depth_texture = model::Texture::create_depth_texture(&device, &config, "depth_texture");
        let sphere = model::load_model("./src/gui/model/sphere.obj", &device, sphere_size).unwrap();

        let instance_buffer = Self::create_instance_buffer(&device, &instances);

        let camera = camera::Camera::new((0.0, 5.0, 10.0), cgmath::Deg(-90.0), cgmath::Deg(-20.0));
        let projection = camera::Projection::new(config.width, config.height, cgmath::Deg(45.0), 0.1, 100.0);
        let camera_controller = camera::CameraController::new(4.0, 0.4);

        let mut camera_uniform = camera::CameraUniform::default(); // edit?
        camera_uniform.update_view_proj(&camera, &projection);

        let camera_buffer = Self::create_uniform_buffer(&device, camera_uniform, "Camera Buffer");

        let camera_bind_group_layout = Self::create_bind_group_layout(&device, "camera_bind_group_layout");
        let camera_bind_group = Self::create_bind_group(&device, &camera_bind_group_layout, &camera_buffer, "camera_bind_group");

        let light_uniform = model::LightUniform::new(light_position); // edit?

        let light_buffer = Self::create_uniform_buffer(&device, light_uniform, "Light Buffer");
        let light_bind_group_layout = Self::create_bind_group_layout(&device, "light_bind_group_layout");
        let light_bind_group = Self::create_bind_group(&device, &light_bind_group_layout, &light_buffer, "light_bind_group");

        let render_pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &light_bind_group_layout],
                push_constant_ranges: &[],
            });
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Normal Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
            };
            Self::create_render_pipeline(
                &device,
                &layout,
                config.format,
                Some(model::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc(), model::InstanceRaw::desc()],
                shader,
            )
        };


        let light_render_pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Light Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &light_bind_group_layout],
                push_constant_ranges: &[],
            });
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Light Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("light.wgsl").into()),
            };
            Self::create_render_pipeline(
                &device,
                &layout,
                config.format,
                Some(model::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc()],
                shader,
            )
        };

        // Initialize iced
        let renderer = {
            let engine = Engine::new(
                &adapter,
                device.clone(),
                queue.clone(),
                surface.get_capabilities(&adapter).formats
                .iter()
                .copied()
                .find(wgpu::TextureFormat::is_srgb)
                .or_else(|| {
                    surface.get_capabilities(&adapter).formats.first().copied()
                })
                .expect("Get preferred format"),
                None,
            );

            Renderer::new(engine, Font::default(), Pixels::from(16))
        };

        // Initialize GUI controls
        let controls = UIControls::new();

        Self {
            window: window_arc,
            surface,
            device,
            queue,
            config,
            size,
            depth_texture,
            sphere,
            instances,
            instance_buffer,
            camera,
            projection,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            light_uniform,
            light_buffer,
            light_bind_group_layout,
            light_bind_group,
            render_pipeline,
            light_render_pipeline,
            viewport,
            renderer,
            uicontrols: controls,
            events: Vec::new(),
            cache: user_interface::Cache::new(),
            modifiers: ModifiersState::default(),
            cursor: mouse::Cursor::Unavailable,
            mouse_pressed: false,
            clipboard,
            sim_state: SimulationDisplayState::Paused,
            reset_requested: false,
        }
    }

    fn create_gpu_instance() -> wgpu::Instance {
        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        })
    }

    fn create_adapter(instance: wgpu::Instance, surface: &Surface) -> Adapter {
        instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(surface),
                force_fallback_adapter: false,
            }
        ).block_on().expect("Failed to find an appropriate adapter")
    }

    fn create_device(adapter: &Adapter) -> (Device, Queue) {
        adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                label: Some("Device"),
                // trace: wgpu::Trace::Off,
            },
            None,
        ).block_on().expect("Failed to create device")
    }

    fn create_surface_config(size: winit::dpi::PhysicalSize<u32>, capabilities: SurfaceCapabilities) -> SurfaceConfiguration {
        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result in all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = capabilities.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(capabilities.formats[0]);

        SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            desired_maximum_frame_latency: 2,
            present_mode: PresentMode::AutoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
        }
    }

    fn create_instance_buffer(device: &Device, instances: &[super::mediation::Instance]) -> wgpu::Buffer {
        let instance_data = instances.iter().map(super::mediation::Instance::to_raw).collect::<Vec<_>>();

        device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&instance_data),
                usage: wgpu::BufferUsages::VERTEX,
            }
        )
    }

    fn create_uniform_buffer<T: bytemuck::NoUninit>(device: &Device, uniform: T, label: &str) -> wgpu::Buffer {
        device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&[uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    fn create_bind_group_layout(device: &Device, label: &str) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some(label),
        })
    }

    fn create_bind_group(device: &Device, bind_group_layout: &wgpu::BindGroupLayout, buffer: &wgpu::Buffer, label: &str)
    -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }
            ],
            label: Some(label),
        })
    }

    fn create_render_pipeline(
            device: &Device,
            render_pipeline_layout: &PipelineLayout,
            color_format: wgpu::TextureFormat,
            depth_format: Option<wgpu::TextureFormat>,
            vertex_layouts: &[wgpu::VertexBufferLayout],
            shader: wgpu::ShaderModuleDescriptor,
            // config: &SurfaceConfiguration
        ) -> RenderPipeline {
        let shader = device.create_shader_module(shader);

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: vertex_layouts, // todo Vertex101
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back), // todo
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
                format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        })
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.viewport = Viewport::with_physical_size(
                Size::new(self.size.width, self.size.height),
                self.window.scale_factor(),
            );
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.depth_texture = model::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
            self.projection.resize(new_size.width, new_size.height);
            self.surface.configure(&self.device, &self.config);
            info!("Resized to {:?}", new_size)
        }
    }

    /// modify State's state depending on input
    fn input(&mut self, event: &WindowEvent) -> bool {
        // Map window event to iced event
        if let Some(event) = conversion::window_event(
            event.clone(),
            self.window.scale_factor(),
            self.modifiers,
        ) {
            self.events.push(event);
        }
        // If there are events pending
        if !self.events.is_empty() {
            self.process_events()
        }
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
                self.camera_controller.process_keyboard(*key, *state) || self.process_keyboard(key, state)
            },
            WindowEvent::MouseWheel { delta, .. } => {
                self.camera_controller.process_scroll(delta);
                true
            }
            WindowEvent::MouseInput {
                button: winit::event::MouseButton::Left,
                state,
                ..
            } => {
                self.mouse_pressed = *state == winit::event::ElementState::Pressed;
                true
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor =
                    mouse::Cursor::Available(conversion::cursor_position(
                        *position,
                        self.viewport.scale_factor(),
                    ));
                true
            }
            WindowEvent::ModifiersChanged(new_modifiers) => {
                self.modifiers = new_modifiers.state();
                true
            }
            _ => false,
        }
    }
    fn process_events(&mut self) {
        // Process events
        let mut interface = UserInterface::build(
            self.uicontrols.view(),
            self.viewport.logical_size(),
            std::mem::take(&mut self.cache),
            &mut self.renderer,
        );

        let mut messages = Vec::new();

        let _ = interface.update(
            &self.events,
            self.cursor,
            &mut self.renderer,
            &mut self.clipboard,
            &mut messages,
        );

        self.events.clear();
        self.cache = interface.into_cache();

        // update our UI with any messages
        for message in messages {
            self.uicontrols.update(message);
        }

        // and request a redraw
        self.window.request_redraw();
    }

    fn process_keyboard(&mut self, key: &winit::keyboard::KeyCode, state: &winit::event::ElementState) -> bool {
        match key {
            winit::keyboard::KeyCode::KeyK => {
                if *state == winit::event::ElementState::Pressed {
                    debug!("K pressed");
                    self.sim_state.toggle();
                    true
                } else {
                    false
                }
            }
            winit::keyboard::KeyCode::KeyR => {
                if *state == winit::event::ElementState::Pressed {
                    debug!("R pressed");
                    self.reset_requested = true;
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Update buffer for next rendering step (consider new state of State)
    fn update(&mut self, time_delta_to_last_render_time: std::time::Duration, instances: Option<Vec<super::mediation::Instance>>, messages: Vec<controls::Message>) {
        // Update UI controls
        for message in messages {
            self.uicontrols.update(message);
        }
        // Update instances
        // if let SimulationDisplayState::Resumed = self.sim_state {
            if let Some(instances) = instances {
                self.instances = instances;
                self.instance_buffer = Self::create_instance_buffer(&self.device, &self.instances);
                // self.queue.write_buffer(
                //     &self.instance_buffer,
                //     0,
                //     bytemuck::cast_slice(&[self.instances[0]]),
                // );
            }
        // }
        // Update camera
        self.camera_controller.update_camera(&mut self.camera, time_delta_to_last_render_time);
        self.camera_uniform.update_view_proj(&self.camera, &self.projection);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        // Update the light
        let old_position: cgmath::Vector3<_> = self.light_uniform.position.into();
        self.light_uniform.position = (cgmath::Quaternion::from_axis_angle(
            (0.0, 1.0, 0.0).into(),
            cgmath::Deg(std::f32::consts::PI * time_delta_to_last_render_time.as_secs_f32()),
        ) * old_position)
            .into();
        self.queue.write_buffer(
            &self.light_buffer,
            0,
            bytemuck::cast_slice(&[self.light_uniform]),
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(
                                // wgpu::Color {
                                //     r: 0.9,
                                //     g: 0.9,
                                //     b: 0.9,
                                //     a: 1.0,
                                // }
                                wgpu::Color {
                                    r: self.uicontrols.background_color().r as f64,
                                    g: self.uicontrols.background_color().g as f64,
                                    b: self.uicontrols.background_color().b as f64,
                                    a: self.uicontrols.background_color().a as f64,
                                }
                            ),
                            store: wgpu::StoreOp::Store,
                        },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));

            render_pass.set_pipeline(&self.light_render_pipeline);
            render_pass.draw_light_model(
                &self.sphere,
                &self.camera_bind_group,
                &self.light_bind_group,
            );

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.draw_model_instanced(
                &self.sphere,
                0..self.instances.len() as u32,
                &self.camera_bind_group,
                &self.light_bind_group,
            );
        }
        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));

        // Draw iced on top
        let mut interface = UserInterface::build(
            self.uicontrols.view(),
            self.viewport.logical_size(),
            std::mem::take(&mut self.cache),
            &mut self.renderer,
        );

        let (state, _) = interface.update(
            &[Event::Window(
                window::Event::RedrawRequested(
                    Instant::now(),
                ),
            )],
            self.cursor,
            &mut self.renderer,
            &mut self.clipboard,
            &mut Vec::new(),
        );

        // Update the mouse cursor
        if let user_interface::State::Updated {
            mouse_interaction,
            ..
        } = state
        {
            self.window.set_cursor(
                conversion::mouse_interaction(
                    mouse_interaction,
                ),
            );
        }

        // Draw the interface
        interface.draw(
            &mut self.renderer,
            &Theme::Dark,
            &renderer::Style::default(),
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
            &view,
            &self.viewport,
        );

        // Present the frame
        frame.present();

        Ok(())
    }
}
