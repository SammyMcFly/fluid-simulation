use cgmath::{InnerSpace, Matrix4, Point3, Rad, SquareMatrix, Vector3, perspective};
use iced_wgpu::wgpu;
use iced_wgpu::wgpu::util::DeviceExt;
use iced_winit::winit;
use iced_winit::winit::dpi::PhysicalPosition;
use iced_winit::winit::event::{DeviceEvent, WindowEvent};
use iced_winit::winit::keyboard::KeyCode;
use std::f32::consts::FRAC_PI_2;

#[cfg(feature = "logging")]
use tracing::debug;

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);

const SAFE_FRAC_PI_2: f32 = FRAC_PI_2 - 0.0001;

/// Camera in standard kartesian coordinates
#[derive(Debug)]
pub struct Camera {
    pub position: Point3<f32>,
    yaw: Rad<f32>,
    pitch: Rad<f32>,
}

impl Camera {
    pub fn new<V: Into<Point3<f32>>, Y: Into<Rad<f32>>, P: Into<Rad<f32>>>(
        position: V,
        yaw: Y,
        pitch: P,
    ) -> Self {
        Self {
            position: position.into(),
            yaw: yaw.into(),
            pitch: pitch.into(),
        }
    }

    pub fn position_in_graphics_coordinates(&self) -> Point3<f32> {
        Point3::new(self.position.x, self.position.z, -self.position.y)
    }

    pub fn calc_matrix(&self) -> Matrix4<f32> {
        let (sin_pitch, cos_pitch) = self.pitch.0.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.0.sin_cos();

        Matrix4::look_to_rh(
            self.position_in_graphics_coordinates(),
            Vector3::new(cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw).normalize(),
            Vector3::unit_y(),
        )
    }
}

/// Camera uniform in graphics coordinates
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    view_position: [f32; 4],
    view_proj: [[f32; 4]; 4],
    inv_view: [[f32; 4]; 4],
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self {
            view: cgmath::Matrix4::identity().into(),
            proj: cgmath::Matrix4::identity().into(),
            view_position: [0.0; 4],
            view_proj: cgmath::Matrix4::identity().into(),
            inv_view: cgmath::Matrix4::identity().into(),
        }
    }
}

impl CameraUniform {
    pub fn update_view_proj(&mut self, camera: &Camera, projection: &Projection) {
        let view = camera.calc_matrix();
        let proj = projection.calc_matrix();
        self.view = view.into();
        self.proj = proj.into();
        self.view_position = camera
            .position_in_graphics_coordinates()
            .to_homogeneous()
            .into();
        self.view_proj = (proj * view).into();
        self.inv_view = view.invert().unwrap_or(Matrix4::identity()).into();
    }
}

pub struct Projection {
    aspect: f32,
    fovy: Rad<f32>,
    znear: f32,
    zfar: f32,
}

impl Projection {
    pub fn new<F: Into<Rad<f32>>>(width: u32, height: u32, fovy: F, znear: f32, zfar: f32) -> Self {
        Self {
            aspect: width as f32 / height as f32,
            fovy: fovy.into(),
            znear,
            zfar,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height as f32;
    }

    pub fn calc_matrix(&self) -> Matrix4<f32> {
        OPENGL_TO_WGPU_MATRIX * perspective(self.fovy, self.aspect, self.znear, self.zfar)
    }
}

#[derive(Debug)]
pub struct CameraController {
    amount_left: f32,
    amount_right: f32,
    amount_forward: f32,
    amount_backward: f32,
    amount_up: f32,
    amount_down: f32,
    rotate_horizontal: f32,
    rotate_vertical: f32,
    scroll: f32,
    speed: f32,
    sensitivity: f32,
    scroll_speed: f32,
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32, scroll_speed: f32) -> Self {
        Self {
            amount_left: 0.0,
            amount_right: 0.0,
            amount_forward: 0.0,
            amount_backward: 0.0,
            amount_up: 0.0,
            amount_down: 0.0,
            rotate_horizontal: 0.0,
            rotate_vertical: 0.0,
            scroll: 0.0,
            speed,
            sensitivity,
            scroll_speed,
        }
    }

    pub fn process_keyboard(&mut self, key: KeyCode, state: winit::event::ElementState) {
        let amount = if state == winit::event::ElementState::Pressed {
            100.0
        } else {
            0.0
        };
        match key {
            KeyCode::KeyW | KeyCode::ArrowUp => {
                self.amount_forward = amount;
                #[cfg(feature = "logging")]
                if state == winit::event::ElementState::Pressed {
                    debug!("Up pressed");
                }
            }
            KeyCode::KeyS | KeyCode::ArrowDown => {
                self.amount_backward = amount;
                #[cfg(feature = "logging")]
                if state == winit::event::ElementState::Pressed {
                    debug!("Down pressed");
                }
            }
            KeyCode::KeyA | KeyCode::ArrowLeft => {
                self.amount_left = amount;
                #[cfg(feature = "logging")]
                if state == winit::event::ElementState::Pressed {
                    debug!("Left pressed");
                }
            }
            KeyCode::KeyD | KeyCode::ArrowRight => {
                self.amount_right = amount;
                #[cfg(feature = "logging")]
                if state == winit::event::ElementState::Pressed {
                    debug!("Right pressed");
                }
            }
            KeyCode::Space => {
                self.amount_up = amount;
                #[cfg(feature = "logging")]
                if state == winit::event::ElementState::Pressed {
                    debug!("Space pressed");
                }
            }
            KeyCode::ShiftLeft => {
                self.amount_down = amount;
                #[cfg(feature = "logging")]
                if state == winit::event::ElementState::Pressed {
                    debug!("Shift pressed");
                }
            }
            _ => (),
        }
    }

    pub fn process_mouse(&mut self, mouse_dx: f64, mouse_dy: f64) {
        // if inverted is set to -1 camera rotation is inverted (if 1 there is no inversion)
        let inversion = -1.;
        self.rotate_horizontal = (inversion * mouse_dx) as f32;
        self.rotate_vertical = (inversion * mouse_dy) as f32;
    }

    pub fn process_scroll(&mut self, delta: &winit::event::MouseScrollDelta) {
        #[cfg(feature = "logging")]
        debug!("Scrolled {:?}", delta);
        self.scroll = match delta {
            // I'm assuming a line is about 100 pixels
            winit::event::MouseScrollDelta::LineDelta(_, scroll) => -scroll * 100.0,
            winit::event::MouseScrollDelta::PixelDelta(PhysicalPosition { y: scroll, .. }) => {
                (-100.0 * (*scroll)) as f32
            }
        };
    }

    pub fn update_camera(&mut self, camera: &mut Camera, time_to_last_update: std::time::Duration) {
        let dt = time_to_last_update.as_secs_f32();

        // Move forward/backward and left/right
        let (yaw_sin, yaw_cos) = camera.yaw.0.sin_cos();
        let forward = Vector3::new(yaw_cos, -yaw_sin, 0.0).normalize();
        let right = Vector3::new(-yaw_sin, -yaw_cos, 0.0).normalize();
        camera.position += forward * (self.amount_forward - self.amount_backward) * self.speed * dt;
        camera.position += right * (self.amount_right - self.amount_left) * self.speed * dt;

        // Move in/out (aka. "zoom")
        // Note: this isn't an actual zoom. The camera's position
        // changes when zooming. I've added this to make it easier
        // to get closer to an object you want to focus on.
        let (pitch_sin, pitch_cos) = camera.pitch.0.sin_cos();
        let scrollward =
            Vector3::new(-pitch_cos * yaw_cos, pitch_cos * yaw_sin, -pitch_sin).normalize();
        camera.position += scrollward * self.scroll * self.scroll_speed * dt;
        self.scroll = 0.0;

        // Move up/down. Since we don't use roll, we can just
        // modify the y coordinate directly.
        camera.position.z += (self.amount_up - self.amount_down) * self.speed * dt;

        // Rotate
        camera.yaw += Rad(self.rotate_horizontal) * self.sensitivity * dt;
        camera.pitch += Rad(-self.rotate_vertical) * self.sensitivity * dt;

        // If process_mouse isn't called every frame, these values
        // will not get set to zero, and the camera will rotate
        // when moving in a non-cardinal direction.
        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;

        // Keep the camera's angle from going too high/low.
        if camera.pitch < -Rad(SAFE_FRAC_PI_2) {
            camera.pitch = -Rad(SAFE_FRAC_PI_2);
        } else if camera.pitch > Rad(SAFE_FRAC_PI_2) {
            camera.pitch = Rad(SAFE_FRAC_PI_2);
        }
    }
}

struct CameraBundleInfo {
    position: Point3<f32>,
    yaw: Rad<f32>,
    pitch: Rad<f32>,
    speed: f32,
    sensitivity: f32,
    scroll_speed: f32,
    fovy: Rad<f32>,
    znear: f32,
    zfar: f32,
}

pub struct CameraBundle {
    pub camera: Camera,
    pub projection: Projection,
    pub controller: CameraController,

    pub uniform: CameraUniform,
    pub buffer: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,

    params: CameraBundleInfo,
}

impl CameraBundle {
    #![allow(clippy::too_many_arguments)]
    pub fn new<
        V: Into<Point3<f32>> + Clone,
        Y: Into<Rad<f32>> + Clone,
        P: Into<Rad<f32>> + Clone,
        F: Into<Rad<f32>> + Clone,
    >(
        gpu_context: &super::gpu_context::GpuContext,
        position: V,
        yaw: Y,
        pitch: P,
        speed: f32,
        sensitivity: f32,
        scroll_speed: f32,
        fovy: F,
        znear: f32,
        zfar: f32,
    ) -> Self {
        let camera = Camera::new(position.clone(), yaw.clone(), pitch.clone());
        let projection = Projection::new(
            gpu_context.config.width,
            gpu_context.config.height,
            fovy.clone(),
            znear,
            zfar,
        );
        let camera_controller = CameraController::new(speed, sensitivity, scroll_speed);

        let mut camera_uniform = CameraUniform::default(); // edit?
        camera_uniform.update_view_proj(&camera, &projection);

        let camera_buffer =
            Self::create_uniform_buffer(gpu_context, camera_uniform, "Camera Buffer");

        let camera_bind_group_layout =
            Self::create_bind_group_layout(gpu_context, "camera_bind_group_layout");
        let camera_bind_group = Self::create_bind_group(
            gpu_context,
            &camera_bind_group_layout,
            &camera_buffer,
            "camera_bind_group",
        );

        Self {
            camera,
            projection,
            controller: camera_controller,
            uniform: camera_uniform,
            buffer: camera_buffer,
            bind_group_layout: camera_bind_group_layout,
            bind_group: camera_bind_group,
            params: CameraBundleInfo {
                position: position.into(),
                yaw: yaw.into(),
                pitch: pitch.into(),
                speed,
                sensitivity,
                scroll_speed,
                fovy: fovy.into(),
                znear,
                zfar,
            },
        }
    }

    fn create_uniform_buffer<T: bytemuck::NoUninit>(
        gpu_context: &super::gpu_context::GpuContext,
        uniform: T,
        label: &str,
    ) -> wgpu::Buffer {
        gpu_context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&[uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    fn create_bind_group_layout(
        gpu_context: &super::gpu_context::GpuContext,
        label: &str,
    ) -> wgpu::BindGroupLayout {
        gpu_context
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

    fn create_bind_group(
        gpu_context: &super::gpu_context::GpuContext,
        bind_group_layout: &wgpu::BindGroupLayout,
        buffer: &wgpu::Buffer,
        label: &str,
    ) -> wgpu::BindGroup {
        gpu_context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout: bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
                label: Some(label),
            })
    }

    pub fn process_window_event(&mut self, event: &winit::event::WindowEvent) {
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
                self.controller.process_keyboard(*key, *state);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.controller.process_scroll(delta);
            }
            _ => (),
        }
    }

    pub fn process_device_event(
        &mut self,
        event: &winit::event::DeviceEvent,
        mouse_right_button_pressed: bool,
    ) {
        #[allow(clippy::single_match)]
        match event {
            DeviceEvent::MouseMotion { delta } => {
                if mouse_right_button_pressed {
                    self.controller.process_mouse(delta.0, delta.1);
                }
            }
            _ => (),
        }
    }

    pub fn update(
        &mut self,
        gpu_context: &super::gpu_context::GpuContext,
        time_to_last_update: std::time::Duration,
    ) {
        self.controller
            .update_camera(&mut self.camera, time_to_last_update);
        self.uniform
            .update_view_proj(&self.camera, &self.projection);
        gpu_context
            .queue
            .write_buffer(&self.buffer, 0, bytemuck::cast_slice(&[self.uniform]));
    }

    pub fn reset(&mut self, gpu_context: &super::gpu_context::GpuContext) {
        self.camera = Camera::new(self.params.position, self.params.yaw, self.params.pitch);
        self.projection = Projection::new(
            gpu_context.config.width,
            gpu_context.config.height,
            self.params.fovy,
            self.params.znear,
            self.params.zfar,
        );
        self.controller = CameraController::new(
            self.params.speed,
            self.params.sensitivity,
            self.params.scroll_speed,
        );

        self.uniform = CameraUniform::default(); // edit?
        self.uniform
            .update_view_proj(&self.camera, &self.projection);

        self.buffer = Self::create_uniform_buffer(gpu_context, self.uniform, "Camera Buffer");

        self.bind_group_layout =
            Self::create_bind_group_layout(gpu_context, "camera_bind_group_layout");
        self.bind_group = Self::create_bind_group(
            gpu_context,
            &self.bind_group_layout,
            &self.buffer,
            "camera_bind_group",
        );
    }
}
