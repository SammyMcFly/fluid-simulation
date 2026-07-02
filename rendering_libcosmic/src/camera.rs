//! Camera math and input handling.
//!

use cgmath::{InnerSpace, Matrix4, Point3, Rad, SquareMatrix, Vector3, perspective};
use std::f32::consts::FRAC_PI_2;

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);

const SAFE_FRAC_PI_2: f32 = FRAC_PI_2 - 0.0001;

// ─── Camera ───────────────────────────────────────────────────

/// Camera in standard cartesian coordinates (z-up)
#[derive(Debug, Clone)]
pub struct Camera {
    pub position: Point3<f32>,
    pub yaw: Rad<f32>,
    pub pitch: Rad<f32>,
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

    /// Position in graphics coordinates (y-up, z-forward swap)
    pub fn position_in_graphics_coordinates(&self) -> Point3<f32> {
        Point3::new(self.position.x, self.position.z, -self.position.y)
    }

    pub fn view_matrix(&self) -> Matrix4<f32> {
        let (sin_pitch, cos_pitch) = self.pitch.0.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.0.sin_cos();

        Matrix4::look_to_rh(
            self.position_in_graphics_coordinates(),
            Vector3::new(cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw).normalize(),
            Vector3::unit_y(),
        )
    }
}

// ─── Projection ───────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Projection {
    pub aspect: f32,
    pub fovy: Rad<f32>,
    pub znear: f32,
    pub zfar: f32,
}

impl Projection {
    pub fn new<F: Into<Rad<f32>>>(width: u32, height: u32, fovy: F, znear: f32, zfar: f32) -> Self {
        Self {
            aspect: width as f32 / height.max(1) as f32,
            fovy: fovy.into(),
            znear,
            zfar,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height.max(1) as f32;
    }

    pub fn projection_matrix(&self) -> Matrix4<f32> {
        OPENGL_TO_WGPU_MATRIX * perspective(self.fovy, self.aspect, self.znear, self.zfar)
    }
}

// ─── CameraUniform ────────────────────────────────────────────

/// GPU-ready uniform data. Layout must match the WGSL struct.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub view_position: [f32; 4],
    pub view_proj: [[f32; 4]; 4],
    pub inv_view: [[f32; 4]; 4],
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self {
            view: Matrix4::identity().into(),
            proj: Matrix4::identity().into(),
            view_position: [0.0; 4],
            view_proj: Matrix4::identity().into(),
            inv_view: Matrix4::identity().into(),
        }
    }
}

impl CameraUniform {
    pub fn update(&mut self, camera: &Camera, projection: &Projection) {
        let view = camera.view_matrix();
        let proj = projection.projection_matrix();
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

// ─── CameraController ─────────────────────────────────────────

#[derive(Debug, Clone)]
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

    /// Process a keyboard key press/release.
    /// Uses WASD + Space/Shift. Returns true if the key was handled.
    pub fn process_key(&mut self, key: Key, pressed: bool) -> bool {
        let amount = if pressed { 100.0 } else { 0.0 };
        match key {
            Key::Forward => {
                self.amount_forward = amount;
                true
            }
            Key::Backward => {
                self.amount_backward = amount;
                true
            }
            Key::Left => {
                self.amount_left = amount;
                true
            }
            Key::Right => {
                self.amount_right = amount;
                true
            }
            Key::Up => {
                self.amount_up = amount;
                true
            }
            Key::Down => {
                self.amount_down = amount;
                true
            }
        }
    }

    /// Process mouse movement (dx, dy in pixels)
    pub fn process_mouse_motion(&mut self, dx: f32, dy: f32) {
        let inversion = -1.0;
        self.rotate_horizontal = inversion * dx;
        self.rotate_vertical = inversion * dy;
    }

    /// Process scroll (positive = zoom in, negative = zoom out)
    pub fn process_scroll(&mut self, delta: f32) {
        self.scroll = delta;
    }

    /// Advance the camera state by `dt` seconds
    pub fn update_camera(&mut self, camera: &mut Camera, dt: f32) {
        // Move forward/backward and left/right
        let (yaw_sin, yaw_cos) = camera.yaw.0.sin_cos();
        let forward = Vector3::new(yaw_cos, -yaw_sin, 0.0).normalize();
        let right = Vector3::new(-yaw_sin, -yaw_cos, 0.0).normalize();
        camera.position += forward * (self.amount_forward - self.amount_backward) * self.speed * dt;
        camera.position += right * (self.amount_right - self.amount_left) * self.speed * dt;

        // Scroll zoom
        let (pitch_sin, pitch_cos) = camera.pitch.0.sin_cos();
        let scrollward =
            Vector3::new(-pitch_cos * yaw_cos, pitch_cos * yaw_sin, -pitch_sin).normalize();
        camera.position += scrollward * self.scroll * self.scroll_speed * dt;
        self.scroll = 0.0;

        // Move up/down
        camera.position.z += (self.amount_up - self.amount_down) * self.speed * dt;

        // Rotate
        camera.yaw += Rad(self.rotate_horizontal) * self.sensitivity * dt;
        camera.pitch += Rad(-self.rotate_vertical) * self.sensitivity * dt;

        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;

        // Clamp pitch
        camera.pitch = camera
            .pitch
            .clamp(-Rad(SAFE_FRAC_PI_2), Rad(SAFE_FRAC_PI_2));
    }
}

/// Abstract key actions (decoupled from winit)
#[derive(Debug, Clone, Copy)]
pub enum Key {
    Forward,
    Backward,
    Left,
    Right,
    Up,
    Down,
}

// ─── CameraState ──────────────

/// All camera state needed by the application.
/// No GPU resources – those live in FluidRenderer (Pipeline).
#[derive(Debug, Clone)]
pub struct CameraState {
    pub camera: Camera,
    pub projection: Projection,
    pub controller: CameraController,
    pub uniform: CameraUniform,
    // Stored initial values for reset
    initial_position: Point3<f32>,
    initial_yaw: Rad<f32>,
    initial_pitch: Rad<f32>,
}

impl CameraState {
    pub fn new<Y: Into<Rad<f32>> + Copy, P: Into<Rad<f32>> + Copy, F: Into<Rad<f32>> + Copy>(
        position: (f32, f32, f32),
        yaw: Y,
        pitch: P,
        speed: f32,
        sensitivity: f32,
        scroll_speed: f32,
        fovy: F,
        znear: f32,
        zfar: f32,
        width: u32,
        height: u32,
    ) -> Self {
        let pos = Point3::new(position.0, position.1, position.2);
        let camera = Camera::new(pos, yaw, pitch);
        let projection = Projection::new(width, height, fovy, znear, zfar);
        let controller = CameraController::new(speed, sensitivity, scroll_speed);
        let mut uniform = CameraUniform::default();
        uniform.update(&camera, &projection);

        Self {
            camera,
            projection,
            controller,
            uniform,
            initial_position: pos,
            initial_yaw: yaw.into(),
            initial_pitch: pitch.into(),
        }
    }

    /// Update camera based on controller input, recompute uniform
    pub fn tick(&mut self, dt: f32) {
        self.controller.update_camera(&mut self.camera, dt);
        self.uniform.update(&self.camera, &self.projection);
    }

    /// Resize projection
    pub fn resize(&mut self, width: u32, height: u32) {
        self.projection.resize(width, height);
        self.uniform.update(&self.camera, &self.projection);
    }

    /// Reset to initial state
    pub fn reset(&mut self) {
        self.camera = Camera::new(self.initial_position, self.initial_yaw, self.initial_pitch);
        self.controller = CameraController::new(
            self.controller.speed,
            self.controller.sensitivity,
            self.controller.scroll_speed,
        );
        self.uniform.update(&self.camera, &self.projection);
    }
}

// ─── Trait Rad::clamp helper ──────────────────────────────────

trait ClampRad {
    fn clamp(self, min: Self, max: Self) -> Self;
}

impl ClampRad for Rad<f32> {
    fn clamp(self, min: Rad<f32>, max: Rad<f32>) -> Rad<f32> {
        Rad(self.0.clamp(min.0, max.0))
    }
}
