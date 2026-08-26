//! Lighting logic.
//!

use cgmath::Rotation3;

/// Light in standard cartesian coordinates (z-up, sim frame).
#[derive(Debug, Clone)]
pub struct Light {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub movement_speed: f32,
}

impl Light {
    pub fn new(position: [f32; 3], color: [f32; 3], movement_speed: f32) -> Self {
        Self {
            position,
            color,
            movement_speed,
        }
    }

    pub fn update_position(&mut self, dt: f32) {
        let old_position: cgmath::Vector3<f32> = self.position.into();
        self.position = (cgmath::Quaternion::from_axis_angle(
            (0.0, 0.0, 1.0).into(),
            cgmath::Deg(std::f32::consts::PI * dt * self.movement_speed),
        ) * old_position)
            .into();
    }
}

/// GPU-ready uniform data. Layout must match the WGSL struct.
///
/// `position` is in SIM frame (z-up), same convention as all other
/// geometry uniforms/buffers. Shaders are responsible for swizzling
/// to render frame (x, z, -y) at the point of use.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightUniform {
    pub position: [f32; 3],
    pub _padding: u32,
    pub color: [f32; 3],
    pub _padding2: u32,
}

impl Default for LightUniform {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            _padding: 0,
            color: [1.0; 3],
            _padding2: 0,
        }
    }
}

impl LightUniform {
    pub fn from_light(light: &Light) -> Self {
        Self {
            position: light.position,
            _padding: 0,
            color: light.color,
            _padding2: 0,
        }
    }

    pub fn update(&mut self, light: &Light) {
        self.position = light.position;
        self.color = light.color;
    }
}

/// All light state needed by the application.
#[derive(Debug, Clone)]
pub struct LightState {
    pub light: Light,
    pub uniform: LightUniform,
    initial_position: [f32; 3],
}

impl LightState {
    pub fn new(position: [f32; 3], color: [f32; 3], movement_speed: f32) -> Self {
        let light = Light::new(position, color, movement_speed);
        let uniform = LightUniform::from_light(&light);
        Self {
            light,
            uniform,
            initial_position: position,
        }
    }

    /// Advance light rotation by dt seconds
    pub fn tick(&mut self, dt: f32) {
        self.light.update_position(dt);
        self.uniform.update(&self.light);
    }

    /// Set light to a specific position (sim frame, z-up)
    pub fn set_position(&mut self, position: [f32; 3]) {
        self.light.position = position;
        self.uniform.update(&self.light);
    }

    /// Reset to initial position
    pub fn reset(&mut self) {
        self.light.position = self.initial_position;
        self.uniform.update(&self.light);
    }
}
