//! Lighting module
//!
use iced_wgpu::wgpu;
use iced_wgpu::wgpu::util::DeviceExt;
use cgmath::Rotation3;



/// Light in standard kartesian coordinates
#[derive(Debug, Copy, Clone)]
pub struct Light {
    pub position: [f32; 3],
    pub color: [f32; 3],
    movement_speed: f32,
}

impl Light {
    pub fn new(position: [f32; 3], color: Option<[f32; 3]>, movement_speed: f32) -> Self {
        let color = color.unwrap_or([1.; 3]);
        Self {
            position,
            color,
            movement_speed,
        }
    }

    pub fn position_in_graphics_coordinates(&self) -> [f32; 3] {
        [self.position[0], self.position[2], -self.position[1]]
    }

    pub fn update_position(&mut self, time_to_last_update: std::time::Duration) {
        let old_position: cgmath::Vector3<_> = self.position.into();
        self.position = (cgmath::Quaternion::from_axis_angle(
            (0.0, 0.0, 1.0).into(),
            cgmath::Deg(std::f32::consts::PI * time_to_last_update.as_secs_f32()*self.movement_speed),
        ) * old_position)
            .into();
    }
}

/// Light uniform in graphics coordinates
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightUniform {
    pub position: [f32; 3],
    // Due to uniforms requiring 16 byte (4 float) spacing, we need to use a padding field here
    pub _padding: u32,
    pub color: [f32; 3],
    // Due to uniforms requiring 16 byte (4 float) spacing, we need to use a padding field here
    pub _padding2: u32,
}

impl Default for LightUniform {
    fn default() -> Self {
        Self {
            position: [0.; 3],
            _padding: 0,
            color: [1.; 3],
            _padding2: 0,
        }
    }
}

impl LightUniform {
    pub fn update(&mut self, light: &Light) {
        self.position = light.position_in_graphics_coordinates();
        self.color = light.color;
    }
}

pub struct LightBundle {
    pub light: Light,
    pub uniform: LightUniform,
    pub buffer: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,

    // pub rotation_speed: f32,
    // pub radius: f32,
}

impl LightBundle {
    pub fn new(
        gpu_context: &super::gpu_context::GpuContext,
        light_position: [f32; 3],
        light_color: Option<[f32; 3]>,
        light_movement_speed: f32,
    ) -> Self {
        let light = Light::new(light_position, light_color, light_movement_speed);
        let mut light_uniform = LightUniform::default();
        light_uniform.update(&light);
        let light_buffer = Self::create_uniform_buffer(gpu_context, light_uniform, "Light Buffer");
        let light_bind_group_layout = Self::create_bind_group_layout(gpu_context, "light_bind_group_layout");
        let light_bind_group = Self::create_bind_group(gpu_context, &light_bind_group_layout, &light_buffer, "light_bind_group");

        Self {
            light,
            uniform: light_uniform,
            buffer: light_buffer,
            bind_group_layout: light_bind_group_layout,
            bind_group: light_bind_group,
        }
    }

    fn create_uniform_buffer<T: bytemuck::NoUninit>(
        gpu_context: &super::gpu_context::GpuContext,
        uniform: T,
        label: &str,
    ) -> wgpu::Buffer {
        gpu_context.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&[uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    fn create_bind_group_layout(
        gpu_context: &super::gpu_context::GpuContext,
        label: &str,
    ) -> wgpu::BindGroupLayout {
        gpu_context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
    )
    -> wgpu::BindGroup {
        gpu_context.device.create_bind_group(&wgpu::BindGroupDescriptor {
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

    pub fn set_light(
        &mut self,
        gpu_context: &super::gpu_context::GpuContext,
        light_position: [f32; 3],
    ) {
        self.light = Light::new(light_position, Some(self.light.color), self.light.movement_speed);
        self.uniform.update(&self.light);
        gpu_context.queue.write_buffer(
            &self.buffer,
            0,
            bytemuck::cast_slice(&[self.uniform]),
        );
    }

    pub fn update(
        &mut self,
        gpu_context: &super::gpu_context::GpuContext,
        time_to_last_update: std::time::Duration,
    ) {
        self.light.update_position(time_to_last_update);
        self.uniform.update(&self.light);
        gpu_context.queue.write_buffer(
            &self.buffer,
            0,
            bytemuck::cast_slice(&[self.uniform]),
        );
    }

    // pub fn reset(&mut self, gpu_context: &super::gpu_context::GpuContext,) {}
}