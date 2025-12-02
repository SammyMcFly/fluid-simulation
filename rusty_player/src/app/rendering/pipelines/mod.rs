//! Pipelines
//!
use iced_wgpu::wgpu;

pub struct Pipelines {
    pub object: wgpu::RenderPipeline,
    pub light: wgpu::RenderPipeline,
}

impl Pipelines {
    pub fn new(
        gpu_context: &super::gpu_context::GpuContext,
        camera: &super::camera::CameraBundle,
        light: &super::lighting::LightBundle,
        depth_format: Option<wgpu::TextureFormat>,
        vertex_layouts: &[wgpu::VertexBufferLayout],
    ) -> Self {
        let object_render_pipeline = Self::create_object_render_pipeline(
            gpu_context,
            camera,
            light,
            depth_format,
            vertex_layouts,
        );
        let light_render_pipeline = Self::create_light_render_pipeline(
            gpu_context,
            camera,
            light,
            depth_format,
            vertex_layouts,
        );

        Self {
            object: object_render_pipeline,
            light: light_render_pipeline,
        }
    }

    fn create_object_render_pipeline(
        gpu_context: &super::gpu_context::GpuContext,
        camera: &super::camera::CameraBundle,
        light: &super::lighting::LightBundle,
        depth_format: Option<wgpu::TextureFormat>,
        vertex_layouts: &[wgpu::VertexBufferLayout],
    ) -> wgpu::RenderPipeline {
        let layout = gpu_context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&camera.bind_group_layout, &light.bind_group_layout],
            push_constant_ranges: &[],
        });
        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("Normal Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        };
        Self::create_render_pipeline(
            gpu_context,
            &layout,
            depth_format,
            vertex_layouts,
            shader,
        )
    }

    fn create_light_render_pipeline(
        gpu_context: &super::gpu_context::GpuContext,
        camera: &super::camera::CameraBundle,
        light: &super::lighting::LightBundle,
        depth_format: Option<wgpu::TextureFormat>,
        vertex_layouts: &[wgpu::VertexBufferLayout],
    ) -> wgpu::RenderPipeline {
        let layout = gpu_context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Light Pipeline Layout"),
            bind_group_layouts: &[&camera.bind_group_layout, &light.bind_group_layout],
            push_constant_ranges: &[],
        });
        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("Light Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("light.wgsl").into()),
        };
        Self::create_render_pipeline(
            gpu_context,
            &layout,
            depth_format,
            vertex_layouts,
            shader,
        )
    }

    fn create_render_pipeline(
        gpu_context: &super::gpu_context::GpuContext,
        render_pipeline_layout: &wgpu::PipelineLayout,
        depth_format: Option<wgpu::TextureFormat>,
        vertex_layouts: &[wgpu::VertexBufferLayout],
        shader: wgpu::ShaderModuleDescriptor,
    ) -> wgpu::RenderPipeline {
        let shader = gpu_context.device.create_shader_module(shader);

        gpu_context.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                    format: gpu_context.config.format,
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
}