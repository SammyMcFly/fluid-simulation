//! Pipelines
//!
use crate::model::VertexBufferLayout;
use iced_wgpu::wgpu;

pub struct Pipelines {
    // pub object: wgpu::RenderPipeline,
    pub light: wgpu::RenderPipeline,
    pub particle: wgpu::RenderPipeline,
    pub mesh_opaque: wgpu::RenderPipeline,
    pub mesh_transparent: wgpu::RenderPipeline,
}

impl Pipelines {
    pub fn new(
        gpu_context: &super::gpu_context::GpuContext,
        camera: &super::camera::CameraBundle,
        light: &super::lighting::LightBundle,
        depth_format: Option<wgpu::TextureFormat>,
    ) -> Self {
        // let object_render_pipeline = Self::create_object_render_pipeline(
        //     gpu_context,
        //     camera,
        //     light,
        //     depth_format,
        // );
        let light_render_pipeline =
            Self::create_light_render_pipeline(gpu_context, camera, light, depth_format);
        let particle_pipeline =
            Self::create_particle_pipeline(gpu_context, camera, light, depth_format);
        let mesh_opaque =
            Self::create_mesh_pipeline(gpu_context, camera, light, depth_format, false);
        let mesh_transparent =
            Self::create_mesh_pipeline(gpu_context, camera, light, depth_format, true);

        Self {
            // object: object_render_pipeline,
            light: light_render_pipeline,
            particle: particle_pipeline,
            mesh_opaque,
            mesh_transparent,
        }
    }

    // fn create_object_render_pipeline(
    //     gpu_context: &super::gpu_context::GpuContext,
    //     camera: &super::camera::CameraBundle,
    //     light: &super::lighting::LightBundle,
    //     depth_format: Option<wgpu::TextureFormat>,
    // ) -> wgpu::RenderPipeline {
    //     let layout = gpu_context
    //         .device
    //         .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
    //             label: Some("Render Pipeline Layout"),
    //             bind_group_layouts: &[&camera.bind_group_layout, &light.bind_group_layout],
    //             push_constant_ranges: &[],
    //         });
    //     let shader = wgpu::ShaderModuleDescriptor {
    //         label: Some("Normal Shader"),
    //         source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    //     };
    //     Self::create_render_pipeline(
    //         gpu_context, &layout,
    //         depth_format,
    //         &[
    //             super::model::ModelVertex::desc(),
    //             super::model::InstanceRaw::desc(),
    //         ],
    //         shader,
    //     )
    // }

    fn create_light_render_pipeline(
        gpu_context: &super::gpu_context::GpuContext,
        camera: &super::camera::CameraBundle,
        light: &super::lighting::LightBundle,
        depth_format: Option<wgpu::TextureFormat>,
    ) -> wgpu::RenderPipeline {
        let layout = gpu_context
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
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
            &[super::model::ModelVertex::desc()],
            shader,
        )
    }

    fn create_particle_pipeline(
        gpu_context: &super::gpu_context::GpuContext,
        camera: &super::camera::CameraBundle,
        light: &super::lighting::LightBundle,
        depth_format: Option<wgpu::TextureFormat>,
    ) -> wgpu::RenderPipeline {
        let layout = gpu_context
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Particle Pipeline Layout"),
                bind_group_layouts: &[&camera.bind_group_layout, &light.bind_group_layout],
                push_constant_ranges: &[],
            });

        // let shader = gpu_context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        //     label: Some("Particle Impostor Shader"),
        //     source: wgpu::ShaderSource::Wgsl(
        //         include_str!("particle_impostor.wgsl").into()
        //     ),
        // });
        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("Particle Impostor Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("particle_impostor.wgsl").into()),
        };

        Self::create_render_pipeline(
            gpu_context,
            &layout,
            depth_format,
            &[
                <super::instances::BillboardInstanceRaw as super::model::VertexBufferLayout>::desc(
                ),
            ],
            shader,
        )
    }

    fn create_mesh_pipeline(
        gpu_context: &super::gpu_context::GpuContext,
        camera: &super::camera::CameraBundle,
        light: &super::lighting::LightBundle,
        depth_format: Option<wgpu::TextureFormat>,
        transparent: bool,
    ) -> wgpu::RenderPipeline {
        let layout = gpu_context
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Mesh Pipeline Layout"),
                bind_group_layouts: &[&camera.bind_group_layout, &light.bind_group_layout],
                push_constant_ranges: &[],
            });

        let shader_src = if transparent {
            include_str!("mesh_transparent.wgsl")
        } else {
            include_str!("mesh_opaque.wgsl")
        };

        let shader = gpu_context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(if transparent {
                    "Mesh Transparent Shader"
                } else {
                    "Mesh Opaque Shader"
                }),
                source: wgpu::ShaderSource::Wgsl(shader_src.into()),
            });

        let blend = if transparent {
            Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
            })
        } else {
            Some(wgpu::BlendState::REPLACE)
        };

        gpu_context
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Mesh Render Pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[super::model::ColoredMeshVertex::desc()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: gpu_context.config.format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    // Disable back-face culling for transparent so both sides render
                    // cull_mode: if transparent { None } else { Some(wgpu::Face::Back) },
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
                    format,
                    // Transparent: read depth but don't write (allows particles behind to show)
                    depth_write_enabled: !transparent,
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

    fn create_render_pipeline(
        gpu_context: &super::gpu_context::GpuContext,
        render_pipeline_layout: &wgpu::PipelineLayout,
        depth_format: Option<wgpu::TextureFormat>,
        vertex_layouts: &[wgpu::VertexBufferLayout],
        shader: wgpu::ShaderModuleDescriptor,
    ) -> wgpu::RenderPipeline {
        let shader = gpu_context.device.create_shader_module(shader);

        gpu_context
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
