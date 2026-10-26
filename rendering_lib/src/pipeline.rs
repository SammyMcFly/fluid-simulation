//! FluidRenderer – all GPU resources that persist between frames.
//! Implements shader::Pipeline (created once by cosmic/iced).

use cosmic::iced::wgpu;
use cosmic::iced::wgpu::util::DeviceExt;
use cosmic::iced::widget::shader;
use std::fmt::Debug;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use crate::camera::CameraUniform;
use crate::lighting::LightUniform;
use crate::model::{ColoredMeshVertex, ModelVertex, VertexBufferLayout};

// ─── Depth Texture ────────────────────────────────────────────

pub struct DepthTexture {
    pub view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
}

impl DepthTexture {
    pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::FORMAT,
            // Needs TEXTURE_BINDING in addition to RENDER_ATTACHMENT: depth
            // peeling samples this texture (as the "opaque depth" reference,
            // or as a ping-pong "previous layer" depth) in a later pass,
            // after having used it as a depth-stencil attachment.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            view,
            width,
            height,
        }
    }
}

// ─── Depth Peeling: Accumulation Texture ──────────────────────

/// Off-screen color target that depth-peeled transparent layers are
/// composited into (via the "under" blend operator), one peel layer at a
/// time, before being blended onto the real color target in a final pass.
pub struct AccumTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
}

impl AccumTexture {
    pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Peel Accumulation Texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            texture,
            view,
            width,
            height,
        }
    }
}

/// Number of depth-peel layers rendered per frame for transparent boundary
/// meshes / fluid mesh. Each layer "peels off" the next-nearest unpeeled
/// surface at every pixel. For a scene of a container boundary (2 layers:
/// near + far wall) plus a handful of inner rigid bodies (2 layers each),
/// a small fixed count comfortably covers the expected overlap depth.
pub const DEPTH_PEEL_LAYERS: u32 = 6;

// ─── Scene GPU Buffers (uploaded per frame in prepare()) ──────

/// GPU buffers for the current scene (particles, meshes, etc.)
/// These change when new simulation data arrives.
#[derive(Default)]
pub struct SceneGpuBuffers {
    pub particle_buffer: Option<wgpu::Buffer>,
    pub particle_count: u32,
    pub boundary_particle_buffer: Option<wgpu::Buffer>,
    pub boundary_particle_count: u32,
    pub mesh_vertex_buffer: Option<wgpu::Buffer>,
    pub mesh_index_buffer: Option<wgpu::Buffer>,
    pub mesh_index_count: u32,
    pub mesh_transparent: bool,
    pub boundary_meshes: Vec<BoundaryMesh>,
    pub sensor_plane_vertex_buffer: Option<wgpu::Buffer>,
    pub sensor_plane_index_buffer: Option<wgpu::Buffer>,
    pub sensor_plane_index_count: u32,
}

// ─── Light Indicator Mesh ─────────────────────────────────────

pub struct LightMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
}

// --- Boundary Mesh --------------------------------------------

pub struct BoundaryMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub instance_buffer: wgpu::Buffer,
}

// ─── Screenshot -------------──────────────────────────────────

/// Screenshot capture state machine
#[derive(Debug, Default)]
pub enum ScreenshotState {
    /// No screenshot in progress
    #[default]
    Idle,
    /// copy_texture_to_buffer was issued in render(), awaiting next prepare() to map
    CopyIssued { pending: PendingScreenshot },
    /// map_async was called, waiting for callback to fire
    MapPending { pending: PendingScreenshot },
}

#[derive(Debug)]
pub struct PendingScreenshot {
    pub width: u32,
    pub height: u32,
    pub target: PendingScreenshotTarget,
    pub id: u64,
}

#[derive(Debug, Clone)]
pub enum PendingScreenshotTarget {
    Directory {
        frame_index: usize,
        directory: std::path::PathBuf,
        overwrite: bool,
    },
    ExplicitPath {
        path: PathBuf,
    },
}

pub trait ScreenshotCommand: Debug + Send + 'static {
    fn write_rendering(
        data: Vec<u8>,
        width: u32,
        height: u32,
        frame_index: usize,
        directory: std::path::PathBuf,
        overwrite: bool,
    ) -> Self;

    fn save_screenshot_to_file(data: Vec<u8>, width: u32, height: u32, file_path: PathBuf) -> Self;
}

// ─── FluidRenderer ────────────────────────────────────────────

pub struct SimulationRenderer {
    // Camera
    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group_layout: wgpu::BindGroupLayout,
    pub camera_bind_group: wgpu::BindGroup,
    // Light
    pub light_buffer: wgpu::Buffer,
    pub light_bind_group_layout: wgpu::BindGroupLayout,
    pub light_bind_group: wgpu::BindGroup,
    // Pipelines
    pub particle_pipeline: wgpu::RenderPipeline,
    pub particle_transparent_pipeline: wgpu::RenderPipeline,
    pub mesh_opaque_pipeline: wgpu::RenderPipeline,
    pub mesh_unlit_pipeline: wgpu::RenderPipeline,
    pub light_pipeline: wgpu::RenderPipeline,
    /// The texture format used by all pipelines (matches swapchain)
    pub surface_format: wgpu::TextureFormat,
    // Depth
    pub depth_texture: Option<DepthTexture>,
    // Light indicator mesh
    pub light_mesh: LightMesh,
    // Scene data (uploaded in prepare)
    pub scene: SceneGpuBuffers,

    // ─── Depth peeling (transparent boundary + fluid meshes) ──────
    /// Bind group layout shared by all peel pipelines: light uniform (binding
    /// 0) plus the two depth textures each peel shader needs to sample
    /// (binding 1: previous-layer depth, binding 2: opaque-geometry depth).
    /// Combining these into ONE bind group is required because the actual
    /// device limit here is `max_bind_groups: 2` (overridden by iced_wgpu,
    /// below wgpu's own default of 4) -- group 0 is already taken by the
    /// camera, leaving exactly one more group for everything else a peel
    /// pipeline needs.
    pub peel_light_bind_group_layout: wgpu::BindGroupLayout,
    pub peel_depth_a: Option<DepthTexture>,
    pub peel_depth_b: Option<DepthTexture>,
    pub accum_texture: Option<AccumTexture>,
    /// Peel bind groups for the on-screen target. "_a" is used while
    /// writing peel_depth_a (so its `prev_depth` binding is peel_depth_b),
    /// and vice versa for "_b" -- see `make_peel_light_bind_groups`.
    pub peel_light_bind_group_a: Option<wgpu::BindGroup>,
    pub peel_light_bind_group_b: Option<wgpu::BindGroup>,
    /// Same, but referencing `offscreen_depth` as the opaque-depth source,
    /// used when a screenshot is being captured.
    pub peel_light_bind_group_offscreen_a: Option<wgpu::BindGroup>,
    pub peel_light_bind_group_offscreen_b: Option<wgpu::BindGroup>,
    pub boundary_peel_first_pipeline: wgpu::RenderPipeline,
    pub boundary_peel_next_pipeline: wgpu::RenderPipeline,
    pub fluid_peel_first_pipeline: wgpu::RenderPipeline,
    pub fluid_peel_next_pipeline: wgpu::RenderPipeline,
    pub composite_bind_group_layout: wgpu::BindGroupLayout,
    pub composite_bind_group: Option<wgpu::BindGroup>,
    pub composite_pipeline: wgpu::RenderPipeline,
    /// Temporary per-layer resolve target: one peel layer's correctly
    /// Z-buffered (REPLACE-blended) color, before being folded into
    /// `accum_texture` via `layer_composite_pipeline`.
    pub layer_color: Option<AccumTexture>,
    pub layer_composite_bind_group: Option<wgpu::BindGroup>,
    pub layer_composite_pipeline: wgpu::RenderPipeline,

    // Screenshot infrastructure
    pub offscreen_texture: Option<wgpu::Texture>,
    pub offscreen_view: Option<wgpu::TextureView>,
    pub offscreen_depth: Option<DepthTexture>,
    pub staging_buffer: Option<wgpu::Buffer>,
    pub staging_mapped: Arc<AtomicBool>,
    pub staging_padded_bpr: u32,
    pub screenshot_width: u32,
    pub screenshot_height: u32,
    pub screenshot_state: Mutex<ScreenshotState>,
    /// ID of the last screenshot request that was fully captured and dispatched.
    /// Prevents re-capturing the same still-pending `ScreenshotRequest` if the
    /// state machine finishes (returns to `Idle`) before the app has observed
    /// completion and issued a new request.
    pub last_completed_screenshot_id: Option<u64>,
}

impl shader::Pipeline for SimulationRenderer {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        // ─── Bind Group Layouts ───────────────────────────────

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Camera BGL"),
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
            });

        let light_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Light BGL"),
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
            });

        // ─── Uniform Buffers ──────────────────────────────────

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera BG"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let light_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light Buffer"),
            size: std::mem::size_of::<LightUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Light BG"),
            layout: &light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_buffer.as_entire_binding(),
            }],
        });

        // ─── Particle Pipeline ────────────────────────────────

        let particle_pipeline = Self::create_particle_pipeline(
            device,
            format,
            &camera_bind_group_layout,
            &light_bind_group_layout,
        );
        let particle_transparent_pipeline = Self::create_particle_pipeline_transparent(
            device,
            format,
            &camera_bind_group_layout,
            &light_bind_group_layout,
        );

        // ─── Mesh Pipelines ───────────────────────────────────

        let mesh_opaque_pipeline = Self::create_mesh_pipeline(
            device,
            format,
            &camera_bind_group_layout,
            &light_bind_group_layout,
            Some(wgpu::Face::Back),
            false,
            false,
            false,
        );
        let mesh_unlit_pipeline =
            Self::create_unlit_mesh_pipeline(device, format, &camera_bind_group_layout);

        // ─── Depth peeling setup ───────────────────────────────

        let peel_light_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Peel Light+Depth BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        let composite_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Composite BGL"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });

        let peel_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Peel Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout, &peel_light_bind_group_layout],
            immediate_size: 0,
        });

        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Composite Pipeline Layout"),
                bind_group_layouts: &[&composite_bind_group_layout],
                immediate_size: 0,
            });

        let boundary_peel_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Boundary Peel Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/mesh_peel_boundary.wgsl").into(),
            ),
        });
        let boundary_peel_first_pipeline = Self::build_peel_pipeline(
            device,
            &peel_pipeline_layout,
            &boundary_peel_shader,
            "fs_first",
            true,
            "Boundary Peel First",
        );
        let boundary_peel_next_pipeline = Self::build_peel_pipeline(
            device,
            &peel_pipeline_layout,
            &boundary_peel_shader,
            "fs_next",
            true,
            "Boundary Peel Next",
        );

        let fluid_peel_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fluid Peel Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mesh_peel_fluid.wgsl").into()),
        });
        let fluid_peel_first_pipeline = Self::build_peel_pipeline(
            device,
            &peel_pipeline_layout,
            &fluid_peel_shader,
            "fs_first",
            false,
            "Fluid Peel First",
        );
        let fluid_peel_next_pipeline = Self::build_peel_pipeline(
            device,
            &peel_pipeline_layout,
            &fluid_peel_shader,
            "fs_next",
            false,
            "Fluid Peel Next",
        );

        let composite_pipeline = Self::create_composite_pipeline(
            device,
            &composite_pipeline_layout,
            format,
            Self::over_blend(),
            "Composite Peel Pipeline (accum -> screen)",
        );
        let layer_composite_pipeline = Self::create_composite_pipeline(
            device,
            &composite_pipeline_layout,
            AccumTexture::FORMAT,
            Self::under_blend(),
            "Composite Peel Pipeline (layer -> accum)",
        );

        // ─── Light Pipeline ───────────────────────────────────

        let light_pipeline = Self::create_light_pipeline(
            device,
            format,
            &camera_bind_group_layout,
            &light_bind_group_layout,
        );

        // ─── Light Indicator Mesh ─────────────────────────────

        let light_mesh = Self::load_sphere_mesh(device);

        Self {
            camera_buffer,
            camera_bind_group_layout,
            camera_bind_group,
            light_buffer,
            light_bind_group_layout,
            light_bind_group,
            particle_pipeline,
            particle_transparent_pipeline,
            mesh_opaque_pipeline,
            mesh_unlit_pipeline,
            light_pipeline,
            surface_format: format,
            depth_texture: None,
            light_mesh,
            scene: SceneGpuBuffers::default(),

            peel_light_bind_group_layout,
            peel_depth_a: None,
            peel_depth_b: None,
            accum_texture: None,
            peel_light_bind_group_a: None,
            peel_light_bind_group_b: None,
            peel_light_bind_group_offscreen_a: None,
            peel_light_bind_group_offscreen_b: None,
            boundary_peel_first_pipeline,
            boundary_peel_next_pipeline,
            fluid_peel_first_pipeline,
            fluid_peel_next_pipeline,
            composite_bind_group_layout,
            composite_bind_group: None,
            composite_pipeline,
            layer_color: None,
            layer_composite_bind_group: None,
            layer_composite_pipeline,

            offscreen_texture: None,
            offscreen_view: None,
            offscreen_depth: None,
            staging_buffer: None,
            staging_mapped: Arc::new(AtomicBool::new(false)),
            staging_padded_bpr: 0,
            screenshot_width: 0,
            screenshot_height: 0,
            screenshot_state: Mutex::new(ScreenshotState::default()),
            last_completed_screenshot_id: None,
        }
    }
}

// ─── Pipeline Creation Helpers ────────────────────────────────

impl SimulationRenderer {
    fn create_particle_pipeline(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        camera_bgl: &wgpu::BindGroupLayout,
        light_bgl: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Impostor Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/particle_impostor.wgsl").into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Particle Pipeline Layout"),
            bind_group_layouts: &[camera_bgl, light_bgl],
            immediate_size: 0,
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Particle Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[BillboardInstance::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DepthTexture::FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    }

    fn create_particle_pipeline_transparent(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        camera_bgl: &wgpu::BindGroupLayout,
        light_bgl: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Impostor Shader (Transparent)"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/particle_impostor.wgsl").into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Particle Transparent Pipeline Layout"),
            bind_group_layouts: &[camera_bgl, light_bgl],
            immediate_size: 0,
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Particle Transparent Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[BillboardInstance::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState {
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
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DepthTexture::FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    }

    fn create_mesh_pipeline(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        camera_bgl: &wgpu::BindGroupLayout,
        light_bgl: &wgpu::BindGroupLayout,
        cull_mode: Option<wgpu::Face>,
        transparent: bool,
        fluid: bool,
        instanced: bool,
    ) -> wgpu::RenderPipeline {
        let shader_src = if transparent && fluid {
            include_str!("shaders/mesh_transparent_fluid.wgsl")
        } else if transparent {
            include_str!("shaders/mesh_transparent.wgsl")
        } else {
            include_str!("shaders/mesh_opaque.wgsl")
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(if transparent {
                "Mesh Transparent"
            } else {
                "Mesh Opaque"
            }),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Mesh Pipeline Layout"),
            bind_group_layouts: &[camera_bgl, light_bgl],
            immediate_size: 0,
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

        let colored_desc = ColoredMeshVertex::desc();
        let pose_desc = MeshPoseInstance::layout();
        let buffers: Vec<wgpu::VertexBufferLayout> = if instanced {
            vec![colored_desc, pose_desc]
        } else {
            vec![colored_desc]
        };

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Mesh Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &buffers,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DepthTexture::FORMAT,
                depth_write_enabled: !transparent,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    }

    fn create_unlit_mesh_pipeline(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        camera_bgl: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Mesh Unlit"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mesh_unlit.wgsl").into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Mesh Unlit Pipeline Layout"),
            bind_group_layouts: &[camera_bgl], // kein Light-BGL
            immediate_size: 0,
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Mesh Unlit Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[ColoredMeshVertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // double-sided (Ebene)
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DepthTexture::FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    }

    fn create_light_pipeline(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        camera_bgl: &wgpu::BindGroupLayout,
        light_bgl: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Light Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/light.wgsl").into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Light Pipeline Layout"),
            bind_group_layouts: &[camera_bgl, light_bgl],
            immediate_size: 0,
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Light Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[ModelVertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DepthTexture::FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    }

    fn load_sphere_mesh(device: &wgpu::Device) -> LightMesh {
        const SPHERE_OBJ: &str = include_str!("model/sphere.obj");
        let mut cursor = std::io::Cursor::new(SPHERE_OBJ.as_bytes());
        let (models, _) = tobj::load_obj_buf(
            &mut cursor,
            &tobj::LoadOptions {
                triangulate: true,
                single_index: true,
                ..Default::default()
            },
            |_| Ok((Vec::new(), ahash::AHashMap::new())),
        )
        .expect("Failed to load sphere.obj");

        let m = &models[0].mesh;
        let scaling = 1.0;
        let vertices: Vec<ModelVertex> = (0..m.positions.len() / 3)
            .map(|i| ModelVertex {
                position: [
                    m.positions[i * 3] / 1.78 * scaling,
                    (m.positions[i * 3 + 1] - 0.89) / 1.78 * scaling,
                    m.positions[i * 3 + 2] / 1.78 * scaling,
                ],
                normal: if m.normals.is_empty() {
                    [0.0, 0.0, 0.0]
                } else {
                    [m.normals[i * 3], m.normals[i * 3 + 1], m.normals[i * 3 + 2]]
                },
            })
            .collect();

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light Sphere VB"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light Sphere IB"),
            contents: bytemuck::cast_slice(&m.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        LightMesh {
            vertex_buffer,
            index_buffer,
            num_indices: m.indices.len() as u32,
        }
    }

    fn build_peel_pipeline(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        shader: &wgpu::ShaderModule,
        fs_entry: &str,
        instanced: bool,
        label: &str,
    ) -> wgpu::RenderPipeline {
        let colored_desc = ColoredMeshVertex::desc();
        let pose_desc = MeshPoseInstance::layout();
        let buffers: Vec<wgpu::VertexBufferLayout> = if instanced {
            vec![colored_desc, pose_desc]
        } else {
            vec![colored_desc]
        };

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &buffers,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some(fs_entry),
                targets: &[Some(wgpu::ColorTargetState {
                    format: AccumTexture::FORMAT,
                    // REPLACE, not "under"-blend: within a single peel layer,
                    // multiple triangles of the SAME non-convex mesh can
                    // overlap at a pixel (e.g. at a fold/crease). The depth
                    // test alone only decides whether a fragment is written
                    // at all -- it does NOT prevent an earlier-arriving,
                    // farther fragment from already having been blended in
                    // before a later-arriving, nearer one overwrites the
                    // depth buffer. With an additive "under" blend, both
                    // fragments' colors end up summed. With REPLACE, each
                    // passing fragment fully overwrites the previous one, so
                    // -- exactly like conventional opaque Z-buffered
                    // rendering -- the final stored color is always just the
                    // single nearest surviving fragment, regardless of draw
                    // order. This resolves ONE correct color per pixel for
                    // this layer; cross-layer "under" compositing happens
                    // separately, in a second pass (see
                    // `composite_layer_into_accum` in primitive.rs).
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                // No culling: with depth peeling, front and back faces of a
                // single (even convex) mesh are naturally captured as separate
                // peel layers via their different depths -- the old
                // backface/frontface two-pass trick is no longer needed.
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DepthTexture::FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    }

    /// Builds a fullscreen-triangle "composite a texture onto the current
    /// target" pipeline. Used for two distinct purposes with different
    /// blend math and target formats:
    /// - `layer_composite_pipeline`: composites ONE resolved peel layer
    ///   (`layer_color`, straight "replace"-resolved, see
    ///   `build_peel_pipeline`) onto the running `accum_texture`, via the
    ///   "under" operator -- this is where cross-layer front-to-back
    ///   compositing actually happens now.
    /// - `composite_pipeline`: composites the final `accum_texture` onto
    ///   the real color target, via the standard "over" operator.
    fn create_composite_pipeline(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        target_format: wgpu::TextureFormat,
        blend: wgpu::BlendState,
        label: &str,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Composite Peel Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/composite_peel.wgsl").into()),
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    }

    /// "Under" blend: `dst_new = src*(1-dst.a) + dst*1`, with `src`
    /// premultiplied by its own alpha. Used to fold one resolved peel layer
    /// into the running accumulation, front-to-back.
    fn under_blend() -> wgpu::BlendState {
        wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        }
    }

    /// "Over" blend: standard straight-alpha compositing of the final
    /// accumulated result onto the real color target.
    fn over_blend() -> wgpu::BlendState {
        wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        }
    }

    /// (Re)creates the peel ping-pong depth textures, accumulation texture
    /// and the composite bind group whenever the viewport size changes.
    /// Does NOT create the online `peel_light_bind_group_a/b` -- those
    /// additionally depend on which opaque-depth texture (on-screen vs.
    /// offscreen) they reference, and are built via
    /// `make_peel_light_bind_groups` in `prepare()`.
    pub fn ensure_peel_resources(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let needs_recreate = self
            .peel_depth_a
            .as_ref()
            .is_none_or(|d| d.width != width || d.height != height);

        if needs_recreate {
            self.peel_depth_a = Some(DepthTexture::new(device, width, height));
            self.peel_depth_b = Some(DepthTexture::new(device, width, height));
            self.accum_texture = Some(AccumTexture::new(device, width, height));
            self.layer_color = Some(AccumTexture::new(device, width, height));

            self.composite_bind_group =
                Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Composite BG (accum)"),
                    layout: &self.composite_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &self.accum_texture.as_ref().unwrap().view,
                        ),
                    }],
                }));
            self.layer_composite_bind_group =
                Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Composite BG (layer)"),
                    layout: &self.composite_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &self.layer_color.as_ref().unwrap().view,
                        ),
                    }],
                }));
        }
    }

    /// Builds the two ping-pong "light + peel depth textures" bind groups
    /// for a given opaque-depth source. `opaque_depth_view` must be either
    /// the on-screen `depth_texture`'s view or the offscreen screenshot
    /// depth texture's view, depending on which target is being peeled for.
    pub fn make_peel_light_bind_groups(
        &self,
        device: &wgpu::Device,
        opaque_depth_view: &wgpu::TextureView,
    ) -> (wgpu::BindGroup, wgpu::BindGroup) {
        let depth_a = &self
            .peel_depth_a
            .as_ref()
            .expect("peel resources not initialized -- call ensure_peel_resources first")
            .view;
        let depth_b = &self
            .peel_depth_b
            .as_ref()
            .expect("peel resources not initialized -- call ensure_peel_resources first")
            .view;

        let make = |prev_depth_view: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Peel Light BG"),
                layout: &self.peel_light_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.light_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(prev_depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(opaque_depth_view),
                    },
                ],
            })
        };

        // "a" is used while WRITING to peel_depth_a, so its prev_depth must
        // be peel_depth_b (the previous iteration's result) -- and vice versa.
        (make(depth_b), make(depth_a))
    }
}

// ─── Billboard Instance (particle data) ──────────────────────

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BillboardInstance {
    pub center: [f32; 3],
    pub radius: f32,
    pub color: [f32; 4],
}

impl BillboardInstance {
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

// --- MeshPoseInstance ---

/// GPU vertex-buffer representation of a rigid-body pose (translation +
/// rotation), uploaded per boundary-mesh instance via `bytemuck::bytes_of`.
///
/// Deliberately a separate type from [`simulation_lib::render_info::RenderPose`] (the
/// serde/bincode-compatible form sent from the backend worker to the
/// frontend over the crossbeam channel): GPU vertex attributes require
/// [alignment/padding] that would be wasted overhead in the wire format, and
/// vice versa, `RenderPose`'s compact layout doesn't satisfy wgpu's
/// alignment rules. `MeshPoseInstance` is constructed via `From<RenderPose>`
/// (or similar) when uploading a `BoundaryMeshDraw` to the GPU.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshPoseInstance {
    pub translation: [f32; 3],
    pub _pad0: f32,
    /// Quaternion (i, j, k, w)
    pub rotation: [f32; 4],
}

impl MeshPoseInstance {
    pub const IDENTITY: Self = Self {
        translation: [0.0; 3],
        _pad0: 0.0,
        rotation: [0.0, 0.0, 0.0, 1.0],
    };

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

impl From<&simulation_lib::render_info::RenderPose> for MeshPoseInstance {
    fn from(p: &simulation_lib::render_info::RenderPose) -> Self {
        Self {
            translation: p.translation,
            _pad0: 0.0,
            rotation: p.rotation,
        }
    }
}
