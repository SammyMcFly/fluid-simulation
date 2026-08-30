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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
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
    pub mesh_transparent_pipeline: wgpu::RenderPipeline,
    pub mesh_transparent_backface_pipeline: wgpu::RenderPipeline,
    pub mesh_transparent_fluid_pipeline: wgpu::RenderPipeline,
    pub mesh_transparent_fluid_backface_pipeline: wgpu::RenderPipeline,
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
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
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
        let mesh_transparent_pipeline = Self::create_mesh_pipeline(
            device,
            format,
            &camera_bind_group_layout,
            &light_bind_group_layout,
            Some(wgpu::Face::Back),
            true,
            false,
            true,
        );
        let mesh_transparent_backface_pipeline = Self::create_mesh_pipeline(
            device,
            format,
            &camera_bind_group_layout,
            &light_bind_group_layout,
            Some(wgpu::Face::Front),
            true,
            false,
            true,
        );
        let mesh_transparent_fluid_pipeline = Self::create_mesh_pipeline(
            device,
            format,
            &camera_bind_group_layout,
            &light_bind_group_layout,
            Some(wgpu::Face::Back),
            true,
            true,
            false,
        );
        let mesh_transparent_fluid_backface_pipeline = Self::create_mesh_pipeline(
            device,
            format,
            &camera_bind_group_layout,
            &light_bind_group_layout,
            Some(wgpu::Face::Front),
            true,
            true,
            false,
        );
        let mesh_unlit_pipeline =
            Self::create_unlit_mesh_pipeline(device, format, &camera_bind_group_layout);

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
            mesh_transparent_pipeline,
            mesh_transparent_backface_pipeline,
            mesh_transparent_fluid_pipeline,
            mesh_transparent_fluid_backface_pipeline,
            mesh_unlit_pipeline,
            light_pipeline,
            surface_format: format,
            depth_texture: None,
            light_mesh,
            scene: SceneGpuBuffers::default(),

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
