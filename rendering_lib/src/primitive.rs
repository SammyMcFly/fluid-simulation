//! FluidFrame – per-frame render data.
//! Implements shader::Primitive (created every frame by FluidViewport::draw()).

use cosmic::iced::Rectangle;
use cosmic::iced::wgpu;
use cosmic::iced::wgpu::util::DeviceExt;
use cosmic::iced::widget::shader;
use cosmic::iced::widget::shader::Viewport;
use crossbeam::channel::Sender;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::camera::CameraUniform;
use crate::lighting::LightUniform;
use crate::model::ColoredMeshVertex;
use crate::pipeline::PendingScreenshot;
use crate::pipeline::PendingScreenshotTarget;
use crate::pipeline::ScreenshotCommand;
use crate::pipeline::ScreenshotState;
use crate::pipeline::{BillboardInstance, DepthTexture, SimulationRenderer};

// ─── CPU-side scene data ──────────────────────────────────────

/// Describes what to render in a single frame.
/// All data here is CPU-side; upload happens in prepare().
#[derive(Debug, Clone)]
pub struct SceneData {
    pub fluid: FluidSceneData,
    pub boundary: BoundarySceneData,
}

impl Default for SceneData {
    fn default() -> Self {
        Self {
            fluid: FluidSceneData::None,
            boundary: BoundarySceneData::None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum FluidSceneData {
    Particles {
        instances: Vec<BillboardInstance>,
    },
    Mesh {
        vertices: Vec<ColoredMeshVertex>,
        indices: Vec<u32>,
        transparent: bool,
    },
    SensorPlane {
        vertices: Vec<ColoredMeshVertex>,
        indices: Vec<u32>,
    },
    None,
}

#[derive(Debug, Clone)]
pub enum BoundarySceneData {
    Particles {
        instances: Vec<BillboardInstance>,
    },
    Mesh {
        vertices: Vec<ColoredMeshVertex>,
        indices: Vec<u32>,
    },
    None,
}

// ─── ScreenshotRequest ────────────────────────────────────────

/// Describes how/where to save the screenshot
#[derive(Debug, Clone)]
pub enum ScreenshotTarget {
    /// Single file to a specific path (button screenshot)
    SingleFile { path: PathBuf },
    /// Sequential frame in a directory (CLI rendering)
    RenderingFrame {
        frame_index: usize,
        output_dir: PathBuf,
    },
}

#[derive(Debug, Clone)]
pub struct ScreenshotRequest {
    pub target: ScreenshotTarget,
}

// ─── FluidFrame ───────────────────────────────────────────────

/// The per-frame primitive. Carries all data needed for one render.
#[derive(Debug, Clone)]
pub struct SimulationFrame<W: ScreenshotCommand> {
    pub camera_uniform: CameraUniform,
    pub light_uniform: LightUniform,
    pub scene: SceneData,
    pub background_color: [f32; 4],
    // Screenshot
    pub readback_request: Option<ScreenshotRequest>,
    pub worker_sender: Option<Sender<W>>,
    /// Shared flag: set to true once readback data is dispatched
    pub screenshot_consumed: Arc<AtomicBool>,
}

impl<W: ScreenshotCommand> shader::Primitive for SimulationFrame<W> {
    type Pipeline = SimulationRenderer;

    fn prepare(
        &self,
        pipeline: &mut SimulationRenderer,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _bounds: &Rectangle,
        viewport: &Viewport,
    ) {
        // ─── Screenshot state machine ─────────────────────────
        {
            let mut state = pipeline.screenshot_state.lock().unwrap();
            match &*state {
                ScreenshotState::Idle => {}
                ScreenshotState::CopyIssued { .. } => {
                    // Copy was submitted last frame. Now safe to map.
                    if let Some(buffer) = &pipeline.staging_buffer {
                        let mapped_flag = pipeline.staging_mapped.clone();
                        buffer
                            .slice(..)
                            .map_async(wgpu::MapMode::Read, move |result| {
                                if result.is_ok() {
                                    mapped_flag.store(true, Ordering::Release);
                                }
                            });
                        let old = std::mem::replace(&mut *state, ScreenshotState::Idle);
                        if let ScreenshotState::CopyIssued { pending } = old {
                            *state = ScreenshotState::MapPending { pending };
                        }
                    }
                }
                ScreenshotState::MapPending { .. } => {
                    if pipeline.staging_mapped.load(Ordering::Acquire) {
                        if let Some(buffer) = &pipeline.staging_buffer {
                            let old = std::mem::replace(&mut *state, ScreenshotState::Idle);
                            if let ScreenshotState::MapPending { pending } = old {
                                let slice = buffer.slice(..);
                                let mapped = slice.get_mapped_range();

                                let row_bytes = (pending.width * 4) as usize;
                                let padded_bpr = pipeline.staging_padded_bpr as usize;
                                let mut rgba = vec![0u8; row_bytes * pending.height as usize];

                                for y in 0..pending.height as usize {
                                    let src = &mapped[y * padded_bpr..y * padded_bpr + row_bytes];
                                    let dst = &mut rgba[y * row_bytes..(y + 1) * row_bytes];
                                    dst.copy_from_slice(src);
                                }

                                drop(mapped);
                                buffer.unmap();
                                pipeline.staging_mapped.store(false, Ordering::Release);

                                // Swizzle BGRA → RGBA if needed
                                if pipeline.surface_format == wgpu::TextureFormat::Bgra8Unorm
                                    || pipeline.surface_format
                                        == wgpu::TextureFormat::Bgra8UnormSrgb
                                {
                                    for pixel in rgba.chunks_exact_mut(4) {
                                        pixel.swap(0, 2);
                                    }
                                }

                                // Dispatch to worker
                                if let Some(sender) = &self.worker_sender {
                                    match pending.target {
                                        PendingScreenshotTarget::Directory {
                                            frame_index,
                                            directory,
                                        } => {
                                            let _ = sender.send(W::write_rendering(
                                                rgba,
                                                pending.width,
                                                pending.height,
                                                frame_index,
                                                directory,
                                            ));
                                        }
                                        PendingScreenshotTarget::ExplicitPath { path } => {
                                            let _ = sender.send(W::save_screenshot_to_file(
                                                rgba,
                                                pending.width,
                                                pending.height,
                                                path,
                                            ));
                                        }
                                    }
                                }

                                self.screenshot_consumed.store(true, Ordering::Release);
                            }
                        }
                    }
                }
            }
        } // Mutex guard dropped here

        // ─── Depth texture ────────────────────────────────────
        let width = viewport.physical_width();
        let height = viewport.physical_height();
        if pipeline
            .depth_texture
            .as_ref()
            .map_or(true, |d| d.width != width || d.height != height)
        {
            pipeline.depth_texture = Some(DepthTexture::new(device, width, height));
        }

        // ─── Camera + Light uniforms ──────────────────────────
        queue.write_buffer(
            &pipeline.camera_buffer,
            0,
            bytemuck::bytes_of(&self.camera_uniform),
        );
        queue.write_buffer(
            &pipeline.light_buffer,
            0,
            bytemuck::bytes_of(&self.light_uniform),
        );

        // ─── Scene buffers ────────────────────────────────────
        self.upload_scene(pipeline, device);

        // ─── Setup offscreen for screenshot ───────────────────
        if self.readback_request.is_some() {
            self.setup_offscreen(pipeline, device, width, height);
        }
    }

    fn draw(
        &self,
        _pipeline: &SimulationRenderer,
        _render_pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        false // Call render instead
    }

    fn render(
        &self,
        pipeline: &SimulationRenderer,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clip_bounds: &Rectangle<u32>,
    ) {
        let Some(depth) = &pipeline.depth_texture else {
            return;
        };

        // ─── 1. Normal render to screen ──────────────────────
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Fluid Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_viewport(
                clip_bounds.x as f32,
                clip_bounds.y as f32,
                clip_bounds.width as f32,
                clip_bounds.height as f32,
                0.0,
                1.0,
            );
            pass.set_scissor_rect(
                clip_bounds.x,
                clip_bounds.y,
                clip_bounds.width,
                clip_bounds.height,
            );

            self.draw_scene(&mut pass, pipeline);
        }

        // ─── 2. Offscreen render for screenshot ──────────────
        if let Some(ref request) = self.readback_request {
            let should_capture = {
                let state = pipeline.screenshot_state.lock().unwrap();
                matches!(*state, ScreenshotState::Idle)
            };

            if should_capture {
                if let (Some(offscreen_view), Some(offscreen_depth)) =
                    (&pipeline.offscreen_view, &pipeline.offscreen_depth)
                {
                    // Render to offscreen
                    {
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("Offscreen Render Pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: offscreen_view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color {
                                        r: self.background_color[0] as f64,
                                        g: self.background_color[1] as f64,
                                        b: self.background_color[2] as f64,
                                        a: self.background_color[3] as f64,
                                    }),
                                    store: wgpu::StoreOp::Store,
                                },
                                depth_slice: None,
                            })],
                            depth_stencil_attachment: Some(
                                wgpu::RenderPassDepthStencilAttachment {
                                    view: &offscreen_depth.view,
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(1.0),
                                        store: wgpu::StoreOp::Store,
                                    }),
                                    stencil_ops: None,
                                },
                            ),
                            ..Default::default()
                        });

                        self.draw_scene(&mut pass, pipeline);
                    }

                    // Copy offscreen → staging
                    if let (Some(texture), Some(buffer)) =
                        (&pipeline.offscreen_texture, &pipeline.staging_buffer)
                    {
                        let width = pipeline.screenshot_width;
                        let height = pipeline.screenshot_height;

                        encoder.copy_texture_to_buffer(
                            wgpu::TexelCopyTextureInfo {
                                texture,
                                mip_level: 0,
                                origin: wgpu::Origin3d::ZERO,
                                aspect: wgpu::TextureAspect::All,
                            },
                            wgpu::TexelCopyBufferInfo {
                                buffer,
                                layout: wgpu::TexelCopyBufferLayout {
                                    offset: 0,
                                    bytes_per_row: Some(pipeline.staging_padded_bpr),
                                    rows_per_image: Some(height),
                                },
                            },
                            wgpu::Extent3d {
                                width,
                                height,
                                depth_or_array_layers: 1,
                            },
                        );

                        // Build target
                        let target = match &request.target {
                            ScreenshotTarget::SingleFile { path } => {
                                PendingScreenshotTarget::ExplicitPath { path: path.clone() }
                            }
                            ScreenshotTarget::RenderingFrame {
                                frame_index,
                                output_dir,
                            } => PendingScreenshotTarget::Directory {
                                frame_index: *frame_index,
                                directory: output_dir.clone(),
                            },
                        };

                        // Transition: Idle → CopyIssued
                        let mut state = pipeline.screenshot_state.lock().unwrap();
                        *state = ScreenshotState::CopyIssued {
                            pending: PendingScreenshot {
                                width,
                                height,
                                target,
                            },
                        };
                    }
                }
            }
        }
    }
}

// ─── Upload helpers ───────────────────────────────────────────

impl<W: ScreenshotCommand> SimulationFrame<W> {
    fn upload_scene(&self, pipeline: &mut SimulationRenderer, device: &wgpu::Device) {
        // Reset
        pipeline.scene = Default::default();

        // ─── Fluid ────────────────────────────────────────────
        match &self.scene.fluid {
            FluidSceneData::Particles { instances } => {
                if !instances.is_empty() {
                    pipeline.scene.particle_buffer = Some(device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("Fluid Particle Buffer"),
                            contents: bytemuck::cast_slice(instances),
                            usage: wgpu::BufferUsages::VERTEX,
                        },
                    ));
                    pipeline.scene.particle_count = instances.len() as u32;
                }
            }
            FluidSceneData::Mesh {
                vertices,
                indices,
                transparent,
            } => {
                if !vertices.is_empty() && !indices.is_empty() {
                    pipeline.scene.mesh_vertex_buffer = Some(device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("Fluid Mesh VB"),
                            contents: bytemuck::cast_slice(vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        },
                    ));
                    pipeline.scene.mesh_index_buffer = Some(device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("Fluid Mesh IB"),
                            contents: bytemuck::cast_slice(indices),
                            usage: wgpu::BufferUsages::INDEX,
                        },
                    ));
                    pipeline.scene.mesh_index_count = indices.len() as u32;
                    pipeline.scene.mesh_transparent = *transparent;
                }
            }
            FluidSceneData::SensorPlane { vertices, indices } => {
                if !vertices.is_empty() && !indices.is_empty() {
                    pipeline.scene.sensor_plane_vertex_buffer = Some(device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("Sensor Plane VB"),
                            contents: bytemuck::cast_slice(vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        },
                    ));
                    pipeline.scene.sensor_plane_index_buffer = Some(device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("Sensor Plane IB"),
                            contents: bytemuck::cast_slice(indices),
                            usage: wgpu::BufferUsages::INDEX,
                        },
                    ));
                    pipeline.scene.sensor_plane_index_count = indices.len() as u32;
                }
            }
            FluidSceneData::None => {}
        }

        // ─── Boundary ─────────────────────────────────────────
        match &self.scene.boundary {
            BoundarySceneData::Particles { instances } => {
                if !instances.is_empty() {
                    pipeline.scene.boundary_particle_buffer = Some(device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("Boundary Particle Buffer"),
                            contents: bytemuck::cast_slice(instances),
                            usage: wgpu::BufferUsages::VERTEX,
                        },
                    ));
                    pipeline.scene.boundary_particle_count = instances.len() as u32;
                }
            }
            BoundarySceneData::Mesh { vertices, indices } => {
                if !vertices.is_empty() && !indices.is_empty() {
                    pipeline.scene.boundary_mesh_vertex_buffer = Some(device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("Boundary Mesh VB"),
                            contents: bytemuck::cast_slice(vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        },
                    ));
                    pipeline.scene.boundary_mesh_index_buffer = Some(device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("Boundary Mesh IB"),
                            contents: bytemuck::cast_slice(indices),
                            usage: wgpu::BufferUsages::INDEX,
                        },
                    ));
                    pipeline.scene.boundary_mesh_index_count = indices.len() as u32;
                }
            }
            BoundarySceneData::None => {}
        }
    }

    fn setup_offscreen(
        &self,
        pipeline: &mut SimulationRenderer,
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) {
        let needs_recreate = pipeline.offscreen_texture.as_ref().map_or(true, |t| {
            let size = t.size();
            size.width != width || size.height != height
        });

        if needs_recreate {
            // Offscreen color texture
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Offscreen Screenshot"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: pipeline.surface_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            pipeline.offscreen_view =
                Some(texture.create_view(&wgpu::TextureViewDescriptor::default()));
            pipeline.offscreen_texture = Some(texture);

            // Offscreen depth texture
            pipeline.offscreen_depth = Some(DepthTexture::new(device, width, height));

            // Staging buffer
            let bytes_per_pixel = 4u32;
            let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            let padded_bpr = ((width * bytes_per_pixel + align - 1) / align) * align;
            let buffer_size = (padded_bpr * height) as u64;

            pipeline.staging_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Screenshot Staging"),
                size: buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            pipeline.staging_padded_bpr = padded_bpr;
        }

        pipeline.screenshot_width = width;
        pipeline.screenshot_height = height;
    }

    fn draw_scene<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, pipeline: &'a SimulationRenderer) {
        // Light indicator
        pass.set_pipeline(&pipeline.light_pipeline);
        pass.set_bind_group(0, &pipeline.camera_bind_group, &[]);
        pass.set_bind_group(1, &pipeline.light_bind_group, &[]);
        pass.set_vertex_buffer(0, pipeline.light_mesh.vertex_buffer.slice(..));
        pass.set_index_buffer(
            pipeline.light_mesh.index_buffer.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        pass.draw_indexed(0..pipeline.light_mesh.num_indices, 0, 0..1);

        // Boundary particles
        if let Some(buf) = &pipeline.scene.boundary_particle_buffer {
            if pipeline.scene.boundary_particle_count > 0 {
                pass.set_pipeline(&pipeline.particle_pipeline);
                pass.set_bind_group(0, &pipeline.camera_bind_group, &[]);
                pass.set_bind_group(1, &pipeline.light_bind_group, &[]);
                pass.set_vertex_buffer(0, buf.slice(..));
                pass.draw(0..6, 0..pipeline.scene.boundary_particle_count);
            }
        }

        // Boundary mesh
        if let (Some(vb), Some(ib)) = (
            &pipeline.scene.boundary_mesh_vertex_buffer,
            &pipeline.scene.boundary_mesh_index_buffer,
        ) {
            if pipeline.scene.boundary_mesh_index_count > 0 {
                pass.set_pipeline(&pipeline.mesh_opaque_pipeline);
                pass.set_bind_group(0, &pipeline.camera_bind_group, &[]);
                pass.set_bind_group(1, &pipeline.light_bind_group, &[]);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..pipeline.scene.boundary_mesh_index_count, 0, 0..1);
            }
        }

        // Fluid particles
        if let Some(buf) = &pipeline.scene.particle_buffer {
            if pipeline.scene.particle_count > 0 {
                pass.set_pipeline(&pipeline.particle_pipeline);
                pass.set_bind_group(0, &pipeline.camera_bind_group, &[]);
                pass.set_bind_group(1, &pipeline.light_bind_group, &[]);
                pass.set_vertex_buffer(0, buf.slice(..));
                pass.draw(0..6, 0..pipeline.scene.particle_count);
            }
        }

        // Sensor plane
        if let (Some(vb), Some(ib)) = (
            &pipeline.scene.sensor_plane_vertex_buffer,
            &pipeline.scene.sensor_plane_index_buffer,
        ) {
            if pipeline.scene.sensor_plane_index_count > 0 {
                pass.set_pipeline(&pipeline.mesh_unlit_pipeline);
                pass.set_bind_group(0, &pipeline.camera_bind_group, &[]);
                // no light_bind_group – pipeline layout has only Camera
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..pipeline.scene.sensor_plane_index_count, 0, 0..1);
            }
        }

        // Fluid mesh
        if let (Some(vb), Some(ib)) = (
            &pipeline.scene.mesh_vertex_buffer,
            &pipeline.scene.mesh_index_buffer,
        ) {
            if pipeline.scene.mesh_index_count > 0 {
                let pip = if pipeline.scene.mesh_transparent {
                    &pipeline.mesh_transparent_pipeline
                } else {
                    &pipeline.mesh_opaque_pipeline
                };
                pass.set_pipeline(pip);
                pass.set_bind_group(0, &pipeline.camera_bind_group, &[]);
                pass.set_bind_group(1, &pipeline.light_bind_group, &[]);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..pipeline.scene.mesh_index_count, 0, 0..1);
            }
        }
    }
}
