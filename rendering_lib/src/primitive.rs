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
use crate::pipeline::BoundaryMesh;
use crate::pipeline::MeshPoseInstance;
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
    Particles { instances: Vec<BillboardInstance> },
    Mesh { meshes: Vec<BoundaryMeshDraw> },
    None,
}

#[derive(Debug, Clone)]
pub struct BoundaryMeshDraw {
    pub vertices: Vec<ColoredMeshVertex>,
    pub indices: Vec<u32>,
    pub pose: MeshPoseInstance,
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
        /// If `true`, overwrite an existing file at this `frame_index` instead of
        /// treating its existence as an error. Used when a visualization change
        /// mid-render causes the current frame to be recaptured with the new
        /// settings.
        overwrite: bool,
    },
}

#[derive(Debug, Clone)]
pub struct ScreenshotRequest {
    pub target: ScreenshotTarget,
    /// Monotonically increasing ID, unique per call to `request_screenshot`.
    /// Used to detect and skip re-captures of an already-completed request
    /// that the app hasn't cleared yet (see `SimulationViewport::request_screenshot`).
    pub id: u64,
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
                    if let Some(buffer) = &pipeline.staging_buffer
                        && pipeline.staging_mapped.load(Ordering::Acquire)
                    {
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
                                || pipeline.surface_format == wgpu::TextureFormat::Bgra8UnormSrgb
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
                                        overwrite,
                                    } => {
                                        let _ = sender.send(W::write_rendering(
                                            rgba,
                                            pending.width,
                                            pending.height,
                                            frame_index,
                                            directory,
                                            overwrite,
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
                            pipeline.last_completed_screenshot_id = Some(pending.id);
                            self.screenshot_consumed.store(true, Ordering::Release);
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
            .is_none_or(|d| d.width != width || d.height != height)
        {
            pipeline.depth_texture = Some(DepthTexture::new(device, width, height));
        }

        // ─── Depth peeling resources (transparent meshes) ─────
        pipeline.ensure_peel_resources(device, width, height);
        {
            let opaque_depth_view = &pipeline.depth_texture.as_ref().unwrap().view;
            let (bg_a, bg_b) = pipeline.make_peel_light_bind_groups(device, opaque_depth_view);
            pipeline.peel_light_bind_group_a = Some(bg_a);
            pipeline.peel_light_bind_group_b = Some(bg_b);
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
            let opaque_depth_view = &pipeline.offscreen_depth.as_ref().unwrap().view;
            let (bg_a, bg_b) = pipeline.make_peel_light_bind_groups(device, opaque_depth_view);
            pipeline.peel_light_bind_group_offscreen_a = Some(bg_a);
            pipeline.peel_light_bind_group_offscreen_b = Some(bg_b);
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

        // ─── 1. Opaque + depth-correct billboard pass ────────
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

            self.draw_opaque_scene(&mut pass, pipeline);
        }

        // ─── 2. Depth-peeled transparency (boundary + fluid mesh) ──
        if let (Some(bg_a), Some(bg_b)) = (
            &pipeline.peel_light_bind_group_a,
            &pipeline.peel_light_bind_group_b,
        ) {
            self.peel_transparency(pipeline, encoder, bg_a, bg_b, Some(clip_bounds));
            self.composite_accum(pipeline, encoder, target, Some(clip_bounds));
        }

        // ─── 3. Offscreen render for screenshot ──────────────
        if let Some(ref request) = self.readback_request {
            let should_capture = {
                let state = pipeline.screenshot_state.lock().unwrap();
                matches!(*state, ScreenshotState::Idle)
                    && pipeline.last_completed_screenshot_id != Some(request.id)
            };

            if let (Some(offscreen_view), Some(offscreen_depth)) =
                (&pipeline.offscreen_view, &pipeline.offscreen_depth)
                && should_capture
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
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &offscreen_depth.view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        ..Default::default()
                    });

                    self.draw_opaque_scene(&mut pass, pipeline);
                }

                if let (Some(bg_a), Some(bg_b)) = (
                    &pipeline.peel_light_bind_group_offscreen_a,
                    &pipeline.peel_light_bind_group_offscreen_b,
                ) {
                    // Offscreen target has no clip_bounds concept -- it IS exactly
                    // screenshot_width x screenshot_height, same as pass 1 above
                    // (which also doesn't set a viewport for the offscreen case).
                    self.peel_transparency(pipeline, encoder, bg_a, bg_b, None);
                    self.composite_accum(pipeline, encoder, offscreen_view, None);
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
                            overwrite,
                        } => PendingScreenshotTarget::Directory {
                            frame_index: *frame_index,
                            directory: output_dir.clone(),
                            overwrite: *overwrite,
                        },
                    };

                    // Transition: Idle → CopyIssued
                    let mut state = pipeline.screenshot_state.lock().unwrap();
                    *state = ScreenshotState::CopyIssued {
                        pending: PendingScreenshot {
                            width,
                            height,
                            target,
                            id: request.id,
                        },
                    };
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
            BoundarySceneData::Mesh { meshes } => {
                for draw in meshes {
                    if draw.vertices.is_empty() || draw.indices.is_empty() {
                        continue;
                    }
                    let vertex_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Boundary Mesh VB"),
                            contents: bytemuck::cast_slice(&draw.vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                    let index_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Boundary Mesh IB"),
                            contents: bytemuck::cast_slice(&draw.indices),
                            usage: wgpu::BufferUsages::INDEX,
                        });
                    let instance_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Boundary Mesh Pose"),
                            contents: bytemuck::bytes_of(&draw.pose),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                    pipeline.scene.boundary_meshes.push(BoundaryMesh {
                        vertex_buffer,
                        index_buffer,
                        index_count: draw.indices.len() as u32,
                        instance_buffer,
                    });
                }
                // if !vertices.is_empty() && !indices.is_empty() {
                //     pipeline.scene.boundary_mesh_vertex_buffer = Some(device.create_buffer_init(
                //         &wgpu::util::BufferInitDescriptor {
                //             label: Some("Boundary Mesh VB"),
                //             contents: bytemuck::cast_slice(vertices),
                //             usage: wgpu::BufferUsages::VERTEX,
                //         },
                //     ));
                //     pipeline.scene.boundary_mesh_index_buffer = Some(device.create_buffer_init(
                //         &wgpu::util::BufferInitDescriptor {
                //             label: Some("Boundary Mesh IB"),
                //             contents: bytemuck::cast_slice(indices),
                //             usage: wgpu::BufferUsages::INDEX,
                //         },
                //     ));
                //     pipeline.scene.boundary_mesh_index_count = indices.len() as u32;
                // }
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

    /// Draws everything that doesn't need depth peeling: opaque/analytically
    /// depth-correct geometry (light indicator, fluid particles, sensor
    /// plane, boundary particles). Boundary meshes and the transparent fluid
    /// mesh are handled separately by `peel_transparency`, since multiple
    /// such meshes can't be correctly depth-sorted via simple draw order
    /// (see `peel_transparency` for why).
    fn draw_opaque_scene<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        pipeline: &'a SimulationRenderer,
    ) {
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

        // Fluid particles
        if let Some(buf) = &pipeline.scene.particle_buffer
            && pipeline.scene.particle_count > 0
        {
            pass.set_pipeline(&pipeline.particle_pipeline);
            pass.set_bind_group(0, &pipeline.camera_bind_group, &[]);
            pass.set_bind_group(1, &pipeline.light_bind_group, &[]);
            pass.set_vertex_buffer(0, buf.slice(..));
            pass.draw(0..6, 0..pipeline.scene.particle_count);
        }

        // Sensor plane
        if let (Some(vb), Some(ib)) = (
            &pipeline.scene.sensor_plane_vertex_buffer,
            &pipeline.scene.sensor_plane_index_buffer,
        ) && pipeline.scene.sensor_plane_index_count > 0
        {
            pass.set_pipeline(&pipeline.mesh_unlit_pipeline);
            pass.set_bind_group(0, &pipeline.camera_bind_group, &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..pipeline.scene.sensor_plane_index_count, 0, 0..1);
        }

        // Boundary particles
        if let Some(buf) = &pipeline.scene.boundary_particle_buffer
            && pipeline.scene.boundary_particle_count > 0
        {
            pass.set_pipeline(&pipeline.particle_transparent_pipeline);
            pass.set_bind_group(0, &pipeline.camera_bind_group, &[]);
            pass.set_bind_group(1, &pipeline.light_bind_group, &[]);
            pass.set_vertex_buffer(0, buf.slice(..));
            pass.draw(0..6, 0..pipeline.scene.boundary_particle_count);
        }
    }

    /// Depth-peels all transparent boundary meshes and the transparent fluid
    /// mesh together, `DEPTH_PEEL_LAYERS` layers deep, compositing each
    /// layer into `pipeline.accum_texture` via the "under" operator.
    ///
    /// Why this replaces simple draw-order/backface-frontface rendering:
    /// with multiple boundary meshes, there is no single draw order that is
    /// correct for every pixel simultaneously (object A can be in front of
    /// object B on one part of the screen and behind it on another). Depth
    /// peeling sidesteps this by repeatedly extracting the next-nearest
    /// surface at every pixel, across ALL transparent meshes at once, using
    /// the hardware depth test rather than a fixed draw order.
    fn peel_transparency(
        &self,
        pipeline: &SimulationRenderer,
        encoder: &mut wgpu::CommandEncoder,
        peel_bind_group_a: &wgpu::BindGroup,
        peel_bind_group_b: &wgpu::BindGroup,
        clip_bounds: Option<&Rectangle<u32>>,
    ) {
        let (Some(depth_a), Some(depth_b), Some(accum), Some(layer_color), Some(layer_bg)) = (
            &pipeline.peel_depth_a,
            &pipeline.peel_depth_b,
            &pipeline.accum_texture,
            &pipeline.layer_color,
            &pipeline.layer_composite_bind_group,
        ) else {
            return;
        };

        for layer in 0..crate::pipeline::DEPTH_PEEL_LAYERS {
            let (depth_view, bind_group) = if layer % 2 == 0 {
                (&depth_a.view, peel_bind_group_a)
            } else {
                (&depth_b.view, peel_bind_group_b)
            };

            let (boundary_pipeline, fluid_pipeline) = if layer == 0 {
                (
                    &pipeline.boundary_peel_first_pipeline,
                    &pipeline.fluid_peel_first_pipeline,
                )
            } else {
                (
                    &pipeline.boundary_peel_next_pipeline,
                    &pipeline.fluid_peel_next_pipeline,
                )
            };

            // ─── Pass A: resolve this one layer into `layer_color` ──────
            // Normal Z-buffered rendering (REPLACE blend, see
            // `build_peel_pipeline`): among all triangles (boundary + fluid
            // mesh) competing for a pixel, only the nearest one that passes
            // the depth test ends up stored here, exactly like opaque
            // rendering -- order-independent, no cross-triangle color
            // accumulation within this single layer.
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Depth Peel Layer Resolve"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &layer_color.view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });

                pass.set_bind_group(0, &pipeline.camera_bind_group, &[]);
                pass.set_bind_group(1, bind_group, &[]);

                // Must match pass 1's viewport exactly -- `in.clip_position.xy`
                // is used as pixel coordinates for the opaque/prev depth
                // texture lookups in the peel shaders, which must refer to
                // the same screen location pass 1 wrote `opaque_depth` at.
                if let Some(cb) = clip_bounds {
                    pass.set_viewport(
                        cb.x as f32,
                        cb.y as f32,
                        cb.width as f32,
                        cb.height as f32,
                        0.0,
                        1.0,
                    );
                    pass.set_scissor_rect(cb.x, cb.y, cb.width, cb.height);
                }

                if let (Some(vb), Some(ib)) = (
                    &pipeline.scene.mesh_vertex_buffer,
                    &pipeline.scene.mesh_index_buffer,
                ) && pipeline.scene.mesh_index_count > 0
                {
                    pass.set_pipeline(fluid_pipeline);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..pipeline.scene.mesh_index_count, 0, 0..1);
                }

                for mesh in &pipeline.scene.boundary_meshes {
                    if mesh.index_count == 0 {
                        continue;
                    }
                    pass.set_pipeline(boundary_pipeline);
                    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    pass.set_vertex_buffer(1, mesh.instance_buffer.slice(..));
                    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                }
            }

            // ─── Pass B: fold this resolved layer into `accum_texture` ──
            // "Under" blend, exactly once per layer -- this is where
            // cross-layer front-to-back transparency compositing actually
            // happens now (previously done, incorrectly, inside pass A).
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Depth Peel Layer Composite"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &accum.view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: if layer == 0 {
                                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                            } else {
                                wgpu::LoadOp::Load
                            },
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });

                if let Some(cb) = clip_bounds {
                    pass.set_viewport(
                        cb.x as f32,
                        cb.y as f32,
                        cb.width as f32,
                        cb.height as f32,
                        0.0,
                        1.0,
                    );
                    pass.set_scissor_rect(cb.x, cb.y, cb.width, cb.height);
                }

                pass.set_pipeline(&pipeline.layer_composite_pipeline);
                pass.set_bind_group(0, layer_bg, &[]);
                pass.draw(0..3, 0..1);
            }
        }
    }

    /// Blends the fully-accumulated peel result onto `color_view`.
    /// `clip_bounds` restricts the viewport/scissor for the on-screen case;
    /// pass `None` for the offscreen screenshot case (full-size target).
    fn composite_accum(
        &self,
        pipeline: &SimulationRenderer,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        clip_bounds: Option<&Rectangle<u32>>,
    ) {
        let Some(composite_bg) = &pipeline.composite_bind_group else {
            return;
        };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Composite Peel Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });

        if let Some(cb) = clip_bounds {
            pass.set_viewport(
                cb.x as f32,
                cb.y as f32,
                cb.width as f32,
                cb.height as f32,
                0.0,
                1.0,
            );
            pass.set_scissor_rect(cb.x, cb.y, cb.width, cb.height);
        }

        pass.set_pipeline(&pipeline.composite_pipeline);
        pass.set_bind_group(0, composite_bg, &[]);
        pass.draw(0..3, 0..1);
    }
}
