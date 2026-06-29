// src/gui/scene_buffers.rs

use iced_wgpu::wgpu;
use iced_wgpu::wgpu::util::DeviceExt;
use simulation_lib::render_info::*;
use simulation_lib::utilities::triangle_mesh::RenderMesh;

use crate::colormap;
use crate::gpu_context::GpuContext;
use crate::instances::{BillboardInstanceRaw, StagingSettings};
use crate::model::ColoredMeshVertex;
use crate::ui::controls::cut::Cut;

// ─── Fluid GPU data ───────────────────────────────────────────

pub enum FluidGpuData {
    Mesh {
        vertex_buffer: wgpu::Buffer,
        index_buffer: wgpu::Buffer,
        num_indices: u32,
        transparent: bool,
    },
    Particles {
        instance_buffer: wgpu::Buffer,
        count: u32,
    },
    SensorPlane {
        vertex_buffer: wgpu::Buffer,
        index_buffer: wgpu::Buffer,
        num_indices: u32,
    },
    None,
}

// ─── Boundary GPU data ────────────────────────────────────────

pub enum BoundaryGpuData {
    Mesh {
        vertex_buffer: wgpu::Buffer,
        index_buffer: wgpu::Buffer,
        num_indices: u32,
    },
    Particles {
        instance_buffer: wgpu::Buffer,
        count: u32,
    },
    None,
}

// ─── Builder ──────────────────────────────────────────────────

pub struct SceneBuffers {
    pub fluid: FluidGpuData,
    pub boundary: BoundaryGpuData,
    pub staging_settings: StagingSettings,
}

impl SceneBuffers {
    pub fn empty() -> Self {
        Self {
            fluid: FluidGpuData::None,
            boundary: BoundaryGpuData::None,
            staging_settings: StagingSettings::default(),
        }
    }

    pub fn build(
        gpu: &GpuContext,
        time_step: &TimeStepInfo,
        staging_settings: StagingSettings,
        cut: &Cut,
        boundary_hidden: bool,
        particle_radius: f32,
    ) -> Self {
        let fluid = Self::build_fluid(gpu, &time_step.fluid, cut, particle_radius);
        let boundary = if boundary_hidden {
            BoundaryGpuData::None
        } else {
            Self::build_boundary(gpu, &time_step.boundary, cut, particle_radius)
        };
        Self {
            fluid,
            boundary,
            staging_settings,
        }
    }

    pub fn needs_update(&self, staging_settings: &StagingSettings) -> bool {
        self.staging_settings != *staging_settings
    }

    // ─── Fluid ────────────────────────────────────────────────

    fn build_fluid(
        gpu: &GpuContext,
        vis: &FluidVisualization,
        cut: &Cut,
        particle_radius: f32,
    ) -> FluidGpuData {
        match vis {
            FluidVisualization::TriangleMesh { mesh } => {
                Self::build_mesh_buffers(gpu, mesh, [0.3, 0.6, 0.9, 0.5], false)
            }
            FluidVisualization::Samples {
                positions,
                coloring,
            } => {
                let colors = Self::resolve_fluid_coloring(coloring, positions.len());
                let instances: Vec<BillboardInstanceRaw> = positions
                    .iter()
                    .zip(colors.iter())
                    .filter(|(pos, _)| cut.cut(pos))
                    .map(|(pos, color)| BillboardInstanceRaw {
                        // coordinate swap: x, z, -y (matching your existing ToRaw)
                        center: [pos[0], pos[2], -pos[1]],
                        radius: particle_radius,
                        color: *color,
                    })
                    .collect();

                let count = instances.len() as u32;
                if count == 0 {
                    return FluidGpuData::None;
                }

                let buffer = gpu
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Fluid Particle Buffer"),
                        contents: bytemuck::cast_slice(&instances),
                        usage: wgpu::BufferUsages::VERTEX,
                    });

                FluidGpuData::Particles {
                    instance_buffer: buffer,
                    count,
                }
            }
            FluidVisualization::SensorPlane { planes } => Self::build_sensor_planes(gpu, planes),
        }
    }

    fn resolve_fluid_coloring(coloring: &FluidColoring, count: usize) -> Vec<[f32; 4]> {
        match coloring {
            FluidColoring::Uniform => vec![[0.2, 0.5, 1.0, 1.0]; count],
            FluidColoring::FluidId { val, max_id } => colormap::id_to_colors(val, *max_id, 1.0),
            FluidColoring::QuantityGraded { quantity } => {
                let values = Self::extract_scalar_values(quantity);
                colormap::values_to_colors(values, 1.0)
            }
        }
    }

    fn extract_scalar_values(quantity: &ScalarQuantity) -> &[f32] {
        match quantity {
            ScalarQuantity::SpeedGraded(v)
            | ScalarQuantity::VolumeGraded(v)
            | ScalarQuantity::DensityGraded(v)
            | ScalarQuantity::DensityErrorGraded(v)
            | ScalarQuantity::PressureGraded(v)
            | ScalarQuantity::KineticEnergyGraded(v) => v,
        }
    }

    // ─── Boundary ─────────────────────────────────────────────

    fn build_boundary(
        gpu: &GpuContext,
        vis: &BoundaryVisualization,
        cut: &Cut,
        particle_radius: f32,
    ) -> BoundaryGpuData {
        match vis {
            BoundaryVisualization::TriangleMesh { mesh, coloring } => {
                let color = match coloring {
                    BoundaryMeshColoring::Original => [0.7, 0.7, 0.7, 1.0], // fallback gray
                    BoundaryMeshColoring::Uniform => [0.6, 0.6, 0.6, 1.0],
                };
                match Self::build_mesh_buffers(gpu, mesh, color, false) {
                    FluidGpuData::Mesh {
                        vertex_buffer,
                        index_buffer,
                        num_indices,
                        ..
                    } => BoundaryGpuData::Mesh {
                        vertex_buffer,
                        index_buffer,
                        num_indices,
                    },
                    _ => BoundaryGpuData::None,
                }
            }
            BoundaryVisualization::Samples {
                positions,
                coloring,
            } => {
                let colors = Self::resolve_boundary_coloring(coloring, positions.len());
                let instances: Vec<BillboardInstanceRaw> = positions
                    .iter()
                    .zip(colors.iter())
                    .filter(|(pos, _)| cut.cut(pos))
                    .map(|(pos, color)| BillboardInstanceRaw {
                        center: [pos[0], pos[2], -pos[1]],
                        radius: particle_radius,
                        color: *color,
                    })
                    .collect();

                let count = instances.len() as u32;
                if count == 0 {
                    return BoundaryGpuData::None;
                }

                let buffer = gpu
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Boundary Particle Buffer"),
                        contents: bytemuck::cast_slice(&instances),
                        usage: wgpu::BufferUsages::VERTEX,
                    });

                BoundaryGpuData::Particles {
                    instance_buffer: buffer,
                    count,
                }
            }
        }
    }

    fn resolve_boundary_coloring(coloring: &BoundarySampleColoring, count: usize) -> Vec<[f32; 4]> {
        match coloring {
            BoundarySampleColoring::Uniform => vec![[0.6, 0.6, 0.6, 1.0]; count],
            BoundarySampleColoring::BoundaryId { val, max_id } => {
                colormap::id_to_colors(val, *max_id, 1.0)
            }
        }
    }

    // ─── Shared helpers ───────────────────────────────────────

    fn build_mesh_buffers(
        gpu: &GpuContext,
        render_mesh: &RenderMesh,
        uniform_color: [f32; 4],
        transparent: bool,
    ) -> FluidGpuData {
        if render_mesh.vertices.is_empty() || render_mesh.indices.is_empty() {
            return FluidGpuData::None;
        }

        let vertices: Vec<ColoredMeshVertex> = render_mesh
            .vertices
            .iter()
            .map(|v| ColoredMeshVertex {
                // coordinate swap to match your GPU convention
                position: [
                    v.position[0] as f32,
                    v.position[2] as f32,
                    -v.position[1] as f32,
                ],
                normal: [v.normal[0] as f32, v.normal[2] as f32, -v.normal[1] as f32],
                color: uniform_color,
            })
            .collect();

        let vertex_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Mesh Vertex Buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let index_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Mesh Index Buffer"),
                contents: bytemuck::cast_slice(&render_mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });

        FluidGpuData::Mesh {
            vertex_buffer,
            index_buffer,
            num_indices: render_mesh.indices.len() as u32,
            transparent,
        }
    }

    fn build_sensor_planes(gpu: &GpuContext, planes: &[SensorPlaneData]) -> FluidGpuData {
        let mut all_vertices: Vec<ColoredMeshVertex> = Vec::new();
        let mut all_indices: Vec<u32> = Vec::new();

        for plane in planes {
            let values = Self::extract_scalar_values(&plane.quantity);
            if plane.positions.is_empty() || values.is_empty() || plane.rows < 2 || plane.cols < 2 {
                continue;
            }

            let colors = colormap::values_to_colors(values, 1.0);
            let rows = plane.rows;
            let cols = plane.cols;
            let base_vertex = all_vertices.len() as u32;

            // Compute per-vertex normals
            let mut normals = vec![[0.0f32; 3]; plane.positions.len()];
            for r in 0..rows - 1 {
                for c in 0..cols - 1 {
                    let i = r * cols + c;
                    let p0 = plane.positions[i];
                    let p1 = plane.positions[i + 1];
                    let p2 = plane.positions[i + cols];
                    let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
                    let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
                    let n = [
                        e1[1] * e2[2] - e1[2] * e2[1],
                        e1[2] * e2[0] - e1[0] * e2[2],
                        e1[0] * e2[1] - e1[1] * e2[0],
                    ];
                    for idx in [i, i + 1, i + cols] {
                        normals[idx][0] += n[0];
                        normals[idx][1] += n[1];
                        normals[idx][2] += n[2];
                    }
                }
            }
            for n in &mut normals {
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                if len > 1e-10 {
                    n[0] /= len;
                    n[1] /= len;
                    n[2] /= len;
                }
            }

            // Vertices
            for ((pos, normal), color) in plane
                .positions
                .iter()
                .zip(normals.iter())
                .zip(colors.iter())
            {
                all_vertices.push(ColoredMeshVertex {
                    position: [pos[0], pos[2], -pos[1]],
                    normal: [normal[0], normal[2], -normal[1]],
                    color: *color,
                });
            }

            // Indices (offset by base_vertex)
            for r in 0..(rows - 1) {
                for c in 0..(cols - 1) {
                    let tl = base_vertex + (r * cols + c) as u32;
                    let tr = tl + 1;
                    let bl = tl + cols as u32;
                    let br = bl + 1;
                    all_indices.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
                }
            }
        }

        if all_vertices.is_empty() {
            return FluidGpuData::None;
        }

        let vertex_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Sensor Planes VB"),
                contents: bytemuck::cast_slice(&all_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let index_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Sensor Planes IB"),
                contents: bytemuck::cast_slice(&all_indices),
                usage: wgpu::BufferUsages::INDEX,
            });

        FluidGpuData::SensorPlane {
            vertex_buffer,
            index_buffer,
            num_indices: all_indices.len() as u32,
        }
    }
}
