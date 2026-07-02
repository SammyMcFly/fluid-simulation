//! scene_builder.rs – Build CPU-side SceneData from simulation TimeStepInfo

use simulation_lib::render_info::*;
use simulation_lib::utilities::triangle_mesh::RenderMesh;

use crate::colormap;
use crate::model::ColoredMeshVertex;
use crate::pipeline::BillboardInstance;
use crate::primitive::{BoundarySceneData, FluidSceneData, SceneData};

pub fn build_scene_data(
    time_step: &TimeStepInfo,
    cut: &crate::cut::Cut,
    boundary_hidden: bool,
    particle_radius: f32,
) -> SceneData {
    let fluid = build_fluid(&time_step.fluid, cut, particle_radius);
    let boundary = if boundary_hidden {
        BoundarySceneData::None
    } else {
        build_boundary(&time_step.boundary, cut, particle_radius)
    };
    SceneData { fluid, boundary }
}

fn build_fluid(vis: &FluidVisualization, cut: &crate::cut::Cut, radius: f32) -> FluidSceneData {
    match vis {
        FluidVisualization::Samples {
            positions,
            coloring,
        } => {
            let colors = resolve_fluid_coloring(coloring, positions.len());
            let instances: Vec<BillboardInstance> = positions
                .iter()
                .zip(colors.iter())
                .filter(|(pos, _)| cut.cut(pos))
                .map(|(pos, color)| BillboardInstance {
                    center: [pos[0], pos[2], -pos[1]],
                    radius,
                    color: *color,
                })
                .collect();

            if instances.is_empty() {
                FluidSceneData::None
            } else {
                FluidSceneData::Particles { instances }
            }
        }
        FluidVisualization::TriangleMesh { mesh } => {
            build_mesh_data(mesh, [0.3, 0.6, 0.9, 0.5], false)
        }
        FluidVisualization::SensorPlane { planes } => build_sensor_plane_data(planes),
    }
}

fn build_boundary(
    vis: &BoundaryVisualization,
    cut: &crate::cut::Cut,
    radius: f32,
) -> BoundarySceneData {
    match vis {
        BoundaryVisualization::Samples {
            positions,
            coloring,
        } => {
            let colors = resolve_boundary_coloring(coloring, positions.len());
            let instances: Vec<BillboardInstance> = positions
                .iter()
                .zip(colors.iter())
                .filter(|(pos, _)| cut.cut(pos))
                .map(|(pos, color)| BillboardInstance {
                    center: [pos[0], pos[2], -pos[1]],
                    radius,
                    color: *color,
                })
                .collect();

            if instances.is_empty() {
                BoundarySceneData::None
            } else {
                BoundarySceneData::Particles { instances }
            }
        }
        BoundaryVisualization::TriangleMesh { mesh, coloring } => {
            let color = match coloring {
                BoundaryMeshColoring::Original => [0.7, 0.7, 0.7, 1.0],
                BoundaryMeshColoring::Uniform => [0.6, 0.6, 0.6, 1.0],
            };
            let (vertices, indices) = mesh_to_cpu(mesh, color);
            if vertices.is_empty() {
                BoundarySceneData::None
            } else {
                BoundarySceneData::Mesh { vertices, indices }
            }
        }
    }
}

fn build_mesh_data(mesh: &RenderMesh, color: [f32; 4], transparent: bool) -> FluidSceneData {
    let (vertices, indices) = mesh_to_cpu(mesh, color);
    if vertices.is_empty() {
        FluidSceneData::None
    } else {
        FluidSceneData::Mesh {
            vertices,
            indices,
            transparent,
        }
    }
}

fn build_sensor_plane_data(planes: &[SensorPlaneData]) -> FluidSceneData {
    let mut all_vertices: Vec<ColoredMeshVertex> = Vec::new();
    let mut all_indices: Vec<u32> = Vec::new();

    for plane in planes {
        let values = extract_scalar_values(&plane.quantity);
        if plane.positions.is_empty() || values.is_empty() || plane.rows < 2 || plane.cols < 2 {
            continue;
        }

        let colors = colormap::values_to_colors(values, 1.0);
        let rows = plane.rows;
        let cols = plane.cols;
        let base_vertex = all_vertices.len() as u32;

        // Per-vertex normals
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
        FluidSceneData::None
    } else {
        FluidSceneData::SensorPlane {
            vertices: all_vertices,
            indices: all_indices,
        }
    }
}

fn mesh_to_cpu(mesh: &RenderMesh, color: [f32; 4]) -> (Vec<ColoredMeshVertex>, Vec<u32>) {
    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let vertices = mesh
        .vertices
        .iter()
        .map(|v| ColoredMeshVertex {
            position: [
                v.position[0] as f32,
                v.position[2] as f32,
                -v.position[1] as f32,
            ],
            normal: [v.normal[0] as f32, v.normal[2] as f32, -v.normal[1] as f32],
            color,
        })
        .collect();
    (vertices, mesh.indices.clone())
}

fn resolve_fluid_coloring(coloring: &FluidColoring, count: usize) -> Vec<[f32; 4]> {
    match coloring {
        FluidColoring::Uniform => vec![[0.2, 0.5, 1.0, 1.0]; count],
        FluidColoring::FluidId { val, max_id } => colormap::id_to_colors(val, *max_id, 1.0),
        FluidColoring::QuantityGraded { quantity } => {
            colormap::values_to_colors(extract_scalar_values(quantity), 1.0)
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
