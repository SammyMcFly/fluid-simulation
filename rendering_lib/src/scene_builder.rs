//! scene_builder.rs – Build CPU-side SceneData from simulation TimeStepInfo

use simulation_lib::render_info::*;
use simulation_lib::utilities::triangle_mesh::RenderMesh;

use crate::colormap::{self, Colormap};
use crate::model::ColoredMeshVertex;
use crate::pipeline::{BillboardInstance, MeshPoseInstance};
use crate::primitive::{BoundaryMeshDraw, BoundarySceneData, FluidSceneData, SceneData};

pub fn build_scene_data(
    time_step: &TimeStepInfo,
    cut: &crate::cut::Cut,
    cut_boundary: bool,
    boundary_hidden: bool,
    bounbary_alpha: f32,
    particle_radius: f32,
    max_mapping: f32,
    colormap: Colormap,
) -> SceneData {
    let fluid = build_fluid(
        &time_step.fluid,
        cut,
        particle_radius,
        max_mapping,
        colormap,
    );
    let boundary = if boundary_hidden {
        BoundarySceneData::None
    } else {
        build_boundary(
            &time_step.boundary,
            cut,
            cut_boundary,
            bounbary_alpha,
            particle_radius,
            colormap,
        )
    };
    SceneData { fluid, boundary }
}

fn build_fluid(
    vis: &FluidVisualization,
    cut: &crate::cut::Cut,
    radius: f32,
    max_mapping: f32,
    colormap: Colormap,
) -> FluidSceneData {
    match vis {
        FluidVisualization::Samples {
            positions,
            coloring,
        } => {
            let colors =
                resolve_fluid_sample_coloring(coloring, positions.len(), max_mapping, colormap);
            let instances: Vec<BillboardInstance> = positions
                .iter()
                .zip(colors.iter())
                .filter(|(pos, _)| cut.cut(pos))
                .map(|(pos, color)| BillboardInstance {
                    center: *pos,
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
        FluidVisualization::TriangleMesh {
            meshes,
            max_fluid_id,
            coloring,
        } => build_fluid_meshes(meshes, *max_fluid_id, coloring, colormap),
        FluidVisualization::SensorPlane { planes, .. } => {
            build_sensor_plane_data(planes, max_mapping, colormap)
        }
    }
}

fn build_fluid_meshes(
    meshes: &[(u32, RenderMesh)],
    max_fluid_id: u32,
    coloring: &FluidMeshColoring,
    colormap: Colormap,
) -> FluidSceneData {
    let mut vertices: Vec<ColoredMeshVertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    const ALPHA: f32 = 0.5; // fluid surface transparency

    for (fluid_id, mesh) in meshes {
        if mesh.vertices.is_empty() || mesh.indices.is_empty() {
            continue;
        }

        // per-fluid color via the same colormap you use elsewhere
        let color = match coloring {
            FluidMeshColoring::FluidId => {
                let rgb = colormap::ids_to_colors(&[*fluid_id], max_fluid_id, colormap, 1.0)[0];
                [rgb[0], rgb[1], rgb[2], ALPHA]
            }
            FluidMeshColoring::Uniform => [0.5, 0.7, 0.8, ALPHA],
        };

        let base = vertices.len() as u32;
        vertices.extend(convert_vertices(mesh, color));
        indices.extend(mesh.indices.iter().map(|&i| i + base));
    }

    if vertices.is_empty() {
        FluidSceneData::None
    } else {
        FluidSceneData::Mesh {
            vertices,
            indices,
            transparent: true,
        }
    }
}

fn build_boundary(
    vis: &BoundaryVisualization,
    cut: &crate::cut::Cut,
    cut_boundary: bool,
    bounbary_alpha: f32,
    radius: f32,
    colormap: Colormap,
) -> BoundarySceneData {
    match vis {
        BoundaryVisualization::Samples {
            positions,
            coloring,
        } => {
            let colors = resolve_boundary_sample_coloring(
                coloring,
                positions.len(),
                colormap,
                bounbary_alpha,
            );
            let instances: Vec<BillboardInstance> = positions
                .iter()
                .zip(colors.iter())
                .filter(|(pos, _)| cut.cut(pos) || !cut_boundary)
                .map(|(pos, color)| BillboardInstance {
                    center: *pos,
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
        BoundaryVisualization::TriangleMesh { meshes, coloring } => {
            let colors = resolve_boundary_mesh_coloring(coloring, colormap, bounbary_alpha);
            let draws: Vec<BoundaryMeshDraw> = meshes
                .iter()
                .zip(colors.iter())
                .filter_map(|((mesh, pose), color)| {
                    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                        return None;
                    }
                    Some(BoundaryMeshDraw {
                        vertices: convert_vertices(mesh, *color),
                        indices: mesh.indices.clone(),
                        pose: MeshPoseInstance::from(pose),
                    })
                })
                .collect();

            if draws.is_empty() {
                BoundarySceneData::None
            } else {
                BoundarySceneData::Mesh { meshes: draws }
            }
        }
    }
}

fn build_sensor_plane_data(
    planes: &[SensorPlaneData],
    max_mapping: f32,
    colormap: Colormap,
) -> FluidSceneData {
    let mut all_vertices: Vec<ColoredMeshVertex> = Vec::new();
    let mut all_indices: Vec<u32> = Vec::new();

    for plane in planes {
        let values = &plane.data;
        if plane.positions.is_empty() || values.is_empty() || plane.rows < 2 || plane.cols < 2 {
            continue;
        }

        let colors = colormap::values_to_colors(values, max_mapping, colormap, 1.0);
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
                position: *pos,
                normal: *normal,
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

fn convert_vertices(mesh: &RenderMesh, color: [f32; 4]) -> Vec<ColoredMeshVertex> {
    mesh.vertices
        .iter()
        .map(|v| ColoredMeshVertex {
            position: [
                v.position[0] as f32,
                v.position[1] as f32,
                v.position[2] as f32,
            ],
            normal: [v.normal[0] as f32, v.normal[1] as f32, v.normal[2] as f32],
            color,
        })
        .collect()
}

fn resolve_fluid_sample_coloring(
    coloring: &FluidSampleColoring,
    count: usize,
    max_mapping: f32,
    colormap: Colormap,
) -> Vec<[f32; 4]> {
    match coloring {
        FluidSampleColoring::Uniform => vec![[0.2, 0.5, 1.0, 1.0]; count],
        FluidSampleColoring::FluidId { id, max_id } => {
            colormap::ids_to_colors(id, *max_id, colormap, 1.0)
        }
        FluidSampleColoring::QuantityGraded { data, .. } => {
            colormap::values_to_colors(data, max_mapping, colormap, 1.0)
        }
    }
}

fn resolve_boundary_sample_coloring(
    coloring: &BoundarySampleColoring,
    count: usize,
    colormap: Colormap,
    bounbary_alpha: f32,
) -> Vec<[f32; 4]> {
    match coloring {
        BoundarySampleColoring::Uniform => vec![[0.6, 0.6, 0.6, bounbary_alpha]; count],
        BoundarySampleColoring::BoundaryId { ids, max_id } => {
            colormap::ids_to_colors(ids, *max_id, colormap, bounbary_alpha)
        }
    }
}

fn resolve_boundary_mesh_coloring(
    coloring: &BoundaryMeshColoring,
    colormap: Colormap,
    bounbary_alpha: f32,
) -> Vec<[f32; 4]> {
    match coloring {
        BoundaryMeshColoring::Original => vec![[0.7, 0.7, 0.7, bounbary_alpha]],
        BoundaryMeshColoring::Uniform => vec![[0.6, 0.6, 0.6, bounbary_alpha]],
        BoundaryMeshColoring::BoundaryId { ids, max_id } => {
            colormap::ids_to_colors(ids, *max_id, colormap, bounbary_alpha)
        }
    }
}
