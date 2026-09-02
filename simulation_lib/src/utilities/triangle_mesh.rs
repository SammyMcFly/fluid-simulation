//! Triangle mesh library
use bincode::{Decode, Encode};
use nalgebra::{Matrix4, Point3, Rotation3, Vector3};
use parry3d_f64::math::Vec3;
use parry3d_f64::shape::{TriMesh, TriMeshFlags};
use serde::{Deserialize, Serialize};

use crate::sph::setup::input::VertexNormalRenderOption;

/// Errors that can occur while loading a triangle mesh asset.
#[derive(Debug, thiserror::Error)]
pub enum MeshError {
    /// The `.obj`/`.mtl` file could not be read or parsed.
    #[error("failed to load mesh '{path}': {source}")]
    Obj {
        path: String,
        #[source]
        source: tobj::LoadError,
    },
}

/// Mesh handle which references a mesh in the [[MeshLibrary]]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshHandle {
    pub idx: usize,
    pub mesh_id: u32,
}

/// Raw loaded geometry
#[derive(Debug, Clone)]
pub struct LoadedMesh {
    /// Vertex positions from OBJ
    pub positions: Vec<Point3<f64>>,
    /// Vertex normals from OBJ
    pub normals: Vec<Vector3<f64>>,
    pub indices: Vec<[u32; 3]>,
}

/// Render vertex (for wgpu)
#[repr(C)]
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    bytemuck::Pod,
    bytemuck::Zeroable,
    PartialEq,
    Serialize,
    Deserialize,
    Encode,
    Decode,
)]
pub struct RenderVertex {
    pub position: [f64; 3],
    pub normal: [f64; 3],
}

/// Render mesh (for wgpu)
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, Encode, Decode)]
pub struct RenderMesh {
    pub vertices: Vec<RenderVertex>,
    pub indices: Vec<u32>,
}

impl RenderMesh {
    pub fn extend(&mut self, other: Self) {
        let offset = self.vertices.len() as u32;
        self.vertices.extend(other.vertices);
        self.indices
            .extend(other.indices.into_iter().map(|i| i + offset));
    }

    pub fn from_loaded_mesh(trimesh: &LoadedMesh) -> Self {
        let vertices: Vec<RenderVertex> = trimesh
            .positions
            .iter()
            .zip(trimesh.normals.iter())
            .map(|(p, n)| RenderVertex {
                position: [p.x, p.y, p.z],
                normal: [n.x, n.y, n.z],
            })
            .collect();

        let indices: Vec<u32> = trimesh.indices.iter().flat_map(|t| *t).collect();

        RenderMesh { vertices, indices }
    }
    pub fn from_trimesh(trimesh: &parry3d_f64::shape::TriMesh) -> Self {
        let vertices: Vec<Point3<f64>> = trimesh
            .vertices()
            .iter()
            .map(|v| Point3::new(v.x, v.y, v.z))
            .collect();
        let triangles = trimesh.indices();

        // Flatten triangle indices
        let indices: Vec<u32> = triangles
            .iter()
            .flat_map(|tri| [tri[0], tri[1], tri[2]])
            .collect();

        // Compute angle-weighted vertex normals
        let mut normals = vec![Vector3::<f64>::zeros(); vertices.len()];

        for tri in triangles {
            let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
            let p0 = &vertices[tri[0] as usize];
            let p1 = &vertices[tri[1] as usize];
            let p2 = &vertices[tri[2] as usize];

            let e01 = p1 - p0;
            let e02 = p2 - p0;
            let e10 = p0 - p1;
            let e12 = p2 - p1;
            let e20 = p0 - p2;
            let e21 = p1 - p2;

            let face_normal = e01.cross(&e02);
            let area = face_normal.norm();
            if area < 1e-10 {
                continue; // degenerate triangle
            }
            let face_normal = face_normal / area;

            // Interior angle at each vertex
            let angle0 = e01
                .normalize()
                .dot(&e02.normalize())
                .clamp(-1.0, 1.0)
                .acos();
            let angle1 = e10
                .normalize()
                .dot(&e12.normalize())
                .clamp(-1.0, 1.0)
                .acos();
            let angle2 = e20
                .normalize()
                .dot(&e21.normalize())
                .clamp(-1.0, 1.0)
                .acos();

            normals[i0] += face_normal * angle0;
            normals[i1] += face_normal * angle1;
            normals[i2] += face_normal * angle2;
        }

        // Normalize
        for n in &mut normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if len > 1e-10 {
                n[0] /= len;
                n[1] /= len;
                n[2] /= len;
            }
        }

        // Build render vertices
        let render_vertices: Vec<RenderVertex> = vertices
            .iter()
            .zip(normals.iter())
            .map(|(pos, n)| RenderVertex {
                position: [pos.x, pos.y, pos.z],
                normal: [n.x, n.y, n.z],
            })
            .collect();

        RenderMesh {
            vertices: render_vertices,
            indices,
        }
    }
}

/// Combined mesh (raw OBJ data, parry3d_f64 TriMesh, and wgpu RenderMesh)
#[derive(Debug, Clone)]
pub struct MeshContainer {
    raw: LoadedMesh,
    /// For SDF queries (angle-weighted pseudo normals for sign determination)
    trimesh: Option<TriMesh>,
    /// For wgpu rendering (angle-weighted smooth normals for shading)
    render: Option<RenderMesh>,
}

impl MeshContainer {
    pub fn new(raw: LoadedMesh) -> Self {
        Self {
            raw,
            trimesh: None,
            render: None,
        }
    }

    /// Get or build parry3d_f64 TriMesh with ORIENTED flag
    /// (angle-weighted pseudo normals for signed distance)
    pub fn trimesh(&mut self) -> &TriMesh {
        if self.trimesh.is_none() {
            self.trimesh = Some(
                TriMesh::with_flags(
                    self.raw
                        .positions
                        .iter()
                        .map(|p| Vec3::new(p.x, p.y, p.z))
                        .collect(),
                    self.raw.indices.clone(),
                    TriMeshFlags::ORIENTED
                        | TriMeshFlags::MERGE_DUPLICATE_VERTICES
                        | TriMeshFlags::FIX_INTERNAL_EDGES,
                )
                .expect("Could not convert mesh to parry3d Trimesh."),
            );
        }
        self.trimesh.as_ref().unwrap()
    }

    /// Get or build render mesh with angle-weighted smooth normals
    pub fn render_mesh(&mut self, render_vertex_normals: VertexNormalRenderOption) -> &RenderMesh {
        if self.render.is_none() {
            match render_vertex_normals {
                VertexNormalRenderOption::FaceNormals => {
                    self.render = Some(RenderMesh::from_loaded_mesh(self.raw()));
                }
                VertexNormalRenderOption::AngleWeightedPseudoNormals => {
                    self.render = Some(RenderMesh::from_trimesh(self.trimesh()));
                }
            }
        }
        self.render.as_ref().unwrap()
    }

    pub fn raw(&self) -> &LoadedMesh {
        &self.raw
    }

    pub fn transform(
        &mut self,
        translation: &[f64; 3],
        rotation_euler_deg: &[f64; 3],
        scale: &[f64; 3],
    ) {
        debug_assert!(scale[0] >= 0.);
        debug_assert!(scale[1] >= 0.);
        debug_assert!(scale[2] >= 0.);

        let transform = build_transform(translation, rotation_euler_deg, scale);

        if transform != Matrix4::identity() {
            self.raw.positions.iter_mut().for_each(|v| {
                *v = transform.transform_point(v);
            });
            self.trimesh = None;
            self.render = None;
        }
    }
}

/// Mesh library
#[derive(Debug, Clone, Default)]
pub struct MeshLibrary {
    pub meshes: Vec<MeshContainer>,
}

impl MeshLibrary {
    pub fn load_obj(&mut self, path: &str) -> Result<(), MeshError> {
        let (models, _) =
            tobj::load_obj(path, &tobj::GPU_LOAD_OPTIONS).map_err(|source| MeshError::Obj {
                path: path.to_string(),
                source,
            })?;

        let mut positions: Vec<Point3<f64>> = Vec::new();
        let mut normals: Vec<Vector3<f64>> = Vec::new();
        let mut indices: Vec<[u32; 3]> = Vec::new();

        for model in &models {
            let m = &model.mesh;
            let offset = positions.len() as u32;

            for i in (0..m.positions.len()).step_by(3) {
                positions.push(Point3::new(
                    m.positions[i].into(),
                    m.positions[i + 1].into(),
                    m.positions[i + 2].into(),
                ));
            }
            if !m.normals.is_empty() {
                for i in (0..m.normals.len()).step_by(3) {
                    normals.push(Vector3::new(
                        m.normals[i] as f64,
                        m.normals[i + 1] as f64,
                        m.normals[i + 2] as f64,
                    ));
                }
            }
            for tri in m.indices.chunks_exact(3) {
                indices.push([tri[0] + offset, tri[1] + offset, tri[2] + offset]);
            }
        }

        self.meshes.push(MeshContainer::new(LoadedMesh {
            positions,
            normals,
            indices,
        }));
        Ok(())
    }

    pub fn get_mesh_container(&self, handle: MeshHandle) -> &MeshContainer {
        &self.meshes[handle.idx]
    }
}

/// Build a 4x4 affine transform: Scale → Rotate (Euler XYZ) → Translate
pub fn build_transform(
    position: &[f64; 3],
    rotation_euler_deg: &[f64; 3],
    scale: &[f64; 3],
) -> Matrix4<f64> {
    let translation =
        Matrix4::new_translation(&Vector3::new(position[0], position[1], position[2]));

    // Convert degrees to radians
    let rx = rotation_euler_deg[0].to_radians();
    let ry = rotation_euler_deg[1].to_radians();
    let rz = rotation_euler_deg[2].to_radians();

    // Euler angles (intrinsic XYZ convention — adjust if needed)
    let rotation = Rotation3::from_euler_angles(rx, ry, rz).to_homogeneous();

    let scale = Matrix4::new_nonuniform_scaling(&Vector3::new(scale[0], scale[1], scale[2]));

    // Order: scale first, then rotate, then translate
    translation * rotation * scale
}
