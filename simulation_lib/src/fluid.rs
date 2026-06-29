/// Module that contains a representation of a collection of samples for an SPH fluid simulation
///
use bincode::{Decode, Encode};
use nalgebra::{Point3, Vector3};
use parry3d_f64::shape::TriMesh;
use serde::{Deserialize, Serialize};
use std::slice::SliceIndex;
use splashsurf_lib::{reconstruct_surface, Parameters, SpatialDecomposition, SurfaceReconstruction};
use splashsurf_lib::nalgebra::Vector3 as SurfVector3;

use crate::utilities::{sampling::sample_volume_shifted, triangle_mesh::{RenderMesh, RenderVertex}};

pub trait Len {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait Positional {
    fn pos_now<I>(&self, id: I) -> &I::Output
    where
        I: SliceIndex<[Point3<f64>]>;
}

/// Fluid, i.e. a collection of samples, which are identified by an ID (usize)
///
#[derive(Debug, Clone, Default)]
pub struct Fluid3D {
    num_active: usize,
    pub fluid_id: Vec<u32>,
    pub position: Vec<Point3<f64>>,
    pub position_prev: Vec<Point3<f64>>,
    pub position_pred: Vec<Point3<f64>>,
    pub velocity: Vec<Vector3<f64>>,
    pub velocity_prev: Vec<Vector3<f64>>,
    pub velocity_pred: Vec<Vector3<f64>>,
    pub acceleration: Vec<Vector3<f64>>,
    pub mass: Vec<f64>,
    /// volume (necessary for sph fluid)
    pub volume: Vec<f64>,
    pub pressure: Vec<f64>,
}

impl Len for Fluid3D {
    fn len(&self) -> usize {
        self.num_active
    }
}

impl Positional for Fluid3D {
    fn pos_now<I>(&self, id: I) -> &I::Output
    where
        I: SliceIndex<[Point3<f64>]>,
    {
        &self.position[id]
    }
}

impl Fluid3D {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add_samples(&mut self, fluid: &TriMesh, fluid_id: u32, rest_density: f64, rest_density_grid_spacing: f64) {
        let position = sample_volume_shifted(fluid, rest_density_grid_spacing);
        let mass = rest_density * rest_density_grid_spacing.powi(3);
        let len = position.len();
        let fluid = Self {
            num_active: len,
            fluid_id: vec![fluid_id; len],
            position,
            position_prev: vec![Point3::origin(); len],
            position_pred: vec![Point3::origin(); len],
            velocity: vec![Vector3::zeros(); len],
            velocity_prev: vec![Vector3::zeros(); len],
            velocity_pred: vec![Vector3::zeros(); len],
            acceleration: vec![Vector3::zeros(); len],
            mass: vec![mass; len],
            volume: vec![0.; len],
            pressure: vec![0.; len],
        };
        self.extend(fluid);
    }

    fn push(&mut self, fluid_id: u32, position: Point3<f64>, velocity: Vector3<f64>, mass: f64) {
        self.fluid_id.push(fluid_id);
        self.position.push(position);
        self.position_prev.push(position);
        self.position_pred.push(Point3::origin());
        self.velocity.push(velocity);
        self.velocity_prev.push(Vector3::zeros());
        self.velocity_pred.push(Vector3::zeros());
        self.acceleration.push(Vector3::zeros());
        self.mass.push(mass);
        self.volume.push(0.);
        self.pressure.push(0.);

        let insert_at = self.num_active;
        let last = self.position.len() - 1;

        if insert_at != last {
            self.swap(insert_at, last);
        }

        self.num_active += 1;
    }

    fn extend(&mut self, other: Self) {
        assert!(self.num_active == self.total_len());
        self.num_active += other.num_active;
        self.fluid_id.extend(other.fluid_id);
        self.position.extend(other.position);
        self.position_prev.extend(other.position_prev);
        self.position_pred.extend(other.position_pred);
        self.velocity.extend(other.velocity);
        self.velocity_prev.extend(other.velocity_prev);
        self.velocity_pred.extend(other.velocity_pred);
        self.acceleration.extend(other.acceleration);
        self.mass.extend(other.mass);
        self.volume.extend(other.volume);
        self.pressure.extend(other.pressure);
    }

    /// Gesamtzahl inkl. inaktiver
    pub fn total_len(&self) -> usize {
        self.position.len()
    }

    pub fn rotate_position(&mut self) {
        std::mem::swap(&mut self.position_prev, &mut self.position);
        std::mem::swap(&mut self.position, &mut self.position_pred);
    }

    pub fn rotate_velocity(&mut self) {
        std::mem::swap(&mut self.velocity_prev, &mut self.velocity);
        std::mem::swap(&mut self.velocity, &mut self.velocity_pred);
    }

    pub fn disable(&mut self, id: usize) {
        assert!(id < self.num_active);
        self.num_active -= 1;
        self.swap(id, self.num_active);
    }

    fn swap(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        self.fluid_id.swap(a, b);
        self.position.swap(a, b);
        self.position_prev.swap(a, b);
        self.position_pred.swap(a, b);
        self.velocity.swap(a, b);
        self.velocity_prev.swap(a, b);
        self.velocity_pred.swap(a, b);
        self.acceleration.swap(a, b);
        self.mass.swap(a, b);
        self.volume.swap(a, b);
        self.pressure.swap(a, b);
    }

    pub fn drop_inactive(&mut self) {
        self.fluid_id.truncate(self.num_active);
        self.position.truncate(self.num_active);
        self.position_prev.truncate(self.num_active);
        self.position_pred.truncate(self.num_active);
        self.velocity.truncate(self.num_active);
        self.velocity_prev.truncate(self.num_active);
        self.velocity_pred.truncate(self.num_active);
        self.acceleration.truncate(self.num_active);
        self.mass.truncate(self.num_active);
        self.volume.truncate(self.num_active);
        self.pressure.truncate(self.num_active);
    }

    pub fn reconstruct_surface(&self) -> RenderMesh {
        let params = Parameters {
            particle_radius: 0.011,
            rest_density: 1000.0,
            compact_support_radius: 0.022,
            cube_size: 0.016,
            iso_surface_threshold: 0.6,
            #[cfg(not(feature = "parallel"))]
            enable_multi_threading: false,
            #[cfg(feature = "parallel")]
            enable_multi_threading: true,
            enable_simd: true,
            global_neighborhood_list: false,
            particle_aabb: None,
            spatial_decomposition: SpatialDecomposition::None,
        };

        let positions: Vec<SurfVector3<f64>> = self
            .position
            .iter()
            .map(|p| SurfVector3::new(p.x, p.y, p.z))
            .collect();

        let reconstruction: SurfaceReconstruction<i64, f64> = reconstruct_surface(
            &positions,
            &params
        )
            .expect("Failed to reconstruct surface");

        let mesh = reconstruction.mesh;
        let vertices = mesh.vertices.as_slice();
        let triangles = mesh.triangles.as_slice();

        // Flatten triangle indices to u32
        let indices: Vec<u32> = triangles
            .iter()
            .flat_map(|tri| tri.iter().map(|&i| i as u32))
            .collect();

        // Work with plain [f64; 3] to avoid nalgebra version conflicts
        let positions: Vec<[f64; 3]> = vertices
            .iter()
            .map(|v| [v.x, v.y, v.z])
            .collect();

        // Compute per-vertex normals (area-weighted average of adjacent face normals)
        let mut normals = vec![[0.0f64; 3]; positions.len()];

        for tri in triangles {
            let v0 = positions[tri[0]];
            let v1 = positions[tri[1]];
            let v2 = positions[tri[2]];

            // edge vectors
            let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

            // cross product
            let face_normal = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];

            for &idx in tri.iter() {
                normals[idx][0] += face_normal[0];
                normals[idx][1] += face_normal[1];
                normals[idx][2] += face_normal[2];
            }
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

        // Build RenderVertex list
        let render_vertices: Vec<RenderVertex> = positions
            .iter()
            .zip(normals.iter())
            .map(|(pos, norm)| RenderVertex {
                position: *pos,
                normal: *norm,
            })
            .collect();

        RenderMesh {
            vertices: render_vertices,
            indices,
        }
    }
}

impl From<SerFluid3D> for Fluid3D {
    fn from(ser_fluid: SerFluid3D) -> Self {
        let len = ser_fluid.position.len();
        Self {
            num_active: len,
            fluid_id: ser_fluid.fluid_id,
            position: ser_fluid.position.iter().map(|pos| (*pos).into()).collect(),
            position_prev: vec![Point3::origin(); len],
            position_pred: vec![Point3::origin(); len],
            velocity: ser_fluid.velocity.iter().map(|vel| (*vel).into()).collect(),
            velocity_prev: vec![Vector3::zeros(); len],
            velocity_pred: vec![Vector3::zeros(); len],
            acceleration: vec![Vector3::zeros(); len],
            mass: vec![ser_fluid.mass; len],
            volume: vec![0.; len],
            pressure: vec![0.; len],
        }
    }
}

/// Compressed and serializable fluid, i.e. a collection of
/// samples, in a 3-dimensional context.
#[derive(Debug, Clone, Default, Serialize, Deserialize, Encode, Decode)]
pub struct SerFluid3D {
    pub fluid_id: Vec<u32>,
    pub position: Vec<[f64; 3]>,
    pub velocity: Vec<[f64; 3]>,
    pub mass: f64,
}

impl From<Fluid3D> for SerFluid3D {
    fn from(fluid: Fluid3D) -> Self {
        Self {
            fluid_id: fluid.fluid_id,
            position: fluid.position.iter().map(|pos| (*pos).into()).collect(),
            velocity: fluid.velocity.iter().map(|vel| (*vel).into()).collect(),
            mass: fluid.mass[0],
        }
    }
}
