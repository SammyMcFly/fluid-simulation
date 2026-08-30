//! Module that contains a representation of a collection of samples for an SPH fluid simulation
//!
use bincode::{Decode, Encode};
use nalgebra::{Point3, Vector3};
use parry3d_f64::shape::TriMesh;
use serde::{Deserialize, Serialize};
use splashsurf_lib::nalgebra::Vector3 as SurfVector3;
use splashsurf_lib::{SpatialDecomposition, SurfaceReconstruction, reconstruct_surface};
use std::collections::BTreeMap;
use std::slice::SliceIndex;

use crate::utilities::{
    sampling::sample_volume_shifted,
    triangle_mesh::{RenderMesh, RenderVertex},
};

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
pub struct Fluid {
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

impl Len for Fluid {
    fn len(&self) -> usize {
        self.num_active
    }
}

impl Positional for Fluid {
    fn pos_now<I>(&self, id: I) -> &I::Output
    where
        I: SliceIndex<[Point3<f64>]>,
    {
        &self.position[id]
    }
}

impl Fluid {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add_samples(
        &mut self,
        fluid: &TriMesh,
        fluid_id: u32,
        rest_density: f64,
        rest_density_grid_spacing: f64,
    ) {
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

    // fn push(&mut self, fluid_id: u32, position: Point3<f64>, velocity: Vector3<f64>, mass: f64) {
    //     self.fluid_id.push(fluid_id);
    //     self.position.push(position);
    //     self.position_prev.push(position);
    //     self.position_pred.push(Point3::origin());
    //     self.velocity.push(velocity);
    //     self.velocity_prev.push(Vector3::zeros());
    //     self.velocity_pred.push(Vector3::zeros());
    //     self.acceleration.push(Vector3::zeros());
    //     self.mass.push(mass);
    //     self.volume.push(0.);
    //     self.pressure.push(0.);

    //     let insert_at = self.num_active;
    //     let last = self.position.len() - 1;

    //     if insert_at != last {
    //         self.swap(insert_at, last);
    //     }

    //     self.num_active += 1;
    // }

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

    /// Reconstruct one mesh per distinct fluid_id.
    /// Returns (fluid_id, mesh) pairs; empty meshes are skipped.
    pub fn reconstruct_surfaces(
        &self,
        rest_density_grid_spacing: f64,
        rest_volume: f64,
        kernel_support_radius: f64,
    ) -> Vec<(u32, RenderMesh)> {
        // Group active particle positions by fluid_id, and remember one mass per group.
        // BTreeMap keeps a stable, sorted iteration order.
        let mut groups: BTreeMap<u32, (Vec<SurfVector3<f64>>, f64)> = BTreeMap::new();

        for i in 0..self.num_active {
            let id = self.fluid_id[i];
            let p = self.position[i];
            let entry = groups
                .entry(id)
                .or_insert_with(|| (Vec::new(), self.mass[i]));
            entry.0.push(SurfVector3::new(p.x, p.y, p.z));
        }

        groups
            .into_iter()
            .filter_map(|(id, (positions, mass))| {
                let rest_density = mass / rest_volume;
                let mesh = Self::reconstruct_single(
                    &positions,
                    rest_density_grid_spacing,
                    rest_density,
                    kernel_support_radius,
                );
                // skip empty reconstructions so the renderer doesn't get zero-index buffers
                (!mesh.indices.is_empty()).then_some((id, mesh))
            })
            .collect()
    }

    /// Reconstruct a single surface from a position slice and one rest density.
    fn reconstruct_single(
        positions: &[SurfVector3<f64>],
        rest_density_grid_spacing: f64,
        rest_density: f64,
        kernel_support_radius: f64,
    ) -> RenderMesh {
        let particle_radius = 0.5 * rest_density_grid_spacing;

        let params = splashsurf_lib::Parameters {
            particle_radius,
            rest_density,
            compact_support_radius: kernel_support_radius,
            cube_size: 0.75 * particle_radius,
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

        // Reconstruct; on failure return an empty mesh instead of panicking.
        let reconstruction: SurfaceReconstruction<i64, f64> =
            match reconstruct_surface(positions, &params) {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!("surface reconstruction failed: {e}");
                    return RenderMesh {
                        vertices: Vec::new(),
                        indices: Vec::new(),
                    };
                }
            };

        let mesh = reconstruction.mesh;
        let triangles = mesh.triangles.as_slice();

        let indices: Vec<u32> = triangles
            .iter()
            .flat_map(|tri| tri.iter().map(|&i| i as u32))
            .collect();

        let positions: Vec<[f64; 3]> = mesh.vertices.iter().map(|v| [v.x, v.y, v.z]).collect();

        // per-vertex normals (area-weighted)
        let mut normals = vec![[0.0f64; 3]; positions.len()];
        for tri in triangles {
            let v0 = positions[tri[0]];
            let v1 = positions[tri[1]];
            let v2 = positions[tri[2]];
            let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
            let fn_ = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            for &idx in tri.iter() {
                normals[idx][0] += fn_[0];
                normals[idx][1] += fn_[1];
                normals[idx][2] += fn_[2];
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

        let render_vertices: Vec<RenderVertex> = positions
            .iter()
            .zip(normals.iter())
            .map(|(pos, norm)| RenderVertex {
                position: *pos,
                normal: *norm,
            })
            .collect();

        #[cfg(feature = "logging")]
        tracing::debug!(
            "reconstructed fluid: {} verts, {} tris (rho={})",
            render_vertices.len(),
            indices.len() / 3,
            rest_density
        );

        RenderMesh {
            vertices: render_vertices,
            indices,
        }
    }
}

/// Compressed and serializable fluid, i.e. a collection of
/// samples, in a 3-dimensional context.
#[derive(Debug, Clone, Default)]
pub struct FluidCheckpoint {
    pub fluid_id: Vec<u32>,
    pub position: Vec<Point3<f64>>,
    pub velocity: Vec<Vector3<f64>>,
    pub mass: Vec<f64>,
}

impl From<Fluid> for FluidCheckpoint {
    fn from(fluid: Fluid) -> Self {
        Self {
            fluid_id: fluid.fluid_id,
            position: fluid.position,
            velocity: fluid.velocity,
            mass: fluid.mass,
        }
    }
}

impl From<FluidCheckpoint> for Fluid {
    fn from(fluid_checkpoint: FluidCheckpoint) -> Self {
        let len = fluid_checkpoint.position.len();
        Self {
            num_active: len,
            fluid_id: fluid_checkpoint.fluid_id,
            position: fluid_checkpoint.position,
            position_prev: vec![Point3::origin(); len],
            position_pred: vec![Point3::origin(); len],
            velocity: fluid_checkpoint.velocity,
            velocity_prev: vec![Vector3::zeros(); len],
            velocity_pred: vec![Vector3::zeros(); len],
            acceleration: vec![Vector3::zeros(); len],
            mass: fluid_checkpoint.mass,
            volume: vec![0.; len],
            pressure: vec![0.; len],
        }
    }
}

/// Compressed and serializable fluid, i.e. a collection of
/// samples, in a 3-dimensional context.
#[derive(Debug, Clone, Default, Serialize, Deserialize, Encode, Decode)]
pub struct SerFluidCheckpoint {
    pub fluid_id: Vec<u32>,
    pub position: Vec<[f64; 3]>,
    pub velocity: Vec<[f64; 3]>,
    pub mass: Vec<f64>,
}

impl From<FluidCheckpoint> for SerFluidCheckpoint {
    fn from(fluid: FluidCheckpoint) -> Self {
        Self {
            fluid_id: fluid.fluid_id,
            position: fluid.position.iter().map(|pos| (*pos).into()).collect(),
            velocity: fluid.velocity.iter().map(|vel| (*vel).into()).collect(),
            mass: fluid.mass,
        }
    }
}

impl From<SerFluidCheckpoint> for FluidCheckpoint {
    fn from(ser_fluid: SerFluidCheckpoint) -> Self {
        Self {
            fluid_id: ser_fluid.fluid_id,
            position: ser_fluid.position.iter().map(|pos| (*pos).into()).collect(),
            velocity: ser_fluid.velocity.iter().map(|vel| (*vel).into()).collect(),
            mass: ser_fluid.mass,
        }
    }
}
