//! Module that contains a representation of a collection of samples for an SPH fluid simulation
//!
use bincode::{Decode, Encode};
use nalgebra::{Point3, Vector3};
use parry3d_f64::shape::TriMesh;
use serde::{Deserialize, Serialize};
use splashsurf_lib::nalgebra::Vector3 as SurfVector3;
use splashsurf_lib::{SpatialDecomposition, SurfaceReconstruction, reconstruct_surface};
use std::collections::BTreeMap;

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

/// Fluid, i.e. a collection of samples, which are identified by an ID (usize)
///
#[derive(Debug, Clone, Default)]
pub struct Fluid {
    num_active: usize,
    pub fluid_id: Vec<u32>,
    pub position: Vec<Point3<f64>>,
    pub velocity: Vec<Vector3<f64>>,
    pub acceleration: Vec<Vector3<f64>>,
    pub mass: Vec<f64>,
    /// volume (necessary for sph fluid)
    pub volume: Vec<f64>,
    pub pressure: Vec<f64>,
    /// Scratch position slots owned by the active [IntegrationScheme].
    /// Count and meaning are scheme-specific (e.g. [Verlet] uses slot 0 to
    /// store the position of the previous time step).
    ///
    /// Sized to [Self::total_len]` by [Self::resize_slots]; kept in sync
    /// by [Self::swap]/[Self::disable]/[Self::drop_inactive] like every other
    /// per-particle field.
    pub integrator_position_slots: Vec<Vec<Point3<f64>>>,
    /// Scratch velocity slots owned by the active [IntegrationScheme].
    /// Count and meaning are scheme-specific.
    ///
    /// Sized to [Self::total_len]` by [Self::resize_slots]; kept in sync
    /// by [Self::swap]/[Self::disable]/[Self::drop_inactive] like every other
    /// per-particle field.
    pub integrator_velocity_slots: Vec<Vec<Vector3<f64>>>,
    /// Scratch position slots owned by the active [PressureSolver].
    /// Count and meaning are scheme-specific (e.g. [IISPH]/[IISPHwOST]
    /// both use slot 0 as predicted position).
    ///
    /// Sized to [Self::total_len]` by [Self::resize_slots]; kept in sync
    /// by [Self::swap]/[Self::disable]/[Self::drop_inactive] like every other
    /// per-particle field.
    pub solver_position_slots: Vec<Vec<Point3<f64>>>,
    /// Scratch position slots owned by the active [PressureSolver].
    /// Count and meaning are scheme-specific (e.g. [IISPH]/[IISPHwOST]
    /// both use slot 0 as predicted velocity).
    ///
    /// Sized to [Self::total_len]` by [Self::resize_slots]; kept in sync
    /// by [Self::swap]/[Self::disable]/[Self::drop_inactive] like every other
    /// per-particle field.
    pub solver_velocity_slots: Vec<Vec<Vector3<f64>>>,
}

impl Len for Fluid {
    fn len(&self) -> usize {
        self.num_active
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
            velocity: vec![Vector3::zeros(); len],
            acceleration: vec![Vector3::zeros(); len],
            mass: vec![mass; len],
            volume: vec![0.; len],
            pressure: vec![0.; len],
            integrator_position_slots: Vec::new(),
            integrator_velocity_slots: Vec::new(),
            solver_position_slots: Vec::new(),
            solver_velocity_slots: Vec::new(),
        };
        self.extend(fluid);
    }

    /// (Re-)sizes the per-role scratch slot pools to [Self::total_len], according to
    /// the counts declared by the active [IntegrationScheme]/[PressureSolver].
    ///
    /// Must be called once [Fluid] is fully populated and [IntegrationScheme]/[PressureSolver]
    /// are known -- [Self::add_samples]/[Self::extend] deliberately leave these pools empty,
    /// since the required counts aren't known until then. See [System::new_boxed] and
    /// [SPHSystem::continue_from_checkpoint].
    pub fn resize_slots(
        &mut self,
        integrator_position_slots: usize,
        integrator_velocity_slots: usize,
        pressure_solver_position_slots: usize,
        pressure_solver_velocity_slots: usize,
    ) {
        let len = self.total_len();
        self.integrator_position_slots = vec![vec![Point3::origin(); len]; integrator_position_slots];
        self.integrator_velocity_slots = vec![vec![Vector3::zeros(); len]; integrator_velocity_slots];
        self.solver_position_slots = vec![vec![Point3::origin(); len]; pressure_solver_position_slots];
        self.solver_velocity_slots = vec![vec![Vector3::zeros(); len]; pressure_solver_velocity_slots];
    }

    fn extend(&mut self, other: Self) {
        assert_eq!(self.num_active, self.total_len());
        self.num_active += other.num_active;
        self.fluid_id.extend(other.fluid_id);
        self.position.extend(other.position);
        self.velocity.extend(other.velocity);
        self.acceleration.extend(other.acceleration);
        self.mass.extend(other.mass);
        self.volume.extend(other.volume);
        self.pressure.extend(other.pressure);
        // Slots are intentionally left untouched here: `extend` is only ever
        // called (via `add_samples`) before `resize_slots` runs, so they're
        // still empty at this point. `resize_slots` sizes them correctly
        // afterward, once `I`/`P` are known.
        assert!(
            self.integrator_position_slots.is_empty()
                && self.integrator_velocity_slots.is_empty()
                && self.solver_position_slots.is_empty()
                && self.solver_velocity_slots.is_empty(),
            "extend() called after resize_slots(); slot growth on extend is not implemented"
        );
    }

    /// Total numer of fluid samples contained in [Fluid].
    pub fn total_len(&self) -> usize {
        self.position.len()
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
        self.velocity.swap(a, b);
        self.acceleration.swap(a, b);
        self.mass.swap(a, b);
        self.volume.swap(a, b);
        self.pressure.swap(a, b);
        for slot in &mut self.integrator_position_slots {
            slot.swap(a, b);
        }
        for slot in &mut self.integrator_velocity_slots {
            slot.swap(a, b);
        }
        for slot in &mut self.solver_position_slots {
            slot.swap(a, b);
        }
        for slot in &mut self.solver_velocity_slots {
            slot.swap(a, b);
        }
    }

    pub fn drop_inactive(&mut self) {
        self.fluid_id.truncate(self.num_active);
        self.position.truncate(self.num_active);
        self.velocity.truncate(self.num_active);
        self.acceleration.truncate(self.num_active);
        self.mass.truncate(self.num_active);
        self.volume.truncate(self.num_active);
        self.pressure.truncate(self.num_active);
        for slot in &mut self.integrator_position_slots {
            slot.truncate(self.num_active);
        }
        for slot in &mut self.integrator_velocity_slots {
            slot.truncate(self.num_active);
        }
        for slot in &mut self.solver_position_slots {
            slot.truncate(self.num_active);
        }
        for slot in &mut self.solver_velocity_slots {
            slot.truncate(self.num_active);
        }
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
            velocity: fluid_checkpoint.velocity,
            acceleration: vec![Vector3::zeros(); len],
            mass: fluid_checkpoint.mass,
            volume: vec![0.; len],
            pressure: vec![0.; len],
            // Left empty on purpose -- sized via `resize_slots` once `I`/`P`
            // are known; see `SPHSystem::continue_from_checkpoint`.
            integrator_position_slots: Vec::new(),
            integrator_velocity_slots: Vec::new(),
            solver_position_slots: Vec::new(),
            solver_velocity_slots: Vec::new(),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn pos(x: f64, y: f64, z: f64) -> Point3<f64> {
        Point3::new(x, y, z)
    }

    fn vel(x: f64, y: f64, z: f64) -> Vector3<f64> {
        Vector3::new(x, y, z)
    }

    /// Builds a `Fluid` with `num_active` explicitly controllable,
    /// independent of `positions.len()` — used to set up mismatched
    /// active/total states (e.g. for `extend`'s panic condition) that
    /// cannot be produced through the public API, since `num_active` is
    /// private and `FluidCheckpoint -> Fluid` always yields
    /// `num_active == len`.
    fn raw_fluid(num_active: usize, positions: Vec<Point3<f64>>, masses: Vec<f64>) -> Fluid {
        let len = positions.len();
        Fluid {
            num_active,
            fluid_id: vec![0; len],
            position: positions,
            velocity: vec![Vector3::zeros(); len],
            acceleration: vec![Vector3::zeros(); len],
            mass: masses,
            volume: vec![0.; len],
            pressure: vec![0.; len],
            integrator_position_slots: Vec::new(),
            integrator_velocity_slots: Vec::new(),
            solver_position_slots: Vec::new(),
            solver_velocity_slots: Vec::new(),
        }
    }

    // ─── extend (private) ──────────────────────────────────────────

    #[test]
    fn extend_appends_all_fields() {
        let mut a = raw_fluid(2, vec![pos(1., 0., 0.), pos(2., 0., 0.)], vec![1., 2.]);
        let b = raw_fluid(1, vec![pos(3., 0., 0.)], vec![3.]);

        a.extend(b);

        assert_eq!(a.len(), 3);
        assert_eq!(a.total_len(), 3);
        assert_eq!(a.position[2], pos(3., 0., 0.));
        assert_eq!(a.mass[2], 3.);
    }

    #[test]
    fn extend_from_empty_self() {
        // Mirrors `add_samples`'s actual usage: a fresh `Fluid::new()`
        // (`num_active == total_len == 0`) extended with the first batch of
        // sampled particles. `extend_appends_all_fields` starts from a
        // non-empty `self`; this covers the zero-particle starting case
        // separately, since `self.num_active == self.total_len()` trivially
        // holds at `0 == 0` but is worth confirming explicitly.
        let mut fluid = Fluid::new();
        let other = raw_fluid(1, vec![pos(1., 0., 0.)], vec![1.]);

        fluid.extend(other);

        assert_eq!(fluid.len(), 1);
        assert_eq!(fluid.position[0], pos(1., 0., 0.));
    }

    #[test]
    #[should_panic]
    fn extend_panics_when_self_has_inactive_particles() {
        // `extend` requires `self.num_active == self.total_len()`.
        // Appending while inactive particles exist at the tail would insert
        // new data past them instead of first truncating via
        // `drop_inactive`, silently corrupting the active/total invariant —
        // so this is asserted rather than handled.
        let mut a = raw_fluid(1, vec![pos(1., 0., 0.), pos(2., 0., 0.)], vec![1., 2.]);
        // a.num_active = 1 but total_len = 2 → mismatch
        let b = raw_fluid(1, vec![pos(3., 0., 0.)], vec![3.]);
        a.extend(b);
    }

    // ─── swap (private) ─────────────────────────────────────────────

    #[test]
    fn swap_exchanges_every_field() {
        // A dedicated, isolated white-box test of `swap` covering ALL
        // fields — including `position_prev`/`position_pred`/
        // `velocity_prev`/`velocity_pred`/`acceleration`/`volume`/
        // `pressure`, which the black-box `disable`-based tests in the
        // external test suite never exercise (they only ever check
        // `position`/`mass`). Catches e.g. "forgot to swap `pressure`" bugs
        // that would otherwise go unnoticed.
        let mut fluid = raw_fluid(2, vec![pos(1., 0., 0.), pos(2., 0., 0.)], vec![1., 2.]);
        fluid.fluid_id = vec![10, 20];
        fluid.velocity = vec![vel(1., 1., 1.), vel(2., 2., 2.)];
        fluid.acceleration = vec![vel(9., 0., 0.), vel(8., 0., 0.)];
        fluid.volume = vec![0.1, 0.2];
        fluid.pressure = vec![100., 200.];
        fluid.integrator_position_slots = vec![vec![pos(1.1, 0., 0.), pos(2.1, 0., 0.)]];
        fluid.integrator_velocity_slots = vec![vec![vel(1.2, 0., 0.), vel(2.2, 0., 0.)]];
        fluid.solver_position_slots = vec![vec![pos(1.3, 0., 0.), pos(2.3, 0., 0.)]];
        fluid.solver_velocity_slots = vec![vec![vel(1.4, 0., 0.), vel(2.4, 0., 0.)]];

        fluid.swap(0, 1);

        assert_eq!(fluid.fluid_id, vec![20, 10]);
        assert_eq!(fluid.position, vec![pos(2., 0., 0.), pos(1., 0., 0.)]);
        assert_eq!(fluid.velocity, vec![vel(2., 2., 2.), vel(1., 1., 1.)]);
        assert_eq!(fluid.acceleration, vec![vel(8., 0., 0.), vel(9., 0., 0.)]);
        assert_eq!(fluid.mass, vec![2., 1.]);
        assert_eq!(fluid.volume, vec![0.2, 0.1]);
        assert_eq!(fluid.pressure, vec![200., 100.]);
        assert_eq!(fluid.integrator_position_slots[0], vec![pos(2.1, 0., 0.), pos(1.1, 0., 0.)]);
        assert_eq!(fluid.integrator_velocity_slots[0], vec![vel(2.2, 0., 0.), vel(1.2, 0., 0.)]);
        assert_eq!(fluid.solver_position_slots[0], vec![pos(2.3, 0., 0.), pos(1.3, 0., 0.)]);
        assert_eq!(fluid.solver_velocity_slots[0], vec![vel(2.4, 0., 0.), vel(1.4, 0., 0.)]);
    }

    #[test]
    fn swap_same_index_is_noop() {
        let mut fluid = raw_fluid(1, vec![pos(1., 0., 0.)], vec![1.]);
        fluid.swap(0, 0);
        assert_eq!(fluid.position[0], pos(1., 0., 0.));
    }

    // ─── extend used to grow after disable + drop_inactive ──────────

    #[test]
    fn disable_then_drop_then_extend() {
        // The realistic "grow again after shrinking" scenario — `extend`
        // is private, so unlike its `push`-based predecessor, this
        // particular sequence can only be exercised from within the crate.
        let mut fluid = raw_fluid(
            3,
            vec![pos(1., 0., 0.), pos(2., 0., 0.), pos(3., 0., 0.)],
            vec![1., 2., 3.],
        );

        fluid.disable(1);
        fluid.drop_inactive();
        assert_eq!(fluid.len(), 2);
        assert_eq!(fluid.total_len(), 2);

        let more = raw_fluid(1, vec![pos(99., 0., 0.)], vec![99.]);
        fluid.extend(more);

        assert_eq!(fluid.len(), 3);
        assert_eq!(fluid.total_len(), 3);
        assert_eq!(fluid.position[2], pos(99., 0., 0.));
    }
}
