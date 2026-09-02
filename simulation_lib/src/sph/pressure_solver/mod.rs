//! Pressure solver algorithm module
pub mod iisph;
pub mod iisph_optimized_source_term;
pub mod sesph;
pub mod sesph_with_splitting;

pub use iisph::IISPH;
pub use iisph_optimized_source_term::IISPHwOST;
pub use sesph::SESPH;
pub use sesph_with_splitting::SESPHwSplitting;

use crate::for_each;
use crate::iteration::for_each_collect;
use crate::neighbor_search::NeighborList;
use crate::sph::CurrentSystemProperties;
use crate::sph::SystemParameters;
use crate::sph::boundary_handling::ForceOntoBoundary;
use crate::sph::boundary_handling::{BoundaryHandling, RequestMode};
use crate::sph::fluid::Fluid;
use crate::sph::kernel::KernelFn;
use crate::sph::setup::input::Parameters;
use crate::utilities::vector;

use nalgebra::Vector3;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub enum PressureSolverVariant {
    SESPH,
    SESPHwSplitting,
    IISPH,
    IISPHwOST,
}

pub trait PressureSolver: Send + Sync + Clone {
    /// Whether this solver correctly supports two-way coupling with
    /// *dynamic* (rigid-body) boundaries.
    ///
    /// Defaults to `true`; override to `false` for solvers with this
    /// limitation. Checked in `SystemConstructor::new`, which refuses to
    /// build a system pairing such a solver with a scene that defines at
    /// least one dynamic boundary.
    const SUPPORTS_DYNAMIC_BOUNDARIES: bool = true;

    fn new(params: &Parameters) -> Self;

    /// Compute pressure
    ///
    /// Contract: Non-pressure accelerations (gravity, viscosity) are already
    /// accumulated in `fluid.acceleration` before this is called.
    fn solve_and_add_acceleration<K: KernelFn>(
        &mut self,
        fluid: &mut Fluid,
        boundary: &mut impl BoundaryHandling,
        neighbors: &NeighborList,
        params: &SystemParameters,
        properties: &mut CurrentSystemProperties,
    );

    /// Return solver-specific measurement data
    fn measurement_info(&self) -> SolverMeasurementInfo {
        SolverMeasurementInfo::default()
    }
}

#[derive(Debug, Clone, Default)]
pub struct SolverMeasurementInfo {
    pub stiffness: f64,
    pub target_density_error: f64,
    pub solver_iterations: u32,
    pub relaxation_factor: f64,
    pub predicted_density_error: f64,
}

/// calculate and set predicted velocity due to currently set acceleration
fn set_pred_vel_by_applying_acc(fluid: &mut Fluid, params: &SystemParameters, to_pred_vel: bool) {
    for_each!(
        mut [fluid.velocity_pred],
        ref [
            vel_now = fluid.velocity,
            acceleration = fluid.acceleration,
        ],
        |id, id_velocity_pred| {
            // select velocity
            let base_vel = if to_pred_vel {
                *id_velocity_pred
            } else {
                vel_now[id]
            };
            let vel = base_vel + params.time_increment * acceleration[id];
            *id_velocity_pred = vel;
        }
    );
}

/// Locally calculate pressure acceleration with a state equation at current time
/// and add it to respective samples
///
/// If `custom_target` is `None`, `fluid.acceleration` is used as the target
/// and the result committed to `fluid.acceleration` is mirrored back onto
/// the boundary.
fn add_pressure_acceleration<K: KernelFn>(
    custom_target: Option<&mut Vec<Vector3<f64>>>,
    fluid: &mut Fluid,
    boundary: &mut impl BoundaryHandling,
    neighbors: &NeighborList,
    params: &SystemParameters,
    with_pred_positions: bool,
    overwrite: bool,
) {
    let is_committed_to = custom_target.is_none();
    let target = if let Some(target) = custom_target {
        target
    } else {
        &mut fluid.acceleration
    };
    // compute pressure acceleration
    let forces_onto_boundary: Vec<ForceOntoBoundary> = for_each_collect!(
        mut [target],
        ref [
            pos_now = fluid.position,
            pos_pred = fluid.position_pred,
            mass = fluid.mass,
            volume = fluid.volume,
            pressure = fluid.pressure,
            neighbors = neighbors,
            boundary = boundary,
        ],
        |id, target_acceleration, local_forces| {
            let mut accu = Vector3::zeros();
            let particle_pos = if with_pred_positions {
                pos_pred[id]
            } else {
                pos_now[id]
            };

            for &neighbor in neighbors.get_neighbors(id) {
                let fluid_neighbor_pos = if with_pred_positions {
                    pos_pred[neighbor]
                } else {
                    pos_now[neighbor]
                };
                let r_vec = vector(&fluid_neighbor_pos, &particle_pos);
                accu -= volume[id] / mass[id]
                    * volume[neighbor]
                    * (pressure[id] + pressure[neighbor])
                    * K::kernel_gradient(&r_vec, params.kernel_support_radius);
            }

            for (i, b) in boundary.iter().enumerate() {
                for &boundary_neighbor in b.get_neighbors(id, RequestMode::Normal) {
                    let weighting = params.boundary_pressure_acceleration_weighting;
                    let r_vec = vector(b.position(boundary_neighbor), &particle_pos);
                    let force = 2. * weighting * volume[id]
                        * b.volume(boundary_neighbor)
                        * pressure[id]
                        * K::kernel_gradient(&r_vec, params.kernel_support_radius);

                    if is_committed_to && b.is_dynamic() {
                        local_forces.push(ForceOntoBoundary {
                            id: i,
                            force,
                            force_location: *b.position(boundary_neighbor),
                        });
                    }
                    accu -= force / mass[id];
                }
            }

            if overwrite {
                *target_acceleration = accu;
            } else {
                *target_acceleration += accu;
            }
        }
    );
    if is_committed_to {
        for force in forces_onto_boundary {
            boundary.add_force_onto_boundary(force);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neighbor_search::{NeighborList, NeighborSearch, SpatialHashing};
    use crate::render_info::BoundaryVisualization;
    use crate::sph::boundary_handling::{Boundary, BoundaryCheckpoint, VolumeMapBoundary};
    use crate::sph::fluid::Len;
    use crate::sph::kernel::CubicBSpline3D;
    use crate::sph::setup::input::{DynamicBoundaryDef, StaticBoundaryDef};
    use crate::utilities::triangle_mesh::MeshContainer;
    use nalgebra::Point3;
    use parry3d_f64::math::Vec3;
    use parry3d_f64::shape::TriMesh;

    // ─── Fixtures / helpers ─────────────────────────────────────────────

    /// Builds `SystemParameters` with explicit control over every field
    /// these tests actually touch, correctly setting `time_increment`
    /// regardless of whether the `cfl_time_step` feature is enabled (under
    /// that feature, `SystemParameters::new` always starts `time_increment`
    /// at `0.0`, so it must be overwritten afterwards via direct field
    /// access — legal here since this test module is a descendant of `sph`).
    fn make_params(
        time_increment: f64,
        kernel_support_radius: f64,
        rest_density_grid_spacing: f64,
        boundary_pressure_acceleration_weighting: f64,
    ) -> SystemParameters {
        #[cfg(not(feature = "cfl_time_step"))]
        let mut params = SystemParameters::new(
            time_increment,
            rest_density_grid_spacing,
            kernel_support_radius,
            -1e9,
            0.0,
            0.0,
            boundary_pressure_acceleration_weighting,
        );
        #[cfg(feature = "cfl_time_step")]
        let mut params = SystemParameters::new(
            0.4,
            0.5,
            rest_density_grid_spacing,
            kernel_support_radius,
            -1e9,
            0.0,
            0.0,
            boundary_pressure_acceleration_weighting,
        );
        #[cfg(feature = "cfl_time_step")]
        {
            params.time_increment = time_increment;
        }
        params
    }

    fn cube_trimesh(side: f64) -> TriMesh {
        let h = side / 2.0;
        let positions = vec![
            Vec3::new(h, h, h),
            Vec3::new(h, h, -h),
            Vec3::new(h, -h, h),
            Vec3::new(h, -h, -h),
            Vec3::new(-h, h, h),
            Vec3::new(-h, h, -h),
            Vec3::new(-h, -h, h),
            Vec3::new(-h, -h, -h),
        ];
        let indices: Vec<[u32; 3]> = vec![
            [4, 2, 0],
            [2, 7, 3],
            [6, 5, 7],
            [1, 7, 5],
            [0, 3, 1],
            [4, 1, 5],
            [4, 6, 2],
            [2, 6, 7],
            [6, 4, 5],
            [1, 3, 7],
            [0, 2, 3],
            [4, 0, 1],
        ];
        TriMesh::new(positions, indices).expect("valid cube mesh")
    }

    fn fluid_with_at_least(min_n: usize) -> Fluid {
        let mesh = cube_trimesh(4.0);
        let mut fluid = Fluid::new();
        fluid.add_samples(&mesh, 0, 1000.0, 0.5);
        assert!(
            fluid.len() >= min_n,
            "expected at least {min_n} sampled particles, got {}",
            fluid.len()
        );
        fluid
    }

    fn build_fluid_neighbor_list(positions: &[Point3<f64>], radius: f64) -> NeighborList {
        let mut ns = SpatialHashing::new(radius);
        let mut neighbor_list = NeighborList::new(positions.len());
        ns.find_samples(radius, positions, positions, &mut neighbor_list);
        neighbor_list
    }

    // ─── Mock boundary, scoped to this test module (see `non_pressure_
    // accelerations`'s test suite for the identical pattern and rationale:
    // `VolumeMapBoundary`'s internals are private, and constructing a real
    // dynamic boundary requires the full mesh-discretization pipeline).

    #[derive(Clone)]
    struct MockSample {
        position: Point3<f64>,
        velocity: Vector3<f64>,
        volume: f64,
    }

    #[derive(Clone, Default)]
    struct MockBoundaryEntry {
        samples: Vec<MockSample>,
        neighbors_normal: Vec<Vec<usize>>,
        neighbors_viscosity: Vec<Vec<usize>>,
        center_of_mass: Option<Point3<f64>>,
        accumulated_acceleration: Vector3<f64>,
    }

    impl Boundary for MockBoundaryEntry {
        fn get_neighbors(&self, id: usize, mode: RequestMode) -> &[usize] {
            let list = match mode {
                RequestMode::Normal => &self.neighbors_normal,
                RequestMode::ViscosityAcceleration => &self.neighbors_viscosity,
            };
            list.get(id).map(|v| v.as_slice()).unwrap_or(&[])
        }

        fn position(&self, id: usize) -> &Point3<f64> {
            &self.samples[id].position
        }

        fn velocity(&self, id: usize) -> &Vector3<f64> {
            &self.samples[id].velocity
        }

        fn volume(&self, id: usize) -> f64 {
            self.samples[id].volume
        }

        fn add_acceleration(&mut self, acceleration: Vector3<f64>) {
            self.accumulated_acceleration += acceleration;
        }

        fn center_of_mass(&self) -> Option<Point3<f64>> {
            self.center_of_mass
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct RecordedForce {
        id: usize,
        force: Vector3<f64>,
        force_location: Point3<f64>,
    }

    #[derive(Clone, Default)]
    struct MockBoundary {
        entries: Vec<MockBoundaryEntry>,
        recorded_forces: Vec<RecordedForce>,
    }

    impl BoundaryHandling for MockBoundary {
        fn new() -> Self {
            Self::default()
        }

        fn is_empty(&self) -> bool {
            self.entries.is_empty()
        }

        fn add_static_boundary(
            &mut self,
            _mesh: &mut MeshContainer,
            _boundary: &StaticBoundaryDef,
            _rest_density_grid_spacing: f64,
            _kernel_support_radius: f64,
        ) {
            unimplemented!("not exercised by pressure_solver tests")
        }

        fn add_dynamic_boundary(
            &mut self,
            _mesh: &mut MeshContainer,
            _boundary: &DynamicBoundaryDef,
            _rest_density_grid_spacing: f64,
            _kernel_support_radius: f64,
        ) {
            unimplemented!("not exercised by pressure_solver tests")
        }

        fn initialize<K: KernelFn>(
            &mut self,
            _neighbor_search: &mut impl NeighborSearch,
            _kernel_support_radius: f64,
            _boundary_rest_volume_weighting: f64,
        ) {
        }

        fn find_boundary_samples(
            &mut self,
            _neighbor_search: &mut impl NeighborSearch,
            _within_range: f64,
            _positions: &[Point3<f64>],
            _rest_density_grid_spacing: f64,
        ) {
            unimplemented!("test fixtures set up neighbors directly, not via search")
        }

        fn iter(&self) -> impl Iterator<Item = &dyn Boundary> + '_ {
            self.entries.iter().map(|b| b as &dyn Boundary)
        }

        fn iter_mut(&mut self) -> impl Iterator<Item = &mut dyn Boundary> + '_ {
            self.entries.iter_mut().map(|b| b as &mut dyn Boundary)
        }

        fn add_force_onto_boundary(&mut self, force: ForceOntoBoundary) {
            self.recorded_forces.push(RecordedForce {
                id: force.id,
                force: force.force,
                force_location: force.force_location,
            });
        }

        fn step_forward_in_time(&mut self, _dt: f64) {}

        fn get_fluid_depth(&self, _fluid_volume: f64) -> f64 {
            0.0
        }

        fn get_visualization(&self, _selector: &BoundaryVisualization) -> BoundaryVisualization {
            unimplemented!("not exercised by pressure_solver tests")
        }

        fn get_checkpoint(&self) -> BoundaryCheckpoint {
            BoundaryCheckpoint::default()
        }

        fn restore_from_checkpoint(&mut self, _state: &BoundaryCheckpoint) {}
    }

    // ─── set_pred_vel_by_applying_acc ────────────────────────────────────

    #[test]
    fn set_pred_vel_from_velocity_matches_formula_and_ignores_stale_pred() {
        let dt = 0.1;
        let params = make_params(dt, 1.0, 0.3, 0.0);
        let mut fluid = fluid_with_at_least(1);
        fluid.velocity[0] = Vector3::new(1.0, 2.0, 3.0);
        fluid.acceleration[0] = Vector3::new(0.0, 0.0, -9.81);
        // Deliberately stale garbage: with `to_pred_vel == false`, this must
        // be fully overwritten, never read from.
        fluid.velocity_pred[0] = Vector3::new(999.0, 999.0, 999.0);

        set_pred_vel_by_applying_acc(&mut fluid, &params, false);

        let expected = fluid.velocity[0] + dt * fluid.acceleration[0];
        assert!((fluid.velocity_pred[0] - expected).norm() < 1e-12);
    }

    #[test]
    fn set_pred_vel_from_pred_vel_compounds_across_repeated_calls() {
        // With `to_pred_vel == true`, each call reads the CURRENT
        // `velocity_pred` (not `velocity`) as the base — so calling this
        // twice in a row must compound: v_pred_2 = v_pred_1 + dt*acc, not
        // reset back to `velocity + dt*acc` each time.
        let dt = 0.1;
        let params = make_params(dt, 1.0, 0.3, 0.0);
        let mut fluid = fluid_with_at_least(1);
        fluid.velocity[0] = Vector3::new(5.0, 5.0, 5.0); // must be ignored entirely
        fluid.acceleration[0] = Vector3::new(1.0, 0.0, 0.0);
        fluid.velocity_pred[0] = Vector3::new(0.0, 0.0, 0.0);

        set_pred_vel_by_applying_acc(&mut fluid, &params, true);
        assert!((fluid.velocity_pred[0] - Vector3::new(dt, 0.0, 0.0)).norm() < 1e-12);

        set_pred_vel_by_applying_acc(&mut fluid, &params, true);
        assert!((fluid.velocity_pred[0] - Vector3::new(2.0 * dt, 0.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn set_pred_vel_with_zero_acceleration_leaves_the_base_velocity_unchanged() {
        let params = make_params(0.1, 1.0, 0.3, 0.0);
        let mut fluid = fluid_with_at_least(1);
        let v = Vector3::new(2.0, -1.0, 0.5);
        fluid.velocity[0] = v;
        fluid.acceleration[0] = Vector3::zeros();

        set_pred_vel_by_applying_acc(&mut fluid, &params, false);
        assert!((fluid.velocity_pred[0] - v).norm() < 1e-12);
    }

    #[test]
    fn set_pred_vel_updates_every_particle_independently() {
        let dt = 0.1;
        let params = make_params(dt, 1.0, 0.3, 0.0);
        let mut fluid = fluid_with_at_least(2);
        fluid.velocity[0] = Vector3::new(1.0, 0.0, 0.0);
        fluid.acceleration[0] = Vector3::new(1.0, 0.0, 0.0);
        fluid.velocity[1] = Vector3::new(0.0, 2.0, 0.0);
        fluid.acceleration[1] = Vector3::new(0.0, 1.0, 0.0);

        set_pred_vel_by_applying_acc(&mut fluid, &params, false);

        assert!((fluid.velocity_pred[0] - Vector3::new(1.0 + dt, 0.0, 0.0)).norm() < 1e-12);
        assert!((fluid.velocity_pred[1] - Vector3::new(0.0, 2.0 + dt, 0.0)).norm() < 1e-12);
    }

    // ─── add_pressure_acceleration: fluid-fluid contribution ─────────────

    #[test]
    fn add_pressure_acceleration_matches_manual_formula_and_accumulates_by_default() {
        let h = 1.0;
        let dx = 0.3;
        let params = make_params(0.1, h, dx, 0.0);

        let mut fluid = fluid_with_at_least(3);
        let positions = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.3, 0.0, 0.0),
            Point3::new(0.0, 0.3, 0.0),
        ];
        let pressures = [100.0, 150.0, 80.0];
        let volumes = [0.02, 0.025, 0.03];
        let masses = [0.5, 0.5, 0.5];
        let preexisting_acc = Vector3::new(0.0, 0.0, -9.81);
        for i in 0..3 {
            fluid.position[i] = positions[i];
            fluid.pressure[i] = pressures[i];
            fluid.volume[i] = volumes[i];
            fluid.mass[i] = masses[i];
            fluid.acceleration[i] = preexisting_acc; // must be ADDED to, not overwritten
        }

        let neighbor_list = build_fluid_neighbor_list(&fluid.position, h);
        let mut boundary = VolumeMapBoundary::default();
        add_pressure_acceleration::<CubicBSpline3D>(
            None,
            &mut fluid,
            &mut boundary,
            &neighbor_list,
            &params,
            false, // with_pred_positions
            false, // overwrite
        );

        for id in 0..3 {
            let mut expected = preexisting_acc;
            for &j in neighbor_list.get_neighbors(id) {
                let r_vec = vector(&positions[j], &positions[id]);
                expected -= volumes[id] / masses[id]
                    * volumes[j]
                    * (pressures[id] + pressures[j])
                    * CubicBSpline3D::kernel_gradient(&r_vec, h);
            }
            assert!(
                (fluid.acceleration[id] - expected).norm() < 1e-9,
                "id={id}: expected {expected:?}, got {:?}",
                fluid.acceleration[id]
            );
        }
    }

    #[test]
    fn add_pressure_acceleration_with_overwrite_discards_preexisting_acceleration() {
        let h = 1.0;
        let params = make_params(0.1, h, 0.3, 0.0);
        let mut fluid = fluid_with_at_least(1);
        fluid.position[0] = Point3::origin();
        fluid.pressure[0] = 0.0; // no neighbors -> zero contribution
        fluid.volume[0] = 0.02;
        fluid.mass[0] = 0.5;
        fluid.acceleration[0] = Vector3::new(42.0, 42.0, 42.0); // must be discarded

        let neighbor_list = NeighborList::new(fluid.len());
        let mut boundary = VolumeMapBoundary::default();
        add_pressure_acceleration::<CubicBSpline3D>(
            None,
            &mut fluid,
            &mut boundary,
            &neighbor_list,
            &params,
            false,
            true, // overwrite
        );

        assert_eq!(fluid.acceleration[0], Vector3::zeros());
    }

    #[test]
    fn add_pressure_acceleration_with_pred_positions_uses_position_pred_for_fluid_particles() {
        let h = 1.0;
        let params = make_params(0.1, h, 0.3, 0.0);
        let mut fluid = fluid_with_at_least(2);

        // Real (`position`) values placed far apart -> would NOT be
        // neighbors within `h`; predicted (`position_pred`) values placed
        // close together -> ARE neighbors. This forces any accidental use
        // of `position` instead of `position_pred` to produce a
        // detectably different (near-zero) result.
        fluid.position[0] = Point3::new(0.0, 0.0, 0.0);
        fluid.position[1] = Point3::new(1000.0, 0.0, 0.0);
        fluid.position_pred[0] = Point3::new(0.0, 0.0, 0.0);
        fluid.position_pred[1] = Point3::new(0.3, 0.0, 0.0);
        fluid.pressure[0] = 100.0;
        fluid.pressure[1] = 100.0;
        fluid.volume[0] = 0.02;
        fluid.volume[1] = 0.02;
        fluid.mass[0] = 0.5;
        fluid.mass[1] = 0.5;
        fluid.acceleration[0] = Vector3::zeros();
        fluid.acceleration[1] = Vector3::zeros();

        // Neighbor search must itself be built on the predicted positions
        // to find this pair as neighbors — mirroring how a real caller
        // would need to re-run the neighbor search on predicted positions
        // before calling with `with_pred_positions = true`.
        let neighbor_list = build_fluid_neighbor_list(&fluid.position_pred, h);
        assert!(!neighbor_list.get_neighbors(0).is_empty());

        let mut boundary = VolumeMapBoundary::default();
        add_pressure_acceleration::<CubicBSpline3D>(
            None,
            &mut fluid,
            &mut boundary,
            &neighbor_list,
            &params,
            true, // with_pred_positions
            true,
        );

        assert!(
            fluid.acceleration[0].norm() > 1e-6,
            "expected a nonzero contribution computed from position_pred, got {:?}",
            fluid.acceleration[0]
        );
    }

    // ─── add_pressure_acceleration: boundary contribution ────────────────

    #[test]
    fn add_pressure_acceleration_boundary_contribution_matches_manual_formula() {
        let h = 1.0;
        let dx = 0.3;
        let weighting = 0.5;
        let params = make_params(0.1, h, dx, weighting);

        let mut fluid = fluid_with_at_least(1);
        fluid.position[0] = Point3::origin();
        fluid.pressure[0] = 200.0;
        fluid.volume[0] = 0.02;
        fluid.mass[0] = 0.5;
        fluid.acceleration[0] = Vector3::zeros();

        let neighbor_list = NeighborList::new(fluid.len());
        let boundary_pos = Point3::new(0.2, 0.0, 0.0);
        let boundary_vol = 0.01;
        let mut boundary = MockBoundary::default();
        boundary.entries.push(MockBoundaryEntry {
            samples: vec![MockSample {
                position: boundary_pos,
                velocity: Vector3::zeros(),
                volume: boundary_vol,
            }],
            neighbors_normal: vec![vec![0]],
            center_of_mass: None, // static
            ..Default::default()
        });

        add_pressure_acceleration::<CubicBSpline3D>(
            None,
            &mut fluid,
            &mut boundary,
            &neighbor_list,
            &params,
            false,
            true,
        );

        let r_vec = vector(&boundary_pos, &fluid.position[0]);
        let force = 2.0
            * weighting
            * fluid.volume[0]
            * boundary_vol
            * fluid.pressure[0]
            * CubicBSpline3D::kernel_gradient(&r_vec, h);
        let expected = -force / fluid.mass[0];

        assert!((fluid.acceleration[0] - expected).norm() < 1e-9);
        assert!(
            boundary.recorded_forces.is_empty(),
            "a static boundary must not receive a reaction force"
        );
    }

    #[test]
    fn add_pressure_acceleration_registers_newtons_third_law_reaction_force_on_dynamic_boundaries()
    {
        let h = 1.0;
        let weighting = 1.0;
        let params = make_params(0.1, h, 0.3, weighting);

        let mut fluid = fluid_with_at_least(1);
        fluid.position[0] = Point3::origin();
        fluid.pressure[0] = 150.0;
        fluid.volume[0] = 0.02;
        fluid.mass[0] = 0.5;
        fluid.acceleration[0] = Vector3::zeros();

        let neighbor_list = NeighborList::new(fluid.len());
        let boundary_pos = Point3::new(0.2, 0.0, 0.0);
        let mut boundary = MockBoundary::default();
        boundary.entries.push(MockBoundaryEntry {
            samples: vec![MockSample {
                position: boundary_pos,
                velocity: Vector3::zeros(),
                volume: 0.01,
            }],
            neighbors_normal: vec![vec![0]],
            center_of_mass: Some(Point3::new(5.0, 0.0, 0.0)), // dynamic
            ..Default::default()
        });

        add_pressure_acceleration::<CubicBSpline3D>(
            None,
            &mut fluid,
            &mut boundary,
            &neighbor_list,
            &params,
            false,
            true,
        );

        // The force felt BY the fluid from this boundary pair is
        // `mass[0] * acceleration[0]` (only contribution present here); by
        // Newton's third law, the registered reaction force onto the
        // boundary must be exactly its negation.
        let force_on_fluid = fluid.mass[0] * fluid.acceleration[0];

        assert_eq!(boundary.recorded_forces.len(), 1);
        let recorded = boundary.recorded_forces[0];
        assert_eq!(recorded.id, 0);
        assert!((recorded.force - (-force_on_fluid)).norm() < 1e-9);
        assert_eq!(recorded.force_location, boundary_pos);
    }

    #[test]
    fn add_pressure_acceleration_with_a_custom_target_never_registers_boundary_forces() {
        // This is the key contract enabling IISPH-style CG iterations to
        // probe an intermediate pressure field (via `custom_target`)
        // without leaking premature reaction forces onto dynamic
        // boundaries — only the final, committed call (`custom_target ==
        // None`) may do that.
        let h = 1.0;
        let params = make_params(0.1, h, 0.3, 1.0);

        let mut fluid = fluid_with_at_least(1);
        fluid.position[0] = Point3::origin();
        fluid.pressure[0] = 150.0;
        fluid.volume[0] = 0.02;
        fluid.mass[0] = 0.5;
        fluid.acceleration[0] = Vector3::new(7.0, 7.0, 7.0); // must stay untouched

        let neighbor_list = NeighborList::new(fluid.len());
        let mut boundary = MockBoundary::default();
        boundary.entries.push(MockBoundaryEntry {
            samples: vec![MockSample {
                position: Point3::new(0.2, 0.0, 0.0),
                velocity: Vector3::zeros(),
                volume: 0.01,
            }],
            neighbors_normal: vec![vec![0]],
            center_of_mass: Some(Point3::new(5.0, 0.0, 0.0)), // dynamic
            ..Default::default()
        });

        let mut custom_target = vec![Vector3::zeros(); fluid.len()];
        add_pressure_acceleration::<CubicBSpline3D>(
            Some(&mut custom_target),
            &mut fluid,
            &mut boundary,
            &neighbor_list,
            &params,
            false,
            true,
        );

        assert!(
            custom_target[0].norm() > 1e-6,
            "expected a nonzero contribution actually written into custom_target"
        );
        assert_eq!(
            fluid.acceleration[0],
            Vector3::new(7.0, 7.0, 7.0),
            "fluid.acceleration must be untouched when writing into a custom_target"
        );
        assert!(
            boundary.recorded_forces.is_empty(),
            "no reaction force must be registered while probing via a custom_target"
        );
    }

    #[test]
    fn add_pressure_acceleration_boundary_term_always_uses_current_position_never_pred() {
        // Unlike the fluid-fluid term, the LOCAL particle's position does
        // switch between `position`/`position_pred` based on
        // `with_pred_positions` — but the boundary neighbor's own position
        // (`b.position(...)`) has no predicted counterpart at all and is
        // always used as-is. This pins down that asymmetry explicitly.
        let h = 1.0;
        let params = make_params(0.1, h, 0.3, 1.0);

        let mut fluid = fluid_with_at_least(1);
        fluid.position[0] = Point3::new(0.2, 0.0, 0.0);
        fluid.position_pred[0] = Point3::new(0.2, 0.0, 0.0); // identical on purpose
        fluid.pressure[0] = 150.0;
        fluid.volume[0] = 0.02;
        fluid.mass[0] = 0.5;
        fluid.acceleration[0] = Vector3::zeros();

        let neighbor_list = NeighborList::new(fluid.len());
        let boundary_pos = Point3::new(0.0, 0.0, 0.0);
        let mut boundary = MockBoundary::default();
        boundary.entries.push(MockBoundaryEntry {
            samples: vec![MockSample {
                position: boundary_pos,
                velocity: Vector3::zeros(),
                volume: 0.01,
            }],
            neighbors_normal: vec![vec![0]],
            center_of_mass: None,
            ..Default::default()
        });

        add_pressure_acceleration::<CubicBSpline3D>(
            None,
            &mut fluid,
            &mut boundary,
            &neighbor_list,
            &params,
            true, // with_pred_positions — irrelevant here since both are equal
            true,
        );

        let r_vec = vector(&boundary_pos, &fluid.position[0]);
        let force = 2.0
            * params.boundary_pressure_acceleration_weighting
            * fluid.volume[0]
            * 0.01
            * fluid.pressure[0]
            * CubicBSpline3D::kernel_gradient(&r_vec, h);
        let expected = -force / fluid.mass[0];

        assert!((fluid.acceleration[0] - expected).norm() < 1e-9);
    }
}
