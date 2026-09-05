//! Implicit imcompressible SPH (SESPH) pressure solver with "optimized source term"
//!
//! Unlike every other `PressureSolver` in this crate, `solve_and_add_acceleration`
//! runs TWO separate global pressure solves per call (EQS1, EQS2), each time
//! pressure is applied via `add_pressure_acceleration` with `custom_target == None`.
//!
//! The actual position/velocity update happens within this solver by computing into
//! solver_position_slots[0]/solver_velocity_slots[0] — the field for the predicted position
//! and velocity. No real IntegrationScheme involvement needed; pair with
//! IntegrationSchemeVariant::TakePredicted.
//!
//! # Limitation: no support for dynamic boundaries
//! To function correctly with this solver, dynamic boundaries will need to be able
//! to update positions and velocities separately. Additionally, support for predicted
//! positions and velocities will need to be added.
//!
//! To prevent silent failures, the combination of this solver running with dynamic
//! boundaries is guarded from running.
use crate::for_each;
use crate::neighbor_search::NeighborList;
use crate::sph::CurrentSystemProperties;
use crate::sph::boundary_handling::{BoundaryHandling, RequestMode};
use crate::sph::fluid::{Fluid, Len};
use crate::sph::kernel::KernelFn;
use crate::sph::pressure_solver::iisph::{IISPH, TerminationCondition};
use crate::sph::pressure_solver::{PressureSolver, SolverMeasurementInfo};
use crate::sph::pressure_solver::{add_pressure_acceleration, set_pred_vel_by_applying_acc};
use crate::sph::setup::input::Parameters;
use crate::sph::{Outer, SystemParameters};
use crate::utilities::vector;

use nalgebra::Matrix3;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Clone)]
pub struct IISPHwOST {
    inner: IISPH,
}

impl PressureSolver for IISPHwOST {
    const POSITION_SLOTS: usize = 1;
    const VELOCITY_SLOTS: usize = 1;
    const MANAGES_OWN_INTEGRATION: bool = true;
    const SUPPORTS_DYNAMIC_BOUNDARIES: bool = false;

    fn new(params: &Parameters) -> Self {
        Self {
            inner: IISPH::new(params),
        }
    }

    fn solve_and_add_acceleration<K: KernelFn>(
        &mut self,
        fluid: &mut Fluid,
        boundary: &mut impl BoundaryHandling,
        neighbor_list: &NeighborList,
        params: &SystemParameters,
        _properties: &mut CurrentSystemProperties,
    ) {
        self.inner.resize_scratch(fluid.len());

        // solve EQS1
        {
            // set predicted velocity by applying non-pressure acceleration
            set_pred_vel_by_applying_acc(fluid, params, false);
            // set source term
            self.inner
                .set_source_term_vde::<K>(fluid, boundary, neighbor_list, params);
            // self.set_source_term_vp(false);
            // solve pressure equation system
            self.inner.resolve_pressure_globally::<K>(
                fluid,
                boundary,
                neighbor_list,
                params,
                false,
                TerminationCondition::AfterIteration(3),
                // TerminationCondition::TargetDensityError(params.target_density_error),
                true,
            );
            // set acceleration to pressure acceleration with pressure from EQS1
            add_pressure_acceleration::<K>(
                None,
                fluid,
                boundary,
                neighbor_list,
                params,
                false,
                true,
            );
        }
        // println!("pressure acc eq1: {}", self.particles[200].acc());
        // set predicted velocity and positions
        {
            // set predicted velocity
            set_pred_vel_by_applying_acc(fluid, params, true);
            // set predicted position
            for_each!(
                mut [fluid.solver_position_slots[0]],
                ref [
                    pos_now = fluid.position,
                    vel_pred = fluid.solver_velocity_slots[0],
                ],
                |id, id_position_pred| {
                    *id_position_pred = pos_now[id] + params.time_increment * vel_pred[id];
                }
            );
        }
        // solve EQS2
        {
            // set source term
            self.inner
                .set_source_term_vp::<K>(fluid, boundary, neighbor_list, params, false);
            // solve pressure equation system
            self.inner.resolve_pressure_globally::<K>(
                fluid,
                boundary,
                neighbor_list,
                params,
                false,
                TerminationCondition::TargetDensityError(self.inner.target_density_error),
                true,
            );
            // set acceleration to pressure acceleration with pressure from EQS2
            add_pressure_acceleration::<K>(
                None,
                fluid,
                boundary,
                neighbor_list,
                params,
                false,
                true,
            );
        }
        // write new positions and resampled velocities to predicted velocity and position field of each particle
        {
            for_each!(
                mut [fluid.solver_position_slots[0], self.inner.pressure_acc_f],
                ref [
                    pos_now = fluid.position,
                    vel_pred = fluid.solver_velocity_slots[0],
                    acceleration = fluid.acceleration,
                    volume = fluid.volume,
                    neighbors = neighbor_list,
                    boundary = boundary,
                ],
                |id, id_position_pred, id_pressure_acc_f| {
                    // calculate new position and store it intermediately
                    let new_pos = *id_position_pred
                        + params.time_increment.powi(2) * acceleration[id]; // TODO uncomment
                    // calculate and set velocity gradient (Jacobian) as predicted velocity
                    let mut jac_vel = Matrix3::zeros();
                    for &neighbor in neighbors.get_neighbors(id) {
                        let r_vec = vector(
                            &pos_now[neighbor],
                            &pos_now[id],
                        );
                        jac_vel -= volume[neighbor]
                            * (vel_pred[id] - vel_pred[neighbor]).outer(
                                &K::kernel_gradient(
                                    &r_vec,
                                    params.kernel_support_radius,
                                ),
                            );
                    }
                    for b in boundary.iter() {
                        for &boundary_neighbor in b.get_neighbors(id, RequestMode::Normal) {
                            let r_vec = vector(
                                b.position(boundary_neighbor),
                                &pos_now[id],
                            );
                            jac_vel -= b.volume(boundary_neighbor)
                                * (vel_pred[id]
                                    - b.velocity(boundary_neighbor))
                                .outer(&K::kernel_gradient(
                                    &r_vec,
                                    params.kernel_support_radius,
                                ));
                        }
                    }
                    // calculate new velocity and intermediately store it as pressure_acc_f to avoid race condition on .vel().pred()
                    // particle.pressure_acc_f[id] = vel_pred[id] + jac_vel*(new_pos - pos_pred[id]); // original "optimized source term" approach
                    // particle.pressure_acc_f[id] = vel_pred[id] + jac_vel*(new_pos - pos_pred[id]) + params.time_increment*particle.acc(); // TODO test
                    // particle.pressure_acc_f[id] = vel_pred[id]; // TODO test
                    *id_pressure_acc_f =
                        vel_pred[id] + params.time_increment * acceleration[id]; // DFSPH approach
                    // store new position in predicted position
                    *id_position_pred = new_pos;
                }
            );
            // move velocity from pressure_acc_f to predicted velocity
            for_each!(
                mut [fluid.solver_velocity_slots[0]],
                ref [
                    pressure_acc_f = self.inner.pressure_acc_f,
                ],
                |id, id_vel_pred| {
                    *id_vel_pred = pressure_acc_f[id];
                }
            );
        }
        // pressure acceleration is applied to particle movement implicitly
        // resulting in the velocity and position predictions
        // Those predicted positions and velocities are commited to in the TakePredicted
        // integration scheme, which is the mandatory pairing with this solver type.
        std::mem::swap(
            &mut fluid.integrator_position_slots[0],
            &mut fluid.solver_position_slots[0],
        );
        std::mem::swap(
            &mut fluid.integrator_velocity_slots[0],
            &mut fluid.solver_velocity_slots[0],
        );
    }

    fn measurement_info(&self) -> SolverMeasurementInfo {
        SolverMeasurementInfo {
            target_density_error: self.inner.target_density_error,
            solver_iterations: self.inner.last_solver_iterations,
            relaxation_factor: self.inner.relaxation_factor,
            predicted_density_error: self.inner.predicted_density_error,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neighbor_search::{NeighborList, NeighborSearch, SpatialHashing};
    use crate::sph::boundary_handling::{
        Boundary, BoundaryCheckpoint, ForceOntoBoundary, VolumeMapBoundary,
    };
    use crate::sph::kernel::CubicBSpline3D;
    use nalgebra::{Point3, Vector3};
    use parry3d_f64::math::Vec3;
    use parry3d_f64::shape::TriMesh;

    // ─── Fixtures / helpers ─────────────────────────────────────────────

    fn make_solver_params(
        target_density_error: f64,
        relaxation_factor: f64,
        min_diagonal_element: f64,
    ) -> Parameters {
        Parameters {
            buffer_length_limit: 100,
            #[cfg(not(feature = "cfl_time_step"))]
            time_increment: 0.001,
            #[cfg(feature = "cfl_time_step")]
            max_time_increment: 0.001,
            #[cfg(feature = "cfl_time_step")]
            cfl_number: 0.4,
            fluid: vec![],
            rest_density_grid_spacing: 0.3,
            kernel_support_radius: 1.0,
            disable_particles_below: -1e9,
            fluid_viscosity: 0.0,
            boundary_viscosity: 0.0,
            boundary_pressure_acceleration_weighting: 0.0,
            boundary_rest_volume_weighting: 0.0,
            stiffness: 0.0,
            target_density_error,
            relaxation_factor,
            min_diagonal_element,
        }
    }

    /// Directly overwrites the private `time_increment` field — legal here
    /// since this test module is a descendant of `sph`.
    fn make_system_params(
        dt: f64,
        kernel_support_radius: f64,
        rest_density_grid_spacing: f64,
        boundary_pressure_acceleration_weighting: f64,
    ) -> SystemParameters {
        #[cfg(not(feature = "cfl_time_step"))]
        let mut params = SystemParameters::new(
            dt,
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
            0.4,
            rest_density_grid_spacing,
            kernel_support_radius,
            -1e9,
            0.0,
            0.0,
            boundary_pressure_acceleration_weighting,
        );
        params.time_increment = dt;
        params
    }

    fn rest_volume_for(rest_density_grid_spacing: f64) -> f64 {
        rest_density_grid_spacing.powi(3)
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
        assert!(fluid.len() >= min_n);
        fluid
    }

    fn with_slots(mut fluid: Fluid) -> Fluid {
        fluid.resize_slots(1, 1, 1, 1); // integrator: 1/1, solver: 1/1
        fluid
    }

    // ─── Mock boundary (scoped to this test module) ─────────────────────

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
            _mesh: &mut crate::utilities::triangle_mesh::MeshContainer,
            _boundary: &crate::sph::setup::input::StaticBoundaryDef,
            _r: f64,
            _k: f64,
        ) {
            unimplemented!("not exercised by IISPHwOST tests")
        }
        fn add_dynamic_boundary(
            &mut self,
            _mesh: &mut crate::utilities::triangle_mesh::MeshContainer,
            _boundary: &crate::sph::setup::input::DynamicBoundaryDef,
            _r: f64,
            _k: f64,
        ) {
            unimplemented!("not exercised by IISPHwOST tests")
        }
        fn initialize<K: KernelFn>(&mut self, _n: &mut impl NeighborSearch, _k: f64, _w: f64) {}
        fn find_boundary_samples(
            &mut self,
            _n: &mut impl NeighborSearch,
            _r: f64,
            _p: &[Point3<f64>],
            _s: f64,
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
        fn get_fluid_depth(&self, _v: f64) -> f64 {
            0.0
        }
        fn get_visualization(
            &self,
            _s: &crate::render_info::BoundaryVisualization,
        ) -> crate::render_info::BoundaryVisualization {
            unimplemented!("not exercised by IISPHwOST tests")
        }
        fn get_checkpoint(&self) -> BoundaryCheckpoint {
            BoundaryCheckpoint::default()
        }
        fn restore_from_checkpoint(&mut self, _s: &BoundaryCheckpoint) {}
    }

    // ─── new / measurement_info ─────────────────────────────────────────

    #[test]
    fn new_delegates_to_inner_iisph_with_matching_fields() {
        let solver = IISPHwOST::new(&make_solver_params(0.02, 0.6, 1e-7));
        assert_eq!(solver.inner.target_density_error, 0.02);
        assert_eq!(solver.inner.relaxation_factor, 0.6);
        assert_eq!(solver.inner.min_diagonal_element, 1e-7);
    }

    #[test]
    fn measurement_info_surfaces_inner_fields() {
        let mut solver = IISPHwOST::new(&make_solver_params(0.02, 0.6, 1e-7));
        solver.inner.last_solver_iterations = 4;
        solver.inner.predicted_density_error = 0.03;

        let info = solver.measurement_info();
        assert_eq!(info.target_density_error, 0.02);
        assert_eq!(info.relaxation_factor, 0.6);
        assert_eq!(info.solver_iterations, 4);
        assert_eq!(info.predicted_density_error, 0.03);
        assert_eq!(info.stiffness, 0.0); // not used by this solver -> default
    }

    // ─── Fully deterministic end-to-end trace: an isolated particle ─────
    //
    // With zero fluid and zero boundary neighbors throughout, every pressure
    // term in both EQS1 and EQS2 vanishes exactly, making the full two-stage
    // pipeline hand-computable:
    //   - pressure ends at 0.0 (isolated -> a_ff == 0 -> `initialize` yields 0)
    //   - fluid.acceleration ends at 0.0 — NOT the preexisting acceleration
    //     `g` set before the call, because BOTH `add_pressure_acceleration`
    //     calls inside this solver use `overwrite == true` with
    //     `custom_target == None`, discarding whatever was in
    //     `fluid.acceleration` beforehand. This deviates from the
    //     `PressureSolver::solve_and_add_acceleration` trait's documented
    //     contract ("non-pressure accelerations already accumulated ...")
    //     — apparently intentional here, since the actual position/velocity
    //     update happens via `position_pred`/`velocity_pred`, meant to be
    //     paired with `IntegrationSchemeVariant::TakePredicted` (see
    //     `Procedures::integration_scheme`'s doc comment).
    //   - velocity_pred ends at `velocity + dt * g` (plain explicit-Euler,
    //     since no pressure correction ever applies to an isolated particle)
    //   - position_pred ends at `position + dt * (velocity + dt * g)`

    #[test]
    fn solve_and_add_acceleration_on_an_isolated_particle_matches_the_exact_hand_derivation() {
        let h = 1.0;
        let dt = 0.05;
        let params = make_system_params(dt, h, 0.3, 0.0);
        let mut solver = IISPHwOST::new(&make_solver_params(0.01, 0.5, 1e-9));

        let mut fluid = with_slots(fluid_with_at_least(1));
        for v in fluid.volume.iter_mut() {
            *v = params.rest_volume;
        }
        let pos0 = Point3::new(1.0, 2.0, 3.0);
        let vel0 = Vector3::new(0.5, 0.0, 0.0);
        let g = Vector3::new(0.0, 0.0, -9.81);
        fluid.position[0] = pos0;
        fluid.velocity[0] = vel0;
        fluid.mass[0] = 0.5;
        fluid.volume[0] = rest_volume_for(0.3);
        fluid.acceleration[0] = g; // preexisting acceleration, expected to be discarded

        let neighbor_list = NeighborList::new(fluid.len());
        let mut boundary = VolumeMapBoundary::default();
        let mut properties = CurrentSystemProperties::default();

        solver.solve_and_add_acceleration::<CubicBSpline3D>(
            &mut fluid,
            &mut boundary,
            &neighbor_list,
            &params,
            &mut properties,
        );

        assert_eq!(fluid.pressure[0], 0.0);
        assert_eq!(
            fluid.acceleration[0],
            Vector3::zeros(),
            "acceleration must be overwritten to zero, not retain the preexisting gravity g"
        );

        let expected_vel_pred = vel0 + dt * g;
        assert!(
            (fluid.integrator_velocity_slots[0][0] - expected_vel_pred).norm() < 1e-9,
            "expected {expected_vel_pred:?}, got {:?}",
            fluid.integrator_velocity_slots[0][0]
        );

        let expected_pos_pred = pos0 + dt * expected_vel_pred;
        assert!(
            (fluid.integrator_position_slots[0][0] - expected_pos_pred).norm() < 1e-9,
            "expected {expected_pos_pred:?}, got {:?}",
            fluid.integrator_position_slots[0][0]
        );
    }

    // ─── Dynamic boundary: force registered twice per call ─────────────

    #[test]
    fn solve_and_add_acceleration_registers_a_reaction_force_once_per_stage_on_dynamic_boundaries()
    {
        // Unlike the other pressure solvers (which call `add_pressure_
        // acceleration` with `custom_target == None` exactly once), this
        // solver commits pressure TWICE per call — once for EQS1, once for
        // EQS2 — so a dynamic boundary neighbor present throughout receives
        // TWO separate reaction-force registrations, not one.
        let h = 1.0;
        let weighting = 1.0;
        let params = make_system_params(0.05, h, 0.3, weighting);
        let mut solver = IISPHwOST::new(&make_solver_params(0.01, 0.5, 1e-9));

        let mut fluid = with_slots(fluid_with_at_least(1));
        for v in fluid.volume.iter_mut() {
            *v = params.rest_volume;
        }
        fluid.position[0] = Point3::origin();
        fluid.velocity[0] = Vector3::zeros();
        fluid.mass[0] = 0.5;
        fluid.volume[0] = rest_volume_for(0.3) * 0.5; // compressed -> nonzero pressure likely
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
        let mut properties = CurrentSystemProperties::default();

        solver.solve_and_add_acceleration::<CubicBSpline3D>(
            &mut fluid,
            &mut boundary,
            &neighbor_list,
            &params,
            &mut properties,
        );

        assert_eq!(
            boundary.recorded_forces.len(),
            2,
            "expected one force registration for EQS1's pressure and one for EQS2's"
        );
    }

    #[test]
    fn solve_and_add_acceleration_registers_no_reaction_force_for_static_boundaries() {
        let h = 1.0;
        let weighting = 1.0;
        let params = make_system_params(0.05, h, 0.3, weighting);
        let mut solver = IISPHwOST::new(&make_solver_params(0.01, 0.5, 1e-9));

        let mut fluid = with_slots(fluid_with_at_least(1));
        for v in fluid.volume.iter_mut() {
            *v = params.rest_volume;
        }
        fluid.position[0] = Point3::origin();
        fluid.velocity[0] = Vector3::zeros();
        fluid.mass[0] = 0.5;
        fluid.volume[0] = rest_volume_for(0.3) * 0.5;
        fluid.acceleration[0] = Vector3::zeros();

        let neighbor_list = NeighborList::new(fluid.len());
        let mut boundary = MockBoundary::default();
        boundary.entries.push(MockBoundaryEntry {
            samples: vec![MockSample {
                position: Point3::new(0.2, 0.0, 0.0),
                velocity: Vector3::zeros(),
                volume: 0.01,
            }],
            neighbors_normal: vec![vec![0]],
            center_of_mass: None, // static
            ..Default::default()
        });
        let mut properties = CurrentSystemProperties::default();

        solver.solve_and_add_acceleration::<CubicBSpline3D>(
            &mut fluid,
            &mut boundary,
            &neighbor_list,
            &params,
            &mut properties,
        );

        assert!(boundary.recorded_forces.is_empty());
    }
}
