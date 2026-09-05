//! State equation SPH (SESPH) or weakly compressible SPH (WCSPH) pressure solver
use crate::for_each;
use crate::neighbor_search::NeighborList;
use crate::sph::CurrentSystemProperties;
use crate::sph::SystemParameters;
use crate::sph::boundary_handling::{BoundaryHandling, RequestMode};
use crate::sph::fluid::{Fluid, Len};
use crate::sph::kernel::KernelFn;
use crate::sph::pressure_solver::{PressureSolver, SolverMeasurementInfo};
use crate::sph::pressure_solver::{add_pressure_acceleration, set_pred_vel_by_applying_acc};
use crate::sph::setup::input::Parameters;
use crate::utilities::vector;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Clone)]
pub struct SESPHwSplitting {
    stiffness: f64,
    density_pred: Vec<f64>,
}

impl PressureSolver for SESPHwSplitting {
    const VELOCITY_SLOTS: usize = 1;

    fn new(params: &Parameters) -> Self {
        Self {
            stiffness: params.stiffness,
            density_pred: Vec::new(),
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
        self.resize_scratch(fluid.len());
        // perform splitting step conditionally
        set_pred_vel_by_applying_acc(fluid, params, false);
        self.calc_predicted_density::<K>(fluid, boundary, neighbor_list, params);
        // compute pressure
        {
            for_each!(
                mut [fluid.pressure],
                ref [
                    density_pred = self.density_pred,
                    mass = fluid.mass,
                ],
                |id, id_pressure| {
                    // select density
                    let id_volume = mass[id] / density_pred[id];
                    // calc pressure with state equation
                    *id_pressure = self.stiffness
                        * f64::max(params.rest_volume / id_volume - 1., 0.);
                    // #[cfg(feature = "logging")]
                    // tracing::debug!("pressure: {}", pressure);
                }
            );
        }
        // add pressure acceleration (compute from pressure)
        add_pressure_acceleration::<K>(None, fluid, boundary, neighbor_list, params, false, false);
    }

    fn measurement_info(&self) -> SolverMeasurementInfo {
        SolverMeasurementInfo {
            stiffness: self.stiffness,
            ..Default::default()
        }
    }
}

impl SESPHwSplitting {
    pub fn resize_scratch(&mut self, len: usize) {
        self.density_pred.resize(len, 0.0);
    }

    fn calc_predicted_density<K: KernelFn>(
        &mut self,
        fluid: &mut Fluid,
        boundary: &impl BoundaryHandling,
        neighbor_list: &NeighborList,
        params: &SystemParameters,
    ) {
        for_each!(
            mut [self.density_pred],
            ref [
                pos_now = fluid.position,
                vel_pred = fluid.solver_velocity_slots[0],
                mass = fluid.mass,
                neighbors = neighbor_list,
                boundary = boundary,
            ],
            |id, density_pred| {
                // reset density
                let mut accu = 0.;
                // add density for every neighbor
                for &neighbor in neighbors.get_neighbors(id) {
                    let r_vec = vector(
                        &pos_now[neighbor],
                        &pos_now[id],
                    );
                    accu += mass[neighbor]
                        * K::kernel_function(
                            &r_vec,
                            params.kernel_support_radius,
                        )
                        + params.time_increment
                            * (vel_pred[id] - vel_pred[neighbor]).dot(&K::kernel_gradient(
                                &r_vec,
                                params.kernel_support_radius,
                            ));
                }
                // add density for every boundary neighbor (mirror mass of moving sample onto boundary sample)
                for b in boundary.iter() {
                    for &boundary_neighbor in b.get_neighbors(id, RequestMode::Normal) {
                        let r_vec = vector(
                            b.position(boundary_neighbor),
                            &pos_now[id],
                        );
                        accu += b.volume(boundary_neighbor)
                            * mass[id]/params.rest_volume
                            * K::kernel_function(
                                &r_vec,
                                params.kernel_support_radius,
                            )
                            + params.time_increment
                                * vel_pred[id]
                                    .dot(&K::kernel_gradient(
                                        &r_vec,
                                        params.kernel_support_radius,
                                    ));
                    }
                }
                *density_pred = accu;
                // #[cfg(feature = "logging")]
                // tracing::debug!("density: {}", fluid.density());
            }
        );
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

    fn make_solver(stiffness: f64) -> SESPHwSplitting {
        SESPHwSplitting {
            stiffness,
            density_pred: Vec::new(),
        }
    }

    /// Builds `SystemParameters` and then directly overwrites the private
    /// `time_increment` field — legal here since this test module is a
    /// descendant of `sph`. This bypasses the `cfl_time_step` feature's
    /// otherwise-inaccessible-from-outside `time_increment` initialization.
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

    fn build_fluid_neighbor_list(positions: &[Point3<f64>], radius: f64) -> NeighborList {
        let mut ns = SpatialHashing::new(radius);
        let mut nl = NeighborList::new(positions.len());
        ns.find_samples(radius, positions, positions, &mut nl);
        nl
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
        _id: usize,
        force: Vector3<f64>,
        _force_location: Point3<f64>,
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
            unimplemented!("not exercised by SESPHwSplitting tests")
        }
        fn add_dynamic_boundary(
            &mut self,
            _mesh: &mut crate::utilities::triangle_mesh::MeshContainer,
            _boundary: &crate::sph::setup::input::DynamicBoundaryDef,
            _r: f64,
            _k: f64,
        ) {
            unimplemented!("not exercised by SESPHwSplitting tests")
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
                _id: force.id,
                force: force.force,
                _force_location: force.force_location,
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
            unimplemented!("not exercised by SESPHwSplitting tests")
        }
        fn get_checkpoint(&self) -> BoundaryCheckpoint {
            BoundaryCheckpoint::default()
        }
        fn restore_from_checkpoint(&mut self, _s: &BoundaryCheckpoint) {}
    }

    // ─── new / measurement_info ─────────────────────────────────────────

    #[test]
    fn new_captures_the_configured_stiffness_and_starts_with_empty_scratch() {
        let solver = make_solver(500.0);
        assert_eq!(solver.stiffness, 500.0);
        assert!(solver.density_pred.is_empty());
    }

    #[test]
    fn measurement_info_reports_only_stiffness() {
        let solver = make_solver(250.0);
        let info = solver.measurement_info();
        assert_eq!(info.stiffness, 250.0);
        assert_eq!(info.target_density_error, 0.0);
        assert_eq!(info.solver_iterations, 0);
        assert_eq!(info.relaxation_factor, 0.0);
        assert_eq!(info.predicted_density_error, 0.0);
    }

    // ─── resize_scratch ─────────────────────────────────────────────────

    #[test]
    fn resize_scratch_grows_and_shrinks_density_pred() {
        let mut solver = make_solver(100.0);
        solver.resize_scratch(5);
        assert_eq!(solver.density_pred.len(), 5);
        assert!(solver.density_pred.iter().all(|&v| v == 0.0));
        solver.resize_scratch(2);
        assert_eq!(solver.density_pred.len(), 2);
    }

    // ─── calc_predicted_density ─────────────────────────────────────────

    #[test]
    fn calc_predicted_density_of_an_isolated_particle_is_zero() {
        let h = 1.0;
        let params = make_system_params(0.05, h, 0.3, 0.0);
        let mut solver = make_solver(100.0);

        let mut fluid = fluid_with_at_least(1);
        fluid.position[0] = Point3::origin();
        fluid.solver_velocity_slots[0][0] = Vector3::new(1.0, 0.0, 0.0);
        fluid.mass[0] = 0.5;
        solver.resize_scratch(fluid.len());

        let neighbor_list = NeighborList::new(fluid.len());
        let boundary = VolumeMapBoundary::default();

        solver.calc_predicted_density::<CubicBSpline3D>(
            &mut fluid,
            &boundary,
            &neighbor_list,
            &params,
        );

        assert_eq!(solver.density_pred[0], 0.0);
    }

    #[test]
    fn calc_predicted_density_matches_manual_formula_for_fluid_neighbors() {
        let h = 1.0;
        let dt = 0.05;
        let params = make_system_params(dt, h, 0.3, 0.0);
        let mut solver = make_solver(100.0);

        let mut fluid = fluid_with_at_least(2);
        fluid.position[0] = Point3::new(0.0, 0.0, 0.0);
        fluid.position[1] = Point3::new(0.3, 0.0, 0.0);
        fluid.solver_velocity_slots[0][0] = Vector3::new(1.0, 0.0, 0.0);
        fluid.solver_velocity_slots[0][1] = Vector3::new(-1.0, 0.0, 0.0);
        fluid.mass[0] = 0.5;
        fluid.mass[1] = 0.6;
        solver.resize_scratch(fluid.len());

        let neighbor_list = build_fluid_neighbor_list(&fluid.position, h);
        let boundary = VolumeMapBoundary::default();

        solver.calc_predicted_density::<CubicBSpline3D>(
            &mut fluid,
            &boundary,
            &neighbor_list,
            &params,
        );

        let mut expected = 0.0;
        for &j in neighbor_list.get_neighbors(0) {
            let r_vec = vector(&fluid.position[j], &fluid.position[0]);
            expected += fluid.mass[j] * CubicBSpline3D::kernel_function(&r_vec, h)
                + dt * (fluid.solver_velocity_slots[0][0] - fluid.solver_velocity_slots[0][j])
                    .dot(&CubicBSpline3D::kernel_gradient(&r_vec, h));
        }
        assert!((solver.density_pred[0] - expected).abs() < 1e-9);
    }

    #[test]
    fn calc_predicted_density_matches_manual_formula_for_a_boundary_neighbor() {
        let h = 1.0;
        let dt = 0.05;
        let params = make_system_params(dt, h, 0.3, 0.0);
        let mut solver = make_solver(100.0);

        let mut fluid = fluid_with_at_least(1);
        fluid.position[0] = Point3::origin();
        fluid.solver_velocity_slots[0][0] = Vector3::new(1.0, 0.0, 0.0);
        fluid.mass[0] = 0.5;
        solver.resize_scratch(fluid.len());

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
            center_of_mass: None,
            ..Default::default()
        });

        solver.calc_predicted_density::<CubicBSpline3D>(
            &mut fluid,
            &boundary,
            &neighbor_list,
            &params,
        );

        let r_vec = vector(&boundary_pos, &fluid.position[0]);
        let expected = boundary_vol
            * (fluid.mass[0] / params.rest_volume)
            * CubicBSpline3D::kernel_function(&r_vec, h)
            + dt * fluid.solver_velocity_slots[0][0]
                .dot(&CubicBSpline3D::kernel_gradient(&r_vec, h));

        assert!((solver.density_pred[0] - expected).abs() < 1e-9);
    }

    // ─── solve_and_add_acceleration: end-to-end ─────────────────────────

    #[test]
    fn solve_and_add_acceleration_on_an_isolated_particle_yields_zero_pressure() {
        // With no neighbors, density_pred == 0.0 -> "volume" = mass/0.0 ==
        // +inf -> rest_volume/+inf - 1.0 == -1.0 -> max(..., 0.0) == 0.0.
        // Documents this degenerate-but-defined outcome explicitly.
        let h = 1.0;
        let params = make_system_params(0.05, h, 0.3, 0.0);
        let mut solver = make_solver(500.0);

        let mut fluid = fluid_with_at_least(1);
        fluid.position[0] = Point3::origin();
        fluid.velocity[0] = Vector3::zeros();
        fluid.mass[0] = 0.5;
        let preexisting = Vector3::new(0.0, 0.0, -9.81);
        fluid.acceleration[0] = preexisting;

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
        assert_eq!(fluid.acceleration[0], preexisting);
    }

    #[test]
    fn solve_and_add_acceleration_matches_manual_formula_for_a_compressed_pair() {
        let h = 1.0;
        let dt = 0.05;
        let params = make_system_params(dt, h, 0.3, 0.0);
        let stiffness = 500.0;
        let mut solver = make_solver(stiffness);

        let mut fluid = fluid_with_at_least(2);
        let positions = [Point3::new(0.0, 0.0, 0.0), Point3::new(0.3, 0.0, 0.0)];
        for i in 0..2 {
            fluid.position[i] = positions[i];
            fluid.velocity[i] = Vector3::zeros();
            fluid.acceleration[i] = Vector3::zeros();
            fluid.mass[i] = 0.5;
        }

        let neighbor_list = build_fluid_neighbor_list(&fluid.position, h);
        let mut boundary = VolumeMapBoundary::default();
        let mut properties = CurrentSystemProperties::default();

        solver.solve_and_add_acceleration::<CubicBSpline3D>(
            &mut fluid,
            &mut boundary,
            &neighbor_list,
            &params,
            &mut properties,
        );

        // Independently recompute density_pred (vel_pred == vel == 0 here,
        // so the divergence term vanishes) and the resulting pressure.
        let mut expected_density_pred = vec![0.0; 2];
        for id in 0..2 {
            for &j in neighbor_list.get_neighbors(id) {
                let r_vec = vector(&positions[j], &positions[id]);
                expected_density_pred[id] +=
                    fluid.mass[j] * CubicBSpline3D::kernel_function(&r_vec, h);
            }
        }
        let expected_pressures: Vec<f64> = expected_density_pred
            .iter()
            .zip(&fluid.mass)
            .map(|(&rho, &m)| {
                let volume = m / rho;
                stiffness * f64::max(params.rest_volume / volume - 1.0, 0.0)
            })
            .collect();

        for (id, &ep) in expected_pressures.iter().enumerate() {
            assert!((fluid.pressure[id] - ep).abs() < 1e-9, "id={id}");
        }
    }

    #[test]
    fn solve_and_add_acceleration_registers_reaction_force_only_for_dynamic_boundaries() {
        let h = 1.0;
        let weighting = 1.0;
        let params = make_system_params(0.05, h, 0.3, weighting);
        let mut solver = make_solver(500.0);

        let mut fluid = fluid_with_at_least(1);
        fluid.position[0] = Point3::origin();
        fluid.velocity[0] = Vector3::zeros();
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
        let mut properties = CurrentSystemProperties::default();

        solver.solve_and_add_acceleration::<CubicBSpline3D>(
            &mut fluid,
            &mut boundary,
            &neighbor_list,
            &params,
            &mut properties,
        );

        let force_on_fluid = fluid.mass[0] * fluid.acceleration[0];
        assert_eq!(boundary.recorded_forces.len(), 1);
        assert!((boundary.recorded_forces[0].force - (-force_on_fluid)).norm() < 1e-9);
    }
}
