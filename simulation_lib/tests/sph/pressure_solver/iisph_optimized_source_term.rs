//! Integration tests for `IISPHwOST`, exercising only its public API.
//! The single private field (`inner: IISPH`) and any deeper reproduction
//! of intermediate CG state are covered separately in this module's own
//! internal test block, since only `IISPH`'s already-public fields would
//! be observable externally through `measurement_info()` anyway.

use nalgebra::{Point3, Vector3};
use parry3d_f64::math::Vec3;
use parry3d_f64::shape::TriMesh;

use simulation_lib::neighbor_search::{NeighborList, NeighborSearch};
use simulation_lib::render_info::BoundaryVisualization;
use simulation_lib::sph::CurrentSystemProperties;
use simulation_lib::sph::SystemParameters;
use simulation_lib::sph::boundary_handling::VolumeMapBoundary;
use simulation_lib::sph::boundary_handling::{
    Boundary, BoundaryCheckpoint, BoundaryHandling, ForceOntoBoundary, RequestMode,
};
use simulation_lib::sph::fluid::{Fluid, Len};
use simulation_lib::sph::kernel::{CubicBSpline3D, KernelFn};
use simulation_lib::sph::pressure_solver::{IISPHwOST, PressureSolver};
use simulation_lib::sph::setup::input::{DynamicBoundaryDef, Parameters, StaticBoundaryDef};
use simulation_lib::utilities::triangle_mesh::MeshContainer;

// ─── Fixtures / helpers (see internal test module for full derivation
// comments) ────────────────────────────────────────────────────────────

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

/// `time_increment` cannot be set externally under `cfl_time_step` (see
/// prior test suites for the same crate); tests needing an exact, nonzero
/// `dt` for hand-derived expectations are gated with
/// `#[cfg(not(feature = "cfl_time_step"))]`.
fn make_system_params(
    dt: f64,
    kernel_support_radius: f64,
    rest_density_grid_spacing: f64,
    boundary_pressure_acceleration_weighting: f64,
) -> SystemParameters {
    #[cfg(not(feature = "cfl_time_step"))]
    {
        SystemParameters::new(
            dt,
            rest_density_grid_spacing,
            kernel_support_radius,
            -1e9,
            0.0,
            0.0,
            boundary_pressure_acceleration_weighting,
        )
    }
    #[cfg(feature = "cfl_time_step")]
    {
        let _ = dt;
        SystemParameters::new(
            0.4,
            0.4,
            rest_density_grid_spacing,
            kernel_support_radius,
            -1e9,
            0.0,
            0.0,
            boundary_pressure_acceleration_weighting,
        )
    }
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

fn dummy_properties() -> CurrentSystemProperties {
    CurrentSystemProperties::default()
}

// ─── Mock boundary ──────────────────────────────────────────────────────

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
    _force: Vector3<f64>,
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
        _mesh: &mut MeshContainer,
        _boundary: &StaticBoundaryDef,
        _r: f64,
        _k: f64,
    ) {
        unimplemented!("not exercised by IISPHwOST tests")
    }
    fn add_dynamic_boundary(
        &mut self,
        _mesh: &mut MeshContainer,
        _boundary: &DynamicBoundaryDef,
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
            _id: force.id,
            _force: force.force,
            _force_location: force.force_location,
        });
    }
    fn step_forward_in_time(&mut self, _dt: f64) {}
    fn get_fluid_depth(&self, _v: f64) -> f64 {
        0.0
    }
    fn get_visualization(&self, _s: &BoundaryVisualization) -> BoundaryVisualization {
        unimplemented!("not exercised by IISPHwOST tests")
    }
    fn get_checkpoint(&self) -> BoundaryCheckpoint {
        BoundaryCheckpoint::default()
    }
    fn restore_from_checkpoint(&mut self, _s: &BoundaryCheckpoint) {}
}

// ─── new / measurement_info ─────────────────────────────────────────────

#[test]
fn measurement_info_surfaces_target_density_error_and_relaxation_factor() {
    let solver = IISPHwOST::new(&make_solver_params(0.02, 0.6, 1e-7));
    let info = solver.measurement_info();
    assert_eq!(info.target_density_error, 0.02);
    assert_eq!(info.relaxation_factor, 0.6);
    // Freshly constructed -> no solve yet -> defaults for the rest.
    assert_eq!(info.solver_iterations, 0);
    assert_eq!(info.predicted_density_error, 0.0);
    assert_eq!(info.stiffness, 0.0);
}

// ─── Deterministic end-to-end trace: isolated particle ─────────────────
//
// See the internal test module's derivation comment for the full,
// hand-worked-out reasoning behind these expected values.

#[cfg(not(feature = "cfl_time_step"))]
#[test]
fn solve_and_add_acceleration_on_an_isolated_particle_matches_the_exact_hand_derivation() {
    let h = 1.0;
    let dt = 0.05;
    let params = make_system_params(dt, h, 0.3, 0.0);
    let mut solver = IISPHwOST::new(&make_solver_params(0.01, 0.5, 1e-9));

    let mut fluid = fluid_with_at_least(1);
    for v in fluid.volume.iter_mut() {
        *v = rest_volume_for(0.3);
    }
    let pos0 = Point3::new(1.0, 2.0, 3.0);
    let vel0 = Vector3::new(0.5, 0.0, 0.0);
    let g = Vector3::new(0.0, 0.0, -9.81);
    fluid.position[0] = pos0;
    fluid.velocity[0] = vel0;
    fluid.mass[0] = 0.5;
    fluid.volume[0] = rest_volume_for(0.3);
    fluid.acceleration[0] = g; // expected to be discarded, not accumulated

    let neighbor_list = NeighborList::new(fluid.len());
    let mut boundary = VolumeMapBoundary::default();
    let mut properties = dummy_properties();

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
        "acceleration must be overwritten to zero, not retain the preexisting gravity g \
         — see the internal test module's derivation comment for why"
    );

    let expected_vel_pred = vel0 + dt * g;
    assert!((fluid.velocity_pred[0] - expected_vel_pred).norm() < 1e-9);

    let expected_pos_pred = pos0 + dt * expected_vel_pred;
    assert!((fluid.position_pred[0] - expected_pos_pred).norm() < 1e-9);
}

// ─── Dynamic boundary: force registered once per stage (2 total) ───────

#[test]
fn solve_and_add_acceleration_registers_a_reaction_force_once_per_stage_on_dynamic_boundaries() {
    let h = 1.0;
    let weighting = 1.0;
    let params = make_system_params(0.05, h, 0.3, weighting);
    let mut solver = IISPHwOST::new(&make_solver_params(0.01, 0.5, 1e-9));

    let mut fluid = fluid_with_at_least(1);
    for v in fluid.volume.iter_mut() {
        *v = rest_volume_for(0.3);
    }
    fluid.position[0] = Point3::origin();
    fluid.velocity[0] = Vector3::zeros();
    fluid.mass[0] = 0.5;
    fluid.volume[0] = rest_volume_for(0.3) * 0.5;
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
        center_of_mass: Some(Point3::new(5.0, 0.0, 0.0)),
        ..Default::default()
    });
    let mut properties = dummy_properties();

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

    let mut fluid = fluid_with_at_least(1);
    for v in fluid.volume.iter_mut() {
        *v = rest_volume_for(0.3);
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
        center_of_mass: None,
        ..Default::default()
    });
    let mut properties = dummy_properties();

    solver.solve_and_add_acceleration::<CubicBSpline3D>(
        &mut fluid,
        &mut boundary,
        &neighbor_list,
        &params,
        &mut properties,
    );

    assert!(boundary.recorded_forces.is_empty());
}

// ─── Trait bounds / basic usability ─────────────────────────────────────

fn assert_implements_pressure_solver<T: PressureSolver>() {}

#[test]
fn iisph_wost_implements_pressure_solver_and_is_cloneable() {
    assert_implements_pressure_solver::<IISPHwOST>();
    let solver = IISPHwOST::new(&make_solver_params(0.02, 0.5, 1e-9));
    let cloned = solver.clone();
    assert_eq!(
        cloned.measurement_info().target_density_error,
        solver.measurement_info().target_density_error
    );
}
