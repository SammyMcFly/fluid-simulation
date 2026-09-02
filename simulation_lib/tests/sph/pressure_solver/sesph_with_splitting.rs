//! Integration tests for `SESPHwSplitting`, exercising only its public
//! API (`PressureSolver::new`, `solve_and_add_acceleration`,
//! `measurement_info`, `resize_scratch`). The private `stiffness`/
//! `density_pred` fields and the private `calc_predicted_density` method
//! are covered separately in this module's own internal test block.

use nalgebra::{Point3, Vector3};
use parry3d_f64::math::Vec3;
use parry3d_f64::shape::TriMesh;

use simulation_lib::neighbor_search::{NeighborList, NeighborSearch, SpatialHashing};
use simulation_lib::render_info::BoundaryVisualization;
use simulation_lib::sph::CurrentSystemProperties;
use simulation_lib::sph::SystemParameters;
use simulation_lib::sph::boundary_handling::{
    Boundary, BoundaryCheckpoint, BoundaryHandling, ForceOntoBoundary, RequestMode,
    VolumeMapBoundary,
};
use simulation_lib::sph::fluid::{Fluid, Len};
use simulation_lib::sph::kernel::{CubicBSpline3D, KernelFn};
use simulation_lib::sph::pressure_solver::{PressureSolver, SESPHwSplitting};
use simulation_lib::sph::setup::input::{DynamicBoundaryDef, Parameters, StaticBoundaryDef};
use simulation_lib::utilities::triangle_mesh::MeshContainer;

// ─── Fixtures / helpers ─────────────────────────────────────────────────

fn make_solver_params(stiffness: f64) -> Parameters {
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
        stiffness,
        target_density_error: 0.0,
        relaxation_factor: 0.0,
        min_diagonal_element: 0.0,
    }
}

/// `time_increment` is a private field on `SystemParameters` and — unlike
/// `rest_volume`, which can be reconstructed via
/// `rest_density_grid_spacing.powi(3)` — has NO publicly derivable value
/// under the `cfl_time_step` feature: freshly constructed
/// `SystemParameters` always start at `time_increment == 0.0` there, with
/// no public way to change it (only the private `set_cfl_time_step`,
/// called from `System::update`, ever does). `dt` is therefore honored
/// only when built WITHOUT that feature; tests needing an exact, nonzero
/// `dt` to independently derive an expected value are gated accordingly.
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

/// Mirrors `SystemParameters::new`'s private `rest_volume` computation
/// (`rest_density_grid_spacing.powi(3)`), since `rest_volume` itself is a
/// private field not accessible from this external test file.
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

fn build_fluid_neighbor_list(positions: &[Point3<f64>], radius: f64) -> NeighborList {
    let mut ns = SpatialHashing::new(radius);
    let mut neighbor_list = NeighborList::new(positions.len());
    ns.find_samples(radius, positions, positions, &mut neighbor_list);
    neighbor_list
}

fn dummy_properties() -> CurrentSystemProperties {
    CurrentSystemProperties::default()
}

// ─── Mock boundary (same pattern as SESPH/IISPH external test suites) ──

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
        _r: f64,
        _k: f64,
    ) {
        unimplemented!("not exercised by SESPHwSplitting tests")
    }
    fn add_dynamic_boundary(
        &mut self,
        _mesh: &mut MeshContainer,
        _boundary: &DynamicBoundaryDef,
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
            id: force.id,
            force: force.force,
            force_location: force.force_location,
        });
    }
    fn step_forward_in_time(&mut self, _dt: f64) {}
    fn get_fluid_depth(&self, _v: f64) -> f64 {
        0.0
    }
    fn get_visualization(&self, _s: &BoundaryVisualization) -> BoundaryVisualization {
        unimplemented!("not exercised by SESPHwSplitting tests")
    }
    fn get_checkpoint(&self) -> BoundaryCheckpoint {
        BoundaryCheckpoint::default()
    }
    fn restore_from_checkpoint(&mut self, _s: &BoundaryCheckpoint) {}
}

// ─── new / measurement_info ─────────────────────────────────────────────

#[test]
fn new_captures_the_configured_stiffness() {
    let solver = SESPHwSplitting::new(&make_solver_params(500.0));
    let info = solver.measurement_info();
    assert_eq!(info.stiffness, 500.0);
}

#[test]
fn measurement_info_reports_default_for_every_other_field() {
    let solver = SESPHwSplitting::new(&make_solver_params(123.0));
    let info = solver.measurement_info();
    assert_eq!(info.target_density_error, 0.0);
    assert_eq!(info.solver_iterations, 0);
    assert_eq!(info.relaxation_factor, 0.0);
    assert_eq!(info.predicted_density_error, 0.0);
}

// ─── resize_scratch ─────────────────────────────────────────────────────

#[test]
fn resize_scratch_does_not_panic_and_is_idempotent_with_solve_and_add_acceleration() {
    // `resize_scratch` itself has no publicly-observable state (the
    // resized buffer, `density_pred`, is private) — its only externally
    // checkable effect is that calling it manually beforehand doesn't
    // break a subsequent `solve_and_add_acceleration` call (which resizes
    // it again internally to the same length).
    let h = 1.0;
    let params = make_system_params(0.05, h, 0.3, 0.0);
    let mut solver = SESPHwSplitting::new(&make_solver_params(500.0));

    let mut fluid = fluid_with_at_least(1);
    fluid.position[0] = Point3::origin();
    fluid.velocity[0] = Vector3::zeros();
    fluid.mass[0] = 0.5;
    fluid.acceleration[0] = Vector3::zeros();

    solver.resize_scratch(fluid.len());
    solver.resize_scratch(fluid.len()); // calling it twice must also be harmless

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

    assert!(fluid.pressure[0].is_finite());
}

// ─── solve_and_add_acceleration: contract checks ────────────────────────

#[test]
fn solve_and_add_acceleration_on_an_isolated_particle_yields_zero_pressure() {
    let h = 1.0;
    let params = make_system_params(0.05, h, 0.3, 0.0);
    let mut solver = SESPHwSplitting::new(&make_solver_params(500.0));

    let mut fluid = fluid_with_at_least(1);
    fluid.position[0] = Point3::origin();
    fluid.velocity[0] = Vector3::zeros();
    fluid.mass[0] = 0.5;
    let preexisting = Vector3::new(0.0, 0.0, -9.81);
    fluid.acceleration[0] = preexisting;

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

    // An isolated particle has density_pred == 0.0, so "volume" =
    // mass/0.0 == +inf -> the state equation clamps to zero pressure.
    assert_eq!(fluid.pressure[0], 0.0);
    assert_eq!(
        fluid.acceleration[0], preexisting,
        "zero pressure -> zero pressure acceleration -> preexisting acceleration preserved"
    );
}

#[cfg(not(feature = "cfl_time_step"))]
#[test]
fn solve_and_add_acceleration_matches_manual_formula_for_a_compressed_pair() {
    let h = 1.0;
    let dt = 0.05;
    let params = make_system_params(dt, h, 0.3, 0.0);
    let rv = rest_volume_for(0.3);
    let stiffness = 500.0;
    let mut solver = SESPHwSplitting::new(&make_solver_params(stiffness));

    let mut fluid = fluid_with_at_least(2);
    let positions = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.3, 0.0, 0.0)];
    for i in 0..2 {
        fluid.position[i] = positions[i];
        fluid.velocity[i] = Vector3::zeros();
        fluid.acceleration[i] = Vector3::zeros();
        fluid.mass[i] = 0.5;
    }

    let neighbor_list = build_fluid_neighbor_list(&fluid.position, h);
    let mut boundary = VolumeMapBoundary::default();
    let mut properties = dummy_properties();

    solver.solve_and_add_acceleration::<CubicBSpline3D>(
        &mut fluid,
        &mut boundary,
        &neighbor_list,
        &params,
        &mut properties,
    );

    let mut expected_density_pred = vec![0.0; 2];
    for id in 0..2 {
        for &j in neighbor_list.get_neighbors(id) {
            let r_vec = simulation_lib::utilities::vector(&positions[j], &positions[id]);
            expected_density_pred[id] += fluid.mass[j] * CubicBSpline3D::kernel_function(&r_vec, h);
        }
    }
    let expected_pressures: Vec<f64> = expected_density_pred
        .iter()
        .zip(&fluid.mass)
        .map(|(&rho, &m)| {
            let volume = m / rho;
            stiffness * f64::max(rv / volume - 1.0, 0.0)
        })
        .collect();

    for id in 0..2 {
        assert!(
            (fluid.pressure[id] - expected_pressures[id]).abs() < 1e-9,
            "id={id}: expected {}, got {}",
            expected_pressures[id],
            fluid.pressure[id]
        );
    }
}

#[test]
fn solve_and_add_acceleration_includes_boundary_contribution_without_reaction_force_for_static_boundaries()
 {
    let h = 1.0;
    let weighting = 0.5;
    let params = make_system_params(0.05, h, 0.3, weighting);
    let mut solver = SESPHwSplitting::new(&make_solver_params(500.0));

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
        center_of_mass: None, // static
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

    assert!(
        boundary.recorded_forces.is_empty(),
        "a static boundary must not receive a reaction force"
    );
}

#[test]
fn solve_and_add_acceleration_registers_newtons_third_law_reaction_force_on_dynamic_boundaries() {
    let h = 1.0;
    let weighting = 1.0;
    let params = make_system_params(0.05, h, 0.3, weighting);
    let mut solver = SESPHwSplitting::new(&make_solver_params(500.0));

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
    let mut properties = dummy_properties();

    solver.solve_and_add_acceleration::<CubicBSpline3D>(
        &mut fluid,
        &mut boundary,
        &neighbor_list,
        &params,
        &mut properties,
    );

    let force_on_fluid = fluid.mass[0] * fluid.acceleration[0];

    assert_eq!(boundary.recorded_forces.len(), 1);
    let recorded = boundary.recorded_forces[0];
    assert_eq!(recorded.id, 0);
    assert!((recorded.force - (-force_on_fluid)).norm() < 1e-9);
    assert_eq!(recorded.force_location, boundary_pos);
}

// ─── Trait bounds / basic usability ─────────────────────────────────────

fn assert_implements_pressure_solver<T: PressureSolver>() {}

#[test]
fn sesph_with_splitting_implements_pressure_solver_and_is_cloneable() {
    assert_implements_pressure_solver::<SESPHwSplitting>();
    let solver = SESPHwSplitting::new(&make_solver_params(100.0));
    let cloned = solver.clone();
    assert_eq!(cloned.measurement_info().stiffness, 100.0);
}
