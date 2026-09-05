//! Integration tests for `IISPH`, exercising only its public API.
//! Private fields (`s_f`, `a_ff`) and private methods
//! (`set_diagonal_element`, `initialize`, `continue_solving`) are covered
//! separately in `iisph`'s own internal test module.

use nalgebra::{Point3, Vector3};
use parry3d_f64::math::Vec3;
use parry3d_f64::shape::TriMesh;

use simulation_lib::neighbor_search::{NeighborList, NeighborSearch};
use simulation_lib::render_info::BoundaryVisualization;
use simulation_lib::sph::SystemParameters;
use simulation_lib::sph::boundary_handling::{
    Boundary, BoundaryCheckpoint, BoundaryHandling, ForceOntoBoundary, RequestMode,
    VolumeMapBoundary,
};
use simulation_lib::sph::fluid::{Fluid, Len};
use simulation_lib::sph::kernel::{CubicBSpline3D, KernelFn};
use simulation_lib::sph::pressure_solver::iisph::TerminationCondition;
use simulation_lib::sph::pressure_solver::{IISPH, PressureSolver};
use simulation_lib::sph::setup::input::{DynamicBoundaryDef, Parameters, StaticBoundaryDef};
use simulation_lib::utilities::triangle_mesh::MeshContainer;
use simulation_lib::utilities::vector;

// ─── Fixtures / helpers ─────────────────────────────────────────────────

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
        // `time_increment` is private and can only be set indirectly by the
        // private `set_cfl_time_step` (called from `System::update()`),
        // which isn't reachable from this external test file. Under this
        // build, freshly constructed `SystemParameters` always start at
        // `time_increment == 0.0` — `dt` therefore cannot be honored here.
        // Tests that need an exact, nonzero `dt` to compute an independently
        // derived expected value are gated with
        // `#[cfg(not(feature = "cfl_time_step"))]` below.
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

/// Mirrors what `System::new_boxed` does via `PressureSolver::POSITION_SLOTS`/
/// `VELOCITY_SLOTS` before any solver method runs on a `Fluid`. `IISPH`
/// declares `POSITION_SLOTS = 1`/`VELOCITY_SLOTS = 1` (see
/// `pressure_solver/iisph.rs`), so every test that calls
/// `set_source_term_vp`, `set_diagonal_element`, `resolve_pressure_globally`,
/// or `solve_and_add_acceleration` needs this — those methods index
/// `fluid.solver_position_slots[0]`/`solver_velocity_slots[0]`
/// unconditionally, even when `with_pred_positions == false`.
fn with_solver_slots(mut fluid: Fluid) -> Fluid {
    fluid.resize_slots(0, 0, 1, 1);
    fluid
}

fn dummy_properties() -> CurrentSystemProperties {
    CurrentSystemProperties::default()
}

// ─── Mock boundary (same pattern as SESPH/pressure_solver test suites) ──

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
        _mesh: &mut MeshContainer,
        _boundary: &StaticBoundaryDef,
        _r: f64,
        _k: f64,
    ) {
        unimplemented!("not exercised by IISPH tests")
    }
    fn add_dynamic_boundary(
        &mut self,
        _mesh: &mut MeshContainer,
        _boundary: &DynamicBoundaryDef,
        _r: f64,
        _k: f64,
    ) {
        unimplemented!("not exercised by IISPH tests")
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
    fn get_visualization(&self, _s: &BoundaryVisualization) -> BoundaryVisualization {
        unimplemented!("not exercised by IISPH tests")
    }
    fn get_checkpoint(&self) -> BoundaryCheckpoint {
        BoundaryCheckpoint::default()
    }
    fn restore_from_checkpoint(&mut self, _s: &BoundaryCheckpoint) {}
}

// ─── new / measurement_info ─────────────────────────────────────────────

#[test]
fn new_copies_solver_parameters_from_parameters() {
    let solver = IISPH::new(&make_solver_params(0.05, 0.6, 1e-7));
    assert_eq!(solver.target_density_error, 0.05);
    assert_eq!(solver.relaxation_factor, 0.6);
    assert_eq!(solver.min_diagonal_element, 1e-7);
    assert_eq!(solver.last_solver_iterations, 0);
    assert_eq!(solver.predicted_density_error, 0.0);
    assert!(solver.pressure_acc_f.is_empty());
}

#[test]
fn measurement_info_reflects_current_solver_state() {
    let mut solver = IISPH::new(&make_solver_params(0.05, 0.6, 1e-7));
    solver.last_solver_iterations = 7;
    solver.predicted_density_error = 0.03;

    let info = solver.measurement_info();
    assert_eq!(info.target_density_error, 0.05);
    assert_eq!(info.relaxation_factor, 0.6);
    assert_eq!(info.solver_iterations, 7);
    assert_eq!(info.predicted_density_error, 0.03);
    assert_eq!(info.stiffness, 0.0); // not used by IISPH -> default
}

// ─── resize_scratch: publicly observable via pressure_acc_f ─────────────

#[test]
fn resize_scratch_resizes_the_public_pressure_acc_f_buffer() {
    let mut solver = IISPH::new(&make_solver_params(0.01, 0.5, 1e-9));
    solver.resize_scratch(4);
    assert_eq!(solver.pressure_acc_f.len(), 4);
    solver.resize_scratch(2);
    assert_eq!(solver.pressure_acc_f.len(), 2);
}

// ─── TerminationCondition: black-box control over iteration count ───────

#[test]
fn resolve_pressure_globally_after_iteration_zero_never_enters_the_loop() {
    let h = 1.0;
    let params = make_system_params(0.05, h, 0.3, 0.0);
    let mut solver = IISPH::new(&make_solver_params(0.01, 0.5, 1e-9));

    let mut fluid = with_solver_slots(fluid_with_at_least(1));
    fluid.position[0] = Point3::origin();
    fluid.volume[0] = rest_volume_for(0.3);
    fluid.mass[0] = 0.5;
    solver.resize_scratch(fluid.len());

    let neighbor_list = NeighborList::new(fluid.len());
    let mut boundary = VolumeMapBoundary::default();

    solver.set_source_term_vp::<CubicBSpline3D>(&fluid, &boundary, &neighbor_list, &params, false);
    solver.resolve_pressure_globally::<CubicBSpline3D>(
        &mut fluid,
        &mut boundary,
        &neighbor_list,
        &params,
        false,
        TerminationCondition::AfterIteration(0),
        true,
    );

    assert_eq!(solver.last_solver_iterations, 0);
}

#[test]
fn resolve_pressure_globally_after_iteration_runs_exactly_the_requested_count() {
    let h = 1.0;
    let params = make_system_params(0.05, h, 0.3, 0.0);
    let mut solver = IISPH::new(&make_solver_params(0.01, 0.5, 1e-9));

    let mut fluid = with_solver_slots(fluid_with_at_least(1));
    fluid.position[0] = Point3::origin();
    fluid.volume[0] = rest_volume_for(0.3) * 0.5;
    fluid.mass[0] = 0.5;
    solver.resize_scratch(fluid.len());

    let neighbor_list = NeighborList::new(fluid.len());
    let mut boundary = VolumeMapBoundary::default();

    solver.set_source_term_vp::<CubicBSpline3D>(&fluid, &boundary, &neighbor_list, &params, false);
    solver.resolve_pressure_globally::<CubicBSpline3D>(
        &mut fluid,
        &mut boundary,
        &neighbor_list,
        &params,
        false,
        TerminationCondition::AfterIteration(3),
        true,
    );

    assert_eq!(solver.last_solver_iterations, 3);
}

// ─── resolve_pressure_globally: pressure clamping (end-to-end, exact) ────

#[test]
fn resolve_pressure_globally_clamps_negative_pressure_to_zero_on_expansion() {
    // Isolated fluid particle + one static boundary neighbor is the only
    // way to get a_ff != 0 without also needing a second fluid particle;
    // with `AfterIteration(0)` the result is entirely determined by
    // `initialize`'s single pass, so no CG convergence needs to be
    // reasoned about.
    let h = 1.0;
    let weighting = 0.5;
    let params = make_system_params(0.05, h, 0.3, weighting);
    let mut solver = IISPH::new(&make_solver_params(0.01, 0.5, 1e-9));

    let mut fluid = with_solver_slots(fluid_with_at_least(1));
    fluid.position[0] = Point3::origin();
    fluid.velocity[0] = Vector3::zeros();
    fluid.solver_velocity_slots[0][0] = Vector3::zeros();
    fluid.acceleration[0] = Vector3::zeros();
    fluid.volume[0] = rest_volume_for(0.3) * 2.0; // expanded -> s_f > 0
    fluid.mass[0] = 0.5;
    solver.resize_scratch(fluid.len());

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
        center_of_mass: None,
        ..Default::default()
    });

    solver.set_source_term_vp::<CubicBSpline3D>(&fluid, &boundary, &neighbor_list, &params, false);
    solver.resolve_pressure_globally::<CubicBSpline3D>(
        &mut fluid,
        &mut boundary,
        &neighbor_list,
        &params,
        false,
        TerminationCondition::AfterIteration(0),
        true, // clamp_pressure
    );

    assert_eq!(
        fluid.pressure[0], 0.0,
        "expansion should never produce negative pressure when clamp_pressure is true"
    );
}

#[cfg(not(feature = "cfl_time_step"))]
#[test]
fn resolve_pressure_globally_matches_manual_formula_on_compression() {
    let h = 1.0;
    let weighting = 0.5;
    let dt = 0.05;
    let params = make_system_params(dt, h, 0.3, weighting);
    let mut solver = IISPH::new(&make_solver_params(0.01, 0.5, 1e-9));

    let mut fluid = with_solver_slots(fluid_with_at_least(1));
    fluid.position[0] = Point3::origin();
    fluid.velocity[0] = Vector3::zeros();
    fluid.solver_velocity_slots[0][0] = Vector3::zeros();
    fluid.acceleration[0] = Vector3::zeros();
    let volume = rest_volume_for(0.3) * 0.5; // compressed -> s_f < 0
    fluid.volume[0] = volume;
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

    solver.set_source_term_vp::<CubicBSpline3D>(&fluid, &boundary, &neighbor_list, &params, false);
    solver.resolve_pressure_globally::<CubicBSpline3D>(
        &mut fluid,
        &mut boundary,
        &neighbor_list,
        &params,
        false,
        TerminationCondition::AfterIteration(0),
        true,
    );

    let r_vec = vector(&boundary_pos, &fluid.position[0]);
    let sum_boundary = boundary_vol * CubicBSpline3D::kernel_gradient(&r_vec, h);
    let c_f = -fluid.mass[0].recip() * fluid.volume[0] * (2.0 * weighting * sum_boundary);
    let a_ff = dt.powi(2) * c_f.dot(&sum_boundary);
    let s_f = 1.0 - rest_volume_for(0.3) / volume;
    let expected_pressure = (0.5 * s_f / a_ff).max(0.0);

    assert!(
        (fluid.pressure[0] - expected_pressure).abs() < 1e-9,
        "expected {expected_pressure}, got {}",
        fluid.pressure[0]
    );
    assert!(
        fluid.pressure[0] > 0.0,
        "compression should yield positive pressure"
    );
}

// ─── solve_and_add_acceleration: end-to-end contract checks ─────────────

#[test]
fn solve_and_add_acceleration_on_an_isolated_particle_at_rest_volume_is_a_noop() {
    // An isolated particle (no fluid or boundary neighbors) can never get
    // a nonzero diagonal element, so regardless of the termination
    // condition it always ends up with zero pressure and therefore zero
    // pressure acceleration — the pre-existing acceleration must survive
    // untouched.
    let h = 1.0;
    let params = make_system_params(0.05, h, 0.3, 0.0);
    let mut solver = IISPH::new(&make_solver_params(0.01, 0.5, 1e-9));

    let mut fluid = with_solver_slots(fluid_with_at_least(1));
    for v in fluid.volume.iter_mut() {
        *v = rest_volume_for(0.3);
    }

    fluid.position[0] = Point3::origin();
    fluid.velocity[0] = Vector3::zeros();
    fluid.volume[0] = rest_volume_for(0.3);
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

    assert_eq!(fluid.pressure[0], 0.0);
    assert_eq!(fluid.acceleration[0], preexisting);
    // `TargetDensityError` always runs at least 2 iterations regardless of
    // convergence — see `continue_solving`'s contract.
    assert_eq!(solver.last_solver_iterations, 2);
    assert_eq!(solver.predicted_density_error, 0.0);
}

#[test]
fn solve_and_add_acceleration_registers_reaction_force_exactly_once_on_a_dynamic_boundary() {
    // Even though `resolve_pressure_globally` internally probes an
    // intermediate pressure field over multiple CG iterations, only the
    // FINAL `add_pressure_acceleration` call (with `custom_target ==
    // None`) at the end of `solve_and_add_acceleration` may register a
    // reaction force — not once per iteration.
    let h = 1.0;
    let weighting = 1.0;
    let params = make_system_params(0.05, h, 0.3, weighting);
    let mut solver = IISPH::new(&make_solver_params(0.01, 0.5, 1e-9));

    let mut fluid = with_solver_slots(fluid_with_at_least(1));
    for v in fluid.volume.iter_mut() {
        *v = rest_volume_for(0.3);
    }

    fluid.position[0] = Point3::origin();
    fluid.velocity[0] = Vector3::zeros();
    fluid.volume[0] = rest_volume_for(0.3) * 0.5;
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

    assert_eq!(
        boundary.recorded_forces.len(),
        1,
        "expected exactly one reaction force, not one per CG iteration"
    );
    let force_on_fluid = fluid.mass[0] * fluid.acceleration[0];
    assert!((boundary.recorded_forces[0].force - (-force_on_fluid)).norm() < 1e-9);
}

// ─── TerminationCondition: basic construction ───────────────────────────

#[test]
fn termination_condition_variants_are_constructible() {
    let _ = TerminationCondition::AfterIteration(5);
    let _ = TerminationCondition::TargetDensityError(0.01);
}

// ─── Trait bounds / basic usability ─────────────────────────────────────

fn assert_implements_pressure_solver<T: PressureSolver>() {}

#[test]
fn iisph_implements_pressure_solver_and_is_cloneable() {
    assert_implements_pressure_solver::<IISPH>();
    let mut solver = IISPH::new(&make_solver_params(0.02, 0.5, 1e-9));
    solver.last_solver_iterations = 3;
    let cloned = solver.clone();
    assert_eq!(cloned.last_solver_iterations, 3);
}
