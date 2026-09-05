//! Integration tests for `SESPH`, exercising only its public API
//! (`PressureSolver::new`, `solve_and_add_acceleration`,
//! `measurement_info`). The private `stiffness` field is never accessed
//! directly — its value is observed exclusively through
//! `measurement_info()`, which is the intended public way to inspect it.

use nalgebra::{Point3, Vector3};
use parry3d_f64::math::Vec3;
use parry3d_f64::shape::TriMesh;

use simulation_lib::neighbor_search::{NeighborList, NeighborSearch, SpatialHashing};
use simulation_lib::render_info::BoundaryVisualization;
use simulation_lib::sph::SystemParameters;
use simulation_lib::sph::boundary_handling::{
    Boundary, BoundaryCheckpoint, BoundaryHandling, ForceOntoBoundary, RequestMode,
    VolumeMapBoundary,
};
use simulation_lib::sph::fluid::{Fluid, Len};
use simulation_lib::sph::kernel::{CubicBSpline3D, KernelFn};
use simulation_lib::sph::pressure_solver::{PressureSolver, SESPH};
use simulation_lib::sph::setup::input::{DynamicBoundaryDef, Parameters, StaticBoundaryDef};
use simulation_lib::utilities::triangle_mesh::MeshContainer;
use simulation_lib::utilities::vector;

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

fn make_system_params(
    kernel_support_radius: f64,
    rest_density_grid_spacing: f64,
    boundary_pressure_acceleration_weighting: f64,
) -> SystemParameters {
    #[cfg(not(feature = "cfl_time_step"))]
    {
        SystemParameters::new(
            0.001,
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
        unimplemented!("not exercised by SESPH tests")
    }

    fn add_dynamic_boundary(
        &mut self,
        _mesh: &mut MeshContainer,
        _boundary: &DynamicBoundaryDef,
        _rest_density_grid_spacing: f64,
        _kernel_support_radius: f64,
    ) {
        unimplemented!("not exercised by SESPH tests")
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
        unimplemented!("not exercised by SESPH tests")
    }

    fn get_checkpoint(&self) -> BoundaryCheckpoint {
        BoundaryCheckpoint::default()
    }

    fn restore_from_checkpoint(&mut self, _state: &BoundaryCheckpoint) {}
}

// ─── new / measurement_info ─────────────────────────────────────────────

#[test]
fn new_captures_the_configured_stiffness() {
    let solver = SESPH::new(&make_solver_params(500.0));
    let info = solver.measurement_info();
    assert_eq!(info.stiffness, 500.0);
}

#[test]
fn measurement_info_reports_default_for_every_other_field() {
    let solver = SESPH::new(&make_solver_params(123.0));
    let info = solver.measurement_info();
    assert_eq!(info.target_density_error, 0.0);
    assert_eq!(info.solver_iterations, 0);
    assert_eq!(info.relaxation_factor, 0.0);
    assert_eq!(info.predicted_density_error, 0.0);
}

// ─── State equation: pressure computation ───────────────────────────────

#[test]
fn pressure_is_zero_when_volume_is_at_or_above_rest_volume() {
    let h = 1.0;
    let params = make_system_params(h, 0.3, 0.0);
    let rv = rest_volume_for(0.3);
    let mut solver = SESPH::new(&make_solver_params(1000.0));

    let mut fluid = fluid_with_at_least(2);
    fluid.position[0] = Point3::new(0.0, 0.0, 0.0);
    fluid.position[1] = Point3::new(1000.0, 0.0, 0.0);
    fluid.volume[0] = rv;
    fluid.volume[1] = rv * 1.5;
    fluid.mass[0] = 0.5;
    fluid.mass[1] = 0.5;
    fluid.acceleration[0] = Vector3::zeros();
    fluid.acceleration[1] = Vector3::zeros();

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

    assert_eq!(fluid.pressure[0], 0.0);
    assert_eq!(fluid.pressure[1], 0.0);
}

#[test]
fn pressure_matches_the_state_equation_formula_when_compressed() {
    let h = 1.0;
    let params = make_system_params(h, 0.3, 0.0);
    let rv = rest_volume_for(0.3);
    let stiffness = 800.0;
    let mut solver = SESPH::new(&make_solver_params(stiffness));

    let mut fluid = fluid_with_at_least(1);
    fluid.position[0] = Point3::origin();
    let compressed_volume = rv * 0.8;
    fluid.volume[0] = compressed_volume;
    fluid.mass[0] = 0.5;
    fluid.acceleration[0] = Vector3::zeros();

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

    let expected = stiffness * (rv / compressed_volume - 1.0);
    assert!(
        (fluid.pressure[0] - expected).abs() < 1e-9,
        "expected {expected}, got {}",
        fluid.pressure[0]
    );
}

#[test]
fn pressure_scales_linearly_with_stiffness() {
    let h = 1.0;
    let params = make_system_params(h, 0.3, 0.0);
    let rv = rest_volume_for(0.3);

    let run = |stiffness: f64| {
        let mut solver = SESPH::new(&make_solver_params(stiffness));
        let mut fluid = fluid_with_at_least(1);
        fluid.position[0] = Point3::origin();
        fluid.volume[0] = rv * 0.5;
        fluid.mass[0] = 0.5;
        fluid.acceleration[0] = Vector3::zeros();
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
        fluid.pressure[0]
    };

    let p1 = run(100.0);
    let p2 = run(200.0);
    assert!((p2 - 2.0 * p1).abs() < 1e-9);
}

#[test]
fn pressure_is_computed_independently_per_particle() {
    let h = 1.0;
    let params = make_system_params(h, 0.3, 0.0);
    let rv = rest_volume_for(0.3);
    let stiffness = 500.0;
    let mut solver = SESPH::new(&make_solver_params(stiffness));

    let mut fluid = fluid_with_at_least(2);
    fluid.position[0] = Point3::new(0.0, 0.0, 0.0);
    fluid.position[1] = Point3::new(1000.0, 0.0, 0.0);
    fluid.volume[0] = rv * 0.9;
    fluid.volume[1] = rv * 0.5;
    fluid.mass[0] = 0.5;
    fluid.mass[1] = 0.5;
    fluid.acceleration[0] = Vector3::zeros();
    fluid.acceleration[1] = Vector3::zeros();

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

    let expected0 = stiffness * (rv / fluid.volume[0] - 1.0);
    let expected1 = stiffness * (rv / fluid.volume[1] - 1.0);
    assert!((fluid.pressure[0] - expected0).abs() < 1e-9);
    assert!((fluid.pressure[1] - expected1).abs() < 1e-9);
}

// ─── Pressure acceleration: fluid-fluid ────────────────────────────────

#[test]
fn solve_and_add_acceleration_matches_manual_formula_for_a_fluid_cluster() {
    let h = 1.0;
    let params = make_system_params(h, 0.3, 0.0);
    let rv = rest_volume_for(0.3);
    let stiffness = 500.0;
    let mut solver = SESPH::new(&make_solver_params(stiffness));

    let mut fluid = fluid_with_at_least(3);
    let positions = [
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.3, 0.0, 0.0),
        Point3::new(0.0, 0.3, 0.0),
    ];
    let volumes = [rv * 0.9, rv * 0.8, rv * 0.7];
    for i in 0..3 {
        fluid.position[i] = positions[i];
        fluid.volume[i] = volumes[i];
        fluid.mass[i] = 0.5;
        fluid.acceleration[i] = Vector3::zeros();
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

    let expected_pressures: Vec<f64> = volumes
        .iter()
        .map(|&v| stiffness * f64::max(rv / v - 1.0, 0.0))
        .collect();
    for (id, ep) in expected_pressures.iter().enumerate() {
        assert!((fluid.pressure[id] - ep).abs() < 1e-9, "id={id}");
    }

    for id in 0..3 {
        let mut expected_acc = Vector3::zeros();
        for &j in neighbor_list.get_neighbors(id) {
            let r_vec = vector(&positions[j], &positions[id]);
            expected_acc -= volumes[id] / fluid.mass[id]
                * volumes[j]
                * (expected_pressures[id] + expected_pressures[j])
                * CubicBSpline3D::kernel_gradient(&r_vec, h);
        }
        assert!(
            (fluid.acceleration[id] - expected_acc).norm() < 1e-9,
            "id={id}: expected {expected_acc:?}, got {:?}",
            fluid.acceleration[id]
        );
    }
}

#[test]
fn solve_and_add_acceleration_accumulates_onto_preexisting_acceleration() {
    let h = 1.0;
    let params = make_system_params(h, 0.3, 0.0);
    let rv = rest_volume_for(0.3);
    let mut solver = SESPH::new(&make_solver_params(500.0));

    let mut fluid = fluid_with_at_least(1);
    fluid.position[0] = Point3::origin();
    fluid.volume[0] = rv; // no compression -> zero pressure contribution
    fluid.mass[0] = 0.5;
    let preexisting = Vector3::new(0.0, 0.0, -9.81);
    fluid.acceleration[0] = preexisting;

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

    assert_eq!(fluid.acceleration[0], preexisting);
}

// ─── Pressure acceleration: boundary contribution ──────────────────────

#[test]
fn solve_and_add_acceleration_includes_boundary_contribution_without_reaction_force_for_static_boundaries()
 {
    let h = 1.0;
    let weighting = 0.5;
    let params = make_system_params(h, 0.3, weighting);
    let rv = rest_volume_for(0.3);
    let stiffness = 500.0;
    let mut solver = SESPH::new(&make_solver_params(stiffness));

    let mut fluid = fluid_with_at_least(1);
    fluid.position[0] = Point3::origin();
    fluid.volume[0] = rv * 0.5;
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

    let r_vec = vector(&boundary_pos, &fluid.position[0]);
    let force = 2.0
        * weighting
        * fluid.volume[0]
        * boundary_vol
        * fluid.pressure[0]
        * CubicBSpline3D::kernel_gradient(&r_vec, h);
    let expected = -force / fluid.mass[0];

    assert!((fluid.acceleration[0] - expected).norm() < 1e-9);
    assert!(boundary.recorded_forces.is_empty());
}

#[test]
fn solve_and_add_acceleration_registers_newtons_third_law_reaction_force_on_dynamic_boundaries() {
    let h = 1.0;
    let weighting = 1.0;
    let params = make_system_params(h, 0.3, weighting);
    let rv = rest_volume_for(0.3);
    let mut solver = SESPH::new(&make_solver_params(500.0));

    let mut fluid = fluid_with_at_least(1);
    fluid.position[0] = Point3::origin();
    fluid.volume[0] = rv * 0.5;
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
fn sesph_implements_pressure_solver_and_is_cloneable() {
    assert_implements_pressure_solver::<SESPH>();
    let solver = SESPH::new(&make_solver_params(100.0));
    let cloned = solver.clone();
    assert_eq!(cloned.measurement_info().stiffness, 100.0);
}
