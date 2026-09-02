//! Integration tests for `render_info`'s public API. The private
//! `from_system` dispatch methods on `FluidVisualization`,
//! `FluidSampleColoring` and `BoundaryVisualization` are covered
//! separately in the module's own internal test block; here, only one
//! end-to-end test exercises the public `TimeStepInfo::from_system` entry
//! point.

use std::rc::Rc;

use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};

use simulation_lib::measurement::Measurement;
use simulation_lib::neighbor_search::NeighborSearchVariant;
use simulation_lib::render_info::{
    BoundarySampleColoring, BoundaryVisualization, FluidSampleColoring, FluidVisualization,
    RenderPose, ScalarQuantity, SimulationParameters, TimeStepInfo,
};
use simulation_lib::sph::boundary_handling::{BoundaryCheckpoint, BoundaryHandlingVariant};
use simulation_lib::sph::fluid::FluidCheckpoint;
use simulation_lib::sph::integration_schemes::IntegrationSchemeVariant;
use simulation_lib::sph::kernel::KernelFnVariant;
use simulation_lib::sph::pressure_solver::PressureSolverVariant;
use simulation_lib::sph::setup::input::{Parameters, Procedures};
use simulation_lib::sph::{SPHSystem, SystemCheckpoint};
use simulation_lib::utilities::triangle_mesh::RenderMesh;

// ─── Fixtures / helpers ─────────────────────────────────────────────────

fn make_parameters(rest_density_grid_spacing: f64, buffer_length_limit: usize) -> Parameters {
    Parameters {
        buffer_length_limit,
        #[cfg(not(feature = "cfl_time_step"))]
        time_increment: 0.001,
        #[cfg(feature = "cfl_time_step")]
        max_time_increment: 0.001,
        #[cfg(feature = "cfl_time_step")]
        cfl_number: 0.4,
        fluid: vec![],
        rest_density_grid_spacing,
        kernel_support_radius: 1.0,
        disable_particles_below: -1e9,
        fluid_viscosity: 0.0,
        boundary_viscosity: 0.0,
        boundary_pressure_acceleration_weighting: 0.0,
        boundary_rest_volume_weighting: 0.0,
        stiffness: 0.0,
        target_density_error: 0.0,
        relaxation_factor: 0.0,
        min_diagonal_element: 0.0,
    }
}

fn make_procedures(boundary_handling: BoundaryHandlingVariant) -> Procedures {
    Procedures {
        kernel_function: KernelFnVariant::CubicBSpline3D,
        integration_scheme: IntegrationSchemeVariant::EulerCromer,
        pressure_solver: PressureSolverVariant::SESPH,
        neighbor_search: NeighborSearchVariant::SpatialHashing,
        boundary_handling,
    }
}

fn sample_measurement() -> Measurement {
    Measurement {
        time: 1.0,
        density: 998.0,
        density_error: 0.1,
        kinetic_energy: 2.0,
        stiffness: 500.0,
        fluid_viscosity: 0.01,
        boundary_viscosity: 0.01,
        fluid_depth: 3.0,
        rest_density_grid_spacing: 0.05,
        kernel_support_radius: 0.1,
        time_step_size: 0.001,
        target_density_error: 0.01,
        solver_iterations: 5,
        relaxation_factor: 0.5,
        time_step_wall_clock_time: 0.002,
        predicted_density_error: 0.02,
    }
}

// ─── SimulationParameters::new ───────────────────────────────────────────

#[test]
fn simulation_parameters_new_casts_and_copies_fields_correctly() {
    let params = make_parameters(0.025, 42);
    let procedures = make_procedures(BoundaryHandlingVariant::VolumeMapBoundary);

    let sim_params = SimulationParameters::new(&procedures, &params, [1.0, 2.0, 3.0], true, false);

    assert!((sim_params.particle_diameter - 0.025f32).abs() < 1e-9);
    assert_eq!(sim_params.buffer_length_limit, 42);
    assert_eq!(sim_params.light_position, [1.0, 2.0, 3.0]);
    assert!(sim_params.is_measured);
    assert!(!sim_params.is_recorded);
}

#[test]
fn simulation_parameters_new_flags_static_sample_boundary_as_explicit() {
    let params = make_parameters(0.05, 10);
    let procedures = make_procedures(BoundaryHandlingVariant::StaticSampleBoundary);

    let sim_params = SimulationParameters::new(&procedures, &params, [0.0, 0.0, 0.0], false, false);

    assert!(sim_params.explicitly_sampled_boundary);
}

#[test]
fn simulation_parameters_new_flags_volume_map_boundary_as_implicit() {
    let params = make_parameters(0.05, 10);
    let procedures = make_procedures(BoundaryHandlingVariant::VolumeMapBoundary);

    let sim_params = SimulationParameters::new(&procedures, &params, [0.0, 0.0, 0.0], false, false);

    assert!(!sim_params.explicitly_sampled_boundary);
}

// ─── SimulationParameters: bincode round trip ───────────────────────────

#[test]
fn simulation_parameters_round_trips_through_bincode_bytes() {
    let params = make_parameters(0.03, 7);
    let procedures = make_procedures(BoundaryHandlingVariant::StaticSampleBoundary);
    let sim_params = SimulationParameters::new(&procedures, &params, [1.0, 1.0, 1.0], true, true);

    let bytes: Vec<u8> = sim_params.clone().into();
    let restored =
        SimulationParameters::try_from(bytes.as_slice()).expect("expected round trip to succeed");

    assert_eq!(restored.particle_diameter, sim_params.particle_diameter);
    assert_eq!(restored.buffer_length_limit, sim_params.buffer_length_limit);
    assert_eq!(restored.light_position, sim_params.light_position);
    assert_eq!(restored.is_measured, sim_params.is_measured);
    assert_eq!(restored.is_recorded, sim_params.is_recorded);
    assert_eq!(
        restored.explicitly_sampled_boundary,
        sim_params.explicitly_sampled_boundary
    );
}

#[test]
fn simulation_parameters_try_from_rejects_garbage_bytes() {
    let garbage = [0u8, 1, 2, 3];
    assert!(SimulationParameters::try_from(&garbage[..]).is_err());
}

// ─── TimeStepInfo: bincode round trip ────────────────────────────────────

#[test]
fn time_step_info_round_trips_through_bincode_bytes() {
    let info = TimeStepInfo {
        time_step_number: 3,
        measurement: sample_measurement(),
        fluid: FluidVisualization::Samples {
            positions: vec![[1.0, 2.0, 3.0]],
            coloring: FluidSampleColoring::Uniform,
        },
        boundary: BoundaryVisualization::Samples {
            positions: vec![[4.0, 5.0, 6.0]],
            coloring: BoundarySampleColoring::Uniform,
        },
    };

    let bytes: Vec<u8> = info.clone().into();
    let restored =
        TimeStepInfo::try_from(bytes.as_slice()).expect("expected round trip to succeed");

    assert_eq!(restored.time_step_number, info.time_step_number);
    assert_eq!(restored.measurement.time, info.measurement.time);
    assert_eq!(restored.measurement.density, info.measurement.density);
    assert_eq!(restored.fluid, info.fluid);
    assert_eq!(restored.boundary, info.boundary);
}

#[test]
fn time_step_info_try_from_rejects_garbage_bytes() {
    let garbage = [9u8; 5];
    assert!(TimeStepInfo::try_from(&garbage[..]).is_err());
}

// ─── RenderPose ───────────────────────────────────────────────────────────

#[test]
fn render_pose_identity_constant_is_a_true_identity() {
    assert_eq!(RenderPose::IDENTITY.translation, [0.0, 0.0, 0.0]);
    assert_eq!(RenderPose::IDENTITY.rotation, [0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn render_pose_from_identity_isometry_equals_render_pose_identity() {
    // Ties together two independently-defined "identity" representations —
    // exactly the kind of check that would catch e.g. an accidentally wrong
    // quaternion component order in `From<Isometry3<f64>>`.
    let pose = RenderPose::from(Isometry3::<f64>::identity());
    assert_eq!(pose, RenderPose::IDENTITY);
}

#[test]
fn render_pose_from_isometry_extracts_translation_and_quaternion_components() {
    let translation = Translation3::new(1.0, 2.0, 3.0);
    let rotation = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), std::f64::consts::FRAC_PI_2);
    let isometry = Isometry3::from_parts(translation, rotation);

    let pose = RenderPose::from(isometry);

    assert!((pose.translation[0] - 1.0f32).abs() < 1e-6);
    assert!((pose.translation[1] - 2.0f32).abs() < 1e-6);
    assert!((pose.translation[2] - 3.0f32).abs() < 1e-6);

    let q = rotation.into_inner();
    assert!((pose.rotation[0] - q.i as f32).abs() < 1e-6);
    assert!((pose.rotation[1] - q.j as f32).abs() < 1e-6);
    assert!((pose.rotation[2] - q.k as f32).abs() < 1e-6);
    assert!((pose.rotation[3] - q.w as f32).abs() < 1e-6);
}

// ─── TimeStepInfo::from_system: end-to-end via the public API ──────────

#[derive(Clone)]
struct MockSystem {
    fluid_ids: Vec<u32>,
    fluid_pos: Vec<[f32; 3]>,
    time_steps_propagated: u64,
    measurement: Measurement,
    boundary_visualization_result: BoundaryVisualization,
}

impl SPHSystem for MockSystem {
    fn time(&self) -> f64 {
        unimplemented!("not exercised by this test")
    }
    fn time_steps_propagated(&self) -> u64 {
        self.time_steps_propagated
    }
    fn step_forward_in_time(&mut self) {
        unimplemented!("not exercised by this test")
    }
    fn take_measurement(&self) -> Measurement {
        self.measurement.clone()
    }
    fn get_fluid_ids(&self) -> Vec<u32> {
        self.fluid_ids.clone()
    }
    fn get_fluid_pos(&self) -> Vec<[f32; 3]> {
        self.fluid_pos.clone()
    }
    fn get_fluid_checkpoint(&self) -> FluidCheckpoint {
        unimplemented!("not exercised by this test")
    }
    fn get_quantity_of_fluid_samples(&self, _quantity: &ScalarQuantity) -> Vec<f32> {
        vec![]
    }
    fn get_quantity_at_positions(
        &mut self,
        _quantity: &ScalarQuantity,
        _positions: &[[f32; 3]],
    ) -> Vec<f32> {
        vec![]
    }
    fn get_fluid_surface(&self) -> Vec<(u32, RenderMesh)> {
        vec![]
    }
    fn get_boundary_visualization(
        &self,
        _selector: &BoundaryVisualization,
    ) -> BoundaryVisualization {
        self.boundary_visualization_result.clone()
    }
    fn get_boundary_checkpoint(&self) -> BoundaryCheckpoint {
        unimplemented!("not exercised by this test")
    }
    fn continue_from_checkpoint(&mut self, _checkpoint: Rc<SystemCheckpoint>) {
        unimplemented!("not exercised by this test")
    }
}

#[test]
fn time_step_info_from_system_wires_together_public_accessors() {
    let mut mock = MockSystem {
        fluid_ids: vec![0, 0],
        fluid_pos: vec![[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        time_steps_propagated: 7,
        measurement: sample_measurement(),
        boundary_visualization_result: BoundaryVisualization::Samples {
            positions: vec![[9.0, 9.0, 9.0]],
            coloring: BoundarySampleColoring::Uniform,
        },
    };

    let selector = TimeStepInfo {
        time_step_number: 0,
        measurement: Measurement::default(),
        fluid: FluidVisualization::Samples {
            positions: vec![],
            coloring: FluidSampleColoring::Uniform,
        },
        boundary: BoundaryVisualization::Samples {
            positions: vec![],
            coloring: BoundarySampleColoring::Uniform,
        },
    };

    let info = TimeStepInfo::from_system(&mut mock, &selector);

    assert_eq!(info.time_step_number, 7);
    assert_eq!(info.measurement.time, mock.measurement.time);
    match info.fluid {
        FluidVisualization::Samples { positions, .. } => {
            assert_eq!(positions, vec![[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        }
        _ => panic!("expected Samples variant"),
    }
    match info.boundary {
        BoundaryVisualization::Samples { positions, .. } => {
            assert_eq!(positions, vec![[9.0, 9.0, 9.0]]);
        }
        _ => panic!("expected Samples variant"),
    }
}
