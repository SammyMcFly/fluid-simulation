//! Integration tests for `sph`'s public API (`SPHSystem`, `System`,
//! `SystemCheckpoint`, `SerSystemCheckpoint`, `Outer`), built end-to-end via
//! `setup::new_boxed_system3d` — exactly as a real caller would. Private
//! items (`SystemParameters`, `CurrentSystemProperties`, `SystemCheckpoint`'s
//! fields) are covered separately in `sph`'s own internal test module.
mod boundary_handling;
mod fluid;
mod integration_schemes;
mod kernel;
mod pressure_solver;
mod setup;

use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use nalgebra::Vector3;

use simulation_lib::neighbor_search::NeighborSearchVariant;
use simulation_lib::render_info::{
    BoundaryMeshColoring, BoundarySampleColoring, BoundaryVisualization, ScalarQuantity,
};
use simulation_lib::sph::boundary_handling::BoundaryHandlingVariant;
use simulation_lib::sph::integration_schemes::IntegrationSchemeVariant;
use simulation_lib::sph::kernel::KernelFnVariant;
use simulation_lib::sph::pressure_solver::PressureSolverVariant;
use simulation_lib::sph::setup::input::{
    Fluid as FluidPhase, FluidDef, Light, Parameters, Procedures, Scene,
};
use simulation_lib::sph::setup::new_boxed_system3d;
use simulation_lib::sph::{Outer, SPHSystem, SystemCheckpoint};

// ─── Fixtures / helpers (mirrors `tests/setup.rs`) ──────────────────────

static TEMP_DIR_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn unique_temp_dir() -> std::path::PathBuf {
    let n = TEMP_DIR_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("sph_mod_test_{}_{n}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("failed to create temp dir");
    dir
}

fn write_cube_obj(dir: &std::path::Path, name: &str) -> std::path::PathBuf {
    let obj = "\
v 1.0 1.0 1.0
v 1.0 1.0 -1.0
v 1.0 -1.0 1.0
v 1.0 -1.0 -1.0
v -1.0 1.0 1.0
v -1.0 1.0 -1.0
v -1.0 -1.0 1.0
v -1.0 -1.0 -1.0
f 5 3 1
f 3 8 4
f 7 6 8
f 2 8 6
f 1 4 2
f 5 2 6
f 5 7 3
f 3 7 8
f 7 5 6
f 2 4 8
f 1 3 4
f 5 1 2
";
    let path = dir.join(name);
    std::fs::write(&path, obj).expect("failed to write temp .obj file");
    path
}

fn make_parameters(fluids: Vec<FluidPhase>) -> Parameters {
    Parameters {
        buffer_length_limit: 100,
        #[cfg(not(feature = "cfl_time_step"))]
        time_increment: 0.001,
        #[cfg(feature = "cfl_time_step")]
        max_time_increment: 0.001,
        #[cfg(feature = "cfl_time_step")]
        cfl_number: 0.4,
        fluid: fluids,
        rest_density_grid_spacing: 0.5,
        kernel_support_radius: 1.0,
        disable_particles_below: -1e9,
        fluid_viscosity: 0.01,
        boundary_viscosity: 0.01,
        boundary_pressure_acceleration_weighting: 1.0,
        boundary_rest_volume_weighting: 1.0,
        stiffness: 500.0,
        target_density_error: 0.01,
        relaxation_factor: 0.5,
        min_diagonal_element: 1e-9,
    }
}

fn make_scene(meshes: HashMap<String, String>, fluid_defs: Vec<FluidDef>) -> Scene {
    Scene {
        light: Light {
            position: [5.0, 8.0, 5.0],
        },
        meshes,
        fluid: fluid_defs,
        boundary: Default::default(),
    }
}

fn cube_fluid_def(mesh_key: &str, fluid_id: u32) -> FluidDef {
    FluidDef {
        mesh: mesh_key.to_string(),
        fluid_id,
        translation: [0.0, 0.0, 0.0],
        rotation_euler_deg: [0.0, 0.0, 0.0],
        scale: [1.0, 1.0, 1.0],
    }
}

fn minimal_fluid_scene(dir: &std::path::Path) -> (Scene, Parameters) {
    let cube_path = write_cube_obj(dir, "cube.obj");
    let mut meshes = HashMap::new();
    meshes.insert("cube".to_string(), cube_path.to_str().unwrap().to_string());
    let scene = make_scene(meshes, vec![cube_fluid_def("cube", 0)]);
    let params = make_parameters(vec![FluidPhase {
        id: 0,
        rest_density: 1000.0,
    }]);
    (scene, params)
}

fn default_procedures() -> Procedures {
    Procedures {
        kernel_function: KernelFnVariant::CubicBSpline3D,
        integration_scheme: IntegrationSchemeVariant::EulerCromer,
        pressure_solver: PressureSolverVariant::SESPH,
        neighbor_search: NeighborSearchVariant::SpatialHashing,
        boundary_handling: BoundaryHandlingVariant::VolumeMapBoundary,
    }
}

fn build_system(scene: &Scene, params: &Parameters) -> Box<dyn SPHSystem> {
    new_boxed_system3d(&default_procedures(), params, scene, None)
        .expect("expected system construction to succeed")
}

// ─── time() / time_steps_propagated() ───────────────────────────────────

#[test]
fn fresh_system_starts_at_time_zero_with_zero_steps_propagated() {
    let dir = unique_temp_dir();
    let (scene, params) = minimal_fluid_scene(&dir);
    let system = build_system(&scene, &params);

    assert_eq!(system.time_steps_propagated(), 0);
    assert!(system.time().abs() < 1e-12);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn step_forward_in_time_increments_the_step_counter() {
    let dir = unique_temp_dir();
    let (scene, params) = minimal_fluid_scene(&dir);
    let mut system = build_system(&scene, &params);

    system.step_forward_in_time();
    assert_eq!(system.time_steps_propagated(), 1);
    system.step_forward_in_time();
    assert_eq!(system.time_steps_propagated(), 2);

    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(not(feature = "cfl_time_step"))]
#[test]
fn step_forward_in_time_advances_time_by_exactly_the_configured_increment() {
    let dir = unique_temp_dir();
    let (scene, params) = minimal_fluid_scene(&dir);
    let mut system = build_system(&scene, &params);

    system.step_forward_in_time();
    system.step_forward_in_time();
    system.step_forward_in_time();

    let expected = 3.0 * params.time_increment;
    assert!(
        (system.time() - expected).abs() < 1e-9,
        "expected {expected}, got {}",
        system.time()
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(feature = "cfl_time_step")]
#[test]
fn step_forward_in_time_never_decreases_time() {
    // Under adaptive time stepping the exact increment isn't predictable
    // from outside, so only monotonicity is checked here.
    let dir = unique_temp_dir();
    let (scene, params) = minimal_fluid_scene(&dir);
    let mut system = build_system(&scene, &params);

    let mut previous = system.time();
    for _ in 0..3 {
        system.step_forward_in_time();
        assert!(system.time() >= previous - 1e-12);
        previous = system.time();
    }

    let _ = std::fs::remove_dir_all(&dir);
}

// ─── take_measurement ────────────────────────────────────────────────────

#[test]
fn take_measurement_reflects_the_configured_parameters() {
    let dir = unique_temp_dir();
    let (scene, params) = minimal_fluid_scene(&dir);
    let system = build_system(&scene, &params);

    let measurement = system.take_measurement();
    assert_eq!(measurement.fluid_viscosity, params.fluid_viscosity);
    assert_eq!(measurement.boundary_viscosity, params.boundary_viscosity);
    assert_eq!(
        measurement.rest_density_grid_spacing,
        params.rest_density_grid_spacing
    );
    assert_eq!(
        measurement.kernel_support_radius,
        params.kernel_support_radius
    );
    assert_eq!(measurement.stiffness, params.stiffness); // SESPH passes stiffness through verbatim

    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(not(feature = "cfl_time_step"))]
#[test]
fn take_measurement_time_step_size_matches_the_fixed_time_increment() {
    let dir = unique_temp_dir();
    let (scene, params) = minimal_fluid_scene(&dir);
    let system = build_system(&scene, &params);

    assert_eq!(
        system.take_measurement().time_step_size,
        params.time_increment
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ─── get_fluid_ids / get_fluid_pos ───────────────────────────────────────

#[test]
fn get_fluid_ids_and_get_fluid_pos_have_matching_nonzero_length() {
    let dir = unique_temp_dir();
    let (scene, params) = minimal_fluid_scene(&dir);
    let system = build_system(&scene, &params);

    let ids = system.get_fluid_ids();
    let positions = system.get_fluid_pos();
    assert!(!ids.is_empty());
    assert_eq!(ids.len(), positions.len());
    assert!(ids.iter().all(|&id| id == 0));

    let _ = std::fs::remove_dir_all(&dir);
}

// ─── get_fluid_checkpoint / continue_from_checkpoint round trip ────────

#[test]
fn continue_from_checkpoint_restores_step_count_and_exact_fluid_state() {
    let dir = unique_temp_dir();
    let (scene, params) = minimal_fluid_scene(&dir);

    let mut system_a = build_system(&scene, &params);
    system_a.step_forward_in_time();
    system_a.step_forward_in_time();

    let checkpoint = SystemCheckpoint::from_sph_system(system_a.as_ref());
    let checkpoint_rc = Rc::new(checkpoint);

    let mut system_b = build_system(&scene, &params);
    system_b.continue_from_checkpoint(checkpoint_rc.clone());

    assert_eq!(
        system_b.time_steps_propagated(),
        checkpoint_rc.get_time_steps_propagated()
    );

    let expected_fluid = checkpoint_rc.get_fluid();
    let actual_fluid = system_b.get_fluid_checkpoint();
    assert_eq!(actual_fluid.fluid_id, expected_fluid.fluid_id);
    assert_eq!(actual_fluid.position, expected_fluid.position);
    assert_eq!(actual_fluid.velocity, expected_fluid.velocity);
    assert_eq!(actual_fluid.mass, expected_fluid.mass);

    // Regression test for a bug where `update()` (called at the end of
    // `continue_from_checkpoint`) unconditionally advanced `current_time`
    // by one further time step under `cfl_time_step`, instead of leaving
    // it at exactly the checkpointed value. Now fixed: the time-advance is
    // only performed in `step_forward_in_time`, never in `update()` itself
    // — so restored time must match the checkpoint exactly under BOTH
    // build configurations, not just without `cfl_time_step`.
    assert!(
        (system_b.time() - checkpoint_rc.get_current_time()).abs() < 1e-9,
        "expected restored time to exactly match the checkpoint's current_time"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ─── get_quantity_of_fluid_samples ──────────────────────────────────────

#[test]
fn get_quantity_of_fluid_samples_returns_one_value_per_particle_for_every_variant() {
    let dir = unique_temp_dir();
    let (scene, params) = minimal_fluid_scene(&dir);
    let system = build_system(&scene, &params);
    let n = system.get_fluid_ids().len();

    for quantity in [
        ScalarQuantity::Speed,
        ScalarQuantity::Volume,
        ScalarQuantity::Density,
        ScalarQuantity::DensityError,
        ScalarQuantity::Pressure,
        ScalarQuantity::KineticEnergy,
    ] {
        let values = system.get_quantity_of_fluid_samples(&quantity);
        assert_eq!(values.len(), n, "length mismatch for {quantity:?}");
        assert!(
            values.iter().all(|v| v.is_finite()),
            "non-finite value for {quantity:?}: {values:?}"
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_freshly_sampled_fluid_starts_at_rest_with_zero_speed_and_kinetic_energy() {
    // `Fluid::add_samples` initializes every particle's velocity to zero,
    // and no time step has run yet -> both Speed and KineticEnergy (which
    // both derive directly from velocity) must be exactly zero.
    let dir = unique_temp_dir();
    let (scene, params) = minimal_fluid_scene(&dir);
    let system = build_system(&scene, &params);

    let speed = system.get_quantity_of_fluid_samples(&ScalarQuantity::Speed);
    let kinetic_energy = system.get_quantity_of_fluid_samples(&ScalarQuantity::KineticEnergy);
    assert!(speed.iter().all(|&v| v == 0.0));
    assert!(kinetic_energy.iter().all(|&v| v == 0.0));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn volume_of_a_freshly_constructed_system_is_positive() {
    // `System::new_boxed` calls `update()` once during construction, which
    // computes `fluid.volume` before the system is ever returned — so even
    // a freshly built, never-stepped system should already report a
    // positive volume for every particle.
    let dir = unique_temp_dir();
    let (scene, params) = minimal_fluid_scene(&dir);
    let system = build_system(&scene, &params);

    let volume = system.get_quantity_of_fluid_samples(&ScalarQuantity::Volume);
    assert!(volume.iter().all(|&v| v > 0.0));

    let _ = std::fs::remove_dir_all(&dir);
}

// ─── get_quantity_at_positions ───────────────────────────────────────────

#[test]
fn get_quantity_at_positions_returns_finite_values_of_matching_length() {
    let dir = unique_temp_dir();
    let (scene, params) = minimal_fluid_scene(&dir);
    let mut system = build_system(&scene, &params);

    let query_positions = system.get_fluid_pos();
    assert!(!query_positions.is_empty());
    let subset = &query_positions[0..1];

    let volume = system.get_quantity_at_positions(&ScalarQuantity::Volume, subset);
    assert_eq!(volume.len(), 1);
    assert!(volume[0].is_finite() && volume[0] > 0.0);

    let speed = system.get_quantity_at_positions(&ScalarQuantity::Speed, subset);
    assert_eq!(speed.len(), 1);
    assert!(speed[0].is_finite());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn get_quantity_at_positions_on_an_empty_position_slice_returns_empty() {
    let dir = unique_temp_dir();
    let (scene, params) = minimal_fluid_scene(&dir);
    let mut system = build_system(&scene, &params);

    let result = system.get_quantity_at_positions(&ScalarQuantity::Volume, &[]);
    assert!(result.is_empty());

    let _ = std::fs::remove_dir_all(&dir);
}

// ─── get_fluid_surface (heavy: real surface reconstruction) ────────────

#[test]
#[ignore = "exercises the real splashsurf surface reconstruction pipeline \
            end-to-end; actual timing hasn't been verified by running this \
            code — run explicitly via `cargo test -- --ignored`."]
fn get_fluid_surface_returns_a_mesh_for_the_present_fluid_id() {
    let dir = unique_temp_dir();
    let (scene, params) = minimal_fluid_scene(&dir);
    let system = build_system(&scene, &params);

    let surfaces = system.get_fluid_surface();
    assert!(
        surfaces.iter().any(|(id, _)| *id == 0),
        "expected a reconstructed surface for fluid id 0"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ─── get_boundary_visualization / get_boundary_checkpoint (no boundary) ─

#[test]
fn get_boundary_visualization_with_no_boundary_geometry_returns_empty_samples() {
    let dir = unique_temp_dir();
    let (scene, params) = minimal_fluid_scene(&dir);
    let system = build_system(&scene, &params);

    let selector = BoundaryVisualization::Samples {
        positions: vec![],
        coloring: BoundarySampleColoring::Uniform,
    };
    match system.get_boundary_visualization(&selector) {
        BoundaryVisualization::Samples { positions, .. } => assert!(positions.is_empty()),
        _ => panic!("expected a Samples result"),
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn get_boundary_visualization_with_no_boundary_geometry_returns_no_meshes() {
    let dir = unique_temp_dir();
    let (scene, params) = minimal_fluid_scene(&dir);
    let system = build_system(&scene, &params);

    let selector = BoundaryVisualization::TriangleMesh {
        meshes: vec![],
        coloring: BoundaryMeshColoring::Original,
    };
    match system.get_boundary_visualization(&selector) {
        BoundaryVisualization::TriangleMesh { meshes, .. } => assert!(meshes.is_empty()),
        _ => panic!("expected a TriangleMesh result"),
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn get_boundary_checkpoint_with_no_boundaries_has_no_dynamic_states() {
    let dir = unique_temp_dir();
    let (scene, params) = minimal_fluid_scene(&dir);
    let system = build_system(&scene, &params);

    assert!(system.get_boundary_checkpoint().dynamic_states.is_empty());

    let _ = std::fs::remove_dir_all(&dir);
}

// ─── SerSystemCheckpoint round trip with a real system ──────────────────

#[test]
fn ser_system_checkpoint_round_trips_a_real_systems_checkpoint_via_ron() {
    let dir = unique_temp_dir();
    let (scene, params) = minimal_fluid_scene(&dir);
    let mut system = build_system(&scene, &params);
    system.step_forward_in_time();

    let checkpoint = SystemCheckpoint::from_sph_system(system.as_ref());
    let ser: simulation_lib::sph::SerSystemCheckpoint = checkpoint.into();

    let text = ron::to_string(&ser).expect("failed to serialize checkpoint to RON");
    let deserialized: simulation_lib::sph::SerSystemCheckpoint =
        ron::from_str(&text).expect("failed to deserialize checkpoint from RON");

    assert_eq!(
        deserialized.time_steps_propagated,
        ser.time_steps_propagated
    );
    assert_eq!(deserialized.fluid.position, ser.fluid.position);

    let _ = std::fs::remove_dir_all(&dir);
}

// ─── Box<dyn SPHSystem> is cloneable (dyn_clone) and clones are independent ─

#[test]
fn boxed_system_is_cloneable_and_clones_evolve_independently() {
    let dir = unique_temp_dir();
    let (scene, params) = minimal_fluid_scene(&dir);
    let mut system = build_system(&scene, &params);

    let cloned = system.clone();
    assert_eq!(
        cloned.time_steps_propagated(),
        system.time_steps_propagated()
    );

    system.step_forward_in_time();
    assert_eq!(system.time_steps_propagated(), 1);
    assert_eq!(
        cloned.time_steps_propagated(),
        0,
        "the clone must not be affected by stepping the original forward"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ─── Outer trait ─────────────────────────────────────────────────────────

#[test]
fn outer_product_matches_the_manual_matrix_formula() {
    let a = Vector3::new(1.0, 2.0, 3.0);
    let b = Vector3::new(4.0, 5.0, 6.0);
    let m = a.outer(&b);

    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(m[(i, j)], a[i] * b[j]);
        }
    }
}

#[test]
fn outer_product_with_a_zero_vector_is_the_zero_matrix() {
    let a = Vector3::new(1.0, 2.0, 3.0);
    let zero = Vector3::new(0.0, 0.0, 0.0);
    let m = a.outer(&zero);
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(m[(i, j)], 0.0);
        }
    }
}

#[test]
fn outer_product_is_not_generally_symmetric() {
    let a = Vector3::new(1.0, 0.0, 0.0);
    let b = Vector3::new(0.0, 1.0, 0.0);
    let m = a.outer(&b);
    assert_eq!(m[(0, 1)], 1.0);
    assert_eq!(m[(1, 0)], 0.0);
}
