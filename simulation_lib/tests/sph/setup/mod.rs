//! Integration tests for `sph::setup`, exercising only its public API.
//!
//! `SetupError`'s exact variant names are unknown (its defining file,
//! `error.rs`, wasn't available), so failure-path tests only check
//! `.is_err()` plus the `Display` message rather than matching on a
//! specific variant.
mod input;

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use simulation_lib::neighbor_search::{NeighborSearchVariant, SpatialHashing};
use simulation_lib::sph::SerSystemCheckpoint;
use simulation_lib::sph::boundary_handling::{
    BoundaryHandling, BoundaryHandlingVariant, SerBoundaryCheckpoint, VolumeMapBoundary,
};
use simulation_lib::sph::fluid::{Len, SerFluidCheckpoint};
use simulation_lib::sph::integration_schemes::{
    EulerCromer, IntegrationSchemeVariant, TakePredicted,
};
use simulation_lib::sph::kernel::{CubicBSpline3D, KernelFnVariant};
use simulation_lib::sph::pressure_solver::{PressureSolverVariant, SESPH};
use simulation_lib::sph::setup::input::{
    BoundaryDefs, Fluid as FluidPhase, FluidDef, Light, Parameters, Procedures, Scene,
};
use simulation_lib::sph::setup::{SetupError, SystemConstructor, new_boxed_system3d};

// ─── Fixtures / helpers ─────────────────────────────────────────────────

static TEMP_DIR_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn unique_temp_dir() -> std::path::PathBuf {
    let n = TEMP_DIR_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("setup_test_{}_{n}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("failed to create temp dir");
    dir
}

/// Unit cube (side length 2, centered at the origin), same geometry/winding
/// used throughout this crate's other test fixtures.
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
        boundary: BoundaryDefs::default(),
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

/// Builds a minimal single-cube-fluid scene + matching parameters, ready
/// to pass to `SystemConstructor::new` / `new_boxed_system3d`.
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

// ─── SystemConstructor::new: success paths ─────────────────────────────

#[test]
fn system_constructor_builds_from_a_simple_fluid_only_scene() {
    let dir = unique_temp_dir();
    let (scene, params) = minimal_fluid_scene(&dir);

    let constructor = SystemConstructor::<
        CubicBSpline3D,
        EulerCromer,
        SESPH,
        SpatialHashing,
        VolumeMapBoundary,
    >::new(&params, &scene, None)
    .expect("expected system construction to succeed");

    assert!(!constructor.fluid.is_empty());
    assert!(constructor.boundary.is_empty());
    assert_eq!(constructor.initial_time_steps_propagated, 0);
    assert_eq!(constructor.initial_current_time, 0.0);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn system_constructor_with_an_empty_scene_still_succeeds() {
    let dir = unique_temp_dir();
    let scene = make_scene(HashMap::new(), vec![]);
    let params = make_parameters(vec![]);

    let constructor = SystemConstructor::<
        CubicBSpline3D,
        EulerCromer,
        SESPH,
        SpatialHashing,
        VolumeMapBoundary,
    >::new(&params, &scene, None)
    .expect("expected an empty scene to still succeed (just with warnings)");

    assert!(constructor.fluid.is_empty());
    assert!(constructor.boundary.is_empty());

    let _ = std::fs::remove_dir_all(&dir);
}

// ─── SystemConstructor::new: failure paths ─────────────────────────────

#[test]
fn system_constructor_fails_for_an_unknown_mesh_reference() {
    let dir = unique_temp_dir();
    let scene = make_scene(HashMap::new(), vec![cube_fluid_def("cube", 0)]);
    let params = make_parameters(vec![FluidPhase {
        id: 0,
        rest_density: 1000.0,
    }]);

    let result = SystemConstructor::<
        CubicBSpline3D,
        EulerCromer,
        SESPH,
        SpatialHashing,
        VolumeMapBoundary,
    >::new(&params, &scene, None);

    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("expected an unknown mesh reference to fail"),
    };
    let message = format!("{err}");
    assert!(
        message.contains("cube"),
        "expected the error to mention the offending mesh name 'cube': {message}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn system_constructor_fails_for_an_undefined_fluid_id() {
    let dir = unique_temp_dir();
    let cube_path = write_cube_obj(&dir, "cube.obj");
    let mut meshes = HashMap::new();
    meshes.insert("cube".to_string(), cube_path.to_str().unwrap().to_string());
    let scene = make_scene(meshes, vec![cube_fluid_def("cube", 7)]);
    let params = make_parameters(vec![]);

    let result = SystemConstructor::<
        CubicBSpline3D,
        EulerCromer,
        SESPH,
        SpatialHashing,
        VolumeMapBoundary,
    >::new(&params, &scene, None);

    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("expected an undefined fluid id to fail"),
    };
    let message = format!("{err}");
    assert!(
        message.contains('7'),
        "expected the error to mention the offending fluid id '7': {message}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn system_constructor_propagates_an_error_for_a_missing_state_file() {
    let dir = unique_temp_dir();
    let (scene, params) = minimal_fluid_scene(&dir);

    let result = SystemConstructor::<
        CubicBSpline3D,
        EulerCromer,
        SESPH,
        SpatialHashing,
        VolumeMapBoundary,
    >::new(
        &params,
        &scene,
        Some("/definitely/does/not/exist/state.ron"),
    );

    assert!(result.is_err());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn system_constructor_propagates_an_error_for_a_malformed_state_file() {
    let dir = unique_temp_dir();
    let (scene, params) = minimal_fluid_scene(&dir);

    let bad_state_path = dir.join("bad_state.ron");
    std::fs::write(&bad_state_path, "this is not valid RON {{{").unwrap();

    let result = SystemConstructor::<
        CubicBSpline3D,
        EulerCromer,
        SESPH,
        SpatialHashing,
        VolumeMapBoundary,
    >::new(&params, &scene, Some(bad_state_path.to_str().unwrap()));

    assert!(result.is_err());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn system_constructor_rejects_iisphwost_with_a_dynamic_boundary() {
    use simulation_lib::sph::pressure_solver::IISPHwOST;
    use simulation_lib::sph::setup::input::{
        BoundaryDefs, DynamicBoundaryDef, VertexNormalRenderOption,
    };

    let dir = unique_temp_dir();
    let cube_path = write_cube_obj(&dir, "cube.obj");
    let mut meshes = HashMap::new();
    meshes.insert("cube".to_string(), cube_path.to_str().unwrap().to_string());

    let mut scene = make_scene(meshes, vec![]);
    scene.boundary = BoundaryDefs {
        statics: vec![],
        dynamic: vec![DynamicBoundaryDef {
            mesh: "cube".to_string(),
            boundary_id: 0,
            density: 1000.0,
            translation: [0.0, 0.0, 0.0],
            rotation_euler_deg: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
            angular_velocity: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
            render_vertex_normals: VertexNormalRenderOption::FaceNormals,
        }],
    };
    let params = make_parameters(vec![]);

    let result = SystemConstructor::<
        CubicBSpline3D,
        TakePredicted,
        IISPHwOST,
        SpatialHashing,
        VolumeMapBoundary,
    >::new(&params, &scene, None);

    assert!(matches!(
        result,
        Err(SetupError::IncompatibleDynamicBoundary)
    ));

    let _ = std::fs::remove_dir_all(&dir);
}

// ─── SystemConstructor::new: resuming from a saved state ───────────────
//
// Requires `ron` as a dev-dependency (same version as the crate's own
// `[dependencies]` entry). A boundary-less scene keeps
// `SerBoundaryCheckpoint::dynamic_states` empty, so this doesn't require
// knowing `SerRigidBodyMotionCheckpoint`'s (unseen) field layout.

#[test]
fn system_constructor_resumes_time_bookkeeping_and_fluid_state_from_a_saved_state() {
    let dir = unique_temp_dir();
    let (scene, params) = minimal_fluid_scene(&dir);

    let checkpoint = SerSystemCheckpoint {
        time_steps_propagated: 42,
        current_time: 12.5,
        fluid: SerFluidCheckpoint {
            fluid_id: vec![0],
            position: vec![[0.1, 0.2, 0.3]],
            velocity: vec![[0.0, 0.0, 0.0]],
            mass: vec![1.0],
        },
        boundary: SerBoundaryCheckpoint {
            dynamic_states: vec![], // no boundaries in this scene
        },
    };
    let ron_text = ron::to_string(&checkpoint).expect("failed to serialize checkpoint to RON");
    let state_path = dir.join("state.ron");
    std::fs::write(&state_path, ron_text).unwrap();

    let constructor = SystemConstructor::<
        CubicBSpline3D,
        EulerCromer,
        SESPH,
        SpatialHashing,
        VolumeMapBoundary,
    >::new(&params, &scene, Some(state_path.to_str().unwrap()))
    .expect("expected resuming from a saved state to succeed");

    assert_eq!(constructor.initial_time_steps_propagated, 42);
    assert_eq!(constructor.initial_current_time, 12.5);
    // Fluid state must come from the checkpoint (1 particle), not from
    // re-sampling the cube mesh (which would yield many more particles).
    assert_eq!(constructor.fluid.len(), 1);

    let _ = std::fs::remove_dir_all(&dir);
}

// ─── new_boxed_system3d: builds and steps forward for every solver ─────

#[test]
fn new_boxed_system3d_builds_and_steps_forward_for_every_pressure_solver() {
    let dir = unique_temp_dir();
    let (scene, params) = minimal_fluid_scene(&dir);

    // Pairing per `Procedures::integration_scheme`'s doc comment: solvers
    // that write directly into position_pred/velocity_pred (IISPHwOST) are
    // paired with `TakePredicted`; conventional acceleration-based solvers
    // are paired with `EulerCromer`.
    let combos = [
        (
            PressureSolverVariant::SESPH,
            IntegrationSchemeVariant::EulerCromer,
        ),
        (
            PressureSolverVariant::SESPHwSplitting,
            IntegrationSchemeVariant::EulerCromer,
        ),
        (
            PressureSolverVariant::IISPH,
            IntegrationSchemeVariant::EulerCromer,
        ),
        (
            PressureSolverVariant::IISPHwOST,
            IntegrationSchemeVariant::TakePredicted,
        ),
    ];

    for (pressure_solver, integration_scheme) in combos {
        let procs = Procedures {
            kernel_function: KernelFnVariant::CubicBSpline3D,
            integration_scheme,
            pressure_solver,
            neighbor_search: NeighborSearchVariant::SpatialHashing,
            boundary_handling: BoundaryHandlingVariant::VolumeMapBoundary,
        };

        let mut system = new_boxed_system3d(&procs, &params, &scene, None)
            .expect("expected system construction to succeed for this solver/integrator pair");

        let time_before = system.time();
        system.step_forward_in_time();
        assert!(
            system.time() >= time_before,
            "expected time to advance (or stay equal) after stepping forward"
        );
        assert_eq!(system.time_steps_propagated(), 1);
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn new_boxed_system3d_builds_for_both_boundary_handling_variants_with_no_boundary_geometry() {
    let dir = unique_temp_dir();
    let (scene, params) = minimal_fluid_scene(&dir);

    for boundary_handling in [
        BoundaryHandlingVariant::VolumeMapBoundary,
        // NOTE: `StaticSampleBoundary`'s implementation wasn't available
        // to verify against; included here only as a basic smoke test.
        BoundaryHandlingVariant::StaticSampleBoundary,
    ] {
        let procs = Procedures {
            kernel_function: KernelFnVariant::CubicBSpline3D,
            integration_scheme: IntegrationSchemeVariant::EulerCromer,
            pressure_solver: PressureSolverVariant::SESPH,
            neighbor_search: NeighborSearchVariant::SpatialHashing,
            boundary_handling,
        };

        let mut system = new_boxed_system3d(&procs, &params, &scene, None)
            .expect("expected system construction to succeed for this boundary handling variant");
        system.step_forward_in_time();
    }

    let _ = std::fs::remove_dir_all(&dir);
}

// ─── Heavy path: an actual boundary mesh with VolumeMapBoundary ─────────

#[test]
#[ignore = "exercises VolumeMapBoundary's real mesh-discretization pipeline \
            (signed distance field + volume map via Gauss-Legendre quadrature) \
            end-to-end through SystemConstructor::new; run explicitly via \
            `cargo test -- --ignored` if you need to verify this wiring."]
fn new_boxed_system3d_builds_with_an_actual_static_boundary_mesh() {
    use simulation_lib::sph::setup::input::{
        BoundaryDefs, StaticBoundaryDef, VertexNormalRenderOption,
    };

    let dir = unique_temp_dir();
    let cube_path = write_cube_obj(&dir, "container.obj");
    let mut meshes = HashMap::new();
    meshes.insert(
        "container".to_string(),
        cube_path.to_str().unwrap().to_string(),
    );

    let mut scene = make_scene(meshes, vec![]);
    scene.boundary = BoundaryDefs {
        statics: vec![StaticBoundaryDef {
            mesh: "container".to_string(),
            boundary_id: 0,
            translation: [0.0, 0.0, 0.0],
            rotation_euler_deg: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
            render_vertex_normals: VertexNormalRenderOption::FaceNormals,
        }],
        dynamic: vec![],
    };

    let mut params = make_parameters(vec![]);
    params.kernel_support_radius = 0.05; // kept small to bound quadrature cost
    params.rest_density_grid_spacing = 0.1;

    let procs = Procedures {
        kernel_function: KernelFnVariant::CubicBSpline3D,
        integration_scheme: IntegrationSchemeVariant::EulerCromer,
        pressure_solver: PressureSolverVariant::SESPH,
        neighbor_search: NeighborSearchVariant::SpatialHashing,
        boundary_handling: BoundaryHandlingVariant::VolumeMapBoundary,
    };

    let system = new_boxed_system3d(&procs, &params, &scene, None)
        .expect("expected construction with an actual boundary mesh to succeed");
    assert_eq!(system.time_steps_propagated(), 0);

    let _ = std::fs::remove_dir_all(&dir);
}
