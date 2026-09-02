//! Integration tests for `sph::setup::input`, exercising only its public
//! API. Private items (`check_time_step_keys`, `default_scale`,
//! `deserialize_scale`) are covered separately in the module's own
//! internal test block.

use std::sync::atomic::{AtomicUsize, Ordering};

use simulation_lib::sph::kernel::KernelFnVariant;
use simulation_lib::sph::pressure_solver::PressureSolverVariant;
use simulation_lib::sph::setup::input::{
    BoundaryDefs, ConfigError, DynamicBoundaryDef, Fluid, FluidDef, Light, ParameterFile,
    Parameters, Procedures, Scene, StaticBoundaryDef, VertexNormalRenderOption,
};

// ─── Fixtures / helpers ─────────────────────────────────────────────────

static TEMP_FILE_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn write_temp_file(contents: &str) -> std::path::PathBuf {
    let n = TEMP_FILE_COUNTER.fetch_add(1, Ordering::SeqCst);
    let path = std::env::temp_dir().join(format!("input_test_{}_{n}.toml", std::process::id()));
    std::fs::write(&path, contents).expect("failed to write temp file");
    path
}

#[cfg(not(feature = "cfl_time_step"))]
fn time_step_toml() -> &'static str {
    "time_increment = 0.001\n"
}
#[cfg(feature = "cfl_time_step")]
fn time_step_toml() -> &'static str {
    "max_time_increment = 0.001\ncfl_number = 0.4\n"
}

/// Text of the *other* build's time-step key, used to test
/// `ConfigError::FeatureMismatch`.
#[cfg(not(feature = "cfl_time_step"))]
fn inactive_time_step_toml() -> &'static str {
    "cfl_number = 0.4\n"
}
#[cfg(feature = "cfl_time_step")]
fn inactive_time_step_toml() -> &'static str {
    "time_increment = 0.001\n"
}

fn valid_parameter_file_text() -> String {
    format!(
        "[procedures]\n\
         kernel_function = \"CubicBSpline3D\"\n\
         integration_scheme = \"EulerCromer\"\n\
         pressure_solver = \"SESPH\"\n\
         neighbor_search = \"SpatialHashing\"\n\
         boundary_handling = \"VolumeMapBoundary\"\n\
         \n\
         [parameters]\n\
         buffer_length_limit = 100\n\
         {}\
         rest_density_grid_spacing = 0.05\n\
         kernel_support_radius = 0.1\n\
         disable_particles_below = -100.0\n\
         fluid_viscosity = 0.01\n\
         boundary_viscosity = 0.01\n\
         boundary_pressure_acceleration_weighting = 1.0\n\
         boundary_rest_volume_weighting = 1.0\n\
         stiffness = 500.0\n\
         target_density_error = 0.01\n\
         relaxation_factor = 0.5\n\
         min_diagonal_element = 1e-9\n\
         fluid = []\n",
        time_step_toml()
    )
}

fn valid_scene_text() -> &'static str {
    "[light]\n\
     position = [1.0, 2.0, 3.0]\n\
     \n\
     [meshes]\n\
     cube = \"cube.obj\"\n"
}

// ─── ConfigError: Display ───────────────────────────────────────────────

#[test]
fn config_error_io_variant_wraps_and_displays_the_source() {
    let result = ParameterFile::from_file("/definitely/does/not/exist_1234.toml");
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("expected a missing file to fail"),
    };
    assert!(matches!(err, ConfigError::Io(_)));
    assert!(format!("{err}").starts_with("I/O error:"));
}

#[test]
fn config_error_toml_variant_wraps_and_displays_the_source() {
    let path = write_temp_file("this is not valid toml {{{");
    let result = ParameterFile::from_file(path.to_str().unwrap());
    let _ = std::fs::remove_file(&path);
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("expected malformed TOML to fail"),
    };
    assert!(matches!(err, ConfigError::Toml(_)));
    assert!(format!("{err}").starts_with("TOML error:"));
}

#[test]
fn config_error_feature_mismatch_displays_the_message_verbatim() {
    let err = ConfigError::FeatureMismatch("some diagnostic message".to_string());
    assert_eq!(format!("{err}"), "some diagnostic message");
}

// ─── ParameterFile::from_file ───────────────────────────────────────────

#[test]
fn parameter_file_parses_a_well_formed_file() {
    let path = write_temp_file(&valid_parameter_file_text());
    let result = ParameterFile::from_file(path.to_str().unwrap());
    let _ = std::fs::remove_file(&path);

    let file = result.expect("expected a well-formed parameter file to parse");
    assert!(matches!(
        file.procedures.pressure_solver,
        PressureSolverVariant::SESPH
    ));
    assert_eq!(file.parameters.buffer_length_limit, 100);
    assert_eq!(file.parameters.stiffness, 500.0);
}

#[test]
fn parameter_file_fails_for_a_nonexistent_file() {
    let result = ParameterFile::from_file("/definitely/does/not/exist_5678.toml");
    assert!(matches!(result, Err(ConfigError::Io(_))));
}

#[test]
fn parameter_file_fails_for_malformed_toml() {
    let path = write_temp_file("not valid toml {{{");
    let result = ParameterFile::from_file(path.to_str().unwrap());
    let _ = std::fs::remove_file(&path);
    assert!(matches!(result, Err(ConfigError::Toml(_))));
}

#[test]
fn parameter_file_fails_when_procedures_section_is_missing() {
    let text = format!(
        "[parameters]\nbuffer_length_limit = 1\n{}\
         rest_density_grid_spacing = 0.05\nkernel_support_radius = 0.1\n\
         disable_particles_below = -1.0\nfluid_viscosity = 0.0\nboundary_viscosity = 0.0\n\
         boundary_pressure_acceleration_weighting = 1.0\nboundary_rest_volume_weighting = 1.0\n\
         stiffness = 1.0\ntarget_density_error = 0.01\nrelaxation_factor = 0.5\n\
         min_diagonal_element = 1e-9\nfluid = []\n",
        time_step_toml()
    );
    let path = write_temp_file(&text);
    let result = ParameterFile::from_file(path.to_str().unwrap());
    let _ = std::fs::remove_file(&path);
    assert!(matches!(result, Err(ConfigError::Toml(_))));
}

#[test]
fn parameter_file_fails_for_an_unknown_top_level_key() {
    let mut text = valid_parameter_file_text();
    text.push_str("\n[bogus_section]\nfoo = 1\n");
    let path = write_temp_file(&text);
    let result = ParameterFile::from_file(path.to_str().unwrap());
    let _ = std::fs::remove_file(&path);
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("expected an unknown top-level section to fail"),
    };
    assert!(matches!(err, ConfigError::Toml(_)));
    assert!(format!("{err}").contains("unknown field"));
}

#[test]
fn parameter_file_reports_feature_mismatch_even_when_the_file_is_otherwise_broken() {
    // `check_time_step_keys` runs on the raw parsed `toml::Table` BEFORE
    // the full structured deserialization is attempted — so a file with
    // the wrong-feature time-step key AND missing every other required
    // field must still surface `FeatureMismatch`, not a generic `Toml`
    // "missing field" error.
    let text = format!("[parameters]\n{}", inactive_time_step_toml());
    let path = write_temp_file(&text);
    let result = ParameterFile::from_file(path.to_str().unwrap());
    let _ = std::fs::remove_file(&path);
    assert!(
        matches!(result, Err(ConfigError::FeatureMismatch(_))),
        "expected FeatureMismatch, got {result:?}"
    );
}

// ─── Scene::from_file ───────────────────────────────────────────────────

#[test]
fn scene_parses_a_well_formed_file() {
    let path = write_temp_file(valid_scene_text());
    let result = Scene::from_file(path.to_str().unwrap());
    let _ = std::fs::remove_file(&path);

    let scene = result.expect("expected a well-formed scene file to parse");
    assert_eq!(scene.light.position, [1.0, 2.0, 3.0]);
    assert_eq!(
        scene.meshes.get("cube").map(String::as_str),
        Some("cube.obj")
    );
}

#[test]
fn scene_defaults_fluid_and_boundary_to_empty_when_omitted() {
    let path = write_temp_file(valid_scene_text());
    let scene = Scene::from_file(path.to_str().unwrap()).unwrap();
    let _ = std::fs::remove_file(&path);

    assert!(scene.fluid.is_empty());
    assert!(scene.boundary.statics.is_empty());
    assert!(scene.boundary.dynamic.is_empty());
}

#[test]
fn scene_fails_when_light_section_is_missing() {
    let text = "[meshes]\ncube = \"cube.obj\"\n";
    let path = write_temp_file(text);
    let result = Scene::from_file(path.to_str().unwrap());
    let _ = std::fs::remove_file(&path);
    assert!(matches!(result, Err(ConfigError::Toml(_))));
}

#[test]
fn scene_fails_when_meshes_section_is_missing() {
    let text = "[light]\nposition = [0.0, 0.0, 0.0]\n";
    let path = write_temp_file(text);
    let result = Scene::from_file(path.to_str().unwrap());
    let _ = std::fs::remove_file(&path);
    assert!(matches!(result, Err(ConfigError::Toml(_))));
}

#[test]
fn scene_fails_for_an_unknown_top_level_key() {
    let mut text = valid_scene_text().to_string();
    text.push_str("\n[bogus]\nfoo = 1\n");
    let path = write_temp_file(&text);
    let result = Scene::from_file(path.to_str().unwrap());
    let _ = std::fs::remove_file(&path);
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("expected an unknown top-level section to fail"),
    };
    assert!(format!("{err}").contains("unknown field"));
}

#[test]
fn scene_fails_for_a_nonexistent_file() {
    let result = Scene::from_file("/definitely/does/not/exist_9999.toml");
    assert!(matches!(result, Err(ConfigError::Io(_))));
}

#[test]
fn scene_fails_for_malformed_toml() {
    let path = write_temp_file("not valid toml {{{");
    let result = Scene::from_file(path.to_str().unwrap());
    let _ = std::fs::remove_file(&path);
    assert!(matches!(result, Err(ConfigError::Toml(_))));
}

#[test]
fn scene_does_not_validate_mesh_or_fluid_id_cross_references() {
    // Per the module doc comment: "Mesh names and fluid ids are *not*
    // validated here." A `[[fluid]]` entry referencing a mesh that isn't
    // declared in `[meshes]` must still parse successfully — cross-file
    // validation happens elsewhere (`SystemConstructor::new`).
    let text = "[light]\n\
                position = [0.0, 0.0, 0.0]\n\
                \n\
                [meshes]\n\
                \n\
                [[fluid]]\n\
                mesh = \"nonexistent_mesh\"\n\
                fluid_id = 999\n";
    let path = write_temp_file(text);
    let result = Scene::from_file(path.to_str().unwrap());
    let _ = std::fs::remove_file(&path);

    let scene = result.expect("Scene::from_file must not validate cross-references");
    assert_eq!(scene.fluid[0].mesh, "nonexistent_mesh");
    assert_eq!(scene.fluid[0].fluid_id, 999);
}

// ─── Nested struct behavior ─────────────────────────────────────────────

#[test]
fn fluid_def_defaults_translation_rotation_and_scale_when_omitted() {
    let def: FluidDef = toml::from_str("mesh = \"cube\"\nfluid_id = 0\n").unwrap();
    assert_eq!(def.translation, [0.0, 0.0, 0.0]);
    assert_eq!(def.rotation_euler_deg, [0.0, 0.0, 0.0]);
    assert_eq!(def.scale, [1.0, 1.0, 1.0]);
}

#[test]
fn fluid_def_rejects_an_unknown_field() {
    let result: Result<FluidDef, _> = toml::from_str("mesh = \"cube\"\nfluid_id = 0\nbogus = 1\n");
    assert!(result.is_err());
}

#[test]
fn static_boundary_def_defaults_render_vertex_normals_to_face_normals() {
    let def: StaticBoundaryDef = toml::from_str("mesh = \"container\"\nboundary_id = 0\n").unwrap();
    assert!(matches!(
        def.render_vertex_normals,
        VertexNormalRenderOption::FaceNormals
    ));
}

#[test]
fn dynamic_boundary_def_requires_density_but_defaults_velocity_fields() {
    let def: DynamicBoundaryDef =
        toml::from_str("mesh = \"cube\"\nboundary_id = 0\ndensity = 1000.0\n").unwrap();
    assert_eq!(def.velocity, [0.0, 0.0, 0.0]);
    assert_eq!(def.angular_velocity, [0.0, 0.0, 0.0]);

    let missing_density: Result<DynamicBoundaryDef, _> =
        toml::from_str("mesh = \"cube\"\nboundary_id = 0\n");
    assert!(
        missing_density.is_err(),
        "density has no default and must be required"
    );
}

#[test]
fn boundary_defs_uses_the_static_keyword_not_the_rust_field_name() {
    // `statics` is renamed to `static` in TOML (since `static` is a Rust
    // keyword); the Rust field name itself must NOT be accepted as a key.
    let good: Result<BoundaryDefs, _> =
        toml::from_str("[[static]]\nmesh = \"cube\"\nboundary_id = 0\n");
    assert!(
        good.is_ok(),
        "expected the renamed 'static' key to work: {good:?}"
    );
    assert_eq!(good.unwrap().statics.len(), 1);

    let bad: Result<BoundaryDefs, _> =
        toml::from_str("[[statics]]\nmesh = \"cube\"\nboundary_id = 0\n");
    assert!(
        bad.is_err(),
        "expected the literal Rust field name 'statics' to be rejected"
    );
}

#[test]
fn vertex_normal_render_option_deserializes_both_variants() {
    assert!(matches!(
        toml::from_str::<VertexNormalRenderOptionWrapper>("v = \"FaceNormals\"")
            .unwrap()
            .v,
        VertexNormalRenderOption::FaceNormals
    ));
    assert!(matches!(
        toml::from_str::<VertexNormalRenderOptionWrapper>("v = \"AngleWeightedPseudoNormals\"")
            .unwrap()
            .v,
        VertexNormalRenderOption::AngleWeightedPseudoNormals
    ));
}

#[derive(serde::Deserialize)]
struct VertexNormalRenderOptionWrapper {
    v: VertexNormalRenderOption,
}

// ─── Procedures: every documented variant name ─────────────────────────

fn make_procedures_text(
    kernel_function: &str,
    integration_scheme: &str,
    pressure_solver: &str,
    neighbor_search: &str,
    boundary_handling: &str,
) -> String {
    format!(
        "kernel_function = \"{kernel_function}\"\n\
         integration_scheme = \"{integration_scheme}\"\n\
         pressure_solver = \"{pressure_solver}\"\n\
         neighbor_search = \"{neighbor_search}\"\n\
         boundary_handling = \"{boundary_handling}\"\n"
    )
}

#[test]
fn procedures_deserializes_every_documented_variant_name() {
    for kernel_function in ["CubicBSpline3D"] {
        for integration_scheme in ["ExplicitEuler", "EulerCromer", "Verlet", "TakePredicted"] {
            for pressure_solver in ["SESPH", "SESPHwSplitting", "IISPH", "IISPHwOST"] {
                for neighbor_search in ["SpatialHashing"] {
                    for boundary_handling in ["StaticSampleBoundary", "VolumeMapBoundary"] {
                        let text = make_procedures_text(
                            kernel_function,
                            integration_scheme,
                            pressure_solver,
                            neighbor_search,
                            boundary_handling,
                        );
                        let result: Result<Procedures, _> = toml::from_str(&text);
                        assert!(
                            result.is_ok(),
                            "failed to deserialize combination {text}: {result:?}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn procedures_doc_example_kernel_function_name_is_actually_invalid() {
    // FLAG: the doc comment on `Procedures` shows
    //   kernel_function = "CubicBSpline"
    // as an example, but the real enum variant is
    // `KernelFnVariant::CubicBSpline3D` — the doc example, taken literally,
    // does NOT deserialize. This test pins down both halves of that
    // discrepancy: the documented spelling fails, the actual variant name
    // succeeds. If the doc comment is ever fixed to say "CubicBSpline3D",
    // update/remove the first half of this test accordingly.
    let text_from_docs = make_procedures_text(
        "CubicBSpline", // as literally written in the doc example
        "EulerCromer",
        "IISPH",
        "SpatialHashing",
        "StaticSampleBoundary",
    );
    let result: Result<Procedures, _> = toml::from_str(&text_from_docs);
    assert!(
        result.is_err(),
        "expected the doc-example spelling 'CubicBSpline' to be rejected, but it parsed: {result:?}"
    );

    let text_correct = make_procedures_text(
        "CubicBSpline3D", // the actual `KernelFnVariant` variant name
        "EulerCromer",
        "IISPH",
        "SpatialHashing",
        "StaticSampleBoundary",
    );
    let result: Result<Procedures, _> = toml::from_str(&text_correct);
    assert!(
        result.is_ok(),
        "expected the actual variant name 'CubicBSpline3D' to parse: {result:?}"
    );
    assert!(matches!(
        result.unwrap().kernel_function,
        KernelFnVariant::CubicBSpline3D
    ));
}

#[test]
fn procedures_rejects_an_unknown_variant_name() {
    let text = make_procedures_text(
        "NotARealKernel",
        "EulerCromer",
        "IISPH",
        "SpatialHashing",
        "StaticSampleBoundary",
    );
    let result: Result<Procedures, _> = toml::from_str(&text);
    assert!(result.is_err());
}

// ─── Fluid / Parameters ─────────────────────────────────────────────────

#[test]
fn fluid_requires_both_id_and_rest_density() {
    let good: Result<Fluid, _> = toml::from_str("id = 0\nrest_density = 1000.0\n");
    assert!(good.is_ok());

    let missing_rest_density: Result<Fluid, _> = toml::from_str("id = 0\n");
    assert!(missing_rest_density.is_err());
}

#[test]
fn parameters_rejects_an_unknown_field() {
    let text = format!(
        "buffer_length_limit = 1\n{}\
         rest_density_grid_spacing = 0.05\nkernel_support_radius = 0.1\n\
         disable_particles_below = -1.0\nfluid_viscosity = 0.0\nboundary_viscosity = 0.0\n\
         boundary_pressure_acceleration_weighting = 1.0\nboundary_rest_volume_weighting = 1.0\n\
         stiffness = 1.0\ntarget_density_error = 0.01\nrelaxation_factor = 0.5\n\
         min_diagonal_element = 1e-9\nfluid = []\nbogus_field = 1\n",
        time_step_toml()
    );
    let result: Result<Parameters, _> = toml::from_str(&text);
    assert!(result.is_err());
}

// ─── Light ───────────────────────────────────────────────────────────────

#[test]
fn light_requires_position() {
    let result: Result<Light, _> = toml::from_str("");
    assert!(result.is_err());
}
