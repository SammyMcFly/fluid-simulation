//! Input module
//!
//! Deserialization of the two TOML files that drive a simulation run:
//!
//! | File | Sections | Entry point |
//! |------|----------|-------------|
//! | Parameter file | `[procedures]`, `[parameters]` | [`ParameterFile::from_file`] |
//! | Scene file | `[light]`, `[meshes]`, `[[fluid]]`, `[boundary]` | [`Scene::from_file`] |
//!
//! Both files are deserialized as a whole, so unknown top-level sections are rejected
//! along with unknown keys.
//!
//! Cross-references between the two files are resolved by id and by name:
//!
//! - [`FluidDef::fluid_id`] must match some [`Fluid::id`] in [`Parameters::fluid`].
//! - [`FluidDef::mesh`], [`StaticBoundaryDef::mesh`] and [`DynamicBoundaryDef::mesh`]
//!   must be keys of [`Scene::meshes`].
//!
//! All configuration structs use `#[serde(deny_unknown_fields)]`: an unrecognized key
//! is an error naming the offending key and the valid alternatives for its table.
//! Keys whose presence depends on a cargo feature are diagnosed separately, with a
//! message naming the required flag rather than reporting an unknown field.

use serde::Deserialize;
use serde::de::{self, Deserializer, SeqAccess, Visitor};
use std::collections::HashMap;

use crate::{
    integration_schemes::IntegrationSchemeVariant,
    neighbor_search::NeighborSearchVariant,
    sph::{
        boundary_handling::BoundaryHandlingVariant, kernel::KernelFnVariant,
        pressure_solver::PressureSolverVariant,
    },
};

/// Errors that can occur while loading and validating the parameter or scene
/// configuration files.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// The file could not be read (missing, permissions, ...).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The file's contents are not valid TOML, or don't match the expected
    /// structure (missing/unknown fields, wrong types, ...).
    #[error("TOML error: {0}")]
    Toml(#[from] toml::de::Error),

    /// A key in `[parameters]` is only meaningful with a `cfl_time_step`
    /// feature configuration different from the one this binary was built
    /// with.
    #[error("{0}")]
    FeatureMismatch(String),
}

/// Full contents of the parameter file: both sections in one document.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParameterFile {
    /// Algorithm selection from `[procedures]`.
    pub procedures: Procedures,

    /// Numerical parameters from `[parameters]`.
    pub parameters: Parameters,
}

impl ParameterFile {
    /// Reads and validates the parameter file at `file_path`.
    ///
    /// # Errors
    ///
    /// Fails if the file cannot be read, is not valid TOML, is missing a top-level
    /// section, contains an unknown key or section, or configures a time-stepping
    /// mode unsupported by this build.
    pub fn from_file(file_path: &str) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(file_path)?;

        // Diagnose feature mismatches before the derived impl reports them as
        // unknown or missing fields.
        let document: toml::Table = toml::from_str(&text)?;
        check_time_step_keys(&document)?;

        // Deserialize from the source text so that TOML errors retain their spans.
        Ok(toml::from_str(&text)?)
    }
}

/// Algorithm selection, read from the `[procedures]` section of the parameter file.
///
/// Every field is required and selects one implementation used to instantiate the
/// generic simulation system. Values in TOML are the enum variant names.
///
/// ```toml
/// [procedures]
/// kernel_function    = "CubicBSpline"
/// integration_scheme = "EulerCromer"
/// pressure_solver    = "IISPH"
/// neighbor_search    = "SpatialHashing"
/// boundary_handling  = "StaticSampleBoundary"
/// ```
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Procedures {
    /// Smoothing kernel used for all SPH interpolations.
    ///
    /// Its support radius is configured separately via
    /// [`Parameters::kernel_support_radius`].
    pub kernel_function: KernelFnVariant,

    /// Time integration scheme advancing particle velocities and positions.
    ///
    /// Choose a scheme consistent with the selected [`Self::pressure_solver`]: solvers
    /// that already produce an integrated predicted state are meant to be paired with
    /// [`IntegrationSchemeVariant::TakePredicted`].
    pub integration_scheme: IntegrationSchemeVariant,

    /// Pressure solver determining how incompressibility is enforced.
    ///
    /// State-equation solvers are governed by [`Parameters::stiffness`]; iterative
    /// solvers by [`Parameters::target_density_error`],
    /// [`Parameters::relaxation_factor`] and [`Parameters::min_diagonal_element`].
    pub pressure_solver: PressureSolverVariant,

    /// Acceleration structure used to find neighboring particles.
    ///
    /// Cell sizing is derived from [`Parameters::kernel_support_radius`].
    pub neighbor_search: NeighborSearchVariant,

    /// Strategy for handling fluid–boundary interaction.
    ///
    /// Sampling-based variants use [`Parameters::rest_density_grid_spacing`] and
    /// [`Parameters::boundary_rest_volume_weighting`].
    pub boundary_handling: BoundaryHandlingVariant,
}

/// Numerical parameters, read from the `[parameters]` section of the parameter file.
///
/// None of the fields have serde defaults: **every** field present in the current build
/// must appear in the file. Which time-stepping fields exist depends on the
/// `cfl_time_step` feature (see [`Self::time_increment`] versus
/// [`Self::max_time_increment`] and [`Self::cfl_number`]).
///
/// Solver-specific fields must be supplied regardless of which solver
/// [`Procedures::pressure_solver`] selects; unused ones are ignored at run time.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Parameters {
    /// Maximum number of simulated time steps buffered between the simulation worker
    /// and the frontend.
    ///
    /// Acts as backpressure: once the buffer is full the worker stalls instead of
    /// running ahead and growing memory without bound.
    pub buffer_length_limit: usize,

    /// Fixed simulation time step.
    ///
    /// Only present when the crate is built **without** the `cfl_time_step` feature;
    /// otherwise use [`Self::max_time_increment`] and [`Self::cfl_number`].
    #[cfg(not(feature = "cfl_time_step"))]
    pub time_increment: f64,

    /// Upper bound for the adaptive time step.
    ///
    /// Only present with the `cfl_time_step` feature; the step actually taken is the
    /// smaller of this value and the CFL-limited step derived from
    /// [`Self::cfl_number`].
    #[cfg(feature = "cfl_time_step")]
    pub max_time_increment: f64,

    /// Courant–Friedrichs–Lewy safety factor limiting the adaptive time step relative
    /// to the maximum particle velocity and the kernel support radius.
    ///
    /// Only present with the `cfl_time_step` feature. Smaller values are more
    /// conservative.
    #[cfg(feature = "cfl_time_step")]
    pub cfl_number: f64,

    /// Fluid phases available in the simulation, given as `[[parameters.fluid]]`
    /// entries.
    ///
    /// Each [`Fluid::id`] can be referenced by [`FluidDef::fluid_id`] in the scene
    /// file; several entries with different rest densities enable multi-fluid scenes.
    pub fluid: Vec<Fluid>,

    /// Spacing of the regular grid used to sample geometry into particles and to
    /// compute rest volumes.
    ///
    /// Together with the fluid rest density this determines particle masses, and it
    /// should be chosen consistently with [`Self::kernel_support_radius`].
    pub rest_density_grid_spacing: f64,

    /// Support radius of the smoothing kernel selected by
    /// [`Procedures::kernel_function`].
    ///
    /// Also sizes the cells of the neighbor search: larger values mean more neighbors
    /// per particle, smoother fields and higher cost per step.
    pub kernel_support_radius: f64,

    /// Height threshold below which particles are deactivated.
    ///
    /// Particles that leave the domain (for example through a leaking boundary) fall
    /// past this level and are excluded from further computation instead of degrading
    /// performance indefinitely.
    pub disable_particles_below: f64,

    /// Artificial viscosity coefficient for fluid–fluid interaction.
    ///
    /// Damps relative motion between neighboring fluid particles; larger values give
    /// a more viscous, more stable but less lively fluid.
    pub fluid_viscosity: f64,

    /// Artificial viscosity coefficient for fluid–boundary interaction.
    ///
    /// Controls the amount of friction at walls: `0.0` approximates a free-slip
    /// boundary, larger values approach no-slip.
    pub boundary_viscosity: f64,

    /// Weighting of the boundary contribution to the pressure acceleration of fluid
    /// particles.
    ///
    /// `1.0` applies the boundary term unmodified; lower values soften the repulsion
    /// at walls, higher values stiffen it.
    pub boundary_pressure_acceleration_weighting: f64,

    /// Scaling factor applied to the computed rest volumes of boundary samples.
    ///
    /// Compensates for over- or under-estimated boundary volumes produced by the
    /// sampling in [`Procedures::boundary_handling`], which otherwise show up as
    /// density errors near walls.
    pub boundary_rest_volume_weighting: f64,

    /// Stiffness constant of the state equation.
    ///
    /// Used by state-equation pressure solvers: higher values enforce
    /// incompressibility more strongly but require a smaller time step. Ignored by
    /// iterative solvers.
    pub stiffness: f64,

    // /// Number of iterations to perform in the iterative pressure solver.
    // solver_iterations: u32,
    /// Density error at which an iterative pressure solver is considered converged,
    /// as a fraction of the rest density.
    ///
    /// Ignored by state-equation solvers.
    pub target_density_error: f64,

    /// Relaxation factor of the iterative pressure solver's update.
    ///
    /// Values below `1.0` under-relax and improve stability; values that are too large
    /// can make the iteration diverge. Ignored by state-equation solvers.
    pub relaxation_factor: f64,

    /// Lower bound on the magnitude of the system matrix diagonal in the iterative
    /// pressure solver.
    ///
    /// Guards against division by near-zero diagonal elements, which occur for
    /// particles with too few neighbors (free surface, isolated droplets). Ignored by
    /// state-equation solvers.
    pub min_diagonal_element: f64,
}

/// Definition of a single fluid phase, given as a `[[parameters.fluid]]` entry.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fluid {
    /// Identifier of this fluid phase.
    ///
    /// Referenced by [`FluidDef::fluid_id`] in the scene file. Ids are expected to be
    /// unique across [`Parameters::fluid`].
    pub id: u32,

    /// Rest density of this fluid phase.
    ///
    /// Together with [`Parameters::rest_density_grid_spacing`] it determines the mass
    /// of the particles sampled for this phase.
    pub rest_density: f64,
}

/// Contents of the scene file: lighting, mesh assets and the geometry placed in the
/// simulation domain.
///
/// Unlike [`Parameters`], the keys of this struct sit at the top level of the file.
///
/// ```toml
/// [light]
/// position = [5.0, 8.0, 5.0]
///
/// [meshes]
/// container = "assets/container.obj"
/// drop      = "assets/sphere.obj"
///
/// [[fluid]]
/// mesh     = "drop"
/// fluid_id = 0
///
/// [[boundary.static]]
/// mesh        = "container"
/// boundary_id = 0
/// ```
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scene {
    /// Light source used when rendering the scene.
    pub light: Light,

    /// Mesh assets as a map from name to file path.
    ///
    /// The names are referenced by [`FluidDef::mesh`], [`StaticBoundaryDef::mesh`] and
    /// [`DynamicBoundaryDef::mesh`], so a mesh reused several times is declared once.
    /// Relative paths are resolved against the process working directory.
    pub meshes: HashMap<String, String>,

    /// Fluid volumes to sample with particles, given as `[[fluid]]` entries.
    ///
    /// Defaults to empty, which yields a simulation without fluid.
    #[serde(default)]
    pub fluid: Vec<FluidDef>,

    /// Static and dynamic boundary geometry.
    ///
    /// Defaults to empty, which yields an unbounded domain.
    #[serde(default)]
    pub boundary: BoundaryDefs,
}

impl Scene {
    /// Reads and deserializes the scene file at `file_path`.
    ///
    /// # Errors
    ///
    /// Fails if the file cannot be read, is not valid TOML, is missing `[light]` or
    /// `[meshes]`, or contains an unknown key or section. Mesh names and fluid ids are
    /// *not* validated here.
    pub fn from_file(file_path: &str) -> Result<Self, ConfigError> {
        // Read the scene file
        let config: Self = toml::from_str(&std::fs::read_to_string(file_path)?)?;
        Ok(config)
    }
}

/// Light source of the scene, given as the `[light]` section.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Light {
    /// World-space position of the point light as `[x, y, z]`.
    pub position: [f64; 3],
}

/// Placement of a fluid volume, given as a `[[fluid]]` entry.
///
/// The referenced mesh is sampled with particles using
/// [`Parameters::rest_density_grid_spacing`]; the transform is applied to the mesh
/// before sampling.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FluidDef {
    /// Name of the mesh to sample, as declared in [`Scene::meshes`].
    pub mesh: String,

    /// Fluid phase assigned to the sampled particles.
    ///
    /// Must match a [`Fluid::id`] in [`Parameters::fluid`].
    pub fluid_id: u32,

    /// Translation applied to the mesh as `[x, y, z]`. Defaults to `[0.0, 0.0, 0.0]`.
    #[serde(default)]
    pub translation: [f64; 3],

    /// Rotation applied to the mesh as Euler angles in **degrees**.
    /// Defaults to `[0.0, 0.0, 0.0]`.
    #[serde(default)]
    pub rotation_euler_deg: [f64; 3],

    /// Scaling applied to the mesh, defaulting to `[1.0, 1.0, 1.0]`.
    ///
    /// Accepts either a single number for uniform scaling or an array of three
    /// numbers: `scale = 2.0` is equivalent to `scale = [2.0, 2.0, 2.0]`.
    #[serde(default = "default_scale", deserialize_with = "deserialize_scale")]
    pub scale: [f64; 3],
}

/// Boundary geometry of the scene, given as the `[boundary]` section.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundaryDefs {
    /// Immobile boundaries, given as `[[boundary.static]]` entries.
    ///
    /// Named `static` in TOML because the Rust identifier is a keyword.
    #[serde(default, rename = "static")]
    pub statics: Vec<StaticBoundaryDef>,

    /// Movable boundaries, given as `[[boundary.dynamic]]` entries.
    #[serde(default)]
    pub dynamic: Vec<DynamicBoundaryDef>,
}

/// Placement of a static boundary, given as a `[[boundary.static]]` entry.
///
/// Static boundaries are sampled once during setup and never move, so their neighbor
/// information can be reused across time steps.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StaticBoundaryDef {
    /// Name of the boundary mesh, as declared in [`Scene::meshes`].
    pub mesh: String,

    /// Identifier of this boundary, used to distinguish boundaries in measurements
    /// and rendering.
    pub boundary_id: u32,

    /// Translation applied to the mesh as `[x, y, z]`. Defaults to `[0.0, 0.0, 0.0]`.
    #[serde(default)]
    pub translation: [f64; 3],

    /// Rotation applied to the mesh as Euler angles in **degrees**.
    /// Defaults to `[0.0, 0.0, 0.0]`.
    #[serde(default)]
    pub rotation_euler_deg: [f64; 3],

    /// Scaling applied to the mesh, defaulting to `[1.0, 1.0, 1.0]`.
    ///
    /// Accepts a single number for uniform scaling or an array of three numbers.
    #[serde(default = "default_scale", deserialize_with = "deserialize_scale")]
    pub scale: [f64; 3],

    /// How vertex normals of this mesh are derived for shading.
    ///
    /// Defaults to [`VertexNormalRenderOption::FaceNormals`].
    #[serde(default)]
    pub render_vertex_normals: VertexNormalRenderOption,
}

/// Dynamic boundary that acts as a rigid body, given as a `[[boundary.dynamic]]` entry.
///
/// Same fields as [`StaticBoundaryDef`], but the geometry is capable of performing rigid body
/// movemnts during the simulation interacting with the simulated fluid.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DynamicBoundaryDef {
    /// Name of the boundary mesh, as declared in [`Scene::meshes`].
    pub mesh: String,

    /// Identifier of this boundary, used to distinguish boundaries in measurements
    /// and rendering.
    pub boundary_id: u32,

    /// Density of the rigid body which is used to calculate the total mass
    /// and the inertia tensor.
    pub density: f64,

    /// Initial translation applied to the mesh as `[x, y, z]`.
    /// Defaults to `[0.0, 0.0, 0.0]`.
    #[serde(default)]
    pub translation: [f64; 3],

    /// Initial rotation around the center of mass of the mesh applied to the mesh as Euler angles in **degrees**.
    /// Defaults to `[0.0, 0.0, 0.0]`.
    #[serde(default)]
    pub rotation_euler_deg: [f64; 3],

    /// Initial linear velocity of rigid body.
    /// Defaults to `[0.0, 0.0, 0.0]`.
    #[serde(default)]
    pub velocity: [f64; 3],

    /// Initial angular velocity of rigid body.
    /// Defaults to `[0.0, 0.0, 0.0]`.
    #[serde(default)]
    pub angular_velocity: [f64; 3],

    /// Scaling applied to the mesh, defaulting to `[1.0, 1.0, 1.0]`.
    ///
    /// Accepts a single number for uniform scaling or an array of three numbers.
    #[serde(default = "default_scale", deserialize_with = "deserialize_scale")]
    pub scale: [f64; 3],

    /// How vertex normals of this mesh are derived for shading.
    ///
    /// Defaults to [`VertexNormalRenderOption::FaceNormals`].
    #[serde(default)]
    pub render_vertex_normals: VertexNormalRenderOption,
}

/// Method used to derive vertex normals of a boundary mesh for shading.
#[derive(Debug, Default, Copy, Clone, Deserialize)]
pub enum VertexNormalRenderOption {
    /// Use the normal of each face, giving faceted shading with hard edges.
    ///
    /// Suitable for boxes and other flat-sided geometry.
    #[default]
    FaceNormals,

    /// Average adjacent face normals weighted by their incident angles, giving smooth
    /// shading.
    ///
    /// Suitable for curved geometry such as spheres and cylinders.
    AngleWeightedPseudoNormals,
}

/// Keys of `[parameters]` that belong to the *other* `cfl_time_step` configuration,
/// paired with the diagnostic shown when one of them is encountered.
///
/// Without this check such a key produces either `unknown field ...` or a
/// `missing field ...` error for its counterpart, neither of which mentions the
/// feature flag that actually causes the mismatch.
#[cfg(feature = "cfl_time_step")]
const INACTIVE_TIME_STEP_KEYS: &[(&str, &str)] = &[(
    "time_increment",
    "`time_increment` configures a fixed time step, but this binary was built with \
     the `cfl_time_step` feature. Use `max_time_increment` and `cfl_number` instead, \
     or rebuild without the feature.",
)];

/// See the `cfl_time_step` variant of this constant.
#[cfg(not(feature = "cfl_time_step"))]
const INACTIVE_TIME_STEP_KEYS: &[(&str, &str)] = &[
    (
        "max_time_increment",
        "`max_time_increment` requires the `cfl_time_step` feature, which this binary \
         was built without. Use `time_increment` instead, or rebuild with \
         `--features cfl_time_step`.",
    ),
    (
        "cfl_number",
        "`cfl_number` requires the `cfl_time_step` feature, which this binary was \
         built without. Use `time_increment` instead, or rebuild with \
         `--features cfl_time_step`.",
    ),
];

/// Rejects `[parameters]` keys that are only meaningful in the other
/// `cfl_time_step` configuration.
///
/// # Errors
///
/// Returns a message naming the offending key and the required feature flag.
fn check_time_step_keys(document: &toml::Table) -> Result<(), ConfigError> {
    let Some(parameters) = document.get("parameters").and_then(toml::Value::as_table) else {
        return Ok(());
    };

    for (key, diagnostic) in INACTIVE_TIME_STEP_KEYS {
        if parameters.contains_key(*key) {
            return Err(ConfigError::FeatureMismatch((*diagnostic).to_string()));
        }
    }

    Ok(())
}

/// Default for the `scale` fields: no scaling.
fn default_scale() -> [f64; 3] {
    [1.0, 1.0, 1.0]
}

/// Deserializes a scale factor from either a single number or an array of three
/// numbers.
///
/// A scalar `s` is expanded to `[s, s, s]`, so both `scale = 2.0` and
/// `scale = [2.0, 2.0, 2.0]` are accepted.
fn deserialize_scale<'de, D>(deserializer: D) -> Result<[f64; 3], D::Error>
where
    D: Deserializer<'de>,
{
    struct ScaleVisitor;

    impl<'de> Visitor<'de> for ScaleVisitor {
        type Value = [f64; 3];

        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "a number or an array of 3 numbers")
        }

        fn visit_f64<E: de::Error>(self, v: f64) -> Result<[f64; 3], E> {
            if v < 0.0 {
                return Err(de::Error::invalid_value(
                    de::Unexpected::Float(v),
                    &"a non-negative scale factor",
                ));
            }
            Ok([v, v, v])
        }

        fn visit_i64<E: de::Error>(self, v: i64) -> Result<[f64; 3], E> {
            if v < 0 {
                return Err(de::Error::invalid_value(
                    de::Unexpected::Signed(v),
                    &"a non-negative scale factor",
                ));
            }
            Ok([v as f64, v as f64, v as f64])
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<[f64; 3], A::Error> {
            let x = seq
                .next_element::<f64>()?
                .ok_or_else(|| de::Error::invalid_length(0, &self))?;
            let y = seq
                .next_element::<f64>()?
                .ok_or_else(|| de::Error::invalid_length(1, &self))?;
            let z = seq
                .next_element::<f64>()?
                .ok_or_else(|| de::Error::invalid_length(2, &self))?;
            for (axis, v) in [("x", x), ("y", y), ("z", z)] {
                if v < 0.0 {
                    return Err(de::Error::invalid_value(
                        de::Unexpected::Float(v),
                        &format!("a non-negative scale factor for {axis}").as_str(),
                    ));
                }
            }
            Ok([x, y, z])
        }
    }

    deserializer.deserialize_any(ScaleVisitor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_field_is_rejected() {
        let err =
            toml::from_str::<Light>("position = [0.0, 0.0, 0.0]\ncolour = \"red\"").unwrap_err();
        assert!(err.to_string().contains("unknown field"), "{err}");
    }

    #[test]
    fn unknown_section_is_rejected() {
        let err = toml::from_str::<Scene>(
            "[light]\nposition = [0.0, 0.0, 0.0]\n[meshes]\n[rendering]\nsamples = 4",
        )
        .unwrap_err();
        assert!(err.to_string().contains("unknown field"), "{err}");
    }

    #[test]
    fn inactive_time_step_key_names_the_feature() {
        #[cfg(feature = "cfl_time_step")]
        let document: toml::Table = toml::from_str("[parameters]\ntime_increment = 0.001").unwrap();
        #[cfg(not(feature = "cfl_time_step"))]
        let document: toml::Table = toml::from_str("[parameters]\ncfl_number = 0.4").unwrap();

        let err = check_time_step_keys(&document).unwrap_err();
        assert!(err.to_string().contains("cfl_time_step"), "{err}");
    }
}
