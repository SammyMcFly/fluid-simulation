//! Error types for constructing an [`SPHSystem`](crate::sph::SPHSystem) from
//! configuration and, optionally, a saved state file.

/// Errors that can occur while building a system from [`Parameters`](super::input::Parameters)
/// / [`Scene`](super::input::Scene) and constructing the concrete `System<...>`.
#[derive(Debug, thiserror::Error)]
pub enum SetupError {
    /// The saved `--state` file could not be read.
    #[error("failed to read saved state file: {0}")]
    Io(#[from] std::io::Error),

    /// The saved `--state` file's contents are not valid RON, or don't match
    /// the expected [`SerSystemCheckpoint`](crate::sph::SerSystemCheckpoint)
    /// structure.
    #[error("failed to parse saved state file: {0}")]
    Ron(#[from] ron::de::SpannedError),

    #[error(transparent)]
    Mesh(#[from] crate::utilities::triangle_mesh::MeshError),

    /// A `mesh` key in `[[fluid]]`, `[[boundary.static]]` or
    /// `[[boundary.dynamic]]` does not match any key in `Scene::meshes`.
    #[error("mesh '{0}' is referenced but not defined in [meshes]")]
    UnknownMesh(String),

    /// A `fluid_id` in `[[fluid]]` does not match any [`Fluid::id`](super::input::Fluid::id)
    /// in `Parameters::fluid`.
    #[error(
        "fluid id {0} is referenced by a [[fluid]] entry but not defined in \
         [[parameters.fluid]]"
    )]
    UndefinedFluidId(u32),
}
