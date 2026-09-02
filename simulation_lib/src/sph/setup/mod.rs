//! Module for scene building and parameter importing
//!
//!
pub mod input;

use crate::neighbor_search::NeighborSearch;
use crate::neighbor_search::*;
use crate::sph::boundary_handling::BoundaryHandling;
use crate::sph::boundary_handling::*;
use crate::sph::fluid::{Fluid, Len};
use crate::sph::integration_schemes::IntegrationScheme;
use crate::sph::integration_schemes::*;
use crate::sph::kernel::*;
use crate::sph::pressure_solver::PressureSolver;
use crate::sph::pressure_solver::*;
use crate::sph::{SPHSystem, SerSystemCheckpoint, System};
use crate::sph::{SystemCheckpoint, SystemParameters};
use crate::utilities::triangle_mesh::{MeshHandle, MeshLibrary};
use input::{Parameters, Procedures, Scene};

use std::collections::HashMap;

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

    /// The selected [`Procedures::pressure_solver`](super::input::Procedures::pressure_solver)
    /// does not correctly support two-way coupling with dynamic boundaries
    /// (see [`PressureSolver::SUPPORTS_DYNAMIC_BOUNDARIES`](crate::sph::pressure_solver::PressureSolver::SUPPORTS_DYNAMIC_BOUNDARIES)),
    /// but the scene defines at least one `[[boundary.dynamic]]` entry.
    #[error(
        "the selected pressure solver does not support dynamic boundaries, but the \
         scene defines at least one; choose a different pressure_solver or remove the \
         dynamic boundary/boundaries"
    )]
    IncompatibleDynamicBoundary,
}

pub struct SystemConstructor<
    K: KernelFn,
    I: IntegrationScheme,
    P: PressureSolver,
    N: NeighborSearch,
    B: BoundaryHandling,
> {
    // pub config: Config,
    pub fluid: Fluid,
    pub boundary: B,
    pub system_parameters: SystemParameters,
    /// Time step counter to resume from when loading a saved `--state` file;
    /// `0` when sampling fresh fluid/boundary geometry from the scene.
    pub initial_time_steps_propagated: u64,
    /// Accumulated physical time to resume from when loading a saved `--state`
    /// file; `0.` when sampling fresh geometry.
    pub initial_current_time: f64,
    _kernel_fn: std::marker::PhantomData<K>,
    pub integrator: I,
    pub pressure_solver: P,
    pub neighbor_search: N,
}

impl<K: KernelFn, I: IntegrationScheme, P: PressureSolver, N: NeighborSearch, B: BoundaryHandling>
    SystemConstructor<K, I, P, N, B>
{
    pub fn new(
        params: &Parameters,
        scene: &Scene,
        sample_state_file_path: Option<&str>,
    ) -> Result<Self, SetupError> {
        let integrator = I::default();
        let pressure_solver: P = P::new(params);
        let mut neighbor_search = N::new(params.kernel_support_radius);

        // load triangle meshes
        let mut meshes = MeshLibrary::default();
        let mut index_map = HashMap::new();
        for (i, mesh) in scene.meshes.keys().enumerate() {
            meshes.load_obj(scene.meshes.get(mesh).unwrap())?;
            index_map.insert(mesh.clone(), i);
        }

        // load fluid samples
        let fluid_rest_densities: HashMap<u32, f64> = params
            .fluid
            .iter()
            .map(|f| (f.id, f.rest_density))
            .collect();
        let saved_checkpoint: Option<SystemCheckpoint> =
            if let Some(file_path) = sample_state_file_path {
                let content = std::fs::read_to_string(file_path)?;
                let ser_state: SerSystemCheckpoint = ron::from_str(&content)?;
                Some(ser_state.into())
            } else {
                None
            };

        let fluid = if let Some(checkpoint) = &saved_checkpoint {
            checkpoint.get_fluid().clone().into()
        } else {
            let mut fluid = Fluid::new();
            for f in &scene.fluid {
                // select mesh
                let mut mesh = meshes
                    .get_mesh_container(MeshHandle {
                        idx: *index_map
                            .get(&f.mesh)
                            .ok_or_else(|| SetupError::UnknownMesh(f.mesh.clone()))?,
                        mesh_id: f.fluid_id,
                    })
                    .clone();
                // apply transformation
                mesh.transform(&f.translation, &f.rotation_euler_deg, &f.scale);

                fluid.add_samples(
                    mesh.trimesh(),
                    f.fluid_id,
                    *fluid_rest_densities
                        .get(&f.fluid_id)
                        .ok_or(SetupError::UndefinedFluidId(f.fluid_id))?,
                    params.rest_density_grid_spacing,
                );
            }
            if fluid.is_empty() {
                tracing::warn!("No fluid is present in simulation.");
            };
            fluid
        };

        // load boundary
        let mut boundary = {
            let mut boundary = B::new();
            for b in &scene.boundary.statics {
                // select mesh
                let mut mesh = meshes
                    .get_mesh_container(MeshHandle {
                        idx: *index_map
                            .get(&b.mesh)
                            .ok_or_else(|| SetupError::UnknownMesh(b.mesh.clone()))?,
                        mesh_id: b.boundary_id,
                    })
                    .clone();

                boundary.add_static_boundary(
                    &mut mesh,
                    b,
                    params.rest_density_grid_spacing,
                    params.kernel_support_radius,
                );
            }
            for b in &scene.boundary.dynamic {
                // select mesh
                let mut mesh = meshes
                    .get_mesh_container(MeshHandle {
                        idx: *index_map
                            .get(&b.mesh)
                            .ok_or_else(|| SetupError::UnknownMesh(b.mesh.clone()))?,
                        mesh_id: b.boundary_id,
                    })
                    .clone();

                boundary.add_dynamic_boundary(
                    &mut mesh,
                    b,
                    params.rest_density_grid_spacing,
                    params.kernel_support_radius,
                );
            }
            if boundary.is_empty() {
                tracing::warn!("No boundary is present in simulation.");
            };
            boundary
        };
        boundary.initialize::<K>(
            &mut neighbor_search,
            params.kernel_support_radius,
            params.boundary_rest_volume_weighting,
        );

        // Reject pressure-solver/boundary combinations known to produce incorrect
        // fluid-boundary coupling — see `PressureSolver::SUPPORTS_DYNAMIC_BOUNDARIES`.
        if !P::SUPPORTS_DYNAMIC_BOUNDARIES && boundary.iter().any(|b| b.is_dynamic()) {
            return Err(SetupError::IncompatibleDynamicBoundary);
        }

        // If resuming from a saved state, overwrite the boundary's dynamic state
        // (pose/velocity) with the saved one. Done AFTER `initialize()` so that,
        // for `StaticSampleBoundary`, neighbor lists and pseudo volumes are already
        // computed relative to the scene's initial pose before being
        // overwritten.
        if let Some(checkpoint) = &saved_checkpoint {
            boundary.restore_from_checkpoint(checkpoint.get_boundary());
        }

        let initial_time_steps_propagated = saved_checkpoint
            .as_ref()
            .map(SystemCheckpoint::get_time_steps_propagated)
            .unwrap_or(0);
        let initial_current_time = saved_checkpoint
            .as_ref()
            .map(SystemCheckpoint::get_current_time)
            .unwrap_or(0.);

        // init system properties
        let system_parameters = SystemParameters::new(
            #[cfg(not(feature = "cfl_time_step"))]
            params.time_increment,
            #[cfg(feature = "cfl_time_step")]
            params.max_time_increment,
            #[cfg(feature = "cfl_time_step")]
            params.cfl_number,
            params.rest_density_grid_spacing,
            params.kernel_support_radius,
            params.disable_particles_below,
            params.fluid_viscosity,
            params.boundary_viscosity,
            params.boundary_pressure_acceleration_weighting,
            params.boundary_rest_volume_weighting,
        );

        let constructor = Self {
            fluid,
            boundary,
            system_parameters,
            initial_time_steps_propagated,
            initial_current_time,
            _kernel_fn: std::marker::PhantomData,
            integrator,
            pressure_solver,
            neighbor_search,
        };
        // create simulation system
        Ok(constructor)
    }
}

macro_rules! create {
    ($params:expr, $scene:expr, $state:expr, $K:ty, $I:ty, $P:ty, $N:ty, $B:ty) => {{
        let constructor = SystemConstructor::<$K, $I, $P, $N, $B>::new($params, $scene, $state)?;
        let system = System::<$K, $I, $P, $N, $B>::new_boxed(constructor);
        Ok(system)
    }};
}

pub fn new_boxed_system3d(
    procs: &Procedures,
    params: &Parameters,
    scene: &Scene,
    state: Option<&str>,
) -> Result<Box<dyn SPHSystem>, SetupError> {
    // Nest macros to build the cartesian product without writing each combination
    macro_rules! with_boundary {
        ($K:ty, $I:ty, $P:ty, $N:ty) => {
            match procs.boundary_handling {
                BoundaryHandlingVariant::StaticSampleBoundary => {
                    create!(params, scene, state, $K, $I, $P, $N, StaticSampleBoundary)
                }
                BoundaryHandlingVariant::VolumeMapBoundary => {
                    create!(params, scene, state, $K, $I, $P, $N, VolumeMapBoundary)
                }
            }
        };
    }

    macro_rules! with_neighbor {
        ($K:ty, $I:ty, $P:ty) => {
            match procs.neighbor_search {
                NeighborSearchVariant::SpatialHashing => with_boundary!($K, $I, $P, SpatialHashing),
            }
        };
    }

    macro_rules! with_pressure {
        ($K:ty, $I:ty) => {
            match procs.pressure_solver {
                PressureSolverVariant::SESPH => with_neighbor!($K, $I, SESPH),
                PressureSolverVariant::SESPHwSplitting => with_neighbor!($K, $I, SESPHwSplitting),
                PressureSolverVariant::IISPH => with_neighbor!($K, $I, IISPH),
                PressureSolverVariant::IISPHwOST => with_neighbor!($K, $I, IISPHwOST),
            }
        };
    }

    macro_rules! with_integrator {
        ($K:ty) => {
            match procs.integration_scheme {
                IntegrationSchemeVariant::ExplicitEuler => with_pressure!($K, ExplicitEuler),
                // IntegrationSchemeVariant::ImplicitEuler => with_pressure!($K, ImplicitEuler),
                IntegrationSchemeVariant::EulerCromer => with_pressure!($K, EulerCromer),
                IntegrationSchemeVariant::Verlet => with_pressure!($K, Verlet),
                IntegrationSchemeVariant::TakePredicted => with_pressure!($K, TakePredicted),
            }
        };
    }

    match procs.kernel_function {
        KernelFnVariant::CubicBSpline3D => with_integrator!(CubicBSpline3D),
    }
}
