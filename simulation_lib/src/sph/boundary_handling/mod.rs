//! Boundary handling module
mod static_sample_boundary;
mod volume_map_boundary;

pub use rigid_body_motion::{
    RigidBodyMotion, RigidBodyMotionCheckpoint, SerRigidBodyMotionCheckpoint,
};
pub use static_sample_boundary::StaticSampleBoundary;
pub use volume_map_boundary::VolumeMapBoundary;

use crate::{
    neighbor_search::NeighborSearch,
    render_info::BoundaryVisualization,
    sph::kernel::KernelFn,
    sph::setup::input::{DynamicBoundaryDef, StaticBoundaryDef},
    utilities::triangle_mesh::MeshContainer,
};
mod rigid_body_motion;

use bincode::{Decode, Encode};
use nalgebra::{Point3, Vector3};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub enum BoundaryHandlingVariant {
    StaticSampleBoundary,
    VolumeMapBoundary,
}

pub trait BoundaryHandling: Send + Sync + Clone {
    fn new() -> Self;

    fn is_empty(&self) -> bool;

    fn add_static_boundary(
        &mut self,
        mesh: &mut MeshContainer,
        boundary: &StaticBoundaryDef,
        rest_density_grid_spacing: f64,
        kernel_support_radius: f64,
    );

    fn add_dynamic_boundary(
        &mut self,
        mesh: &mut MeshContainer,
        boundary: &DynamicBoundaryDef,
        rest_density_grid_spacing: f64,
        kernel_support_radius: f64,
    );

    fn initialize<K: KernelFn>(
        &mut self,
        neighbor_search: &mut impl NeighborSearch,
        kernel_support_radius: f64,
        boundary_rest_volume_weighting: f64,
    );

    fn find_boundary_samples(
        &mut self,
        neighbor_search: &mut impl NeighborSearch,
        within_range: f64,
        positions: &[Point3<f64>],
        rest_density_grid_spacing: f64,
    );

    fn iter(&self) -> impl Iterator<Item = &dyn Boundary> + '_;

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut dyn Boundary> + '_;

    fn add_force_onto_boundary(&mut self, force: ForceOntoBoundary);

    fn step_forward_in_time(&mut self, dt: f64);

    fn get_fluid_depth(&self, fluid_volume: f64) -> f64;

    fn get_visualization(&self, selector: &BoundaryVisualization) -> BoundaryVisualization;

    /// Captures the dynamic state of all boundaries for later restoration via
    /// [`Self::restore_from_checkpoint`].
    fn get_checkpoint(&self) -> BoundaryCheckpoint;

    /// Restores the dynamic state of all boundaries from a previously captured
    /// [`BoundaryCheckpointState`].
    ///
    /// The static geometry (sample count, mesh) must already match; this only
    /// overwrites position, velocity and rigid-body state. Also recomputes
    /// `position`/`velocity` from the restored rigid-body pose so both stay
    /// consistent.
    fn restore_from_checkpoint(&mut self, state: &BoundaryCheckpoint);
}

pub trait Boundary: Send + Sync {
    fn get_neighbors(&self, id: usize, mode: RequestMode) -> &[usize];

    fn position(&self, id: usize) -> &Point3<f64>;

    fn velocity(&self, id: usize) -> &Vector3<f64>;

    fn volume(&self, id: usize) -> f64;

    fn is_dynamic(&self) -> bool {
        self.center_of_mass().is_some()
    }

    fn add_acceleration(&mut self, acceleration: Vector3<f64>);

    fn center_of_mass(&self) -> Option<Point3<f64>>;
}

#[derive(Debug, Clone, Copy, Default)]
pub enum RequestMode {
    #[default]
    Normal,
    ViscosityAcceleration,
}

pub struct ForceOntoBoundary {
    pub id: usize,
    pub force: Vector3<f64>,
    pub force_location: Point3<f64>,
}

/// Snapshot of the dynamic state of all boundaries in a [`BoundaryHandling`]
/// implementation, sufficient to resume rigid-body motion and fluid–boundary
/// coupling from a specific point in time.
///
/// Static boundaries contribute `None`: their geometry, sample positions and volume
/// never change, so nothing needs to be captured for them.
#[derive(Debug, Clone, Default)]
pub struct BoundaryCheckpoint {
    /// One entry per boundary, in the same order as the boundaries are stored
    /// internally (matching [`BoundaryHandling::iter`]).
    pub dynamic_states: Vec<Option<RigidBodyMotionCheckpoint>>,
}

/// Serializable counterpart to [`BoundaryCheckpoint`].
#[derive(Debug, Clone, Default, Serialize, Deserialize, Encode, Decode)]
pub struct SerBoundaryCheckpoint {
    pub dynamic_states: Vec<Option<SerRigidBodyMotionCheckpoint>>,
}

impl From<BoundaryCheckpoint> for SerBoundaryCheckpoint {
    fn from(s: BoundaryCheckpoint) -> Self {
        Self {
            dynamic_states: s
                .dynamic_states
                .into_iter()
                .map(|d| d.map(Into::into))
                .collect(),
        }
    }
}

impl From<SerBoundaryCheckpoint> for BoundaryCheckpoint {
    fn from(s: SerBoundaryCheckpoint) -> Self {
        Self {
            dynamic_states: s
                .dynamic_states
                .into_iter()
                .map(|d| d.map(Into::into))
                .collect(),
        }
    }
}
