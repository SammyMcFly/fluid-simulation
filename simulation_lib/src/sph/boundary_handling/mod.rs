//! Boundary handling module
use nalgebra::{Isometry3, Matrix3, Point3, Quaternion, Translation3, UnitQuaternion, Vector3};
use serde::Deserialize;

use crate::{
    neighbor_search::NeighborSearch,
    render_info::BoundaryVisualization,
    setup::input::{DynamicBoundaryDef, StaticBoundaryDef},
    sph::kernel::KernelFn,
    utilities::triangle_mesh::MeshContainer,
};

mod static_sample_boundary;
mod volume_maps;

pub use static_sample_boundary::SampleBoundary;
pub use volume_maps::VolumeMaps;

#[derive(Debug, Deserialize)]
pub enum BoundaryHandlingVariant {
    SampleBoundary,
    VolumeMaps,
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
    fn checkpoint_state(&self) -> BoundaryCheckpoint;

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

    fn pos_now(&self, id: usize) -> &Point3<f64>;

    fn vel_now(&self, id: usize) -> &Vector3<f64>;

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

#[derive(Debug, Clone)]
struct RigidBodyMotion {
    mass: f64,
    /// Inverse inertia tensor in the local frame — constant over time.
    inertia_tensor_inv_body: Matrix3<f64>,
    center_of_mass: Point3<f64>,
    orientation: UnitQuaternion<f64>,
    linear_velocity: Vector3<f64>,
    angular_momentum: Vector3<f64>,
    force: Vector3<f64>,
    torque: Vector3<f64>,
    // // derived variables
    // inertia_tensor_inv_world: Option<Matrix3<f64>>,
    // angular_velocity: Option<Vector3<f64>>,
}

pub struct ForceOntoBoundary {
    pub id: usize,
    pub force: Vector3<f64>,
    pub force_location: Point3<f64>,
}

impl RigidBodyMotion {
    pub fn new(
        mass: f64,
        inertia_tensor_body: Matrix3<f64>,
        inertia_tensor_inv_body: Matrix3<f64>,
        center_of_mass: Point3<f64>,
        orientation: UnitQuaternion<f64>,
        linear_velocity: Vector3<f64>,
        angular_velocity: Vector3<f64>,
    ) -> Self {
        let r = orientation.to_rotation_matrix();
        let inertia_tensor_world = r.matrix() * inertia_tensor_body * r.matrix().transpose();
        Self {
            mass,
            inertia_tensor_inv_body,
            center_of_mass,
            orientation,
            linear_velocity,
            angular_momentum: inertia_tensor_world * angular_velocity,
            force: Vector3::zeros(),
            torque: Vector3::zeros(),
            // inertia_tensor_inv_world: None,
            // angular_velocity: Some(angular_velocity),
        }
    }

    /// Current rigid-body pose: body/local frame -> world frame.
    #[inline]
    pub fn pose(&self) -> Isometry3<f64> {
        Isometry3::from_parts(
            Translation3::from(self.center_of_mass.coords),
            self.orientation,
        )
    }

    /// World-space point -> body/local-space point.
    #[inline]
    pub fn world_to_local(&self, p_world: &Point3<f64>) -> Point3<f64> {
        self.pose().inverse_transform_point(p_world)
    }

    /// World-space point -> body/local-space point.
    #[inline]
    pub fn local_to_world(&self, p_world: &Point3<f64>) -> Point3<f64> {
        self.pose().transform_point(p_world)
    }

    /// Local-space direction/gradient -> world-space direction/gradient.
    #[inline]
    pub fn local_to_world_vector(&self, v_local: &Vector3<f64>) -> Vector3<f64> {
        self.pose().rotation.transform_vector(v_local)
    }

    /// Inverse inertia tensor in WORLD frame: I_world^-1 = R * I_body^-1 * R^T
    fn inertia_tensor_inv_world(&self) -> Matrix3<f64> {
        let r = self.orientation.to_rotation_matrix();
        r.matrix() * self.inertia_tensor_inv_body * r.matrix().transpose()
    }

    pub fn angular_velocity(&self) -> Vector3<f64> {
        self.inertia_tensor_inv_world() * self.angular_momentum
    }

    pub fn velocity_at_cm(&self) -> Vector3<f64> {
        self.linear_velocity
    }

    pub fn velocity_at_point(&self, p_world: &Point3<f64>) -> Vector3<f64> {
        self.linear_velocity
            + self
                .angular_velocity()
                .cross(&(p_world - self.center_of_mass))
    }

    pub fn reset_forces(&mut self) {
        self.force = Vector3::zeros();
        self.torque = Vector3::zeros();
    }

    pub fn add_force(&mut self, force: ForceOntoBoundary) {
        self.force += force.force;
        self.torque += (force.force_location - self.center_of_mass).cross(&force.force);
    }

    /// Performs time step by integrating with the Euler-Cromer time integration scheme
    pub fn step_forward_in_time(&mut self, dt: f64) {
        // Linear
        let linear_acceleration = self.force / self.mass;
        self.linear_velocity += linear_acceleration * dt;
        self.center_of_mass += self.linear_velocity * dt;

        // Angular: dL/dt = torque (exact, no correction term needed in world frame)
        self.angular_momentum += self.torque * dt;

        // Integrate orientation quaternion: dq/dt = 0.5 * (0, omega) * q
        let omega_quat = Quaternion::from_parts(0.0, self.angular_velocity());
        let q_dot = omega_quat * self.orientation.into_inner() * 0.5;
        let new_q = self.orientation.into_inner() + q_dot * dt;
        self.orientation = UnitQuaternion::from_quaternion(new_q);

        self.reset_forces();
    }

    pub fn checkpoint_state(&self) -> RigidBodyMotionState {
        RigidBodyMotionState {
            center_of_mass: self.center_of_mass,
            orientation: self.orientation,
            linear_velocity: self.linear_velocity,
            angular_momentum: self.angular_momentum,
            force: self.force,
            torque: self.torque,
        }
    }

    pub fn restore_from_checkpoint(&mut self, state: &RigidBodyMotionState) {
        self.center_of_mass = state.center_of_mass;
        self.orientation = state.orientation;
        self.linear_velocity = state.linear_velocity;
        self.angular_momentum = state.angular_momentum;
        self.force = state.force;
        self.torque = state.torque;
    }
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
    pub dynamic_states: Vec<Option<RigidBodyMotionState>>,
}

/// Serializable snapshot of a [`RigidBodyMotion`]'s time-dependent state.
///
/// Excludes [`RigidBodyMotion::mass`] and the local inverse inertia tensor, which are
/// constant for the lifetime of a boundary and are already reconstructed correctly
/// when the boundary is set up from the scene file.
#[derive(Debug, Clone, Copy)]
pub struct RigidBodyMotionState {
    pub center_of_mass: Point3<f64>,
    pub orientation: UnitQuaternion<f64>,
    pub linear_velocity: Vector3<f64>,
    pub angular_momentum: Vector3<f64>,
    /// Force accumulated for the *next* integration step.
    ///
    /// Populated by fluid–boundary pressure coupling during `System::update`, which
    /// runs *after* [`RigidBodyMotion::step_forward_in_time`] has reset it — i.e. right
    /// before the next step would consume it. Must be preserved, or resuming from this
    /// checkpoint would silently drop this pending force for one step.
    pub force: Vector3<f64>,
    /// See [`Self::force`].
    pub torque: Vector3<f64>,
}
