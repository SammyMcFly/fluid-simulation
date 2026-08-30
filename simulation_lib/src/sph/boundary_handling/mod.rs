//! Boundary handling module
use bincode::{Decode, Encode};
use nalgebra::{Isometry3, Matrix3, Point3, Quaternion, Translation3, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};

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
    // Derived variables
    inertia_tensor_inv_world: Matrix3<f64>,
    angular_velocity: Vector3<f64>,
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
        let mut motion = Self {
            mass,
            inertia_tensor_inv_body,
            center_of_mass,
            orientation,
            linear_velocity,
            angular_momentum: inertia_tensor_world * angular_velocity,
            force: Vector3::zeros(),
            torque: Vector3::zeros(),
            // Placeholder values, immediately overwritten by `update_derived`
            // below — avoids duplicating the correct initialization logic.
            inertia_tensor_inv_world: Matrix3::identity(),
            angular_velocity: Vector3::zeros(),
        };
        motion.update_derived();
        motion
    }

    /// Recomputes `inertia_tensor_inv_world` and `angular_velocity` from the
    /// current `orientation`/`angular_momentum`.
    ///
    /// Must be called after every change to either field — currently in
    /// `new`, `step_forward_in_time` and `restore_from_checkpoint` — since
    /// `inertia_tensor_inv_world()`/`angular_velocity()` return the cached
    /// values directly without checking staleness.
    fn update_derived(&mut self) {
        let r = self.orientation.to_rotation_matrix();
        self.inertia_tensor_inv_world =
            r.matrix() * self.inertia_tensor_inv_body * r.matrix().transpose();
        self.angular_velocity = self.inertia_tensor_inv_world * self.angular_momentum;
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
        self.inertia_tensor_inv_world
    }

    pub fn angular_velocity(&self) -> Vector3<f64> {
        self.angular_velocity
    }

    pub fn velocity_at_cm(&self) -> Vector3<f64> {
        self.linear_velocity
    }

    pub fn velocity_at_point(&self, p_world: &Point3<f64>) -> Vector3<f64> {
        self.linear_velocity
            + self
                .angular_velocity
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
        self.angular_velocity = self.inertia_tensor_inv_world * self.angular_momentum;

        // Integrate orientation quaternion: dq/dt = 0.5 * (0, omega) * q
        let omega_quat = Quaternion::from_parts(0.0, self.angular_velocity);
        let q_dot = omega_quat * self.orientation.into_inner() * 0.5;
        let new_q = self.orientation.into_inner() + q_dot * dt;
        self.orientation = UnitQuaternion::from_quaternion(new_q);

        self.reset_forces();
        // `orientation` and `angular_momentum` both just changed above —
        // refresh the cached derived quantities before they're read again.
        self.update_derived();
    }

    pub fn get_checkpoint(&self) -> RigidBodyMotionCheckpoint {
        RigidBodyMotionCheckpoint {
            center_of_mass: self.center_of_mass,
            orientation: self.orientation,
            linear_velocity: self.linear_velocity,
            angular_momentum: self.angular_momentum,
            force: self.force,
            torque: self.torque,
        }
    }

    pub fn restore_from_checkpoint(&mut self, state: &RigidBodyMotionCheckpoint) {
        self.center_of_mass = state.center_of_mass;
        self.orientation = state.orientation;
        self.linear_velocity = state.linear_velocity;
        self.angular_momentum = state.angular_momentum;
        self.force = state.force;
        self.torque = state.torque;
        // `orientation`/`angular_momentum` were just overwritten from the
        // checkpoint — refresh the cache to match.
        self.update_derived();
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
    pub dynamic_states: Vec<Option<RigidBodyMotionCheckpoint>>,
}

/// Snapshot of a [`RigidBodyMotion`]'s time-dependent state.
///
/// Excludes [`RigidBodyMotion::mass`] and the local inverse inertia tensor, which are
/// constant for the lifetime of a boundary and are already reconstructed correctly
/// when the boundary is set up from the scene file.
#[derive(Debug, Clone, Copy)]
pub struct RigidBodyMotionCheckpoint {
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

/// Serializable counterpart to [`RigidBodyMotionState`], for persisting a boundary's
/// dynamic state to disk (e.g. via [`crate::sph::Checkpoint`]'s serializable form).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Encode, Decode)]
pub struct SerRigidBodyMotionCheckpoint {
    pub center_of_mass: [f64; 3],
    /// Quaternion as (i, j, k, w).
    pub orientation: [f64; 4],
    pub linear_velocity: [f64; 3],
    pub angular_momentum: [f64; 3],
    pub force: [f64; 3],
    pub torque: [f64; 3],
}

impl From<RigidBodyMotionCheckpoint> for SerRigidBodyMotionCheckpoint {
    fn from(s: RigidBodyMotionCheckpoint) -> Self {
        let q = s.orientation.into_inner();
        Self {
            center_of_mass: s.center_of_mass.into(),
            orientation: [q.i, q.j, q.k, q.w],
            linear_velocity: s.linear_velocity.into(),
            angular_momentum: s.angular_momentum.into(),
            force: s.force.into(),
            torque: s.torque.into(),
        }
    }
}

impl From<SerRigidBodyMotionCheckpoint> for RigidBodyMotionCheckpoint {
    fn from(s: SerRigidBodyMotionCheckpoint) -> Self {
        Self {
            center_of_mass: Point3::from(s.center_of_mass),
            orientation: UnitQuaternion::from_quaternion(Quaternion::new(
                s.orientation[3], // w
                s.orientation[0], // i
                s.orientation[1], // j
                s.orientation[2], // k
            )),
            linear_velocity: Vector3::from(s.linear_velocity),
            angular_momentum: Vector3::from(s.angular_momentum),
            force: Vector3::from(s.force),
            torque: Vector3::from(s.torque),
        }
    }
}
