//! Rigid-body dynamics for two-way coupled dynamic boundaries.
//!
//! [`RigidBodyMotion`] integrates the pose (position + orientation) and
//! momentum (linear + angular) of a boundary that is free to move under
//! forces/torques accumulated from fluid-boundary pressure and viscosity
//! coupling (see [`super::Boundary::add_acceleration`] and
//! [`super::BoundaryHandling::add_force_onto_boundary`]).

use bincode::{Decode, Encode};
use nalgebra::{Isometry3, Matrix3, Point3, Quaternion, Translation3, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};

use super::ForceOntoBoundary;

/// Rigid-body state and Euler-Cromer time integration for a single dynamic
/// boundary.
///
/// Fields are private; all state is accessed through the methods below, so
/// that derived quantities ([`Self::angular_velocity`], the world-frame
/// inverse inertia tensor) can be safely cached and kept consistent — see
/// [`Self::update_derived`].
#[derive(Debug, Clone)]
pub struct RigidBodyMotion {
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
            inertia_tensor_inv_world: Matrix3::identity(),
            angular_velocity: Vector3::zeros(),
        };
        motion.update_derived_variables();
        motion
    }

    /// Recomputes `inertia_tensor_inv_world` and `angular_velocity` from the
    /// current `orientation`/`angular_momentum`.
    ///
    /// Must be called after every change to either field — currently in
    /// `new`, `step_forward_in_time` and `restore_from_checkpoint`.
    fn update_derived_variables(&mut self) {
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

    /// Total mass of the rigid body.
    #[inline]
    pub fn mass(&self) -> f64 {
        self.mass
    }

    /// Center of mass of the rigid body.
    #[inline]
    pub fn center_of_mass(&self) -> Point3<f64> {
        self.center_of_mass
    }

    /// World-space point -> body/local-space point.
    #[inline]
    pub fn world_to_local(&self, p_world: &Point3<f64>) -> Point3<f64> {
        self.pose().inverse_transform_point(p_world)
    }

    /// Body/local-space point -> world-space point.
    #[inline]
    pub fn local_to_world(&self, p_world: &Point3<f64>) -> Point3<f64> {
        self.pose().transform_point(p_world)
    }

    /// Local-space direction/gradient -> world-space direction/gradient.
    #[inline]
    pub fn local_to_world_vector(&self, v_local: &Vector3<f64>) -> Vector3<f64> {
        self.pose().rotation.transform_vector(v_local)
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
        self.update_derived_variables();
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
        self.update_derived_variables();
    }
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

/// Serializable counterpart to [`RigidBodyMotionCheckpoint`].
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Asserts that the cached derived fields (`inertia_tensor_inv_world`,
    /// `angular_velocity`) are exactly what `update_derived` would compute
    /// from the current `orientation`/`angular_momentum`/
    /// `inertia_tensor_inv_body` — i.e. the cache is not stale.
    ///
    /// This is the central regression guard for the caching bug class we
    /// found earlier (using the pre-integration `angular_velocity` for the
    /// quaternion update, but forgetting to refresh the cache with the
    /// *new* orientation afterwards): rather than hand-deriving expected
    /// physical values for every call site, it checks the cache is always
    /// self-consistent with its own inputs.
    fn assert_derived_cache_is_consistent(m: &RigidBodyMotion) {
        let r = m.orientation.to_rotation_matrix();
        let expected_inv_world = r.matrix() * m.inertia_tensor_inv_body * r.matrix().transpose();
        let expected_angular_velocity = expected_inv_world * m.angular_momentum;
        assert!(
            (m.inertia_tensor_inv_world - expected_inv_world).norm() < 1e-12,
            "inertia_tensor_inv_world cache is stale"
        );
        assert!(
            (m.angular_velocity - expected_angular_velocity).norm() < 1e-12,
            "angular_velocity cache is stale"
        );
    }

    #[test]
    fn new_leaves_no_stale_placeholder_values() {
        // `new`'s placeholder `inertia_tensor_inv_world: Matrix3::identity()`
        // must be immediately overwritten by `update_derived` — checking the
        // private field directly, which no external test can do (there is
        // no public getter for it, only its effect via `angular_velocity()`).
        let motion = RigidBodyMotion::new(
            2.0,
            Matrix3::from_diagonal(&Vector3::new(1., 2., 3.)),
            Matrix3::from_diagonal(&Vector3::new(1., 0.5, 1. / 3.)),
            Point3::origin(),
            UnitQuaternion::identity(),
            Vector3::zeros(),
            Vector3::new(1., 1., 1.),
        );

        assert_ne!(motion.inertia_tensor_inv_world, Matrix3::identity());
        assert_derived_cache_is_consistent(&motion);
    }

    #[test]
    fn update_derived_recomputes_cache_from_scratch() {
        // Isolates `update_derived` from `new`/`step_forward_in_time`
        // entirely, by constructing via a direct struct literal (only
        // possible with in-module field access) with deliberately wrong
        // placeholder cache values.
        let mut motion = RigidBodyMotion {
            mass: 1.0,
            inertia_tensor_inv_body: Matrix3::from_diagonal(&Vector3::new(1., 2., 3.)),
            center_of_mass: Point3::origin(),
            orientation: UnitQuaternion::from_axis_angle(
                &Vector3::z_axis(),
                std::f64::consts::FRAC_PI_2,
            ),
            linear_velocity: Vector3::zeros(),
            angular_momentum: Vector3::new(1., 1., 1.),
            force: Vector3::zeros(),
            torque: Vector3::zeros(),
            inertia_tensor_inv_world: Matrix3::zeros(), // deliberately wrong
            angular_velocity: Vector3::zeros(),         // deliberately wrong
        };

        motion.update_derived_variables();

        assert_derived_cache_is_consistent(&motion);
    }

    #[test]
    fn derived_cache_stays_consistent_after_step_forward_in_time() {
        // Anisotropic inertia + nonzero torque so orientation actually
        // changes meaningfully each step — the exact scenario in which the
        // "stale cache" bug class would surface.
        let mut motion = RigidBodyMotion::new(
            1.0,
            Matrix3::from_diagonal(&Vector3::new(1., 2., 3.)),
            Matrix3::from_diagonal(&Vector3::new(1., 0.5, 1. / 3.)),
            Point3::origin(),
            UnitQuaternion::identity(),
            Vector3::zeros(),
            Vector3::new(0.1, 0.2, 0.3),
        );

        for _ in 0..5 {
            motion.add_force(ForceOntoBoundary {
                id: 0,
                force: Vector3::new(0., 0., 0.),
                force_location: Point3::new(1., 0., 0.),
            });
            motion.torque += Vector3::new(0.5, 0.3, 0.1); // synthetic nonzero torque
            motion.step_forward_in_time(0.01);
            assert_derived_cache_is_consistent(&motion);
        }
    }

    #[test]
    fn derived_cache_stays_consistent_after_restore_from_checkpoint() {
        let mut motion = RigidBodyMotion::new(
            1.0,
            Matrix3::from_diagonal(&Vector3::new(1., 2., 3.)),
            Matrix3::from_diagonal(&Vector3::new(1., 0.5, 1. / 3.)),
            Point3::origin(),
            UnitQuaternion::identity(),
            Vector3::zeros(),
            Vector3::zeros(),
        );

        motion.restore_from_checkpoint(&RigidBodyMotionCheckpoint {
            center_of_mass: Point3::new(1., 2., 3.),
            orientation: UnitQuaternion::from_axis_angle(
                &Vector3::z_axis(),
                std::f64::consts::FRAC_PI_2,
            ),
            linear_velocity: Vector3::zeros(),
            angular_momentum: Vector3::new(1., 1., 0.),
            force: Vector3::zeros(),
            torque: Vector3::zeros(),
        });

        assert_derived_cache_is_consistent(&motion);
    }
}
