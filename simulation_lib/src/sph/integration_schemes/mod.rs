//! Integration schemes
pub mod euler_cromer;
pub mod explicit_euler;
pub mod take_predicted;
pub mod verlet;

pub use euler_cromer::EulerCromer;
pub use explicit_euler::ExplicitEuler;
pub use take_predicted::TakePredicted;
pub use verlet::Verlet;

use crate::sph::fluid::Fluid;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub enum IntegrationSchemeVariant {
    ExplicitEuler,
    EulerCromer,
    Verlet,
    TakePredicted,
}

pub trait IntegrationScheme: Send + Sync + Default + Clone {
    /// Number of `Fluid::integrator_position_slots` this scheme requires.
    const POSITION_SLOTS: usize = 0;

    /// Number of `Fluid::integrator_velocity_slots` this scheme requires.
    const VELOCITY_SLOTS: usize = 0;

    /// Advance positions and velocities by one time step.
    ///
    /// Contract: acceleration has already been computed before this is called.
    fn integrate(&mut self, fluid: &mut Fluid, dt: f64);
}
