//! Integration schemes
use crate::fluid::Fluid;
use serde::Deserialize;

pub mod euler_cromer;
pub mod explicit_euler;
// pub mod implicit_euler;
pub mod take_predicted;
pub mod verlet;

pub use euler_cromer::EulerCromer;
pub use explicit_euler::ExplicitEuler;
// pub use implicit_euler::ImplicitEuler;
pub use take_predicted::TakePredicted;
pub use verlet::Verlet;

#[derive(Debug, Deserialize)]
pub enum IntegrationSchemeVariant {
    ExplicitEuler,
    // ImplicitEuler,
    EulerCromer,
    Verlet,
    TakePredicted,
}

pub trait IntegrationScheme: Send + Sync + Default + Clone {
    /// Advance positions and velocities by one time step.
    ///
    /// Contract: acceleration has already been computed before this is called.
    fn integrate(&mut self, fluid: &mut Fluid, dt: f64);
}
