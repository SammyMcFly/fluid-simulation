/// Integration schemes
use crate::sph::sample::Fluid3D;

pub mod euler_cromer;
pub mod explicit_euler;
pub mod implicit_euler;
pub mod take_predicted;
pub mod verlet;

pub use euler_cromer::EulerCromer;
pub use explicit_euler::ExplicitEuler;
pub use implicit_euler::ImplicitEuler;
pub use take_predicted::TakePredicted;
pub use verlet::Verlet;

pub trait IntegrationScheme: Send + Sync {
    /// Advance positions and velocities by one time step.
    ///
    /// Contract: acceleration has already been computed before this is called.
    fn integrate(&mut self, fluid: &mut Fluid3D, dt: f64);
}
