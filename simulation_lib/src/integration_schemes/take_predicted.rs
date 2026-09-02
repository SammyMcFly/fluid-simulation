//! Integration scheme that takes the predicted position and velocity as new current position and velocity
use nalgebra::Vector3;

use crate::fluid::Fluid;
use crate::integration_schemes::IntegrationScheme;

#[derive(Default, Clone)]
pub struct TakePredicted;

impl IntegrationScheme for TakePredicted {
    fn integrate(&mut self, fluid: &mut Fluid, _dt: f64) {
        fluid.rotate_position();
        fluid.rotate_velocity();
    }
}
