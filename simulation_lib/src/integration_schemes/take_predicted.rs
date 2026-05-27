/// Integration scheme that takes the predicted position and velocity as new current position and velocity
use crate::integration_schemes::IntegrationScheme;
use crate::sample::Fluid3D;
use nalgebra::Vector3;

pub struct TakePredicted {
    position_pred: Vec<Vector3<f64>>,
    velocity_pred: Vec<Vector3<f64>>,
}

impl IntegrationScheme for TakePredicted {
    fn integrate(&mut self, fluid: &mut Fluid3D, _dt: f64) {
        fluid.rotate_position();
        fluid.rotate_velocity();
    }
}
