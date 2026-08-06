//! Euler-Cromer integration scheme
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::fluid::Fluid3D;
use crate::for_each;
use crate::integration_schemes::IntegrationScheme;

#[derive(Default, Clone)]
pub struct EulerCromer;

impl IntegrationScheme for EulerCromer {
    fn integrate(&mut self, fluid: &mut Fluid3D, dt: f64) {
        // Rotate buffers
        fluid.rotate_position();
        fluid.rotate_velocity();
        // position = old position_prev (will be overwritten), position_prev = old position
        for_each!(
            mut [fluid.position, fluid.velocity],
            ref [
                pos_prev = fluid.position_prev,   // = old "position"
                vel_prev = fluid.velocity_prev,   // = old "velocity"
                acceleration = fluid.acceleration,
            ],
            |id, id_pos_now, id_vel_now| {
                // update velocities
                *id_vel_now = vel_prev[id] + dt * acceleration[id];
                // update positions
                *id_pos_now = pos_prev[id] + dt * *id_vel_now;
            }
        );
    }
}
