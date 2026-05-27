/// Verlet integration scheme
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use crate::for_each;
use crate::sph::integration_schemes::IntegrationScheme;
use crate::sph::sample::Fluid3D;

pub struct Verlet;

impl IntegrationScheme for Verlet {
    fn integrate(&mut self, fluid: &mut Fluid3D, dt: f64) {
        // Rotate buffers
        fluid.rotate_position();
        fluid.rotate_velocity();
        // position = old position_prev (will be overwritten), position_prev = old position
        for_each!(
            mut [fluid.position, fluid.velocity],
            ref [
                pos_prev = fluid.position_prev,   // = old "position"
                pos_pred = fluid.position_pred,   // = old "position_prev"
                acceleration = fluid.acceleration,
            ],
            |id, id_pos_now, id_vel_now| {
                // update positions
                *id_pos_now =  2.0 * pos_prev[id]
                    - pos_pred[id]
                    + dt.powi(2) * acceleration[id];
                // update velocities
                *id_vel_now = (*id_pos_now - pos_prev[id])
                    / dt;
            }
        );
    }
}
