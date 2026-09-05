//! Explicit Euler integration scheme
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::for_each;
use crate::sph::fluid::Fluid;
use crate::sph::integration_schemes::IntegrationScheme;

#[derive(Default, Clone)]
pub struct ExplicitEuler;

impl IntegrationScheme for ExplicitEuler {
    fn integrate(&mut self, fluid: &mut Fluid, dt: f64) {
        for_each!(
            mut [fluid.position, fluid.velocity],
            ref [
                acceleration = fluid.acceleration,
            ],
            |id, id_pos_now, id_vel_now| {
                let pos_prev = *id_pos_now;
                let vel_prev = *id_vel_now;
                *id_pos_now = pos_prev + dt * vel_prev;
                *id_vel_now = vel_prev + dt * acceleration[id];
            }
        );
    }
}
