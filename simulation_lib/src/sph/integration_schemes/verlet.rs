//! Verlet integration scheme
use crate::for_each;
use crate::sph::fluid::Fluid;
use crate::sph::integration_schemes::IntegrationScheme;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

#[derive(Default, Clone)]
pub struct Verlet;

impl IntegrationScheme for Verlet {
    const POSITION_SLOTS: usize = 1;

    fn integrate(&mut self, fluid: &mut Fluid, dt: f64) {
        debug_assert_eq!(
            fluid.integrator_position_slots.len(),
            Self::POSITION_SLOTS,
            "did `resize_slots` get called with `Verlet::POSITION_SLOTS`?"
        );
        // slot 0 = x(t-1), i.e. position one step before the current
        // `fluid.position` = x(t)
        for_each!(
            mut [fluid.position, fluid.velocity, fluid.integrator_position_slots[0]],
            ref [
                acceleration = fluid.acceleration,
            ],
            |id, id_pos_now, id_vel_now, id_pos_prev| {
                let new_pos =
                    *id_pos_now + (*id_pos_now - *id_pos_prev) + dt.powi(2) * acceleration[id];
                *id_vel_now = (new_pos - *id_pos_now) / dt;
                *id_pos_prev = *id_pos_now;
                *id_pos_now = new_pos;
            }
        );
    }
}
