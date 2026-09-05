//! Integration scheme that takes the predicted position and velocity as new current position and velocity
use crate::sph::fluid::Fluid;
use crate::sph::integration_schemes::IntegrationScheme;

#[derive(Default, Clone)]
pub struct TakePredicted;

impl IntegrationScheme for TakePredicted {
    const POSITION_SLOTS: usize = 1;
    const VELOCITY_SLOTS: usize = 1;

    fn integrate(&mut self, fluid: &mut Fluid, _dt: f64) {
        debug_assert_eq!(
            fluid.integrator_position_slots.len(),
            Self::POSITION_SLOTS,
            "did `resize_slots` get called with `TakePredicted::POSITION_SLOTS`?"
        );
        debug_assert_eq!(
            fluid.integrator_velocity_slots.len(),
            Self::VELOCITY_SLOTS,
            "did `resize_slots` get called with `TakePredicted::VELOCITY_SLOTS`?"
        );
        std::mem::swap(&mut fluid.position, &mut fluid.integrator_position_slots[0]);
        std::mem::swap(&mut fluid.velocity, &mut fluid.integrator_velocity_slots[0]);
    }
}
