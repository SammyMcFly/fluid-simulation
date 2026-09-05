//! State equation SPH (SESPH) or weakly compressible SPH (WCSPH) pressure solver
use crate::for_each;
use crate::neighbor_search::NeighborList;
use crate::sph::CurrentSystemProperties;
use crate::sph::SystemParameters;
use crate::sph::boundary_handling::BoundaryHandling;
use crate::sph::fluid::Fluid;
use crate::sph::kernel::KernelFn;
use crate::sph::pressure_solver::add_pressure_acceleration;
use crate::sph::pressure_solver::{PressureSolver, SolverMeasurementInfo};
use crate::sph::setup::input::Parameters;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Clone)]
pub struct SESPH {
    stiffness: f64,
}

impl PressureSolver for SESPH {
    fn new(params: &Parameters) -> Self {
        Self {
            stiffness: params.stiffness,
        }
    }

    // Calculate and update pressure for all particles for the current point in time.
    ///
    /// Function uses a state equation to calculate the pressure locally.
    fn solve_and_add_acceleration<K: KernelFn>(
        &mut self,
        fluid: &mut Fluid,
        boundary: &mut impl BoundaryHandling,
        neighbor_list: &NeighborList,
        params: &SystemParameters,
        _properties: &mut CurrentSystemProperties,
    ) {
        {
            for_each!(
                mut [fluid.pressure],
                ref [
                    volume = fluid.volume,
                ],
                |id, id_pressure| {
                    // select density
                    let id_volume = volume[id];
                    // calc pressure with state equation
                    *id_pressure = self.stiffness
                        * f64::max(params.rest_volume / id_volume - 1., 0.);
                    // #[cfg(feature = "logging")]
                    // tracing::debug!("pressure: {}", pressure);
                }
            );
        }
        // add pressure acceleration (compute from pressure)
        add_pressure_acceleration::<K>(
            None,
            fluid,
            boundary,
            neighbor_list,
            params,
            false,
            false,
            true,
        );
    }

    fn measurement_info(&self) -> SolverMeasurementInfo {
        SolverMeasurementInfo {
            stiffness: self.stiffness,
            ..Default::default()
        }
    }
}
