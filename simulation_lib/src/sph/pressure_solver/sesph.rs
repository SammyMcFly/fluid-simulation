/// State equation SPH (SESPH) or weakly compressible SPH (WCSPH) pressure solver
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::for_each;
use crate::sph::pressure_solver::{PressureSolver, SolverMeasurementInfo};
use crate::sph::kernel::KernelFn;
use crate::sample::{Fluid3D, Boundary3D};
use crate::sph::SystemParameters;
use crate::sph::CurrentSystemProperties;
use crate::sph::pressure_solver::add_pressure_acceleration;
use crate::neighbor_search::NeighborList;

pub struct SESPH {
    stiffness: f64,
}

impl PressureSolver for SESPH {
    // Calculate and update pressure for all particles for the current point in time.
    ///
    /// Function uses a state equation to calculate the pressure locally.
    fn solve_and_add_acceleration<K: KernelFn>(
        &mut self,
        fluid: &mut Fluid3D,
        boundary: &Boundary3D,
        neighbors: &NeighborList,
        boundary_neighbors: &NeighborList,
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
                    // if cfg!(feature = "logging") {
                    //     debug!("pressure: {}", pressure);
                    // }
                }
            );
        }
        // add pressure acceleration (compute from pressure)
        add_pressure_acceleration::<K>(
            None,
            fluid,
            boundary,
            neighbors,
            boundary_neighbors,
            params,
            false,
            false,
        );
    }

    fn measurement_info(&self) -> SolverMeasurementInfo {
        SolverMeasurementInfo {
            stiffness: self.stiffness,
            ..Default::default()
        }
    }
}

impl SESPH {
    pub fn new(stiffness: f64) -> Self {
        Self {
            stiffness,
        }
    }
}