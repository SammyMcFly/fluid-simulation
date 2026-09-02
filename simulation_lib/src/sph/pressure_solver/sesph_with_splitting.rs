//! State equation SPH (SESPH) or weakly compressible SPH (WCSPH) pressure solver
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::fluid::{Fluid, Len};
use crate::for_each;
use crate::neighbor_search::NeighborList;
use crate::setup::input::Parameters;
use crate::sph::CurrentSystemProperties;
use crate::sph::SystemParameters;
use crate::sph::boundary_handling::{BoundaryHandling, RequestMode};
use crate::sph::kernel::KernelFn;
use crate::sph::pressure_solver::{PressureSolver, SolverMeasurementInfo};
use crate::sph::pressure_solver::{add_pressure_acceleration, set_pred_vel_by_applying_acc};
use crate::utilities::vector;

#[derive(Clone)]
pub struct SESPHwSplitting {
    stiffness: f64,
    density_pred: Vec<f64>,
}

impl PressureSolver for SESPHwSplitting {
    fn new(params: &Parameters) -> Self {
        Self {
            stiffness: params.stiffness,
            density_pred: Vec::new(),
        }
    }

    fn solve_and_add_acceleration<K: KernelFn>(
        &mut self,
        fluid: &mut Fluid,
        boundary: &mut impl BoundaryHandling,
        neighbor_list: &NeighborList,
        params: &SystemParameters,
        _properties: &mut CurrentSystemProperties,
    ) {
        self.resize_scratch(fluid.len());
        // perform splitting step conditionally
        set_pred_vel_by_applying_acc(fluid, params, false);
        self.calc_predicted_density::<K>(fluid, boundary, neighbor_list, params);
        // compute pressure
        {
            for_each!(
                mut [fluid.pressure],
                ref [
                    density_pred = self.density_pred,
                    mass = fluid.mass,
                ],
                |id, id_pressure| {
                    // select density
                    let id_volume = mass[id] / density_pred[id];
                    // calc pressure with state equation
                    *id_pressure = self.stiffness
                        * f64::max(params.rest_volume / id_volume - 1., 0.);
                    // #[cfg(feature = "logging")]
                    // tracing::debug!("pressure: {}", pressure);
                }
            );
        }
        // add pressure acceleration (compute from pressure)
        add_pressure_acceleration::<K>(None, fluid, boundary, neighbor_list, params, false, false);
    }

    fn measurement_info(&self) -> SolverMeasurementInfo {
        SolverMeasurementInfo {
            stiffness: self.stiffness,
            ..Default::default()
        }
    }
}

impl SESPHwSplitting {
    pub fn resize_scratch(&mut self, len: usize) {
        self.density_pred.resize(len, 0.0);
    }

    fn calc_predicted_density<K: KernelFn>(
        &mut self,
        fluid: &mut Fluid,
        boundary: &impl BoundaryHandling,
        neighbor_list: &NeighborList,
        params: &SystemParameters,
    ) {
        for_each!(
            mut [self.density_pred],
            ref [
                pos_now = fluid.position,
                vel_pred = fluid.velocity_pred,
                mass = fluid.mass,
                neighbors = neighbor_list,
                boundary = boundary,
            ],
            |id, density_pred| {
                // reset density
                let mut accu = 0.;
                // add density for every neighbor
                for &neighbor in neighbors.get_neighbors(id) {
                    let r_vec = vector(
                        &pos_now[neighbor],
                        &pos_now[id],
                    );
                    accu += mass[neighbor]
                        * K::kernel_function(
                            &r_vec,
                            params.kernel_support_radius,
                        )
                        + params.time_increment
                            * (vel_pred[id] - vel_pred[neighbor]).dot(&K::kernel_gradient(
                                &r_vec,
                                params.kernel_support_radius,
                            ));
                }
                // add density for every boundary neighbor (mirror mass of moving sample onto boundary sample)
                for b in boundary.iter() {
                    for &boundary_neighbor in b.get_neighbors(id, RequestMode::Normal) {
                        let r_vec = vector(
                            b.position(boundary_neighbor),
                            &pos_now[id],
                        );
                        accu += b.volume(boundary_neighbor)
                            * mass[id]/params.rest_volume
                            * K::kernel_function(
                                &r_vec,
                                params.kernel_support_radius,
                            )
                            + params.time_increment
                                * vel_pred[id]
                                    .dot(&K::kernel_gradient(
                                        &r_vec,
                                        params.kernel_support_radius,
                                    ));
                    }
                }
                *density_pred = accu;
                // #[cfg(feature = "logging")]
                // tracing::debug!("density: {}", fluid.density());
            }
        );
    }
}
