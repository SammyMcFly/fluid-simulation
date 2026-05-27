/// State equation SPH (SESPH) or weakly compressible SPH (WCSPH) pressure solver
#[cfg(feature = "parallelized_sph")]
use rayon::prelude::*;

use crate::for_each;
use crate::sph::pressure_solver::PressureSolver;
use crate::sph::kernel::KernelFn;
use crate::sample::{Fluid3D, Boundary3D, Len, Positional};
use crate::sph::SystemParameters;
use crate::sph::CurrentSystemProperties;

use crate::sph::pressure_solver::{set_pred_vel_by_applying_acc, add_pressure_acceleration};
use crate::sph::direction;

pub struct SESPHwSplitting {
    stiffness: f64,
    density_pred: Vec<f64>,
}

impl PressureSolver for SESPHwSplitting {
    fn solve_and_add_acceleration<K: KernelFn>(
        &mut self,
        fluid: &mut Fluid3D,
        boundary: &Boundary3D,
        params: &SystemParameters,
        _properties: &mut CurrentSystemProperties,
    ) {
        self.resize_scratch(fluid.len());
        // perform splitting step conditionally
        set_pred_vel_by_applying_acc(fluid, params, false);
        self.calc_predicted_density::<K>(fluid, boundary, params,);
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
            params,
            false,
            false,
        );
    }
}

impl SESPHwSplitting {
    pub fn new(stiffness: f64) -> Self {
        Self {
            stiffness,
            density_pred: Vec::new(),
        }
    }

    pub fn resize_scratch(&mut self, len: usize) {
        self.density_pred.resize(len, 0.0);
    }

    fn calc_predicted_density<K: KernelFn>(
        &mut self,
        fluid: &mut Fluid3D,
        boundary: &Boundary3D,
        params: &SystemParameters,
    ) {
        for_each!(
            mut [self.density_pred],
            ref [
                pos_now = fluid.position,
                vel_pred = fluid.velocity_pred,
                mass = fluid.mass,
                neighbors = fluid.neighbors,
                boundary_neighbors = fluid.boundary_neighbors,
            ],
            |id, density_pred| {
                // reset density
                let mut accu = 0.;
                // add density for every neighbor
                for &neighbor in &neighbors[id] {
                    let r_vec = direction(
                        &pos_now[neighbor],
                        &pos_now[id],
                    );
                    let dist = r_vec.norm();
                    accu += mass[neighbor]
                        * K::value(
                            dist,
                            params.smoothing_length,
                        )
                        + params.time_increment
                            * (vel_pred[id] - vel_pred[neighbor]).dot(&K::gradient(
                                &r_vec,
                                dist,
                                params.smoothing_length,
                            ));
                }
                // add density for every boundary neighbor (mirror mass of moving sample onto boundary sample)
                for &boundary_neighbor in &boundary_neighbors[id] {
                    let r_vec = direction(
                        boundary.pos_now(boundary_neighbor),
                        &pos_now[id],
                    );
                    let dist = r_vec.norm();
                    accu += *boundary.volume(boundary_neighbor)
                        * params.rest_density
                        * K::value(
                            dist,
                            params.smoothing_length,
                        )
                        + params.time_increment
                            * vel_pred[id]
                                .dot(&K::gradient(
                                    &r_vec,
                                    dist,
                                    params.smoothing_length,
                                ));
                }
                *density_pred = accu;
                // if cfg!(feature = "logging") {
                // debug!("density: {}", fluid.density());
                // }
            }
        );
    }
}