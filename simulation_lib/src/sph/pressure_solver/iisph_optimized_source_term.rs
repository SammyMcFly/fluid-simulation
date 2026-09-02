//! Implicit imcompressible SPH (SESPH) pressure solver with "optimized source term"
//!
//! # Limitation: no support for dynamic boundaries
//!
//! Unlike every other `PressureSolver` in this crate, `solve_and_add_acceleration`
//! runs TWO separate global pressure solves per call (EQS1, EQS2), each time
//! pressure is applied via `add_pressure_acceleration` with `custom_target == None`.
//! To function correctly with this solver, dynamic boundaries will need to be able
//! to update positions and velocities separately. Additionally, support for predicted
//! positions and velocities will need to be added.
//!
//! To prevent silent failures, the combination of this solver running with dynamic
//! boundaries is guarded from running.
use crate::for_each;
use crate::neighbor_search::NeighborList;
use crate::sph::CurrentSystemProperties;
use crate::sph::boundary_handling::{BoundaryHandling, RequestMode};
use crate::sph::fluid::{Fluid, Len};
use crate::sph::kernel::KernelFn;
use crate::sph::pressure_solver::iisph::{IISPH, TerminationCondition};
use crate::sph::pressure_solver::{PressureSolver, SolverMeasurementInfo};
use crate::sph::pressure_solver::{add_pressure_acceleration, set_pred_vel_by_applying_acc};
use crate::sph::setup::input::Parameters;
use crate::sph::{Outer, SystemParameters};
use crate::utilities::vector;

use nalgebra::Matrix3;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Clone)]
pub struct IISPHwOST {
    inner: IISPH,
}

impl PressureSolver for IISPHwOST {
    const SUPPORTS_DYNAMIC_BOUNDARIES: bool = false;

    fn new(params: &Parameters) -> Self {
        Self {
            inner: IISPH::new(params),
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
        self.inner.resize_scratch(fluid.len());

        // solve EQS1
        {
            // set predicted velocity by applying non-pressure acceleration
            set_pred_vel_by_applying_acc(fluid, params, false);
            // set source term
            self.inner
                .set_source_term_vde::<K>(fluid, boundary, neighbor_list, params);
            // self.set_source_term_vp(false);
            // solve pressure equation system
            self.inner.resolve_pressure_globally::<K>(
                fluid,
                boundary,
                neighbor_list,
                params,
                false,
                TerminationCondition::AfterIteration(3),
                // TerminationCondition::TargetDensityError(params.target_density_error),
                true,
            );
            // set acceleration to pressure acceleration with pressure from EQS1
            add_pressure_acceleration::<K>(
                None,
                fluid,
                boundary,
                neighbor_list,
                params,
                false,
                true,
            );
        }
        // println!("pressure acc eq1: {}", self.particles[200].acc());
        // set predicted velocity and positions
        {
            // set predicted velocity
            set_pred_vel_by_applying_acc(fluid, params, true);
            // set predicted position
            for_each!(
                mut [fluid.position_pred],
                ref [
                    pos_now = fluid.position,
                    vel_pred = fluid.velocity_pred,
                ],
                |id, id_position_pred| {
                    *id_position_pred = pos_now[id] + params.time_increment * vel_pred[id];
                }
            );
        }
        // solve EQS2
        {
            // set source term
            self.inner
                .set_source_term_vp::<K>(fluid, boundary, neighbor_list, params, false);
            // solve pressure equation system
            self.inner.resolve_pressure_globally::<K>(
                fluid,
                boundary,
                neighbor_list,
                params,
                false,
                TerminationCondition::TargetDensityError(self.inner.target_density_error),
                true,
            );
            // set acceleration to pressure acceleration with pressure from EQS2
            add_pressure_acceleration::<K>(
                None,
                fluid,
                boundary,
                neighbor_list,
                params,
                false,
                true,
            );
        }
        // write new positions and resampled velocities to predicted velocity and position field of each particle
        {
            for_each!(
                mut [fluid.position_pred, self.inner.pressure_acc_f],
                ref [
                    pos_now = fluid.position,
                    vel_pred = fluid.velocity_pred,
                    acceleration = fluid.acceleration,
                    volume = fluid.volume,
                    neighbors = neighbor_list,
                    boundary = boundary,
                    // s_f = self.s_f,
                    // a_ff = self.a_ff,
                ],
                |id, id_position_pred, id_pressure_acc_f| {
                    // calculate new position and store it intermediately
                    let new_pos = *id_position_pred
                        + params.time_increment.powi(2) * acceleration[id]; // TODO uncomment
                    // calculate and set velocity gradient (Jacobian) as predicted velocity
                    let mut jac_vel = Matrix3::zeros();
                    for &neighbor in neighbors.get_neighbors(id) {
                        let r_vec = vector(
                            &pos_now[neighbor],
                            &pos_now[id],
                        );
                        jac_vel -= volume[neighbor]
                            * (vel_pred[id] - vel_pred[neighbor]).outer(
                                &K::kernel_gradient(
                                    &r_vec,
                                    params.kernel_support_radius,
                                ),
                            );
                    }
                    for b in boundary.iter() {
                        for &boundary_neighbor in b.get_neighbors(id, RequestMode::Normal) {
                            let r_vec = vector(
                                b.position(boundary_neighbor),
                                &pos_now[id],
                            );
                            jac_vel -= b.volume(boundary_neighbor)
                                * (vel_pred[id]
                                    - b.velocity(boundary_neighbor))
                                .outer(&K::kernel_gradient(
                                    &r_vec,
                                    params.kernel_support_radius,
                                ));
                        }
                    }
                    // calculate new velocity and intermediately store it as pressure_acc_f to avoid race condition on .vel().pred()
                    // particle.pressure_acc_f[id] = vel_pred[id] + jac_vel*(new_pos - pos_pred[id]); // original "optimized source term" approach
                    // particle.pressure_acc_f[id] = vel_pred[id] + jac_vel*(new_pos - pos_pred[id]) + params.time_increment*particle.acc(); // TODO test
                    // particle.pressure_acc_f[id] = vel_pred[id]; // TODO test
                    *id_pressure_acc_f =
                        vel_pred[id] + params.time_increment * acceleration[id]; // DFSPH approach
                    // store new position in predicted position
                    *id_position_pred = new_pos;
                }
            );
            // move velocity from pressure_acc_f to predicted velocity
            for_each!(
                mut [fluid.velocity_pred],
                ref [
                    pressure_acc_f = self.inner.pressure_acc_f,
                ],
                |id, id_vel_pred| {
                    *id_vel_pred = pressure_acc_f[id];
                }
            );
        }
        // pressure acceleration is applied to particle movement implicitly
        // resulting in the velocity and position predictions
    }

    fn measurement_info(&self) -> SolverMeasurementInfo {
        SolverMeasurementInfo {
            target_density_error: self.inner.target_density_error,
            solver_iterations: self.inner.last_solver_iterations,
            relaxation_factor: self.inner.relaxation_factor,
            predicted_density_error: self.inner.predicted_density_error,
            ..Default::default()
        }
    }
}
