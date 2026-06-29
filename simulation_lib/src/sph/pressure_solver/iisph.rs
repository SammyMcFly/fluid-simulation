/// Implicit imcompressible SPH (SESPH) pressure self
use nalgebra::Vector3;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "logging")]
use tracing::{debug, warn}; // debug, error, info, span, trace, warn,

use crate::for_each;
use crate::setup::input::Parameters;
use crate::sph::boundary_handling::BoundaryHandling;
use crate::sph::pressure_solver::{PressureSolver, SolverMeasurementInfo};
use crate::sph::kernel::KernelFn;
use crate::fluid::{Fluid3D, Len};
use crate::sph::SystemParameters;
use crate::sph::CurrentSystemProperties;
use crate::sph::pressure_solver::{set_pred_vel_by_applying_acc, add_pressure_acceleration};
use crate::utilities::vector;
use crate::neighbor_search::NeighborList;

#[allow(dead_code)]
pub enum TerminationCondition {
    AfterIteration(u32),
    TargetDensityError(f64),
}

pub struct IISPH {
    // pub solver_iterations: u32,
    pub target_density_error: f64,
    pub relaxation_factor: f64,
    pub min_diagonal_element: f64,
    // Scratch buffers
    s_f: Vec<f64>,
    a_ff: Vec<f64>,
    pub pressure_acc_f: Vec<Vector3<f64>>,
    pub last_solver_iterations: u32,
    pub predicted_density_error: f64,
}

impl PressureSolver for IISPH {
    fn new(params: &Parameters) -> Self {
        Self {
            target_density_error: params.target_density_error,
            relaxation_factor: params.relaxation_factor,
            min_diagonal_element: params.min_diagonal_element,
            s_f: Vec::new(),
            a_ff: Vec::new(),
            pressure_acc_f: Vec::new(),
            last_solver_iterations: u32::default(),
            predicted_density_error: f64::default(),
        }
    }

    fn solve_and_add_acceleration<K: KernelFn>(
        &mut self,
        fluid: &mut Fluid3D,
        boundary: &impl BoundaryHandling,
        neighbor_list: &NeighborList,
        params: &SystemParameters,
        _properties: &mut CurrentSystemProperties,
    ) {
        self.resize_scratch(fluid.len());

        // solve pressure equation system
        {
            // set predicted velocity by applying non-pressure acceleration
            set_pred_vel_by_applying_acc(fluid, params, false);
            // set source term
            Self::set_source_term_vp::<K>(
                self,
                fluid,
                boundary,
                neighbor_list,
                params,
                false,
            );
            // self.set_source_term_vde();
            // println!("s_f: {}", self.particles[200].s_f);
            // solve pressure equation system
            self.resolve_pressure_globally::<K>(
                fluid,
                boundary,
                neighbor_list,
                params,
                false,
                // TerminationCondition::AfterIteration(params.solver_iterations),
                TerminationCondition::TargetDensityError(self.target_density_error),
                true,
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
        );
    }

    fn measurement_info(&self) -> SolverMeasurementInfo {
        SolverMeasurementInfo {
            target_density_error: self.target_density_error,
            solver_iterations: self.last_solver_iterations,
            relaxation_factor: self.relaxation_factor,
            predicted_density_error: self.predicted_density_error,
            ..Default::default()
        }
    }
}

impl IISPH {
    pub fn resize_scratch(&mut self, len: usize) {
        self.s_f.resize(len, 0.0);
        self.a_ff.resize(len, 0.0);
        self.pressure_acc_f.resize(len, Vector3::zeros());
    }

    /// Calculate source term for velocity divergence eliminating linear equation system for pressure
    pub fn set_source_term_vde<K: KernelFn>(
        &mut self,
        fluid: &Fluid3D,
        boundary: &impl BoundaryHandling,
        neighbor_list: &NeighborList,
        params: &SystemParameters,
    ) {
        // compute source term s_f of pressure linear equation system
        for_each!(
            mut [self.s_f],
            ref [
                pos_now = fluid.position,
                vel_pred = fluid.velocity_pred,
                volume = fluid.volume,
                neighbors = neighbor_list,
                boundary = boundary
            ],
            |id, id_s_f| {
                let mut accu = 0.;
                for &neighbor in neighbors.get_neighbors(id) {
                    let r_vec = vector(
                        &pos_now[neighbor],
                        &pos_now[id],
                    );
                    accu -= params.time_increment
                        * volume[neighbor]
                        * (vel_pred[id] - vel_pred[neighbor]).dot(
                            &K::kernel_gradient(
                                &r_vec,
                                params.kernel_support_radius,
                            ),
                        );
                }
                for &boundary_neighbor in boundary.get_neighbors(id) {
                    let r_vec = vector(
                        boundary.pos_now(boundary_neighbor),
                        &pos_now[id],
                    );
                    accu -= params.time_increment
                        * *boundary.volume(boundary_neighbor)
                        * (vel_pred[id]
                            - *boundary.vel_now(boundary_neighbor))
                        .dot(&K::kernel_gradient(
                            &r_vec,
                            params.kernel_support_radius,
                        ));
                }
                *id_s_f = accu;
                // if i == 200 {
                //     println!("source term vel.div.: {}", particle.s_f[id]);
                // }
            }
        );
    }

    /// Calculate source term for volume preserving linear equation system for pressure
    pub fn set_source_term_vp<K: KernelFn>(
        &mut self,
        fluid: &Fluid3D,
        boundary: &impl BoundaryHandling,
        neighbor_list: &NeighborList,
        params: &SystemParameters,
        with_pred_positions: bool,
    ) {
        // compute source term s_f of pressure linear equation system
        for_each!(
            mut [self.s_f],
            ref [
                pos_now = fluid.position,
                pos_pred = fluid.position_pred,
                vel_pred = fluid.velocity_pred,
                volume = fluid.volume,
                neighbors = neighbor_list,
                boundary = boundary
            ],
            |id, id_s_f| {
                let mut accu = 1. - params.rest_volume / volume[id];
                for &neighbor in neighbors.get_neighbors(id) {
                    // select positions
                    let particle_pos = if with_pred_positions {
                        pos_pred[id]
                    } else {
                        pos_now[id]
                    };
                    let fluid_neighbor_pos = if with_pred_positions {
                        pos_pred[neighbor]
                    } else {
                        pos_now[neighbor]
                    };

                    let r_vec = vector(
                        &fluid_neighbor_pos,
                        &particle_pos,
                    );
                    accu -= params.time_increment
                        * volume[neighbor]
                        * (vel_pred[id] - vel_pred[neighbor]).dot(
                            &K::kernel_gradient(
                                &r_vec,
                                params.kernel_support_radius,
                            ),
                        );
                }
                for &boundary_neighbor in boundary.get_neighbors(id) {
                    // select position
                    let particle_pos = if with_pred_positions {
                        pos_pred[id]
                    } else {
                        pos_now[id]
                    };

                    let r_vec = vector(
                        boundary.pos_now(boundary_neighbor),
                        &particle_pos,
                    );
                    accu -= params.time_increment
                        * *boundary.volume(boundary_neighbor)
                        * (vel_pred[id]
                            - *boundary.vel_now(boundary_neighbor))
                        .dot(&K::kernel_gradient(
                            &r_vec,
                            params.kernel_support_radius,
                        ));
                }
                *id_s_f = accu;
                // if i == 200 {
                //     println!("source term vol.pre.: {}", particle.s_f[id]);
                // }
            }
        );
    }

    /// compute diagonal element A_ff
    fn set_diagonal_element<K: KernelFn>(
        &mut self,
        fluid: &mut Fluid3D,
        boundary: &impl BoundaryHandling,
        neighbor_list: &NeighborList,
        params: &SystemParameters,
        with_pred_positions: bool,
    ) {
        // compute diagonal element A_ff
        for_each!(
            mut [self.a_ff],
            ref [
                pos_now = fluid.position,
                pos_pred = fluid.position_pred,
                mass = fluid.mass,
                volume = fluid.volume,
                neighbors = neighbor_list,
                boundary = boundary,
            ],
            |id, id_a_ff| {
                // calc intermediate variables
                let mut sum_fluid = Vector3::zeros();
                let mut sum_fluid2 = 0.;
                for &neighbor in neighbors.get_neighbors(id) {
                    // select positions
                    let particle_pos = if with_pred_positions {
                        pos_pred[id]
                    } else {
                        pos_now[id]
                    };
                    let fluid_neighbor_pos = if with_pred_positions {
                        pos_pred[neighbor]
                    } else {
                        pos_now[neighbor]
                    };

                    let r_vec = vector(
                        &fluid_neighbor_pos,
                        &particle_pos,
                    );
                    sum_fluid += volume[neighbor]
                        * K::kernel_gradient(
                            &r_vec,
                            params.kernel_support_radius,
                        );

                    sum_fluid2 -= params.time_increment.powi(2)
                        * volume[id]
                        * volume[neighbor].powi(2)
                        / mass[neighbor]
                        * K::kernel_gradient(
                            &r_vec,
                            params.kernel_support_radius,
                        )
                        .norm_squared();
                }
                let mut sum_boundary = Vector3::zeros();
                for &boundary_neighbor in boundary.get_neighbors(id) {
                    // select position
                    let particle_pos = if with_pred_positions {
                        pos_pred[id]
                    } else {
                        pos_now[id]
                    };

                    let r_vec = vector(
                        boundary.pos_now(boundary_neighbor),
                        &particle_pos,
                    );
                    sum_boundary += *boundary.volume(boundary_neighbor)
                        * K::kernel_gradient(
                            &r_vec,
                            params.kernel_support_radius,
                        );
                }
                // select weighting
                let weighting = params.boundary_pressure_acceleration_weighting;
                // calc intermediate variable c_f
                let c_f = -volume[id] / mass[id]
                    * (sum_fluid + 2. * weighting * sum_boundary);
                // use intermediate variables to calc a_ff
                *id_a_ff = params.time_increment.powi(2)
                    * c_f.dot(&(sum_fluid + sum_boundary))
                    + sum_fluid2;
            }
        );
    }

    fn initialize(
        &mut self,
        fluid: &mut Fluid3D,
        clamp_pressure: bool,
    ) {
        for_each!(
            mut [self.a_ff, fluid.pressure],
            ref [
                s_f = self.s_f,
            ],
            |id, id_a_ff, id_pressure| {
                // initialize pressure with fixed result of first self iteration
                // Update pressure
                if *id_a_ff > self.min_diagonal_element
                    || *id_a_ff < -self.min_diagonal_element
                {
                    let p_next_iter =
                        self.relaxation_factor * s_f[id] / *id_a_ff;
                    // particle.set_pressure(0.); // TODO remove
                    if clamp_pressure {
                        // TODO uncomment
                        *id_pressure = p_next_iter.max(0.);
                    } else {
                        *id_pressure = p_next_iter;
                    }
                } else {
                    *id_pressure = 0.;
                }
                assert!(*id_a_ff <= 0.);
            }
        );
    }

    fn continue_solving(
        termination_condition: &TerminationCondition,
        solver_iteration: u32,
        predicted_density_error: f64,
    ) -> bool {
        match termination_condition {
            TerminationCondition::AfterIteration(number) => solver_iteration < *number,
            TerminationCondition::TargetDensityError(tde) => {
                let min_solver_iterations = 2;
                let max_solver_iteration = u32::MAX;
                // let max_solver_iteration = 100;
                (solver_iteration < min_solver_iterations || predicted_density_error > *tde) && solver_iteration < max_solver_iteration
            }
        }
    }

    /// Globally calculate pressure by solving a linear equation system at current time
    /// and update respective particles' fields
    ///
    /// For the implementation the following document was closedly followed:
    /// Notes on  Ihmsen et al. ”Implicit Incompressible SPH” by  Matthias Teschner, University of Freiburg
    pub fn resolve_pressure_globally<K: KernelFn>(
        &mut self,
        fluid: &mut Fluid3D,
        boundary: &impl BoundaryHandling,
        neighbor_list: &NeighborList,
        params: &SystemParameters,
        with_pred_positions: bool,
        termination_condition: TerminationCondition,
        clamp_pressure: bool,
    ) {
        // compute diagonal element A_ff
        self.set_diagonal_element::<K>(
            fluid,
            boundary,
            neighbor_list,
            params,
            with_pred_positions,
        );
        // Set initial guess
        self.initialize(fluid, clamp_pressure);
        // Solve linear equation system until a sufficiently accurate result is obtained
        let mut solver_iteration = 0;
        let mut predicted_density_error = f64::INFINITY;
        // for _solver_iteration in 0..self.properties.solver_iterations {
        while Self::continue_solving(
            &termination_condition,
            solver_iteration,
            predicted_density_error,
        ) {
            // compute intermediate pressure acceleration
            add_pressure_acceleration::<K>(
                Some(&mut self.pressure_acc_f),
                fluid,
                boundary,
                neighbor_list,
                params,
                with_pred_positions,
                true,
            );

            // perform self iteration for all fluid particles
            let mut pred_density_errors: Vec<f64> = vec![0.0; fluid.len()];
            for_each!(
                mut [fluid.pressure, pred_density_errors],
                ref [
                    pos_now = fluid.position,
                    pos_pred = fluid.position_pred,
                    pressure_acc_f = self.pressure_acc_f,
                    volume = fluid.volume,
                    neighbors = neighbor_list,
                    boundary = boundary,
                    s_f = self.s_f,
                    a_ff = self.a_ff,
                ],
                |id, id_pressure, id_pred_density_errors| {
                    // calculate the divergence of the velocity change due to the pressure acceleration: a_dot_p_f
                    let mut a_dot_p_f = 0.;
                    for &neighbor in neighbors.get_neighbors(id) {
                        // select positions
                        let particle_pos = if with_pred_positions {
                            pos_pred[id]
                        } else {
                            pos_now[id]
                        };
                        let fluid_neighbor_pos = if with_pred_positions {
                            pos_pred[neighbor]
                        } else {
                            pos_now[neighbor]
                        };

                        let r_vec = vector(
                            &fluid_neighbor_pos,
                            &particle_pos,
                        );
                        a_dot_p_f += params.time_increment.powi(2)
                            * volume[neighbor]
                            * (pressure_acc_f[id] - pressure_acc_f[neighbor])
                                .dot(&K::kernel_gradient(
                                    &r_vec,
                                    params.kernel_support_radius,
                                ));
                    }
                    for &boundary_neighbor in boundary.get_neighbors(id) {
                        // select positions
                        let particle_pos = if with_pred_positions {
                            pos_pred[id]
                        } else {
                            pos_now[id]
                        };

                        let r_vec = vector(
                            boundary.pos_now(boundary_neighbor),
                            &particle_pos,
                        );
                        a_dot_p_f += params.time_increment.powi(2)
                            * *boundary.volume(boundary_neighbor)
                            * pressure_acc_f[id]
                                .dot(&K::kernel_gradient(
                                    &r_vec,
                                    params.kernel_support_radius,
                                ));
                    }
                    // Update pressure
                    if a_ff[id] < -self.min_diagonal_element {
                        // || particle.a_ff[id] > params.min_diagonal_element {
                        let p_next_iter = *id_pressure
                            + self.relaxation_factor * (s_f[id] - a_dot_p_f)
                                / a_ff[id];
                        // particle.set_pressure(p_next_iter.max(0.));
                        if clamp_pressure {
                            *id_pressure = p_next_iter.max(0.);
                        } else {
                            *id_pressure = p_next_iter;
                        }
                    }
                    // Calculate and send absolute value of predicted density error
                    // if particle.s_f[id] < 0. {
                    if (s_f[id] < 0. && clamp_pressure)
                        || (!clamp_pressure
                            && a_ff[id] < -self.min_diagonal_element)
                    {
                        *id_pred_density_errors = (a_dot_p_f - s_f[id]).abs();
                    } else {
                        *id_pred_density_errors = 0.;
                    }
                }
            );
            // accumulate average_predicted_density_error
            #[cfg(not(feature = "parallel"))]
            let total_error: f64 = pred_density_errors.iter().sum();
            #[cfg(feature = "parallel")]
            let total_error: f64 = pred_density_errors.par_iter().sum();
            let count = pred_density_errors.len();
            predicted_density_error = if count > 0 {
                total_error / count as f64 * 100.0
            } else {
                0.0
            };
            #[cfg(feature = "logging")]
            debug!("solver_iteration {}", solver_iteration);
            #[cfg(feature = "logging")]
            debug!("average_relative_predicted_density_error (%): {predicted_density_error}");

            solver_iteration += 1;
            #[cfg(feature = "logging")]
            if solver_iteration == 100 {
                warn!("Number of global pressure self iterations >= 100");
            }
        }
        #[cfg(feature = "logging")]
        debug!("final number of self iterations: {solver_iteration} (+1)");
        #[cfg(feature = "logging")]
        debug!("final average_relative_predicted_density_error (%): {predicted_density_error}");

        self.last_solver_iterations = solver_iteration;
        self.predicted_density_error = predicted_density_error;
    }
}
