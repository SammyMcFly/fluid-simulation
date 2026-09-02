//! Pressure solver algorithm module
pub mod iisph;
pub mod iisph_optimized_source_term;
pub mod sesph;
pub mod sesph_with_splitting;

pub use iisph::IISPH;
pub use iisph_optimized_source_term::IISPHwOST;
pub use sesph::SESPH;
pub use sesph_with_splitting::SESPHwSplitting;

use crate::for_each;
use crate::iteration::for_each_collect;
use crate::neighbor_search::NeighborList;
use crate::sph::CurrentSystemProperties;
use crate::sph::SystemParameters;
use crate::sph::boundary_handling::ForceOntoBoundary;
use crate::sph::boundary_handling::{BoundaryHandling, RequestMode};
use crate::sph::fluid::Fluid;
use crate::sph::kernel::KernelFn;
use crate::sph::setup::input::Parameters;
use crate::utilities::vector;

use nalgebra::Vector3;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub enum PressureSolverVariant {
    SESPH,
    SESPHwSplitting,
    IISPH,
    IISPHwOST,
}

pub trait PressureSolver: Send + Sync + Clone {
    /// Whether this solver correctly supports two-way coupling with
    /// *dynamic* (rigid-body) boundaries.
    ///
    /// Defaults to `true`; override to `false` for solvers with this
    /// limitation. Checked in `SystemConstructor::new`, which refuses to
    /// build a system pairing such a solver with a scene that defines at
    /// least one dynamic boundary.
    const SUPPORTS_DYNAMIC_BOUNDARIES: bool = true;

    fn new(params: &Parameters) -> Self;

    /// Compute pressure
    ///
    /// Contract: Non-pressure accelerations (gravity, viscosity) are already
    /// accumulated in `fluid.acceleration` before this is called.
    fn solve_and_add_acceleration<K: KernelFn>(
        &mut self,
        fluid: &mut Fluid,
        boundary: &mut impl BoundaryHandling,
        neighbors: &NeighborList,
        params: &SystemParameters,
        properties: &mut CurrentSystemProperties,
    );

    /// Return solver-specific measurement data
    fn measurement_info(&self) -> SolverMeasurementInfo {
        SolverMeasurementInfo::default()
    }
}

#[derive(Debug, Clone, Default)]
pub struct SolverMeasurementInfo {
    pub stiffness: f64,
    pub target_density_error: f64,
    pub solver_iterations: u32,
    pub relaxation_factor: f64,
    pub predicted_density_error: f64,
}

/// calculate and set predicted velocity due to currently set acceleration
fn set_pred_vel_by_applying_acc(fluid: &mut Fluid, params: &SystemParameters, to_pred_vel: bool) {
    for_each!(
        mut [fluid.velocity_pred],
        ref [
            vel_now = fluid.velocity,
            acceleration = fluid.acceleration,
        ],
        |id, id_velocity_pred| {
            // select velocity
            let base_vel = if to_pred_vel {
                *id_velocity_pred
            } else {
                vel_now[id]
            };
            let vel = base_vel + params.time_increment * acceleration[id];
            *id_velocity_pred = vel;
        }
    );
}

/// Locally calculate pressure acceleration with a state equation at current time
/// and add it to respective samples
///
/// If `custom_target` is `None`, `fluid.acceleration` is used as the target
/// and the result committed to `fluid.acceleration` is mirrored back onto
/// the boundary.
fn add_pressure_acceleration<K: KernelFn>(
    custom_target: Option<&mut Vec<Vector3<f64>>>,
    fluid: &mut Fluid,
    boundary: &mut impl BoundaryHandling,
    neighbors: &NeighborList,
    params: &SystemParameters,
    with_pred_positions: bool,
    overwrite: bool,
) {
    let is_committed_to = custom_target.is_none();
    let target = if let Some(target) = custom_target {
        target
    } else {
        &mut fluid.acceleration
    };
    // compute pressure acceleration
    let forces_onto_boundary: Vec<ForceOntoBoundary> = for_each_collect!(
        mut [target],
        ref [
            pos_now = fluid.position,
            pos_pred = fluid.position_pred,
            mass = fluid.mass,
            volume = fluid.volume,
            pressure = fluid.pressure,
            neighbors = neighbors,
            boundary = boundary,
        ],
        |id, target_acceleration, local_forces| {
            let mut accu = Vector3::zeros();
            let particle_pos = if with_pred_positions {
                pos_pred[id]
            } else {
                pos_now[id]
            };

            for &neighbor in neighbors.get_neighbors(id) {
                let fluid_neighbor_pos = if with_pred_positions {
                    pos_pred[neighbor]
                } else {
                    pos_now[neighbor]
                };
                let r_vec = vector(&fluid_neighbor_pos, &particle_pos);
                accu -= volume[id] / mass[id]
                    * volume[neighbor]
                    * (pressure[id] + pressure[neighbor])
                    * K::kernel_gradient(&r_vec, params.kernel_support_radius);
            }

            for (i, b) in boundary.iter().enumerate() {
                for &boundary_neighbor in b.get_neighbors(id, RequestMode::Normal) {
                    let weighting = params.boundary_pressure_acceleration_weighting;
                    let r_vec = vector(b.position(boundary_neighbor), &particle_pos);
                    let force = 2. * weighting * volume[id]
                        * b.volume(boundary_neighbor)
                        * pressure[id]
                        * K::kernel_gradient(&r_vec, params.kernel_support_radius);

                    if is_committed_to && b.is_dynamic() {
                        local_forces.push(ForceOntoBoundary {
                            id: i,
                            force,
                            force_location: *b.position(boundary_neighbor),
                        });
                    }
                    accu -= force / mass[id];
                }
            }

            if overwrite {
                *target_acceleration = accu;
            } else {
                *target_acceleration += accu;
            }
        }
    );
    // for_each!(
    //     mut [target],
    //     ref [
    //         pos_now = fluid.position,
    //         pos_pred = fluid.position_pred,
    //         mass = fluid.mass,
    //         volume = fluid.volume,
    //         pressure = fluid.pressure,
    //         neighbors = neighbors,
    //         boundary = boundary,
    //     ],
    //     |id, target_acceleration| {
    //         let mut accu = Vector3::zeros();
    //         let particle_pos = if with_pred_positions {
    //             pos_pred[id]
    //         } else {
    //             pos_now[id]
    //         };
    //         // add pressure acceleration from other moving particles
    //         for &neighbor in neighbors.get_neighbors(id) {
    //             // select positions
    //             let fluid_neighbor_pos = if with_pred_positions {
    //                 pos_pred[neighbor]
    //             } else {
    //                 pos_now[neighbor]
    //             };
    //             // calc acceleration
    //             let r_vec = vector(
    //                 &fluid_neighbor_pos,
    //                 &particle_pos,
    //             );
    //             accu -= volume[id] / mass[id]
    //                 * volume[neighbor]
    //                 * (pressure[id] + pressure[neighbor])
    //                 * K::kernel_gradient(
    //                     &r_vec,
    //                     params.kernel_support_radius,
    //                 );
    //         }
    //         // add pressure acceleration from boundary particles
    //         for (i, b) in boundary.iter().enumerate() {
    //             for &boundary_neighbor in b.get_neighbors(id, RequestMode::Normal) {
    //                 // select weighting
    //                 let weighting = params.boundary_pressure_acceleration_weighting;
    //                 // calc acceleration
    //                 // mirror only pressure into boundary particle, set density to rest density
    //                 let r_vec = vector(
    //                     b.pos_now(boundary_neighbor),
    //                     &particle_pos,
    //                 );
    //                 let force = 2. * weighting * volume[id]
    //                     * b.volume(boundary_neighbor)
    //                     * pressure[id]
    //                     * K::kernel_gradient(
    //                         &r_vec,
    //                         params.kernel_support_radius,
    //                     );
    //                 if is_committed_to && b.is_dynamic() {
    //                     forces_onto_boundary.push(ForceOntoBoundary {
    //                         id: i,
    //                         force,
    //                         force_location: *b.pos_now(boundary_neighbor),
    //                     });
    //                 }
    //                 accu -= force / mass[id];
    //             }
    //         }
    //         if overwrite {
    //             *target_acceleration = accu;
    //         } else {
    //             *target_acceleration += accu;
    //         }
    //     }
    // );
    if is_committed_to {
        for force in forces_onto_boundary {
            boundary.add_force_onto_boundary(force);
        }
    }
}
