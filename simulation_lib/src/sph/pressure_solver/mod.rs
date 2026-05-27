/// Pressure solver algorithm module
use nalgebra::Vector3;
#[cfg(feature = "parallelized_sph")]
use rayon::prelude::*;

use crate::for_each;
use crate::sph::kernel::KernelFn;
use crate::sample::{Fluid3D, Boundary3D, Positional};
use crate::sph::SystemParameters;
use crate::sph::CurrentSystemProperties;
use crate::sph::direction;

pub mod sesph;
pub mod sesph_with_splitting;
pub mod iisph;
pub mod iisph_optimized_source_term;

pub use sesph::SESPH;
pub use sesph_with_splitting::SESPHwSplitting;
pub use iisph::IISPH;
pub use iisph_optimized_source_term::IISPHwOST;

pub trait PressureSolver: Send + Sync {
    /// Compute pressure
    ///
    /// Contract: Non-pressure accelerations (gravity, viscosity) are already
    /// accumulated in `fluid.acceleration` before this is called.
    fn solve_and_add_acceleration<K: KernelFn>(
        &mut self,
        fluid: &mut Fluid3D,
        boundary: &Boundary3D,
        params: &SystemParameters,
        properties: &mut CurrentSystemProperties,
    );
}

/// calculate and set predicted velocity due to currently set acceleration
fn set_pred_vel_by_applying_acc(fluid: &mut Fluid3D, params: &SystemParameters, to_pred_vel: bool) {
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
fn add_pressure_acceleration<K: KernelFn>(
    custom_target: Option<&mut Vec<Vector3<f64>>>,
    fluid: &mut Fluid3D,
    boundary: &Boundary3D,
    params: &SystemParameters,
    with_pred_positions: bool,
    overwrite: bool
) {
    let target = if let Some(target) = custom_target {
        target
    } else {
        &mut fluid.acceleration
    };
    // compute pressure acceleration
    for_each!(
        mut [target],
        ref [
            pos_now = fluid.position,
            pos_pred = fluid.position_pred,
            mass = fluid.mass,
            volume = fluid.volume,
            pressure = fluid.pressure,
            neighbors = fluid.neighbors,
            boundary_neighbors = fluid.boundary_neighbors
        ],
        |id, target_acceleration| {
            let mut accu = Vector3::zeros();
            // add pressure acceleration from other moving particles
            for &neighbor in &neighbors[id] {
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
                // calc acceleration
                let r_vec = direction(
                    &fluid_neighbor_pos,
                    &particle_pos,
                );
                let dist = r_vec.norm();
                accu -= volume[id] / mass[id]
                    * volume[neighbor]
                    * (pressure[id] + pressure[neighbor])
                    * K::gradient(
                        &r_vec,
                        dist,
                        params.smoothing_length,
                    );
            }
            // add pressure acceleration from boundary particles
            for &boundary_neighbor in &boundary_neighbors[id] {
                // select weighting
                let weighting = params.boundary_pressure_acceleration_weighting;
                // select position
                let particle_pos = if with_pred_positions {
                    pos_pred[id]
                } else {
                    pos_now[id]
                };
                // calc acceleration
                // mirror only pressure into boundary particle, set density to rest density
                let r_vec = direction(
                    boundary.pos_now(boundary_neighbor),
                    &particle_pos,
                );
                let dist = r_vec.norm();
                accu -= 2. * weighting * volume[id] / mass[id]
                    * *boundary.volume(boundary_neighbor)
                    * pressure[id]
                    * K::gradient(
                        &r_vec,
                        dist,
                        params.smoothing_length,
                    );
            }
            if overwrite {
                *target_acceleration = accu;
            } else {
                *target_acceleration += accu;
            }
        }
    );
}