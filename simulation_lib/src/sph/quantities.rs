use nalgebra::{Point3, Vector3};
/// Volume calculation module
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::for_each;
use crate::neighbor_search::NeighborList;
use crate::sph::SystemParameters;
use crate::sph::boundary_handling::{BoundaryHandling, RequestMode};
use crate::sph::kernel::KernelFn;
use crate::utilities::vector;

/// Calculate and set volume for all positions at the current point in time
pub fn get_volume<K: KernelFn>(
    volume: &mut Vec<f64>,
    position_eval: &Vec<Point3<f64>>,
    neighboring_samples: &NeighborList,
    sample_positions: &Vec<Point3<f64>>,
    boundary: &impl BoundaryHandling,
    params: &SystemParameters,
) {
    for_each!(
        mut [volume],
        ref [
            position_eval = position_eval,
            neighbors = neighboring_samples,
            sample_positions = sample_positions,
            boundary = boundary,
        ],
        |id, id_volume| {
            let mut accu = 0.;
            // add volume for every neighbor
            for &neighbor in neighbors.get_neighbors(id) {
                let r_vec = vector(
                    &sample_positions[neighbor],
                    &position_eval[id],
                );
                accu += params.rest_volume
                    * K::kernel_function(
                        &r_vec,
                        params.kernel_support_radius,
                    );
            }
            // add volume contribution from boundary
            for &boundary_neighbor in boundary.get_neighbors(id, RequestMode::Normal) {
                let r_vec = vector(
                    boundary.pos_now(boundary_neighbor),
                    &position_eval[id],
                );
                accu += boundary.volume(boundary_neighbor)
                    * K::kernel_function(
                        &r_vec,
                        params.kernel_support_radius,
                    );
            }
            *id_volume = params.rest_volume / accu;
        }
    );
}

/// Calculate and set speed for all positions at the current point in time
pub fn get_speed<K: KernelFn>(
    speed: &mut Vec<f64>,
    position_eval: &Vec<Point3<f64>>,
    neighboring_samples: &NeighborList,
    sample_positions: &Vec<Point3<f64>>,
    sample_velocities: &Vec<Vector3<f64>>,
    sample_volumes: &Vec<f64>,
    boundary: &impl BoundaryHandling,
    params: &SystemParameters,
) {
    for_each!(
        mut [speed],
        ref [
            pos_now_eval = position_eval,
            neighbors = neighboring_samples,
            sample_positions = sample_positions,
            sample_velocities = sample_velocities,
            sample_volumes = sample_volumes,
            boundary = boundary,
            params = params,
        ],
        |id, id_speed| {
            let mut accu = Vector3::zeros();
            // add velocity for every neighbor
            for &neighbor in neighbors.get_neighbors(id) {
                let r_vec = vector(
                    &pos_now_eval[id],
                    &sample_positions[neighbor],
                );
                accu += sample_velocities[neighbor]
                    * sample_volumes[neighbor]
                    * K::kernel_function(
                        &r_vec,
                        params.kernel_support_radius,
                    );
            }
            // add contribution from boundary
            for &boundary_neighbor in boundary.get_neighbors(id, RequestMode::Normal) {
                let r_vec = vector(
                    &pos_now_eval[id],
                    boundary.pos_now(boundary_neighbor),
                );
                accu += *boundary.vel_now(boundary_neighbor)
                    * boundary.volume(boundary_neighbor)
                    * K::kernel_function(
                        &r_vec,
                        params.kernel_support_radius,
                    );
            }
            *id_speed = accu.norm();
        }
    );
}

/// Calculate and set density for all positions at the current point in time
pub fn get_density<K: KernelFn>(
    density: &mut Vec<f64>,
    position_eval: &Vec<Point3<f64>>,
    neighboring_samples: &NeighborList,
    sample_positions: &Vec<Point3<f64>>,
    sample_masses: &Vec<f64>,
    // boundary: &impl BoundaryHandling,
    params: &SystemParameters,
) {
    for_each!(
        mut [density],
        ref [
            position_eval = position_eval,
            neighbors = neighboring_samples,
            sample_positions = sample_positions,
            sample_masses = sample_masses,
            // boundary = boundary,
            params = params,
        ],
        |id, id_density| {
            let mut accu = 0.;
            // add volume for every neighbor
            for &neighbor in neighbors.get_neighbors(id) {
                let r_vec = vector(
                    &position_eval[id],
                    &sample_positions[neighbor],
                );
                accu += sample_masses[neighbor]
                    * K::kernel_function(
                        &r_vec,
                        params.kernel_support_radius,
                    );
            }
            // // add contribution from boundary
            // for &boundary_neighbor in boundary.get_neighbors(id) {
            //     let r_vec = vector(
            //         &position_eval[id],
            //         boundary.pos_now(boundary_neighbor),
            //     );
            //     accu += *boundary.density(boundary_neighbor)
            //         *boundary.volume(boundary_neighbor)
            //         * K::kernel_function(
            //             &r_vec,
            //             params.kernel_support_radius,
            //         );
            // }
            *id_density = accu;
        }
    );
}

/// Calculate and set density for all positions at the current point in time
pub fn get_density_error<K: KernelFn>(
    density_err: &mut Vec<f64>,
    position_eval: &Vec<Point3<f64>>,
    neighboring_samples: &NeighborList,
    sample_positions: &Vec<Point3<f64>>,
    sample_volumes: &Vec<f64>,
    // boundary: &impl BoundaryHandling,
    params: &SystemParameters,
) {
    for_each!(
        mut [density_err],
        ref [
            position_eval = position_eval,
            neighbors = neighboring_samples,
            sample_positions = sample_positions,
            sample_volumes = sample_volumes,
            // boundary = boundary,
            params = params,
        ],
        |id, id_density_err| {
            let mut accu = 0.;
            // add volume for every neighbor
            for &neighbor in neighbors.get_neighbors(id) {
                let r_vec = vector(
                    &position_eval[id],
                    &sample_positions[neighbor],
                );
                let err = if sample_volumes[neighbor] < params.rest_volume {
                    params.rest_volume / sample_volumes[neighbor] - 1.
                } else {
                    continue;
                };
                accu += err
                    * sample_volumes[neighbor]
                    * K::kernel_function(
                        &r_vec,
                        params.kernel_support_radius,
                    );
            }
            // // add contribution from boundary
            // for &boundary_neighbor in boundary.get_neighbors(id) {
            //     let r_vec = vector(
            //         &position_eval[id],
            //         boundary.pos_now(boundary_neighbor),
            //     );
            //     accu += *boundary.density(boundary_neighbor)
            //         *boundary.volume(boundary_neighbor)
            //         * K::kernel_function(
            //             &r_vec,
            //             params.kernel_support_radius,
            //         );
            // }
            *id_density_err = 100. * accu;
        }
    );
}

/// Calculate and set speed for all positions at the current point in time
pub fn get_pressure<K: KernelFn>(
    pressure: &mut Vec<f64>,
    position_eval: &Vec<Point3<f64>>,
    neighboring_samples: &NeighborList,
    sample_positions: &Vec<Point3<f64>>,
    sample_volumes: &Vec<f64>,
    sample_pressure: &Vec<f64>,
    // boundary: &impl BoundaryHandling,
    params: &SystemParameters,
) {
    for_each!(
        mut [pressure],
        ref [
            pos_now_eval = position_eval,
            neighbors = neighboring_samples,
            sample_positions = sample_positions,
            sample_volumes = sample_volumes,
            sample_pressure = sample_pressure,
            // boundary = boundary,
            params = params,
        ],
        |id, id_pressure| {
            let mut accu = 0.;
            // add velocity for every neighbor
            for &neighbor in neighbors.get_neighbors(id) {
                let r_vec = vector(
                    &pos_now_eval[id],
                    &sample_positions[neighbor],
                );
                accu += sample_pressure[neighbor]
                    * sample_volumes[neighbor]
                    * K::kernel_function(
                        &r_vec,
                        params.kernel_support_radius,
                    );
            }
            // // add contribution from boundary
            // for &boundary_neighbor in boundary.get_neighbors(id) {
            //     let r_vec = vector(
            //         &pos_now_eval[id],
            //         boundary.pos_now(boundary_neighbor),
            //     );
            //     accu += *boundary.vel_now(boundary_neighbor)
            //         * *boundary.volume(boundary_neighbor)
            //         * K::kernel_function(
            //             &r_vec,
            //             params.kernel_support_radius,
            //         );
            // }
            *id_pressure = accu;
        }
    );
}

/// Calculate and set speed for all positions at the current point in time
pub fn get_kinetic_energy<K: KernelFn>(
    kinetic_energy: &mut Vec<f64>,
    position_eval: &Vec<Point3<f64>>,
    neighboring_samples: &NeighborList,
    sample_positions: &Vec<Point3<f64>>,
    sample_velocities: &Vec<Vector3<f64>>,
    sample_volumes: &Vec<f64>,
    sample_masses: &Vec<f64>,
    // boundary: &impl BoundaryHandling,
    params: &SystemParameters,
) {
    for_each!(
        mut [kinetic_energy],
        ref [
            pos_now_eval = position_eval,
            neighbors = neighboring_samples,
            sample_positions = sample_positions,
            sample_velocities = sample_velocities,
            sample_volumes = sample_volumes,
            sample_masses = sample_masses,
            // boundary = boundary,
            params = params,
        ],
        |id, id_kinetic_energy| {
            let mut accu = 0.;
            // add velocity for every neighbor
            for &neighbor in neighbors.get_neighbors(id) {
                let r_vec = vector(
                    &pos_now_eval[id],
                    &sample_positions[neighbor],
                );
                accu += 0.5 * sample_masses[neighbor]
                    * sample_velocities[neighbor].norm_squared()
                    * sample_volumes[neighbor]
                    * K::kernel_function(
                        &r_vec,
                        params.kernel_support_radius,
                    );
            }
            // // add contribution from boundary
            // for &boundary_neighbor in boundary.get_neighbors(id) {
            //     let r_vec = vector(
            //         &pos_now_eval[id],
            //         boundary.pos_now(boundary_neighbor),
            //     );
            //     accu += 0.5 * boundary.density(boundary_neighbor)
            //         * boundary.vel_now(boundary_neighbor).norm_squared()
            //         * boundary.volume(boundary_neighbor).powi(2)
            //         * K::kernel_function(
            //                 &r_vec,
            //             params.kernel_support_radius,
            //         );
            // }
            *id_kinetic_energy = accu;
        }
    );
}
