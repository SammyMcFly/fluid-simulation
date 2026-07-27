/// Acceleration module
use nalgebra::Vector3;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::fluid::Fluid3D;
use crate::sph::SystemParameters;
use crate::sph::vector;
use crate::neighbor_search::NeighborList;

/// reset acceleration, i. e. set it to 0.
pub fn reset_acceleration(
    fluid: &mut Fluid3D,
) {
    for_each!(
        mut [fluid.acceleration],
        ref [],
        |_id, id_acceleration| {
            *id_acceleration = Vector3::zeros();
        }
    );
}

/// Add gravity acceleration to all not boundary particles
pub fn add_gravity_acceleration(
    fluid: &mut Fluid3D,
) {
    for_each!(
        mut [fluid.acceleration],
        ref [position = fluid.position],
        |id, id_acceleration| {
            let strength_of_gravity = 9.81;
            // gravitate downwards
            *id_acceleration +=  Vector3::new(0.0, 0.0, -strength_of_gravity);
            // gravitate around point
            // let gravitation_center = Point3::new(0.0, 0.0, 0.0);
            // let direction = vector(&position[id], &gravitation_center);
            // let direction_normalized = direction/direction.norm();
            // *id_acceleration +=  strength_of_gravity*direction_normalized;
        }
    );
}

/// Calculate viscosity acceleration at current time and add it to respective particles
pub fn add_viscosity_acceleration<K: KernelFn>(
    fluid: &mut Fluid3D,
    boundary: &impl BoundaryHandling,
    neighbors: &NeighborList,
    params: &SystemParameters,
) {
    for_each!(
        mut [fluid.acceleration],
        ref [
            pos_now = fluid.position,
            vel_now = fluid.velocity,
            volume = fluid.volume,
            neighbors = neighbors,
            boundary = boundary
        ],
        |id, id_acceleration| {
            let mut accu = Vector3::zeros();
            // add viscostiy acceleration from other moving particles
            for &neighbor in neighbors.get_neighbors(id) {
                let r_vec = vector(
                    &pos_now[neighbor],
                    &pos_now[id],
                );
                accu += params.fluid_viscosity
                    * 2.
                    * (3. + 2.)
                    * volume[neighbor]
                    * (vel_now[id] - vel_now[neighbor])
                        .dot(&(pos_now[id] - pos_now[neighbor]))
                    / ((pos_now[id] - pos_now[neighbor])
                        .norm_squared()
                        + 0.01 * params.rest_density_grid_spacing.powi(2))
                    * K::kernel_gradient(
                        &r_vec,
                        params.kernel_support_radius,
                    );
            }
            // add viscostiy acceleration contribution from boundary
            for &boundary_neighbor in boundary.get_neighbors(id) {
                let r_vec = vector(
                    boundary.pos_now(boundary_neighbor),
                    &pos_now[id],
                );
                accu += params.boundary_viscosity
                    * 2.
                    * (3. + 2.)
                    * *boundary.volume(boundary_neighbor)
                    * (vel_now[id] - *boundary.vel_now(boundary_neighbor))
                        .dot(
                            &(pos_now[id]
                                - *boundary.pos_now(boundary_neighbor)),
                        )
                    / ((pos_now[id]
                        - *boundary.pos_now(boundary_neighbor))
                    .norm_squared()
                        + 0.01 * params.rest_density_grid_spacing.powi(2))
                    * K::kernel_gradient(
                        &r_vec,
                        params.kernel_support_radius,
                    );
            }
            *id_acceleration += accu;
        }
    );
}
