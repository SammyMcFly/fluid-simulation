//! Acceleration module
use nalgebra::Vector3;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::fluid::Fluid3D;
use crate::for_each;
use crate::iteration::for_each_collect;
use crate::neighbor_search::NeighborList;
use crate::sph::SystemParameters;
use crate::sph::boundary_handling::BoundaryHandling;
use crate::sph::boundary_handling::ForceOntoBoundary;
use crate::sph::boundary_handling::RequestMode;
use crate::sph::kernel::KernelFn;
use crate::sph::vector;

/// reset acceleration, i. e. set it to 0.
pub fn reset_acceleration<B: BoundaryHandling>(fluid: &mut Fluid3D) {
    for_each!(
        mut [fluid.acceleration],
        ref [],
        |_id, id_acceleration| {
            *id_acceleration = Vector3::zeros();
        }
    );
}

/// Add gravity acceleration to all not boundary particles
pub fn add_gravity_acceleration<B: BoundaryHandling>(fluid: &mut Fluid3D, boundary: &mut B) {
    let strength_of_gravity = 9.81;
    // // gravitate downwards
    // for_each!(
    //     mut [fluid.acceleration],
    //     ref [],
    //     |_id, id_acceleration| {
    //         *id_acceleration +=  Vector3::new(0.0, 0.0, -strength_of_gravity);
    //     }
    // );
    // for b in boundary.iter_mut() {
    //     b.add_acceleration(Vector3::new(0.0, 0.0, -strength_of_gravity));
    // }
    // gravitate around point
    for_each!(
        mut [fluid.acceleration],
        ref [position = fluid.position],
        |id, id_acceleration| {
            use nalgebra::Point3;
            let gravitation_center = Point3::new(0.0, 0.0, 0.0);
            let direction = vector(&position[id], &gravitation_center);
            let direction_normalized = direction/direction.norm();
            *id_acceleration +=  strength_of_gravity*direction_normalized;
        }
    );
    for b in boundary.iter_mut() {
        use nalgebra::Point3;
        let gravitation_center = Point3::new(0.0, 0.0, 0.0);
        if let Some(cm) = b.center_of_mass() {
            let direction = vector(&cm, &gravitation_center);
            let direction_normalized = direction / direction.norm();
            b.add_acceleration(strength_of_gravity * direction_normalized);
        }
    }
}

/// Calculate viscosity acceleration at current time and add it to respective particles
pub fn add_viscosity_acceleration<K: KernelFn>(
    fluid: &mut Fluid3D,
    boundary: &mut impl BoundaryHandling,
    neighbors: &NeighborList,
    params: &SystemParameters,
) {
    let forces_onto_boundary: Vec<ForceOntoBoundary> = for_each_collect!(
        mut [fluid.acceleration],
        ref [
            pos_now = fluid.position,
            vel_now = fluid.velocity,
            mass = fluid.mass,
            volume = fluid.volume,
            neighbors = neighbors,
            boundary = boundary
        ],
        |id, id_acceleration, local_forces| {
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
            for (i, b) in boundary.iter().enumerate() {
                for &boundary_neighbor in b.get_neighbors(id, RequestMode::ViscosityAcceleration) {
                    let r_vec = vector(
                        b.pos_now(boundary_neighbor),
                        &pos_now[id],
                    );
                    let acceleration = params.boundary_viscosity
                        * 2.
                        * (3. + 2.)
                        * b.volume(boundary_neighbor)
                        * (vel_now[id] - *b.vel_now(boundary_neighbor))
                            .dot(
                                &(pos_now[id]
                                    - *b.pos_now(boundary_neighbor)),
                            )
                        / ((pos_now[id]
                            - *b.pos_now(boundary_neighbor))
                        .norm_squared()
                            + 0.01 * params.rest_density_grid_spacing.powi(2))
                        * K::kernel_gradient(
                            &r_vec,
                            params.kernel_support_radius,
                        );
                    if b.is_dynamic() {
                        local_forces.push(ForceOntoBoundary {
                            id: i,
                            force: -mass[id] * acceleration,
                            force_location: *b.pos_now(boundary_neighbor),
                        });
                    }
                    accu += acceleration;
                }
            }
            *id_acceleration += accu;
        }
    );
    for force in forces_onto_boundary {
        boundary.add_force_onto_boundary(force);
    }
}
