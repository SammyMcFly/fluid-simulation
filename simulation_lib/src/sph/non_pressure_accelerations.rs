/// Acceleration module
use nalgebra::Vector3;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::for_each;
use crate::sph::kernel::KernelFn;
use crate::sample::{Fluid3D, Boundary3D, Positional};
use crate::sph::SystemParameters;
use crate::sph::direction;

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
pub fn add_gravity(
    fluid: &mut Fluid3D,
) {
    for_each!(
        mut [fluid.acceleration],
        ref [],
        |_id, id_acceleration| {
            let strength_of_gravity = 9.81;
            // gravitate downwards
            let accu = Vector3::new(0.0, 0.0, -strength_of_gravity);
            // gravitate around point
            // let gravitation_center = Vector3::new(0.0, 0.0, 0.0);
            // let accu = strength_of_gravity*(gravitation_center-fluid.pos_now(id));

            *id_acceleration += accu;
        }
    );
}

/// Calculate spring acceleration at current time and add it to respective particles
#[cfg(feature = "springs")]
pub fn add_spring_acceleration(&mut self) {
    for Spring {
        indices: (i1, i2),
        k,
        l,
        ..
    } in &self.springs
    {
        // if cfg!(feature = "logging") {
        //     debug!("Calculate spring force");
        // }
        // calculate force for spring
        let force = k / l
            * ((self.particles[*i2].pos().now() - self.particles[*i1].pos().now())
                - (*l * (self.particles[*i2].pos().now() - self.particles[*i1].pos().now())
                    / (self.particles[*i2].pos().now() - self.particles[*i1].pos().now())
                        .norm()));

        let m: f64 = self.particles[*i1].mass();
        self.particles[*i1].add_acc(force / m);
        let m: f64 = self.particles[*i2].mass();
        self.particles[*i2].add_acc(-force / m);
    }
    // calculate other forces here
}

/// Calculate viscosity acceleration at current time and add it to respective particles
pub fn add_viscosity_acceleration<K: KernelFn>(
    fluid: &mut Fluid3D,
    boundary: &Boundary3D,
    params: &SystemParameters,
) {
    for_each!(
        mut [fluid.acceleration],
        ref [
            pos_now = fluid.position,
            vel_now = fluid.velocity,
            volume = fluid.volume,
            neighbors = fluid.neighbors,
            boundary_neighbors = fluid.boundary_neighbors
        ],
        |id, id_acceleration| {
            let mut accu = Vector3::zeros();
            // add viscostiy acceleration from other moving particles
            for &neighbor in &neighbors[id] {
                let r_vec = direction(
                    &pos_now[neighbor],
                    &pos_now[id],
                );
                let dist = r_vec.norm();
                accu += params.fluid_viscosity
                    * 2.
                    * (3. + 2.)
                    * volume[neighbor]
                    * (vel_now[id] - vel_now[neighbor])
                        .dot(&(pos_now[id] - pos_now[neighbor]))
                    / ((pos_now[id] - pos_now[neighbor])
                        .norm_squared()
                        + 0.01 * params.smoothing_length.powi(2))
                    * K::gradient(
                        &r_vec,
                        dist,
                        params.smoothing_length,
                    );
            }
            // add viscostiy acceleration from boundary particles
            for &boundary_neighbor in &boundary_neighbors[id] {
                let r_vec = direction(
                    boundary.pos_now(boundary_neighbor),
                    &pos_now[id],
                );
                let dist = r_vec.norm();
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
                        + 0.01 * params.smoothing_length.powi(2))
                    * K::gradient(
                        &r_vec,
                        dist,
                        params.smoothing_length,
                    );
            }
            *id_acceleration += accu;
        }
    );
}