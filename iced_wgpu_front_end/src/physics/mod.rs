//! ## Physics based simulation backend
//!
//! Contains all necessary components to initialize a scene and simulate the trajectories
//! of its containing particles by propagating the system time.
//!
use std::sync::{Arc, Mutex};

use nalgebra::{Matrix3, Vector3};
use num_traits::identities::Zero;
use serde::Deserialize;
use std::io::Write;
use rayon::prelude::*;

// use tracing::{debug, error, info, span, trace, warn};
use tracing::{debug};


pub mod particle;
use particle::*;
pub mod spring;
use spring::*;
pub mod uniform_grid;

use crate::measure;

use super::setup;



/// Cubic B-spline kernel function
pub fn cubic_b_spline_3d(distance: f64, smoothing_length: f64) -> f64 {
    let normalized_distance = distance/smoothing_length;
    if normalized_distance < 1. {
        let prefactor = 1./4./std::f64::consts::PI/smoothing_length.powi(3);
        prefactor*((2.-normalized_distance).powi(3)-4.*(1.-normalized_distance).powi(3))
    } else if normalized_distance < 2. {
        let prefactor = 1./4./std::f64::consts::PI/smoothing_length.powi(3);
        prefactor*(2.-normalized_distance).powi(3)
    } else {
        0.
    }
}

/// Gradient of cubic B-spline kernel function
pub fn cubic_b_spline_3d_gradient(distance: f64, smoothing_length: f64, direction: Vector3<f64>) -> Vector3<f64> {
    let normalized_distance = distance/smoothing_length;
    if normalized_distance == 0. { // if distance is 0 direction is invalid -> return Vector3::zeros()
        Vector3::zeros()
    } else if normalized_distance < 1. {
        let prefactor = 1./4./std::f64::consts::PI/smoothing_length.powi(4);
        direction/distance*prefactor*(-3.*(2.-normalized_distance).powi(2)+12.*(1.-normalized_distance).powi(2))
    } else if normalized_distance < 2. {
        let prefactor = 1./4./std::f64::consts::PI/smoothing_length.powi(4);
        direction/distance*prefactor*(-3.*(2.-normalized_distance).powi(2))
    } else {
        Vector3::zeros()
    }
}


/// Method for propagating time in a simulated physical system
#[derive(Debug, Clone, Deserialize)]
pub enum PropagationMethod {
    ExplicitEuler,
    ImplicitEuler,
    EulerCromer,
    Verlet,
}

/// Configuration of a physical system to be simulated
#[derive(Debug, Clone)]
pub struct SystemProperties {
    time_inc: f64,
    particle_mass: f64,
    /// Smooting length h
    smoothing_length: f64,
    /// disable particles below this threshold
    disable_particles_below: f64,
    rest_density: f64, // rho_0
    /// Grid spacing when particles are ordered in a cubic grid at rest density
    rest_density_grid_spacing: f64,
    average_density: f64,
    fluid_depth: f64,
    viscosity: f64,
    stiffness: f64,
    kernel_fn: fn(distance: f64, particle_size: f64) -> f64,
    kernel_gradient_fn: fn(distance: f64, particle_size: f64, direction: Vector3<f64>) -> Vector3<f64>,
}

impl SystemProperties {
    pub fn new(
        time_inc: f64,
        rest_density: f64,
        rest_density_grid_spacing: f64,
        smoothing_length: f64,
        disable_particles_below: f64,
        fluid_depth: f64,
        viscosity: f64,
        stiffness: f64,
        kernel_fn: fn(distance: f64, particle_size: f64) -> f64,
        kernel_gradient_fn: fn(distance: f64, particle_size: f64, direction: Vector3<f64>) -> Vector3<f64>,
    ) -> Self {
        let average_density = 0.;
        Self {
            time_inc,
            particle_mass: rest_density*rest_density_grid_spacing.powi(3),
            smoothing_length,
            disable_particles_below,
            rest_density,
            rest_density_grid_spacing,
            average_density,
            fluid_depth,
            viscosity,
            stiffness,
            kernel_fn,
            kernel_gradient_fn,
        }
    }

    pub fn smoothing_length(&self) -> f64 {
        self.smoothing_length
    }
}

///  3D implementation of a physical system to be simulated
#[derive(Debug, Clone)]
pub struct System3D {
    /// Collection of all fluid particles
    pub particles: Vec<Particle3D>,
    /// Uniform grid for fluid particles
    ///
    /// Accelerates neighbor search
    particle_grid: uniform_grid::UniformGrid,
    /// Collection of all boundary (not moving) particles
    pub boundary_particles: Vec<BoundaryParticle3D>,
    /// Uniform grid for boundary particles
    ///
    /// Accelerates neighbor search
    boundary_particle_grid: uniform_grid::UniformGrid,
    /// Springs connecting different particles
    ///
    /// Spring stores indices of particles connected to via spring force,
    /// spring force coeff (k) and rest length (l)
    springs: Vec<Spring>,
    /// Time
    time_steps_propagated: u64,
    /// Properties of the system
    properties: SystemProperties,
    measurement_series: Option<Arc<Mutex<measure::MeasurementSeries>>>,
}

impl System3D {
    pub fn new(
        systemconfig: setup::System3DConfig,
    ) -> Self {
        let particle_grid = uniform_grid::UniformGrid::new(systemconfig.system_properties.smoothing_length());
        let mut boundary_particle_grid = uniform_grid::UniformGrid::new(systemconfig.system_properties.smoothing_length());
        boundary_particle_grid.populate_boundary_particles(&systemconfig.boundary_particles);
        let mut system = Self {
            particles: systemconfig.particles,
            particle_grid,
            boundary_particles: systemconfig.boundary_particles,
            boundary_particle_grid,
            springs: systemconfig.springs,
            time_steps_propagated: 0,
            properties: systemconfig.system_properties,
            measurement_series: systemconfig.measurement_series,
        };
        // set boundary mass such that the density is equal to the fluids rest density
        system.init_boundary_mass();
        // Update uniform grid
        system.update();
        // calculate initial accelerations
        system.calc_acceleration();
        // take initial measurement
        system.measure();
        system
    }

    fn init_boundary_mass(&mut self) {
        for boundary_particle_index in 0..self.boundary_particles.len() {
            // add inverse volume for every boundary neighbor
            let mut inverse_volume = 0.;
            // get boundary neighbors of boundary particles
            for boundary_neighbor in self.boundary_particle_grid.get_particles_in_kernel_range(
                    &self.boundary_particles[boundary_particle_index].pos(),
                    &self.boundary_particles
                ) {
                let distance = self.boundary_particles[boundary_particle_index].get_distance(&self.boundary_particles[boundary_neighbor].pos());
                inverse_volume += (self.properties.kernel_fn)(distance, self.properties.smoothing_length);
            }
            // calculate mass with rest density of fluid
            let pseudo_mass = self.properties.rest_density/inverse_volume;
            self.boundary_particles[boundary_particle_index].set_mass(pseudo_mass);
            // debug!("boundary particle {} has position: {}", boundary_particle_index, self.boundary_particles[boundary_particle_index].pos());
            // debug!("boundary particle {} has mass: {}", boundary_particle_index, self.boundary_particles[boundary_particle_index].mass());
        }
    }

    pub fn save_state(&self, file_path: &str) -> std::io::Result<()> {
        let ron_string = ron::to_string(&self.get_serializable_particles()).unwrap();
        let mut file = std::fs::File::create(file_path)?;
        file.write_all(ron_string.as_bytes())?;
        Ok(())
    }

    pub fn time(&self) -> f64 {
        (self.time_steps_propagated as f64)*self.properties.time_inc
    }

    /// Calculate 2-norm of maximum velocity of any particle
    fn calc_max_speed(&self) -> f64 {
        let mut max_speed = 0.;
        for particle in &self.particles {
            if particle.is_enabled() {
                let speed = particle.vel().now().norm();
                if speed > max_speed {
                    max_speed = speed;
                }
            }
        }
        // debug!("Maximum speed: {}", max_speed);
        max_speed
    }

    /// Calculate average kinetic energy for all fluid particles
    fn calc_average_kinetic_energy(&self) -> f64 {
        let mut average_kin_energy = 0.;
        let mut count = 0.;
        for particle in &self.particles {
            if particle.is_enabled() {
                average_kin_energy += 1./2.*particle.mass()*particle.vel().now().norm_squared();
                count += 1.;
            }
        }
        if count != 0. {
            average_kin_energy /= count;
        }
        // debug!("Average kin. energy: {}", average_kin_energy);
        average_kin_energy
    }

    /// Calculate average mass density for all fluid particles
    fn calc_average_mass_density(&self) -> f64 {
        let mut average_density = 0.;
        let mut count = 0.;
        for particle in &self.particles {
            if particle.is_enabled() {
                if particle.density() > self.properties.rest_density {
                    average_density += particle.density();
                } else {
                    average_density += self.properties.rest_density;

                }
                count += 1.;
            }
        }
        if count != 0. {
            average_density /= count;
        }
        // debug!("Average density: {}, rest density: {}", average_density, self.properties.rest_density);
        average_density
    }

    fn update_particle_neighbors(&mut self) {
        // for particle_index in 0..self.particles.len() {
        //     if self.particles[particle_index].is_enabled() {
        //         // update neighbors
        //         let neighbors = self.particle_grid.get_particles_in_kernel_range(&self.particles[particle_index].pos().now(), &self.particles);
        //         self.particles[particle_index].set_neighbors(neighbors);
        //         // update boundary neighbors
        //         let boundary_neighbors = self.boundary_particle_grid.get_particles_in_kernel_range(&self.particles[particle_index].pos().now(), &self.boundary_particles);
        //         self.particles[particle_index].set_boundary_neighbors(boundary_neighbors);
        //     }
        // }
        let immutable_clone_of_particles = self.particles.clone();
        self.particles.par_iter_mut().for_each(|particle| {
            if particle.is_enabled() {
                // update neighbors
                let neighbors = self.particle_grid.get_particles_in_kernel_range(&particle.pos().now(), &immutable_clone_of_particles);
                particle.set_neighbors(neighbors);
                // update boundary neighbors
                let boundary_neighbors = self.boundary_particle_grid.get_particles_in_kernel_range(&particle.pos().now(), &self.boundary_particles);
                particle.set_boundary_neighbors(boundary_neighbors);
            }
        });
    }

    fn update_density(&mut self) {
        // for particle_index in 0..self.particles.len() {
        //     if self.particles[particle_index].is_enabled() {
        //         // reset density
        //         self.particles[particle_index].set_density(0.);
        //         // add density for every neighbor
        //         for &neighbor in &self.particles[particle_index].neighbors.clone() {
        //             let distance = self.particles[particle_index].get_distance(&self.particles[neighbor].pos().now());
        //             let density = self.particles[neighbor].mass()
        //                 *(self.properties.kernel_fn)(distance, self.properties.smoothing_length);
        //             self.particles[particle_index].add_density(density);
        //         }
        //         // add density for every boundary neighbor (mirror mass of moving particle onto boundary particle)
        //         for &boundary_neighbor in &self.particles[particle_index].boundary_neighbors.clone() {
        //             // add density for every neighbor
        //             let distance = self.particles[particle_index].get_distance(&self.boundary_particles[boundary_neighbor].pos());
        //             let density =
        //                 // self.particles[particle_index].mass()
        //                 self.boundary_particles[boundary_neighbor].mass()
        //                 *(self.properties.kernel_fn)(distance, self.properties.smoothing_length);
        //             self.particles[particle_index].add_density(density);
        //         }
        //         // debug!("density: {}", self.particles[particle_index].density());
        //     }
        // }
        let immutable_clone_of_particles = self.particles.clone();
        self.particles.par_iter_mut().for_each(|particle| {
            if particle.is_enabled() {
                // reset density
                particle.set_density(0.);
                // add density for every neighbor
                for &neighbor in &particle.neighbors.clone() {
                    let distance = particle.get_distance(&immutable_clone_of_particles[neighbor].pos().now());
                    let density = immutable_clone_of_particles[neighbor].mass()
                        *(self.properties.kernel_fn)(distance, self.properties.smoothing_length);
                    particle.add_density(density);
                }
                // add density for every boundary neighbor (mirror mass of moving particle onto boundary particle)
                for &boundary_neighbor in &particle.boundary_neighbors.clone() {
                    // add density for every neighbor
                    let distance = particle.get_distance(&self.boundary_particles[boundary_neighbor].pos());
                    let density =
                        self.boundary_particles[boundary_neighbor].mass()
                        *(self.properties.kernel_fn)(distance, self.properties.smoothing_length);
                    particle.add_density(density);
                }
                // debug!("density: {}", particle.density());
            }
        });
    }

    fn update_pressure(&mut self) {
        for particle_index in 0..self.particles.len() {
            if self.particles[particle_index].is_enabled() {
                let pressure = self.properties.stiffness*f64::max(
                    self.particles[particle_index].density()/self.properties.rest_density - 1.,
                    0.
                );
                self.particles[particle_index].set_pressure(pressure);
                // debug!("pressure: {}", pressure);
            }
        }
    }

    /// Add gravity acceleration to all not boundary particles
    fn add_gravity(&mut self) {
        for particle in &mut self.particles {
            particle.add_acc(Vector3::new(0.0, 0.0, -9.81));
        }
    }

    /// Calculate spring acceleration at current time and add it to respective particles
    fn add_spring_acceleration(&mut self) {
        for Spring { indices: (i1, i2), k, l, ..} in &self.springs {
            // debug!("Calculate spring force");
            // calculate force for spring
            let force = k/l
                *((self.particles[*i2].pos().now()-self.particles[*i1].pos().now())
                - (*l*(self.particles[*i2].pos().now()-self.particles[*i1].pos().now())
                /(self.particles[*i2].pos().now()-self.particles[*i1].pos().now()).norm()));

            let m: f64 = self.particles[*i1].mass();
            self.particles[*i1].add_acc(force/m);
            let m: f64 = self.particles[*i2].mass();
            self.particles[*i2].add_acc(-force/m);
        }
        // calculate other forces here
    }

    /// Calculate viscosity acceleration at current time and add it to respective particles
    fn add_viscosity_acceleration(&mut self) {
        // for particle_index in 0..self.particles.len() {
        //     if self.particles[particle_index].is_enabled() {
        //         // add viscostiy acceleration from other moving particles
        //         for &neighbor in &self.particles[particle_index].neighbors.clone() {
        //             let distance = self.particles[particle_index].get_distance(&self.particles[neighbor].pos().now());
        //             let direction = self.particles[particle_index].get_direction(&self.particles[neighbor].pos().now());
        //             let acc = self.properties.viscosity*2.*self.particles[neighbor].mass()/self.particles[particle_index].density()
        //                 *(self.particles[particle_index].vel().now()-self.particles[neighbor].vel().now()).dot(&(self.particles[particle_index].pos().now()-self.particles[neighbor].pos().now()))
        //                 /((self.particles[particle_index].pos().now()-self.particles[neighbor].pos().now()).norm_squared()+0.01*self.properties.smoothing_length.powi(2))
        //                 *(self.properties.kernel_gradient_fn)(distance, self.properties.smoothing_length, direction);
        //             self.particles[particle_index].add_acc(acc);
        //         }
        //         // add viscostiy acceleration from boundary particles
        //         for &boundary_neighbor in &self.particles[particle_index].boundary_neighbors.clone() {
        //             let distance = self.particles[particle_index].get_distance(&self.boundary_particles[boundary_neighbor].pos());
        //             let direction = self.particles[particle_index].get_direction(&self.boundary_particles[boundary_neighbor].pos());
        //             let acc = self.properties.viscosity*2.*self.particles[particle_index].mass()/self.particles[particle_index].density()
        //                 *(self.particles[particle_index].vel().now()).dot(&(self.particles[particle_index].pos().now()-self.boundary_particles[boundary_neighbor].pos()))
        //                 /((self.particles[particle_index].pos().now()-self.boundary_particles[boundary_neighbor].pos()).norm_squared()+0.01*self.properties.smoothing_length.powi(2))
        //                 *(self.properties.kernel_gradient_fn)(distance, self.properties.smoothing_length, direction);
        //             self.particles[particle_index].add_acc(acc);
        //         }
        //     }
        // }
        let immutable_clone_of_particles = self.particles.clone();
        self.particles.par_iter_mut().for_each(|particle| {
            if particle.is_enabled() {
                // add viscostiy acceleration from other moving particles
                for &neighbor in &particle.neighbors.clone() {
                    let distance = particle.get_distance(&immutable_clone_of_particles[neighbor].pos().now());
                    let direction = particle.get_direction(&immutable_clone_of_particles[neighbor].pos().now());
                    let acc = self.properties.viscosity*2.*(3.+2.)*immutable_clone_of_particles[neighbor].mass()/particle.density()
                        *(particle.vel().now()-immutable_clone_of_particles[neighbor].vel().now()).dot(&(particle.pos().now()-immutable_clone_of_particles[neighbor].pos().now()))
                        /((particle.pos().now()-immutable_clone_of_particles[neighbor].pos().now()).norm_squared()+0.01*self.properties.smoothing_length.powi(2))
                        *(self.properties.kernel_gradient_fn)(distance, self.properties.smoothing_length, direction);
                    particle.add_acc(acc);
                }
                // add viscostiy acceleration from boundary particles
                for &boundary_neighbor in &particle.boundary_neighbors.clone() {
                    let distance = particle.get_distance(&self.boundary_particles[boundary_neighbor].pos());
                    let direction = particle.get_direction(&self.boundary_particles[boundary_neighbor].pos());
                    let acc = self.properties.viscosity*2.*(3.+2.)*self.boundary_particles[boundary_neighbor].mass()/self.properties.rest_density
                        *(particle.vel().now()).dot(&(particle.pos().now()-self.boundary_particles[boundary_neighbor].pos()))
                        /((particle.pos().now()-self.boundary_particles[boundary_neighbor].pos()).norm_squared()+0.01*self.properties.smoothing_length.powi(2))
                        *(self.properties.kernel_gradient_fn)(distance, self.properties.smoothing_length, direction);
                    particle.add_acc(acc);
                }
            }
        });
    }

    /// Calculate pressure acceleration at current time and add it to respective particles
    fn add_pressure_acceleration(&mut self) {
        // for particle_index in 0..self.particles.len() {
        //     if self.particles[particle_index].is_enabled() {
        //         // let mut test_kernel_gradient = Vector3::zero();
        //         // add pressure acceleration from other moving particles
        //         for &neighbor in &self.particles[particle_index].neighbors.clone() {
        //             let distance = self.particles[particle_index].get_distance(&self.particles[neighbor].pos().now());
        //             let direction = self.particles[particle_index].get_direction(&self.particles[neighbor].pos().now());
        //             let acc = -self.particles[neighbor].mass()
        //                 *(self.particles[particle_index].pressure()/self.particles[particle_index].density().powi(2) + self.particles[neighbor].pressure()/self.particles[neighbor].density().powi(2))
        //                 *(self.properties.kernel_gradient_fn)(distance, self.properties.smoothing_length, direction);
        //             self.particles[particle_index].add_acc(acc);
        //             // test_kernel_gradient += acc;
        //         }
        //         // add pressure acceleration from boundary particles
        //         for &boundary_neighbor in &self.particles[particle_index].boundary_neighbors.clone() {
        //             let distance = self.particles[particle_index].get_distance(&self.boundary_particles[boundary_neighbor].pos());
        //             let direction = self.particles[particle_index].get_direction(&self.boundary_particles[boundary_neighbor].pos());
        //             let acc = -self.boundary_particles[boundary_neighbor].mass()
        //                 // mirror pressure and density
        //                 // *2.*self.particles[particle_index].pressure()/self.particles[particle_index].density().powi(2)
        //                 // mirror only pressure, set density to rest density
        //                 *self.particles[particle_index].pressure()*(1./self.particles[particle_index].density().powi(2)+1./self.properties.rest_density.powi(2))
        //                 *(self.properties.kernel_gradient_fn)(distance, self.properties.smoothing_length, direction);
        //             self.particles[particle_index].add_acc(acc);
        //             // test_kernel_gradient += acc;
        //         }
        //         // debug!("kernel gradient: {}", test_kernel_gradient);
        //     }
        // }
        let immutable_clone_of_particles = self.particles.clone();
        self.particles.par_iter_mut().for_each(|particle| {
            if particle.is_enabled() {
                // let mut test_kernel_gradient = Vector3::zero();
                // add pressure acceleration from other moving particles
                for &neighbor in &particle.neighbors.clone() {
                    let distance = particle.get_distance(&immutable_clone_of_particles[neighbor].pos().now());
                    let direction = particle.get_direction(&immutable_clone_of_particles[neighbor].pos().now());
                    let acc =
                        -immutable_clone_of_particles[neighbor].mass()
                        *(particle.pressure()/particle.density().powi(2) + immutable_clone_of_particles[neighbor].pressure()/immutable_clone_of_particles[neighbor].density().powi(2))
                        *(self.properties.kernel_gradient_fn)(distance, self.properties.smoothing_length, direction);
                    particle.add_acc(acc);
                    // test_kernel_gradient += acc;
                }
                // add pressure acceleration from boundary particles
                for &boundary_neighbor in &particle.boundary_neighbors.clone() {
                    let distance = particle.get_distance(&self.boundary_particles[boundary_neighbor].pos());
                    let direction = particle.get_direction(&self.boundary_particles[boundary_neighbor].pos());
                    // mirror pressure into boundary particle
                    // -particle.mass()
                    let acc =
                        -self.boundary_particles[boundary_neighbor].mass()
                        *particle.pressure()*(1./particle.density().powi(2)+1./self.properties.rest_density.powi(2))
                        *(self.properties.kernel_gradient_fn)(distance, self.properties.smoothing_length, direction);
                    particle.add_acc(acc);
                    // test_kernel_gradient += acc;
                }
                // debug!("kernel gradient: {}", test_kernel_gradient);
            }
        });
    }

    /// Calculate acceleration at current time
    ///
    /// Supports: Gravity, spring force, viscosity and pressure acceleration
    fn calc_acceleration(&mut self) {
        // update neighbors field of all fluid particles
        self.update_particle_neighbors();
        // compute density and pressure
        self.update_density();
        self.update_pressure();
        // reset acceleration
        for particle in &mut self.particles {
            particle.set_acc(Vector3::zero());
        }
        // add gravity acceleration
        self.add_gravity();
        // add spring acceleration
        self.add_spring_acceleration();
        // add viscosity acceleration
        self.add_viscosity_acceleration();
        // add pressure acceleration
        self.add_pressure_acceleration();
    }

    pub fn inc_time(&mut self, method: &PropagationMethod) {
        // increment time one step
        match method {
            PropagationMethod::ExplicitEuler => {
                for particle in &mut self.particles {
                    if particle.is_enabled() {
                        // update positions
                        particle.set_new_pos(particle.pos().now() + self.properties.time_inc*particle.vel().now());
                        // update velocities
                        particle.set_new_vel(particle.vel().now() + self.properties.time_inc*particle.acc());
                    }
                }
            },
            PropagationMethod::ImplicitEuler => { // Conjugate Gradient implementation
                // init fractions of the Jacobi matrix that belong to the springs
                for Spring {
                        indices: (i1, i2),
                        k,
                        l,
                        matrix_s
                } in &mut self.springs {
                    // calculate spacial derivative of spring force of spring
                    // between vert[i1] and vert[i2] applied to vert[i1] with respect to vert[i1].pos
                    let x_i2_outer_x_i1 =
                            (self.particles[*i2].pos().now()-self.particles[*i1].pos().now())
                            .outer(&(self.particles[*i2].pos().now()-self.particles[*i1].pos().now()));

                    *matrix_s = *k / *l
                        *(-Matrix3::identity()+ *l /(self.particles[*i2].pos().now()-self.particles[*i1].pos().now()).norm()
                        *(Matrix3::identity()-1.0/(self.particles[*i2].pos().now()-self.particles[*i1].pos().now()).norm().powi(2)
                        *x_i2_outer_x_i1));
                }
                // init variables for iterative numeric solver
                for v in &mut self.particles {
                    let vel = v.vel().now();
                    v.set_pred_vel(vel);
                    v.d_l = v.vel().now()+self.properties.time_inc*v.acc()-v.vel().pred();
                }
                for Spring {
                        indices: (i1, i2),
                        matrix_s,
                        ..
                } in &self.springs {
                    let v_pred = self.particles[*i1].vel().pred();
                    let m = self.particles[*i1].mass();
                    self.particles[*i1].d_l += self.properties.time_inc.powi(2)/m * (*matrix_s) * v_pred;

                    let v_pred = self.particles[*i2].vel().pred();
                    let m = self.particles[*i2].mass();
                    self.particles[*i2].d_l -= self.properties.time_inc.powi(2)/m * (*matrix_s) * v_pred;
                }
                for v in &mut self.particles {
                    v.r_l = v.d_l;
                }
                // solve numerically iteratively
                for _ in 0..5 {
                    // refresh a_times_d_i
                    for v in &mut self.particles {
                        v.a_times_d_l = v.d_l;
                    }
                    for Spring {
                            indices: (i1, i2),
                            matrix_s, ..} in &self.springs {
                        let d_l = self.particles[*i1].d_l;
                        let m = self.particles[*i1].mass();
                        self.particles[*i1].d_l -= self.properties.time_inc.powi(2)/m * (*matrix_s) * d_l;

                        let d_l = self.particles[*i2].d_l;
                        let m = self.particles[*i2].mass();
                        self.particles[*i2].d_l += self.properties.time_inc.powi(2)/m * (*matrix_s) * d_l;
                    }
                    // do numeric solver iteration
                    for v in &mut self.particles {
                        v.alpha_l = v.r_l.dot(&v.r_l)/(v.d_l.dot(&v.a_times_d_l));
                        let vel = v.vel().pred()+v.alpha_l*v.d_l;
                        v.set_pred_vel(vel);
                        let r_l_old = v.r_l;
                        v.r_l -= v.alpha_l*v.a_times_d_l;
                        v.d_l = v.r_l + v.r_l.dot(&v.r_l)/(r_l_old.dot(&r_l_old))*v.d_l;
                    }
                }

                for v in &mut self.particles {
                    // function produces NaN values for a 0 acceleration
                    // this check prevents spreading of NaN values
                    if v.acc() != Vector3::zeros() {
                        // set velocity from numeric solver as new velocity
                        v.accept_pred_vel();
                    }
                    // update positions with new velocities: v_i(t+h)
                    v.set_new_pos(v.pos().now() + self.properties.time_inc*v.vel().now());
                }

            }
            PropagationMethod::EulerCromer => {
                for particle in &mut self.particles {
                    if particle.is_enabled() {
                        // update velocities
                        particle.set_new_vel(particle.vel().now() + self.properties.time_inc*particle.acc());
                        // update positions with new velocities: v_i(t+h)
                        particle.set_new_pos(particle.pos().now() + self.properties.time_inc*particle.vel().now());
                    }
                }
            },
            PropagationMethod::Verlet => {
                // update positions
                for particle in &mut self.particles {
                    if particle.is_enabled() {
                        // update positions
                        let t = 2.0*particle.pos().now() - particle.pos().prev() + self.properties.time_inc.powi(2)*particle.acc();
                        particle.set_new_pos(t);
                        // update velocities
                        let t = (particle.pos().now() - particle.pos().prev())/self.properties.time_inc;
                        particle.set_new_vel(t);
                    }
                }
            },
        }
        self.time_steps_propagated += 1;
        // Update uniform grid
        self.update();
        // calculate new accelerations
        self.calc_acceleration();
        // Measure (physical) quantities at current time step
        self.measure();
    }

    /// Update uniform grid and disabled particles
    fn update(&mut self) {
        // disable irrelevant particles (NOTE: Disabled particles must not be connected via spring)
        for particle in &mut self.particles {
            if particle.pos().now()[2] < self.properties.disable_particles_below {
                particle.disable();
            }
        }
        // update uniform grid of fluid particles
        self.particle_grid.clear();
        self.particle_grid.populate(&self.particles);
    }

    fn measure(&mut self) {
        self.properties.average_density = self.calc_average_mass_density();
        // debug!("{}, {}", self.properties.average_density, self.properties.rest_density);
        // self.properties.max_speed =
        // let cfl_coeff = self.calc_max_speed()*self.properties.time_inc/self.properties.rest_density_grid_spacing;
        // debug!("cfl coefficient: {}", cfl_coeff);
        if let Some(ms) = &self.measurement_series {
            let measurement = measure::Measurement {
                time: self.time(),
                density: self.properties.average_density/self.properties.rest_density,
                kinetic_energy: self.calc_average_kinetic_energy(),
                stiffness: self.properties.stiffness,
                viscosity: self.properties.viscosity,
                fluid_depth: self.properties.fluid_depth,
                rest_density_grid_spacing: self.properties.rest_density_grid_spacing,
                smoothing_length: self.properties.smoothing_length,
                rest_density: self.properties.rest_density,
                time_step_size: self.properties.time_inc,
            };
            ms.lock().unwrap().push_back(measurement);
        }
    }

    pub fn get_average_mass_density(&self) -> f32 {
        self.properties.average_density as f32
    }

    fn get_serializable_particles(&self) -> Vec<SerParticle3D> {
        self.particles.clone().into_iter().map(|p| p.into()).collect()
    }
}

pub trait Outer {
    type OuterProductType;
    fn outer(&self, other: &Self) -> Self::OuterProductType;
}

impl<N: Copy + std::ops::Mul<N, Output=N> + Zero> Outer for Vector3<N> {
    type OuterProductType = Matrix3<N>;

    fn outer(&self, other: &Self) -> Self::OuterProductType {
        Matrix3::new(
            self[0]*other[0],
            self[0]*other[1],
            self[0]*other[2],
            self[1]*other[0],
            self[1]*other[1],
            self[1]*other[2],
            self[2]*other[0],
            self[2]*other[1],
            self[2]*other[2])
    }
}
