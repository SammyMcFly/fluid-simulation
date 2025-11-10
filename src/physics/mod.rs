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
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "logging")]
use tracing::{debug}; // debug, error, info, span, trace, warn,

pub mod particle;
use particle::*;

#[cfg(feature = "springs")]
pub mod spring;
#[cfg(feature = "springs")]
use spring::*;

pub mod uniform_grid;

use crate::measure;

use super::setup;

/// Calculate the distance between two 3D points
pub fn distance(from: &Vector3<f64>, to: &Vector3<f64>) -> f64 {
    (to - from).norm()
}

/// Direction from particle1 towards particle2
pub fn direction(from: &Vector3<f64>, towards: &Vector3<f64>) -> Vector3<f64> {
    towards - from
}

/// Cubic B-spline kernel function
pub fn cubic_b_spline_3d(position_1: &Vector3<f64>, position_2: &Vector3<f64>, smoothing_length: f64) -> f64 {
    let distance = distance(position_1, position_2);
    // normalize
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
pub fn cubic_b_spline_3d_gradient(position_1: &Vector3<f64>, position_2: &Vector3<f64>, smoothing_length: f64) -> Vector3<f64> {
    // calculate distance between positions
    let distance = distance(position_1, position_2);
    // calculate direction direction from position 2 to 1
    let inv_direction = direction(position_2, position_1);
    // normalize
    let normalized_distance = distance/smoothing_length;
    if normalized_distance == 0. { // if distance is 0 direction is invalid -> return Vector3::zeros()
        Vector3::zeros()
    } else if normalized_distance < 1. {
        let prefactor = 1./4./std::f64::consts::PI/smoothing_length.powi(4);
        inv_direction/distance*prefactor*(-3.*(2.-normalized_distance).powi(2)+12.*(1.-normalized_distance).powi(2))
    } else if normalized_distance < 2. {
        let prefactor = 1./4./std::f64::consts::PI/smoothing_length.powi(4);
        inv_direction/distance*prefactor*(-3.*(2.-normalized_distance).powi(2))
    } else {
        Vector3::zeros()
    }
}


/// Method for propagating time in a simulated physical system
#[derive(Debug, Clone, Deserialize)]
pub enum PropagationMethod {
    ExplicitEuler,
    #[cfg(feature = "implicit_euler")]
    ImplicitEuler,
    EulerCromer,
    Verlet,
}

/// Configuration of a physical system to be simulated
#[derive(Debug, Clone)]
pub struct SystemProperties {
    time_inc: f64,
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
    boundary_pressure_acceleration_weighting: f64,
    #[cfg(feature = "local_pressure")]
    stiffness: f64,
    #[cfg(feature = "global_pressure")]
    relaxation_factor: f64,
    #[cfg(feature = "global_pressure")]
    min_diagonal_element: f64,
    kernel_fn: fn(position_1: &Vector3<f64>, position_2: &Vector3<f64>, smoothing_length: f64) -> f64,
    kernel_gradient_fn: fn(position_1: &Vector3<f64>, position_2: &Vector3<f64>, smoothing_length: f64) -> Vector3<f64>,
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
        boundary_pressure_acceleration_weighting: f64,
        #[cfg(feature = "local_pressure")]
        stiffness: f64,
        #[cfg(feature = "global_pressure")]
        relaxation_factor: f64,
        #[cfg(feature = "global_pressure")]
        min_diagonal_element: f64,
        kernel_fn: fn(position_1: &Vector3<f64>, position_2: &Vector3<f64>, smoothing_length: f64) -> f64,
        kernel_gradient_fn: fn(position_1: &Vector3<f64>, position_2: &Vector3<f64>, smoothing_length: f64) -> Vector3<f64>,
    ) -> Self {
        let average_density = 0.;
        Self {
            time_inc,
            smoothing_length,
            disable_particles_below,
            rest_density,
            rest_density_grid_spacing,
            average_density,
            fluid_depth,
            viscosity,
            boundary_pressure_acceleration_weighting,
            #[cfg(feature = "local_pressure")]
            stiffness,
            #[cfg(feature = "global_pressure")]
            relaxation_factor,
            #[cfg(feature = "global_pressure")]
            min_diagonal_element,
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
    #[cfg(feature = "springs")]
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
            #[cfg(feature = "springs")]
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

    /// Calculate and set pseudo mass of all boundary particles
    #[cfg(not(feature = "pseudo_mass_boundary"))]
    fn init_boundary_mass(&mut self) {
        for boundary_particle_index in 0..self.boundary_particles.len() {
            // simple mass
            self.boundary_particles[boundary_particle_index].set_mass(self.properties.rest_density*self.properties.rest_density_grid_spacing.powi(3));
        }
    }

        /// Calculate and set pseudo mass of all boundary particles
    #[cfg(feature = "pseudo_mass_boundary")]
    fn init_boundary_mass(&mut self) {
        for boundary_particle_index in 0..self.boundary_particles.len() {
            // add inverse volume for every boundary neighbor
            let mut inverse_volume = 0.;
            // get boundary neighbors of boundary particles
            for boundary_neighbor in self.boundary_particle_grid.get_particles_in_kernel_range(
                    &self.boundary_particles[boundary_particle_index].pos(),
                    &self.boundary_particles
                ) {
                inverse_volume += (self.properties.kernel_fn)(
                    &self.boundary_particles[boundary_particle_index].pos(),
                    &self.boundary_particles[boundary_neighbor].pos(),
                    self.properties.smoothing_length,
                );
            }
            // calculate mass with rest density of fluid
            let pseudo_mass = self.properties.rest_density/inverse_volume;
            self.boundary_particles[boundary_particle_index].set_mass(pseudo_mass);
            // if cfg!(feature = "logging") {
                // debug!("boundary particle {} has position: {}", boundary_particle_index, self.boundary_particles[boundary_particle_index].pos());
                // debug!("boundary particle {} has mass: {}", boundary_particle_index, self.boundary_particles[boundary_particle_index].mass());
            // }
        }
    }

    /// Store the current state of all fluid particles to a file
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
        // if cfg!(feature = "logging") {
            // debug!("Maximum speed: {}", max_speed);
        // }
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
        // if cfg!(feature = "logging") {
        //     debug!("Average kin. energy: {}", average_kin_energy);
        // }
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
        // if cfg!(feature = "logging") {
            // debug!("Average density: {}, rest density: {}", average_density, self.properties.rest_density);
        // }
        average_density
    }

    /// Perform neighbor search for all fluid particles
    #[cfg(not(feature = "parallel"))]
    fn update_particle_neighbors(&mut self) {
        for particle_index in 0..self.particles.len() {
            if self.particles[particle_index].is_enabled() {
                // update neighbors
                let neighbors = self.particle_grid.get_particles_in_kernel_range(&self.particles[particle_index].pos().now(), &self.particles);
                self.particles[particle_index].set_neighbors(neighbors);
                // update boundary neighbors
                let boundary_neighbors = self.boundary_particle_grid.get_particles_in_kernel_range(&self.particles[particle_index].pos().now(), &self.boundary_particles);
                self.particles[particle_index].set_boundary_neighbors(boundary_neighbors);
            }
        }
    }

    /// Perform neighbor search for all fluid particles
    #[cfg(feature = "parallel")]
    fn update_particle_neighbors(&mut self) {
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

    /// Calculate and update density for all particles for the current point in time
    #[cfg(not(feature = "parallel"))]
    fn update_density(&mut self) {
        for particle_index in 0..self.particles.len() {
            if self.particles[particle_index].is_enabled() {
                // reset density
                self.particles[particle_index].set_density(0.);
                // add density for every neighbor
                for &neighbor in &self.particles[particle_index].neighbors.clone() {
                    let density = self.particles[neighbor].mass()
                        *(self.properties.kernel_fn)(
                            &self.particles[particle_index].pos().now(),
                            &self.particles[neighbor].pos().now(),
                            self.properties.smoothing_length,
                        );
                    self.particles[particle_index].add_density(density);
                }
                // add density for every boundary neighbor (mirror mass of moving particle onto boundary particle)
                for &boundary_neighbor in &self.particles[particle_index].boundary_neighbors.clone() {
                    // add density for every neighbor
                    let density =
                        // self.particles[particle_index].mass()
                        self.boundary_particles[boundary_neighbor].mass()
                        *(self.properties.kernel_fn)(
                            &self.particles[particle_index].pos().now(),
                            &self.boundary_particles[boundary_neighbor].pos(),
                            self.properties.smoothing_length,
                        );
                    self.particles[particle_index].add_density(density);
                }
                // if cfg!(feature = "logging") {
                //     debug!("density: {}", self.particles[particle_index].density());
                // }
            }
        }
    }

    /// Calculate and update density for all particles for the current point in time
    #[cfg(feature = "parallel")]
    fn update_density(&mut self) {
        let immutable_clone_of_particles = self.particles.clone();
        self.particles.par_iter_mut().for_each(|particle| {
            if particle.is_enabled() {
                // reset density
                particle.set_density(0.);
                // add density for every neighbor
                for &neighbor in &particle.neighbors.clone() {
                    let density = immutable_clone_of_particles[neighbor].mass()
                        *(self.properties.kernel_fn)(
                            &particle.pos().now(),
                            &immutable_clone_of_particles[neighbor].pos().now(),
                            self.properties.smoothing_length,
                        );
                    particle.add_density(density);
                }
                // add density for every boundary neighbor (mirror mass of moving particle onto boundary particle)
                for &boundary_neighbor in &particle.boundary_neighbors.clone() {
                    // add density for every neighbor
                    let density =
                        self.boundary_particles[boundary_neighbor].mass()
                        *(self.properties.kernel_fn)(
                            &particle.pos().now(),
                            &self.boundary_particles[boundary_neighbor].pos(),
                            self.properties.smoothing_length,
                        );
                    particle.add_density(density);
                }
                // if cfg!(feature = "logging") {
                    // debug!("density: {}", particle.density());
                // }
            }
        });
    }

    // perform splitting step conditionally
    #[cfg(all(not(feature = "parallel"), feature = "splitting"))]
    fn calc_predicted_density(&mut self) {
        for particle_index in 0..self.particles.len() {
            if self.particles[particle_index].is_enabled() {
                // reset density
                self.particles[particle_index].pred_density = 0.;
                // add density for every neighbor
                for &neighbor in &self.particles[particle_index].neighbors.clone() {
                    let density = self.particles[neighbor].mass()
                        *(self.properties.kernel_fn)(
                            &self.particles[particle_index].pos().now(),
                            &self.particles[neighbor].pos().now(),
                            self.properties.smoothing_length,
                        )
                        +self.properties.time_inc*(self.particles[particle_index].vel().pred()-self.particles[neighbor].vel().pred())
                        .dot(&(self.properties.kernel_gradient_fn)(
                            &self.particles[particle_index].pos().now(),
                            &self.particles[neighbor].pos().now(),
                            self.properties.smoothing_length,
                        ));
                    self.particles[particle_index].pred_density += density;
                }
                // add density for every boundary neighbor (mirror mass of moving particle onto boundary particle)
                for &boundary_neighbor in &self.particles[particle_index].boundary_neighbors.clone() {
                    // add density for every neighbor
                    let density =
                        // self.particles[particle_index].mass()
                        self.boundary_particles[boundary_neighbor].mass()
                        *(self.properties.kernel_fn)(
                            &self.particles[particle_index].pos().now(),
                            &self.boundary_particles[boundary_neighbor].pos(),
                            self.properties.smoothing_length,
                        )
                        +self.properties.time_inc*self.particles[particle_index].vel().pred()
                        .dot(&(self.properties.kernel_gradient_fn)(
                            &self.particles[particle_index].pos().now(),
                            &self.boundary_particles[boundary_neighbor].pos(),
                            self.properties.smoothing_length,
                        ));
                    self.particles[particle_index].pred_density += density;
                }
                // if cfg!(feature = "logging") {
                //     debug!("density: {}", self.particles[particle_index].density());
                // }
            }
        }
    }

    // perform splitting step conditionally
    #[cfg(all(feature = "parallel", feature = "splitting"))]
    fn calc_predicted_density(&mut self) {
        let immutable_clone_of_particles = self.particles.clone();
        self.particles.par_iter_mut().for_each(|particle| {
            if particle.is_enabled() {
                // reset density
                particle.pred_density = 0.;
                // add density for every neighbor
                for &neighbor in &particle.neighbors.clone() {
                    let density = immutable_clone_of_particles[neighbor].mass()
                        *(self.properties.kernel_fn)(
                            &particle.pos().now(),
                            &immutable_clone_of_particles[neighbor].pos().now(),
                            self.properties.smoothing_length,
                        )
                        +self.properties.time_inc*(particle.vel().pred()-immutable_clone_of_particles[neighbor].vel().pred())
                        .dot(&(self.properties.kernel_gradient_fn)(
                            &particle.pos().now(),
                            &immutable_clone_of_particles[neighbor].pos().now(),
                            self.properties.smoothing_length,
                        ));
                    particle.pred_density += density;
                }
                // add density for every boundary neighbor (mirror mass of moving particle onto boundary particle)
                for &boundary_neighbor in &particle.boundary_neighbors.clone() {
                    // add density for every neighbor
                    let density =
                        self.boundary_particles[boundary_neighbor].mass()
                        *(self.properties.kernel_fn)(
                            &particle.pos().now(),
                            &self.boundary_particles[boundary_neighbor].pos(),
                            self.properties.smoothing_length,
                        )
                        +self.properties.time_inc*particle.vel().pred()
                        .dot(&(self.properties.kernel_gradient_fn)(
                            &particle.pos().now(),
                            &self.boundary_particles[boundary_neighbor].pos(),
                            self.properties.smoothing_length,
                        ));
                    particle.pred_density += density;
                }
                // if cfg!(feature = "logging") {
                    // debug!("density: {}", particle.density());
                // }
            }
        });
    }

    /// Add gravity acceleration to all not boundary particles
    fn add_gravity(&mut self) {
        for particle in &mut self.particles {
            if particle.is_enabled() {
                let strength_of_gravity = 9.81;
                // gravitate downwards
                let acc = Vector3::new(0.0, 0.0, -strength_of_gravity);
                // gravitate around point
                // let gravitation_center = Vector3::new(0.0, 0.0, 0.0);
                // let acc = strength_of_gravity*(gravitation_center-particle.pos().now());

                particle.add_acc(acc);
            }
        }
    }

    /// Calculate spring acceleration at current time and add it to respective particles
    #[cfg(feature = "springs")]
    fn add_spring_acceleration(&mut self) {
        for Spring { indices: (i1, i2), k, l, ..} in &self.springs {
            // if cfg!(feature = "logging") {
            //     debug!("Calculate spring force");
            // }
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
    #[cfg(not(feature = "parallel"))]
    fn add_viscosity_acceleration(&mut self) {
        for particle_index in 0..self.particles.len() {
            if self.particles[particle_index].is_enabled() {
                // add viscostiy acceleration from other moving particles
                for &neighbor in &self.particles[particle_index].neighbors.clone() {
                    let acc = self.properties.viscosity*2.*self.particles[neighbor].mass()/self.particles[neighbor].density()
                        *(self.particles[particle_index].vel().now()-self.particles[neighbor].vel().now()).dot(&(self.particles[particle_index].pos().now()-self.particles[neighbor].pos().now()))
                        /((self.particles[particle_index].pos().now()-self.particles[neighbor].pos().now()).norm_squared()+0.01*self.properties.smoothing_length.powi(2))
                        *(self.properties.kernel_gradient_fn)(
                            &self.particles[particle_index].pos().now(),
                            &self.particles[neighbor].pos().now(),
                            self.properties.smoothing_length,
                        );
                    self.particles[particle_index].add_acc(acc);
                }
                // add viscostiy acceleration from boundary particles
                for &boundary_neighbor in &self.particles[particle_index].boundary_neighbors.clone() {
                    let acc = self.properties.viscosity*2.*self.boundary_particles[boundary_neighbor].mass()/self.properties.rest_density
                        *(self.particles[particle_index].vel().now()).dot(&(self.particles[particle_index].pos().now()-self.boundary_particles[boundary_neighbor].pos()))
                        /((self.particles[particle_index].pos().now()-self.boundary_particles[boundary_neighbor].pos()).norm_squared()+0.01*self.properties.smoothing_length.powi(2))
                        *(self.properties.kernel_gradient_fn)(
                            &self.particles[particle_index].pos().now(),
                            &self.boundary_particles[boundary_neighbor].pos(),
                            self.properties.smoothing_length,
                        );
                    self.particles[particle_index].add_acc(acc);
                }
            }
        }
    }

    /// Calculate viscosity acceleration at current time and add it to respective particles
    #[cfg(feature = "parallel")]
    fn add_viscosity_acceleration(&mut self) {
        let immutable_clone_of_particles = self.particles.clone();
        self.particles.par_iter_mut().for_each(|particle| {
            if particle.is_enabled() {
                // add viscostiy acceleration from other moving particles
                for &neighbor in &particle.neighbors.clone() {
                    let acc = self.properties.viscosity*2.*(3.+2.)*immutable_clone_of_particles[neighbor].mass()/immutable_clone_of_particles[neighbor].density()
                        *(particle.vel().now()-immutable_clone_of_particles[neighbor].vel().now()).dot(&(particle.pos().now()-immutable_clone_of_particles[neighbor].pos().now()))
                        /((particle.pos().now()-immutable_clone_of_particles[neighbor].pos().now()).norm_squared()+0.01*self.properties.smoothing_length.powi(2))
                        *(self.properties.kernel_gradient_fn)(
                            &particle.pos().now(),
                            &immutable_clone_of_particles[neighbor].pos().now(),
                            self.properties.smoothing_length,
                        );
                    particle.add_acc(acc);
                    // #[cfg(feature = "logging")]
                    // debug!("acc: {}", acc);
                }
                // add viscostiy acceleration from boundary particles
                for &boundary_neighbor in &particle.boundary_neighbors.clone() {
                    let acc = self.properties.viscosity*2.*(3.+2.)*self.boundary_particles[boundary_neighbor].mass()/self.properties.rest_density
                        *(particle.vel().now()).dot(&(particle.pos().now()-self.boundary_particles[boundary_neighbor].pos()))
                        /((particle.pos().now()-self.boundary_particles[boundary_neighbor].pos()).norm_squared()+0.01*self.properties.smoothing_length.powi(2))
                        *(self.properties.kernel_gradient_fn)(
                            &particle.pos().now(),
                            &self.boundary_particles[boundary_neighbor].pos(),
                            self.properties.smoothing_length,
                        );
                    particle.add_acc(acc);
                    // #[cfg(feature = "logging")]
                    // debug!("acc b: {}", acc);
                }
            }
        });
    }

    /// Calculate and update pressure for all particles for the current point in time.
    ///
    /// Function uses a state equation to calculate the pressure locally.
    #[cfg(feature = "local_pressure")]
    fn update_pressure_locally(&mut self) {
        for particle_index in 0..self.particles.len() {
            if self.particles[particle_index].is_enabled() {
                // select density
                #[cfg(not(feature = "splitting"))]
                let particle_density = self.particles[particle_index].density();
                #[cfg(feature = "splitting")]
                let particle_density = self.particles[particle_index].pred_density;
                // calc pressure with state equation
                let pressure = self.properties.stiffness*f64::max(
                    particle_density/self.properties.rest_density - 1.,
                    0.
                );
                self.particles[particle_index].set_pressure(pressure);
                // if cfg!(feature = "logging") {
                //     debug!("pressure: {}", pressure);
                // }
            }
        }
    }

    /// Calculate pressure acceleration at current time and add it to respective particles
    #[cfg(not(feature = "parallel"))]
    fn add_pressure_acceleration(&mut self) {
        // computer pressure acceleration
        for particle_index in 0..self.particles.len() {
            if self.particles[particle_index].is_enabled() {
                // add pressure acceleration from other moving particles
                for &neighbor in &self.particles[particle_index].neighbors.clone() {
                    // select density
                    #[cfg(all(not(feature = "splitting"), not(feature = "global_pressure")))]
                    let particle_density = self.particles[particle_index].density();
                    #[cfg(feature = "splitting")]
                    let particle_density = self.particles[particle_index].pred_density;
                    #[cfg(feature = "global_pressure")]
                    let particle_density = self.properties.rest_density;
                    #[cfg(all(not(feature = "splitting"), not(feature = "global_pressure")))]
                    let neighbor_density = self.particles[neighbor].density();
                    #[cfg(feature = "splitting")]
                    let neighbor_density = self.particles[neighbor].pred_density;
                    #[cfg(feature = "global_pressure")]
                    let neighbor_density = self.properties.rest_density;
                    // calc acceleration
                    let acc = -self.particles[neighbor].mass()
                        *(self.particles[particle_index].pressure()/particle_density.powi(2) + self.particles[neighbor].pressure()/neighbor_density.powi(2))
                        *(self.properties.kernel_gradient_fn)(
                            &self.particles[particle_index].pos().now(),
                            &self.particles[neighbor].pos().now(),
                            self.properties.smoothing_length,
                        );
                    #[cfg(not(feature = "global_pressure"))]
                    self.particles[particle_index].add_acc(acc);
                    #[cfg(feature = "global_pressure")]
                    self.particles[particle_index].add_acc(acc);
                    // #[cfg(feature = "logging")]
                    // debug!("kernel gradient: {}", (self.properties.kernel_gradient_fn)(
                    //     &self.particles[particle_index].pos().now(),
                    //     &self.particles[neighbor].pos().now(),
                    //     self.properties.smoothing_length,
                    // ));
                    // #[cfg(feature = "logging")]
                    // debug!("positions: {}, {}", self.particles[particle_index].pos().now(), self.particles[neighbor].pos().now());
                }
                // add pressure acceleration from boundary particles
                for &boundary_neighbor in &self.particles[particle_index].boundary_neighbors.clone() {
                    // select density
                    #[cfg(all(not(feature = "splitting"), not(feature = "global_pressure")))]
                    let particle_density = self.particles[particle_index].density();
                    #[cfg(feature = "splitting")]
                    let particle_density = self.particles[particle_index].pred_density;
                    #[cfg(feature = "global_pressure")]
                    let particle_density = self.properties.rest_density;
                    // calc acceleration
                    // mirror only pressure into boundary particle, set density to rest density
                    let acc = -self.properties.boundary_pressure_acceleration_weighting
                        *self.boundary_particles[boundary_neighbor].mass()
                        *self.particles[particle_index].pressure()*(1./particle_density.powi(2)+1./self.properties.rest_density.powi(2))
                        *(self.properties.kernel_gradient_fn)(
                            &self.particles[particle_index].pos().now(),
                            &self.boundary_particles[boundary_neighbor].pos(),
                            self.properties.smoothing_length,
                        );
                    self.particles[particle_index].add_acc(acc);
                }
            }
        }
    }

    /// Locally calculate pressure acceleration with a state equation at current time
    /// and add it to respective particles
    #[cfg(feature = "parallel")]
    fn add_pressure_acceleration(&mut self) {
        // computer pressure acceleration
        let immutable_clone_of_particles = self.particles.clone();
        self.particles.par_iter_mut().for_each(|particle| {
            if particle.is_enabled() {
                // add pressure acceleration from other moving particles
                for &neighbor in &particle.neighbors.clone() {
                    // select density
                    #[cfg(all(not(feature = "splitting"), not(feature = "global_pressure")))]
                    let particle_density = particle.density();
                    #[cfg(feature = "splitting")]
                    let particle_density = particle.pred_density;
                    #[cfg(feature = "global_pressure")]
                    let particle_density = self.properties.rest_density;
                    #[cfg(all(not(feature = "splitting"), not(feature = "global_pressure")))]
                    let neighbor_density = immutable_clone_of_particles[neighbor].density();
                    #[cfg(feature = "splitting")]
                    let neighbor_density = immutable_clone_of_particles[neighbor].pred_density;
                    #[cfg(feature = "global_pressure")]
                    let neighbor_density = self.properties.rest_density;
                    // calc acceleration
                    let acc =
                        -immutable_clone_of_particles[neighbor].mass()
                        *(particle.pressure()/particle_density.powi(2)
                            + immutable_clone_of_particles[neighbor].pressure()/neighbor_density.powi(2))
                        *(self.properties.kernel_gradient_fn)(
                            &particle.pos().now(),
                            &immutable_clone_of_particles[neighbor].pos().now(),
                            self.properties.smoothing_length,
                        );
                    particle.add_acc(acc);
                }
                // add pressure acceleration from boundary particles
                for &boundary_neighbor in &particle.boundary_neighbors.clone() {
                    // select density
                    #[cfg(all(not(feature = "splitting"), not(feature = "global_pressure")))]
                    let particle_density = particle.density();
                    #[cfg(feature = "splitting")]
                    let particle_density = particle.pred_density;
                    #[cfg(feature = "global_pressure")]
                    let particle_density = self.properties.rest_density;
                    // calc acceleration
                    // mirror only pressure into boundary particle, set density to rest density
                    let acc = -self.properties.boundary_pressure_acceleration_weighting
                        *self.boundary_particles[boundary_neighbor].mass()
                        *particle.pressure()*(1./particle_density.powi(2)+1./self.properties.rest_density.powi(2))
                        *(self.properties.kernel_gradient_fn)(
                            &particle.pos().now(),
                            &self.boundary_particles[boundary_neighbor].pos(),
                            self.properties.smoothing_length,
                        );
                    particle.add_acc(acc);
                }
            }
        });
    }

    /// Globally calculate pressure by solving a linear equation system at current time
    /// and update respective particles' fields
    ///
    /// For the implementation the following document was closedly followed:
    /// Notes on  Ihmsen et al. ”Implicit Incompressible SPH” by  Matthias Teschner, University of Freiburg
    #[cfg(all(not(feature = "parallel"), feature = "global_pressure"))]
    fn resolve_pressure_globally(&mut self) {
        // calculate predicted velocity due to non-pressure acceleration and set respective field at particle
        for particle_index in 0..self.particles.len() {
            if self.particles[particle_index].is_enabled() {
                let vel = self.particles[particle_index].vel().now()+self.properties.time_inc*self.particles[particle_index].acc();
                self.particles[particle_index].set_pred_vel(vel);
            }
        }
        // compute source term s_f of pressure linear equation system
        for particle_index in 0..self.particles.len() {
            if self.particles[particle_index].is_enabled() {
                self.particles[particle_index].s_f = self.properties.rest_density - self.particles[particle_index].density();
                for &neighbor in &self.particles[particle_index].neighbors.clone() {
                    self.particles[particle_index].s_f -= self.properties.time_inc*self.particles[neighbor].mass()
                        *(self.particles[particle_index].vel().pred() - self.particles[neighbor].vel().pred()).dot(
                            &(self.properties.kernel_gradient_fn)(
                                &self.particles[particle_index].pos().now(),
                                &self.particles[neighbor].pos().now(),
                                self.properties.smoothing_length,
                            )
                        );
                }
                for &boundary_neighbor in &self.particles[particle_index].boundary_neighbors.clone() {
                    self.particles[particle_index].s_f -= self.properties.time_inc*self.boundary_particles[boundary_neighbor].mass()
                        *(self.particles[particle_index].vel().pred() - self.boundary_particles[boundary_neighbor].vel()).dot(
                            &(self.properties.kernel_gradient_fn)(
                                &self.particles[particle_index].pos().now(),
                                &self.boundary_particles[boundary_neighbor].pos(),
                                self.properties.smoothing_length,
                            )
                        );
                }
                // #[cfg(feature = "logging")]
                // debug!("rho_f {}", self.particles[particle_index].density());
                // #[cfg(feature = "logging")]
                // debug!("s_f {}", self.particles[particle_index].s_f);
            }
        }
        // compute diagonal element A_ff
        for particle_index in 0..self.particles.len() {
            if self.particles[particle_index].is_enabled() {
                self.particles[particle_index].a_ff = 0.;
                // calc intermediate variable 1
                let mut c_1 = Vector3::zeros();
                for &neighbor in &self.particles[particle_index].neighbors.clone() {
                    c_1 += self.particles[neighbor].mass()/self.properties.rest_density.powi(2)*
                        (self.properties.kernel_gradient_fn)(
                                &self.particles[particle_index].pos().now(),
                                &self.particles[neighbor].pos().now(),
                                self.properties.smoothing_length,
                            );
                }
                // calc intermediate variable 2
                let mut c_2 = Vector3::zeros();
                for &boundary_neighbor in &self.particles[particle_index].boundary_neighbors.clone() {
                    c_2 += 2.*self.properties.boundary_pressure_acceleration_weighting
                        *self.boundary_particles[boundary_neighbor].mass()/self.properties.rest_density.powi(2)
                        *(self.properties.kernel_gradient_fn)(
                            &self.particles[particle_index].pos().now(),
                            &self.boundary_particles[boundary_neighbor].pos(),
                            self.properties.smoothing_length,
                        );
                }
                // use intermediate variables to calc a_ff
                for &neighbor in &self.particles[particle_index].neighbors.clone() {
                    self.particles[particle_index].a_ff += self.properties.time_inc.powi(2)*self.particles[neighbor].mass()
                        *(-c_1-c_2).dot(
                            &(self.properties.kernel_gradient_fn)(
                                &self.particles[particle_index].pos().now(),
                                &self.particles[neighbor].pos().now(),
                                self.properties.smoothing_length,
                            )
                        );
                    self.particles[particle_index].a_ff += self.properties.time_inc.powi(2)*self.particles[neighbor].mass()*
                        self.particles[particle_index].mass()/self.properties.rest_density.powi(2)
                        *(self.properties.kernel_gradient_fn)(
                                &self.particles[neighbor].pos().now(),
                                &self.particles[particle_index].pos().now(),
                                self.properties.smoothing_length,
                            ).dot(
                            &(self.properties.kernel_gradient_fn)(
                                &self.particles[particle_index].pos().now(),
                                &self.particles[neighbor].pos().now(),
                                self.properties.smoothing_length,
                            )
                        );
                }
                for &boundary_neighbor in &self.particles[particle_index].boundary_neighbors.clone() {
                    self.particles[particle_index].a_ff += self.properties.time_inc.powi(2)*self.boundary_particles[boundary_neighbor].mass()
                        *(-c_1-c_2).dot(
                            &(self.properties.kernel_gradient_fn)(
                                &self.particles[particle_index].pos().now(),
                                &self.boundary_particles[boundary_neighbor].pos(),
                                self.properties.smoothing_length,
                            )
                        );
                }
            }
        }
        // // compute diagonal element A_ff
        // for particle_index in 0..self.particles.len() {
        //     if self.particles[particle_index].is_enabled() {
        //         self.particles[particle_index].a_ff = 0.;
        //         // calc intermediate variable
        //         let mut int_var = Vector3::zeros();
        //         for &neighbor in &self.particles[particle_index].neighbors.clone() {
        //             int_var -= self.particles[neighbor].mass()/self.particles[particle_index].density().powi(2)
        //                 *(self.properties.kernel_gradient_fn)(
        //                     &self.particles[particle_index].pos().now(),
        //                     &self.particles[neighbor].pos().now(),
        //                     self.properties.smoothing_length,
        //                 );
        //         }
        //         for &boundary_neighbor in &self.particles[particle_index].boundary_neighbors.clone() {
        //             int_var -= 2.*self.properties.boundary_pressure_acceleration_weighting
        //                 *self.boundary_particles[boundary_neighbor].mass()/self.particles[particle_index].density().powi(2)
        //                 *(self.properties.kernel_gradient_fn)(
        //                     &self.particles[particle_index].pos().now(),
        //                     &self.boundary_particles[boundary_neighbor].pos(),
        //                     self.properties.smoothing_length,
        //                 );
        //         }
        //         // use intermediate variables to calc a_ff
        //         for &neighbor in &self.particles[particle_index].neighbors.clone() {
        //             self.particles[particle_index].a_ff += self.properties.time_inc.powi(2)*self.particles[neighbor].mass()
        //                 *int_var.dot(
        //                     &(self.properties.kernel_gradient_fn)(
        //                         &self.particles[particle_index].pos().now(),
        //                         &self.particles[neighbor].pos().now(),
        //                         self.properties.smoothing_length,
        //                     )
        //                 );
        //         }
        //         for &neighbor in &self.particles[particle_index].neighbors.clone() {
        //             self.particles[particle_index].a_ff += self.properties.time_inc.powi(2)*self.particles[neighbor].mass()*
        //                 self.particles[particle_index].mass()/self.particles[particle_index].density().powi(2)
        //                 *(self.properties.kernel_gradient_fn)(
        //                         &self.particles[neighbor].pos().now(),
        //                         &self.particles[particle_index].pos().now(),
        //                         self.properties.smoothing_length,
        //                     ).dot(
        //                     &(self.properties.kernel_gradient_fn)(
        //                         &self.particles[particle_index].pos().now(),
        //                         &self.particles[neighbor].pos().now(),
        //                         self.properties.smoothing_length,
        //                     )
        //                 );
        //         }
        //         for &boundary_neighbor in &self.particles[particle_index].boundary_neighbors.clone() {
        //             self.particles[particle_index].a_ff += self.properties.time_inc.powi(2)*self.boundary_particles[boundary_neighbor].mass()
        //                 *int_var.dot(
        //                     &(self.properties.kernel_gradient_fn)(
        //                         &self.particles[particle_index].pos().now(),
        //                         &self.boundary_particles[boundary_neighbor].pos(),
        //                         self.properties.smoothing_length,
        //                     )
        //                 );
        //         }
        //     }
        // }
        // initialize pressure and pressure acceleration to zero // todo: move into first loop of this function
        for particle_index in 0..self.particles.len() {
            if self.particles[particle_index].is_enabled() {
                self.particles[particle_index].set_pressure(0.);
            }
        }
        // Solve linear equation system until a sufficiently accurate result is obtained
        for _solver_iteration in 0..5 {
            // computer pressure acceleration // todo: move this into function
            for particle_index in 0..self.particles.len() {
                if self.particles[particle_index].is_enabled() {
                    // reset pressure acceleration
                    self.particles[particle_index].pressure_acc_f = Vector3::zeros();
                    // add pressure acceleration from other moving particles
                    for &neighbor in &self.particles[particle_index].neighbors.clone() {
                        let acc = -self.particles[neighbor].mass()
                            *(self.particles[particle_index].pressure()/self.properties.rest_density.powi(2)
                                +self.particles[neighbor].pressure()/self.properties.rest_density.powi(2))
                            *(self.properties.kernel_gradient_fn)(
                                &self.particles[particle_index].pos().now(),
                                &self.particles[neighbor].pos().now(),
                                self.properties.smoothing_length,
                            );
                        self.particles[particle_index].pressure_acc_f += acc;
                    }
                    // add pressure acceleration from boundary particles
                    for &boundary_neighbor in &self.particles[particle_index].boundary_neighbors.clone() {
                        let acc = -self.properties.boundary_pressure_acceleration_weighting
                            *self.boundary_particles[boundary_neighbor].mass()
                            *2.*self.particles[particle_index].pressure()/self.properties.rest_density.powi(2) // mirror pressure
                            *(self.properties.kernel_gradient_fn)(
                                &self.particles[particle_index].pos().now(),
                                &self.boundary_particles[boundary_neighbor].pos(),
                                self.properties.smoothing_length,
                            );
                        self.particles[particle_index].pressure_acc_f += acc;
                    }
                }
            }
            // // computer pressure acceleration // todo: move this into function
            // for particle_index in 0..self.particles.len() {
            //     if self.particles[particle_index].is_enabled() {
            //         // reset pressure acceleration
            //         self.particles[particle_index].pressure_acc_f = Vector3::zeros();
            //         // add pressure acceleration from other moving particles
            //         for &neighbor in &self.particles[particle_index].neighbors.clone() {
            //             let acc = -self.particles[neighbor].mass()
            //                 *(self.particles[particle_index].pressure()/self.particles[particle_index].density().powi(2)
            //                     +self.particles[neighbor].pressure()/self.particles[neighbor].density().powi(2))
            //                 *(self.properties.kernel_gradient_fn)(
            //                     &self.particles[particle_index].pos().now(),
            //                     &self.particles[neighbor].pos().now(),
            //                     self.properties.smoothing_length,
            //                 );
            //             self.particles[particle_index].pressure_acc_f += acc;
            //         }
            //         // add pressure acceleration from boundary particles
            //         for &boundary_neighbor in &self.particles[particle_index].boundary_neighbors.clone() {
            //             let acc = -self.properties.boundary_pressure_acceleration_weighting
            //                 *self.boundary_particles[boundary_neighbor].mass()
            //                 // mirror pressure
            //                 *self.particles[particle_index].pressure()*(1./self.particles[particle_index].density().powi(2)+1./self.properties.rest_density.powi(2))
            //                 *(self.properties.kernel_gradient_fn)(
            //                     &self.particles[particle_index].pos().now(),
            //                     &self.boundary_particles[boundary_neighbor].pos(),
            //                     self.properties.smoothing_length,
            //                 );
            //             self.particles[particle_index].pressure_acc_f += acc;
            //         }
            //     }
            // }
            // perform solver iteration for all fluid particles
            for particle_index in 0..self.particles.len() {
                if self.particles[particle_index].is_enabled() {
                    // calculate the divergence of the velocity change due to the pressure acceleration: a_dot_p_f
                    let mut a_dot_p_f = 0.;
                    for &neighbor in &self.particles[particle_index].neighbors.clone() {
                        a_dot_p_f += self.properties.time_inc.powi(2)*self.particles[neighbor].mass()
                            *(self.particles[particle_index].pressure_acc_f - self.particles[neighbor].pressure_acc_f).dot(
                                &(self.properties.kernel_gradient_fn)(
                                    &self.particles[particle_index].pos().now(),
                                    &self.particles[neighbor].pos().now(),
                                    self.properties.smoothing_length,
                                )
                            );
                    }
                    for &boundary_neighbor in &self.particles[particle_index].boundary_neighbors.clone() {
                        a_dot_p_f += self.properties.time_inc.powi(2)*self.boundary_particles[boundary_neighbor].mass()
                            *self.particles[particle_index].pressure_acc_f.dot(
                                &(self.properties.kernel_gradient_fn)(
                                    &self.particles[particle_index].pos().now(),
                                    &self.boundary_particles[boundary_neighbor].pos(),
                                    self.properties.smoothing_length,
                                )
                            );
                    }
                    // Update pressure
                    if self.particles[particle_index].a_ff > self.properties.min_diagonal_element
                            || self.particles[particle_index].a_ff < -self.properties.min_diagonal_element {
                        let p_next_iter = self.particles[particle_index].pressure() + self.properties.relaxation_factor
                            *(self.particles[particle_index].s_f - a_dot_p_f)/self.particles[particle_index].a_ff;
                        self.particles[particle_index].set_pressure(f64::max(p_next_iter, 0.));
                    }
                    // Calculate predicted density error
                    let _predicted_density_error = a_dot_p_f - self.particles[particle_index].s_f;
                    // if particle_index == 1 && self.particles[particle_index].pressure() != 0. {
                    //     #[cfg(feature = "logging")]
                    //     debug!("_solver_iteration {}", _solver_iteration);
                    //     #[cfg(feature = "logging")]
                    //     debug!("_predicted_density_error {}", _predicted_density_error);
                    //     #[cfg(feature = "logging")]
                    //     debug!("a_ff {}", self.particles[particle_index].a_ff);
                    //     #[cfg(feature = "logging")]
                    //     debug!("s_f {}", self.particles[particle_index].s_f);
                    //     #[cfg(feature = "logging")]
                    //     debug!("pressure_acc_f {}", self.particles[particle_index].pressure_acc_f);
                    //     #[cfg(feature = "logging")]
                    //     debug!("pressure {}", self.particles[particle_index].pressure());
                    //     #[cfg(feature = "logging")]
                    //     debug!("a_dot_p_f {}", a_dot_p_f);
                    // }
                }
            }
        }
    }

    /// Globally calculate pressure by solving a linear equation system at current time
    /// and update respective particles' fields
    ///
    /// For the implementation the following document was closedly followed:
    /// Notes on  Ihmsen et al. ”Implicit Incompressible SPH” by  Matthias Teschner, University of Freiburg
    #[cfg(all(feature = "parallel", feature = "global_pressure"))]
    fn resolve_pressure_globally(&mut self) {
        // calculate predicted velocity due to non-pressure acceleration and set respective field at particle
        for particle_index in 0..self.particles.len() {
            if self.particles[particle_index].is_enabled() {
                let vel = self.particles[particle_index].vel().now()+self.properties.time_inc*self.particles[particle_index].acc();
                self.particles[particle_index].set_pred_vel(vel);
            }
        }
        // compute source term s_f of pressure linear equation system
        let immutable_clone_of_particles = self.particles.clone();
        self.particles.par_iter_mut().for_each(|particle| {
            if particle.is_enabled() {
                particle.s_f = self.properties.rest_density - particle.density();
                for &neighbor in &particle.neighbors.clone() {
                    particle.s_f -= self.properties.time_inc*immutable_clone_of_particles[neighbor].mass()
                        *(particle.vel().pred() - immutable_clone_of_particles[neighbor].vel().pred()).dot(
                            &(self.properties.kernel_gradient_fn)(
                                &particle.pos().now(),
                                &immutable_clone_of_particles[neighbor].pos().now(),
                                self.properties.smoothing_length,
                            )
                        );
                }
                for &boundary_neighbor in &particle.boundary_neighbors.clone() {
                    particle.s_f -= self.properties.time_inc*self.boundary_particles[boundary_neighbor].mass()
                        *(particle.vel().pred() - self.boundary_particles[boundary_neighbor].vel()).dot(
                            &(self.properties.kernel_gradient_fn)(
                                &particle.pos().now(),
                                &self.boundary_particles[boundary_neighbor].pos(),
                                self.properties.smoothing_length,
                            )
                        );
                }
            }
        });
        // compute diagonal element A_ff
        let immutable_clone_of_particles = self.particles.clone();
        self.particles.par_iter_mut().for_each(|particle| {
            if particle.is_enabled() {
                particle.a_ff = 0.;
                // calc intermediate variable 1
                let mut c_1 = Vector3::zeros();
                for &neighbor in &particle.neighbors.clone() {
                    c_1 += immutable_clone_of_particles[neighbor].mass()/self.properties.rest_density.powi(2)*
                        (self.properties.kernel_gradient_fn)(
                                &particle.pos().now(),
                                &immutable_clone_of_particles[neighbor].pos().now(),
                                self.properties.smoothing_length,
                            );
                }
                // calc intermediate variable 2
                let mut c_2 = Vector3::zeros();
                for &boundary_neighbor in &particle.boundary_neighbors.clone() {
                    c_2 += 2.*self.properties.boundary_pressure_acceleration_weighting
                        *self.boundary_particles[boundary_neighbor].mass()/self.properties.rest_density.powi(2)*
                        (self.properties.kernel_gradient_fn)(
                            &particle.pos().now(),
                            &self.boundary_particles[boundary_neighbor].pos(),
                            self.properties.smoothing_length,
                        );
                }
                // use intermediate variables to calc a_ff
                for &neighbor in &particle.neighbors.clone() {
                    particle.a_ff += self.properties.time_inc.powi(2)*immutable_clone_of_particles[neighbor].mass()
                        *(-c_1-c_2).dot(
                            &(self.properties.kernel_gradient_fn)(
                                &particle.pos().now(),
                                &immutable_clone_of_particles[neighbor].pos().now(),
                                self.properties.smoothing_length,
                            )
                        );
                    particle.a_ff += self.properties.time_inc.powi(2)*immutable_clone_of_particles[neighbor].mass()*
                        particle.mass()/self.properties.rest_density.powi(2)
                        *(self.properties.kernel_gradient_fn)(
                                &immutable_clone_of_particles[neighbor].pos().now(),
                                &particle.pos().now(),
                                self.properties.smoothing_length,
                            ).dot(
                            &(self.properties.kernel_gradient_fn)(
                                &particle.pos().now(),
                                &immutable_clone_of_particles[neighbor].pos().now(),
                                self.properties.smoothing_length,
                            )
                        );
                }
                for &boundary_neighbor in &particle.boundary_neighbors.clone() {
                    particle.a_ff += self.properties.time_inc.powi(2)*self.boundary_particles[boundary_neighbor].mass()
                        *(-c_1-c_2).dot(
                            &(self.properties.kernel_gradient_fn)(
                                &particle.pos().now(),
                                &self.boundary_particles[boundary_neighbor].pos(),
                                self.properties.smoothing_length,
                            )
                        );
                }
            }
        });
        // // compute diagonal element A_ff
        // let immutable_clone_of_particles = self.particles.clone();
        // self.particles.par_iter_mut().for_each(|particle| {
        //     if particle.is_enabled() {
        //         particle.a_ff = 0.;
        //         // calc intermediate variable
        //         let mut int_var = Vector3::zeros();
        //         for &neighbor in &particle.neighbors.clone() {
        //             int_var -= immutable_clone_of_particles[neighbor].mass()/particle.density().powi(2)
        //                 *(self.properties.kernel_gradient_fn)(
        //                     &particle.pos().now(),
        //                     &immutable_clone_of_particles[neighbor].pos().now(),
        //                     self.properties.smoothing_length,
        //                 );
        //         }
        //         for &boundary_neighbor in &particle.boundary_neighbors.clone() {
        //             int_var -= 2.*self.properties.boundary_pressure_acceleration_weighting
        //                 *self.boundary_particles[boundary_neighbor].mass()/particle.density().powi(2)
        //                 *(self.properties.kernel_gradient_fn)(
        //                     &particle.pos().now(),
        //                     &self.boundary_particles[boundary_neighbor].pos(),
        //                     self.properties.smoothing_length,
        //                 );
        //         }
        //         // use intermediate variables to calc a_ff
        //         for &neighbor in &particle.neighbors.clone() {
        //             particle.a_ff += self.properties.time_inc.powi(2)*immutable_clone_of_particles[neighbor].mass()
        //                 *int_var.dot(
        //                     &(self.properties.kernel_gradient_fn)(
        //                         &particle.pos().now(),
        //                         &immutable_clone_of_particles[neighbor].pos().now(),
        //                         self.properties.smoothing_length,
        //                     )
        //                 );
        //         }
        //         for &neighbor in &particle.neighbors.clone() {
        //             particle.a_ff += self.properties.time_inc.powi(2)*immutable_clone_of_particles[neighbor].mass()*
        //                 particle.mass()/particle.density().powi(2)
        //                 *(self.properties.kernel_gradient_fn)(
        //                         &immutable_clone_of_particles[neighbor].pos().now(),
        //                         &particle.pos().now(),
        //                         self.properties.smoothing_length,
        //                     ).dot(
        //                     &(self.properties.kernel_gradient_fn)(
        //                         &particle.pos().now(),
        //                         &immutable_clone_of_particles[neighbor].pos().now(),
        //                         self.properties.smoothing_length,
        //                     )
        //                 );
        //         }
        //         for &boundary_neighbor in &particle.boundary_neighbors.clone() {
        //             particle.a_ff += self.properties.time_inc.powi(2)*self.boundary_particles[boundary_neighbor].mass()
        //                 *int_var.dot(
        //                     &(self.properties.kernel_gradient_fn)(
        //                         &particle.pos().now(),
        //                         &self.boundary_particles[boundary_neighbor].pos(),
        //                         self.properties.smoothing_length,
        //                     )
        //                 );
        //         }
        //     }
        // });
        // initialize pressure and pressure acceleration to zero // todo: move into first loop of this function
        for particle_index in 0..self.particles.len() {
            if self.particles[particle_index].is_enabled() {
                self.particles[particle_index].set_pressure(0.);
            }
        }
        // Solve linear equation system until a sufficiently accurate result is obtained
        for _solver_iteration in 0..5 {
            // computer pressure acceleration // todo: move this into function
            let immutable_clone_of_particles = self.particles.clone();
            self.particles.par_iter_mut().for_each(|particle| {
                if particle.is_enabled() {
                    // reset pressure acceleration
                    particle.pressure_acc_f = Vector3::zeros();
                    // add pressure acceleration from other moving particles
                    for &neighbor in &particle.neighbors.clone() {
                        let acc = -immutable_clone_of_particles[neighbor].mass()
                            *(particle.pressure()/self.properties.rest_density.powi(2)
                                +immutable_clone_of_particles[neighbor].pressure()/self.properties.rest_density.powi(2))
                            *(self.properties.kernel_gradient_fn)(
                                &particle.pos().now(),
                                &immutable_clone_of_particles[neighbor].pos().now(),
                                self.properties.smoothing_length,
                            );
                        particle.pressure_acc_f += acc;
                    }
                    // add pressure acceleration from boundary particles
                    for &boundary_neighbor in &particle.boundary_neighbors.clone() {
                        let acc = -self.properties.boundary_pressure_acceleration_weighting
                            *self.boundary_particles[boundary_neighbor].mass()
                            *2.*particle.pressure()/self.properties.rest_density.powi(2) // mirror pressure
                            *(self.properties.kernel_gradient_fn)(
                                &particle.pos().now(),
                                &self.boundary_particles[boundary_neighbor].pos(),
                                self.properties.smoothing_length,
                            );
                        particle.pressure_acc_f += acc;
                    }
                }
            });
            // // computer pressure acceleration // todo: move this into function
            // let immutable_clone_of_particles = self.particles.clone();
            // self.particles.par_iter_mut().for_each(|particle| {
            //     if particle.is_enabled() {
            //         // reset pressure acceleration
            //         particle.pressure_acc_f = Vector3::zeros();
            //         // add pressure acceleration from other moving particles
            //         for &neighbor in &particle.neighbors.clone() {
            //             let acc = -immutable_clone_of_particles[neighbor].mass()
            //                 *(particle.pressure()/particle.density().powi(2)
            //                     +immutable_clone_of_particles[neighbor].pressure()/immutable_clone_of_particles[neighbor].density().powi(2))
            //                 *(self.properties.kernel_gradient_fn)(
            //                     &particle.pos().now(),
            //                     &immutable_clone_of_particles[neighbor].pos().now(),
            //                     self.properties.smoothing_length,
            //                 );
            //             particle.pressure_acc_f += acc;
            //         }
            //         // add pressure acceleration from boundary particles
            //         for &boundary_neighbor in &particle.boundary_neighbors.clone() {
            //             let acc = -self.properties.boundary_pressure_acceleration_weighting
            //                 *self.boundary_particles[boundary_neighbor].mass()
            //                 // mirror pressure
            //                 *particle.pressure()*(1./particle.density().powi(2)+1./self.properties.rest_density.powi(2))
            //                 *(self.properties.kernel_gradient_fn)(
            //                     &particle.pos().now(),
            //                     &self.boundary_particles[boundary_neighbor].pos(),
            //                     self.properties.smoothing_length,
            //                 );
            //             particle.pressure_acc_f += acc;
            //         }
            //     }
            // });
            // perform solver iteration for all fluid particles
            let immutable_clone_of_particles = self.particles.clone();
            self.particles.par_iter_mut().for_each(|particle| {
                if particle.is_enabled() {
                    // calculate the divergence of the velocity change due to the pressure acceleration: a_dot_p_f
                    let mut a_dot_p_f = 0.;
                    for &neighbor in &particle.neighbors.clone() {
                        a_dot_p_f += self.properties.time_inc.powi(2)*immutable_clone_of_particles[neighbor].mass()
                            *(particle.pressure_acc_f - immutable_clone_of_particles[neighbor].pressure_acc_f).dot(
                                &(self.properties.kernel_gradient_fn)(
                                    &particle.pos().now(),
                                    &immutable_clone_of_particles[neighbor].pos().now(),
                                    self.properties.smoothing_length,
                                )
                            );
                    }
                    for &boundary_neighbor in &particle.boundary_neighbors.clone() {
                        a_dot_p_f += self.properties.time_inc.powi(2)*self.boundary_particles[boundary_neighbor].mass()
                            *particle.pressure_acc_f.dot(
                                &(self.properties.kernel_gradient_fn)(
                                    &particle.pos().now(),
                                    &self.boundary_particles[boundary_neighbor].pos(),
                                    self.properties.smoothing_length,
                                )
                            );
                    }
                    // Update pressure
                    if particle.a_ff > self.properties.min_diagonal_element
                            || particle.a_ff < -self.properties.min_diagonal_element {
                        let p_next_iter = particle.pressure() + self.properties.relaxation_factor
                            *(particle.s_f - a_dot_p_f)/particle.a_ff;
                        particle.set_pressure(f64::max(p_next_iter, 0.));
                    }
                    // Calculate predicted density error
                    let _predicted_density_error = a_dot_p_f - particle.s_f;
                }
            });
        }
    }

    #[cfg(feature = "splitting")]
    fn calc_predicted_velocity(&mut self) {
        for particle in &mut self.particles {
            if particle.is_enabled() {
                // update velocities
                particle.set_pred_vel(particle.vel().now() + self.properties.time_inc*particle.acc());
            }
        }
    }

    /// Calculate non-pressure accelerations and add them to each particles acceleration
    fn add_non_pressure_acceleration(&mut self) {
        // add gravity acceleration
        self.add_gravity();
        // add spring acceleration
        #[cfg(feature = "springs")]
        self.add_spring_acceleration();
        // add viscosity acceleration
        self.add_viscosity_acceleration();
    }

    /// Calculate acceleration at current time
    ///
    /// Supports: Gravity, spring force, viscosity and pressure acceleration
    #[cfg(feature = "local_pressure")]
    fn calc_acceleration(&mut self) {
        // reset acceleration
        for particle in &mut self.particles {
            particle.set_acc(Vector3::zero());
        }
        // add non-pressure acceleration
        self.add_non_pressure_acceleration();
        // perform splitting step conditionally
        #[cfg(feature = "splitting")]
        self.calc_predicted_velocity();
        #[cfg(feature = "splitting")]
        self.calc_predicted_density();
        // compute pressure
        self.update_pressure_locally();
        // add pressure acceleration
        self.add_pressure_acceleration();
    }

    /// Calculate acceleration at current time
    ///
    /// Supports: Gravity, spring force, viscosity and pressure acceleration
    #[cfg(feature = "global_pressure")]
    fn calc_acceleration(&mut self) {
        // reset acceleration
        for particle in &mut self.particles {
            particle.set_acc(Vector3::zero());
        }
        // add non-pressure acceleration
        self.add_non_pressure_acceleration();
        // solve pressure equation system
        self.resolve_pressure_globally();
        // add pressure acceleration due to pressure from pressure equation system
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
            #[cfg(feature = "implicit_euler")]
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
        // Measure (physical) quantities at current time step
        self.measure();
    }

    /// Update particle properties and uniform grid
    fn update(&mut self) {
        // disable irrelevant particles: particles below threshold (NOTE: Disabled particles must not be connected via spring)
        for particle in &mut self.particles {
            if particle.pos().now()[2] < self.properties.disable_particles_below {
                particle.disable();
            }
        }
        // update uniform grid of fluid particles
        self.particle_grid.clear();
        self.particle_grid.populate(&self.particles);
        // update neighbors of all fluid particles
        self.update_particle_neighbors();
        // compute density
        self.update_density();
        // calculate new accelerations
        self.calc_acceleration();
    }

    fn measure(&mut self) {
        self.properties.average_density = self.calc_average_mass_density();
        // if cfg!(feature = "logging") {
        //     debug!("{}, {}", self.properties.average_density, self.properties.rest_density);
        //     let max_speed = self.calc_max_speed();
        //     let cfl_coeff = max_speed*self.properties.time_inc/self.properties.rest_density_grid_spacing;
        //     debug!("time: {}, cfl coefficient: {}, max speed: {}", self.time(), cfl_coeff, max_speed);
        // }
        if let Some(ms) = &self.measurement_series {
            let measurement = measure::Measurement {
                time: self.time(),
                density: self.properties.average_density/self.properties.rest_density,
                kinetic_energy: self.calc_average_kinetic_energy(),
                #[cfg(feature = "local_pressure")]
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
        // #[cfg(feature = "logging")]
        // debug!("average mass density: {}", self.properties.average_density as f32);
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
