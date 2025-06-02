//! ## Physics based simulation backend
//!
//! Contains all necessary components to initialize a scene and simulate the trajectories
//! of its containing particles by propagating the system time.
//!
use std::{sync::{Arc, Mutex}, vec};

use nalgebra::{Matrix3, Vector3};
use num_traits::identities::Zero;
use serde::Deserialize;

// use tracing::{debug, error, info, span, trace, warn};
use tracing::{debug};


pub mod particle;
use particle::*;
pub mod spring;
use spring::*;
pub mod uniform_grid;




/// Cubic spline kernel function
pub fn cubic_spline_3d(distance: f64, smoothing_length: f64) -> f64 {
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

/// Gradient of cubic spline kernel function
pub fn cubic_spline_3d_gradient(distance: f64, smoothing_length: f64, direction: Vector3<f64>) -> Vector3<f64> {
    let normalized_distance = distance/smoothing_length;
    if normalized_distance == 0. { // if distance is 0 direction is invalid -> return Vector3::zeros()
        Vector3::zeros()
    } else if normalized_distance < 1. {
        let prefactor = 1./4./std::f64::consts::PI/smoothing_length.powi(5);
        direction/normalized_distance*prefactor*(-3.*(2.-normalized_distance).powi(2)+12.*(1.-normalized_distance).powi(2))
    } else if normalized_distance < 2. {
        let prefactor = 1./4./std::f64::consts::PI/smoothing_length.powi(5);
        direction/normalized_distance*prefactor*(-3.*(2.-normalized_distance).powi(2))
    } else {
        Vector3::zeros()
    }
}


/// Method for propagating time in a simulated physical system
#[derive(Debug, Deserialize)]
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
    /// Smooting length, often denoted as h
    particle_diameter: f64,
    rest_density: f64, // rho_0
    average_density: f64,
    viscosity: f64,
    stiffness: f64,
    kernel_fn: fn(distance: f64, particle_size: f64) -> f64,
    kernel_gradient_fn: fn(distance: f64, particle_size: f64, direction: Vector3<f64>) -> Vector3<f64>,
}

impl SystemProperties {
    pub fn new(
        time_inc: f64,
        particle_mass: f64,
        particle_diameter: f64,
        viscosity: f64,
        stiffness: f64,
        kernel_fn: fn(distance: f64, particle_size: f64) -> f64,
        kernel_gradient_fn: fn(distance: f64, particle_size: f64, direction: Vector3<f64>) -> Vector3<f64>,
    ) -> Self {
        let rest_density = particle_mass/particle_diameter.powi(3);
        let average_density = 0.;
        Self {
            time_inc,
            particle_mass,
            particle_diameter,
            rest_density,
            average_density,
            viscosity,
            stiffness,
            kernel_fn,
            kernel_gradient_fn,
        }
    }
}

///  3D implementation of a physical system to be simulated
#[derive(Debug, Clone)]
pub struct System3D {
    /// Collection of all moving particles
    particles: Vec<Particle3D>,
    /// Uniform grid for moving particles
    ///
    /// Accelerates neighbor search
    particle_grid: uniform_grid::UniformGrid,
    /// Collection of all boundary (not moving) particles
    boundary_particles: Vec<BoundaryParticle3D>,
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
    time: f64,
    /// Properties of the system
    properties: SystemProperties,
    /// Queue for queuing all particle position for visualization
    pub queue: Arc<Mutex<super::mediation::IntermediateQueue>>,
    /// Controls for communicating with front end
    pub controls: Arc<Mutex<super::mediation::IntermediateControls>>,
}

impl System3D {
    pub fn new(
        particles: Vec<Particle3D>,
        boundary_particles: Vec<BoundaryParticle3D>,
        springs: Vec<Spring>,
        system_properties: SystemProperties,
        queue: Arc<Mutex<super::mediation::IntermediateQueue>>,
        controls: Arc<Mutex<super::mediation::IntermediateControls>>,
    ) -> Self {
        let particle_grid = uniform_grid::UniformGrid::new(system_properties.particle_diameter);
        let mut boundary_particle_grid = uniform_grid::UniformGrid::new(system_properties.particle_diameter);
        boundary_particle_grid.populate_boundary_particles(&boundary_particles);
        let mut system = Self {
            particles,
            particle_grid,
            boundary_particles,
            boundary_particle_grid,
            springs,
            time: 0.0,
            properties: system_properties,
            queue,
            controls,
        };
        // Add initial positions to queue and update uniform grid
        system.update();
        system
    }

    pub fn from_config(
        config: &super::setup::Config,
        system_properties: SystemProperties,
        queue: Arc<Mutex<super::mediation::IntermediateQueue>>,
        controls: Arc<Mutex<super::mediation::IntermediateControls>>,
    ) -> Self {
        // init particles
        let mut particles = vec![];
        for i in 0..config.scene.particles.n_particles_x {
            for j in 0..config.scene.particles.n_particles_y {
                for k in 0..config.scene.particles.n_particles_z {
                    let x = (i as f64)*config.scene.particles.particle_spacing+config.scene.particles.x_offset;
                    let y = (j as f64)*config.scene.particles.particle_spacing+config.scene.particles.y_offset;
                    let z = (k as f64)*config.scene.particles.particle_spacing+config.scene.particles.z_offset;
                    let particle = Particle3D::new(
                        [Vector3::new(x, y, z), Vector3::new(x, y, z)],
                        Vector3::new(0., 0., 0.),
                        config.params.particle_mass,
                        None);
                    particles.push(particle);
                }
            }
        }
        // init boundary particles
        let mut boundary_particles = vec![];
        // init floor
        for i in 0..config.scene.boundary_particles.n_floor_particles_x {
            for j in 0..config.scene.boundary_particles.n_floor_particles_y {
                for k in 0..config.scene.boundary_particles.n_floor_particles_z {
                    let x = (i as f64)*config.scene.boundary_particles.particle_spacing+config.scene.boundary_particles.x_offset;
                    let y = (j as f64)*config.scene.boundary_particles.particle_spacing+config.scene.boundary_particles.y_offset;
                    let z = (k as f64)*config.scene.boundary_particles.particle_spacing+config.scene.boundary_particles.z_offset;
                    let boundary_particle = BoundaryParticle3D::new(
                        Vector3::new(x, y, z),
                        [0.0,0.0,0.0]);
                    boundary_particles.push(boundary_particle);
                }
            }
        }
        // init walls ()
        for i in 0..config.scene.boundary_particles.n_floor_particles_x {
            for j in 0..config.scene.boundary_particles.n_floor_particles_y {
                for k in config.scene.boundary_particles.n_floor_particles_z..(config.scene.boundary_particles.n_floor_particles_z+config.scene.boundary_particles.wall_height) {
                    let x = (i as f64)*config.scene.boundary_particles.particle_spacing+config.scene.boundary_particles.x_offset;
                    let y = (j as f64)*config.scene.boundary_particles.particle_spacing+config.scene.boundary_particles.y_offset;
                    let z = (k as f64)*config.scene.boundary_particles.particle_spacing+config.scene.boundary_particles.z_offset;
                    // filter for particles at the edge of the floor
                    if x < (config.scene.boundary_particles.x_offset + config.scene.boundary_particles.x_wall_thickness as f64 * config.scene.boundary_particles.particle_spacing)
                            || x > (config.scene.boundary_particles.x_offset + ((config.scene.boundary_particles.n_floor_particles_x as f64 - 1.) - config.scene.boundary_particles.x_wall_thickness as f64) * config.scene.boundary_particles.particle_spacing)
                            || y < (config.scene.boundary_particles.y_offset + config.scene.boundary_particles.y_wall_thickness as f64 * config.scene.boundary_particles.particle_spacing)
                            || y > (config.scene.boundary_particles.y_offset + ((config.scene.boundary_particles.n_floor_particles_y as f64 - 1.) - config.scene.boundary_particles.y_wall_thickness as f64) * config.scene.boundary_particles.particle_spacing) {
                        let boundary_particle = BoundaryParticle3D::new(
                            Vector3::new(x, y, z),
                            [0.0,0.0,0.0]);
                        boundary_particles.push(boundary_particle);
                    }
                }
            }
        }
        // init springs
        let mut springs = vec![];
        // add springs configured in config file here

        Self::new(
            particles,
            boundary_particles,
            springs,
            system_properties,
            queue,
            controls,
        )
    }

    pub fn update_density(&mut self) {
        for particle_index in 0..self.particles.len() {
            // reset density
            self.particles[particle_index].set_density(0.);
            // add density for every neighbor
            for &neighbor in &self.particles[particle_index].neighbors.clone() {
                let distance = self.particles[particle_index].get_distance(&self.particles[neighbor].pos().now());
                let density = self.particles[neighbor].mass()
                    *(self.properties.kernel_fn)(distance, self.properties.particle_diameter);
                self.particles[particle_index].add_density(density);
            }
            // add density for every boundary neighbor (mirror mass of moving particle onto boundary particle)
            for &boundary_neighbors in &self.particles[particle_index].boundary_neighbors.clone() {
                // add density for every neighbor
                let distance = self.particles[particle_index].get_distance(&self.boundary_particles[boundary_neighbors].pos());
                let density = self.particles[particle_index].mass()
                    *(self.properties.kernel_fn)(distance, self.properties.particle_diameter);
                self.particles[particle_index].add_density(density);
            }
            // debug!("rest density: {}", self.properties.rest_density);
        }
        // calculate average density
        self.properties.average_density = 0.;
        let mut count = 0.;
        for particle in &self.particles {
            if particle.density() >= self.properties.rest_density {
                self.properties.average_density += particle.density();
                count += 1.;
            }
        }
        if count != 0. {
            self.properties.average_density /= count;
        }
        debug!("Average density: {}, rest density: {}, contributing particles: {}", self.properties.average_density, self.properties.rest_density, count);
    }

    pub fn update_pressure(&mut self) {
        for particle_index in 0..self.particles.len() {
            let pressure = self.properties.stiffness*f64::max(
                self.particles[particle_index].density()/self.properties.rest_density - 1.,
                0.
            );
            self.particles[particle_index].set_pressure(pressure);
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
        for particle_index in 0..self.particles.len() {
            // add viscostiy acceleration from other moving particles
            for &neighbor in &self.particles[particle_index].neighbors.clone() {
                let distance = self.particles[particle_index].get_distance(&self.particles[neighbor].pos().now());
                let direction = self.particles[particle_index].get_direction(&self.particles[neighbor].pos().now());
                let acc = self.properties.viscosity*2.*self.particles[neighbor].mass()/self.particles[particle_index].density()
                    *(self.particles[particle_index].vel().now()-self.particles[neighbor].vel().now()).dot(&(self.particles[particle_index].pos().now()-self.particles[neighbor].pos().now()))
                    /((self.particles[particle_index].pos().now()-self.particles[neighbor].pos().now()).norm_squared()+0.01*self.properties.particle_diameter.powi(2))
                    *(self.properties.kernel_gradient_fn)(distance, self.properties.particle_diameter, direction);
                self.particles[particle_index].add_acc(acc);
            }
            // add viscostiy acceleration from boundary particles
            for &boundary_neighbor in &self.particles[particle_index].boundary_neighbors.clone() {
                let distance = self.particles[particle_index].get_distance(&self.boundary_particles[boundary_neighbor].pos());
                let direction = self.particles[particle_index].get_direction(&self.boundary_particles[boundary_neighbor].pos());
                let acc = self.properties.viscosity*2.*self.particles[particle_index].mass()/self.particles[particle_index].density()
                    *(self.particles[particle_index].vel().now()-Vector3::new(0., 0., 0.)).dot(&(self.particles[particle_index].pos().now()-self.boundary_particles[boundary_neighbor].pos()))
                    /((self.particles[particle_index].pos().now()-self.boundary_particles[boundary_neighbor].pos()).norm_squared()+0.01*self.properties.particle_diameter.powi(2))
                    *(self.properties.kernel_gradient_fn)(distance, self.properties.particle_diameter, direction);
                self.particles[particle_index].add_acc(acc);
            }
        }
    }

    /// Calculate pressure acceleration at current time and add it to respective particles
    fn add_pressure_acceleration(&mut self) {
        for particle_index in 0..self.particles.len() {
            // add pressure acceleration from other moving particles
            for &neighbor in &self.particles[particle_index].neighbors.clone() {
                let distance = self.particles[particle_index].get_distance(&self.particles[neighbor].pos().now());
                let direction = self.particles[particle_index].get_direction(&self.particles[neighbor].pos().now());
                let acc = -self.particles[neighbor].mass()
                    *(self.particles[particle_index].pressure()/self.particles[particle_index].density().powi(2) + self.particles[neighbor].pressure()/self.particles[neighbor].density().powi(2))
                    *(self.properties.kernel_gradient_fn)(distance, self.properties.particle_diameter, direction);
                self.particles[particle_index].add_acc(acc);
            }
            // add pressure acceleration from boundary particles
            for &boundary_neighbor in &self.particles[particle_index].boundary_neighbors.clone() {
                let distance = self.particles[particle_index].get_distance(&self.boundary_particles[boundary_neighbor].pos());
                let direction = self.particles[particle_index].get_direction(&self.boundary_particles[boundary_neighbor].pos());
                let acc = -self.particles[particle_index].mass()
                    *2.*self.particles[particle_index].pressure()/self.particles[particle_index].density().powi(2)
                    *(self.properties.kernel_gradient_fn)(distance, self.properties.particle_diameter, direction);
                self.particles[particle_index].add_acc(acc);
            }
        }
    }

    /// Calculate acceleration at current time
    ///
    /// Supports: Gravity, spring force, viscosity and pressure acceleration
    fn calc_acceleration(&mut self) {
        for particle_index in 0..self.particles.len() {
            // update neighbors
            let neighbors = self.particle_grid.get_particles_in_kernel_range(&self.particles[particle_index].pos().now(), &self.particles);
            self.particles[particle_index].set_neighbors(neighbors);
            // update boundary neighbors
            let boundary_neighbors = self.boundary_particle_grid.get_particles_in_kernel_range(&self.particles[particle_index].pos().now(), &self.boundary_particles);
            self.particles[particle_index].set_boundary_neighbors(boundary_neighbors);
        }
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

    pub fn inc_time(&mut self, method: PropagationMethod) {
        // calculate accelerations
        self.calc_acceleration();
        // increment time one step
        match method {
            PropagationMethod::ExplicitEuler => {
                for v in &mut self.particles {
                    // update positions
                    v.set_new_pos(v.pos().now() + self.properties.time_inc*v.vel().now());
                    // update velocities
                    v.set_new_vel(v.vel().now() + self.properties.time_inc*v.acc());
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
                for v in &mut self.particles {
                    // update velocities
                    v.set_new_vel(v.vel().now() + self.properties.time_inc*v.acc());
                    // update positions with new velocities: v_i(t+h)
                    v.set_new_pos(v.pos().now() + self.properties.time_inc*v.vel().now());
                }
            },
            PropagationMethod::Verlet => {
                // update positions
                for v in &mut self.particles {
                    // update positions
                    let t = 2.0*v.pos().now() - v.pos().prev() + self.properties.time_inc.powi(2)*v.acc();
                    v.set_new_pos(t);
                    // update velocities
                    let t = (v.pos().now() - v.pos().prev())/self.properties.time_inc;
                    v.set_new_vel(t);
                }
            },
        }
        self.time += self.properties.time_inc;
        self.update();
    }

    fn update(&mut self) {
        // update color
        for particle in &mut self.particles {
            particle.update_color();
        }
        // update uniform grid
        self.particle_grid.clear();
        self.particle_grid.populate(&self.particles);
    }

    fn particles_as_instances(&self) -> Vec<super::mediation::Instance> {
        let mut result = Vec::new();
        // add moving particles
        for particle in &self.particles {
            let instance = super::mediation::Instance {
                position: Matrix3::new(1., 0., 0., 0., 0., 1., 0., -1., 0.) // map y to -z axis and z to y axis
                    *particle.pos().now().map(|v| { v as f32 }),
                color: particle.color(),
            };
            result.push(instance);
        }
        // add boundary particles
        for particle in &self.boundary_particles {
            let instance = super::mediation::Instance {
                position: Matrix3::new(1., 0., 0., 0., 0., 1., 0., -1., 0.) // map y to -z axis and z to y axis
                    *particle.pos().map(|v| { v as f32 }),
                color: particle.color(),
            };
            result.push(instance);
        }
        result
    }

    /// Forward new particle positions to graphics output
    /// by queueing particles to visualization queue
    pub fn queue_for_visualization(&self) {
        // debug!("queued\n");
        self.queue.lock().unwrap().push_back(self.particles_as_instances());
        self.controls.lock().unwrap().set_average_density(self.properties.average_density as f32);
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
