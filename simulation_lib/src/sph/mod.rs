/// # Core SPH simulation
///
/// Contains the simulated system, the information of the individual samples
/// and provides the methods for propagating the system in time.
///
use nalgebra::{Matrix3, Vector3};
use num_traits::Zero;
#[cfg(feature = "parallelized_sph")]
use rayon::prelude::*;
#[cfg(feature = "logging")]
use tracing::{debug, warn}; // debug, error, info, span, trace, warn,

pub mod kernel;
pub mod pressure_solver;
mod non_pressure_accelerations;
mod volume;

#[cfg(feature = "springs")]
pub mod spring;

#[cfg(feature = "springs")]
use spring::*;
use crate::sample::*;
use crate::sph::non_pressure_accelerations::*;
use pressure_solver::PressureSolver;
use crate::neighbor_search::UniformGrid;
use crate::TimeStepInfo;
use crate::for_each;
use crate::measurement;
use crate::setup;
use kernel::KernelFn;
use crate::integration_schemes::IntegrationScheme;
use volume::update_volume;

/// Calculate the distance between two 3D points
fn distance(from: &Vector3<f64>, to: &Vector3<f64>) -> f64 {
    (to - from).norm()
}

/// Direction from particle1 towards particle2
fn direction(from: &Vector3<f64>, towards: &Vector3<f64>) -> Vector3<f64> {
    towards - from
}

#[allow(dead_code)]
enum TerminationCondition {
    AfterIteration(u32),
    TargetDensityError(f64),
}

/// Information about the system at the current time
#[derive(Debug, Clone, Default)]
pub struct CurrentSystemProperties {
    average_density: f64,
    fluid_depth: f64,
    solver_iterations: u32,
    /// wall clock time passed calculating current time step
    time_step_wall_clock_time: f64,
    predicted_density_error: f64,
}

impl CurrentSystemProperties {
    pub fn set_fluid_depth(&mut self, fluid_depth: f64) {
        self.fluid_depth = fluid_depth;
    }

    fn update(&mut self, average_density: f64) {
        self.average_density = average_density;
    }
}

/// Simulation parameters of a to be simulated system
#[derive(Debug, Clone)]
pub struct SystemParameters {
    time_increment: f64,
    #[cfg(feature = "cfl_time_step")]
    max_time_increment: f64,
    #[cfg(feature = "cfl_time_step")]
    pub cfl_number: f64,
    /// Smooting length h
    smoothing_length: f64,
    /// disable particles below this threshold
    disable_particles_below: f64,
    rest_density: f64, // rho_0
    rest_volume: f64,  // rho_0
    /// Grid spacing when particles are ordered in a cubic grid at rest density
    rest_density_grid_spacing: f64,
    fluid_viscosity: f64,
    boundary_viscosity: f64,
    boundary_pressure_acceleration_weighting: f64,
    boundary_rest_volume_weighting: f64,
}

impl SystemParameters {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        #[cfg(not(feature = "cfl_time_step"))] time_increment: f64,
        #[cfg(feature = "cfl_time_step")] max_time_increment: f64,
        #[cfg(feature = "cfl_time_step")] cfl_number: f64,
        rest_density: f64,
        rest_density_grid_spacing: f64,
        smoothing_length: f64,
        disable_particles_below: f64,
        fluid_viscosity: f64,
        boundary_viscosity: f64,
        boundary_pressure_acceleration_weighting: f64,
        boundary_rest_volume_weighting: f64,
    ) -> Self {
        Self {
            #[cfg(not(feature = "cfl_time_step"))]
            time_increment,
            #[cfg(feature = "cfl_time_step")]
            time_increment: 0.,
            #[cfg(feature = "cfl_time_step")]
            max_time_increment,
            #[cfg(feature = "cfl_time_step")]
            cfl_number,
            smoothing_length,
            disable_particles_below,
            rest_density,
            rest_volume: rest_density_grid_spacing.powi(3),
            rest_density_grid_spacing,
            fluid_viscosity,
            boundary_viscosity,
            boundary_pressure_acceleration_weighting,
            boundary_rest_volume_weighting,
        }
    }

    #[cfg(feature = "cfl_time_step")]
    fn set_cfl_time_step(&mut self, max_speed: f64) {
        let cfl_time_step = self.cfl_number * self.rest_density_grid_spacing / max_speed;
        self.time_increment = self.max_time_increment.min(cfl_time_step);
    }
}

///  3D implementation of a physical system to be simulated
#[derive(Debug, Clone)]
pub struct System3D<K: KernelFn, I: IntegrationScheme, P: PressureSolver> {
    /// Collection of all fluid samples
    fluid: Fluid3D,
    /// Uniform grid for fluid samples
    ///
    /// Accelerates neighbor search
    fluid_neighbor_search: UniformGrid,
    /// Collection of all boundary (not moving) samples
    boundary: Boundary3D,
    /// Uniform grid for boundary particles
    ///
    /// Accelerates neighbor search
    boundary_neighbor_search: UniformGrid,
    /// Springs connecting different samples
    ///
    /// Spring stores indices of samples connected to via spring force,
    /// spring force coeff (k) and rest length (l)
    #[cfg(feature = "springs")]
    springs: Vec<Spring>,
    /// Time
    time_steps_propagated: u64,
    /// Properties of the system
    parameters: SystemParameters,
    properties: CurrentSystemProperties,
    _kernel_fn: std::marker::PhantomData<K>,
    integrator: I,
    pressure_solver: P,
}

impl<K: KernelFn, I: IntegrationScheme, P: PressureSolver> System3D<K, I, P> {
    pub fn new(
        systemconfig: setup::System3DConfig,
        integrator: I,
        pressure_solver: P,
    ) -> Self {
        let particle_grid =
            UniformGrid::new(systemconfig.system_parameters.smoothing_length);
        let mut boundary_particle_grid =
            UniformGrid::new(systemconfig.system_parameters.smoothing_length);
        boundary_particle_grid.populate_boundary_particles(&systemconfig.boundary);
        let mut system = Self {
            fluid: systemconfig.fluid,
            fluid_neighbor_search: particle_grid,
            boundary: systemconfig.boundary,
            boundary_neighbor_search: boundary_particle_grid,
            #[cfg(feature = "springs")]
            springs: systemconfig.springs,
            time_steps_propagated: 0,
            parameters: systemconfig.system_parameters,
            properties: systemconfig.properties,
            _kernel_fn: std::marker::PhantomData,
            integrator,
            pressure_solver,
        };
        // set boundary mass such that the density is equal to the fluids rest density
        system.init_boundary_volume();
        // Update uniform grid
        system.update();
        system
    }

    /// Calculate and set pseudo mass of all boundary particles
    #[cfg(not(feature = "pseudo_volume_boundary"))]
    fn init_boundary_volume(&mut self) {
        for boundary_particle_index in 0..self.boundary.len() {
            // simple mass
            self.boundary
                .set_volume(boundary_particle_index, self.parameters.rest_volume);
        }
    }

    /// Calculate and set pseudo mass of all boundary particles
    #[cfg(feature = "pseudo_volume_boundary")]
    fn init_boundary_volume(&mut self) {
        for boundary_particle_index in 0..self.boundary.len() {
            // add inverse volume for every boundary neighbor
            let mut inverse_volume = 0.;
            // get boundary neighbors of boundary particles
            for boundary_neighbor in self.boundary_neighbor_search.get_particles_in_kernel_range(
                self.boundary.pos_now(boundary_particle_index),
                &self.boundary.position,
            ) {
                let dist = distance(
                    self.boundary.pos_now(boundary_particle_index),
                    self.boundary.pos_now(boundary_neighbor),
                );
                inverse_volume += K::value(dist, self.parameters.smoothing_length);
            }
            // calculate mass with rest density of fluid
            let pseudo_volume = self.parameters.boundary_rest_volume_weighting / inverse_volume;
            self.boundary
                .set_volume(boundary_particle_index, pseudo_volume);
            // #[cfg(feature = "logging")]
            // debug!("boundary particle {} has position: {}", boundary_particle_index, self.boundary_particles[boundary_particle_index].pos());
            // #[cfg(feature = "logging")]
            // debug!("boundary particle {} has mass: {}", boundary_particle_index, self.boundary_particles[boundary_particle_index].mass());
        }
    }

    pub fn time(&self) -> f64 {
        (self.time_steps_propagated as f64) * self.parameters.time_increment
    }

    /// Calculate 2-norm of maximum velocity of any particle
    fn calc_max_speed(&self) -> f64 {
        #[cfg(not(feature = "parallelized_sph"))]
        {
            self.fluid
                .velocity
                .iter()
                .map(|vel| vel.norm())
                .fold(0.0_f64, f64::max)
        }
        #[cfg(feature = "parallelized_sph")]
        {
            self.fluid
                .velocity
                .par_iter()
                .map(|vel| vel.norm())
                .reduce(|| 0.0_f64, f64::max)
        }
    }

    /// Calculate average kinetic energy for all fluid particles
    fn calc_average_kinetic_energy(&self) -> f64 {
        #[cfg(not(feature = "parallelized_sph"))]
        let (total_energy, count) = {
            self.fluid
                .velocity
                .iter()
                .zip(self.fluid.mass.iter())
                .map(|(vel, mass)| 0.5 * mass * vel.norm_squared())
                .fold((0.0_f64, 0_u64), |(sum, cnt), energy| {
                    (sum + energy, cnt + 1)
                })
        };
        #[cfg(feature = "parallelized_sph")]
        let (total_energy, count) = self
            .fluid
            .velocity
            .par_iter()
            .zip(&self.fluid.mass)
            .map(|(vel, mass)| 0.5 * mass * vel.norm_squared())
            .fold(
                || (0.0_f64, 0_u64),
                |(sum, cnt), energy| (sum + energy, cnt + 1),
            )
            .reduce(
                || (0.0, 0),
                |(sum_a, cnt_a), (sum_b, cnt_b)| (sum_a + sum_b, cnt_a + cnt_b),
            );
        if count > 0 {
            total_energy / count as f64
        } else {
            0.0
        }
    }

    // fn save_pressure_profile(&self) -> Result<(), Box<dyn Error>> {
    //     let mut bin_daten: HashMap<i32, (f64, u32)> = HashMap::new();

    //     for particle in &self.particles {
    //         if particle.is_enabled() {
    //             let rounded: f64 = particle.pos_now().z.round();
    //             let bin = rounded as i32;
    //             let entry = bin_daten.entry(bin).or_insert((0.0, 0));
    //             entry.0 += particle.pressure();
    //             entry.1 += 1;
    //         }
    //     }

    //     let file = File::create(format!("pressure_profile_at_time_{}.csv", self.time()))?;
    //     let mut writer = csv::Writer::from_writer(file);

    //     writer.write_record(["Bin", "Average", "Count"])?;

    //     // Sort
    //     let mut sorted_bins: Vec<_> = bin_daten.into_iter().collect();
    //     sorted_bins.sort_by_key(|k| k.0);

    //     for (bin, (summe, anzahl)) in sorted_bins {
    //         let average = summe / anzahl as f64;
    //         writer.write_record(&[
    //             bin.to_string(),
    //             format!("{}", average),
    //             anzahl.to_string(),
    //         ])?;
    //     }

    //     writer.flush()?;
    //     Ok(())
    // }

    // fn save_kinetic_energy_profile(&self) -> Result<(), Box<dyn Error>> {
    //     let mut bin_daten: HashMap<i32, (f64, u32)> = HashMap::new();

    //     for particle in &self.particles {
    //         if particle.is_enabled() {
    //             let kin_energy = 1./2.*particle.mass()*particle.vel_now(id).norm_squared();
    //             let rounded: f64 = particle.pos_now().z.round();
    //             let bin = rounded as i32;
    //             let entry = bin_daten.entry(bin).or_insert((0.0, 0));
    //             entry.0 += kin_energy;
    //             entry.1 += 1;
    //         }
    //     }

    //     let file = File::create(format!("kin_energy_profile_at_time_{}.csv", self.time()))?;
    //     let mut writer = csv::Writer::from_writer(file);

    //     writer.write_record(["Bin", "Average", "Count"])?;

    //     // Sort
    //     let mut sorted_bins: Vec<_> = bin_daten.into_iter().collect();
    //     sorted_bins.sort_by_key(|k| k.0);

    //     for (bin, (summe, anzahl)) in sorted_bins {
    //         let average = summe / anzahl as f64;
    //         writer.write_record(&[
    //             bin.to_string(),
    //             format!("{}", average),
    //             anzahl.to_string(),
    //         ])?;
    //     }

    //     writer.flush()?;
    //     Ok(())
    // }

    /// Calculate average mass density for all fluid particles
    fn calc_average_mass_density(&self) -> f64 {
        #[cfg(not(feature = "parallelized_sph"))]
        let (total_mass_density, count) = {
            self.fluid
                .volume
                .iter()
                .zip(self.fluid.mass.iter())
                .map(|(volume, mass)| {
                    if *volume < self.parameters.rest_volume {
                        mass / volume
                    } else {
                        mass / self.parameters.rest_volume
                    }
                })
                .fold((0.0_f64, 0_u64), |(sum, cnt), d| (sum + d, cnt + 1))
        };
        #[cfg(feature = "parallelized_sph")]
        let (total_mass_density, count) = self
            .fluid
            .volume
            .par_iter()
            .zip(&self.fluid.mass)
            .map(|(volume, mass)| {
                if *volume < self.parameters.rest_volume {
                    mass / volume
                } else {
                    mass / self.parameters.rest_volume
                }
            })
            .fold(
                || (0.0_f64, 0_u64),
                |(sum, cnt), mass_density| (sum + mass_density, cnt + 1),
            )
            .reduce(
                || (0.0, 0),
                |(sum_a, cnt_a), (sum_b, cnt_b)| (sum_a + sum_b, cnt_a + cnt_b),
            );
        if count > 0 {
            total_mass_density / count as f64
        } else {
            0.0
        }
    }

    /// Perform neighbor search for all fluid particles
    ///
    /// Adds fluid neighbors and boundary neighbors as neighbors
    fn update_particle_neighbors(&mut self) {
        for_each!(
            mut [self.fluid.neighbors, self.fluid.boundary_neighbors],
            ref [pos_now = self.fluid.position],
            |id, id_neighbors, id_boundary_neighbors| {
                // update neighbors
                let neighbors = self
                    .fluid_neighbor_search
                    .get_particles_in_kernel_range(&pos_now[id], pos_now);
                *id_neighbors = neighbors;
                // update boundary neighbors
                let boundary_neighbors = self
                    .boundary_neighbor_search
                    .get_particles_in_kernel_range(&pos_now[id], &self.boundary.position);
                *id_boundary_neighbors = boundary_neighbors;
            }
        );
    }

    /// Calculate non-pressure accelerations and add them to each particles acceleration
    fn add_non_pressure_acceleration(&mut self) {
        // add gravity acceleration
        add_gravity(&mut self.fluid);
        // add spring acceleration
        #[cfg(feature = "springs")]
        add_spring_acceleration();
        // add viscosity acceleration
        add_viscosity_acceleration::<K>(&mut self.fluid, &self.boundary, &self.parameters);
    }

    /// Calculate acceleration at current time
    ///
    /// Supports: Gravity, spring force, viscosity and pressure acceleration
    // #[cfg(feature = "local_pressure")]
    fn calc_acceleration(&mut self) {
        // reset acceleration
        reset_acceleration(&mut self.fluid);
        // add non-pressure acceleration
        self.add_non_pressure_acceleration();
        // compute pressure
        self.pressure_solver.solve_and_add_acceleration::<K>(
            &mut self.fluid,
            &self.boundary,
            &self.parameters,
            &mut self.properties,
        );
    }

    /// Step forward in time one time increment.
    ///
    /// This includes calculating all parameters of the system at the next point in time.
    pub fn step_forward_in_time(&mut self) {
        // measure wall clock time for time step
        let start = std::time::Instant::now();

        self.integrator.integrate(&mut self.fluid, self.parameters.time_increment);

        self.time_steps_propagated += 1;
        // Update
        self.update();
        // measure wall clock time for time step
        self.properties.time_step_wall_clock_time = start.elapsed().as_secs_f64();
    }

    /// Update particle properties and uniform grid
    fn update(&mut self) {
        // remove irrelevant particles: particles below threshold
        // (NOTE: Removed particles must not be connected via spring
        let mut id = 0;
        while id < self.fluid.len() {
            if self.fluid.position[id][2] < self.parameters.disable_particles_below {
                self.fluid.disable(id);
            } else {
                id += 1;
            }
        }
        self.fluid.drop_inactive(); // truncate
        // update uniform grid of fluid particles
        self.fluid_neighbor_search.clear();
        self.fluid_neighbor_search.populate(&self.fluid);
        // update neighbors of all fluid particles
        self.update_particle_neighbors();
        // compute density
        update_volume::<K>(&mut self.fluid, &self.boundary, &self.parameters);
        // calculate new accelerations
        self.calc_acceleration();
        // update properties
        self.properties.update(self.calc_average_mass_density());
        // set new cfl time step conditionally
        #[cfg(any(feature = "logging", feature = "cfl_time_step"))]
        let max_speed = self.calc_max_speed();
        #[cfg(feature = "cfl_time_step")]
        self.parameters.set_cfl_time_step(max_speed);
        #[cfg(all(feature = "logging", not(feature = "cfl_time_step")))]
        {
            debug!(
                "cfl number: {}",
                self.parameters.time_increment * max_speed
                    / self.parameters.rest_density_grid_spacing
            );
        }
        // // take and store additional measurements
        // if self.time() >= 2.0 && self.time() < 2.1 {
        //     let _ = self.save_pressure_profile();
        //     let _ = self.save_kinetic_energy_profile();
        // }
    }

    /// Measure (physical) quantities at current time step
    pub fn push_back_measurement(&mut self, series: &mut measurement::MeasurementSeries) {
        // if cfg!(feature = "logging") {
        //     debug!("{}, {}", self.properties.average_density, self.properties.rest_density);
        //     let max_speed = self.calc_max_speed();
        //     let cfl_coeff = max_speed*self.properties.time_increment/self.properties.rest_density_grid_spacing;
        //     debug!("time: {}, cfl coefficient: {}, max speed: {}", self.time(), cfl_coeff, max_speed);
        // }

        // series.push_back(measurement::Measurement {
        //     time: self.time(),
        //     density: self.properties.average_density,
        //     kinetic_energy: self.calc_average_kinetic_energy(),
        //     stiffness: self.parameters.stiffness,
        //     fluid_viscosity: self.parameters.fluid_viscosity,
        //     boundary_viscosity: self.parameters.boundary_viscosity,
        //     fluid_depth: self.properties.fluid_depth,
        //     rest_density_grid_spacing: self.parameters.rest_density_grid_spacing,
        //     smoothing_length: self.parameters.smoothing_length,
        //     rest_density: self.parameters.rest_density,
        //     time_step_size: self.parameters.time_increment,
        //     target_density_error: 0.,
        //     // target_density_error: 0.,
        //     target_density_error: self.parameters.target_density_error,
        //     solver_iterations: 0,
        //     solver_iterations: self.properties.solver_iterations,
        //     relaxation_factor: 0.,
        //     relaxation_factor: self.parameters.relaxation_factor,
        //     time_step_wall_clock_time: self.properties.time_step_wall_clock_time,
        //     predicted_density_error: self.properties.predicted_density_error,
        // });
    }

    fn get_serializable_particles(&self) -> SerFluid3D {
        self.fluid.clone().into()
    }

    fn get_serializable_boundary_particles(&self) -> SerBoundary3D {
        self.boundary.clone().into()
    }

    pub fn get_time_step_info(&self) -> TimeStepInfo {
        TimeStepInfo {
            time: self.time() as f32,
            time_increment: self.parameters.time_increment as f32,
            average_density: self.properties.average_density as f32,
            fluid: self.get_serializable_particles(),
            boundary: self.get_serializable_boundary_particles(),
        }
    }
}

pub trait Outer {
    type OuterProductType;
    fn outer(&self, other: &Self) -> Self::OuterProductType;
}

impl<N: Copy + std::ops::Mul<N, Output = N> + Zero> Outer for Vector3<N> {
    type OuterProductType = Matrix3<N>;

    fn outer(&self, other: &Self) -> Self::OuterProductType {
        Matrix3::new(
            self[0] * other[0],
            self[0] * other[1],
            self[0] * other[2],
            self[1] * other[0],
            self[1] * other[1],
            self[1] * other[2],
            self[2] * other[0],
            self[2] * other[1],
            self[2] * other[2],
        )
    }
}
