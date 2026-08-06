//! # Core SPH simulation
//!
//! Contains the simulated system, the information of the individual samples
//! and provides the methods for propagating the system in time.
pub mod boundary_handling;
pub mod kernel;
mod non_pressure_accelerations;
pub mod pressure_solver;
mod quantities;

use crate::fluid::*;
use crate::integration_schemes::IntegrationScheme;
use crate::measurement::{self, Measurement};
use crate::neighbor_search::{NeighborList, NeighborSearch};
use crate::render_info::{BoundaryVisualization, ScalarQuantity};
use crate::setup;
use crate::sph::boundary_handling::BoundaryHandling;
use crate::sph::non_pressure_accelerations::*;
use crate::sph::quantities::{
    get_density, get_density_error, get_kinetic_energy, get_pressure, get_speed,
};
use crate::utilities::triangle_mesh::RenderMesh;
use crate::utilities::vector;
use kernel::KernelFn;
use pressure_solver::PressureSolver;
use quantities::get_volume;

use dyn_clone::DynClone;
use nalgebra::{Matrix3, Point3, Vector3};
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::rc::Rc;
#[cfg(feature = "logging")]
use tracing::{debug, warn}; // debug, error, info, span, trace, warn,

pub trait SPHSystem: DynClone {
    fn time(&self) -> f64;
    fn time_steps_propagated(&self) -> u64;

    /// Step forward in time one time increment.
    ///
    /// This includes calculating all parameters of the system at the next point in time.
    fn step_forward_in_time(&mut self);

    fn continue_from_checkpoint(&mut self, checkpoint: Rc<Checkpoint>);

    /// Measure (physical) quantities at current time step
    fn take_measurement(&self) -> Measurement;

    fn get_fluid_ids(&self) -> Vec<u32>;
    fn get_fluid_pos(&self) -> Vec<[f32; 3]>;
    fn get_serialized_fluid(&self) -> SerFluid3D;
    fn get_quantity_of_fluid_samples(&self, quantity: &ScalarQuantity) -> Vec<f32>;
    fn get_quantity_at_positions(
        &mut self,
        quantity: &ScalarQuantity,
        positions: &[[f32; 3]],
    ) -> Vec<f32>;

    fn get_fluid_surface(&self) -> Vec<(u32, RenderMesh)>;

    fn get_boundary_visualization(&self, selector: &BoundaryVisualization)
    -> BoundaryVisualization;
}

dyn_clone::clone_trait_object!(SPHSystem);

///  3D implementation of a physical system to be simulated
#[derive(Debug, Clone)]
pub struct System3D<
    K: KernelFn,
    I: IntegrationScheme,
    P: PressureSolver,
    N: NeighborSearch,
    B: BoundaryHandling,
> {
    /// Time step number
    time_steps_propagated: u64,
    /// Properties of the system
    parameters: SystemParameters,
    properties: CurrentSystemProperties,
    /// Collection of all fluid samples
    fluid: Fluid3D,
    /// List of fluid neighbors
    fluid_neighbor_list: NeighborList,
    /// Collection of all boundary (not moving) samples
    boundary: B,
    _kernel_fn: std::marker::PhantomData<K>,
    integrator: I,
    pressure_solver: P,
    neighbor_search: N,
}

impl<
    K: KernelFn + Clone + 'static,
    I: IntegrationScheme + Clone + 'static,
    P: PressureSolver + Clone + 'static,
    N: NeighborSearch + Clone + 'static,
    B: BoundaryHandling + Clone + 'static,
> SPHSystem for System3D<K, I, P, N, B>
{
    fn time(&self) -> f64 {
        #[cfg(not(feature = "cfl_time_step"))]
        return (self.time_steps_propagated as f64) * self.parameters.time_increment;
        #[cfg(feature = "cfl_time_step")]
        return self.parameters.current_time;
    }

    fn time_steps_propagated(&self) -> u64 {
        self.time_steps_propagated
    }

    /// Step forward in time one time increment.
    ///
    /// This includes calculating all parameters of the system at the next point in time.
    fn step_forward_in_time(&mut self) {
        // measure wall clock time for time step
        let start = std::time::Instant::now();

        self.integrator
            .integrate(&mut self.fluid, self.parameters.time_increment);

        self.time_steps_propagated += 1;
        // Update
        self.update();
        // measure wall clock time for time step
        self.properties.time_step_wall_clock_time = start.elapsed().as_secs_f64();
    }

    fn continue_from_checkpoint(&mut self, checkpoint: Rc<Checkpoint>) {
        self.time_steps_propagated = checkpoint.get_time_steps_propagated();
        self.fluid = checkpoint.get_fluid().clone().into();
        // self.boundary = checkpoint.get_boundary().clone();
        self.update();
    }

    /// Measure (physical) quantities at current time step
    fn take_measurement(&self) -> Measurement {
        // if cfg!(feature = "logging") {
        //     debug!(
        //         "{}, {}",
        //         self.properties.average_density, self.properties.rest_density
        //     );
        //     let max_speed = self.calc_max_speed();
        //     let cfl_coeff = max_speed * self.properties.time_increment
        //         / self.properties.rest_density_grid_spacing;
        //     debug!(
        //         "time: {}, cfl coefficient: {}, max speed: {}",
        //         self.time(),
        //         cfl_coeff,
        //         max_speed
        //     );
        // }

        let solver_info = self.pressure_solver.measurement_info();

        measurement::Measurement {
            time: self.time(),
            density: self.properties.average_density,
            density_error: self.calc_average_mass_density_error(),
            kinetic_energy: self.calc_average_kinetic_energy(),
            stiffness: solver_info.stiffness,
            fluid_viscosity: self.parameters.fluid_viscosity,
            boundary_viscosity: self.parameters.boundary_viscosity,
            fluid_depth: self.properties.fluid_depth,
            rest_density_grid_spacing: self.parameters.rest_density_grid_spacing,
            kernel_support_radius: self.parameters.kernel_support_radius,
            time_step_size: self.parameters.time_increment,
            target_density_error: solver_info.target_density_error,
            solver_iterations: solver_info.solver_iterations,
            relaxation_factor: solver_info.relaxation_factor,
            time_step_wall_clock_time: self.properties.time_step_wall_clock_time,
            predicted_density_error: solver_info.predicted_density_error,
        }
    }

    fn get_fluid_ids(&self) -> Vec<u32> {
        self.fluid.fluid_id.clone()
    }

    fn get_fluid_pos(&self) -> Vec<[f32; 3]> {
        self.fluid
            .position
            .iter()
            .map(|pos| [pos.x as f32, pos.y as f32, pos.z as f32])
            .collect()
    }

    fn get_serialized_fluid(&self) -> SerFluid3D {
        self.fluid.clone().into()
    }

    fn get_quantity_of_fluid_samples(&self, quantity: &ScalarQuantity) -> Vec<f32> {
        match quantity {
            ScalarQuantity::Speed => self
                .fluid
                .velocity
                .iter()
                .map(|vel| vel.norm() as f32)
                .collect(),
            ScalarQuantity::Volume => self.fluid.volume.iter().map(|vol| *vol as f32).collect(),
            ScalarQuantity::Density => self
                .fluid
                .volume
                .iter()
                .zip(&self.fluid.mass)
                .map(|(vol, mass)| (*mass / *vol) as f32)
                .collect(),
            ScalarQuantity::DensityError => self
                .fluid
                .volume
                .iter()
                .map(|vol| {
                    if *vol < self.parameters.rest_volume {
                        (100. * (self.parameters.rest_volume / *vol - 1.)) as f32
                    } else {
                        0.
                    }
                })
                .collect(),
            ScalarQuantity::Pressure => self.fluid.pressure.iter().map(|p| *p as f32).collect(),
            ScalarQuantity::KineticEnergy => self
                .fluid
                .velocity
                .iter()
                .zip(&self.fluid.mass)
                .map(|(vel, mass)| (0.5 * *mass * vel.norm_squared()) as f32)
                .collect(),
        }
    }

    fn get_quantity_at_positions(
        &mut self,
        quantity: &ScalarQuantity,
        positions: &[[f32; 3]],
    ) -> Vec<f32> {
        let mut neighbor_list = NeighborList::new(positions.len());
        let positions: &Vec<Point3<f64>> = &positions
            .iter()
            .map(|pos| Point3::new(pos[0] as f64, pos[1] as f64, pos[2] as f64))
            .collect();
        self.neighbor_search.find_samples(
            self.parameters.kernel_support_radius,
            positions,
            &self.fluid.position,
            &mut neighbor_list,
        );
        self.boundary.find_boundary_samples(
            &mut self.neighbor_search,
            self.parameters.kernel_support_radius,
            positions,
            self.parameters.rest_density_grid_spacing,
        );
        let mut q = vec![0.; positions.len()];
        match quantity {
            ScalarQuantity::Speed => {
                get_speed::<K>(
                    &mut q,
                    positions,
                    &neighbor_list,
                    &self.fluid.position,
                    &self.fluid.velocity,
                    &self.fluid.volume,
                    &self.boundary,
                    &self.parameters,
                );
                q.iter().map(|speed| *speed as f32).collect()
            }
            ScalarQuantity::Volume => {
                get_volume::<K>(
                    &mut q,
                    positions,
                    &neighbor_list,
                    &self.fluid.position,
                    &self.boundary,
                    &self.parameters,
                );
                q.iter().map(|vol| *vol as f32).collect()
            }
            ScalarQuantity::Density => {
                get_density::<K>(
                    &mut q,
                    positions,
                    &neighbor_list,
                    &self.fluid.position,
                    &self.fluid.mass,
                    // &self.boundary,
                    &self.parameters,
                );
                q.iter().map(|speed| *speed as f32).collect()
            }
            ScalarQuantity::DensityError => {
                get_density_error::<K>(
                    &mut q,
                    positions,
                    &neighbor_list,
                    &self.fluid.position,
                    &self.fluid.mass,
                    // &self.boundary,
                    &self.parameters,
                );
                q.iter().map(|speed| *speed as f32).collect()
            }
            ScalarQuantity::Pressure => {
                get_pressure::<K>(
                    &mut q,
                    positions,
                    &neighbor_list,
                    &self.fluid.position,
                    &self.fluid.volume,
                    &self.fluid.pressure,
                    // &self.boundary,
                    &self.parameters,
                );
                q.iter().map(|speed| *speed as f32).collect()
            }
            ScalarQuantity::KineticEnergy => {
                get_kinetic_energy::<K>(
                    &mut q,
                    positions,
                    &neighbor_list,
                    &self.fluid.position,
                    &self.fluid.velocity,
                    &self.fluid.volume,
                    &self.fluid.mass,
                    // &self.boundary,
                    &self.parameters,
                );
                q.iter().map(|speed| *speed as f32).collect()
            }
        }
    }

    fn get_fluid_surface(&self) -> Vec<(u32, RenderMesh)> {
        self.fluid.reconstruct_surfaces(
            self.parameters.rest_density_grid_spacing,
            self.parameters.rest_volume,
            self.parameters.kernel_support_radius,
        )
    }

    fn get_boundary_visualization(
        &self,
        selector: &BoundaryVisualization,
    ) -> BoundaryVisualization {
        self.boundary.get_visualization(selector)
    }
}

impl<
    K: KernelFn + Clone + 'static,
    I: IntegrationScheme + Clone + 'static,
    P: PressureSolver + Clone + 'static,
    N: NeighborSearch + Clone + 'static,
    B: BoundaryHandling + Clone + 'static,
> System3D<K, I, P, N, B>
{
    pub fn new_boxed(constructor: setup::System3DConstructor<K, I, P, N, B>) -> Box<dyn SPHSystem> {
        let len = constructor.fluid.len();
        let mut system = Self {
            fluid: constructor.fluid,
            fluid_neighbor_list: NeighborList::new(len),
            boundary: constructor.boundary,
            time_steps_propagated: 0,
            parameters: constructor.system_parameters,
            properties: CurrentSystemProperties::default(),
            _kernel_fn: std::marker::PhantomData,
            integrator: constructor.integrator,
            pressure_solver: constructor.pressure_solver,
            neighbor_search: constructor.neighbor_search,
        };
        // system.init_boundary_volume();
        // Update uniform grid
        system.update();
        Box::new(system) as Box<dyn SPHSystem>
    }

    pub fn time(&self) -> f64 {
        (self.time_steps_propagated as f64) * self.parameters.time_increment
    }

    /// Calculate 2-norm of maximum velocity of any particle
    fn calc_max_speed(&self) -> f64 {
        #[cfg(not(feature = "parallel"))]
        {
            self.fluid
                .velocity
                .iter()
                .map(|vel| vel.norm())
                .fold(0.0_f64, f64::max)
        }
        #[cfg(feature = "parallel")]
        {
            self.fluid
                .velocity
                .par_iter()
                .map(|vel| vel.norm())
                .reduce(|| 0.0_f64, f64::max)
        }
    }

    /// Calculate average mass density for all fluid particles
    fn calc_average_mass_density(&self) -> f64 {
        #[cfg(not(feature = "parallel"))]
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
        #[cfg(feature = "parallel")]
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

    /// Calculate average mass density error over all fluid particles
    fn calc_average_mass_density_error(&self) -> f64 {
        #[cfg(not(feature = "parallel"))]
        let (total_mass_density, count) = {
            self.fluid
                .volume
                .iter()
                .map(|volume| {
                    if *volume < self.parameters.rest_volume {
                        self.parameters.rest_volume / volume
                    } else {
                        1.
                    }
                })
                .fold((0.0_f64, 0_u64), |(sum, cnt), d| (sum + d, cnt + 1))
        };
        #[cfg(feature = "parallel")]
        let (total_mass_density_error, count) = self
            .fluid
            .volume
            .par_iter()
            .map(|volume| {
                if *volume < self.parameters.rest_volume {
                    self.parameters.rest_volume / volume
                } else {
                    1.
                }
            })
            .fold(
                || (0.0_f64, 0_u64),
                |(sum, cnt), mass_density_error| (sum + mass_density_error, cnt + 1),
            )
            .reduce(
                || (0.0, 0),
                |(sum_a, cnt_a), (sum_b, cnt_b)| (sum_a + sum_b, cnt_a + cnt_b),
            );
        if count > 0 {
            100. * (total_mass_density_error / count as f64 - 1.)
        } else {
            0.0
        }
    }

    /// Calculate average kinetic energy over all fluid particles
    fn calc_average_kinetic_energy(&self) -> f64 {
        #[cfg(not(feature = "parallel"))]
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
        #[cfg(feature = "parallel")]
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

    /// Calculate non-pressure accelerations and add them to each particles acceleration
    fn add_non_pressure_acceleration(&mut self) {
        // add gravity acceleration
        add_gravity_acceleration(&mut self.fluid);
        // add viscosity acceleration
        add_viscosity_acceleration::<K>(
            &mut self.fluid,
            &self.boundary,
            &self.fluid_neighbor_list,
            &self.parameters,
        );
    }

    /// Calculate acceleration at current time
    ///
    /// Supports: Gravity, viscosity and pressure acceleration
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
            &self.fluid_neighbor_list,
            &self.parameters,
            &mut self.properties,
        );
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
        // truncate arrays
        self.fluid.drop_inactive();
        // update neighbors of fluid particles
        self.neighbor_search.find_samples(
            self.parameters.kernel_support_radius,
            &self.fluid.position,
            &self.fluid.position,
            &mut self.fluid_neighbor_list,
        );
        self.boundary.find_boundary_samples(
            &mut self.neighbor_search,
            self.parameters.kernel_support_radius,
            &self.fluid.position,
            self.parameters.rest_density_grid_spacing,
        );
        // compute density
        get_volume::<K>(
            &mut self.fluid.volume,
            &self.fluid.position,
            &self.fluid_neighbor_list,
            &self.fluid.position,
            &self.boundary,
            &self.parameters,
        );
        // calculate new accelerations
        self.calc_acceleration();
        // update properties
        self.properties.update(self.calc_average_mass_density());
        // set new cfl time step conditionally
        #[cfg(feature = "cfl_time_step")]
        {
            self.parameters.current_time += self.parameters.time_increment;
        }
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

/// Information about the system at the current time
#[derive(Debug, Clone, Default)]
pub struct CurrentSystemProperties {
    average_density: f64,
    fluid_depth: f64,
    /// wall clock time passed calculating current time step
    time_step_wall_clock_time: f64,
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
    current_time: f64,
    #[cfg(feature = "cfl_time_step")]
    max_time_increment: f64,
    #[cfg(feature = "cfl_time_step")]
    pub cfl_number: f64,
    /// Smooting length h
    kernel_support_radius: f64,
    /// disable particles below this threshold
    disable_particles_below: f64,
    rest_volume: f64,
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
        rest_density_grid_spacing: f64,
        kernel_support_radius: f64,
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
            current_time: 0.,
            #[cfg(feature = "cfl_time_step")]
            max_time_increment,
            #[cfg(feature = "cfl_time_step")]
            cfl_number,
            kernel_support_radius,
            disable_particles_below,
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

// pub struct Checkpointy<B: BoundaryHandling> {
pub struct Checkpoint {
    time_steps_propagated: u64,
    fluid: SerFluid3D,
    // boundary: B,
}

// impl<B: BoundaryHandling> Checkpointy<B> {
impl Checkpoint {
    pub fn from_sph_system(system: &dyn SPHSystem) -> Self {
        Self {
            time_steps_propagated: system.time_steps_propagated(),
            fluid: system.get_serialized_fluid(),
            // boundary: self.boundary.get_checkpoint_data(),
        }
    }

    pub fn get_time_steps_propagated(&self) -> u64 {
        self.time_steps_propagated
    }

    pub fn get_fluid(&self) -> &SerFluid3D {
        &self.fluid
    }

    // pub fn get_boundary(&self) -> &dyn BoundaryHandling {
    //     &self.boundary
    // }
}
