//! Module for scene building and parameter importing
//!
//!
mod scenes;

use std::sync::{Arc, Mutex};
use serde::Deserialize;

use super::physics::particle::{Particle3D, SerParticle3D, BoundaryParticle3D};
use super::physics::spring::Spring;
use super::physics::{SystemProperties, PropagationMethod, cubic_b_spline_3d, cubic_b_spline_3d_gradient};
use super::mediation;
use super::measure;


#[derive(Debug, Deserialize)]
pub struct Setup {
    pub parameters: Parameters,
    pub light: Light,
    pub scene: SceneVariant,
}

#[derive(Debug, Deserialize)]
pub struct Parameters {
    pub buffer_length_limit: u32,
    pub time_inc: f64,
    pub viscosity: f64,
    pub stiffness: f64,
    pub integration_scheme: PropagationMethod,
    pub rest_density: f64,
    pub rest_density_grid_spacing: f64,
    pub smoothing_length: f64,
    pub disable_particles_below: f64,
}

#[derive(Debug, Deserialize)]
pub struct Light {
    pub position: [f32; 3],
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", content = "parameters")]
pub enum SceneVariant {
    NoLidCube(scenes::NoLidCube),
    Spiral(scenes::Spiral),
}

impl Scene for SceneVariant {
    fn get_boundary(&self, rest_density_grid_spacing: f64) -> Vec<BoundaryParticle3D> {
        match self {
            Self::NoLidCube(variant) => variant.get_boundary(rest_density_grid_spacing),
            Self::Spiral(variant) => variant.get_boundary(rest_density_grid_spacing),
        }
    }
    fn get_fluid(&self, rest_density: f64, rest_density_grid_spacing: f64) -> Vec<Particle3D> {
        match self {
            Self::NoLidCube(variant) => variant.get_fluid(rest_density, rest_density_grid_spacing),
            Self::Spiral(variant) => variant.get_fluid(rest_density, rest_density_grid_spacing),
        }
    }
    fn get_springs(&self) -> Vec<Spring> {
        match self {
            Self::NoLidCube(variant) => variant.get_springs(),
            Self::Spiral(variant) => variant.get_springs(),
        }
    }
    fn calc_fluid_depth(&self, rest_density_grid_spacing: f64) -> f64 {
        match self {
            Self::NoLidCube(variant) => variant.calc_fluid_depth(rest_density_grid_spacing),
            Self::Spiral(variant) => variant.calc_fluid_depth(rest_density_grid_spacing),
        }
    }
}

trait Scene {
    fn get_boundary(&self, rest_density_grid_spacing: f64) -> Vec<BoundaryParticle3D>;
    fn get_fluid(&self, rest_density: f64, rest_density_grid_spacing: f64) -> Vec<Particle3D>;
    fn get_springs(&self) -> Vec<Spring>;
    fn calc_fluid_depth(&self, rest_density_grid_spacing: f64) -> f64;
}


pub struct System3DConfig {
    pub particles: Vec<Particle3D>,
    pub boundary_particles: Vec<BoundaryParticle3D>,
    pub springs: Vec<Spring>,
    pub system_properties: SystemProperties,
    pub controls: Arc<Mutex<super::mediation::IntermediateControls>>,
    pub measurement_series: Option<Arc<Mutex<measure::MeasurementSeries>>>,
}

pub struct System3DConfigConstructor {
    config: Setup,
    build: Option<System3DConfig>,
}

impl System3DConfigConstructor {
    fn load_config(file_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Read the scene config file
        let config_file_content = std::fs::read_to_string(file_path)?;
        // Parse the content into the Config struct
        Ok(Self {
            config: toml::from_str(&config_file_content)?,
            build: None,
        })
    }

    fn load_particles(file_path: &str) -> Result<Vec<Particle3D>, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(file_path)?;
        let particles: Vec<SerParticle3D> = ron::from_str(&content).unwrap();
        Ok(particles.into_iter().map(|p| p.into()).collect())
    }

    fn get_system_properties(&self) -> SystemProperties {
        SystemProperties::new(
            self.config.parameters.time_inc,
            self.config.parameters.rest_density,
            self.config.parameters.rest_density_grid_spacing,
            self.config.parameters.smoothing_length,
            self.config.parameters.disable_particles_below,
            self.config.scene.calc_fluid_depth(self.config.parameters.rest_density_grid_spacing),
            self.config.parameters.viscosity,
            self.config.parameters.stiffness,
            cubic_b_spline_3d,
            cubic_b_spline_3d_gradient
        )
    }

    fn build(
        &mut self,
        particles: Vec<Particle3D>,
        boundary_particles: Vec<BoundaryParticle3D>,
        springs: Vec<Spring>,
        system_properties: SystemProperties,
        controls: Arc<Mutex<super::mediation::IntermediateControls>>,
        measurement_series: Option<Arc<Mutex<measure::MeasurementSeries>>>,
    ) {
        self.build = Some(System3DConfig { particles, boundary_particles, springs, system_properties, controls, measurement_series, });
    }

    pub fn new(
        config_file_path: &str,
        particle_state_file_path: Option<&str>,
        controls: Arc<Mutex<mediation::IntermediateControls>>,
        measurement_series: Option<Arc<Mutex<measure::MeasurementSeries>>>,
    ) -> Result<(Self, u32, PropagationMethod), Box<dyn std::error::Error>> {
        // load config file
        let mut constructor = Self::load_config(config_file_path)?;

        // load particles
        let particles = if let Some(particle_state_file_path) = particle_state_file_path {
            Self::load_particles(particle_state_file_path)?
        } else {
            constructor.config.scene.get_fluid(constructor.config.parameters.rest_density, constructor.config.parameters.rest_density_grid_spacing)
        };

        // copy buffer length
        let buffer_length = constructor.config.parameters.buffer_length_limit;

        // copy integration scheme
        let integration_scheme = constructor.config.parameters.integration_scheme.clone();

        // load boundary
        let boundary_particles = constructor.config.scene.get_boundary(constructor.config.parameters.rest_density_grid_spacing);
        // load springs
        let springs = constructor.config.scene.get_springs();

        // hand over time_inc
        controls.lock().unwrap().set_time_inc(constructor.config.parameters.time_inc as f32);
        // hand over particle size
        controls.lock().unwrap().set_particle_diameter(constructor.config.parameters.rest_density_grid_spacing as f32);
        // hand over rest density
        controls.lock().unwrap().set_rest_density(constructor.config.parameters.rest_density as f32);
        // hand over light position
        controls.lock().unwrap().set_light_position(constructor.config.light.position);

        // init system properties
        let system_properties = constructor.get_system_properties();

        constructor.build(
            particles,
            boundary_particles,
            springs,
            system_properties,
            controls,
            measurement_series,
        );
        // create simulation system
        Ok((constructor, buffer_length, integration_scheme))
    }

    pub fn finish(self) -> System3DConfig {
        self.build.unwrap()
    }
}













// pub trait SimulationSystem {
//     fn calc_acceleration(&mut self);
//     fn particles(&self) -> &Vec<Particle3D>;
//     fn springs(&self) -> &Vec<Spring>;
//     fn update(&mut self);
// }


// ///  3D implementation of a physical system to be simulated
// #[derive(Debug, Clone)]
// pub struct System3D {
//     /// Collection of all particles
//     particles: Vec<Particle3D>,
//     /// Uniform grid
//     ///
//     /// Accelerates neighbor search
//     uniform_grid: uniform_grid::UniformGrid,
//     /// Springs connecting different particles
//     ///
//     /// Spring stores indices of particles connected to via spring force,
//     /// spring force coeff (k) and rest length (l)
//     springs: Vec<Spring>,
//     time: f64,
//     time_inc: f64,
//     kernel_support: f64,
//     /// Queue for queuing all particle position for visualization
//     pub queue: Arc<Mutex<super::queue::IntermediateQueue>>,
// }

// impl System3D {
//     pub fn new(particles: Vec<Particle3D>, springs: Vec<Spring>, time_inc: f64, kernel_support: f64)
//     -> Self {
//         let queue = super::queue::IntermediateQueue::default();
//         let uniform_grid = uniform_grid::UniformGrid::new(&particles, kernel_support);
//         let system = Self {
//             particles,
//             uniform_grid,
//             springs,
//             time: 0.0,
//             time_inc,
//             kernel_support,
//             queue: Arc::new(Mutex::new(queue)),
//         };
//         // Add initial positions to queue
//         system.queue.lock().unwrap().push_back(system.particles_as_instances());
//         system
//     }

//     pub fn calc_kernel(&self, kernel_fn: fn(distance: f64, kernel_support: f64) -> f64) -> f64 {
//         todo!()
//     }

//     pub fn calc_kernel_gradient(&self, kernel_gradient_fn: fn(distance: f64, kernel_support: f64, direction: Vector3<f64>) -> f64) -> Vector3<f64> {
//         todo!()
//     }

//     /// Calculate acceleration at current time
//     ///
//     /// Supports: Spring force acceleration
//     fn calc_acceleration(&mut self) {
//         // calculate spring forces
//         for super::physics::spring::Spring { indices: (i1, i2), k, l, ..} in &self.springs {
//             // calculate force for spring
//             let force = k/l
//                 *((self.particles[*i2].pos().now()-self.particles[*i1].pos().now())
//                 - (*l*(self.particles[*i2].pos().now()-self.particles[*i1].pos().now())
//                 /(self.particles[*i2].pos().now()-self.particles[*i1].pos().now()).norm()));

//             let m: f64 = self.particles[*i1].m();
//             self.particles[*i1].set_new_acc(force/m);
//             let m: f64 = self.particles[*i2].m();
//             self.particles[*i2].set_new_acc(-force/m);
//         }
//         // calculate other forces here
//     }
//     fn update(&mut self) {
//         // forward new particle positions to graphics output
//         self.queue.lock().unwrap().push_back(self.particles_as_instances());
//         // update uniform grid
//         self.uniform_grid.clear();
//         self.uniform_grid.populate(&self.particles);
//     }

//     fn particles_as_instances(&self) -> Vec<super::queue::Instance> {
//         let mut result = Vec::new();
//         for v in &self.particles {
//             let instance = super::queue::Instance {
//                 position: v.pos().now().map(|v| { v as f32 }),
//                 color: v.color(),
//             };
//             result.push(instance);
//         }
//         result
//     }
// }

// pub fn inc_time(system: &mut impl SimulationSystem, method: super::physics::PropagationMethod, time_inc: f64) {
//     // calculate accelerations
//     system.calc_acceleration();
//     // increment time one step
//     match method {
//         super::physics::PropagationMethod::ExplicitEuler => {
//             for v in &mut system.particles() {
//                 // update positions
//                 v.set_new_pos(v.pos().now() + time_inc*v.vel().now());
//                 // update velocities
//                 v.set_new_vel(v.vel().now() + time_inc*v.acc());
//             }
//         },
//         super::physics::PropagationMethod::ImplicitEuler => { // Conjugate Gradient implementation
//             // init fractions of the Jacobi matrix that belong to the springs
//             for super::physics::spring::Spring {
//                     indices: (i1, i2),
//                     k,
//                     l,
//                     matrix_s
//             } in &mut system.springs() {
//                 // calculate spacial derivative of spring force of spring
//                 // between vert[i1] and vert[i2] applied to vert[i1] with respect to vert[i1].pos
//                 let x_i2_outer_x_i1 =
//                         (self.particles[*i2].pos().now()-system.particles()[*i1].pos().now())
//                         .outer(&(system.particles()[*i2].pos().now()-system.particles()[*i1].pos().now()));

//                 *matrix_s = *k / *l
//                     *(-Matrix3::identity()+ *l /(system.particles()[*i2].pos().now()-system.particles()[*i1].pos().now()).norm()
//                     *(Matrix3::identity()-1.0/(system.particles()[*i2].pos().now()-system.particles()[*i1].pos().now()).norm().powi(2)
//                     *x_i2_outer_x_i1));
//             }
//             // init variables for iterative numeric solver
//             for v in &mut system.particles() {
//                 let vel = v.vel().now();
//                 v.set_pred_vel(vel);
//                 v.d_l = v.vel().now()+time_inc*v.acc()-v.vel().pred();
//             }
//             for super::physics::spring::Spring {
//                     indices: (i1, i2),
//                     matrix_s,
//                     ..
//             } in system.springs() {
//                 let v_pred = system.particles()[*i1].vel().pred();
//                 let m = system.particles()[*i1].m();
//                 system.particles()[*i1].d_l += time_inc.powi(2)/m * (*matrix_s) * v_pred;

//                 let v_pred = system.particles()[*i2].vel().pred();
//                 let m = system.particles()[*i2].m();
//                 system.particles()[*i2].d_l -= time_inc.powi(2)/m * (*matrix_s) * v_pred;
//             }
//             for v in &mut system.particles() {
//                 v.r_l = v.d_l;
//             }
//             // solve numerically iteratively
//             for _ in 0..5 {
//                 // refresh a_times_d_i
//                 for v in &mut system.particles() {
//                     v.a_times_d_l = v.d_l;
//                 }
//                 for super::physics::spring::Spring {
//                         indices: (i1, i2),
//                         matrix_s, ..} in system.springs() {
//                     let d_l = system.particles()[*i1].d_l;
//                     let m = system.particles()[*i1].m();
//                     system.particles()[*i1].d_l -= time_inc.powi(2)/m * (*matrix_s) * d_l;

//                     let d_l = system.particles()[*i2].d_l;
//                     let m = system.particles()[*i2].m();
//                     system.particles()[*i2].d_l += time_inc.powi(2)/m * (*matrix_s) * d_l;
//                 }
//                 // do numeric solver iteration
//                 for v in &mut system.particles() {
//                     v.alpha_l = v.r_l.dot(&v.r_l)/(v.d_l.dot(&v.a_times_d_l));
//                     let vel = v.vel().pred()+v.alpha_l*v.d_l;
//                     v.set_pred_vel(vel);
//                     let r_l_old = v.r_l;
//                     v.r_l -= v.alpha_l*v.a_times_d_l;
//                     v.d_l = v.r_l + v.r_l.dot(&v.r_l)/(r_l_old.dot(&r_l_old))*v.d_l;
//                 }
//             }

//             for v in &mut system.particles() {
//                 // set velocity from numeric solver as new velocity
//                 v.accept_pred_vel();
//                 // update positions with new velocities: v_i(t+h)
//                 v.set_new_pos(v.pos().now() + time_inc*v.vel().now());
//             }

//         }
//         super::physics::PropagationMethod::EulerCromer => {
//             for v in &mut system.particles() {
//                 // update velocities
//                 v.set_new_vel(v.vel().now() + time_inc*v.acc());
//                 // update positions with new velocities: v_i(t+h)
//                 v.set_new_pos(v.pos().now() + time_inc*v.vel().now());
//             }
//         },
//         super::physics::PropagationMethod::Verlet => {
//             // update positions
//             for v in &mut system.particles() {
//                 // update positions
//                 let t = 2.0*v.pos().now() - v.pos().prev() + time_inc.powi(2)*v.acc();
//                 v.set_new_pos(t);
//                 // update velocities
//                 let t = (v.pos().now() - v.pos().prev())/time_inc;
//                 v.set_new_vel(t);
//             }
//         },
//     }
//     self.time += time_inc;
//     system.update();
// }
