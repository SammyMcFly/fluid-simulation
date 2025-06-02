//! Module for scene building and parameter importing
//!
//!
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Config {
    // pub gui: GuiConfig,
    pub params: Parameters,
    pub scene: SceneConfig,

}

// #[derive(Debug, Deserialize)]
// struct GuiConfig {
//     name: String,
//     version: String,
// }

#[derive(Debug, Deserialize)]
pub struct SceneConfig {
    pub particles: ParticleConfig,
    pub boundary_particles: BoundaryParticleConfig,
    // pub springs: SpringConfig,
    pub light: Lightconfig,
}

#[derive(Debug, Deserialize)]
pub struct Parameters {
    pub buffer_length_limit: u32,
    pub time_inc: f64,
    pub viscosity: f64,
    pub stiffness: f64,
    pub integration_scheme: super::physics::PropagationMethod,
    pub particle_mass: f64,
    pub particle_diameter: f64,
    // pub smoothing_length: f64,
}

#[derive(Debug, Deserialize)]
pub struct ParticleConfig {
    pub n_particles_x: usize,
    pub n_particles_y: usize,
    pub n_particles_z: usize,
    pub x_offset: f64,
    pub y_offset: f64,
    pub z_offset: f64,
    pub particle_spacing: f64,
}

#[derive(Debug, Deserialize)]
pub struct BoundaryParticleConfig {
    pub n_floor_particles_x: usize,
    pub n_floor_particles_y: usize,
    pub n_floor_particles_z: usize,
    pub wall_height: usize,
    pub x_wall_thickness: usize,
    pub y_wall_thickness: usize,
    pub x_offset: f64,
    pub y_offset: f64,
    pub z_offset: f64,
    pub particle_spacing: f64,
}

#[derive(Debug, Deserialize)]
pub struct Lightconfig {
    pub position: [f32; 3],
}

// #[derive(Debug, Deserialize)]
// pub struct SpringConfig {

// }



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
