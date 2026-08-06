//! Implicit Euler integration scheme
use crate::fluid::{Fluid3D, Len};
use crate::for_each;
use crate::integration_schemes::IntegrationScheme;
use nalgebra::Vector3;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

#[derive(Default)]
pub struct ImplicitEuler {
    // Scratch buffers (reused each step, values don't carry over)
    pub d_l: Vec<Vector3<f64>>,
    pub r_l: Vec<Vector3<f64>>,
    pub alpha_l: Vec<f64>,
    pub a_times_d_l: Vec<Vector3<f64>>,
}

impl ImplicitEuler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ensure scratch buffers match current particle count
    fn resize_scratch(&mut self, len: usize) {
        self.d_l.resize(len, Vector3::zeros());
        self.r_l.resize(len, Vector3::zeros());
        self.alpha_l.resize(len, 0.0);
        self.a_times_d_l.resize(len, Vector3::zeros());
    }
}

// impl IntegrationScheme for ImplicitEuler {
//     fn integrate(&mut self, fluid: &mut Fluid3D, _dt: f64) {
//         self.resize_scratch(fluid.len());
//         // Conjugate Gradient implementation
//         // init fractions of the Jacobi matrix that belong to the springs
//         for Spring {
//             indices: (i1, i2),
//             k,
//             l,
//             matrix_s,
//         } in &mut self.springs
//         {
//             // calculate spacial derivative of spring force of spring
//             // between vert[i1] and vert[i2] applied to vert[i1] with respect to vert[i1].pos
//             let x_i2_outer_x_i1 = (self.particles[*i2].pos().now()
//                 - self.particles[*i1].pos().now())
//             .outer(&(self.particles[*i2].pos().now() - self.particles[*i1].pos().now()));

//             *matrix_s = *k / *l
//                 * (-Matrix3::identity()
//                     + *l / (self.particles[*i2].pos().now()
//                         - self.particles[*i1].pos().now())
//                     .norm()
//                         * (Matrix3::identity()
//                             - 1.0
//                                 / (self.particles[*i2].pos().now()
//                                     - self.particles[*i1].pos().now())
//                                 .norm()
//                                 .powi(2)
//                                 * x_i2_outer_x_i1));
//         }
//         // init variables for iterative numeric solver
//         for v in &mut self.particles {
//             let vel = v.vel().now();
//             v.set_pred_vel(vel);
//             v.d_l =
//                 v.vel().now() + self.properties.time_increment * v.acc() - v.vel().pred();
//         }
//         for Spring {
//             indices: (i1, i2),
//             matrix_s,
//             ..
//         } in &self.springs
//         {
//             let v_pred = self.particles[*i1].vel().pred();
//             let m = self.particles[*i1].mass();
//             self.particles[*i1].d_l +=
//                 self.properties.time_increment.powi(2) / m * (*matrix_s) * v_pred;

//             let v_pred = self.particles[*i2].vel().pred();
//             let m = self.particles[*i2].mass();
//             self.particles[*i2].d_l -=
//                 self.properties.time_increment.powi(2) / m * (*matrix_s) * v_pred;
//         }
//         for v in &mut self.particles {
//             v.r_l = v.d_l;
//         }
//         // solve numerically iteratively
//         for _ in 0..5 {
//             // refresh a_times_d_i
//             for v in &mut self.particles {
//                 v.a_times_d_l = v.d_l;
//             }
//             for Spring {
//                 indices: (i1, i2),
//                 matrix_s,
//                 ..
//             } in &self.springs
//             {
//                 let d_l = self.particles[*i1].d_l;
//                 let m = self.particles[*i1].mass();
//                 self.particles[*i1].d_l -=
//                     self.properties.time_increment.powi(2) / m * (*matrix_s) * d_l;

//                 let d_l = self.particles[*i2].d_l;
//                 let m = self.particles[*i2].mass();
//                 self.particles[*i2].d_l +=
//                     self.properties.time_increment.powi(2) / m * (*matrix_s) * d_l;
//             }
//             // do numeric solver iteration
//             for v in &mut self.particles {
//                 v.alpha_l = v.r_l.dot(&v.r_l) / (v.d_l.dot(&v.a_times_d_l));
//                 let vel = v.vel().pred() + v.alpha_l * v.d_l;
//                 v.set_pred_vel(vel);
//                 let r_l_old = v.r_l;
//                 v.r_l -= v.alpha_l * v.a_times_d_l;
//                 v.d_l = v.r_l + v.r_l.dot(&v.r_l) / (r_l_old.dot(&r_l_old)) * v.d_l;
//             }
//         }

//         for v in &mut self.particles {
//             // function produces NaN values for a 0 acceleration
//             // this check prevents spreading of NaN values
//             if v.acc() != Vector3::zeros() {
//                 // set velocity from numeric solver as new velocity
//                 v.accept_pred_vel();
//             }
//             // update positions with new velocities: v_i(t+h)
//             v.set_new_pos(v.pos().now() + self.properties.time_increment * v.vel().now());
//         }
//     }
// }
