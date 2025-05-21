use std::sync::{Arc, Mutex};

use nalgebra::{DMatrix, DVector, Vector3};

// use tracing::{debug, error, info, span, trace, warn};






trait Vertex {
    type Pos;
    type Vel;
    type Acc;

    type PosBuffer;
    type VelBuffer;
    type AccBuffer;

    fn pos(&self) -> Self::PosBuffer;
    fn vel(&self) -> Self::VelBuffer;
    fn acc(&self) -> Self::AccBuffer;

    fn set_new_pos(&mut self, pos: Self::Pos);
    fn set_new_vel(&mut self, vel: Self::Vel);
    fn set_new_acc(&mut self, acc: Self::Acc);
    fn m(&self) -> f64;
}

/// Vertex in a 1-dimensional context
#[derive(Debug, Clone)]
pub struct Vertex1D{
    position: [f64; 2],
    velocity: f64,
    acceleration: f64,
    mass: f64,
    color: [f32; 3],
}

impl Vertex for Vertex1D {
    type Pos = f64;
    type Vel = f64;
    type Acc = f64;

    type PosBuffer = [f64; 2];
    type VelBuffer = f64;
    type AccBuffer = f64;

    fn pos(&self) -> Self::PosBuffer {
        self.position
    }
    fn vel(&self) -> Self::VelBuffer{
        self.velocity
    }
    fn acc(&self) -> Self::AccBuffer {
        self.acceleration
    }

    fn set_new_pos(&mut self, pos: Self::Pos){
        self.position[1] = self.position[0];
        self.position[0] = pos;
    }
    fn set_new_vel(&mut self, vel: Self::Vel) {
        self.velocity = vel;
    }
    fn set_new_acc(&mut self, acc: Self::Acc) {
        self.acceleration = acc;
    }
    fn m(&self) -> f64 {
        self.mass
    }
}

impl Vertex1D {
    pub fn new(
        position: [f64; 2],
        velocity: f64,
        mass: f64,
        color: [f32; 3],
    ) -> Self {
        Self {
            position: Q3::new(position[0], position[1]),
            velocity,
            acceleration: 0.0,
            mass,
            color,
        }
    }
}

pub enum PropagationMethod {
    ExplicitEuler,
    ImplicitEulerWithMatrix,
    ImplicitEuler,
    EulerCromer,
    Verlet,
}

#[derive(Debug, Clone)]
pub struct System1D {
    /// Vec of all vertices
    vert: Vec<Vertex1D>,
    /// stores indices of vertices the vertex is connected to via spring force
    /// (vertex index, k: spring force coeff, L: rest length)
    edges: Vec<Vec<(usize, f64, f64)>>,
    time: f64,
    time_inc: f64,
    /// Queue for queuing all vertex position for visualization
    pub queue: Arc<Mutex<queue::VisualizationQueue>>,
}

impl System1D {
    pub fn new(vertices: Vec<Vertex1D>, edges: Vec<Vec<(usize, f64, f64)>>, time_inc: f64)
    -> Self {
        let queue = queue::VisualizationQueue::default();
        Self {
            vert: vertices,
            edges,
            time: 0.0,
            time_inc,
            queue: Arc::new(Mutex::new(queue)),
        }
    }

    fn isolate_vertex_positions(&self) -> Vec<queue::Instance> {
        let mut result = Vec::new();
        for v in &self.vert {
            let instance = queue::Instance {
                position: Vector3::new(v.pos()[0] as f32, 0.0, 0.0),
                color: v.color,
            };
            result.push(instance);
        }
        result
    }

    /// Spring force acceleration
    fn calc_current_acc(&mut self) {
        // calculate spring forces
        for i in 0..self.vert.len() {
            for (spring_conn, k_ij, l_ij) in &self.edges[i] {
                let acc = k_ij/l_ij/self.vert[i].m()
                    *((self.vert[*spring_conn].pos()[0]-self.vert[i].pos()[0])
                    - (l_ij*(self.vert[*spring_conn].pos()[0]-self.vert[i].pos()[0])
                    /(self.vert[*spring_conn].pos()[0]-self.vert[i].pos()[0]).powi(2).powf(0.5)));
                self.vert[i].set_new_acc(acc);
            }
        }
    }

    pub fn inc_time(&mut self, method: PropagationMethod) {
        // calculate accelerations
        self.calc_current_acc();
        // increment time one step
        match method {
            PropagationMethod::ExplicitEuler => {
                for v in &mut self.vert {
                    // update positions
                    v.set_new_pos(v.pos()[0] + self.time_inc*v.vel());
                    // update velocities
                    v.set_new_vel(v.vel() + self.time_inc*v.acc());
                }
            },
            PropagationMethod::ImplicitEulerWithMatrix => { // Conjugate Gradient implementation
                let dim = self.vert.len();
                // build v_0 = v_t
                let mut v_t = DVector::<f64>::zeros(dim); // Replace f64 with Matrix3x3::<f64> for 3D
                // build a_t
                let mut a_t = DVector::<f64>::zeros(dim); // Replace f64 with Matrix3x3::<f64> for 3D
                // build A
                let mut mat_cap_a = DMatrix::<f64>::zeros(dim, dim); // Replace f64 with Matrix3x3::<f64> for 3D

                for (i, v) in self.vert.iter().enumerate() {
                    v_t[i] = v.vel();
                    a_t[i] = v.acc();
                    let mut mat_j_ii= 0.0;
                    for (spring_conn, k_ij, l_ij) in &self.edges[i] {
                        mat_j_ii = k_ij/l_ij/v.m()
                            *(-1.0+l_ij/(self.vert[*spring_conn].pos()[0]-v.pos()[0]).powi(2).powf(0.5)
                            *(1.0-1.0/(self.vert[*spring_conn].pos()[0]-v.pos()[0]).powi(2)
                            *(self.vert[*spring_conn].pos()[0]-v.pos()[0]).powi(2))); // Replace 1.0 with identity matrix 3x3, insert matrix (dyadisches) product
                    }
                    mat_cap_a[(i, i)] = 1.0-self.time_inc.powi(2)*mat_j_ii; // Replace 1.0 with identity matrix 3x3
                    for (spring_conn, _, _) in &self.edges[i] {
                        mat_cap_a[(i, *spring_conn)] = self.time_inc.powi(2)*v.m()/self.vert[*spring_conn].m()*mat_j_ii;
                    }
                }

                let s = v_t.clone() + self.time_inc*a_t;

                let mut v_l = v_t;

                let mut r_l = s - &mat_cap_a.clone()*v_l.clone();
                let mut d_l = r_l.clone();
                for _ in 0..5 {
                    let alpha_l = if r_l == DVector::<f64>::zeros(dim)
                            && d_l == DVector::<f64>::zeros(dim) { //prevent div by 0
                        1.0
                    } else {
                        r_l.dot(&r_l)/(d_l.dot(&(mat_cap_a.clone()*d_l.clone())))
                    };
                    v_l += alpha_l*d_l.clone();
                    let r_old = r_l.clone();
                    r_l -= alpha_l*(mat_cap_a.clone()*d_l.clone());

                    d_l = if r_l == DVector::<f64>::zeros(dim)
                            && r_old == DVector::<f64>::zeros(dim) { //prevent div by 0
                        d_l
                    } else {
                        r_l.clone() + r_l.dot(&r_l)/r_old.dot(&r_old)*d_l
                    }
                }

                for i in 0..self.vert.len() {
                    self.vert[i].set_new_vel(v_l[i]);
                }

                for v in &mut self.vert {
                    // update positions with new velocities: v_i(t+h)
                    v.set_new_pos(v.pos()[0] + self.time_inc*v.vel());
                }
            },
            PropagationMethod::ImplicitEuler => {

            }
            PropagationMethod::EulerCromer => {
                for v in &mut self.vert {
                    // update velocities
                    v.set_new_vel(v.vel() + self.time_inc*v.acc());
                    // update positions with new velocities: v_i(t+h)
                    v.set_new_pos(v.pos()[0] + self.time_inc*v.vel());
                }
            },
            PropagationMethod::Verlet => { // todo: Does not work
                // update positions
                for v in &mut self.vert {
                    // update positions
                    let t = 2.0*v.pos()[0] - v.pos()[1] + self.time_inc.powi(2)*v.acc();
                    v.set_new_pos(t);
                    // update velocities
                    let t = (v.pos()[0] - v.pos()[1])/self.time_inc;
                    v.set_new_vel(t);
                }
            },
        }
        self.time += self.time_inc;
        self.queue.lock().unwrap().push_back(self.isolate_vertex_positions());
    }
}
