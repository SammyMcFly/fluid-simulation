/// Module that contains a representation of a collection of samples for an SPH fluid simulation
///
use bincode::{Decode, Encode};
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use std::slice::SliceIndex;

pub trait Len {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait Expandable {
    fn push(&mut self, position: Vector3<f64>, velocity: Vector3<f64>, mass: f64);
    fn extend(&mut self, other: Self);
}

pub trait Positional {
    fn pos_now<I>(&self, id: I) -> &I::Output
    where
        I: SliceIndex<[Vector3<f64>]>;
}

/// Fluid, i.e. a collection of samples, which are identified by an ID (usize)
///
#[derive(Debug, Clone, Default)]
pub struct Fluid3D {
    num_active: usize,
    pub position: Vec<Vector3<f64>>,
    pub position_prev: Vec<Vector3<f64>>,
    pub position_pred: Vec<Vector3<f64>>,
    pub velocity: Vec<Vector3<f64>>,
    pub velocity_prev: Vec<Vector3<f64>>,
    pub velocity_pred: Vec<Vector3<f64>>,
    pub acceleration: Vec<Vector3<f64>>,
    pub mass: Vec<f64>,
    /// volume (necessary for sph fluid)
    pub volume: Vec<f64>,
    pub pressure: Vec<f64>,
    /// neighbors
    pub neighbors: Vec<Vec<usize>>,
    /// boundary neighbors
    pub boundary_neighbors: Vec<Vec<usize>>,
    // local pressure with splitting variable
    #[cfg(feature = "splitting")]
    pub density_pred: Vec<f64>,
    // global pressure solver variables
    #[cfg(feature = "global_pressure")]
    pub s_f: Vec<f64>,
    #[cfg(feature = "global_pressure")]
    pub a_ff: Vec<f64>,
    #[cfg(feature = "global_pressure")]
    pub pressure_acc_f: Vec<Vector3<f64>>,
    // implicit euler variables
    #[cfg(feature = "implicit_euler")]
    pub d_l: Vec<Vector3<f64>>,
    #[cfg(feature = "implicit_euler")]
    pub r_l: Vec<Vector3<f64>>,
    #[cfg(feature = "implicit_euler")]
    pub alpha_l: Vec<f64>,
    #[cfg(feature = "implicit_euler")]
    pub a_times_d_l: Vec<Vector3<f64>>,
}

impl Len for Fluid3D {
    fn len(&self) -> usize {
        self.num_active
    }
}

impl Expandable for Fluid3D {
    fn push(&mut self, position: Vector3<f64>, velocity: Vector3<f64>, mass: f64) {
        self.position.push(position);
        self.position_prev.push(position);
        self.position_pred.push(Vector3::zeros());
        self.velocity.push(velocity);
        self.velocity_prev.push(Vector3::zeros());
        self.velocity_pred.push(Vector3::zeros());
        self.acceleration.push(Vector3::zeros());
        self.mass.push(mass);
        self.volume.push(0.);
        self.pressure.push(0.);
        self.neighbors.push(Vec::new());
        self.boundary_neighbors.push(Vec::new());
        #[cfg(feature = "splitting")]
        self.density_pred.push(0.);
        #[cfg(feature = "global_pressure")]
        self.s_f.push(0.);
        #[cfg(feature = "global_pressure")]
        self.a_ff.push(0.);
        #[cfg(feature = "global_pressure")]
        self.pressure_acc_f.push(Vector3::zeros());
        #[cfg(feature = "implicit_euler")]
        self.d_l.push(Vector3::zeros());
        #[cfg(feature = "implicit_euler")]
        self.r_l.push(Vector3::zeros());
        #[cfg(feature = "implicit_euler")]
        self.alpha_l.push(0.);
        #[cfg(feature = "implicit_euler")]
        self.a_times_d_l.push(Vector3::zeros());

        let insert_at = self.num_active;
        let last = self.position.len() - 1;

        if insert_at != last {
            self.swap(insert_at, last);
        }

        self.num_active += 1;
    }

    fn extend(&mut self, other: Self) {
        assert!(self.num_active == self.total_len());
        self.position.extend(other.position);
        self.position_prev.extend(other.position_prev);
        self.position_pred.extend(other.position_pred);
        self.velocity.extend(other.velocity);
        self.velocity_prev.extend(other.velocity_prev);
        self.velocity_pred.extend(other.velocity_pred);
        self.acceleration.extend(other.acceleration);
        self.mass.extend(other.mass);
        self.volume.extend(other.volume);
        self.pressure.extend(other.pressure);
        self.neighbors.extend(other.neighbors);
        self.boundary_neighbors.extend(other.boundary_neighbors);
        #[cfg(feature = "splitting")]
        self.density_pred.extend(other.density_pred);
        #[cfg(feature = "global_pressure")]
        self.s_f.extend(other.s_f);
        #[cfg(feature = "global_pressure")]
        self.a_ff.extend(other.a_ff);
        #[cfg(feature = "global_pressure")]
        self.pressure_acc_f.extend(other.pressure_acc_f);
        #[cfg(feature = "implicit_euler")]
        self.d_l.extend(other.d_l);
        #[cfg(feature = "implicit_euler")]
        self.r_l.extend(other.r_l);
        #[cfg(feature = "implicit_euler")]
        self.alpha_l.extend(other.alpha_l);
        #[cfg(feature = "implicit_euler")]
        self.a_times_d_l.extend(other.a_times_d_l);
    }
}

impl Positional for Fluid3D {
    fn pos_now<I>(&self, id: I) -> &I::Output
    where
        I: SliceIndex<[Vector3<f64>]>,
    {
        &self.position[id]
    }
}

impl Fluid3D {
    /// Gesamtzahl inkl. inaktiver
    pub fn total_len(&self) -> usize {
        self.position.len()
    }

    // pub fn rotate_position(&mut self, pos: &mut Vec<Vector3<f64>>) {
    //     std::mem::swap(&mut self.position, pos);
    // }
    // pub fn rotate_velocity(&mut self, vel: &mut Vec<Vector3<f64>>) {
    //     std::mem::swap(&mut self.velocity, vel);
    // }

    pub fn accept_pred_pos(&mut self) {
        std::mem::swap(&mut self.position_prev, &mut self.position);
        std::mem::swap(&mut self.position, &mut self.position_pred);
    }

    pub fn accept_pred_vel(&mut self) {
        std::mem::swap(&mut self.velocity_prev, &mut self.velocity);
        std::mem::swap(&mut self.velocity, &mut self.velocity_pred);
    }

    pub fn disable(&mut self, id: usize) {
        assert!(id < self.num_active);
        self.num_active -= 1;
        self.swap(id, self.num_active);
    }

    fn swap(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        self.position.swap(a, b);
        self.position_prev.swap(a, b);
        self.position_pred.swap(a, b);
        self.velocity.swap(a, b);
        self.velocity_prev.swap(a, b);
        self.velocity_pred.swap(a, b);
        self.acceleration.swap(a, b);
        self.mass.swap(a, b);
        self.volume.swap(a, b);
        self.pressure.swap(a, b);
        self.neighbors.swap(a, b);
        self.boundary_neighbors.swap(a, b);
        #[cfg(feature = "splitting")]
        self.pred_density.swap(a, b);
        #[cfg(feature = "global_pressure")]
        self.s_f.swap(a, b);
        #[cfg(feature = "global_pressure")]
        self.a_ff.swap(a, b);
        #[cfg(feature = "global_pressure")]
        self.pressure_acc_f.swap(a, b);
        #[cfg(feature = "implicit_euler")]
        self.d_l.swap(a, b);
        #[cfg(feature = "implicit_euler")]
        self.r_l.swap(a, b);
        #[cfg(feature = "implicit_euler")]
        self.alpha_l.swap(a, b);
        #[cfg(feature = "implicit_euler")]
        self.a_times_d_l.swap(a, b);
    }

    pub fn drop_inactive(&mut self) {
        self.position.truncate(self.num_active);
        self.position_prev.truncate(self.num_active);
        self.position_pred.truncate(self.num_active);
        self.velocity.truncate(self.num_active);
        self.velocity_prev.truncate(self.num_active);
        self.velocity_pred.truncate(self.num_active);
        self.acceleration.truncate(self.num_active);
        self.mass.truncate(self.num_active);
        self.volume.truncate(self.num_active);
        self.pressure.truncate(self.num_active);
        self.neighbors.truncate(self.num_active);
        self.boundary_neighbors.truncate(self.num_active);
        #[cfg(feature = "splitting")]
        self.pred_density.truncate(self.num_active);
        #[cfg(feature = "global_pressure")]
        self.s_f.truncate(self.num_active);
        #[cfg(feature = "global_pressure")]
        self.a_ff.truncate(self.num_active);
        #[cfg(feature = "global_pressure")]
        self.pressure_acc_f.truncate(self.num_active);
        #[cfg(feature = "implicit_euler")]
        self.d_l.truncate(self.num_active);
        #[cfg(feature = "implicit_euler")]
        self.r_l.truncate(self.num_active);
        #[cfg(feature = "implicit_euler")]
        self.alpha_l.truncate(self.num_active);
        #[cfg(feature = "implicit_euler")]
        self.a_times_d_l.truncate(self.num_active);
    }
}

impl From<SerFluid3D> for Fluid3D {
    fn from(ser_fluid: SerFluid3D) -> Self {
        let len = ser_fluid.position.len();
        Self {
            num_active: len,
            position: ser_fluid.position.iter().map(|pos| (*pos).into()).collect(),
            position_prev: vec![Vector3::zeros(); len],
            position_pred: vec![Vector3::zeros(); len],
            velocity: ser_fluid.velocity.iter().map(|vel| (*vel).into()).collect(),
            velocity_prev: vec![Vector3::zeros(); len],
            velocity_pred: vec![Vector3::zeros(); len],
            acceleration: vec![Vector3::zeros(); len],
            mass: vec![ser_fluid.mass; len],
            volume: vec![0.; len],
            pressure: vec![0.; len],
            neighbors: vec![Vec::new(); len],
            boundary_neighbors: vec![Vec::new(); len],
            #[cfg(feature = "splitting")]
            density_pred: vec![0.; len],
            #[cfg(feature = "global_pressure")]
            s_f: vec![0.; len],
            #[cfg(feature = "global_pressure")]
            a_ff: vec![0.; len],
            #[cfg(feature = "global_pressure")]
            pressure_acc_f: vec![Vector3::zeros(); len],
            #[cfg(feature = "implicit_euler")]
            d_l: vec![Vector3::zeros(); len],
            #[cfg(feature = "implicit_euler")]
            r_l: vec![Vector3::zeros(); len],
            #[cfg(feature = "implicit_euler")]
            alpha_l: vec![0.; len],
            #[cfg(feature = "implicit_euler")]
            a_times_d_l: vec![Vector3::zeros(); len],
        }
    }
}

/// Compressed and serializable fluid, i.e. a collection of
/// samples, in a 3-dimensional context.
#[derive(Debug, Clone, Default, Serialize, Deserialize, Encode, Decode)]
pub struct SerFluid3D {
    pub position: Vec<[f64; 3]>,
    pub velocity: Vec<[f64; 3]>,
    pub mass: f64,
}

impl From<Fluid3D> for SerFluid3D {
    fn from(fluid: Fluid3D) -> Self {
        Self {
            position: fluid.position.iter().map(|pos| (*pos).into()).collect(),
            velocity: fluid.velocity.iter().map(|vel| (*vel).into()).collect(),
            mass: fluid.mass[0],
        }
    }
}

impl SerFluid3D {
    pub fn vel_now(&self, id: usize) -> [f64; 3] {
        self.velocity[id]
    }
}

/// Boundary represented by samples, which are identified by an ID (usize)
#[derive(Debug, Clone, Default)]
pub struct Boundary3D {
    pub position: Vec<Vector3<f64>>,
    velocity: Vec<Vector3<f64>>,
    /// volume (necessary for sph fluid)
    volume: Vec<f64>,
}

impl Len for Boundary3D {
    fn len(&self) -> usize {
        self.position.len()
    }
}

impl Expandable for Boundary3D {
    fn push(&mut self, position: Vector3<f64>, velocity: Vector3<f64>, volume: f64) {
        self.position.push(position);
        self.velocity.push(velocity);
        self.volume.push(volume);
    }

    fn extend(&mut self, other: Self) {
        self.position.extend(other.position);
        self.velocity.extend(other.velocity);
        self.volume.extend(other.volume);
    }
}

impl Positional for Boundary3D {
    fn pos_now<I>(&self, id: I) -> &I::Output
    where
        I: SliceIndex<[Vector3<f64>]>,
    {
        &self.position[id]
    }
}

impl Boundary3D {
    pub fn vel_now<I>(&self, id: I) -> &I::Output
    where
        I: SliceIndex<[Vector3<f64>]>,
    {
        &self.velocity[id]
    }

    pub fn set_volume(&mut self, id: usize, volume: f64) {
        self.volume[id] = volume;
    }

    pub fn volume<I>(&self, id: I) -> &I::Output
    where
        I: SliceIndex<[f64]>,
    {
        &self.volume[id]
    }
}

impl From<SerBoundary3D> for Boundary3D {
    fn from(particle: SerBoundary3D) -> Self {
        let len = particle.position.len();
        Self {
            position: particle.position.iter().map(|pos| (*pos).into()).collect(),
            velocity: particle.velocity.iter().map(|vel| (*vel).into()).collect(),
            volume: vec![0.; len],
        }
    }
}

/// Compressed and serializable particle in a 3-dimensional context
#[derive(Debug, Clone, Default, Serialize, Deserialize, Encode, Decode)]
pub struct SerBoundary3D {
    pub position: Vec<[f64; 3]>,
    pub velocity: Vec<[f64; 3]>,
}

impl From<Boundary3D> for SerBoundary3D {
    fn from(particle: Boundary3D) -> Self {
        Self {
            position: particle.position.iter().map(|pos| (*pos).into()).collect(),
            velocity: particle.velocity.iter().map(|vel| (*vel).into()).collect(),
        }
    }
}

// impl Positional for SerBoundary3D {
//     fn pos_now<I>(&self, id: I) -> &I::Output
//     where
//         I: SliceIndex<[Vector3<f64>]>,
//     {
//         &self.position[id].into()
//     }
// }

// impl SerBoundary3D {
//     pub fn vel_now(&self, id: usize) -> [f64; 3] {
//         self.velocity[id]
//     }
// }
