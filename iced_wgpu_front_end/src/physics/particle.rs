// use tracing::{debug, error, info, span, trace, warn};
use nalgebra::Vector3;



#[derive(Debug, Clone, Copy)]
struct Cycle3(usize); // Cycle<static u8>

impl Cycle3 {
    fn new(val:usize) -> Self {
        Cycle3(val)
    }
}

impl std::ops::Add<u8> for Cycle3 {
    type Output = usize;

    fn add(self, rhs: u8) -> Self::Output {
        assert!(rhs==1);
        if self.0 == 2 {
            0
        } else {
            self.0 + 1
        }
    }
}

impl std::ops::AddAssign<u8> for Cycle3 {
    fn add_assign(&mut self, rhs: u8) {
        assert!(rhs==1);
        if self.0 == 2 {
            self.0 = 0;
        } else {
            self.0 += 1;
        }
    }
}

impl std::ops::Sub<u8> for Cycle3 {
    type Output = usize;

    fn sub(self, rhs: u8) -> Self::Output {
        assert!(rhs==1);
        if self.0 == 0 {
            2
        } else {
            self.0 - 1
        }
    }
}

impl std::ops::SubAssign<u8> for Cycle3 {
    fn sub_assign(&mut self, rhs: u8) {
        assert!(rhs==1);
        if self.0 == 0 {
            self.0 = 2;
        } else {
            self.0 -= 1;
        }
    }
}

// Cant accomodate Runge-Kutta 4th (Q4) order or Adams-Bashforth/Adams-Moulton (Q5/Q7)
#[derive(Debug, Clone, Copy)]
pub struct Q3<T>
where
    T: Default + Copy
{
    data: [T; 3],
    now: Cycle3,
}

impl<T> Q3<T>
where
    T: Default + Copy
{
    pub fn new(val_now: T, val_prev: T) -> Self {
        Self {
            data: [val_prev, val_now, T::default()],
            now: Cycle3::new(1),
        }
    }
    pub fn now(&self) -> T {
        self.data[self.now.0]
    }
    pub fn prev(&self) -> T  {
        self.data[self.now-1]
    }
    pub fn pred(&self) -> T  {
        self.data[self.now+1]
    }
    pub fn update_now(&mut self, val: T) {
        self.now += 1;
        self.data[self.now.0] = val;
    }
    pub fn set_pred(&mut self, val: T) {
        self.data[self.now+1] = val;
    }
    pub fn accept_pred(&mut self) {
        self.now += 1;
    }
}

pub trait ParticleQ3 {
    type Pos;
    type Vel;
    type Acc;

    type PosBuffer;
    type VelBuffer;
    type AccBuffer;

    type Color;

    fn pos(&self) -> Self::PosBuffer;
    fn vel(&self) -> Self::VelBuffer;
    fn acc(&self) -> Self::AccBuffer;

    fn set_new_pos(&mut self, pos: Self::Pos);
    fn set_new_vel(&mut self, vel: Self::Vel);
    fn set_acc(&mut self, acc: Self::Acc);
    fn add_acc(&mut self, acc: Self::Acc);

    fn set_pred_pos(&mut self, pos: Self::Pos);
    fn set_pred_vel(&mut self, vel: Self::Vel);
    fn accept_pred_pos(&mut self);
    fn accept_pred_vel(&mut self);

    fn mass(&self) -> f64;
    fn density(&self) -> f64;
    fn set_density(&mut self, density: f64);
    fn add_density(&mut self, density: f64);
    fn pressure(&self) -> f64;
    fn set_pressure(&mut self, pressure: f64);
    fn color(&self) -> Self::Color;
}

pub trait GridParticle {
    fn get_distance(&self, other: &Vector3<f64>) -> f64;
}

// # 3D Implementation

/// Particle in a 3-dimensional context
#[derive(Debug, Clone)]
pub struct Particle3D {
    position: Q3<Vector3<f64>>,
    velocity: Q3<Vector3<f64>>,
    acceleration: Vector3<f64>,
    mass: f64,
    /// density (necessary for sph fluid)
    density: f64,
    pressure: f64,
    /// RGB-color values between 0.0 and 1.0
    custom_color: bool,
    color: [f32; 3],
    /// neighbors
    pub neighbors: Vec<usize>,
    /// boundary neighbors
    pub boundary_neighbors: Vec<usize>,
    /// implicit euler variables
    pub d_l: Vector3<f64>,
    pub r_l: Vector3<f64>,
    pub alpha_l: f64,
    pub a_times_d_l: Vector3<f64>,
}

impl ParticleQ3 for Particle3D {
    type Pos = Vector3<f64>;
    type Vel = Vector3<f64>;
    type Acc = Vector3<f64>;

    type PosBuffer = Q3<Vector3<f64>>;
    type VelBuffer = Q3<Vector3<f64>>;
    type AccBuffer = Vector3<f64>;

    type Color = [f32; 3];

    fn pos(&self) -> Self::PosBuffer {
        self.position
    }
    fn vel(&self) -> Self::VelBuffer {
        self.velocity
    }
    fn acc(&self) -> Self::AccBuffer {
        self.acceleration
    }

    fn set_new_pos(&mut self, pos: Self::Pos) {
        self.position.update_now(pos);
    }
    fn set_new_vel(&mut self, vel: Self::Vel) {
        self.velocity.update_now(vel);
    }
    fn set_acc(&mut self, acc: Self::Acc) {
        self.acceleration = acc;
    }
    fn add_acc(&mut self, acc: Self::Acc) {
        self.acceleration += acc;
    }

    fn set_pred_pos(&mut self, pos: Self::Pos) {
        self.position.set_pred(pos);
    }
    fn set_pred_vel(&mut self, vel: Self::Vel) {
        self.velocity.set_pred(vel);
    }
    fn accept_pred_pos(&mut self) {
        self.position.accept_pred();
    }
    fn accept_pred_vel(&mut self) {
        self.velocity.accept_pred();
    }

    fn mass(&self) -> f64 {
        self.mass
    }
    fn density(&self) -> f64 {
        self.density
    }
    fn set_density(&mut self, density: f64) {
        self.density = density;
    }
    fn add_density(&mut self, density: f64) {
        self.density += density;
    }
    fn pressure(&self) -> f64 {
        self.pressure
    }
    fn set_pressure(&mut self, pressure: f64) {
        self.pressure = pressure;
    }
    fn color(&self) -> [f32; 3] {
        self.color
    }
}

impl GridParticle for Particle3D {
    fn get_distance(&self, other: &Vector3<f64>) -> f64 {
        ((self.position.now().x-other.x).powi(2)
            +(self.position.now().y-other.y).powi(2)
            +(self.position.now().z-other.z).powi(2)).sqrt()
    }
}

impl Particle3D {
    pub fn new(
        position: [Vector3<f64>; 2],
        velocity: Vector3<f64>,
        mass: f64,
        color: Option<[f32; 3]>,
    ) -> Self {
        let (custom_color, color) = if let Some(color) = color {
            (true, color)
        } else {
            (false, [0.0,0.0,1.0])
        };
        Self {
            position: Q3::new(position[0], position[1]),
            velocity: Q3::new(velocity, Vector3::default()),
            acceleration: Vector3::default(),
            mass,
            density: f64::default(),
            pressure: f64::default(),
            custom_color,
            color,
            neighbors: vec![],
            boundary_neighbors: vec![],
            d_l: Vector3::default(),
            r_l: Vector3::default(),
            alpha_l: f64::default(),
            a_times_d_l: Vector3::default()
        }
    }

    pub fn update_color(&mut self) {
        if !self.custom_color {
            let whiteness = f64::min(self.vel().now().norm()/10., 1.);
            self.color = [ whiteness as f32, whiteness as f32, 1. ];
        }
    }

    pub fn set_neighbors(&mut self, neighbors: Vec<usize>) {
        self.neighbors = neighbors;
    }

    pub fn set_boundary_neighbors(&mut self, boundary_neighbors: Vec<usize>) {
        self.boundary_neighbors = boundary_neighbors;
    }

    /// Direction from particle1 towards particle2
    pub fn get_direction(&self, other: &Vector3<f64>) -> Vector3<f64> {
        self.position.now()-other
    }
}

/// Boundary particle in a 3-dimensional context
#[derive(Debug, Clone)]
pub struct BoundaryParticle3D {
    position: Vector3<f64>,
    // mass: f64,
    /// RGB-color values between 0.0 and 1.0
    color: [f32; 3],
}

impl GridParticle for BoundaryParticle3D {
    fn get_distance(&self, other: &Vector3<f64>) -> f64 {
        (self.position-other).norm()
    }
}

impl BoundaryParticle3D {
    pub fn new(
        position: Vector3<f64>,
        // mass: f64,
        color: [f32; 3],
    ) -> Self {
        Self {
            position,
            // mass,
            color,
        }
    }

    pub fn pos(&self) -> Vector3<f64> {
        self.position
    }

    // fn mass(&self) -> f64 {
    //     self.mass
    // }

    pub fn color(&self) -> [f32; 3] {
        self.color
    }
}

