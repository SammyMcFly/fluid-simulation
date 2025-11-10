use nalgebra::Vector3;
use serde::{Serialize, Deserialize};

// #[cfg(feature = "logging")]
// use tracing::{debug, error, info, span, trace, warn};


/// Struct that represents the natural Numbers modulo 3
#[derive(Debug, Clone, Copy)]
struct Cycle3(usize); // Cycle<static u8>

impl Cycle3 {
    fn new(val:usize) -> Self {
        Cycle3(val)
    }
}

impl std::ops::Add<u8> for Cycle3 {
    type Output = usize;

    /// Add one to Cycle3. Right hand side must be 1.
    fn add(self, _: u8) -> Self::Output {
        // assert!(rhs==1);
        if self.0 == 2 {
            0
        } else {
            self.0 + 1
        }
    }
}

impl std::ops::AddAssign<u8> for Cycle3 {
    /// Add one to Cycle3. Right hand side must be 1.
    fn add_assign(&mut self, _: u8) {
        // assert!(rhs==1);
        if self.0 == 2 {
            self.0 = 0;
        } else {
            self.0 += 1;
        }
    }
}

impl std::ops::Sub<u8> for Cycle3 {
    type Output = usize;

    /// Subtract one to Cycle3. Right hand side must be 1.
    fn sub(self, _: u8) -> Self::Output {
        // assert!(rhs==1);
        if self.0 == 0 {
            2
        } else {
            self.0 - 1
        }
    }
}

impl std::ops::SubAssign<u8> for Cycle3 {
    /// Subtract one to Cycle3. Right hand side must be 1.
    fn sub_assign(&mut self, _: u8) {
        // assert!(rhs==1);
        if self.0 == 0 {
            self.0 = 2;
        } else {
            self.0 -= 1;
        }
    }
}

// Cant accomodate Runge-Kutta 4th (Q4) order or Adams-Bashforth/Adams-Moulton (needed Q5/Q7)
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

pub trait Positional {
    fn pos_now(&self) -> Vector3<f64>;
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
}

pub trait Initializable {
    fn new(
        position: [Vector3<f64>; 2],
        velocity: Vector3<f64>,
        mass: f64,
    ) -> Self;
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
    disabled: bool,
    /// neighbors
    pub neighbors: Vec<usize>,
    /// boundary neighbors
    pub boundary_neighbors: Vec<usize>,
    // local pressure with splitting variable
    #[cfg(feature = "splitting")]
    pub pred_density: f64,
    // global pressure solver variables
    #[cfg(feature = "global_pressure")]
    pub s_f: f64,
    #[cfg(feature = "global_pressure")]
    pub a_ff: f64,
    #[cfg(feature = "global_pressure")]
    pub pressure_acc_f: Vector3<f64>,
    // implicit euler variables
    #[cfg(feature = "implicit_euler")]
    pub d_l: Vector3<f64>,
    #[cfg(feature = "implicit_euler")]
    pub r_l: Vector3<f64>,
    #[cfg(feature = "implicit_euler")]
    pub alpha_l: f64,
    #[cfg(feature = "implicit_euler")]
    pub a_times_d_l: Vector3<f64>,
}

impl Positional for Particle3D {
    fn pos_now(&self) -> Vector3<f64> {
        self.position.now()
    }
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
}

impl Initializable for Particle3D {
    fn new(
            position: [Vector3<f64>; 2],
            velocity: Vector3<f64>,
            mass: f64,
        ) -> Self {
        Self {
            position: Q3::new(position[0], position[1]),
            velocity: Q3::new(velocity, Vector3::default()),
            acceleration: Vector3::default(),
            mass,
            density: f64::default(),
            pressure: f64::default(),
            disabled: false,
            neighbors: vec![],
            boundary_neighbors: vec![],
            #[cfg(feature = "splitting")]
            pred_density: f64::default(),
            #[cfg(feature = "global_pressure")]
            s_f: f64::default(),
            #[cfg(feature = "global_pressure")]
            a_ff: f64::default(),
            #[cfg(feature = "global_pressure")]
            pressure_acc_f: Vector3::default(),
            #[cfg(feature = "implicit_euler")]
            d_l: Vector3::default(),
            #[cfg(feature = "implicit_euler")]
            r_l: Vector3::default(),
            #[cfg(feature = "implicit_euler")]
            alpha_l: f64::default(),
            #[cfg(feature = "implicit_euler")]
            a_times_d_l: Vector3::default()
        }
    }
}

impl Particle3D {
    pub fn set_neighbors(&mut self, neighbors: Vec<usize>) {
        self.neighbors = neighbors;
    }

    pub fn set_boundary_neighbors(&mut self, boundary_neighbors: Vec<usize>) {
        self.boundary_neighbors = boundary_neighbors;
    }

    pub fn is_enabled(&self) -> bool {
        !self.disabled
    }

    pub fn disable(&mut self) {
        self.disabled = true;
    }

    pub fn set_mass(&mut self, mass: f64) {
        self.mass = mass;
    }
}

/// Boundary particle in a 3-dimensional context
#[derive(Debug, Clone)]
pub struct BoundaryParticle3D {
    position: Vector3<f64>,
    mass: f64,
    #[cfg(feature = "global_pressure")]
    velocity: Vector3<f64>,
}

impl Positional for BoundaryParticle3D {
    fn pos_now(&self) -> Vector3<f64> {
        self.position
    }
}

impl Initializable for BoundaryParticle3D {
    fn new(
            position: [Vector3<f64>; 2],
            _: Vector3<f64>,
            mass: f64,
        ) -> Self {
        Self {
            position: position[0],
            mass,
            #[cfg(feature = "global_pressure")]
            velocity: Vector3::zeros(),
        }
    }
}

impl BoundaryParticle3D {
    pub fn pos(&self) -> Vector3<f64> {
        self.position
    }

    pub fn set_mass(&mut self, mass: f64) {
        self.mass = mass;
    }

    pub fn mass(&self) -> f64 {
        self.mass
    }

    #[cfg(feature = "global_pressure")]
    pub fn vel(&self) -> Vector3<f64> {
        self.velocity
    }
}

impl From<SerParticle3D> for Particle3D {
    fn from(particle: SerParticle3D) -> Self {
        Self {
            position: Q3::new(particle.position[0].into(), particle.position[2].into()),
            velocity: Q3::new(particle.velocity[0].into(), particle.velocity[2].into()),
            acceleration: particle.acceleration.into(),
            mass: particle.mass,
            density: particle.density,
            pressure: particle.pressure,
            // custom_color: particle.custom_color,
            // color: particle.color,
            disabled: particle.disabled,
            neighbors: particle.neighbors,
            boundary_neighbors: particle.boundary_neighbors,
            #[cfg(feature = "splitting")]
            pred_density: f64::default(),
            #[cfg(feature = "global_pressure")]
            s_f: f64::default(),
            #[cfg(feature = "global_pressure")]
            a_ff: f64::default(),
            #[cfg(feature = "global_pressure")]
            pressure_acc_f: Vector3::default(),
            #[cfg(feature = "implicit_euler")]
            d_l: Vector3::default(),
            #[cfg(feature = "implicit_euler")]
            r_l: Vector3::default(),
            #[cfg(feature = "implicit_euler")]
            alpha_l: f64::default(),
            #[cfg(feature = "implicit_euler")]
            a_times_d_l: Vector3::default()
        }
    }
}

/// Serializable particle in a 3-dimensional context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerParticle3D {
    position: [[f64; 3]; 3],
    velocity: [[f64; 3]; 3],
    acceleration: [f64; 3],
    mass: f64,
    /// density (necessary for sph fluid)
    density: f64,
    pressure: f64,
    disabled: bool,
    /// neighbors
    pub neighbors: Vec<usize>,
    /// boundary neighbors
    pub boundary_neighbors: Vec<usize>,
}

impl From<Particle3D> for SerParticle3D {
    fn from(particle: Particle3D) -> Self {
        Self {
            position: [particle.position.now().into(), particle.position.pred().into(), particle.position.prev().into()],
            velocity: [particle.velocity.now().into(), particle.velocity.pred().into(), particle.velocity.prev().into()],
            acceleration: particle.acceleration.into(),
            mass: particle.mass,
            density: particle.density,
            pressure: particle.pressure,
            disabled: particle.disabled,
            neighbors: particle.neighbors,
            boundary_neighbors: particle.boundary_neighbors,
        }
    }
}

