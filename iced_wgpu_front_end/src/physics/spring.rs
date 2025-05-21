use nalgebra::Matrix3;



/// Entity representing a Spring
#[derive(Debug, Clone)]
pub struct Spring {
    /// Particle indices which are connected by the spring
    pub indices: (usize, usize),
    /// Spring force coeff
    pub k: f64,
    /// Rest length
    pub l: f64,
    /// Implicit euler variable: S
    pub matrix_s: Matrix3<f64>,
}

impl Spring {
    /// Construct an instance of [[Spring]]
    pub fn new(indices: (usize, usize), k: f64, l: f64,) -> Self {
        Self {
            indices,
            k,
            l,
            matrix_s: Matrix3::default(),
        }
    }
}
