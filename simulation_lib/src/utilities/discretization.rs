/// Discretization helpers
use gauss_quad::GaussLegendre;
use nalgebra::{Point3, Vector3};
use std::collections::HashMap;
use std::f64::consts::PI;
use thiserror::Error as ThisError;

/// Integrates f(x, y, z) over a sphere with radius 'radius' using the Gauß-Legendre quadrature.
pub fn gauss_legendre_integrate<F>(f: F, radius: f64, n: usize) -> f64
where
    F: Fn(f64, f64, f64) -> f64,
{
    let quad = GaussLegendre::new(n.try_into().unwrap());

    quad.integrate(0.0, radius, |r| {
        r.powi(2)
            * quad.integrate(0.0, PI, |theta| {
                theta.sin()
                    * quad.integrate(0.0, 2. * PI, |phi| {
                        let x = r * theta.sin() * phi.cos();
                        let y = r * theta.sin() * phi.sin();
                        let z = r * theta.cos();
                        f(x, y, z)
                    })
            })
    })
}

/// A cubic serendipity discretization of a 3D scalar function within a predefined grid domain.
///
/// Provides discretized function and function gradient.
#[derive(Debug)]
pub struct CubicSerendipityDiscretization {
    x_min: Point3<f64>,
    dx: f64,
    n: [usize; 3], // cells per axis
    ref_nodes: Vec<Point3<f64>>,
    offsets: Vec<[usize; 3]>, // per-node lattice offset within a cell
    values: HashMap<[usize; 3], f64>, // global nodal values, keyed by lattice index
}

#[derive(Debug, ThisError)]
#[error("Point is out of bounds.")]
pub struct OutOfBoundsError;

impl CubicSerendipityDiscretization {
    /// Build the discretization: sample `f` once at every (shared) node.
    pub fn new<F: Fn(&Point3<f64>) -> f64>(
        x_min: Point3<f64>,
        x_max: Point3<f64>,
        dx: f64,
        f: &F,
    ) -> Self {
        let n = [
            ((x_max[0] - x_min[0]) / dx).round() as usize,
            ((x_max[1] - x_min[1]) / dx).round() as usize,
            ((x_max[2] - x_min[2]) / dx).round() as usize,
        ];
        let ref_nodes = Self::reference_nodes();
        let offsets: Vec<[usize; 3]> = ref_nodes
            .iter()
            .map(|&p_ref| {
                [
                    Self::to_offset(p_ref.x),
                    Self::to_offset(p_ref.y),
                    Self::to_offset(p_ref.z),
                ]
            })
            .collect();

        let mut values = HashMap::new();
        // fine-lattice physical position: x_min + (lattice/3)*dx
        for cz in 0..n[2] {
            for cy in 0..n[1] {
                for cx in 0..n[0] {
                    let base = [3 * cx, 3 * cy, 3 * cz];
                    for off in &offsets {
                        let key = [base[0] + off[0], base[1] + off[1], base[2] + off[2]];
                        values.entry(key).or_insert_with(|| {
                            let p = Point3::new(
                                x_min[0] + key[0] as f64 / 3.0 * dx,
                                x_min[1] + key[1] as f64 / 3.0 * dx,
                                x_min[2] + key[2] as f64 / 3.0 * dx,
                            );
                            f(&p)
                        });
                    }
                }
            }
        }

        Self {
            x_min,
            dx,
            n,
            ref_nodes,
            offsets,
            values,
        }
    }

    /// Reference coord (in {-1,-1/3,1/3,1}) -> lattice offset (in {0,1,2,3}).
    fn to_offset(r: f64) -> usize {
        (((r + 1.0) * 1.5).round()) as usize
    }

    fn reference_nodes() -> Vec<Point3<f64>> {
        let mut nodes = Vec::with_capacity(32);
        let signs = [-1.0, 1.0];
        for &z in &signs {
            for &y in &signs {
                for &x in &signs {
                    nodes.push(Point3::new(x, y, z));
                }
            }
        }
        let thirds = [-1. / 3., 1. / 3.];
        for &z in &signs {
            for &y in &signs {
                for &t in &thirds {
                    nodes.push(Point3::new(t, y, z));
                }
            }
        }
        for &z in &signs {
            for &x in &signs {
                for &t in &thirds {
                    nodes.push(Point3::new(x, t, z));
                }
            }
        }
        for &y in &signs {
            for &x in &signs {
                for &t in &thirds {
                    nodes.push(Point3::new(x, y, t));
                }
            }
        }
        nodes
    }

    fn is_corner(c: f64) -> bool {
        (c.abs() - 1.0).abs() < 1e-9
    }

    fn shape_functions(nodes: &[Point3<f64>], xi: f64, eta: f64, zeta: f64) -> Vec<f64> {
        let r2 = xi * xi + eta * eta + zeta * zeta;
        nodes
            .iter()
            .map(|&p_ref| {
                match (
                    Self::is_corner(p_ref.x),
                    Self::is_corner(p_ref.y),
                    Self::is_corner(p_ref.z),
                ) {
                    (true, true, true) => {
                        (1.0 / 64.0)
                            * (1.0 + xi * p_ref.x)
                            * (1.0 + eta * p_ref.y)
                            * (1.0 + zeta * p_ref.z)
                            * (9.0 * r2 - 19.0)
                    }
                    (false, true, true) => {
                        (9.0 / 64.0)
                            * (1.0 - xi * xi)
                            * (1.0 + 9.0 * xi * p_ref.x)
                            * (1.0 + eta * p_ref.y)
                            * (1.0 + zeta * p_ref.z)
                    }
                    (true, false, true) => {
                        (9.0 / 64.0)
                            * (1.0 - eta * eta)
                            * (1.0 + 9.0 * eta * p_ref.y)
                            * (1.0 + xi * p_ref.x)
                            * (1.0 + zeta * p_ref.z)
                    }
                    (true, true, false) => {
                        (9.0 / 64.0)
                            * (1.0 - zeta * zeta)
                            * (1.0 + 9.0 * zeta * p_ref.z)
                            * (1.0 + xi * p_ref.x)
                            * (1.0 + eta * p_ref.y)
                    }
                    _ => 0.0,
                }
            })
            .collect()
    }

    /// Computes the gradient of the shape functions in regards to the reference coordinates xi, eta, and zeta.
    fn shape_function_gradients(
        nodes: &[Point3<f64>],
        xi: f64,
        eta: f64,
        zeta: f64,
    ) -> Vec<Vector3<f64>> {
        let r2 = xi * xi + eta * eta + zeta * zeta;
        nodes
            .iter()
            .map(|&p_ref| {
                match (
                    Self::is_corner(p_ref.x),
                    Self::is_corner(p_ref.y),
                    Self::is_corner(p_ref.z),
                ) {
                    (true, true, true) => Vector3::new(
                        (1.0 / 64.0)
                            * (1.0 + eta * p_ref.y)
                            * (1.0 + zeta * p_ref.z)
                            * (p_ref.x * (9.0 * r2 - 19.0) + (1.0 + xi * p_ref.x) * 18.0 * xi),
                        (1.0 / 64.0)
                            * (1.0 + xi * p_ref.x)
                            * (1.0 + zeta * p_ref.z)
                            * (p_ref.y * (9.0 * r2 - 19.0) + (1.0 + eta * p_ref.y) * 18.0 * eta),
                        (1.0 / 64.0)
                            * (1.0 + xi * p_ref.x)
                            * (1.0 + eta * p_ref.y)
                            * (p_ref.z * (9.0 * r2 - 19.0) + (1.0 + zeta * p_ref.z) * 18.0 * zeta),
                    ),
                    (false, true, true) => Vector3::new(
                        (9.0 / 64.0)
                            * (1.0 + eta * p_ref.y)
                            * (1.0 + zeta * p_ref.z)
                            * (2. * xi * (1.0 + 9.0 * xi * p_ref.x)
                                + (1.0 - xi * xi) * 9.0 * p_ref.x),
                        (9.0 / 64.0)
                            * (1.0 - xi * xi)
                            * (1.0 + 9.0 * xi * p_ref.x)
                            * p_ref.y
                            * (1.0 + zeta * p_ref.z),
                        (9.0 / 64.0)
                            * (1.0 - xi * xi)
                            * (1.0 + 9.0 * xi * p_ref.x)
                            * (1.0 + eta * p_ref.y)
                            * p_ref.z,
                    ),
                    (true, false, true) => Vector3::new(
                        (9.0 / 64.0)
                            * (1.0 - eta * eta)
                            * (1.0 + 9.0 * eta * p_ref.y)
                            * p_ref.x
                            * (1.0 + zeta * p_ref.z),
                        (9.0 / 64.0)
                            * (1.0 + xi * p_ref.x)
                            * (1.0 + zeta * p_ref.z)
                            * (2.0 * eta * (1.0 + 9.0 * eta * p_ref.y)
                                + (1.0 - eta * eta) * 9.0 * p_ref.y),
                        (9.0 / 64.0)
                            * (1.0 - eta * eta)
                            * (1.0 + 9.0 * eta * p_ref.y)
                            * (1.0 + xi * p_ref.x)
                            * p_ref.z,
                    ),
                    (true, true, false) => Vector3::new(
                        (9.0 / 64.0)
                            * (1.0 - zeta * zeta)
                            * (1.0 + 9.0 * zeta * p_ref.z)
                            * p_ref.x
                            * (1.0 + eta * p_ref.y),
                        (9.0 / 64.0)
                            * (1.0 - zeta * zeta)
                            * (1.0 + 9.0 * zeta * p_ref.z)
                            * (1.0 + xi * p_ref.x)
                            * p_ref.y,
                        (9.0 / 64.0)
                            * (1.0 + xi * p_ref.x)
                            * (1.0 + eta * p_ref.y)
                            * (2.0 * zeta * (1.0 + 9.0 * zeta * p_ref.z)
                                + (1.0 - zeta * zeta) * 9.0 * p_ref.z),
                    ),
                    _ => Vector3::new(0.0, 0.0, 0.0),
                }
            })
            .collect()
    }

    fn get_cube_idx(&self, p: &Point3<f64>) -> Result<[usize; 3], OutOfBoundsError> {
        let mut c = [0usize; 3];
        for d in 0..3 {
            let idx = ((p[d] - self.x_min[d]) / self.dx).floor() as isize;
            if idx < 0 || idx >= self.n[d] as isize {
                return Err(OutOfBoundsError {});
            }
            c[d] = idx as usize;
        }
        Ok(c)
    }

    /// Evaluate the interpolant anywhere in the grid.
    pub fn function(&self, p: &Point3<f64>) -> Result<f64, OutOfBoundsError> {
        let c = self.get_cube_idx(p)?;
        let o = [
            self.x_min[0] + c[0] as f64 * self.dx,
            self.x_min[1] + c[1] as f64 * self.dx,
            self.x_min[2] + c[2] as f64 * self.dx,
        ];
        let xi = 2.0 * (p[0] - o[0]) / self.dx - 1.0;
        let eta = 2.0 * (p[1] - o[1]) / self.dx - 1.0;
        let zeta = 2.0 * (p[2] - o[2]) / self.dx - 1.0;

        let base = [3 * c[0], 3 * c[1], 3 * c[2]];
        let shp = Self::shape_functions(&self.ref_nodes, xi, eta, zeta);

        Ok(self
            .offsets
            .iter()
            .zip(&shp)
            .map(|(off, &ni)| {
                let key = [base[0] + off[0], base[1] + off[1], base[2] + off[2]];
                self.values[&key] * ni
            })
            .sum())
    }

    /// Evaluate the interpolant anywhere in the grid.
    pub fn gradient(&self, p: &Point3<f64>) -> Result<Vector3<f64>, OutOfBoundsError> {
        let c = self.get_cube_idx(p)?;
        let o = [
            self.x_min[0] + c[0] as f64 * self.dx,
            self.x_min[1] + c[1] as f64 * self.dx,
            self.x_min[2] + c[2] as f64 * self.dx,
        ];
        let xi = 2.0 * (p[0] - o[0]) / self.dx - 1.0;
        let eta = 2.0 * (p[1] - o[1]) / self.dx - 1.0;
        let zeta = 2.0 * (p[2] - o[2]) / self.dx - 1.0;

        let base = [3 * c[0], 3 * c[1], 3 * c[2]];
        let shp = Self::shape_function_gradients(&self.ref_nodes, xi, eta, zeta);

        Ok(self
            .offsets
            .iter()
            .zip(&shp)
            .map(|(off, &ni)| {
                let key = [base[0] + off[0], base[1] + off[1], base[2] + off[2]];
                2. / self.dx * self.values[&key] * ni
            })
            .sum())
    }
}
