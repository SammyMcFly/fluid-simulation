//! Discretization helpers
use gauss_quad::GaussLegendre;
use nalgebra::{Point3, Vector3};
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use thiserror::Error as ThisError;

#[derive(Debug, ThisError)]
pub enum EvaluationError {
    #[error("Point is out of bounds.")]
    OutOfBounds,
    #[error("Point lies in a pruned cell with no discretization values.")]
    PrunedCell,
}

/// Integrates f(x, y, z) over a ball with radius 'radius' using the Gauß-Legendre quadrature.
pub fn gauss_legendre_integrate<F>(
    f: &F,
    center: &Point3<f64>,
    radius: f64,
    order: usize,
) -> Result<f64, EvaluationError>
where
    F: Fn(&Point3<f64>) -> Result<f64, EvaluationError>,
{
    let quad = GaussLegendre::new(order.try_into().unwrap());
    let error: RefCell<Option<EvaluationError>> = RefCell::new(None);

    let result = quad.integrate(0.0, radius, |r| {
        if error.borrow().is_some() {
            return 0.0; // short-circuit remaining quadrature points
        }
        r.powi(2)
            * quad.integrate(0.0, PI, |theta| {
                if error.borrow().is_some() {
                    return 0.0; // short-circuit remaining quadrature points
                }
                theta.sin()
                    * quad.integrate(0.0, 2. * PI, |phi| {
                        if error.borrow().is_some() {
                            return 0.0; // short-circuit remaining quadrature points
                        }
                        let p = Point3::new(
                            center.x + r * theta.sin() * phi.cos(),
                            center.y + r * theta.sin() * phi.sin(),
                            center.z + r * theta.cos(),
                        );
                        match f(&p) {
                            Ok(v) => v,
                            Err(e) => {
                                *error.borrow_mut() = Some(e);
                                0.0
                            }
                        }
                    })
            })
    });

    match error.into_inner() {
        Some(e) => Err(e),
        None => Ok(result),
    }
}

/// A cubic serendipity discretization of a 3D scalar function within a predefined grid domain.
///
/// Provides discretized function and function gradient.
#[derive(Debug, Clone)]
pub struct CubicSerendipityDiscretization {
    x_min: Point3<f64>,
    x_max: Point3<f64>,
    dx: f64,
    n: [usize; 3], // cells per axis
    ref_nodes: Vec<Point3<f64>>,
    offsets: Vec<[usize; 3]>, // per-node lattice offset within a cell
    values: HashMap<[usize; 3], f64>, // global nodal values, keyed by lattice index
}

impl CubicSerendipityDiscretization {
    /// Build the discretization: sample `f` once at every (shared) node.
    ///
    /// The function f is only discretized in space where lower_bound < f(x) < upper_bound.
    pub fn new<F: Fn(&Point3<f64>) -> Result<f64, EvaluationError> + Sync>(
        x_min: Point3<f64>,
        x_max: Point3<f64>,
        f_value_prune_lower_bound: Option<f64>,
        f_value_prune_upper_bound: Option<f64>,
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

        // Collect all unique node keys across all cells
        let offsets_ref = &offsets;
        let all_keys: HashSet<[usize; 3]> = (0..n[2])
            .flat_map(|cz| {
                (0..n[1]).flat_map(move |cy| {
                    (0..n[0]).flat_map(move |cx| {
                        let base = [3 * cx, 3 * cy, 3 * cz];
                        offsets_ref
                            .iter()
                            .map(move |off| [base[0] + off[0], base[1] + off[1], base[2] + off[2]])
                    })
                })
            })
            .collect();

        // Compute all node values in parallel; None signals an EvaluationError
        #[cfg(not(feature = "parallel"))]
        let all_keys_iter = all_keys.iter();
        #[cfg(feature = "parallel")]
        let all_keys_iter = all_keys.par_iter();
        let all_values: HashMap<[usize; 3], Option<f64>> = all_keys_iter
            .map(|&key| {
                let p = Point3::new(
                    x_min[0] + key[0] as f64 / 3.0 * dx,
                    x_min[1] + key[1] as f64 / 3.0 * dx,
                    x_min[2] + key[2] as f64 / 3.0 * dx,
                );
                (key, f(&p).ok())
            })
            .collect();

        // Apply pruning sequentially
        let mut values = HashMap::new();
        for cz in 0..n[2] {
            for cy in 0..n[1] {
                for cx in 0..n[0] {
                    let base = [3 * cx, 3 * cy, 3 * cz];

                    let node_values: Vec<f64> = offsets
                        .iter()
                        .filter_map(|off| {
                            *all_values
                                .get(&[base[0] + off[0], base[1] + off[1], base[2] + off[2]])
                                .unwrap()
                        })
                        .collect();

                    // Prune if any node had an EvaluationError
                    if node_values.len() != offsets.len() {
                        continue;
                    }

                    let prune_low = f_value_prune_lower_bound
                        .is_some_and(|t| node_values.iter().all(|&v| v < t));
                    let prune_high = f_value_prune_upper_bound
                        .is_some_and(|t| node_values.iter().all(|&v| v > t));

                    if !prune_low && !prune_high {
                        for (off, val) in offsets.iter().zip(node_values.iter()) {
                            let key = [base[0] + off[0], base[1] + off[1], base[2] + off[2]];
                            values.entry(key).or_insert(*val);
                        }
                    }
                }
            }
        }

        Self {
            x_min,
            x_max,
            dx,
            n,
            ref_nodes,
            offsets,
            values,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Anzahl Zellen pro Achse (nur für Diagnosemeldungen).
    pub(crate) fn cell_count(&self) -> [usize; 3] {
        self.n
    }

    /// Abstand zwischen benachbarten Interpolationsknoten (dx / 3) —
    /// die tatsächliche Auflösung des Gitters (nur für Diagnosemeldungen).
    pub(crate) fn node_spacing(&self) -> f64 {
        self.dx / 3.0
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
                            * (-2. * xi * (1.0 + 9.0 * xi * p_ref.x)
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
                            * (-2.0 * eta * (1.0 + 9.0 * eta * p_ref.y)
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
                            * (-2.0 * zeta * (1.0 + 9.0 * zeta * p_ref.z)
                                + (1.0 - zeta * zeta) * 9.0 * p_ref.z),
                    ),
                    _ => Vector3::new(0.0, 0.0, 0.0),
                }
            })
            .collect()
    }

    fn get_cube_idx(&self, p: &Point3<f64>) -> Result<[usize; 3], EvaluationError> {
        let mut c = [0usize; 3];
        for d in 0..3 {
            let idx = ((p[d] - self.x_min[d]) / self.dx).floor() as isize;
            if p[d] < self.x_min[d] || p[d] > self.x_max[d] {
                return Err(EvaluationError::OutOfBounds {});
            }
            // point with p[d] == self.x_max[d] is included by clamping it to last valid cell
            c[d] = (idx.min(self.n[d] as isize - 1)) as usize;
        }
        Ok(c)
    }

    /// Evaluate the interpolant anywhere in the grid.
    pub fn function(&self, p: &Point3<f64>) -> Result<f64, EvaluationError> {
        let c = self.get_cube_idx(p)?;
        let base = [3 * c[0], 3 * c[1], 3 * c[2]];

        if self.offsets.iter().any(|off| {
            !self
                .values
                .contains_key(&[base[0] + off[0], base[1] + off[1], base[2] + off[2]])
        }) {
            return Err(EvaluationError::PrunedCell);
        }

        let o = [
            self.x_min[0] + c[0] as f64 * self.dx,
            self.x_min[1] + c[1] as f64 * self.dx,
            self.x_min[2] + c[2] as f64 * self.dx,
        ];
        let xi = 2.0 * (p[0] - o[0]) / self.dx - 1.0;
        let eta = 2.0 * (p[1] - o[1]) / self.dx - 1.0;
        let zeta = 2.0 * (p[2] - o[2]) / self.dx - 1.0;

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

    // /// Evaluate the interpolant anywhere in the grid.
    // pub fn gradient(&self, p: &Point3<f64>) -> Result<Vector3<f64>, EvaluationError> {
    //     let c = self.get_cube_idx(p)?;
    //     let base = [3 * c[0], 3 * c[1], 3 * c[2]];

    //     if self.offsets.iter().any(|off| {
    //         !self
    //             .values
    //             .contains_key(&[base[0] + off[0], base[1] + off[1], base[2] + off[2]])
    //     }) {
    //         return Err(EvaluationError::PrunedCell);
    //     }

    //     let o = [
    //         self.x_min[0] + c[0] as f64 * self.dx,
    //         self.x_min[1] + c[1] as f64 * self.dx,
    //         self.x_min[2] + c[2] as f64 * self.dx,
    //     ];
    //     let xi = 2.0 * (p[0] - o[0]) / self.dx - 1.0;
    //     let eta = 2.0 * (p[1] - o[1]) / self.dx - 1.0;
    //     let zeta = 2.0 * (p[2] - o[2]) / self.dx - 1.0;

    //     let shp = Self::shape_function_gradients(&self.ref_nodes, xi, eta, zeta);

    //     Ok(self
    //         .offsets
    //         .iter()
    //         .zip(&shp)
    //         .map(|(off, &ni)| {
    //             let key = [base[0] + off[0], base[1] + off[1], base[2] + off[2]];
    //             2. / self.dx * self.values[&key] * ni
    //         })
    //         .sum())
    // }

    /// Numerical derivative along one axis via central difference.
    ///
    /// Both `p + h` and `p - h` must be evaluable; if either evaluation fails
    /// (out of bounds, or in a pruned cell), that error is propagated as-is.
    /// There is currently no fallback to a one-sided difference near domain or
    /// pruned-cell boundaries — see the corresponding test in this module for
    /// the concrete failure mode this causes in `find_boundary_samples`.
    fn directional_derivative(
        &self,
        p: &Point3<f64>,
        axis: usize,
        h: f64,
    ) -> Result<f64, EvaluationError> {
        let mut p_plus = *p;
        let mut p_minus = *p;
        p_plus[axis] += h;
        p_minus[axis] -= h;

        match (self.function(&p_plus), self.function(&p_minus)) {
            (Ok(f_plus), Ok(f_minus)) => Ok((f_plus - f_minus) / (2.0 * h)),
            (Ok(_), Err(e)) => Err(e),
            (Err(e), Ok(_)) => Err(e),
            (Err(e), Err(_)) => Err(e),
        }
    }

    /// Evaluate the gradient anywhere in the grid using central differences.
    pub fn gradient(&self, p: &Point3<f64>) -> Result<Vector3<f64>, EvaluationError> {
        // Step size: half the internal node spacing (dx / 3) is a natural choice
        // since that's the resolution of the underlying discretization.
        let h = self.dx / 6.0;

        Ok(Vector3::new(
            self.directional_derivative(p, 0, h)?,
            self.directional_derivative(p, 1, h)?,
            self.directional_derivative(p, 2, h)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── to_offset / is_corner ──────────────────────────────────────

    #[test]
    fn to_offset_maps_reference_coordinates_to_lattice_offsets() {
        assert_eq!(CubicSerendipityDiscretization::to_offset(-1.0), 0);
        assert_eq!(CubicSerendipityDiscretization::to_offset(-1.0 / 3.0), 1);
        assert_eq!(CubicSerendipityDiscretization::to_offset(1.0 / 3.0), 2);
        assert_eq!(CubicSerendipityDiscretization::to_offset(1.0), 3);
    }

    #[test]
    fn is_corner_identifies_extreme_reference_coordinates() {
        assert!(CubicSerendipityDiscretization::is_corner(1.0));
        assert!(CubicSerendipityDiscretization::is_corner(-1.0));
        assert!(!CubicSerendipityDiscretization::is_corner(1.0 / 3.0));
        assert!(!CubicSerendipityDiscretization::is_corner(-1.0 / 3.0));
        assert!(!CubicSerendipityDiscretization::is_corner(0.0));
    }

    #[test]
    fn is_corner_tolerates_floating_point_noise() {
        assert!(CubicSerendipityDiscretization::is_corner(1.0 + 1e-12));
        assert!(CubicSerendipityDiscretization::is_corner(-1.0 - 1e-12));
    }

    // ─── reference_nodes ─────────────────────────────────────────────

    #[test]
    fn reference_nodes_has_32_unique_nodes() {
        let nodes = CubicSerendipityDiscretization::reference_nodes();
        assert_eq!(nodes.len(), 32);
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                assert!(
                    (nodes[i] - nodes[j]).norm() > 1e-9,
                    "nodes {i} and {j} coincide: {:?} == {:?}",
                    nodes[i],
                    nodes[j]
                );
            }
        }
    }

    #[test]
    fn reference_nodes_coordinates_are_all_valid_lattice_values() {
        let valid = [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0];
        for node in CubicSerendipityDiscretization::reference_nodes() {
            for c in [node.x, node.y, node.z] {
                assert!(
                    valid.iter().any(|&v| (v - c).abs() < 1e-9),
                    "coordinate {c} is not one of {valid:?}"
                );
            }
        }
    }

    #[test]
    fn reference_nodes_are_either_corners_or_single_edge_nodes() {
        // Serendipity elements omit face/interior nodes: every node is
        // either a full corner (3 corner coordinates) or an edge-midpoint
        // node (exactly 2 corner coordinates, 1 interior).
        for node in CubicSerendipityDiscretization::reference_nodes() {
            let corner_count = [node.x, node.y, node.z]
                .iter()
                .filter(|&&c| CubicSerendipityDiscretization::is_corner(c))
                .count();
            assert!(
                corner_count == 2 || corner_count == 3,
                "node {node:?} has {corner_count} corner coordinates, expected 2 or 3"
            );
        }
    }

    // ─── shape_functions ─────────────────────────────────────────────

    #[test]
    fn shape_functions_satisfy_kronecker_delta_at_reference_nodes() {
        // Fundamental nodal-interpolation property: N_i(node_j) = delta_ij.
        // Without this, `function()`'s weighted sum wouldn't even reproduce
        // the sampled node values at their own locations. If this fails, it
        // is a genuine correctness bug in `shape_functions`, not a flaw in
        // this test.
        let nodes = CubicSerendipityDiscretization::reference_nodes();
        for (j, node_j) in nodes.iter().enumerate() {
            let shp = CubicSerendipityDiscretization::shape_functions(
                &nodes, node_j.x, node_j.y, node_j.z,
            );
            assert_eq!(shp.len(), 32);
            for (i, &value) in shp.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (value - expected).abs() < 1e-9,
                    "N_{i}(node_{j}) = {value}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn shape_functions_sum_to_one_partition_of_unity() {
        let nodes = CubicSerendipityDiscretization::reference_nodes();
        for &(xi, eta, zeta) in &[
            (0.0, 0.0, 0.0),
            (0.3, -0.2, 0.5),
            (-0.9, 0.9, -0.1),
            (0.99, -0.99, 0.0),
        ] {
            let shp = CubicSerendipityDiscretization::shape_functions(&nodes, xi, eta, zeta);
            let sum: f64 = shp.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-9,
                "sum of shape functions at ({xi}, {eta}, {zeta}) = {sum}, expected 1.0"
            );
        }
    }

    // ─── shape_function_gradients (currently dead code) ─────────────

    #[test]
    fn shape_function_gradients_match_finite_difference_of_shape_functions() {
        // `shape_function_gradients` is currently dead code (only
        // referenced from the commented-out analytical `gradient` method).
        // This documents that its formulas are at least internally
        // consistent with `shape_functions`, in case it's ever revived.
        let nodes = CubicSerendipityDiscretization::reference_nodes();
        let h = 1e-6;

        for &(xi, eta, zeta) in &[(0.1, 0.2, -0.3), (-0.5, 0.4, 0.6), (0.0, 0.0, 0.0)] {
            let analytical =
                CubicSerendipityDiscretization::shape_function_gradients(&nodes, xi, eta, zeta);

            let d_dxi: Vec<f64> = {
                let plus =
                    CubicSerendipityDiscretization::shape_functions(&nodes, xi + h, eta, zeta);
                let minus =
                    CubicSerendipityDiscretization::shape_functions(&nodes, xi - h, eta, zeta);
                plus.iter()
                    .zip(&minus)
                    .map(|(p, m)| (p - m) / (2.0 * h))
                    .collect()
            };
            let d_deta: Vec<f64> = {
                let plus =
                    CubicSerendipityDiscretization::shape_functions(&nodes, xi, eta + h, zeta);
                let minus =
                    CubicSerendipityDiscretization::shape_functions(&nodes, xi, eta - h, zeta);
                plus.iter()
                    .zip(&minus)
                    .map(|(p, m)| (p - m) / (2.0 * h))
                    .collect()
            };
            let d_dzeta: Vec<f64> = {
                let plus =
                    CubicSerendipityDiscretization::shape_functions(&nodes, xi, eta, zeta + h);
                let minus =
                    CubicSerendipityDiscretization::shape_functions(&nodes, xi, eta, zeta - h);
                plus.iter()
                    .zip(&minus)
                    .map(|(p, m)| (p - m) / (2.0 * h))
                    .collect()
            };

            for i in 0..32 {
                let numerical = Vector3::new(d_dxi[i], d_deta[i], d_dzeta[i]);
                assert!(
                    (analytical[i] - numerical).norm() < 1e-4,
                    "node {i} at ({xi},{eta},{zeta}): analytical {:?} vs numerical {numerical:?}",
                    analytical[i]
                );
            }
        }
    }

    // ─── get_cube_idx ────────────────────────────────────────────────

    #[test]
    fn get_cube_idx_locates_interior_point() {
        let disc = CubicSerendipityDiscretization::new(
            Point3::new(0., 0., 0.),
            Point3::new(2., 1., 1.),
            None,
            None,
            1.0,
            &|_p: &Point3<f64>| Ok(0.0),
        );
        assert_eq!(
            disc.get_cube_idx(&Point3::new(0.5, 0.5, 0.5)).unwrap(),
            [0, 0, 0]
        );
        assert_eq!(
            disc.get_cube_idx(&Point3::new(1.5, 0.5, 0.5)).unwrap(),
            [1, 0, 0]
        );
    }

    #[test]
    fn get_cube_idx_clamps_exact_upper_boundary_to_last_cell() {
        let disc = CubicSerendipityDiscretization::new(
            Point3::new(0., 0., 0.),
            Point3::new(2., 1., 1.),
            None,
            None,
            1.0,
            &|_p: &Point3<f64>| Ok(0.0),
        );
        assert_eq!(
            disc.get_cube_idx(&Point3::new(2.0, 1.0, 1.0)).unwrap(),
            [1, 0, 0]
        );
    }

    #[test]
    fn get_cube_idx_rejects_points_outside_domain() {
        let disc = CubicSerendipityDiscretization::new(
            Point3::new(0., 0., 0.),
            Point3::new(2., 1., 1.),
            None,
            None,
            1.0,
            &|_p: &Point3<f64>| Ok(0.0),
        );
        assert!(matches!(
            disc.get_cube_idx(&Point3::new(-0.01, 0.5, 0.5)),
            Err(EvaluationError::OutOfBounds)
        ));
        assert!(matches!(
            disc.get_cube_idx(&Point3::new(2.01, 0.5, 0.5)),
            Err(EvaluationError::OutOfBounds)
        ));
    }

    // ─── new(): pruning behavior & shared-node retention ────────────

    #[test]
    fn pruned_cells_still_contribute_shared_boundary_nodes_from_unpruned_neighbors() {
        // Two cells along x: [0,1] and [1,2]. `f` is below the prune
        // threshold for x <= 1.0 (all of cell 0's nodes, INCLUDING the
        // shared face at x=1) and above it for x > 1.0 (some of cell 1's
        // nodes) — so cell 0 is pruned, cell 1 is not, and the physically
        // shared node at x=1 must survive via cell 1's contribution even
        // though cell 0 (which also "owns" that node) was skipped entirely.
        let disc = CubicSerendipityDiscretization::new(
            Point3::new(0., 0., 0.),
            Point3::new(2., 1., 1.),
            Some(5.0),
            None,
            1.0,
            &|p: &Point3<f64>| Ok(if p.x <= 1.0 { 0.0 } else { 10.0 }),
        );

        // Corner node at lattice [3,0,0] == physical (1,0,0): shared by
        // both cells, contributed by unpruned cell 1.
        assert!(disc.values.contains_key(&[3, 0, 0]));
        // Corner node at lattice [0,0,0] == physical (0,0,0): exclusive to
        // pruned cell 0, never inserted.
        assert!(!disc.values.contains_key(&[0, 0, 0]));

        assert!(matches!(
            disc.function(&Point3::new(0.5, 0.5, 0.5)),
            Err(EvaluationError::PrunedCell)
        ));
    }

    #[test]
    fn function_prunes_cell_if_any_node_evaluation_errors() {
        // A single-cell domain where the sampling function itself errors
        // for one specific node — `new` must treat that cell as
        // unbuildable (pruned), rather than propagating the error out of
        // `new` (which has no `Result` return type) or panicking.
        let disc = CubicSerendipityDiscretization::new(
            Point3::new(0., 0., 0.),
            Point3::new(1., 1., 1.),
            None,
            None,
            1.0,
            &|p: &Point3<f64>| {
                if (p.x - 1.0).abs() < 1e-9 {
                    Err(EvaluationError::OutOfBounds)
                } else {
                    Ok(0.0)
                }
            },
        );

        assert!(matches!(
            disc.function(&Point3::new(0.5, 0.5, 0.5)),
            Err(EvaluationError::PrunedCell)
        ));
    }

    // ─── directional_derivative ──────────────────────────────────────

    #[test]
    fn directional_derivative_computes_central_difference_in_interior() {
        let disc = CubicSerendipityDiscretization::new(
            Point3::new(-1., -1., -1.),
            Point3::new(1., 1., 1.),
            None,
            None,
            1.0,
            &|p: &Point3<f64>| Ok(3.0 * p.x),
        );

        let d = disc
            .directional_derivative(&Point3::new(0., 0., 0.), 0, 0.1)
            .unwrap();
        assert!((d - 3.0).abs() < 1e-2);
    }

    #[test]
    fn directional_derivative_propagates_error_without_one_sided_fallback() {
        // `directional_derivative` is a pure central-difference approximation:
        // if evaluating `function` at EITHER `p + h` or `p - h` fails (out of
        // bounds, or in a pruned cell), that error is propagated as-is — there
        // is no fallback to a one-sided difference near domain or pruned-cell
        // boundaries.
        //
        // Consequence: `gradient()` (and therefore `signed_distance_gradient`/
        // `volume_map_value` in `volume_map_boundary`) can fail purely because
        // a point lies close to a domain/pruned-cell boundary, even where
        // `function` itself would evaluate successfully at that exact point.
        // This is a real limitation of the current gradient evaluation, not
        // just a theoretical edge case — see `find_boundary_samples`, which
        // silently skips any fluid particle whose gradient evaluation fails
        // this way.
        //
        // This test pins down the current (fallback-free) behavior; if a
        // one-sided fallback is added later, update this test to expect
        // `Ok(..)`.
        let disc = CubicSerendipityDiscretization::new(
            Point3::new(-1., -1., -1.),
            Point3::new(1., 1., 1.),
            None,
            None,
            2.0,
            &|_p: &Point3<f64>| Ok(0.0),
        );

        // p + h = 0.95 + 0.1 = 1.05, exceeds x_max = 1.0.
        let result = disc.directional_derivative(&Point3::new(0.95, 0., 0.), 0, 0.1);
        assert!(matches!(result, Err(EvaluationError::OutOfBounds)));
    }
}
