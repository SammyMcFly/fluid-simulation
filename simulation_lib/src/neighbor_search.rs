/// Module provides the necessary entities for an efficient
/// neighbor search implementation
///
use nalgebra::Vector3;
use rustc_hash::FxHashMap; // Faster than: // use std::collections::HashMap;

use crate::sample::{Len, Positional};

type UniformGridCell = Vector3<i32>;

/// Calculate the distance between two 3D points
pub fn distance(from: &Vector3<f64>, to: &Vector3<f64>) -> f64 {
    (to - from).norm()
}



/// Container that stores indices of samples in a hash map depending
/// on their spatial position on the uniform grid.
#[derive(Debug, Clone)]
pub struct UniformGrid {
    // hash_map: HashMap<u64, Vec<usize>>,
    hash_map: FxHashMap<u64, Vec<usize>>,
    cell_size: f64,
}

impl UniformGrid {
    /// Initialize
    pub fn new(cell_size: f64) -> Self {
        // let hash_map: HashMap<u64, Vec<usize>> = HashMap::new();
        let hash_map: FxHashMap<u64, Vec<usize>> = FxHashMap::default();
        Self {
            hash_map,
            cell_size,
        }
    }

    /// Insert fluid samples from array into hash map
    pub fn populate(&mut self, fluid: &(impl Len + Positional)) {
        (0..fluid.len()).for_each(|id| {
            let cell = self.get_cell(fluid.pos_now(id));
            let cell_hash = Self::hash(cell.x, cell.y, cell.z);
            self.hash_map.entry(cell_hash).or_default().push(id);
        });
    }

    /// Insert boundary samples from array into hash map
    pub fn populate_boundary_particles(&mut self, boundary: &(impl Len + Positional)) {
        (0..boundary.len()).for_each(|id| {
            let cell = self.get_cell(boundary.pos_now(id));
            let cell_hash = Self::hash(cell.x, cell.y, cell.z);
            self.hash_map.entry(cell_hash).or_default().push(id);
        });
    }

    /// Given a sample's position, calculate its grid cell
    fn get_cell(&self, position: &Vector3<f64>) -> UniformGridCell {
        UniformGridCell::new(
            (position.x / self.cell_size).floor() as i32,
            (position.y / self.cell_size).floor() as i32,
            (position.z / self.cell_size).floor() as i32,
        )
    }

    /// Reversibly map signed integers to unsigned integers
    fn map_int_to_uint(k: i32) -> u64 {
        if k >= 0 {
            (k as u64) * 2
        } else {
            ((-k) as u64) * 2 + 1
        }
    }

    /// Hash function for hashing triples of integers (k,l,m),
    /// that define a position on the grid
    ///
    /// Handles negative and positive positions on grid
    fn hash(k: i32, l: i32, m: i32) -> u64 {
        const P1: u64 = 73856093;
        const P2: u64 = 19349663;
        const P3: u64 = 83492791;

        let unsigned_k = Self::map_int_to_uint(k);
        let unsigned_l = Self::map_int_to_uint(l);
        let unsigned_m = Self::map_int_to_uint(m);

        (unsigned_k * P1) ^ (unsigned_l * P2) ^ (unsigned_m * P3)
    }

    /// Get the indices of samples, which are within the twice the cell size of a
    /// specific position.
    pub fn get_particles_in_kernel_range(
        &self,
        position: &Vector3<f64>,
        other_positions: &[Vector3<f64>],
    ) -> Vec<usize> {
        let mut particles_in_kernel_range = Vec::new();
        let cell = self.get_cell(position);

        for dx in -2..=2 {
            for dy in -2..=2 {
                for dz in -2..=2 {
                    let neighbor_cell = (cell.x + dx, cell.y + dy, cell.z + dz);
                    let hash = Self::hash(neighbor_cell.0, neighbor_cell.1, neighbor_cell.2);
                    if let Some(neighbors) = self.hash_map.get(&hash) {
                        for &neighbor in neighbors {
                            // Distance check
                            if distance(&other_positions[neighbor], position) < 2. * self.cell_size
                            {
                                particles_in_kernel_range.push(neighbor);
                            }
                        }
                    }
                }
            }
        }
        particles_in_kernel_range
    }

    /// Clear hash map
    pub fn clear(&mut self) {
        self.hash_map.clear();
    }
}
