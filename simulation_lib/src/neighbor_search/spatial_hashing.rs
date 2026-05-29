/// Spatial hashing neighbor search algorithm
use nalgebra::Vector3;
use rustc_hash::FxHashMap; // Faster than: // use std::collections::HashMap;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::for_each;
use crate::neighbor_search::{NeighborSearch, distance};

type UniformGridCell = Vector3<i32>;

/// Spatial hashing neighbor search structure.
///
/// Indices of samples are stored in a hash map depending
/// on their spatial position on the uniform grid. The lookup
/// of the samples in a certain cell or neighboring cells is
/// efficient and because of this it accelerates neighbor search.
/// This strategy is called spacial hashing.
#[derive(Debug, Clone)]
pub struct SpatialHashing {
    fluid_cells: FxHashMap<u64, Vec<usize>>,
    boundary_cells: FxHashMap<u64, Vec<usize>>,
    cell_size: f64,
}

impl SpatialHashing {
    /// Initialize spatial hashing with given cell size.
    ///
    /// # Performance guidance
    ///
    /// `cell_size` should be close to the search range (`within_range`) passed to
    /// `find_neighbors`. Specifically:
    ///
    /// - **`cell_size ≈ range`** → searches a 3×3×3 = 27 cell neighborhood (optimal)
    /// - **`cell_size ≈ range / 2`** → searches a 5×5×5 = 125 cell neighborhood (fewer false positives, more overhead)
    /// - **`cell_size >> range`** → few cells searched, but each cell contains many
    ///   particles that fail the distance check
    ///
    /// A good default is `cell_size = within_range`
    pub fn new(cell_size: f64) -> Self {
        Self {
            fluid_cells: FxHashMap::default(),
            boundary_cells: FxHashMap::default(),
            cell_size,
        }
    }

    /// Create a spatial hashing structure with optimal cell size for the given range.
    pub fn from_range(within_range: f64) -> Self {
        Self::new(within_range/2.)
    }

    /// Insert samples from array into target hash map
    fn populate(
        target: &mut FxHashMap<u64, Vec<usize>>,
        fluid_positions: &[Vector3<f64>],
        cell_size: f64,
    ) {
        (0..fluid_positions.len()).for_each(|id| {
            let cell = Self::get_cell(&fluid_positions[id], cell_size);
            let cell_hash = Self::hash(cell.x, cell.y, cell.z);
            target.entry(cell_hash).or_default().push(id);
        });
    }

    /// Given a sample's position, calculate its grid cell
    fn get_cell(position: &Vector3<f64>, cell_size: f64) -> UniformGridCell {
        UniformGridCell::new(
            (position.x / cell_size).floor() as i32,
            (position.y / cell_size).floor() as i32,
            (position.z / cell_size).floor() as i32,
        )
    }

    /// Reversibly map signed integers to unsigned integers
    fn map_int_to_uint(k: i32) -> u64 {
        if k >= 0 {
            (k as u64) * 2
        } else {
            ((k as i64).unsigned_abs()) * 2 + 1
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
    fn get_particles_in_range(
        source_cells: &FxHashMap<u64, Vec<usize>>,
        position: &Vector3<f64>,
        other_positions: &[Vector3<f64>],
        cell_size: f64,
        range: f64,
    ) -> Vec<usize> {
        let mut particles_in_kernel_range = Vec::new();
        let cell = Self::get_cell(position, cell_size);
        let multiples_of_cell_size = (range / cell_size).ceil() as i32;

        for dx in -multiples_of_cell_size..=multiples_of_cell_size {
            for dy in -multiples_of_cell_size..=multiples_of_cell_size {
                for dz in -multiples_of_cell_size..=multiples_of_cell_size {
                    let neighbor_cell = (cell.x + dx, cell.y + dy, cell.z + dz);
                    let hash = Self::hash(neighbor_cell.0, neighbor_cell.1, neighbor_cell.2);
                    if let Some(neighbors) = source_cells.get(&hash) {
                        for &neighbor in neighbors {
                            // Distance check
                            if distance(&other_positions[neighbor], position) < range
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
}

impl NeighborSearch for SpatialHashing {
    /// Perform neighbor search for all fluid particles
    ///
    /// Adds fluid neighbors and boundary neighbors as neighbors
    fn find_neighbors(
        &mut self,
        within_range: f64,
        fluid_positions: &[Vector3<f64>],
        boundary_positions: &[Vector3<f64>],
        fluid_neighbors: &mut super::NeighborList,
        boundary_neighbors: &mut super::NeighborList,
    ) {
        fluid_neighbors.resize(fluid_positions.len());
        boundary_neighbors.resize(fluid_positions.len());
        fluid_neighbors.clear();
        boundary_neighbors.clear();

        // Build grid from fluid positions
        self.fluid_cells.clear();
        Self::populate(&mut self.fluid_cells, fluid_positions, self.cell_size);

        // Build grid from boundary positions
        self.boundary_cells.clear();
        Self::populate(&mut self.boundary_cells, boundary_positions, self.cell_size);

        for_each!(
            mut [fluid_neighbors.neighbors_mut(), boundary_neighbors.neighbors_mut()],
            ref [fluid_pos = fluid_positions, boundary_pos = boundary_positions],
            |id, id_neighbors, id_boundary_neighbors| {
                // update neighbors
                let neighbors = Self::get_particles_in_range(
                    &self.fluid_cells,
                    &fluid_pos[id],
                    fluid_pos,
                    self.cell_size,
                    within_range,
                );
                *id_neighbors = neighbors;
                // update boundary neighbors
                let boundary_neighbors = Self::get_particles_in_range(
                    &self.boundary_cells,
                    &fluid_pos[id],
                    boundary_pos,
                    self.cell_size,
                    within_range,
                );
                *id_boundary_neighbors = boundary_neighbors;
            }
        );
        fluid_neighbors.flatten();
        boundary_neighbors.flatten();
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::neighbor_search::NeighborList;

    // ─── Helper functions ───────────────────────────────────────────────

    fn pos(x: f64, y: f64, z: f64) -> Vector3<f64> {
        Vector3::new(x, y, z)
    }

    /// Collect neighbors of particle `id` as a sorted Vec for order-independent comparison
    fn sorted_neighbors(nl: &NeighborList, id: usize) -> Vec<usize> {
        let mut v = nl.get_neighbors(id).to_vec();
        v.sort();
        v
    }

    // ─── get_cell tests ─────────────────────────────────────────────────

    #[test]
    fn get_cell_origin() {
        let cell = SpatialHashing::get_cell(&pos(0.5, 0.5, 0.5), 1.0);
        assert_eq!(cell, Vector3::new(0, 0, 0));
    }

    #[test]
    fn get_cell_positive() {
        let cell = SpatialHashing::get_cell(&pos(2.3, 4.7, 1.1), 1.0);
        assert_eq!(cell, Vector3::new(2, 4, 1));
    }

    #[test]
    fn get_cell_negative() {
        let cell = SpatialHashing::get_cell(&pos(-0.5, -1.5, -3.9), 1.0);
        assert_eq!(cell, Vector3::new(-1, -2, -4));
    }

    #[test]
    fn get_cell_with_different_cell_size() {
        let cell = SpatialHashing::get_cell(&pos(1.5, 3.0, 0.9), 2.0);
        assert_eq!(cell, Vector3::new(0, 1, 0));
    }

    #[test]
    fn get_cell_on_boundary() {
        // Exactly on cell boundary → floor puts it in the lower cell
        let cell = SpatialHashing::get_cell(&pos(1.0, 2.0, 3.0), 1.0);
        assert_eq!(cell, Vector3::new(1, 2, 3));
    }

    // ─── map_int_to_uint tests ──────────────────────────────────────────

    #[test]
    fn map_int_to_uint_positive() {
        assert_eq!(SpatialHashing::map_int_to_uint(0), 0);
        assert_eq!(SpatialHashing::map_int_to_uint(1), 2);
        assert_eq!(SpatialHashing::map_int_to_uint(5), 10);
    }

    #[test]
    fn map_int_to_uint_negative() {
        assert_eq!(SpatialHashing::map_int_to_uint(-1), 3);
        assert_eq!(SpatialHashing::map_int_to_uint(-5), 11);
    }

    #[test]
    fn map_int_to_uint_is_injective() {
        // No two different inputs should map to the same output
        let values: Vec<u64> = (-100..=100)
            .map(SpatialHashing::map_int_to_uint)
            .collect();
        let mut unique = values.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(values.len(), unique.len());
    }

    // ─── hash tests ─────────────────────────────────────────────────────

    #[test]
    fn hash_same_input_same_output() {
        let h1 = SpatialHashing::hash(1, 2, 3);
        let h2 = SpatialHashing::hash(1, 2, 3);
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_different_input_likely_different_output() {
        let h1 = SpatialHashing::hash(0, 0, 0);
        let h2 = SpatialHashing::hash(1, 0, 0);
        let h3 = SpatialHashing::hash(0, 1, 0);
        let h4 = SpatialHashing::hash(0, 0, 1);
        // While collisions are possible, adjacent cells should not collide
        assert_ne!(h1, h2);
        assert_ne!(h1, h3);
        assert_ne!(h1, h4);
    }

    #[test]
    fn hash_handles_negatives() {
        // Should not panic
        let _ = SpatialHashing::hash(-10, -20, -30);
        let _ = SpatialHashing::hash(-1, 0, 1);
    }

    // ─── populate tests ─────────────────────────────────────────────────

    #[test]
    fn populate_empty_positions() {
        let mut map = FxHashMap::default();
        SpatialHashing::populate(&mut map, &[], 1.0);
        assert!(map.is_empty());
    }

    #[test]
    fn populate_single_particle() {
        let mut map = FxHashMap::default();
        let positions = vec![pos(0.5, 0.5, 0.5)];
        SpatialHashing::populate(&mut map, &positions, 1.0);

        let hash = SpatialHashing::hash(0, 0, 0);
        assert_eq!(map.get(&hash).unwrap(), &vec![0]);
    }

    #[test]
    fn populate_two_particles_same_cell() {
        let mut map = FxHashMap::default();
        let positions = vec![pos(0.1, 0.1, 0.1), pos(0.9, 0.9, 0.9)];
        SpatialHashing::populate(&mut map, &positions, 1.0);

        let hash = SpatialHashing::hash(0, 0, 0);
        let cell_contents = map.get(&hash).unwrap();
        assert!(cell_contents.contains(&0));
        assert!(cell_contents.contains(&1));
    }

    #[test]
    fn populate_two_particles_different_cells() {
        let mut map = FxHashMap::default();
        let positions = vec![pos(0.5, 0.5, 0.5), pos(5.5, 5.5, 5.5)];
        SpatialHashing::populate(&mut map, &positions, 1.0);

        let hash0 = SpatialHashing::hash(0, 0, 0);
        let hash5 = SpatialHashing::hash(5, 5, 5);
        assert_eq!(map.get(&hash0).unwrap(), &vec![0]);
        assert_eq!(map.get(&hash5).unwrap(), &vec![1]);
    }

    // ─── get_particles_in_range tests ───────────────────────────────────

    #[test]
    fn particles_in_range_finds_close_neighbor() {
        let mut map = FxHashMap::default();
        let positions = vec![pos(0.0, 0.0, 0.0), pos(0.5, 0.0, 0.0)];
        SpatialHashing::populate(&mut map, &positions, 1.0);

        let result = SpatialHashing::get_particles_in_range(
            &map, &positions[0], &positions, 1.0, 1.0,
        );
        assert!(result.contains(&0)); // self
        assert!(result.contains(&1)); // close neighbor
    }

    #[test]
    fn particles_in_range_excludes_far_neighbor() {
        let mut map = FxHashMap::default();
        let positions = vec![pos(0.0, 0.0, 0.0), pos(10.0, 0.0, 0.0)];
        SpatialHashing::populate(&mut map, &positions, 1.0);

        let result = SpatialHashing::get_particles_in_range(
            &map, &positions[0], &positions, 1.0, 2.0,
        );
        assert!(result.contains(&0));  // self
        assert!(!result.contains(&1)); // too far
    }

    #[test]
    fn particles_in_range_excludes_at_exact_range() {
        let mut map = FxHashMap::default();
        // distance = exactly 2.0, range < 2.0 uses strict less-than
        let positions = vec![pos(0.0, 0.0, 0.0), pos(2.0, 0.0, 0.0)];
        SpatialHashing::populate(&mut map, &positions, 1.0);

        let result = SpatialHashing::get_particles_in_range(
            &map, &positions[0], &positions, 1.0, 2.0,
        );
        assert!(!result.contains(&1)); // exactly at range → excluded (strict <)
    }

    #[test]
    fn particles_in_range_negative_positions() {
        let mut map = FxHashMap::default();
        let positions = vec![pos(-1.0, -1.0, -1.0), pos(-1.5, -1.0, -1.0)];
        SpatialHashing::populate(&mut map, &positions, 1.0);

        let result = SpatialHashing::get_particles_in_range(
            &map, &positions[0], &positions, 1.0, 1.0,
        );
        assert!(result.contains(&1));
    }
}