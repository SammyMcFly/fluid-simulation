/// Module provides the necessary entities for an efficient
/// neighbor search implementation
use nalgebra::Point3;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::Deserialize;

use crate::for_each;
pub mod spatial_hashing;

pub use spatial_hashing::SpatialHashing;


#[derive(Debug, Deserialize)]
pub enum NeighborSearchVariant {
    SpatialHashing,
}

pub trait NeighborSearch: Send + Sync {
    /// Initialize spatial hashing with given cell size.
    fn new(within_range: f64) -> Self;
    fn find_samples(
        &mut self,
        within_range: f64,
        positions: &[Point3<f64>],
        sample_positions: &[Point3<f64>],
        neighbor_list: &mut NeighborList,
    );
}

/// Calculate the distance between two 3D points
pub fn distance(from: &Point3<f64>, to: &Point3<f64>) -> f64 {
    (to - from).norm()
}

/// Struct that stores neighbors of samples in a flat array
#[derive(Debug, Clone, Default)]
pub struct NeighborList {
    /// Flat neighbor list
    data: Vec<usize>,
    /// Index list to point to start of the neighbor list of each sample
    offsets: Vec<usize>,
    /// Unflattened neighbor list which is necessary for parallelization
    unflattened_data: Vec<Vec<usize>>,
}

impl NeighborList {
    pub fn new(len: usize) -> Self {
        Self {
            data: vec![usize::default(); len],
            offsets: vec![usize::default(); len+1],
            unflattened_data: vec![Vec::new(); len],
        }
    }

    pub fn resize(&mut self, len: usize) {
        self.data.resize(len, usize::default());
        self.offsets.resize(len+1, usize::default());
        self.unflattened_data.resize(len, Vec::new());
    }

    pub fn clear(&mut self) {
        self.data.clear();
        self.offsets.clear();
        for_each!(
            mut [self.unflattened_data],
            ref [],
            |_id, id_neighbors| {
                id_neighbors.clear();
            }
        );
    }

    /// Get mutable reference to unflattened neighbor list: one Vec<usize> per sample
    ///
    /// Contract: Always call flatten after updating data in unflattened array
    pub fn neighbors_mut(&mut self) -> &mut[Vec<usize>] {
        &mut self.unflattened_data
    }

    /// Flatten neighbor list
    fn flatten(&mut self) {
        self.data.clear();
        self.offsets.clear();

        let total_neighbors: usize = self.unflattened_data.iter().map(|v| v.len()).sum();
        let num_particles = self.unflattened_data.len();

        self.data.reserve(total_neighbors);
        self.offsets.reserve(num_particles + 1);

        self.offsets.push(0);
        for nbrs in &self.unflattened_data {
            self.data.extend_from_slice(nbrs);
            self.offsets.push(self.data.len());
        }
    }

    /// Get indices of neighbor of sample with identifier 'id'
    pub fn get_neighbors(&self, id: usize) -> &[usize] {
        &self.data[self.offsets[id]..self.offsets[id+1]]
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_empty_neighbor_list() {
        let nl = NeighborList::new(5);
        assert_eq!(nl.unflattened_data.len(), 5);
        assert!(nl.unflattened_data.iter().all(|v| v.is_empty()));
    }

    #[test]
    fn flatten_empty_list() {
        let mut nl = NeighborList::new(3);
        nl.flatten();
        assert_eq!(nl.offsets.len(), 4); // num_particles + 1
        assert_eq!(nl.data.len(), 0);
        for i in 0..3 {
            assert_eq!(nl.get_neighbors(i), &[]);
        }
    }

    #[test]
    fn flatten_single_particle_no_neighbors() {
        let mut nl = NeighborList::new(1);
        nl.flatten();
        assert_eq!(nl.get_neighbors(0), &[]);
    }

    #[test]
    fn flatten_single_particle_with_neighbors() {
        let mut nl = NeighborList::new(1);
        nl.neighbors_mut()[0] = vec![1, 2, 3];
        nl.flatten();
        assert_eq!(nl.get_neighbors(0), &[1, 2, 3]);
    }

    #[test]
    fn flatten_multiple_particles() {
        let mut nl = NeighborList::new(4);
        nl.neighbors_mut()[0] = vec![1, 2];
        nl.neighbors_mut()[1] = vec![0, 2, 3];
        nl.neighbors_mut()[2] = vec![];
        nl.neighbors_mut()[3] = vec![0];
        nl.flatten();

        assert_eq!(nl.get_neighbors(0), &[1, 2]);
        assert_eq!(nl.get_neighbors(1), &[0, 2, 3]);
        assert_eq!(nl.get_neighbors(2), &[]);
        assert_eq!(nl.get_neighbors(3), &[0]);
    }

    #[test]
    fn flatten_preserves_order() {
        let mut nl = NeighborList::new(2);
        nl.neighbors_mut()[0] = vec![5, 3, 7, 1];
        nl.neighbors_mut()[1] = vec![9, 0];
        nl.flatten();

        assert_eq!(nl.get_neighbors(0), &[5, 3, 7, 1]);
        assert_eq!(nl.get_neighbors(1), &[9, 0]);
    }

    #[test]
    fn clear_resets_all_data() {
        let mut nl = NeighborList::new(3);
        nl.neighbors_mut()[0] = vec![1, 2];
        nl.neighbors_mut()[1] = vec![0];
        nl.neighbors_mut()[2] = vec![0, 1];
        nl.flatten();

        nl.clear();

        assert!(nl.data.is_empty());
        assert!(nl.offsets.is_empty());
        assert!(nl.unflattened_data.iter().all(|v| v.is_empty()));
    }

    #[test]
    fn resize_grows() {
        let mut nl = NeighborList::new(2);
        nl.neighbors_mut()[0] = vec![1];
        nl.neighbors_mut()[1] = vec![0];

        nl.resize(5);

        assert_eq!(nl.unflattened_data.len(), 5);
        // Existing data preserved
        assert_eq!(nl.unflattened_data[0], vec![1]);
        assert_eq!(nl.unflattened_data[1], vec![0]);
        // New entries are empty
        assert!(nl.unflattened_data[2].is_empty());
        assert!(nl.unflattened_data[3].is_empty());
        assert!(nl.unflattened_data[4].is_empty());
    }

    #[test]
    fn resize_shrinks() {
        let mut nl = NeighborList::new(5);
        nl.neighbors_mut()[0] = vec![1, 2];
        nl.neighbors_mut()[4] = vec![0];

        nl.resize(2);

        assert_eq!(nl.unflattened_data.len(), 2);
        assert_eq!(nl.unflattened_data[0], vec![1, 2]);
    }

    #[test]
    fn flatten_called_twice() {
        let mut nl = NeighborList::new(2);
        nl.neighbors_mut()[0] = vec![1];
        nl.neighbors_mut()[1] = vec![0];
        nl.flatten();

        // Modify and flatten again
        nl.neighbors_mut()[0] = vec![1, 3, 5];
        nl.neighbors_mut()[1] = vec![];
        nl.flatten();

        assert_eq!(nl.get_neighbors(0), &[1, 3, 5]);
        assert_eq!(nl.get_neighbors(1), &[]);
    }

    #[test]
    fn full_workflow_clear_repopulate_flatten() {
        let mut nl = NeighborList::new(3);

        // First iteration
        nl.neighbors_mut()[0] = vec![1, 2];
        nl.neighbors_mut()[1] = vec![0, 2];
        nl.neighbors_mut()[2] = vec![0, 1];
        nl.flatten();
        assert_eq!(nl.get_neighbors(0), &[1, 2]);
        assert_eq!(nl.get_neighbors(1), &[0, 2]);
        assert_eq!(nl.get_neighbors(2), &[0, 1]);

        // Second iteration (simulates next timestep)
        nl.clear();
        nl.neighbors_mut()[0] = vec![2];
        nl.neighbors_mut()[1] = vec![];
        nl.neighbors_mut()[2] = vec![0, 1];
        nl.flatten();
        assert_eq!(nl.get_neighbors(0), &[2]);
        assert_eq!(nl.get_neighbors(1), &[]);
        assert_eq!(nl.get_neighbors(2), &[0, 1]);
    }

    #[test]
    fn get_data_length_consistency() {
        let mut nl = NeighborList::new(4);
        nl.neighbors_mut()[0] = vec![1, 2, 3];
        nl.neighbors_mut()[1] = vec![0];
        nl.neighbors_mut()[2] = vec![0, 1, 3, 4];
        nl.neighbors_mut()[3] = vec![];
        nl.flatten();

        let total: usize = (0..4).map(|i| nl.get_neighbors(i).len()).sum();
        assert_eq!(total, nl.data.len());
        assert_eq!(nl.offsets.len(), 5); // num_particles + 1
    }

    #[test]
    fn large_neighbor_list() {
        let n = 1000;
        let mut nl = NeighborList::new(n);
        for i in 0..n {
            // Each particle neighbors the next 5 (wrapping)
            nl.neighbors_mut()[i] = (1..=5).map(|d| (i + d) % n).collect();
        }
        nl.flatten();

        for i in 0..n {
            let expected: Vec<usize> = (1..=5).map(|d| (i + d) % n).collect();
            assert_eq!(nl.get_neighbors(i), expected.as_slice());
        }
    }
}
