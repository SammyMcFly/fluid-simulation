//! Module provides the necessary entities for an efficient
//! neighbor search implementation
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

pub trait NeighborSearch: Send + Sync + Clone {
    /// Initialize spatial hashing with given cell size.
    fn new(search_range: f64) -> Self;
    fn find_samples(
        &mut self,
        search_range: f64,
        positions: &[Point3<f64>],
        sample_positions: &[Point3<f64>],
        neighbor_list: &mut NeighborList,
    );
}

/// Calculate the distance between two 3D points
pub fn distance(from: &Point3<f64>, to: &Point3<f64>) -> f64 {
    (to - from).norm()
}

pub use neighbor_list::NeighborList;

mod neighbor_list {
    use super::*;
    /// Struct that stores neighbors of samples in a flat array.
    ///
    /// The only way to (re)populate this list is [`NeighborList::rebuild`],
    /// which resizes, clears, fills and flattens atomically. `resize`,
    /// `clear` and `flatten` are private to this module — visible only here
    /// and in `neighbor_search::spatial_hashing` (a descendant module) via
    /// `pub(super)`, and to this module's own `tests` submodule — so no
    /// other caller can observe or misuse the intermediate,
    /// not-yet-flattened state.
    #[derive(Debug, Clone)]
    pub struct NeighborList {
        /// Flat neighbor list: indices of neighboring samples
        indices: Vec<usize>,
        /// Index list to point to start of the neighbor list of each sample
        offsets: Vec<usize>,
        /// Unflattened neighbor list which is necessary for parallelization
        unflattened_indices: Vec<Vec<usize>>,
    }

    /// Transient handle granting mutable access to a [`NeighborList`]'s
    /// unflattened per-sample buffers. Only ever constructed inside
    /// [`NeighborList::rebuild`] and passed by `&mut` reference to its
    /// `fill` callback — it cannot be obtained any other way, and cannot
    /// outlive that single call.
    pub(super) struct NeighborListBuilder<'a> {
        unflattened_indices: &'a mut Vec<Vec<usize>>,
    }

    impl<'a> NeighborListBuilder<'a> {
        /// Get mutable reference to unflattened neighbor list: one `Vec<usize>` per sample.
        pub(super) fn neighbors_mut(&mut self) -> &mut [Vec<usize>] {
            self.unflattened_indices
        }
    }

    impl Default for NeighborList {
        fn default() -> Self {
            Self::new(0)
        }
    }

    impl NeighborList {
        pub fn new(len: usize) -> Self {
            Self {
                indices: vec![0; len],
                offsets: vec![0; len + 1],
                unflattened_indices: vec![Vec::new(); len],
            }
        }

        /// Number of samples this list holds a neighbor list for — i.e. the
        /// valid range of `id` for `get_neighbors(id)`.
        ///
        /// Not to be confused with `volume_maps::NeighborList::len`, which
        /// counts flattened neighbor *entries* rather than samples — the two
        /// types serve different roles despite the shared "NeighborList" name.
        pub fn num_samples(&self) -> usize {
            self.unflattened_indices.len()
        }

        fn resize(&mut self, len: usize) {
            // self.indices.resize(len, usize::default());
            // self.offsets.resize(len + 1, usize::default());
            self.unflattened_indices.resize(len, Vec::new());
        }

        fn clear(&mut self) {
            self.indices.clear();
            self.offsets.clear();
            for_each!(
                mut [self.unflattened_indices],
                ref [],
                |_id, id_neighbors| {
                    id_neighbors.clear();
                }
            );
        }

        /// Flatten neighbor list
        fn flatten(&mut self) {
            self.indices.clear();
            self.offsets.clear();

            let total_neighbors: usize = self.unflattened_indices.iter().map(|v| v.len()).sum();
            let num_particles = self.unflattened_indices.len();

            self.indices.reserve(total_neighbors);
            self.offsets.reserve(num_particles + 1);

            self.offsets.push(0);
            for nbrs in &self.unflattened_indices {
                self.indices.extend_from_slice(nbrs);
                self.offsets.push(self.indices.len());
            }
        }

        /// Rebuilds this neighbor list from scratch for `num_samples` samples.
        ///
        /// `fill` receives `&mut self` to populate the per-sample buffers via
        /// [`Self::neighbors_mut`]. Resizing, clearing, filling and flattening
        /// happen atomically within this single call — an insane, intermediate
        /// state is never observable from outside this method, so `get_neighbors`
        /// /`pos_now`/`vel_now`/`volume` can only ever see a fully flattened,
        /// consistent list.
        pub(super) fn rebuild(
            &mut self,
            len: usize,
            fill: impl FnOnce(&mut NeighborListBuilder<'_>),
        ) {
            self.resize(len);
            self.clear();
            let mut builder = NeighborListBuilder {
                unflattened_indices: &mut self.unflattened_indices,
            };
            fill(&mut builder);
            self.flatten();
        }

        /// Get indices of neighbor of sample with identifier 'id'
        pub fn get_neighbors(&self, id: usize) -> &[usize] {
            &self.indices[self.offsets[id]..self.offsets[id + 1]]
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn new_creates_empty_neighbor_list() {
            let nl = NeighborList::new(5);
            assert_eq!(nl.unflattened_indices.len(), 5);
            assert!(nl.unflattened_indices.iter().all(|v| v.is_empty()));
        }

        #[test]
        fn new_without_flatten_is_safe_to_query() {
            // Regression test for the exact issue that motivated `NeighborList`'s
            // `new`/`resize` consistency: querying a freshly constructed list
            // (mirroring `Default::default()` / the state before the first
            // `rebuild` call) must not panic, even though `flatten` was never
            // called. `new`'s `offsets = vec![0; len + 1]` combined with an empty
            // `indices` makes every `get_neighbors` call return an empty slice.
            let nl = NeighborList::new(3);
            for i in 0..3 {
                assert_eq!(nl.get_neighbors(i), &[]);
            }
        }

        #[test]
        fn rebuild_from_default_grows_correctly() {
            // Mirrors actual production usage: `NeighborList::default()` (i.e.
            // `new(0)`) is held across time steps, and the first real call is
            // `rebuild` growing it to the current sample count — never `new(n)`
            // directly with the final size, as most other tests do for brevity.
            let mut nl = NeighborList::default();
            nl.rebuild(3, |b| {
                let neighbors = b.neighbors_mut();
                neighbors[0] = vec![1, 2];
                neighbors[2] = vec![0];
            });

            assert_eq!(nl.get_neighbors(0), &[1, 2]);
            assert_eq!(nl.get_neighbors(1), &[]);
            assert_eq!(nl.get_neighbors(2), &[0]);
        }

        #[test]
        fn flatten_empty_list() {
            // `flatten` remains private but directly callable here, since `tests`
            // is a descendant of `neighbor_list`. No `rebuild` needed: this test
            // exercises `flatten` in isolation on freshly-constructed, all-empty
            // data.
            let mut nl = NeighborList::new(3);
            nl.flatten();
            assert_eq!(nl.offsets.len(), 4); // num_particles + 1
            assert_eq!(nl.indices.len(), 0);
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
            // `neighbors_mut` now lives on `NeighborListBuilder`, only reachable
            // inside `rebuild`'s `fill` callback.
            let mut nl = NeighborList::new(1);
            nl.rebuild(1, |b| {
                b.neighbors_mut()[0] = vec![1, 2, 3];
            });
            assert_eq!(nl.get_neighbors(0), &[1, 2, 3]);
        }

        #[test]
        fn flatten_multiple_particles() {
            let mut nl = NeighborList::new(4);
            nl.rebuild(4, |b| {
                let neighbors = b.neighbors_mut();
                neighbors[0] = vec![1, 2];
                neighbors[1] = vec![0, 2, 3];
                neighbors[2] = vec![];
                neighbors[3] = vec![0];
            });

            assert_eq!(nl.get_neighbors(0), &[1, 2]);
            assert_eq!(nl.get_neighbors(1), &[0, 2, 3]);
            assert_eq!(nl.get_neighbors(2), &[]);
            assert_eq!(nl.get_neighbors(3), &[0]);
        }

        #[test]
        fn flatten_preserves_order() {
            let mut nl = NeighborList::new(2);
            nl.rebuild(2, |b| {
                let neighbors = b.neighbors_mut();
                neighbors[0] = vec![5, 3, 7, 1];
                neighbors[1] = vec![9, 0];
            });

            assert_eq!(nl.get_neighbors(0), &[5, 3, 7, 1]);
            assert_eq!(nl.get_neighbors(1), &[9, 0]);
        }

        #[test]
        fn flatten_preserves_duplicate_indices() {
            // `flatten` performs no deduplication — verbatim copying is required,
            // since SPH kernel sums over `get_neighbors` would silently produce
            // wrong results (missing contributions) if duplicates were ever
            // dropped. Documents this invariant explicitly rather than leaving it
            // as an unstated assumption.
            let mut nl = NeighborList::new(1);
            nl.rebuild(1, |b| {
                b.neighbors_mut()[0] = vec![2, 2, 3, 2];
            });

            assert_eq!(nl.get_neighbors(0), &[2, 2, 3, 2]);
        }

        #[test]
        fn clear_resets_all_data() {
            let mut nl = NeighborList::new(3);
            nl.rebuild(3, |b| {
                let neighbors = b.neighbors_mut();
                neighbors[0] = vec![1, 2];
                neighbors[1] = vec![0];
                neighbors[2] = vec![0, 1];
            });

            nl.clear();

            assert!(nl.indices.is_empty());
            assert!(nl.offsets.is_empty());
            assert!(nl.unflattened_indices.iter().all(|v| v.is_empty()));
        }

        #[test]
        fn rebuild_fill_noop_clears_previous_data() {
            // `rebuild` always clears before invoking `fill` — even if `fill` does
            // nothing at all, previously set neighbors must not survive. Documents
            // that clearing is unconditional, not merely a side effect of `fill`
            // happening to overwrite every slot in other tests.
            let mut nl = NeighborList::new(2);
            nl.rebuild(2, |b| {
                let neighbors = b.neighbors_mut();
                neighbors[0] = vec![1];
                neighbors[1] = vec![0];
            });

            nl.rebuild(2, |_| {
                // fill does nothing
            });

            assert_eq!(nl.get_neighbors(0), &[]);
            assert_eq!(nl.get_neighbors(1), &[]);
        }

        #[test]
        fn resize_grows() {
            // `resize` only touches `unflattened_indices`, so this test writes to
            // that field directly instead of going through `rebuild` (which
            // would additionally call `clear`/`flatten`, obscuring what `resize`
            // alone does).
            let mut nl = NeighborList::new(2);
            nl.unflattened_indices[0] = vec![1];
            nl.unflattened_indices[1] = vec![0];

            nl.resize(5);

            assert_eq!(nl.unflattened_indices.len(), 5);
            // Existing data preserved
            assert_eq!(nl.unflattened_indices[0], vec![1]);
            assert_eq!(nl.unflattened_indices[1], vec![0]);
            // New entries are empty
            assert!(nl.unflattened_indices[2].is_empty());
            assert!(nl.unflattened_indices[3].is_empty());
            assert!(nl.unflattened_indices[4].is_empty());
        }

        #[test]
        fn resize_shrinks() {
            let mut nl = NeighborList::new(5);
            nl.unflattened_indices[0] = vec![1, 2];
            nl.unflattened_indices[4] = vec![0];

            nl.resize(2);

            assert_eq!(nl.unflattened_indices.len(), 2);
            assert_eq!(nl.unflattened_indices[0], vec![1, 2]);
        }

        #[test]
        fn rebuild_shrink_then_grow_has_no_stale_data() {
            // `Vec::resize` truncates (drops) elements when shrinking rather than
            // caching them for reuse, so growing back afterwards must yield fresh,
            // empty slots — not stale data from before the shrink. Not obvious from
            // reading `resize` in isolation; worth a direct regression test given
            // how easy it would be to break with a future optimization (e.g.
            // swap-remove-based shrinking that leaves old tail data around).
            let mut nl = NeighborList::new(5);
            nl.rebuild(5, |b| {
                let neighbors = b.neighbors_mut();
                for (i, n) in neighbors.iter_mut().enumerate() {
                    *n = vec![i];
                }
            });

            nl.rebuild(2, |b| {
                b.neighbors_mut()[0] = vec![99];
            });

            nl.rebuild(5, |b| {
                b.neighbors_mut()[0] = vec![42];
                // slots 1..5 intentionally left untouched by `fill`
            });

            assert_eq!(nl.get_neighbors(0), &[42]);
            for i in 1..5 {
                assert_eq!(
                    nl.get_neighbors(i),
                    &[],
                    "slot {i} should be empty, not stale"
                );
            }
        }

        #[test]
        fn flatten_called_twice() {
            let mut nl = NeighborList::new(2);
            nl.rebuild(2, |b| {
                let neighbors = b.neighbors_mut();
                neighbors[0] = vec![1];
                neighbors[1] = vec![0];
            });

            // Modify and flatten again
            nl.rebuild(2, |b| {
                let neighbors = b.neighbors_mut();
                neighbors[0] = vec![1, 3, 5];
                neighbors[1] = vec![];
            });

            assert_eq!(nl.get_neighbors(0), &[1, 3, 5]);
            assert_eq!(nl.get_neighbors(1), &[]);
        }

        #[test]
        fn full_workflow_clear_repopulate_flatten() {
            let mut nl = NeighborList::new(3);

            // First iteration
            nl.rebuild(3, |b| {
                let neighbors = b.neighbors_mut();
                neighbors[0] = vec![1, 2];
                neighbors[1] = vec![0, 2];
                neighbors[2] = vec![0, 1];
            });
            assert_eq!(nl.get_neighbors(0), &[1, 2]);
            assert_eq!(nl.get_neighbors(1), &[0, 2]);
            assert_eq!(nl.get_neighbors(2), &[0, 1]);

            // Second iteration (simulates next timestep) — `rebuild` itself
            // performs the resize/clear/fill/flatten sequence, so there is no
            // separate `clear()` call needed here anymore.
            nl.rebuild(3, |b| {
                let neighbors = b.neighbors_mut();
                neighbors[0] = vec![2];
                neighbors[1] = vec![];
                neighbors[2] = vec![0, 1];
            });
            assert_eq!(nl.get_neighbors(0), &[2]);
            assert_eq!(nl.get_neighbors(1), &[]);
            assert_eq!(nl.get_neighbors(2), &[0, 1]);
        }

        #[test]
        fn get_data_length_consistency() {
            let mut nl = NeighborList::new(4);
            nl.rebuild(4, |b| {
                let neighbors = b.neighbors_mut();
                neighbors[0] = vec![1, 2, 3];
                neighbors[1] = vec![0];
                neighbors[2] = vec![0, 1, 3, 4];
                neighbors[3] = vec![];
            });

            let total: usize = (0..4).map(|i| nl.get_neighbors(i).len()).sum();
            assert_eq!(total, nl.indices.len());
            assert_eq!(nl.offsets.len(), 5); // num_particles + 1
        }

        #[test]
        fn large_neighbor_list() {
            let n = 1000;
            let mut nl = NeighborList::new(n);
            nl.rebuild(n, |b| {
                let neighbors = b.neighbors_mut();
                for (i, neighbor) in neighbors.iter_mut().enumerate() {
                    // Each particle neighbors the next 5 (wrapping)
                    *neighbor = (1..=5).map(|d| (i + d) % n).collect();
                }
            });

            for i in 0..n {
                let expected: Vec<usize> = (1..=5).map(|d| (i + d) % n).collect();
                assert_eq!(nl.get_neighbors(i), expected.as_slice());
            }
        }

        #[test]
        #[should_panic]
        fn get_neighbors_out_of_bounds_panics() {
            // Documents current behavior explicitly: `get_neighbors` uses direct
            // indexing (`self.offsets[id]`), so an out-of-range `id` panics rather
            // than returning e.g. an empty slice or `None`. If this is ever
            // deliberately changed (e.g. to `.get(id)`-based bounds checking for
            // robustness against upstream indexing bugs), this test should be
            // updated/removed accordingly rather than silently left to fail.
            let nl = NeighborList::new(2);
            let _ = nl.get_neighbors(5);
        }
    }
}
