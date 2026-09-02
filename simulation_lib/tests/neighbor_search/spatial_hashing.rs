use nalgebra::Point3;
use simulation_lib::neighbor_search::{NeighborList, NeighborSearch, SpatialHashing};

// ─── Helper functions ───────────────────────────────────────────────

fn pos(x: f64, y: f64, z: f64) -> Point3<f64> {
    Point3::new(x, y, z)
}

/// Collect neighbors of particle `id` as a sorted Vec for order-independent comparison
fn sorted_neighbors(nl: &NeighborList, id: usize) -> Vec<usize> {
    let mut v = nl.get_neighbors(id).to_vec();
    v.sort();
    v
}

// ─── find_samples (full integration) tests ──────────────────────────

#[test]
fn find_neighbors_empty_positions() {
    let mut sh = SpatialHashing::new(1.0);
    let fluid_pos: Vec<Point3<f64>> = vec![];
    let mut fluid_nbrs = NeighborList::new(0);

    sh.find_samples(2.0, &fluid_pos, &fluid_pos, &mut fluid_nbrs);
    // Should not panic
}

#[test]
fn find_neighbors_single_particle_no_boundary() {
    let mut sh = SpatialHashing::new(1.0);
    let fluid_pos = vec![pos(0.0, 0.0, 0.0)];
    let boundary_pos: Vec<Point3<f64>> = vec![];
    let mut fluid_nbrs = NeighborList::new(1);
    let mut boundary_nbrs = NeighborList::new(1);

    // Two separate calls: fluid-fluid and fluid-boundary neighbor search are
    // independent invocations of `find_samples`, each with its own output
    // `NeighborList` — unlike the old (stale) API this file previously
    // assumed, there is no single call that produces both at once.
    sh.find_samples(2.0, &fluid_pos, &fluid_pos, &mut fluid_nbrs);
    sh.find_samples(2.0, &fluid_pos, &boundary_pos, &mut boundary_nbrs);

    // Finds itself (self is always within range 0 of itself)
    assert_eq!(fluid_nbrs.get_neighbors(0), &[0]);
    assert_eq!(boundary_nbrs.get_neighbors(0), &[]);
}

#[test]
fn find_neighbors_two_close_particles() {
    let mut sh = SpatialHashing::new(1.0);
    let fluid_pos = vec![pos(0.0, 0.0, 0.0), pos(0.5, 0.0, 0.0)];
    let mut fluid_nbrs = NeighborList::new(2);

    sh.find_samples(2.0, &fluid_pos, &fluid_pos, &mut fluid_nbrs);

    assert!(sorted_neighbors(&fluid_nbrs, 0).contains(&1));
    assert!(sorted_neighbors(&fluid_nbrs, 1).contains(&0));
}

#[test]
fn find_neighbors_two_far_particles() {
    let mut sh = SpatialHashing::new(1.0);
    let fluid_pos = vec![pos(0.0, 0.0, 0.0), pos(100.0, 0.0, 0.0)];
    let mut fluid_nbrs = NeighborList::new(2);

    sh.find_samples(2.0, &fluid_pos, &fluid_pos, &mut fluid_nbrs);

    // Each only finds itself
    assert_eq!(fluid_nbrs.get_neighbors(0), &[0]);
    assert_eq!(fluid_nbrs.get_neighbors(1), &[1]);
}

#[test]
fn find_neighbors_boundary_particles_found() {
    let mut sh = SpatialHashing::new(1.0);
    let fluid_pos = vec![pos(0.0, 0.0, 0.0)];
    let boundary_pos = vec![pos(0.5, 0.0, 0.0), pos(100.0, 0.0, 0.0)];
    let mut boundary_nbrs = NeighborList::new(1);

    sh.find_samples(2.0, &fluid_pos, &boundary_pos, &mut boundary_nbrs);

    // Boundary particle 0 is close, particle 1 is far
    let b_nbrs = sorted_neighbors(&boundary_nbrs, 0);
    assert!(b_nbrs.contains(&0));
    assert!(!b_nbrs.contains(&1));
}

#[test]
fn find_neighbors_symmetry() {
    let mut sh = SpatialHashing::new(1.0);
    let fluid_pos = vec![pos(0.0, 0.0, 0.0), pos(1.0, 0.0, 0.0), pos(0.0, 1.0, 0.0)];
    let mut fluid_nbrs = NeighborList::new(3);

    sh.find_samples(1.5, &fluid_pos, &fluid_pos, &mut fluid_nbrs);

    // If i is neighbor of j, then j is neighbor of i
    for i in 0..3 {
        for &j in fluid_nbrs.get_neighbors(i) {
            if i != j {
                assert!(
                    fluid_nbrs.get_neighbors(j).contains(&i),
                    "Particle {} has neighbor {}, but not vice versa",
                    i,
                    j
                );
            }
        }
    }
}

#[test]
fn find_neighbors_cluster() {
    let mut sh = SpatialHashing::new(1.0);
    // Tight cluster: all within range of each other
    let fluid_pos = vec![
        pos(0.0, 0.0, 0.0),
        pos(0.3, 0.0, 0.0),
        pos(0.0, 0.3, 0.0),
        pos(0.0, 0.0, 0.3),
    ];
    let mut fluid_nbrs = NeighborList::new(4);

    sh.find_samples(1.0, &fluid_pos, &fluid_pos, &mut fluid_nbrs);

    // Every particle should be neighbor of every other
    for i in 0..4 {
        let nbrs = sorted_neighbors(&fluid_nbrs, i);
        for j in 0..4 {
            assert!(nbrs.contains(&j), "Particle {} missing neighbor {}", i, j);
        }
    }
}

#[test]
fn find_neighbors_two_separate_clusters() {
    let mut sh = SpatialHashing::new(1.0);
    // Two clusters far apart
    let fluid_pos = vec![
        // Cluster A
        pos(0.0, 0.0, 0.0),
        pos(0.5, 0.0, 0.0),
        // Cluster B
        pos(50.0, 50.0, 50.0),
        pos(50.5, 50.0, 50.0),
    ];
    let mut fluid_nbrs = NeighborList::new(4);

    sh.find_samples(2.0, &fluid_pos, &fluid_pos, &mut fluid_nbrs);

    // Cluster A particles are neighbors of each other
    assert!(sorted_neighbors(&fluid_nbrs, 0).contains(&1));
    assert!(sorted_neighbors(&fluid_nbrs, 1).contains(&0));
    // Cluster B particles are neighbors of each other
    assert!(sorted_neighbors(&fluid_nbrs, 2).contains(&3));
    assert!(sorted_neighbors(&fluid_nbrs, 3).contains(&2));
    // No cross-cluster neighbors
    assert!(!sorted_neighbors(&fluid_nbrs, 0).contains(&2));
    assert!(!sorted_neighbors(&fluid_nbrs, 0).contains(&3));
    assert!(!sorted_neighbors(&fluid_nbrs, 2).contains(&0));
    assert!(!sorted_neighbors(&fluid_nbrs, 2).contains(&1));
}

#[test]
fn find_neighbors_respects_range_parameter() {
    let mut sh = SpatialHashing::new(1.0);
    let fluid_pos = vec![pos(0.0, 0.0, 0.0), pos(1.5, 0.0, 0.0)];
    let mut fluid_nbrs = NeighborList::new(2);

    // Range = 1.0 → distance 1.5 is out of range
    sh.find_samples(1.0, &fluid_pos, &fluid_pos, &mut fluid_nbrs);
    assert!(!fluid_nbrs.get_neighbors(0).contains(&1));

    // Range = 2.0 → distance 1.5 is within range
    sh.find_samples(2.0, &fluid_pos, &fluid_pos, &mut fluid_nbrs);
    assert!(fluid_nbrs.get_neighbors(0).contains(&1));
}

#[test]
fn find_neighbors_diagonal_3d() {
    let mut sh = SpatialHashing::new(1.0);
    // Distance = sqrt(0.5^2 + 0.5^2 + 0.5^2) = sqrt(0.75) ≈ 0.866
    let fluid_pos = vec![pos(0.0, 0.0, 0.0), pos(0.5, 0.5, 0.5)];
    let mut fluid_nbrs = NeighborList::new(2);

    sh.find_samples(1.0, &fluid_pos, &fluid_pos, &mut fluid_nbrs);

    assert!(fluid_nbrs.get_neighbors(0).contains(&1));
    assert!(fluid_nbrs.get_neighbors(1).contains(&0));
}

#[test]
fn find_neighbors_duplicate_positions() {
    // Coincident particles (distance = 0) occur in practice — e.g. two
    // fluid samples at an identical grid point during initialization, or
    // degenerate boundary samples. Distance 0 must always be `< range` for
    // any `range > 0`, so duplicates must find each other.
    let mut sh = SpatialHashing::new(1.0);
    let fluid_pos = vec![pos(1.0, 1.0, 1.0), pos(1.0, 1.0, 1.0)];
    let mut fluid_nbrs = NeighborList::new(2);

    sh.find_samples(0.5, &fluid_pos, &fluid_pos, &mut fluid_nbrs);

    assert!(fluid_nbrs.get_neighbors(0).contains(&1));
    assert!(fluid_nbrs.get_neighbors(1).contains(&0));
}

#[test]
fn find_neighbors_across_cell_boundary() {
    let mut sh = SpatialHashing::new(1.0);
    // Particles in adjacent cells but within range
    let fluid_pos = vec![pos(0.9, 0.0, 0.0), pos(1.1, 0.0, 0.0)];
    let mut fluid_nbrs = NeighborList::new(2);

    sh.find_samples(1.0, &fluid_pos, &fluid_pos, &mut fluid_nbrs);

    // Distance = 0.2, should be neighbors
    assert!(fluid_nbrs.get_neighbors(0).contains(&1));
    assert!(fluid_nbrs.get_neighbors(1).contains(&0));
}

#[test]
fn find_neighbors_called_multiple_times() {
    let mut sh = SpatialHashing::new(1.0);
    let mut fluid_nbrs = NeighborList::new(2);

    // First call: close particles
    let fluid_pos_1 = vec![pos(0.0, 0.0, 0.0), pos(0.5, 0.0, 0.0)];
    sh.find_samples(2.0, &fluid_pos_1, &fluid_pos_1, &mut fluid_nbrs);
    assert!(fluid_nbrs.get_neighbors(0).contains(&1));

    // Second call: particles moved apart. `find_samples` (via `rebuild`)
    // must fully replace stale data from the previous call, not merely add
    // to it.
    let fluid_pos_2 = vec![pos(0.0, 0.0, 0.0), pos(100.0, 0.0, 0.0)];
    sh.find_samples(2.0, &fluid_pos_2, &fluid_pos_2, &mut fluid_nbrs);
    assert!(!fluid_nbrs.get_neighbors(0).contains(&1));
}

#[test]
fn find_neighbors_particle_count_grows_between_calls() {
    // Exercises `rebuild`'s resize path through the actual production call
    // site (`find_samples`), not just via direct `NeighborList` unit tests.
    // Mirrors a real simulation where the fluid particle count can increase
    // between time steps (though currently only ever decreases via
    // `drop_inactive` — this covers the more demanding growth direction).
    let mut sh = SpatialHashing::new(1.0);
    let mut nl = NeighborList::new(2);

    let small = vec![pos(0.0, 0.0, 0.0), pos(0.5, 0.0, 0.0)];
    sh.find_samples(1.0, &small, &small, &mut nl);
    assert!(nl.get_neighbors(0).contains(&1));

    let grown = vec![
        pos(0.0, 0.0, 0.0),
        pos(0.5, 0.0, 0.0),
        pos(0.3, 0.0, 0.0),
        pos(100.0, 0.0, 0.0),
    ];
    sh.find_samples(1.0, &grown, &grown, &mut nl);

    assert!(sorted_neighbors(&nl, 0).contains(&2));
    assert_eq!(nl.get_neighbors(3), &[3]); // far particle only finds itself
}

#[test]
fn find_neighbors_particle_count_shrinks_between_calls() {
    // The inverse direction of the previous test — and the direction that
    // actually occurs in production via `Fluid::drop_inactive`. Confirms no
    // stale neighbor data from removed particle slots leaks into the
    // smaller list.
    let mut sh = SpatialHashing::new(1.0);
    let mut nl = NeighborList::new(4);

    let large = vec![
        pos(0.0, 0.0, 0.0),
        pos(0.5, 0.0, 0.0),
        pos(0.3, 0.0, 0.0),
        pos(100.0, 0.0, 0.0),
    ];
    sh.find_samples(1.0, &large, &large, &mut nl);

    let shrunk = vec![pos(0.0, 0.0, 0.0), pos(0.5, 0.0, 0.0)];
    sh.find_samples(1.0, &shrunk, &shrunk, &mut nl);

    assert!(nl.get_neighbors(0).contains(&1));
    assert!(nl.get_neighbors(1).contains(&0));
}

#[test]
fn find_neighbors_large_cell_size() {
    let mut sh = SpatialHashing::new(10.0);
    let fluid_pos = vec![pos(0.0, 0.0, 0.0), pos(5.0, 0.0, 0.0)];
    let mut fluid_nbrs = NeighborList::new(2);

    // Range < distance → not neighbors even though in same cell
    sh.find_samples(2.0, &fluid_pos, &fluid_pos, &mut fluid_nbrs);
    assert!(!fluid_nbrs.get_neighbors(0).contains(&1));

    // Range > distance → neighbors
    sh.find_samples(6.0, &fluid_pos, &fluid_pos, &mut fluid_nbrs);
    assert!(fluid_nbrs.get_neighbors(0).contains(&1));
}

#[test]
fn find_neighbors_cell_size_independent_of_query_range() {
    // `SpatialHashing::new`'s doc comment states `within_range` (used as
    // `cell_size`) is only a *performance* tuning parameter — correctness
    // must hold even if the range passed to `find_samples` differs
    // significantly from the value used at construction. This is exactly
    // the scenario in `sph.rs`: `neighbor_search` is constructed once with
    // `kernel_support_radius`, but reused for both fluid-fluid neighbor
    // search and (via `get_quantity_at_positions`) arbitrary sensor-plane
    // queries with potentially different ranges.
    let mut sh = SpatialHashing::new(0.1); // cell_size far smaller than query range
    let fluid_pos = vec![pos(0.0, 0.0, 0.0), pos(0.5, 0.0, 0.0)];
    let mut fluid_nbrs = NeighborList::new(2);

    sh.find_samples(1.0, &fluid_pos, &fluid_pos, &mut fluid_nbrs);

    assert!(fluid_nbrs.get_neighbors(0).contains(&1));
    assert!(fluid_nbrs.get_neighbors(1).contains(&0));
}
