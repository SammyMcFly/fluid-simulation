
use nalgebra::Vector3;
use simulation_lib::neighbor_search::{NeighborList, SpatialHashing, NeighborSearch};

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

// ─── find_neighbors (full integration) tests ────────────────────────

#[test]
fn find_neighbors_empty_positions() {
    let mut sh = SpatialHashing::new(1.0);
    let fluid_pos: Vec<Vector3<f64>> = vec![];
    let boundary_pos: Vec<Vector3<f64>> = vec![];
    let mut fluid_nbrs = NeighborList::new(0);
    let mut boundary_nbrs = NeighborList::new(0);

    sh.find_neighbors(2.0, &fluid_pos, &boundary_pos, &mut fluid_nbrs, &mut boundary_nbrs);
    // Should not panic
}

#[test]
fn find_neighbors_single_particle_no_boundary() {
    let mut sh = SpatialHashing::new(1.0);
    let fluid_pos = vec![pos(0.0, 0.0, 0.0)];
    let boundary_pos: Vec<Vector3<f64>> = vec![];
    let mut fluid_nbrs = NeighborList::new(1);
    let mut boundary_nbrs = NeighborList::new(1);

    sh.find_neighbors(2.0, &fluid_pos, &boundary_pos, &mut fluid_nbrs, &mut boundary_nbrs);

    // Finds itself
    assert_eq!(fluid_nbrs.get_neighbors(0), &[0]);
    assert_eq!(boundary_nbrs.get_neighbors(0), &[]);
}

#[test]
fn find_neighbors_two_close_particles() {
    let mut sh = SpatialHashing::new(1.0);
    let fluid_pos = vec![pos(0.0, 0.0, 0.0), pos(0.5, 0.0, 0.0)];
    let boundary_pos: Vec<Vector3<f64>> = vec![];
    let mut fluid_nbrs = NeighborList::new(2);
    let mut boundary_nbrs = NeighborList::new(2);

    sh.find_neighbors(2.0, &fluid_pos, &boundary_pos, &mut fluid_nbrs, &mut boundary_nbrs);

    assert!(sorted_neighbors(&fluid_nbrs, 0).contains(&1));
    assert!(sorted_neighbors(&fluid_nbrs, 1).contains(&0));
}

#[test]
fn find_neighbors_two_far_particles() {
    let mut sh = SpatialHashing::new(1.0);
    let fluid_pos = vec![pos(0.0, 0.0, 0.0), pos(100.0, 0.0, 0.0)];
    let boundary_pos: Vec<Vector3<f64>> = vec![];
    let mut fluid_nbrs = NeighborList::new(2);
    let mut boundary_nbrs = NeighborList::new(2);

    sh.find_neighbors(2.0, &fluid_pos, &boundary_pos, &mut fluid_nbrs, &mut boundary_nbrs);

    // Each only finds itself
    assert_eq!(fluid_nbrs.get_neighbors(0), &[0]);
    assert_eq!(fluid_nbrs.get_neighbors(1), &[1]);
}

#[test]
fn find_neighbors_boundary_particles_found() {
    let mut sh = SpatialHashing::new(1.0);
    let fluid_pos = vec![pos(0.0, 0.0, 0.0)];
    let boundary_pos = vec![pos(0.5, 0.0, 0.0), pos(100.0, 0.0, 0.0)];
    let mut fluid_nbrs = NeighborList::new(1);
    let mut boundary_nbrs = NeighborList::new(1);

    sh.find_neighbors(2.0, &fluid_pos, &boundary_pos, &mut fluid_nbrs, &mut boundary_nbrs);

    // Boundary particle 0 is close, particle 1 is far
    let b_nbrs = sorted_neighbors(&boundary_nbrs, 0);
    assert!(b_nbrs.contains(&0));
    assert!(!b_nbrs.contains(&1));
}

#[test]
fn find_neighbors_symmetry() {
    let mut sh = SpatialHashing::new(1.0);
    let fluid_pos = vec![
        pos(0.0, 0.0, 0.0),
        pos(1.0, 0.0, 0.0),
        pos(0.0, 1.0, 0.0),
    ];
    let boundary_pos: Vec<Vector3<f64>> = vec![];
    let mut fluid_nbrs = NeighborList::new(3);
    let mut boundary_nbrs = NeighborList::new(3);

    sh.find_neighbors(1.5, &fluid_pos, &boundary_pos, &mut fluid_nbrs, &mut boundary_nbrs);

    // If i is neighbor of j, then j is neighbor of i
    for i in 0..3 {
        for &j in fluid_nbrs.get_neighbors(i) {
            if i != j {
                assert!(
                    fluid_nbrs.get_neighbors(j).contains(&i),
                    "Particle {} has neighbor {}, but not vice versa", i, j
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
    let boundary_pos: Vec<Vector3<f64>> = vec![];
    let mut fluid_nbrs = NeighborList::new(4);
    let mut boundary_nbrs = NeighborList::new(4);

    sh.find_neighbors(1.0, &fluid_pos, &boundary_pos, &mut fluid_nbrs, &mut boundary_nbrs);

    // Every particle should be neighbor of every other
    for i in 0..4 {
        let nbrs = sorted_neighbors(&fluid_nbrs, i);
        for j in 0..4 {
            assert!(
                nbrs.contains(&j),
                "Particle {} missing neighbor {}", i, j
            );
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
    let boundary_pos: Vec<Vector3<f64>> = vec![];
    let mut fluid_nbrs = NeighborList::new(4);
    let mut boundary_nbrs = NeighborList::new(4);

    sh.find_neighbors(2.0, &fluid_pos, &boundary_pos, &mut fluid_nbrs, &mut boundary_nbrs);

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
    let boundary_pos: Vec<Vector3<f64>> = vec![];
    let mut fluid_nbrs = NeighborList::new(2);
    let mut boundary_nbrs = NeighborList::new(2);

    // Range = 1.0 → distance 1.5 is out of range
    sh.find_neighbors(1.0, &fluid_pos, &boundary_pos, &mut fluid_nbrs, &mut boundary_nbrs);
    assert!(!fluid_nbrs.get_neighbors(0).contains(&1));

    // Range = 2.0 → distance 1.5 is within range
    sh.find_neighbors(2.0, &fluid_pos, &boundary_pos, &mut fluid_nbrs, &mut boundary_nbrs);
    assert!(fluid_nbrs.get_neighbors(0).contains(&1));
}

#[test]
fn find_neighbors_diagonal_3d() {
    let mut sh = SpatialHashing::new(1.0);
    // Distance = sqrt(0.5^2 + 0.5^2 + 0.5^2) = sqrt(0.75) ≈ 0.866
    let fluid_pos = vec![pos(0.0, 0.0, 0.0), pos(0.5, 0.5, 0.5)];
    let boundary_pos: Vec<Vector3<f64>> = vec![];
    let mut fluid_nbrs = NeighborList::new(2);
    let mut boundary_nbrs = NeighborList::new(2);

    sh.find_neighbors(1.0, &fluid_pos, &boundary_pos, &mut fluid_nbrs, &mut boundary_nbrs);

    assert!(fluid_nbrs.get_neighbors(0).contains(&1));
    assert!(fluid_nbrs.get_neighbors(1).contains(&0));
}

#[test]
fn find_neighbors_across_cell_boundary() {
    let mut sh = SpatialHashing::new(1.0);
    // Particles in adjacent cells but within range
    let fluid_pos = vec![pos(0.9, 0.0, 0.0), pos(1.1, 0.0, 0.0)];
    let boundary_pos: Vec<Vector3<f64>> = vec![];
    let mut fluid_nbrs = NeighborList::new(2);
    let mut boundary_nbrs = NeighborList::new(2);

    sh.find_neighbors(1.0, &fluid_pos, &boundary_pos, &mut fluid_nbrs, &mut boundary_nbrs);

    // Distance = 0.2, should be neighbors
    assert!(fluid_nbrs.get_neighbors(0).contains(&1));
    assert!(fluid_nbrs.get_neighbors(1).contains(&0));
}

#[test]
fn find_neighbors_called_multiple_times() {
    let mut sh = SpatialHashing::new(1.0);
    let boundary_pos: Vec<Vector3<f64>> = vec![];
    let mut fluid_nbrs = NeighborList::new(2);
    let mut boundary_nbrs = NeighborList::new(2);

    // First call: close particles
    let fluid_pos_1 = vec![pos(0.0, 0.0, 0.0), pos(0.5, 0.0, 0.0)];
    sh.find_neighbors(2.0, &fluid_pos_1, &boundary_pos, &mut fluid_nbrs, &mut boundary_nbrs);
    assert!(fluid_nbrs.get_neighbors(0).contains(&1));

    // Second call: particles moved apart
    let fluid_pos_2 = vec![pos(0.0, 0.0, 0.0), pos(100.0, 0.0, 0.0)];
    sh.find_neighbors(2.0, &fluid_pos_2, &boundary_pos, &mut fluid_nbrs, &mut boundary_nbrs);
    assert!(!fluid_nbrs.get_neighbors(0).contains(&1));
}

#[test]
fn find_neighbors_large_cell_size() {
    let mut sh = SpatialHashing::new(10.0);
    let fluid_pos = vec![pos(0.0, 0.0, 0.0), pos(5.0, 0.0, 0.0)];
    let boundary_pos: Vec<Vector3<f64>> = vec![];
    let mut fluid_nbrs = NeighborList::new(2);
    let mut boundary_nbrs = NeighborList::new(2);

    // Range < distance → not neighbors even though in same cell
    sh.find_neighbors(2.0, &fluid_pos, &boundary_pos, &mut fluid_nbrs, &mut boundary_nbrs);
    assert!(!fluid_nbrs.get_neighbors(0).contains(&1));

    // Range > distance → neighbors
    sh.find_neighbors(6.0, &fluid_pos, &boundary_pos, &mut fluid_nbrs, &mut boundary_nbrs);
    assert!(fluid_nbrs.get_neighbors(0).contains(&1));
}
