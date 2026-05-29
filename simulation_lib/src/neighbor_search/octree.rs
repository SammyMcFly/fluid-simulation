/// Octree neighbor search algorithm according to Fernández-Fernández et al.
///
/// https://doi.org/10.1145/3550454.3555523
use nalgebra::Vector3;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::for_each;
use crate::neighbor_search::{NeighborSearch, distance};

/// Spatial hashing neighbor search structure.
///
/// With this method, neighbor search is a three step process:
/// - Particles are assigned to cells
/// - Contruction of the octree
/// - Brute-force neighbor search within the octree's leaves
///
///
#[derive(Debug, Clone)]
pub struct Octree {
    cell_size: f64,
    z_index: Vec<u64>,
    cell_offsets: Vec<usize>,
}

impl Octree {
    /// Initialize spatial hashing with given cell size.
    ///
    /// # Performance guidance
    ///
    /// A good default is `cell_size = 1.5 * within_range` (see original paper).
    pub fn new(cell_size: f64) -> Self {
        Self {
            cell_size,
            z_index: Vec::new(),
            cell_offsets: Vec::new(),
        }
    }

    /// Create a spatial hashing structure with optimal cell size for the given range.
    pub fn from_range(within_range: f64) -> Self {
        Self::new(within_range/2.)
    }

    fn assign_to_cells() {

    }

    fn build_octree() {

    }


}

impl NeighborSearch for Octree {
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


    }
}

/// Axis-aligned bounding box
#[derive(Clone, Copy)]
struct OctreeNode {
    interior_cells: [f64; 3],
    exterior_cells: [f64; 3],
}

/// Axis-aligned bounding box
#[derive(Clone, Copy)]
struct AABB {
    min: [f64; 3],
    max: [f64; 3],
}

/// Spread bits of a 21-bit integer into every third bit position
fn expand_bits(mut v: u64) -> u64 {
    // We only look at the lowest 21 bits
    v &= 0x1fffff;
    v = (v | v << 32) & 0x1f00000000ffff;
    v = (v | v << 16) & 0x1f0000ff0000ff;
    v = (v | v <<  8) & 0x100f00f00f00f00f;
    v = (v | v <<  4) & 0x10c30c30c30c30c3;
    v = (v | v <<  2) & 0x1249249249249249;
    v
}

/// Compact every third bit back into a contiguous 21-bit integer
fn compact_bits(mut v: u64) -> u64 {
    v &= 0x1249249249249249;
    v = (v | v >>  2) & 0x10c30c30c30c30c3;
    v = (v | v >>  4) & 0x100f00f00f00f00f;
    v = (v | v >>  8) & 0x1f0000ff0000ff;
    v = (v | v >> 16) & 0x1f00000000ffff;
    v = (v | v >> 32) & 0x1fffff;
    v
}

/// Encode a f64 3D position into a 63-bit Morton code.
/// Positions are normalized relative to the given bounding box.
/// Resolution: 21 bits per axis (2,097,152 subdivisions).
fn encode_morton3d(pos: [f64; 3], bounds: &AABB) -> u64 {
    const GRID: f64 = (1 << 21) as f64; // 2_097_152

    // Normalize to [0, 1]
    let nx = ((pos[0] - bounds.min[0]) / (bounds.max[0] - bounds.min[0])).clamp(0.0, 1.0);
    let ny = ((pos[1] - bounds.min[1]) / (bounds.max[1] - bounds.min[1])).clamp(0.0, 1.0);
    let nz = ((pos[2] - bounds.min[2]) / (bounds.max[2] - bounds.min[2])).clamp(0.0, 1.0);

    // Quantize to 21-bit integers
    let ix = (nx * (GRID - 1.0)) as u64;
    let iy = (ny * (GRID - 1.0)) as u64;
    let iz = (nz * (GRID - 1.0)) as u64;

    // Interleave
    expand_bits(ix) | (expand_bits(iy) << 1) | (expand_bits(iz) << 2)
}

/// Decode a 63-bit Morton code back into a f64 3D position.
fn decode_morton3d(code: u64, bounds: &AABB) -> [f64; 3] {
    const GRID: f64 = (1 << 21) as f64;

    let ix = compact_bits(code);
    let iy = compact_bits(code >> 1);
    let iz = compact_bits(code >> 2);

    let nx = ix as f64 / (GRID - 1.0);
    let ny = iy as f64 / (GRID - 1.0);
    let nz = iz as f64 / (GRID - 1.0);

    [
        bounds.min[0] + nx * (bounds.max[0] - bounds.min[0]),
        bounds.min[1] + ny * (bounds.max[1] - bounds.min[1]),
        bounds.min[2] + nz * (bounds.max[2] - bounds.min[2]),
    ]
}
