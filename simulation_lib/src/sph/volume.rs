/// Volume calculation module
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::for_each;
use crate::sph::kernel::KernelFn;
use crate::sample::{Fluid3D, Boundary3D, Positional};
use crate::sph::{SystemParameters, vector};
use crate::neighbor_search::NeighborList;

/// Calculate and update volume for all particles for the current point in time
pub fn update_volume<K: KernelFn>(
    fluid: &mut Fluid3D,
    boundary: &Boundary3D,
    neighbors: &NeighborList,
    boundary_neighbors: &NeighborList,
    params: &SystemParameters,
) {
    for_each!(
        mut [fluid.volume],
        ref [
            pos_now = fluid.position,
            neighbors = neighbors,
            boundary_neighbors = boundary_neighbors,
        ],
        |id, id_volume| {
            // reset volume
            *id_volume = 0.;
            let mut accu = 0.;
            // add volume for every neighbor
            for &neighbor in neighbors.get_neighbors(id) {
                let r_vec = vector(
                    &pos_now[neighbor],
                    &pos_now[id],

                );
                accu += params.rest_volume
                    * K::kernel_function(
                        &r_vec,
                        params.kernel_support_radius,
                    );
            }
            // add volume for every boundary neighbor (mirror mass of moving particle onto boundary particle)
            for &boundary_neighbor in boundary_neighbors.get_neighbors(id) {
                let r_vec = vector(
                    boundary.pos_now(boundary_neighbor),
                    &pos_now[id],
                );
                accu += *boundary.volume(boundary_neighbor)
                    * K::kernel_function(
                        &r_vec,
                        params.kernel_support_radius,
                    );
            }
            *id_volume += params.rest_volume / accu;
        }
    );
}