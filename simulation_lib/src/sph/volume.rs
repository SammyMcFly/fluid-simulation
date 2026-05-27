/// Volume calculation module
#[cfg(feature = "parallelized_sph")]
use rayon::prelude::*;

use crate::for_each;
use crate::sph::kernel::KernelFn;
use crate::sample::{Fluid3D, Boundary3D, Positional};
use crate::sph::SystemParameters;
use crate::sph::distance;

/// Calculate and update volume for all particles for the current point in time
pub fn update_volume<K: KernelFn>(
    fluid: &mut Fluid3D,
    boundary: &Boundary3D,
    params: &SystemParameters,
) {
    for_each!(
        mut [fluid.volume],
        ref [
            pos_now = fluid.position,
            neighbors = fluid.neighbors,
            boundary_neighbors = fluid.boundary_neighbors,
        ],
        |id, id_volume| {
            // reset volume
            *id_volume = 0.;
            let mut accu = 0.;
            // add volume for every neighbor
            for &neighbor in &neighbors[id] {
                let dist = distance(
                    &pos_now[id],
                    &pos_now[neighbor],
                );
                accu += params.rest_volume
                    * K::value(
                        dist,
                        params.smoothing_length,
                    );
            }
            // add volume for every boundary neighbor (mirror mass of moving particle onto boundary particle)
            for &boundary_neighbor in &boundary_neighbors[id] {
                let dist = distance(
                    &pos_now[id],
                    boundary.pos_now(boundary_neighbor),
                );
                accu += *boundary.volume(boundary_neighbor)
                    * K::value(
                        dist,
                        params.smoothing_length,
                    );
            }
            *id_volume += params.rest_volume / accu;
        }
    );
}