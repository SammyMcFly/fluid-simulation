//! Volume calculation module
use nalgebra::{Point3, Vector3};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::for_each;
use crate::neighbor_search::NeighborList;
use crate::sph::SystemParameters;
use crate::sph::boundary_handling::{BoundaryHandling, RequestMode};
use crate::sph::kernel::KernelFn;
use crate::utilities::vector;

/// Calculate and set volume for all positions at the current point in time
pub fn get_volume<K: KernelFn>(
    volume: &mut Vec<f64>,
    position_eval: &Vec<Point3<f64>>,
    neighboring_samples: &NeighborList,
    sample_positions: &Vec<Point3<f64>>,
    boundary: &impl BoundaryHandling,
    params: &SystemParameters,
) {
    for_each!(
        mut [volume],
        ref [
            position_eval = position_eval,
            neighbors = neighboring_samples,
            sample_positions = sample_positions,
            boundary = boundary,
        ],
        |id, id_volume| {
            let mut accu = 0.;
            // add volume for every neighbor
            for &neighbor in neighbors.get_neighbors(id) {
                let r_vec = vector(
                    &sample_positions[neighbor],
                    &position_eval[id],
                );
                accu += params.rest_volume
                    * K::kernel_function(
                        &r_vec,
                        params.kernel_support_radius,
                    );
            }
            // add volume contribution from boundary
            for b in boundary.iter() {
                for &boundary_neighbor in b.get_neighbors(id, RequestMode::Normal) {
                    let r_vec = vector(
                        b.position(boundary_neighbor),
                        &position_eval[id],
                    );
                    accu += b.volume(boundary_neighbor)
                        * K::kernel_function(
                            &r_vec,
                            params.kernel_support_radius,
                        );
                }
            }
            *id_volume = params.rest_volume / accu;
        }
    );
}

/// Calculate and set speed for all positions at the current point in time
#[allow(clippy::too_many_arguments)]
pub fn get_speed<K: KernelFn>(
    speed: &mut Vec<f64>,
    position_eval: &Vec<Point3<f64>>,
    neighboring_samples: &NeighborList,
    sample_positions: &Vec<Point3<f64>>,
    sample_velocities: &Vec<Vector3<f64>>,
    sample_volumes: &Vec<f64>,
    boundary: &impl BoundaryHandling,
    params: &SystemParameters,
) {
    for_each!(
        mut [speed],
        ref [
            pos_now_eval = position_eval,
            neighbors = neighboring_samples,
            sample_positions = sample_positions,
            sample_velocities = sample_velocities,
            sample_volumes = sample_volumes,
            boundary = boundary,
            params = params,
        ],
        |id, id_speed| {
            let mut accu = Vector3::zeros();
            // add velocity for every neighbor
            for &neighbor in neighbors.get_neighbors(id) {
                let r_vec = vector(
                    &pos_now_eval[id],
                    &sample_positions[neighbor],
                );
                accu += sample_velocities[neighbor]
                    * sample_volumes[neighbor]
                    * K::kernel_function(
                        &r_vec,
                        params.kernel_support_radius,
                    );
            }
            // add contribution from boundary
            for b in boundary.iter() {
                for &boundary_neighbor in b.get_neighbors(id, RequestMode::Normal) {
                    let r_vec = vector(
                        &pos_now_eval[id],
                        b.position(boundary_neighbor),
                    );
                    accu += *b.velocity(boundary_neighbor)
                        * b.volume(boundary_neighbor)
                        * K::kernel_function(
                            &r_vec,
                            params.kernel_support_radius,
                        );
                }
            }
            *id_speed = accu.norm();
        }
    );
}

/// Calculate and set density for all positions at the current point in time
pub fn get_density<K: KernelFn>(
    density: &mut Vec<f64>,
    position_eval: &Vec<Point3<f64>>,
    neighboring_samples: &NeighborList,
    sample_positions: &Vec<Point3<f64>>,
    sample_masses: &Vec<f64>,
    // boundary: &impl BoundaryHandling,
    params: &SystemParameters,
) {
    for_each!(
        mut [density],
        ref [
            position_eval = position_eval,
            neighbors = neighboring_samples,
            sample_positions = sample_positions,
            sample_masses = sample_masses,
            // boundary = boundary,
            params = params,
        ],
        |id, id_density| {
            let mut accu = 0.;
            // add volume for every neighbor
            for &neighbor in neighbors.get_neighbors(id) {
                let r_vec = vector(
                    &position_eval[id],
                    &sample_positions[neighbor],
                );
                accu += sample_masses[neighbor]
                    * K::kernel_function(
                        &r_vec,
                        params.kernel_support_radius,
                    );
            }
            // // add contribution from boundary
            // for &boundary_neighbor in boundary.get_neighbors(id) {
            //     let r_vec = vector(
            //         &position_eval[id],
            //         boundary.pos_now(boundary_neighbor),
            //     );
            //     accu += *boundary.density(boundary_neighbor)
            //         *boundary.volume(boundary_neighbor)
            //         * K::kernel_function(
            //             &r_vec,
            //             params.kernel_support_radius,
            //         );
            // }
            *id_density = accu;
        }
    );
}

/// Calculate and set density for all positions at the current point in time
pub fn get_density_error<K: KernelFn>(
    density_err: &mut Vec<f64>,
    position_eval: &Vec<Point3<f64>>,
    neighboring_samples: &NeighborList,
    sample_positions: &Vec<Point3<f64>>,
    sample_volumes: &Vec<f64>,
    // boundary: &impl BoundaryHandling,
    params: &SystemParameters,
) {
    for_each!(
        mut [density_err],
        ref [
            position_eval = position_eval,
            neighbors = neighboring_samples,
            sample_positions = sample_positions,
            sample_volumes = sample_volumes,
            // boundary = boundary,
            params = params,
        ],
        |id, id_density_err| {
            let mut accu = 0.;
            // add volume for every neighbor
            for &neighbor in neighbors.get_neighbors(id) {
                let r_vec = vector(
                    &position_eval[id],
                    &sample_positions[neighbor],
                );
                let err = if sample_volumes[neighbor] < params.rest_volume {
                    params.rest_volume / sample_volumes[neighbor] - 1.
                } else {
                    continue;
                };
                accu += err
                    * sample_volumes[neighbor]
                    * K::kernel_function(
                        &r_vec,
                        params.kernel_support_radius,
                    );
            }
            // // add contribution from boundary
            // for &boundary_neighbor in boundary.get_neighbors(id) {
            //     let r_vec = vector(
            //         &position_eval[id],
            //         boundary.pos_now(boundary_neighbor),
            //     );
            //     accu += *boundary.density(boundary_neighbor)
            //         *boundary.volume(boundary_neighbor)
            //         * K::kernel_function(
            //             &r_vec,
            //             params.kernel_support_radius,
            //         );
            // }
            *id_density_err = 100. * accu;
        }
    );
}

/// Calculate and set speed for all positions at the current point in time
pub fn get_pressure<K: KernelFn>(
    pressure: &mut Vec<f64>,
    position_eval: &Vec<Point3<f64>>,
    neighboring_samples: &NeighborList,
    sample_positions: &Vec<Point3<f64>>,
    sample_volumes: &Vec<f64>,
    sample_pressure: &Vec<f64>,
    // boundary: &impl BoundaryHandling,
    params: &SystemParameters,
) {
    for_each!(
        mut [pressure],
        ref [
            pos_now_eval = position_eval,
            neighbors = neighboring_samples,
            sample_positions = sample_positions,
            sample_volumes = sample_volumes,
            sample_pressure = sample_pressure,
            // boundary = boundary,
            params = params,
        ],
        |id, id_pressure| {
            let mut accu = 0.;
            // add velocity for every neighbor
            for &neighbor in neighbors.get_neighbors(id) {
                let r_vec = vector(
                    &pos_now_eval[id],
                    &sample_positions[neighbor],
                );
                accu += sample_pressure[neighbor]
                    * sample_volumes[neighbor]
                    * K::kernel_function(
                        &r_vec,
                        params.kernel_support_radius,
                    );
            }
            // // add contribution from boundary
            // for &boundary_neighbor in boundary.get_neighbors(id) {
            //     let r_vec = vector(
            //         &pos_now_eval[id],
            //         boundary.pos_now(boundary_neighbor),
            //     );
            //     accu += *boundary.vel_now(boundary_neighbor)
            //         * *boundary.volume(boundary_neighbor)
            //         * K::kernel_function(
            //             &r_vec,
            //             params.kernel_support_radius,
            //         );
            // }
            *id_pressure = accu;
        }
    );
}

/// Calculate and set speed for all positions at the current point in time
#[allow(clippy::too_many_arguments)]
pub fn get_kinetic_energy<K: KernelFn>(
    kinetic_energy: &mut Vec<f64>,
    position_eval: &Vec<Point3<f64>>,
    neighboring_samples: &NeighborList,
    sample_positions: &Vec<Point3<f64>>,
    sample_velocities: &Vec<Vector3<f64>>,
    sample_volumes: &Vec<f64>,
    sample_masses: &Vec<f64>,
    // boundary: &impl BoundaryHandling,
    params: &SystemParameters,
) {
    for_each!(
        mut [kinetic_energy],
        ref [
            pos_now_eval = position_eval,
            neighbors = neighboring_samples,
            sample_positions = sample_positions,
            sample_velocities = sample_velocities,
            sample_volumes = sample_volumes,
            sample_masses = sample_masses,
            // boundary = boundary,
            params = params,
        ],
        |id, id_kinetic_energy| {
            let mut accu = 0.;
            // add velocity for every neighbor
            for &neighbor in neighbors.get_neighbors(id) {
                let r_vec = vector(
                    &pos_now_eval[id],
                    &sample_positions[neighbor],
                );
                accu += 0.5 * sample_masses[neighbor]
                    * sample_velocities[neighbor].norm_squared()
                    * sample_volumes[neighbor]
                    * K::kernel_function(
                        &r_vec,
                        params.kernel_support_radius,
                    );
            }
            // // add contribution from boundary
            // for &boundary_neighbor in boundary.get_neighbors(id) {
            //     let r_vec = vector(
            //         &pos_now_eval[id],
            //         boundary.pos_now(boundary_neighbor),
            //     );
            //     accu += 0.5 * boundary.density(boundary_neighbor)
            //         * boundary.vel_now(boundary_neighbor).norm_squared()
            //         * boundary.volume(boundary_neighbor).powi(2)
            //         * K::kernel_function(
            //                 &r_vec,
            //             params.kernel_support_radius,
            //         );
            // }
            *id_kinetic_energy = accu;
        }
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neighbor_search::{NeighborSearch, SpatialHashing};
    use crate::sph::boundary_handling::VolumeMapBoundary;
    use crate::sph::kernel::CubicBSpline3D;

    // ─── Fixtures / helpers ─────────────────────────────────────────────

    fn make_params(kernel_support_radius: f64, rest_density_grid_spacing: f64) -> SystemParameters {
        #[cfg(not(feature = "cfl_time_step"))]
        {
            SystemParameters::new(
                0.001,
                rest_density_grid_spacing,
                kernel_support_radius,
                -1e9,
                0.0,
                0.0,
                1.0,
            )
        }
        #[cfg(feature = "cfl_time_step")]
        {
            SystemParameters::new(
                0.01,
                0.4,
                rest_density_grid_spacing,
                kernel_support_radius,
                -1e9,
                0.0,
                0.0,
                1.0,
            )
        }
    }

    /// A `BoundaryHandling` instance with zero boundaries — used to isolate
    /// the pure fluid-fluid contribution in every test below. Populating a
    /// real boundary with actual samples (via `find_boundary_samples`) is
    /// out of scope here and is covered by `volume_map_boundary`'s own test
    /// suite; what's verified here is only that the boundary-iteration code
    /// path is correctly wired and contributes exactly zero when empty.
    fn empty_boundary() -> VolumeMapBoundary {
        VolumeMapBoundary::default()
    }

    fn build_neighbor_list(
        eval_positions: &[Point3<f64>],
        sample_positions: &[Point3<f64>],
        support_radius: f64,
    ) -> NeighborList {
        let mut ns = SpatialHashing::new(support_radius);
        let mut neighbor_list = NeighborList::new(eval_positions.len());
        ns.find_samples(
            support_radius,
            eval_positions,
            sample_positions,
            &mut neighbor_list,
        );
        neighbor_list
    }

    /// `kernel_function` is radially symmetric, so the argument order of
    /// the difference vector doesn't matter — this sidesteps depending on
    /// `crate::utilities::vector`'s exact sign convention.
    fn kernel_weight(a: &Point3<f64>, b: &Point3<f64>, h: f64) -> f64 {
        CubicBSpline3D::kernel_function(&(a - b), h)
    }

    /// Builds a cubic lattice of points at spacing `dx`, covering
    /// `[-radius, radius]` along every axis (inclusive), centered at the
    /// origin — used for the "recovers rest density" sanity checks below.
    fn cubic_lattice(radius: f64, dx: f64) -> Vec<Point3<f64>> {
        let steps = (radius / dx).round() as i64;
        let mut points = Vec::new();
        for ix in -steps..=steps {
            for iy in -steps..=steps {
                for iz in -steps..=steps {
                    points.push(Point3::new(ix as f64 * dx, iy as f64 * dx, iz as f64 * dx));
                }
            }
        }
        points
    }

    // ─── get_volume ───────────────────────────────────────────────────────

    #[test]
    fn get_volume_matches_manual_formula_for_a_small_cluster() {
        let h = 1.0;
        let dx = 0.3;
        let params = make_params(h, dx);
        let boundary = empty_boundary();

        let positions = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.3, 0.0, 0.0),
            Point3::new(0.0, 0.3, 0.0),
        ];
        let neighbor_list = build_neighbor_list(&positions, &positions, h);

        let mut volume = vec![0.0; positions.len()];
        get_volume::<CubicBSpline3D>(
            &mut volume,
            &positions,
            &neighbor_list,
            &positions,
            &boundary,
            &params,
        );

        for id in 0..positions.len() {
            let accu: f64 = neighbor_list
                .get_neighbors(id)
                .iter()
                .map(|&j| params.rest_volume * kernel_weight(&positions[j], &positions[id], h))
                .sum();
            let expected = params.rest_volume / accu;
            assert!(
                (volume[id] - expected).abs() < 1e-9,
                "id={id}: expected {expected}, got {}",
                volume[id]
            );
        }
    }

    #[test]
    fn get_volume_of_a_completely_isolated_particle_is_infinite() {
        // Documents a real edge case: with zero neighbor contributions, the
        // denominator `accu` is exactly 0.0, so the division produces
        // `+inf` rather than a well-defined volume. This is the actual,
        // current behavior — flagged here explicitly rather than silently
        // relied upon.
        let h = 0.1;
        let dx = 0.05;
        let params = make_params(h, dx);
        let boundary = empty_boundary();

        let eval_positions = vec![Point3::new(0.0, 0.0, 0.0)];
        let sample_positions = vec![Point3::new(1000.0, 1000.0, 1000.0)];
        let neighbor_list = build_neighbor_list(&eval_positions, &sample_positions, h);
        assert!(neighbor_list.get_neighbors(0).is_empty());

        let mut volume = vec![0.0];
        get_volume::<CubicBSpline3D>(
            &mut volume,
            &eval_positions,
            &neighbor_list,
            &sample_positions,
            &boundary,
            &params,
        );

        assert!(volume[0].is_infinite());
    }

    #[test]
    fn get_volume_supports_evaluation_at_points_distinct_from_the_samples() {
        // Mirrors real usage from `System::get_quantity_at_positions`:
        // `position_eval` can be an arbitrary query point set, entirely
        // distinct from `sample_positions`.
        let h = 1.0;
        let dx = 0.3;
        let params = make_params(h, dx);
        let boundary = empty_boundary();

        let sample_positions = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.3, 0.0, 0.0)];
        let eval_positions = vec![Point3::new(0.15, 0.0, 0.0)];

        let neighbor_list = build_neighbor_list(&eval_positions, &sample_positions, h);
        assert!(!neighbor_list.get_neighbors(0).is_empty());

        let mut volume = vec![0.0];
        get_volume::<CubicBSpline3D>(
            &mut volume,
            &eval_positions,
            &neighbor_list,
            &sample_positions,
            &boundary,
            &params,
        );

        let accu: f64 = neighbor_list
            .get_neighbors(0)
            .iter()
            .map(|&j| {
                params.rest_volume * kernel_weight(&sample_positions[j], &eval_positions[0], h)
            })
            .sum();
        assert!((volume[0] - params.rest_volume / accu).abs() < 1e-9);
    }

    #[test]
    fn get_volume_on_empty_input_does_not_panic() {
        let params = make_params(1.0, 0.3);
        let boundary = empty_boundary();
        let positions: Vec<Point3<f64>> = Vec::new();
        let neighbor_list = build_neighbor_list(&positions, &positions, 1.0);
        let mut volume: Vec<f64> = Vec::new();

        get_volume::<CubicBSpline3D>(
            &mut volume,
            &positions,
            &neighbor_list,
            &positions,
            &boundary,
            &params,
        );

        assert!(volume.is_empty());
    }

    // ─── Physical sanity check: rest-spacing lattice recovers rest values ──

    #[test]
    fn get_density_of_a_uniform_lattice_at_rest_spacing_approximates_rest_density() {
        // The defining physical contract of the SPH density estimator: for
        // a fluid sampled at its own rest spacing, evaluating density at
        // the center of a sufficiently large, densely packed neighborhood
        // should recover something close to the nominal rest density.
        let h = 0.2;
        let dx: f64 = 0.05; // h / dx == 4
        let rest_density = 1000.0;
        let rest_volume = dx.powi(3);
        let mass = rest_density * rest_volume;

        let positions = cubic_lattice(h, dx);
        let masses = vec![mass; positions.len()];
        let params = make_params(h, dx);

        let eval_positions = vec![Point3::origin()];
        let neighbor_list = build_neighbor_list(&eval_positions, &positions, h);

        let mut density = vec![0.0];
        get_density::<CubicBSpline3D>(
            &mut density,
            &eval_positions,
            &neighbor_list,
            &positions,
            &masses,
            &params,
        );

        let relative_error = (density[0] - rest_density).abs() / rest_density;
        assert!(
            relative_error < 0.15,
            "density estimate {} deviates from rest density {rest_density} by {:.1}%",
            density[0],
            relative_error * 100.0
        );
    }

    #[test]
    fn get_volume_of_a_uniform_lattice_at_rest_spacing_approximates_rest_volume() {
        // Analogous sanity check via the SPH partition-of-unity property
        // (Σ_j V_j W_ij ≈ 1 for a densely, evenly sampled neighborhood).
        let h = 0.2;
        let dx = 0.05;
        let params = make_params(h, dx);
        let boundary = empty_boundary();

        let positions = cubic_lattice(h, dx);
        let eval_positions = vec![Point3::origin()];
        let neighbor_list = build_neighbor_list(&eval_positions, &positions, h);

        let mut volume = vec![0.0];
        get_volume::<CubicBSpline3D>(
            &mut volume,
            &eval_positions,
            &neighbor_list,
            &positions,
            &boundary,
            &params,
        );

        let rest_volume = dx.powi(3);
        let relative_error = (volume[0] - rest_volume).abs() / rest_volume;
        assert!(
            relative_error < 0.15,
            "volume estimate {} deviates from rest volume {rest_volume} by {:.1}%",
            volume[0],
            relative_error * 100.0
        );
    }

    // ─── get_speed ──────────────────────────────────────────────────────

    #[test]
    fn get_speed_matches_manual_formula() {
        let h = 1.0;
        let params = make_params(h, 0.3);
        let boundary = empty_boundary();

        let positions = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.4, 0.0, 0.0),
            Point3::new(0.0, 0.4, 0.0),
        ];
        let velocities = vec![
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 2.0, 0.0),
            Vector3::new(0.0, 0.0, 3.0),
        ];
        let volumes = vec![0.02, 0.03, 0.025];
        let neighbor_list = build_neighbor_list(&positions, &positions, h);

        let mut speed = vec![0.0; positions.len()];
        get_speed::<CubicBSpline3D>(
            &mut speed,
            &positions,
            &neighbor_list,
            &positions,
            &velocities,
            &volumes,
            &boundary,
            &params,
        );

        for id in 0..positions.len() {
            let accu: Vector3<f64> = neighbor_list
                .get_neighbors(id)
                .iter()
                .map(|&j| {
                    velocities[j] * volumes[j] * kernel_weight(&positions[id], &positions[j], h)
                })
                .fold(Vector3::zeros(), |acc, v| acc + v);
            assert!((speed[id] - accu.norm()).abs() < 1e-9, "id={id}");
        }
    }

    // ─── get_density_error ──────────────────────────────────────────────

    #[test]
    fn get_density_error_ignores_neighbors_at_or_above_rest_volume() {
        let h = 1.0;
        let params = make_params(h, 0.3);
        let rest_volume = params.rest_volume;

        let positions = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.2, 0.0, 0.0),
            Point3::new(0.0, 0.2, 0.0),
        ];
        let volumes = vec![rest_volume, rest_volume * 1.5, rest_volume * 0.5];
        let neighbor_list = build_neighbor_list(&positions, &positions, h);

        let mut density_err = vec![0.0; positions.len()];
        get_density_error::<CubicBSpline3D>(
            &mut density_err,
            &positions,
            &neighbor_list,
            &positions,
            &volumes,
            &params,
        );

        let mut expected = 0.0;
        for &j in neighbor_list.get_neighbors(0) {
            if volumes[j] < rest_volume {
                let err = rest_volume / volumes[j] - 1.0;
                expected += err * volumes[j] * kernel_weight(&positions[0], &positions[j], h);
            }
        }
        expected *= 100.0;

        assert!((density_err[0] - expected).abs() < 1e-9);
    }

    // ─── get_pressure ───────────────────────────────────────────────────

    #[test]
    fn get_pressure_matches_manual_formula() {
        let h = 1.0;
        let params = make_params(h, 0.3);

        let positions = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.3, 0.0, 0.0),
            Point3::new(0.0, 0.3, 0.0),
        ];
        let volumes = vec![0.02, 0.025, 0.03];
        let pressures = vec![100.0, 200.0, 50.0];
        let neighbor_list = build_neighbor_list(&positions, &positions, h);

        let mut pressure = vec![0.0; positions.len()];
        get_pressure::<CubicBSpline3D>(
            &mut pressure,
            &positions,
            &neighbor_list,
            &positions,
            &volumes,
            &pressures,
            &params,
        );

        for id in 0..positions.len() {
            let expected: f64 = neighbor_list
                .get_neighbors(id)
                .iter()
                .map(|&j| {
                    pressures[j] * volumes[j] * kernel_weight(&positions[id], &positions[j], h)
                })
                .sum();
            assert!((pressure[id] - expected).abs() < 1e-9, "id={id}");
        }
    }

    // ─── get_kinetic_energy ─────────────────────────────────────────────

    #[test]
    fn get_kinetic_energy_matches_manual_formula() {
        let h = 1.0;
        let params = make_params(h, 0.3);

        let positions = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.3, 0.0, 0.0),
            Point3::new(0.0, 0.3, 0.0),
        ];
        let velocities = vec![
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 2.0, 0.0),
            Vector3::new(1.0, 1.0, 0.0),
        ];
        let volumes = vec![0.02, 0.025, 0.03];
        let masses = vec![0.5, 0.6, 0.7];
        let neighbor_list = build_neighbor_list(&positions, &positions, h);

        let mut kinetic_energy = vec![0.0; positions.len()];
        get_kinetic_energy::<CubicBSpline3D>(
            &mut kinetic_energy,
            &positions,
            &neighbor_list,
            &positions,
            &velocities,
            &volumes,
            &masses,
            &params,
        );

        for id in 0..positions.len() {
            let expected: f64 = neighbor_list
                .get_neighbors(id)
                .iter()
                .map(|&j| {
                    0.5 * masses[j]
                        * velocities[j].norm_squared()
                        * volumes[j]
                        * kernel_weight(&positions[id], &positions[j], h)
                })
                .sum();
            assert!((kinetic_energy[id] - expected).abs() < 1e-9, "id={id}");
        }
    }
}
