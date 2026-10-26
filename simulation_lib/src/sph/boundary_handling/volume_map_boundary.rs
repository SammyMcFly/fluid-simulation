//! Implicit frictional boundary handling via volume maps
use crate::for_each_mut;
use crate::iteration::for_each_collect;
use crate::neighbor_search::NeighborSearch;
use crate::render_info::{
    BoundaryMeshColoring, BoundarySampleColoring, BoundaryVisualization, RenderPose,
};
use crate::sph::boundary_handling::{
    Boundary, BoundaryCheckpoint, BoundaryConstructionError, BoundaryHandling, ForceOntoBoundary,
    RequestMode, RigidBodyMotion, RigidBodyMotionCheckpoint,
};
use crate::sph::kernel::{CubicBSpline3D, KernelFn};
use crate::sph::setup::input::{DynamicBoundaryDef, StaticBoundaryDef};
use crate::utilities::discretization::{
    CubicSerendipityDiscretization, DiscretizationError, EvaluationError,
    gauss_legendre_integrate_ball,
};
use crate::utilities::triangle_mesh::{MeshContainer, RenderMesh};
use crate::utilities::{Outer, euler_deg_to_quaternion, mean_fischer_burmeister};

use gauss_quad::GaussLegendre;
use nalgebra::{Isometry3, Matrix3, Point3, Vector3};
use num_traits::Zero;
use parry3d_f64::mass_properties::MassProperties;
use parry3d_f64::math::Pose;
use parry3d_f64::query::PointQuery;
use parry3d_f64::shape::{Shape, TriMesh};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::panic;
use std::slice::SliceIndex;
use std::sync::OnceLock;

fn ball_volume(radius: f64) -> f64 {
    4.0 / 3.0 * std::f64::consts::PI * radius.powi(3)
}

#[derive(Debug, Default, Clone)]
pub struct VolumeMapBoundary {
    boundaries: Vec<BoundaryType>,
    rest_density_grid_spacing: f64,
    kernel_support_radius: f64,
    p: Vec<f64>,                 // p_c^k
    v_err: Vec<f64>,             // V_c^err(p_c^k)
    lambda: Vec<Vector3<f64>>,   // lambda_c^k
    v_tang: Vec<Vector3<f64>>,   // v_c^tang(lambda_c^k)
    s_p: Vec<f64>,               // NNCG-search direction
    s_lambda: Vec<Vector3<f64>>, // NNCG-search direction
    extra_force: Vec<Vector3<f64>>,
    extra_torque: Vec<Vector3<f64>>,
    v_trial: Vec<Vector3<f64>>,
    omega_trial: Vec<Vector3<f64>>,
}

impl VolumeMapBoundary {
    const GAUSS_LEGENDRE_QUADRATURE_DEGREE: usize = 30;
    const CONTACT_HANDLING_GAUSS_LEGENDRE_QUADRATURE_DEGREE: usize = 8;
    /// Target contact-cell size, expressed as a multiple of
    /// `kernel_support_radius`.
    const CONTACT_CELL_SIZE_FACTOR: f64 = 0.5;

    const FRICTION_COEFFICIENT: f64 = 0.5;

    const RELAXATION: f64 = 0.5;
    const MIN_DIAGONAL_ELEMENT: f64 = 1e-35;
    const ITERATIONS_WARN_LIMIT: u32 = 100;
    const FB_TOLERANCE_PRESSURE: f64 = 1e-6;
    const FB_TOLERANCE_FRICTION: f64 = 1e-3;

    fn broad_phase(bodies: &[BoundaryType]) -> Vec<(usize, usize)> {
        #[cfg(not(feature = "parallel"))]
        let world_aabbs: Vec<WorldAABB> = bodies.iter().map(|b| b.world_aabb()).collect();
        #[cfg(feature = "parallel")]
        let world_aabbs: Vec<WorldAABB> = bodies.par_iter().map(|b| b.world_aabb()).collect();

        for_each_collect!(
            ref [bodies],
            ref [aabbs = world_aabbs],
            |i, body_i, pairs| {
                for j in (i + 1)..bodies.len() {
                    if !body_i.is_dynamic() && !bodies[j].is_dynamic() {
                        continue;
                    }
                    if aabbs[i].overlap(&aabbs[j]) {
                        pairs.push((i, j));
                    }
                }
            }
        )
    }

    /// Generates fresh contact cells for the current configuration.
    ///
    /// Per cell, this evaluates both SDFs at a small set of fixed
    /// quadrature points (never re-queried after this call — positions
    /// are frozen for the entire Jacobi/NNCG solve of this time step) and
    /// caches everything needed to build the pressure LCP.
    fn generate_contact_cells(&self) -> Vec<ContactCell> {
        let pairs = Self::broad_phase(&self.boundaries);

        // Cheap, sequential: build the flat (body_a, body_b, SubCell) task list.
        // No SDF evaluation happens yet, so sequential cost here is negligible.
        let mut tasks: Vec<(usize, usize, SubCell)> = Vec::new();
        for (body_a, body_b) in pairs {
            let aabb_a = self.boundaries[body_a].world_aabb();
            let aabb_b = self.boundaries[body_b].world_aabb();
            let Some(overlap) = aabb_a.intersection(&aabb_b) else {
                continue;
            };
            let cell_size = self.kernel_support_radius * Self::CONTACT_CELL_SIZE_FACTOR;
            for sub_cell in subdivide_overlap(&overlap, cell_size) {
                tasks.push((body_a, body_b, sub_cell));
            }
        }

        // Expensive part (SDF + gradient quadrature, incl. the pruning check):
        // runs in parallel over the FLAT task list -- much better load
        // balancing than parallelizing over `pairs` directly.
        for_each_collect!(
            ref [tasks],
            ref [boundaries = self.boundaries, ksr = self.kernel_support_radius],
            |_id, task, active_cells| {
                let (body_a, body_b, sub_cell) = task;
                let center = sub_cell.from + (sub_cell.to - sub_cell.from) * 0.5;
                let half_diag = (sub_cell.to - sub_cell.from).norm() * 0.5;

                // Safety factor accounts for the discretized SDF not perfectly satisfying
                // the eikonal condition (interpolation can locally overshoot ‖∇d‖ ≈ 1,
                // observed up to ~2x elsewhere in this codebase — see find_boundary_samples).
                const GRADIENT_OVERSHOOT_SAFETY_FACTOR: f64 = 2.0;
                let bound = GRADIENT_OVERSHOOT_SAFETY_FACTOR * half_diag;

                // No early return/continue: both SDF lookups and the pruning
                // check are threaded through nested `if let`/`if` instead, so
                // this closure body works unchanged whether it's the body of
                // `.for_each(...)` (sequential) or `.fold(...)` (parallel) --
                // see note on why a bare `return;` is NOT safe here.
                if let (Ok(phi_a_center), Ok(phi_b_center)) = (
                    boundaries[*body_a].signed_distance(&center),
                    boundaries[*body_b].signed_distance(&center),
                ) {
                    // A contact cell can only contribute if it dips below BOTH
                    // surfaces (phi < 0 for both) somewhere inside it — s_a/s_b
                    // are exactly zero whenever phi_a/phi_b >= 0, independent
                    // of kernel_support_radius.
                    if phi_a_center - bound < 0. && phi_b_center - bound < 0.
                        && let Some(cell) = build_contact_cell(
                            *body_a,
                            *body_b,
                            sub_cell,
                            boundaries,
                            *ksr,
                            Self::CONTACT_HANDLING_GAUSS_LEGENDRE_QUADRATURE_DEGREE,
                        )
                    {
                        active_cells.push(cell);
                    }
                }
            }
        )
    }

    fn converged(convergence_measurement: f64, tolerance: f64, solver_iteration: u32) -> bool {
        // println!(
        //     "convergence_measurement: {}, tolerance: {}, solver_iteration: {}",
        //     convergence_measurement, tolerance, solver_iteration
        // );
        if solver_iteration >= Self::ITERATIONS_WARN_LIMIT {
            tracing::warn!("Solver exceeded {} iterations", Self::ITERATIONS_WARN_LIMIT);
        }
        if convergence_measurement < tolerance {
            return true;
        }
        false
    }

    /// Solves the rigid-rigid contact pressure LCP for all currently
    /// active cells via NNCG-accelerated Jacobi iteration (Probst &
    /// Teschner, Algorithm 2, extended with the SDF-quadrature-based
    /// `V_c^err`), and commits the resulting contact forces/torques onto
    /// the involved `RigidBodyMotion`s.
    ///
    /// Note: this only *adds* to `motion.force`/`motion.torque` — it does
    /// NOT call `step_forward_in_time`. The caller (`step_forward_in_time`
    /// of this `BoundaryHandling` impl) must still integrate afterwards.
    fn solve_for_p_c(
        &mut self,
        active_cells: &mut [ContactCell],
        dt: f64,
        fb_tolerance_pressure: f64,
        fb_tolerance_friction: f64,
    ) {
        let n = active_cells.len();
        self.p = vec![0.0; n];
        self.v_err = vec![0.0; n];
        self.lambda = vec![Vector3::zeros(); n];
        self.v_tang = vec![Vector3::zeros(); n];
        self.s_p = vec![0.0; n];
        if n == 0 {
            return;
        }

        let n_bodies = self.boundaries.len();
        for_each_mut!(
            mut [active_cells],
            ref [boundaries = self.boundaries, dt_ref = dt],
            |_id, cell| {
                compute_diagonal(cell, boundaries, *dt_ref);
            }
        );
        let body_cell_index = BodyCellIndex::build(active_cells, n_bodies);

        self.extra_force.clear();
        self.extra_force.resize(n_bodies, Vector3::zeros());
        self.extra_torque.clear();
        self.extra_torque.resize(n_bodies, Vector3::zeros());
        self.v_trial.clear();
        self.v_trial.resize(n_bodies, Vector3::zeros());
        self.omega_trial.clear();
        self.omega_trial.resize(n_bodies, Vector3::zeros());
        accumulate_extra_forces(
            active_cells,
            &body_cell_index,
            &self.p,
            &self.lambda,
            &self.boundaries,
            &mut self.extra_force,
            &mut self.extra_torque,
        );
        compute_trial_velocities(
            &self.boundaries,
            dt,
            &self.extra_force,
            &self.extra_torque,
            &mut self.v_trial,
            &mut self.omega_trial,
        );
        for_each_mut!(
            mut [self.v_err],
            ref [cells = active_cells, v_trial = self.v_trial, omega_trial = self.omega_trial, dt_ref = dt],
            |id, v_err| {
                *v_err = predicted_volume_error(&cells[id], *dt_ref, v_trial, omega_trial);
            }
        );
        for_each_mut!(
            mut [self.v_tang],
            ref [cells = active_cells, v_trial = self.v_trial, omega_trial = self.omega_trial, boundaries = self.boundaries],
            |id, v_tang| {
                *v_tang = predicted_tangential_velocity(&cells[id], boundaries, v_trial, omega_trial);
            }
        );

        let (p1, lambda1) = projected_jacobi(
            active_cells,
            &body_cell_index,
            &self.p,
            &self.v_err,
            &self.lambda,
            &self.v_tang,
            n_bodies,
            Self::MIN_DIAGONAL_ELEMENT,
            Self::RELAXATION,
            Self::FRICTION_COEFFICIENT,
        );
        let mut grad_p: Vec<f64> = p1.iter().zip(&self.p).map(|(n, o)| -(n - o)).collect();
        let mut grad_lambda: Vec<Vector3<f64>> = lambda1
            .iter()
            .zip(&self.lambda)
            .map(|(n, o)| -(n - o))
            .collect();
        self.s_p = grad_p.iter().map(|g| -g).collect();
        self.s_lambda = grad_lambda.iter().map(|g| -g).collect();
        self.p = p1;
        self.lambda = lambda1;
        let mut prev_grad_p_norm_squared: f64 = grad_p.iter().map(|g| g * g).sum();
        let mut prev_grad_lambda_norm_squared: f64 = grad_lambda.iter().map(|g| g.dot(g)).sum();

        accumulate_extra_forces(
            active_cells,
            &body_cell_index,
            &self.p,
            &self.lambda,
            &self.boundaries,
            &mut self.extra_force,
            &mut self.extra_torque,
        );
        compute_trial_velocities(
            &self.boundaries,
            dt,
            &self.extra_force,
            &self.extra_torque,
            &mut self.v_trial,
            &mut self.omega_trial,
        );
        for_each_mut!(
            mut [self.v_err],
            ref [cells = active_cells, v_trial = self.v_trial, omega_trial = self.omega_trial, dt_ref = dt],
            |id, v| {
                *v = predicted_volume_error(&cells[id], *dt_ref, v_trial, omega_trial);
            }
        );
        for_each_mut!(
            mut [self.v_tang],
            ref [cells = active_cells, v_trial = self.v_trial, omega_trial = self.omega_trial, boundaries = self.boundaries],
            |id, v_tang| {
                *v_tang = predicted_tangential_velocity(&cells[id], boundaries, v_trial, omega_trial);
            }
        );

        let mut preconditioned_p = vec![0.0; n];
        let mut v_tang_norm = vec![0.0; n];
        let mut preconditioned_deviation_from_max_dissipation = vec![0.0; n];
        preconditioned_p
            .iter_mut()
            .zip(self.p.iter().zip(active_cells.iter()))
            .for_each(|(pp, (p, cell))| {
                *pp = p * cell.a_cc_pressure.max(Self::MIN_DIAGONAL_ELEMENT)
            });
        v_tang_norm
            .iter_mut()
            .zip(self.v_tang.iter())
            .for_each(|(v_t_n, v_t)| *v_t_n = v_t.norm());
        preconditioned_deviation_from_max_dissipation
            .iter_mut()
            .zip(
                active_cells
                    .iter()
                    .zip(self.p.iter().zip(self.lambda.iter())),
            )
            .for_each(|(dev, (cell, (p, lambda)))| {
                *dev = cell.a_cc_friction.max(Self::MIN_DIAGONAL_ELEMENT)
                    * (Self::FRICTION_COEFFICIENT * p * cell.f_a.norm() - lambda.norm());
            });

        let mut solver_iteration = 0;
        while !(Self::converged(
            mean_fischer_burmeister(&preconditioned_p, &self.v_err),
            fb_tolerance_pressure,
            solver_iteration,
        ) && Self::converged(
            mean_fischer_burmeister(&v_tang_norm, &preconditioned_deviation_from_max_dissipation),
            fb_tolerance_friction,
            solver_iteration,
        )) {
            // let merit: f64 = self
            //     .v_err
            //     .iter()
            //     .map(|v| v.min(0.0).powi(2))
            //     .sum::<f64>()
            //     .sqrt();
            // tracing::info!(solver_iteration, merit, "merit");
            let (p_jacobi, lambda_jacobi) = projected_jacobi(
                active_cells,
                &body_cell_index,
                &self.p,
                &self.v_err,
                &self.lambda,
                &self.v_tang,
                n_bodies,
                Self::MIN_DIAGONAL_ELEMENT,
                Self::RELAXATION,
                Self::FRICTION_COEFFICIENT,
            );
            grad_p = p_jacobi
                .iter()
                .zip(&self.p)
                .map(|(n, o)| -(n - o))
                .collect();
            grad_lambda = lambda_jacobi
                .iter()
                .zip(&self.lambda)
                .map(|(n, o)| -(n - o))
                .collect();
            let grad_p_norm_squared: f64 = grad_p.iter().map(|g| g * g).sum();
            let grad_lambda_norm_squared: f64 = grad_lambda.iter().map(|g| g.dot(g)).sum();

            // if grad_norm_squared.sqrt() < Self::GRADIENT_TOLERANCE {
            //     self.p = p_jacobi;
            //     break;
            // }
            {
                let beta = (grad_p_norm_squared + grad_lambda_norm_squared)
                    / (prev_grad_p_norm_squared + prev_grad_lambda_norm_squared)
                        .max(f64::MIN_POSITIVE);
                // tracing::info!("beta: {}", beta);
                if beta > 1.0 {
                    // tracing::info!("NNCG reset at iteration: {}", solver_iteration);
                    self.s_p = vec![0.0; n];
                    self.s_lambda = vec![Vector3::zeros(); n];
                    self.p = p_jacobi;
                    self.lambda = lambda_jacobi;
                } else {
                    // self.p = p_jacobi
                    //     .iter()
                    //     .zip(&self.s)
                    //     .map(|(r, si)| r + beta * si)
                    //     .collect();
                    // self.s = grad
                    //     .iter()
                    //     .zip(&self.s)
                    //     .map(|(g, si)| beta * si - g)
                    //     .collect();
                    let p_nncg: Vec<f64> = p_jacobi
                        .iter()
                        .zip(&self.s_p)
                        .map(|(p_jacobi, search_direction)| p_jacobi + beta * search_direction)
                        .collect();
                    let lambda_nncg: Vec<Vector3<f64>> = lambda_jacobi
                        .iter()
                        .zip(&self.s_lambda)
                        .map(|(lambda_jacobi, search_direction)| {
                            lambda_jacobi + beta * search_direction
                        })
                        .collect();

                    // let mut n_fallback = 0usize;
                    // for p in &p_nncg {
                    //     if *p < 0.0 {
                    //         n_fallback += 1;
                    //         // ... existing fallback logic
                    //     }
                    // }
                    // tracing::info!(solver_iteration, n_fallback, beta, "nncg step");

                    // Pressure
                    //
                    // Apply proximal operator: Clamp to set of allowed pressure (p_c >= 0)

                    // Per-contact feasibility: only the contacts that actually overshoot fall
                    // back to the plain projected-Jacobi value (with their own search direction
                    // zeroed); the rest keep the accelerated step and conjugate direction.
                    let mut p_out = vec![0.0; n];
                    let mut s_p_out = vec![0.0; n];
                    for_each_mut!(
                        mut [p_out, s_p_out],
                        ref [p_jacobi = p_jacobi, p_nncg = p_nncg, grad = grad_p, s_prev = self.s_p, beta_ref = beta],
                        |i, p_out, s_p_out| {
                            if p_nncg[i] < 0.0 {
                                *p_out = p_jacobi[i];
                                // *p_out = 0.0;
                                *s_p_out = 0.0;
                            } else {
                                *p_out = p_nncg[i];
                                *s_p_out = *beta_ref * s_prev[i] - grad[i];
                            }
                        }
                    );
                    // Friction
                    //
                    // Apply proximal operator -> project and clamp to friction cone
                    let mut lambda_out = vec![Vector3::zeros(); n];
                    let mut s_lambda_out = vec![Vector3::zeros(); n];
                    for_each_mut!(
                        mut [lambda_out, s_lambda_out],
                        ref [lambda_jacobi = lambda_jacobi, lambda_nncg = lambda_nncg, grad = grad_lambda, s_prev = self.s_lambda, beta_ref = beta, cells = active_cells, p = self.p],
                        |i, l_out, s_l_out| {
                            let tangential_proj = cells[i].tangential_projection * lambda_nncg[i];
                            let tangential_proj_mag = tangential_proj.norm();
                            let lambda_max_mag = Self::FRICTION_COEFFICIENT * p[i] * cells[i].f_a.norm();
                            if tangential_proj_mag > lambda_max_mag {
                                *l_out = lambda_jacobi[i];
                                // *l_out = lambda_max_mag * tangential_proj / tangential_proj_mag;
                                *s_l_out = Vector3::zeros();
                            } else {
                                *l_out = tangential_proj;
                                *s_l_out = *beta_ref * s_prev[i] - grad[i];
                            }
                        }
                    );

                    self.p = p_out;
                    self.s_p = s_p_out;
                    self.lambda = lambda_out;
                    self.s_lambda = s_lambda_out;
                }

                //                 let beta = ((grad_p_norm_squared + grad_lambda_norm_squared)
                //                     / (prev_grad_p_norm_squared + prev_grad_lambda_norm_squared)
                //                         .max(f64::MIN_POSITIVE))
                //                 .clamp(0.0, 1.0);
                //                 // let beta = 0.;
                //
                //                 let p_nncg: Vec<f64> = p_jacobi
                //                     .iter()
                //                     .zip(&self.s_p)
                //                     .map(|(p_jacobi, search_direction)| p_jacobi + beta * search_direction)
                //                     .collect();
                //                 let lambda_nncg: Vec<Vector3<f64>> = lambda_jacobi
                //                     .iter()
                //                     .zip(&self.s_lambda)
                //                     .map(|(lambda_jacobi, search_direction)| {
                //                         lambda_jacobi + beta * search_direction
                //                     })
                //                     .collect();
                //
                //                 // let mut n_fallback = 0usize;
                //                 // for p in &p_nncg {
                //                 //     if *p < 0.0 {
                //                 //         n_fallback += 1;
                //                 //         // ... existing fallback logic
                //                 //     }
                //                 // }
                //                 // tracing::info!(solver_iteration, n_fallback, beta, "nncg step");
                //
                //                 // Pressure
                //                 //
                //                 // Apply proximal operator: Clamp to set of allowed pressure (p_c >= 0)
                //                 //
                //                 // Per-contact feasibility: only the contacts that actually overshoot fall
                //                 // back to the plain projected-Jacobi value (with their own search direction
                //                 // zeroed); the rest keep the accelerated step and conjugate direction.
                //                 let mut p_out = vec![0.0; n];
                //                 let mut s_out = vec![0.0; n];
                //                 for_each_mut!(
                //                     mut [p_out, s_out],
                //                     ref [p_jacobi = p_jacobi, p_nncg = p_nncg, grad = grad_p, s_prev = self.s_p, beta_ref = beta],
                //                     |i, p_out, s_out| {
                //                         if p_nncg[i] < 0.0 {
                //                             // *p_out = p_jacobi[i];
                //                             *p_out = 0.0;
                //                             *s_out = 0.0;
                //                         } else {
                //                             *p_out = p_nncg[i];
                //                             *s_out = *beta_ref * s_prev[i] - grad[i];
                //                         }
                //                     }
                //                 );
                //                 // Friction
                //                 //
                //                 // Apply proximal operator -> project and clamp to friction cone
                //                 let mut lambda_out = vec![Vector3::zeros(); n];
                //                 let mut s_lambda_out = vec![Vector3::zeros(); n];
                //                 for_each_mut!(
                //                     mut [lambda_out, s_lambda_out],
                //                     ref [lambda_jacobi = lambda_jacobi, lambda_nncg = lambda_nncg, grad = grad_lambda, s_prev = self.s_lambda, beta_ref = beta, cells = active_cells, p = self.p],
                //                     |i, l_out, s_l_out| {
                //                         let tangential_proj = cells[i].tangential_projection * lambda_nncg[i];
                //                         let tangential_proj_mag = tangential_proj.norm();
                //                         let lambda_max_mag = Self::FRICTION_COEFFICIENT * p[i] * cells[i].f_a.norm();
                //                         if tangential_proj_mag > lambda_max_mag {
                //                             // *l_out = lambda_jacobi[i];
                //                             *l_out = lambda_max_mag * tangential_proj / tangential_proj_mag;
                //                             *s_l_out = Vector3::zeros();
                //                         } else {
                //                             *l_out = tangential_proj;
                //                             *s_l_out = *beta_ref * s_prev[i] - grad[i];
                //                         }
                //                     }
                //                 );
                //
                //                 self.p = p_out;
                //                 self.s_p = s_out;
                //                 self.lambda = lambda_out;
                //                 self.s_lambda = s_lambda_out;
            }
            prev_grad_p_norm_squared = grad_p_norm_squared;
            prev_grad_lambda_norm_squared = grad_lambda_norm_squared;
            // let min_pressure = self.p.iter().copied().fold(f64::INFINITY, f64::min);
            // tracing::info!(min_pressure, "Minimum pressure after NNCG extrapolation");

            accumulate_extra_forces(
                active_cells,
                &body_cell_index,
                &self.p,
                &self.lambda,
                &self.boundaries,
                &mut self.extra_force,
                &mut self.extra_torque,
            );
            compute_trial_velocities(
                &self.boundaries,
                dt,
                &self.extra_force,
                &self.extra_torque,
                &mut self.v_trial,
                &mut self.omega_trial,
            );
            for_each_mut!(
                mut [self.v_err],
                ref [cells = active_cells, v_trial = self.v_trial, omega_trial = self.omega_trial, dt_ref = dt],
                |id, v| {
                    *v = predicted_volume_error(&cells[id], *dt_ref, v_trial, omega_trial);
                }
            );
            for_each_mut!(
                mut [self.v_tang],
                ref [cells = active_cells, v_trial = self.v_trial, omega_trial = self.omega_trial, boundaries = self.boundaries],
                |id, v_tang| {
                    *v_tang = predicted_tangential_velocity(&cells[id], boundaries, v_trial, omega_trial);
                }
            );

            preconditioned_p
                .iter_mut()
                .zip(self.p.iter().zip(active_cells.iter()))
                .for_each(|(pp, (p, cell))| {
                    *pp = p * cell.a_cc_pressure.max(Self::MIN_DIAGONAL_ELEMENT)
                });
            v_tang_norm
                .iter_mut()
                .zip(self.v_tang.iter())
                .for_each(|(v_n, v)| *v_n = v.norm());
            preconditioned_deviation_from_max_dissipation
                .iter_mut()
                .zip(
                    active_cells
                        .iter()
                        .zip(self.p.iter().zip(self.lambda.iter())),
                )
                .for_each(|(dev, (cell, (p, lambda)))| {
                    *dev = cell.a_cc_friction.max(Self::MIN_DIAGONAL_ELEMENT)
                        * (Self::FRICTION_COEFFICIENT * p * cell.f_a.norm() - lambda.norm());
                });

            solver_iteration += 1;
        }
        tracing::info!("final solver_iteration: {}", solver_iteration);

        // Commit: fold the converged contact pressure into each body's
        // force/torque accumulator. The subsequent `step_forward_in_time`
        // call (in the caller) integrates this together with whatever
        // explicit forces were already accumulated this step.
        accumulate_extra_forces(
            active_cells,
            &body_cell_index,
            &self.p,
            &self.lambda,
            &self.boundaries,
            &mut self.extra_force,
            &mut self.extra_torque,
        );
        for (i, b) in self.boundaries.iter_mut().enumerate() {
            if let Some(m) = b.motion_mut() {
                m.add_force_and_torque(self.extra_force[i], self.extra_torque[i]);
            }
        }
    }
}

/// For each body, the set of active contact cells touching it (as either
/// body_a or body_b), built once per timestep right after
/// `generate_contact_cells`. Reused, unchanged, across every subsequent
/// Jacobi/NNCG iteration -- turns the per-iteration scatter-add in
/// `accumulate_extra_forces`/`projected_jacobi` into an embarrassingly
/// parallel per-body gather.
#[derive(Debug, Default, Clone)]
struct BodyCellIndex {
    /// Flattened (cell_index, is_body_a) pairs.
    entries: Vec<(usize, bool)>,
    /// `offsets[body] .. offsets[body + 1]` indexes into `entries`.
    offsets: Vec<usize>,
}

impl BodyCellIndex {
    fn build(active_cells: &[ContactCell], n_bodies: usize) -> Self {
        let mut per_body: Vec<Vec<(usize, bool)>> = vec![Vec::new(); n_bodies];
        for (cell_idx, cell) in active_cells.iter().enumerate() {
            per_body[cell.body_a].push((cell_idx, true));
            per_body[cell.body_b].push((cell_idx, false));
        }
        let mut entries = Vec::with_capacity(active_cells.len() * 2);
        let mut offsets = Vec::with_capacity(n_bodies + 1);
        offsets.push(0);
        for list in &per_body {
            entries.extend_from_slice(list);
            offsets.push(entries.len());
        }
        Self { entries, offsets }
    }

    #[inline]
    fn entries_for(&self, body: usize) -> &[(usize, bool)] {
        &self.entries[self.offsets[body]..self.offsets[body + 1]]
    }
}

#[derive(Debug, Clone)]
struct ContactCell {
    body_a: usize,
    body_b: usize,
    /// Frozen for the entire solve of this time step.
    quad_points: Vec<QuadPointData>,
    /// `V_c(t)`, computed once at generation time.
    volume: f64,
    // Calculated once per timestep by `compute_diagonal`:
    center: Point3<f64>,
    f_a: Vector3<f64>,
    tau_a_u: Vector3<f64>,
    f_b: Vector3<f64>,
    tau_b_u: Vector3<f64>,
    tangential_projection: Matrix3<f64>,
    a_cc_pressure: f64,
    a_cc_friction: f64,
}

/// Evaluates both SDFs at the quadrature points of a single sub-cell,
/// builds a [`ContactCell`] if the overlap is non-trivial, or returns
/// `None` otherwise. Quadrature points with no overlap contribution
/// (`s_a == 0 || s_b == 0`) are discarded, not stored.
fn build_contact_cell(
    body_a: usize,
    body_b: usize,
    sub: &SubCell,
    boundaries: &[BoundaryType],
    kernel_support_radius: f64,
    quadrature_degree: usize,
) -> Option<ContactCell> {
    let mut v_c = 0.0;
    let mut quad_points = Vec::with_capacity(quadrature_degree.pow(3));
    let scaling = 2.0;

    for (x_g, w_g) in quadrature_points(&sub.from, &sub.to, quadrature_degree) {
        let Ok(phi_a) = boundaries[body_a].signed_distance(&x_g) else {
            continue;
        };
        let Ok(phi_b) = boundaries[body_b].signed_distance(&x_g) else {
            continue;
        };

        let s_a = cubic_extension_fn(
            scaling * phi_a + kernel_support_radius,
            kernel_support_radius,
        );
        let s_b = cubic_extension_fn(
            scaling * phi_b + kernel_support_radius,
            kernel_support_radius,
        );
        if s_a <= 0.0 || s_b <= 0.0 {
            continue;
        }

        let Ok(grad_a) = boundaries[body_a].signed_distance_gradient(&x_g) else {
            continue;
        };
        let Ok(grad_b) = boundaries[body_b].signed_distance_gradient(&x_g) else {
            continue;
        };

        v_c += w_g * s_a * s_b;
        quad_points.push(QuadPointData {
            pos: x_g,
            weight: w_g,
            grad_phi_a_world: grad_a,
            grad_phi_b_world: grad_b,
            s_a,
            s_b,
            s_prime_a: scaling
                * cubic_extension_fn_derivative(
                    scaling * phi_a + kernel_support_radius,
                    kernel_support_radius,
                ),
            s_prime_b: scaling
                * cubic_extension_fn_derivative(
                    scaling * phi_b + kernel_support_radius,
                    kernel_support_radius,
                ),
        });
    }

    if v_c <= 0.0 || quad_points.is_empty() {
        return None;
    }

    Some(ContactCell {
        body_a,
        body_b,
        quad_points,
        volume: v_c,
        center: Point3::origin(),
        f_a: Vector3::zeros(),
        tau_a_u: Vector3::zeros(),
        f_b: Vector3::zeros(),
        tau_b_u: Vector3::zeros(),
        tangential_projection: Matrix3::identity(),
        a_cc_pressure: 0.0,
        a_cc_friction: 0.0,
    })
}

/// All quantities derivable purely from geometry at a single quadrature
/// point, cached once per time step — positions are frozen for the
/// entire solve, only velocities vary across Jacobi/NNCG iterations.
#[derive(Debug, Clone, Copy)]
struct QuadPointData {
    pos: Point3<f64>,
    weight: f64,
    grad_phi_a_world: Vector3<f64>,
    grad_phi_b_world: Vector3<f64>,
    s_a: f64,
    s_b: f64,
    s_prime_a: f64,
    s_prime_b: f64,
}

fn gauss_nodes_weights(order: usize) -> &'static (Vec<f64>, Vec<f64>) {
    static CACHE: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();
    CACHE.get_or_init(|| {
        let quad = GaussLegendre::new(order.try_into().unwrap());
        let nodes: Vec<f64> = quad.nodes().copied().collect();
        let weights: Vec<f64> = quad.weights().copied().collect();
        (nodes, weights)
    })
}

fn quadrature_points(
    from: &Point3<f64>,
    to: &Point3<f64>,
    degree: usize,
) -> Vec<(Point3<f64>, f64)> {
    let half = (to - from) * 0.5;
    let center = from + half;

    let (nodes, weights) = gauss_nodes_weights(degree);
    let n = nodes.len();

    let mut out = Vec::with_capacity(n * n * n);
    for i in 0..n {
        let (nx, wx) = (nodes[i], weights[i]);
        for j in 0..n {
            let (ny, wy) = (nodes[j], weights[j]);
            for k in 0..n {
                let (nz, wz) = (nodes[k], weights[k]);
                let pos = center + Vector3::new(nx * half.x, ny * half.y, nz * half.z);
                let weight = wx * wy * wz * half.x * half.y * half.z;
                out.push((pos, weight));
            }
        }
    }
    out
}

/// Precomputes the unit sensitivity vectors `f_A, τ_A^u, f_B, τ_B^u` and
/// the diagonal element `a_cc` for a single contact cell. Called once per
/// time step, before any Jacobi iteration.
fn compute_diagonal(cell: &mut ContactCell, boundaries: &[BoundaryType], dt: f64) {
    let com_a = boundaries[cell.body_a].motion().map(|m| m.center_of_mass());
    let com_b = boundaries[cell.body_b].motion().map(|m| m.center_of_mass());

    let mut x_c_times_v_c = Point3::origin();
    let mut f_a = Vector3::zeros();
    let mut tau_a_u = Vector3::zeros();
    let mut f_b = Vector3::zeros();
    let mut tau_b_u = Vector3::zeros();

    for qp in &cell.quad_points {
        let w_g = qp.weight * qp.s_a * qp.s_b;
        x_c_times_v_c += w_g * qp.pos.coords;

        let w_a = qp.weight * qp.s_prime_a * qp.s_b;
        f_a += w_a * qp.grad_phi_a_world;
        if let Some(c) = com_a {
            tau_a_u += w_a * (qp.pos - c).cross(&qp.grad_phi_a_world);
        }

        let w_b = qp.weight * qp.s_prime_b * qp.s_a;
        f_b += w_b * qp.grad_phi_b_world;
        if let Some(c) = com_b {
            tau_b_u += w_b * (qp.pos - c).cross(&qp.grad_phi_b_world);
        }
    }

    cell.center = x_c_times_v_c / cell.volume;
    cell.f_a = f_a;
    cell.tau_a_u = tau_a_u;
    cell.f_b = f_b;
    cell.tau_b_u = tau_b_u;
    let normal_a = cell.f_a / cell.f_a.norm().max(1e-15);
    cell.tangential_projection = Matrix3::identity() - normal_a.outer(&normal_a);

    // A static body (motion() == None) contributes zero compliance —
    // equivalent to infinite mass / zero inverse inertia.
    let term_a = boundaries[cell.body_a]
        .motion()
        .map(|m| f_a.dot(&f_a) / m.mass() + tau_a_u.dot(&(m.inv_inertia_world() * tau_a_u)))
        .unwrap_or(0.0);
    let term_b = boundaries[cell.body_b]
        .motion()
        .map(|m| f_b.dot(&f_b) / m.mass() + tau_b_u.dot(&(m.inv_inertia_world() * tau_b_u)))
        .unwrap_or(0.0);

    cell.a_cc_pressure = dt * dt * (term_a + term_b);
    // tracing::info!("a_cc: {}", cell.a_cc);

    let term_a = boundaries[cell.body_a]
        .motion()
        .map(|m| {
            Matrix3::identity() / m.mass()
                - (cell.center - m.center_of_mass()).cross_matrix()
                    * m.inv_inertia_world()
                    * (cell.center - m.center_of_mass()).cross_matrix()
        })
        .unwrap_or(Matrix3::zeros());
    let term_b = boundaries[cell.body_b]
        .motion()
        .map(|m| {
            Matrix3::identity() / m.mass()
                - (cell.center - m.center_of_mass()).cross_matrix()
                    * m.inv_inertia_world()
                    * (cell.center - m.center_of_mass()).cross_matrix()
        })
        .unwrap_or(Matrix3::zeros());

    cell.a_cc_friction = 1. / 3. * (dt * cell.tangential_projection * (term_a + term_b)).trace();
}

/// Accumulates, per body, the total contact force/torque implied by the
/// current pressure iterate `p` across all active cells touching that
/// body (as either `body_a` or `body_b`).
// fn accumulate_extra_forces(
//     active_cells: &[ContactCell],
//     p: &[f64],
//     extra_force: &mut [Vector3<f64>],
//     extra_torque: &mut [Vector3<f64>],
// ) {
//     for f in extra_force.iter_mut() {
//         *f = Vector3::zeros();
//     }
//     for t in extra_torque.iter_mut() {
//         *t = Vector3::zeros();
//     }
//     for (cell, &p_c) in active_cells.iter().zip(p.iter()) {
//         extra_force[cell.body_a] += p_c * cell.f_a;
//         extra_torque[cell.body_a] += p_c * cell.tau_a_u;
//         extra_force[cell.body_b] += p_c * cell.f_b;
//         extra_torque[cell.body_b] += p_c * cell.tau_b_u;
//     }
// }
fn accumulate_extra_forces(
    active_cells: &[ContactCell],
    body_cell_index: &BodyCellIndex,
    p: &[f64],
    lambda: &[Vector3<f64>],
    boundaries: &[BoundaryType],
    extra_force: &mut [Vector3<f64>],
    extra_torque: &mut [Vector3<f64>],
) {
    for_each_mut!(
        mut [extra_force, extra_torque],
        ref [cells = active_cells, p = p, lambda = lambda, boundaries = boundaries, idx = body_cell_index],
        |body, force, torque| {
            let mut f = Vector3::zeros();
            let mut t = Vector3::zeros();
            for &(cell_idx, is_a) in idx.entries_for(body) {
                let c = &cells[cell_idx];
                let com_a = boundaries[c.body_a].motion().map(|m| m.center_of_mass());
                let com_b = boundaries[c.body_b].motion().map(|m| m.center_of_mass());
                let p_c = p[cell_idx];
                let lambda_c = lambda[cell_idx];
                if is_a {
                    f += p_c * c.f_a + lambda_c;
                    if let Some(com_a) = com_a {
                        t += p_c * c.tau_a_u + (c.center - com_a).cross(&lambda_c);
                    }
                } else {
                    f += p_c * c.f_b - lambda_c;
                    if let Some(com_b) = com_b {
                        t += p_c * c.tau_b_u - (c.center - com_b).cross(&lambda_c);
                    }
                }
            }
            *force = f;
            *torque = t;
        }
    );
}

fn compute_trial_velocities(
    boundaries: &[BoundaryType],
    dt: f64,
    extra_force: &[Vector3<f64>],
    extra_torque: &[Vector3<f64>],
    v_trial: &mut [Vector3<f64>],
    omega_trial: &mut [Vector3<f64>],
) {
    for_each_mut!(
        mut [v_trial, omega_trial],
        ref [boundary = boundaries, extra_force = extra_force, extra_torque = extra_torque, dt_ref = dt],
        |id, v, omega| {
            let (vv, oo) = match boundary[id].motion() {
                None => (Vector3::zeros(), Vector3::zeros()),
                Some(m) => (
                    m.trial_velocity(*dt_ref, &extra_force[id]),
                    m.trial_angular_velocity(*dt_ref, &extra_torque[id]),
                ),
            };
            *v = vv;
            *omega = oo;
        }
    );
}

/// `V_c^err(t+Δt) = -V_c(t) - Δt·V̇_c(t+Δt)` for a single cell, given the
/// per-body additional contact force/torque implied by the current
/// pressure iterate.
#[inline]
fn predicted_volume_error(
    cell: &ContactCell,
    dt: f64,
    trial_velocity: &[Vector3<f64>],
    trial_angular_velocity: &[Vector3<f64>],
) -> f64 {
    -cell.volume
        + dt * (trial_velocity[cell.body_a].dot(&cell.f_a)
            + trial_angular_velocity[cell.body_a].dot(&cell.tau_a_u)
            + trial_velocity[cell.body_b].dot(&cell.f_b)
            + trial_angular_velocity[cell.body_b].dot(&cell.tau_b_u))
}

/// `v_c^tang(t+Δt)` for a single cell.
#[inline]
fn predicted_tangential_velocity(
    cell: &ContactCell,
    boundaries: &[BoundaryType],
    trial_velocity: &[Vector3<f64>],
    trial_angular_velocity: &[Vector3<f64>],
) -> Vector3<f64> {
    let com_a = boundaries[cell.body_a].motion().map(|m| m.center_of_mass());
    let com_b = boundaries[cell.body_b].motion().map(|m| m.center_of_mass());
    if let Some(com_a) = com_a
        && let Some(com_b) = com_b
    {
        let v_rel = trial_velocity[cell.body_a]
            + trial_angular_velocity[cell.body_a].cross(&(cell.center - com_a))
            - trial_velocity[cell.body_b]
            - trial_angular_velocity[cell.body_b].cross(&(cell.center - com_b));
        cell.tangential_projection * v_rel
    } else {
        Vector3::zeros()
    }
}

/// Projected relaxed Jacobi step (ρ_c^{k+1} = max(0, ρ_c^k + α_c·V_c^err)),
/// with the `n_max` normalization needed since a contact cell couples
/// *two* bodies rather than linking a single particle to one body.
fn projected_jacobi(
    active_cells: &[ContactCell],
    body_cell_index: &BodyCellIndex,
    p: &[f64],
    v_err: &[f64],
    lambda: &[Vector3<f64>],
    v_tang: &[Vector3<f64>],
    n_bodies: usize,
    min_diagonal_element: f64,
    relaxation: f64,
    friction_coefficient: f64,
) -> (Vec<f64>, Vec<Vector3<f64>>) {
    let mut n_active = vec![0u32; n_bodies];
    for_each_mut!(
        mut [n_active],
        ref [idx = body_cell_index, p = p, v_err = v_err],
        |body, n| {
            *n = idx.entries_for(body)
                .iter()
                .filter(|&&(cell_idx, _)| v_err[cell_idx] < 0.0 || p[cell_idx] > 0.0)
                .count() as u32;
        }
    );

    (
        active_cells
            .iter()
            .zip(p.iter().zip(v_err.iter()))
            .map(|(cell, (&p_c, &verr_c))| {
                let n_max = n_active[cell.body_a].max(n_active[cell.body_b]).max(1) as f64;
                let alpha = relaxation / n_max / cell.a_cc_pressure.max(min_diagonal_element);

                // apply proximal operator -> clamp to set of allowed pressure
                (p_c - alpha * verr_c).max(0.0)
            })
            .collect(),
        active_cells
            .iter()
            .enumerate()
            .zip(lambda.iter().zip(v_tang.iter()))
            .map(|((cell_idx, cell), (&lambda_c, &v_tang_c))| {
                let n_max = n_active[cell.body_a].max(n_active[cell.body_b]).max(1) as f64;
                // let alpha = relaxation / n_max / cell.a_cc_friction.max(min_diagonal_element);
                let lambda_max_mag = friction_coefficient * p[cell_idx] * cell.f_a.norm();

                // λ*
                // apply proximal operator -> project and clamp to friction cone (projection is not necessary here, as v_tang_c is in the tangetial plane by construction)
                let lambda_c_star_raw = -v_tang_c / cell.a_cc_friction.max(min_diagonal_element);
                let lambda_c_star_raw_norm = lambda_c_star_raw.norm();
                let lambda_c_star = if lambda_c_star_raw_norm > lambda_max_mag {
                    lambda_max_mag * lambda_c_star_raw / lambda_c_star_raw_norm
                } else {
                    lambda_c_star_raw
                };

                // λ_{l+1}
                // apply proximal operator -> project and clamp to friction cone (projection is not necessary here, as v_tang_c is in the tangetial plane by construction)
                let lambda_c_new = lambda_c + relaxation / n_max * lambda_c_star;
                let lambda_c_new_norm = lambda_c_new.norm();
                if lambda_c_new_norm > lambda_max_mag {
                    lambda_max_mag * lambda_c_new / lambda_c_new_norm
                } else {
                    lambda_c_new
                }
            })
            .collect(),
    )
}

impl BoundaryHandling for VolumeMapBoundary {
    fn new() -> Self {
        Self::default()
    }

    fn is_empty(&self) -> bool {
        self.boundaries.is_empty()
    }

    fn add_static_boundary(
        &mut self,
        mesh: &mut MeshContainer,
        boundary: &StaticBoundaryDef,
        rest_density_grid_spacing: f64,
        kernel_support_radius: f64,
        progress_cb: Option<&mut dyn FnMut(f32) -> bool>,
    ) -> Result<(), BoundaryConstructionError> {
        mesh.transform(
            &boundary.translation,
            &boundary.rotation_euler_deg,
            &boundary.scale,
        );
        let trimesh = mesh.trimesh();
        let dx = rest_density_grid_spacing * 4.; // TODO
        let fields = DiscretizedBoundaryFields::new(
            trimesh,
            dx,
            kernel_support_radius,
            Self::GAUSS_LEGENDRE_QUADRATURE_DEGREE,
            progress_cb,
        )
        .map_err(|_| BoundaryConstructionError::Interrupted)?;

        let local_aabb = trimesh.local_aabb();
        let bounding_box = LocalAABB {
            center: Point3::new(
                local_aabb.center().x,
                local_aabb.center().y,
                local_aabb.center().z,
            ),
            half_extents: Vector3::new(
                local_aabb.half_extents().x,
                local_aabb.half_extents().y,
                local_aabb.half_extents().z,
            ),
        };

        let boundary = BoundaryType::StaticBoundary {
            signed_distance_field: fields.signed_distance_field,
            volume_map: fields.volume_map,
            render_mesh: mesh.render_mesh(boundary.render_vertex_normals).clone(),
            render_mesh_id: boundary.boundary_id,
            boundary_neighbor_list: NeighborList::default(),
            boundary_neighbor_list_viscosity: NeighborList::default(),
            bounding_box,
        };
        self.rest_density_grid_spacing = rest_density_grid_spacing;
        self.kernel_support_radius = kernel_support_radius;
        self.boundaries.push(boundary);
        Ok(())
    }

    fn add_dynamic_boundary(
        &mut self,
        mesh: &mut MeshContainer,
        boundary: &DynamicBoundaryDef,
        rest_density_grid_spacing: f64,
        kernel_support_radius: f64,
        progress_cb: Option<&mut dyn FnMut(f32) -> bool>,
    ) -> Result<(), BoundaryConstructionError> {
        mesh.transform(&[0., 0., 0.], &[0., 0., 0.], &boundary.scale);
        let trimesh = mesh.trimesh();
        // calc local center of mass, total mass and inertia tensor
        let mass_props: MassProperties = trimesh.mass_properties(boundary.density);
        // translate the mesh such that its local center of mass lies in the origin
        mesh.transform(
            &[
                -mass_props.local_com[0],
                -mass_props.local_com[1],
                -mass_props.local_com[2],
            ],
            &[0., 0., 0.],
            &[1., 1., 1.],
        );
        let trimesh = mesh.trimesh();
        // calc position of the center of mass in the global coordinate system
        let orientation = euler_deg_to_quaternion(boundary.rotation_euler_deg);
        let local_com_vec = Vector3::new(
            mass_props.local_com[0],
            mass_props.local_com[1],
            mass_props.local_com[2],
        );
        let global_center_of_mass = Point3::from(Vector3::from(boundary.translation))
            + orientation.transform_vector(&local_com_vec);
        // discretize the mesh
        let dx = rest_density_grid_spacing * 4.; // TODO
        let fields = DiscretizedBoundaryFields::new(
            trimesh,
            dx,
            kernel_support_radius,
            Self::GAUSS_LEGENDRE_QUADRATURE_DEGREE,
            progress_cb,
        )
        .map_err(|_| BoundaryConstructionError::Interrupted)?;

        let local_aabb = trimesh.local_aabb();
        let bounding_box = LocalAABB {
            center: Point3::new(
                local_aabb.center().x,
                local_aabb.center().y,
                local_aabb.center().z,
            ),
            half_extents: Vector3::new(
                local_aabb.half_extents().x,
                local_aabb.half_extents().y,
                local_aabb.half_extents().z,
            ),
        };

        let boundary = BoundaryType::DynamicBoundary {
            signed_distance_field: fields.signed_distance_field,
            volume_map: fields.volume_map,
            render_mesh: mesh.render_mesh(boundary.render_vertex_normals).clone(),
            render_mesh_id: boundary.boundary_id,
            state: RigidBodyMotion::new(
                mass_props.mass(),
                nalgebra::Matrix3::from(mass_props.reconstruct_inertia_matrix().to_cols_array_2d()),
                nalgebra::Matrix3::from(
                    mass_props
                        .reconstruct_inverse_inertia_matrix()
                        .to_cols_array_2d(),
                ),
                global_center_of_mass,
                orientation,
                Vector3::from(boundary.velocity),
                Vector3::from(boundary.angular_velocity),
            ),
            boundary_neighbor_list: NeighborList::default(),
            boundary_neighbor_list_viscosity: NeighborList::default(),
            bounding_box,
        };
        self.rest_density_grid_spacing = rest_density_grid_spacing;
        self.kernel_support_radius = kernel_support_radius;
        self.boundaries.push(boundary);
        Ok(())
    }

    fn initialize<K: KernelFn>(
        &mut self,
        _neighbor_search: &mut impl NeighborSearch,
        _kernel_support_radius: f64,
        _boundary_rest_volume_weighting: f64,
    ) {
    }

    fn find_boundary_samples(
        &mut self,
        _neighbor_search: &mut impl NeighborSearch,
        search_range: f64,
        positions: &[Point3<f64>],
        rest_density_grid_spacing: f64,
    ) {
        let num_samples = positions.len();

        for b in &mut self.boundaries {
            struct PerParticleNeighbors {
                pos: Vec<Point3<f64>>,
                vel: Vec<Vector3<f64>>,
                vol: Vec<f64>,
                v_pos: Vec<Point3<f64>>,
                v_vel: Vec<Vector3<f64>>,
                v_vol: Vec<f64>,
            }

            #[cfg(not(feature = "parallel"))]
            let pos_iter = positions.iter();
            #[cfg(feature = "parallel")]
            let pos_iter = positions.par_iter();
            let results: Vec<PerParticleNeighbors> = pos_iter
                .map(|pos| {
                    let mut r = PerParticleNeighbors {
                        pos: Vec::new(),
                        vel: Vec::new(),
                        vol: Vec::new(),
                        v_pos: Vec::new(),
                        v_vel: Vec::new(),
                        v_vol: Vec::new(),
                    };

                    let signed_distance = if let Ok(sd) = b.signed_distance(pos)
                        && sd < search_range
                    {
                        sd
                    } else {
                        return r;
                    };

                    let signed_distance_gradient = if let Ok(sdg) =
                        b.signed_distance_gradient(pos)
                    { // Skip particles with unphysical of signed distance gradient: For SDF the eikonal condition should hold: ‖∇d‖ ≈ 1
                        if sdg.norm() > 0.5 && sdg.norm() < 2.0 {
                            sdg
                        } else {
                            #[cfg(feature = "logging")]
                            tracing::warn!(
                                "Skipping particle with unphysical signed distance gradient: ‖∇d‖ = {}",
                                sdg.norm()
                            );
                            return r;
                        }
                    } else {
                        return r;
                    };

                    let volume = if let Ok(v) = b.volume_map_value(pos)
                        && v > 0.
                    { // Skip particles with very small volume or negative volume due to cubic serendipity interpolation
                        v
                    } else {
                        return r;
                    };

                    // Clamp volume to v_max range to avoid excessive volumes due to cubic serendipity interpolation
                    let volume = volume.min(ball_volume(search_range));

                    // let d = signed_distance.abs();
                    let d = (signed_distance + 0.25 * rest_density_grid_spacing)
                        .max(rest_density_grid_spacing);
                    let point_on_boundary =
                        pos - signed_distance_gradient / signed_distance_gradient.norm() * d;

                    r.pos.push(point_on_boundary);
                    r.vel.push(b.velocity_at_point_on_boundary(&point_on_boundary));
                    r.vol.push(volume);

                    #[cfg(feature = "logging")]
                    tracing::debug!(
                        "pos: {}, sd: {}, sdg: {:?}, vol: {}, point_ob: {:?}",
                        pos, signed_distance, signed_distance_gradient, volume, point_on_boundary
                    );

                    let vec_p_to_boundary = point_on_boundary - pos;
                    let mut vec_temp = Vector3::new(1., 0., 0.);
                    if (vec_p_to_boundary / vec_p_to_boundary.norm())
                        .dot(&vec_temp)
                        .abs()
                        > 0.9
                    {
                        vec_temp = Vector3::new(0., 1., 0.);
                    }
                    let t1 = vec_p_to_boundary.cross(&vec_temp);
                    let t2 = vec_p_to_boundary.cross(&t1);
                    let t1 = t1 / t1.norm();
                    let t2 = t2 / t2.norm();
                    let d = 0.5 * search_range;

                    let p_visc_1 = point_on_boundary + d * t1;
                    let p_visc_2 = point_on_boundary - d * t1;
                    let p_visc_3 = point_on_boundary + d * t2;
                    let p_visc_4 = point_on_boundary - d * t2;

                    r.v_pos.extend_from_slice(&[
                        p_visc_1,
                        p_visc_2,
                        p_visc_3,
                        p_visc_4,
                    ]);
                    r.v_vel.extend(&[
                        b.velocity_at_point_on_boundary(&p_visc_1),
                        b.velocity_at_point_on_boundary(&p_visc_2),
                        b.velocity_at_point_on_boundary(&p_visc_3),
                        b.velocity_at_point_on_boundary(&p_visc_4),
                    ]);
                    r.v_vol.extend_from_slice(&[0.25 * volume; 4]);

                r
            })
            .collect();

            // Sequential write-back
            b.boundary_neighbor_list_mut()
                .rebuild(num_samples, 0, |nl| {
                    let (neighbor_pos, neighbor_vel, neighbor_vol, neighbor_indices) =
                        nl.neighbors_mut();
                    for (i, r) in results.iter().enumerate() {
                        for (j, ((pos, vel), vol)) in
                            r.pos.iter().zip(&r.vel).zip(&r.vol).enumerate()
                        {
                            neighbor_pos[i].push(*pos);
                            neighbor_vel[i].push(*vel);
                            neighbor_vol[i].push(*vol);
                            neighbor_indices[i].push(j);
                        }
                    }
                });

            let len = b.boundary_neighbor_list().len();

            b.boundary_neighbor_list_viscosity_mut()
                .rebuild(num_samples, len, |nl| {
                    let (neighbor_v_pos, neighbor_v_vel, neighbor_v_vol, neighbor_v_indices) =
                        nl.neighbors_mut();
                    for (i, r) in results.iter().enumerate() {
                        for j in 0..(r.v_pos.len() / 4) {
                            let base = 4 * j;
                            for k in 0..4 {
                                neighbor_v_pos[i].push(r.v_pos[4 * j + k]);
                                neighbor_v_vel[i].push(r.v_vel[4 * j + k]);
                                neighbor_v_vol[i].push(r.v_vol[4 * j + k]);
                            }
                            neighbor_v_indices[i].extend_from_slice(&[
                                base,
                                base + 1,
                                base + 2,
                                base + 3,
                            ]);
                        }
                    }
                });
        }
    }

    fn iter(&self) -> impl Iterator<Item = &dyn Boundary> + '_ {
        self.boundaries.iter().map(|b| b as &dyn Boundary)
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut dyn Boundary> + '_ {
        self.boundaries.iter_mut().map(|b| b as &mut dyn Boundary)
    }

    fn add_force_onto_boundary(&mut self, force: ForceOntoBoundary) {
        if let Some(b) = self.boundaries.get_mut(force.id) {
            b.add_force_onto_boundary(force);
        }
    }

    fn step_forward_in_time(&mut self, dt: f64) {
        // Rigid-rigid contact: it adds to forces and torques to bodies due to contact handling.
        let mut active_cells = self.generate_contact_cells();
        self.solve_for_p_c(
            &mut active_cells,
            dt,
            Self::FB_TOLERANCE_PRESSURE,
            Self::FB_TOLERANCE_FRICTION,
        );

        for b in &mut self.boundaries {
            b.step_forward_in_time(dt);
        }
    }

    fn get_fluid_depth(&self, fluid_volume: f64) -> f64 {
        0.
    }

    fn get_visualization(&self, selector: &BoundaryVisualization) -> BoundaryVisualization {
        match selector {
            BoundaryVisualization::TriangleMesh { coloring, .. } => {
                let coloring = match coloring {
                    BoundaryMeshColoring::Original => BoundaryMeshColoring::Original,
                    BoundaryMeshColoring::Uniform => BoundaryMeshColoring::Uniform,
                    BoundaryMeshColoring::BoundaryId { .. } => {
                        let ids = self
                            .boundaries
                            .iter()
                            .map(|b| b.render_mesh_id())
                            .collect::<Vec<_>>();
                        let max_id = *ids.iter().max().unwrap_or(&0);
                        BoundaryMeshColoring::BoundaryId { ids, max_id }
                    }
                };
                BoundaryVisualization::TriangleMesh {
                    meshes: self
                        .boundaries
                        .iter()
                        .map(|b| (b.render_mesh_local().clone(), b.render_pose()))
                        .collect(),
                    coloring,
                }
            }
            BoundaryVisualization::Samples { .. } => {
                tracing::error!("Cannot provide samples as there are none.");
                BoundaryVisualization::Samples {
                    positions: vec![],
                    coloring: BoundarySampleColoring::Uniform,
                }
            }
        }
    }

    fn get_checkpoint(&self) -> BoundaryCheckpoint {
        BoundaryCheckpoint {
            dynamic_states: self
                .boundaries
                .iter()
                .map(BoundaryType::checkpoint_state)
                .collect(),
        }
    }

    fn restore_from_checkpoint(&mut self, state: &BoundaryCheckpoint) {
        if self.boundaries.len() != state.dynamic_states.len() {
            #[cfg(feature = "logging")]
            tracing::warn!(
                "Boundary checkpoint has {} entries, but {} boundaries exist; \
                     skipping boundary restore.",
                state.dynamic_states.len(),
                self.boundaries.len()
            );
            return;
        }
        for (boundary, saved) in self.boundaries.iter_mut().zip(&state.dynamic_states) {
            boundary.restore_from_checkpoint(saved);
        }
    }
}

// ─── Discretized boundary fields and helper structs ------------------------

struct DiscretizedBoundaryFields {
    signed_distance_field: CubicSerendipityDiscretization,
    volume_map: CubicSerendipityDiscretization,
}

impl DiscretizedBoundaryFields {
    /// Create new [`DiscretizedBoundaryFields`] from the given mesh and grid spacing.
    ///
    /// # Note on grid resolution
    ///
    /// Pruning operates node-based (sampling the 32 serendipity nodes per
    /// cell), not analytically. For at least one node to fall inside the
    /// relevant band (`±padding_sd`, or volume > 0), the node spacing
    /// (`dx / 3`) must be significantly smaller than both the band width and
    /// the smallest relevant object dimension. Otherwise, a cell that is
    /// actually affected can be discarded entirely by mistake (see the
    /// `is_empty()` check below for the runtime safeguard).
    ///
    /// Concretely: if `dx` is too coarse relative to `padding_sd` (or
    /// `padding_vm`) and the mesh's own size, every node of a cell can end up
    /// lying outside the padding band even though the surface geometrically
    /// passes right through that cell — causing the entire cell (and
    /// potentially the entire field) to be pruned as "outside", with no
    /// boundary samples ever being produced from it.
    fn new(
        trimesh: &TriMesh,
        dx: f64,
        kernel_support_radius: f64,
        quadrature_degree: usize,
        mut progress_cb: Option<&mut dyn FnMut(f32) -> bool>,
    ) -> Result<Self, DiscretizationError> {
        #[cfg(feature = "logging")]
        tracing::info!("Start cubic serendipity discretization.");
        let identity = Pose::identity();
        let aabb = trimesh.aabb(&identity);
        let aabb_min = Point3::new(aabb.mins.x, aabb.mins.y, aabb.mins.z);
        let aabb_max = Point3::new(aabb.maxs.x, aabb.maxs.y, aabb.maxs.z);

        let padding_sd = 3.1 * kernel_support_radius;
        let sd_field = TriangleMeshWrapper::new(trimesh);

        let padding_sd_vec = Vector3::new(padding_sd, padding_sd, padding_sd);
        let x_min = aabb_min - padding_sd_vec;
        let x_max = aabb_max + padding_sd_vec;

        let sdfn = CubicSerendipityDiscretization::new(
            x_min,
            x_max,
            Some(-padding_sd),
            Some(padding_sd),
            dx,
            &|p| sd_field.signed_distance(p),
            None,
        )?;
        if sdfn.is_empty() {
            let [nx, ny, nz] = sdfn.cell_count();
            let node_spacing = sdfn.node_spacing();
            panic!(
                "Signed-distance field is completely empty after discretization: \
                 no interpolation node fell inside the padding band ±{padding_sd} \
                 around the mesh AABB [{aabb_min} .. {aabb_max}]. \
                 Grid: {nx}x{ny}x{nz} cell(s), dx = {dx}, node spacing = dx/3 = {node_spacing}. \
                 The node spacing must be small relative to the padding band width \
                 and the mesh's own size (roughly node_spacing <= padding / 2) for \
                 at least one node to survive pruning. \
                 Reduce `rest_density_grid_spacing` (dx) or increase `kernel_support_radius`."
            );
        }
        let sd_field_wrapped = SDFnWrapper::new(&sdfn);

        #[cfg(feature = "logging")]
        tracing::info!("Finished cubic serendipity discretization.");
        #[cfg(feature = "logging")]
        tracing::info!("Start volume integration.");

        let padding_vm = 2. * kernel_support_radius + dx / 6.0; // add dx / 6.0 so that the central difference can be evaluated with h = dx / 6.0

        let padding_vm_vec = Vector3::new(padding_vm, padding_vm, padding_vm);
        let x_min = aabb_min - padding_vm_vec;
        let x_max = aabb_max + padding_vm_vec;

        let vm = CubicSerendipityDiscretization::new(
            x_min,
            x_max,
            Some(0.),
            None,
            dx,
            &|p| {
                sd_field_wrapped
                    .volume(p, kernel_support_radius, quadrature_degree)
                    .map(|v| v.clamp(0.0, ball_volume(kernel_support_radius)))
            },
            progress_cb,
        )?;
        if vm.is_empty() {
            let [nx, ny, nz] = vm.cell_count();
            let node_spacing = vm.node_spacing();
            panic!(
                "Volume map is completely empty after discretization: \
                 no interpolation node fell inside the padding band ±{padding_vm} \
                 around the mesh AABB [{aabb_min} .. {aabb_max}]. \
                 Grid: {nx}x{ny}x{nz} cell(s), dx = {dx}, node spacing = dx/3 = {node_spacing}. \
                 Reduce `rest_density_grid_spacing` (dx) or increase `kernel_support_radius`."
            );
        }

        #[cfg(feature = "logging")]
        tracing::info!("Finished volume integration.");

        Ok(DiscretizedBoundaryFields {
            signed_distance_field: sdfn,
            volume_map: vm,
        })
    }
}

struct TriangleMeshWrapper<'a> {
    boundary: &'a TriMesh,
    // max_dist: f64,
}

impl<'a> TriangleMeshWrapper<'a> {
    fn new(
        boundary: &'a TriMesh,
        // max_dist: f64,
    ) -> Self {
        Self {
            boundary,
            // max_dist,
        }
    }

    fn signed_distance(&self, point: &Point3<f64>) -> Result<f64, EvaluationError> {
        let pt = glam::DVec3::new(point.x, point.y, point.z);
        let (proj, feature) = self.boundary.project_local_point_and_get_feature(pt);
        let diff = pt - glam::DVec3::new(proj.point.x, proj.point.y, proj.point.z);
        let dist = diff.length();

        // Outward normal vector of feature:
        let n = self
            .boundary
            .feature_normal(feature)
            .expect("Could not get pseudo normal");
        if diff.dot(glam::DVec3::new(n.x, n.y, n.z)) < 0.0 {
            Ok(-dist)
        } else {
            Ok(dist)
        }
    }
}

struct SDFnWrapper<'a> {
    sdfn: &'a CubicSerendipityDiscretization,
    // max_dist: f64,
}

impl<'a> SDFnWrapper<'a> {
    fn new(sdfn: &'a CubicSerendipityDiscretization) -> Self {
        Self { sdfn }
    }

    fn volume(
        &self,
        point: &Point3<f64>,
        kernel_support_radius: f64,
        quadrature_degree: usize,
    ) -> Result<f64, EvaluationError> {
        let sd_center = match self.sdfn.function(point) {
            Ok(v) => v,
            Err(_) => return Ok(0.0), // pruned cell → outside band → treat as 0
        };

        // All points in the ball are outside kernel support → integral is exactly 0
        if sd_center >= 2.0 * kernel_support_radius {
            return Ok(0.0);
        }

        // All points in the ball are inside the surface → integrand is exactly 1
        if sd_center <= -kernel_support_radius {
            return Ok(4.0 / 3.0 * std::f64::consts::PI * kernel_support_radius.powi(3));
        }

        gauss_legendre_integrate_ball(
            &|p| match self.sdfn.function(p) {
                Ok(sd) => Ok(cubic_extension_fn(sd, kernel_support_radius)),
                Err(e) => Err(e),
            },
            point,
            kernel_support_radius,
            quadrature_degree,
        )
    }
}

fn cubic_extension_fn(signed_distance: f64, kernel_support_radius: f64) -> f64 {
    if signed_distance >= kernel_support_radius {
        return 0.;
    }
    if signed_distance <= 0. {
        return 1.;
    }
    let r_vec = Vector3::new(signed_distance, 0., 0.);
    CubicBSpline3D::kernel_function(&r_vec, kernel_support_radius)
        / CubicBSpline3D::kernel_function(&Vector3::zero(), kernel_support_radius)
}

/// `d/dx` of [`cubic_extension_fn`], i.e. `γ*'(x)`.
///
/// Derived in closed form from the known Monaghan cubic spline shape
/// (verified against Bender et al.'s `γ*` definition — Eq. 13), rather
/// than relying on an assumed derivative method on `CubicBSpline3D`.
fn cubic_extension_fn_derivative(signed_distance: f64, kernel_support_radius: f64) -> f64 {
    if signed_distance >= kernel_support_radius || signed_distance <= 0.0 {
        return 0.0;
    }
    let r_vec = Vector3::new(signed_distance, 0., 0.);
    CubicBSpline3D::kernel_gradient(&r_vec, kernel_support_radius).x
        / CubicBSpline3D::kernel_function(&Vector3::zero(), kernel_support_radius)
}

// ─── Boundaries ───────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum BoundaryType {
    /// Static boundary
    StaticBoundary {
        signed_distance_field: CubicSerendipityDiscretization,
        volume_map: CubicSerendipityDiscretization,
        render_mesh: RenderMesh,
        render_mesh_id: u32,
        boundary_neighbor_list: NeighborList,
        boundary_neighbor_list_viscosity: NeighborList,
        bounding_box: LocalAABB,
    },
    /// Dynamic boundary: performs rigid-body motion with two-way coupling with fluid
    DynamicBoundary {
        signed_distance_field: CubicSerendipityDiscretization,
        volume_map: CubicSerendipityDiscretization,
        render_mesh: RenderMesh,
        render_mesh_id: u32,
        state: RigidBodyMotion,
        boundary_neighbor_list: NeighborList,
        boundary_neighbor_list_viscosity: NeighborList,
        bounding_box: LocalAABB,
    },
}

#[derive(Debug, Clone)]
struct LocalAABB {
    center: Point3<f64>,
    half_extents: Vector3<f64>,
}

#[derive(Debug, Clone)]
struct WorldAABB {
    min: Point3<f64>,
    max: Point3<f64>,
}

impl WorldAABB {
    fn overlap(&self, b: &WorldAABB) -> bool {
        self.min.x <= b.max.x
            && self.max.x >= b.min.x
            && self.min.y <= b.max.y
            && self.max.y >= b.min.y
            && self.min.z <= b.max.z
            && self.max.z >= b.min.z
    }

    fn intersection(&self, other: &WorldAABB) -> Option<WorldAABB> {
        let min = Point3::new(
            self.min.x.max(other.min.x),
            self.min.y.max(other.min.y),
            self.min.z.max(other.min.z),
        );
        let max = Point3::new(
            self.max.x.min(other.max.x),
            self.max.y.min(other.max.y),
            self.max.z.min(other.max.z),
        );
        (min.x <= max.x && min.y <= max.y && min.z <= max.z).then_some(WorldAABB { min, max })
    }
}

#[derive(Debug, Clone)]
struct SubCell {
    from: Point3<f64>,
    to: Point3<f64>,
}

fn subdivide_overlap(overlap: &WorldAABB, target_cell_size: f64) -> Vec<SubCell> {
    let extent = overlap.max - overlap.min;
    let n = [
        ((extent.x / target_cell_size).ceil() as usize).max(1),
        ((extent.y / target_cell_size).ceil() as usize).max(1),
        ((extent.z / target_cell_size).ceil() as usize).max(1),
    ];
    let cell_size = Vector3::new(
        extent.x / n[0] as f64,
        extent.y / n[1] as f64,
        extent.z / n[2] as f64,
    );

    let mut cells = Vec::with_capacity(n[0] * n[1] * n[2]);
    for iz in 0..n[2] {
        for iy in 0..n[1] {
            for ix in 0..n[0] {
                let from = overlap.min
                    + Vector3::new(
                        ix as f64 * cell_size.x,
                        iy as f64 * cell_size.y,
                        iz as f64 * cell_size.z,
                    );
                cells.push(SubCell {
                    from,
                    to: from + cell_size,
                });
            }
        }
    }
    cells
}

impl Boundary for BoundaryType {
    fn get_neighbors(&self, id: usize, mode: RequestMode) -> &[usize] {
        match mode {
            RequestMode::Normal => self.boundary_neighbor_list().get_neighbors(id),
            RequestMode::ViscosityAcceleration => {
                self.boundary_neighbor_list_viscosity().get_neighbors(id)
            }
        }
    }

    fn position(&self, id: usize) -> &Point3<f64> {
        let num_neighbors = self.boundary_neighbor_list().len();
        if id < num_neighbors {
            self.boundary_neighbor_list().pos_now(id)
        } else {
            self.boundary_neighbor_list_viscosity()
                .pos_now(id - num_neighbors)
        }
    }

    fn velocity(&self, id: usize) -> &Vector3<f64> {
        let num_neighbors = self.boundary_neighbor_list().len();
        if id < num_neighbors {
            self.boundary_neighbor_list().vel_now(id)
        } else {
            self.boundary_neighbor_list_viscosity()
                .vel_now(id - num_neighbors)
        }
    }

    fn volume(&self, id: usize) -> f64 {
        let num_neighbors = self.boundary_neighbor_list().len();
        if id < num_neighbors {
            *self.boundary_neighbor_list().volume(id)
        } else {
            *self
                .boundary_neighbor_list_viscosity()
                .volume(id - num_neighbors)
        }
    }

    fn add_acceleration(&mut self, acceleration: Vector3<f64>) {
        let force = match self {
            Self::StaticBoundary { .. } => None,
            Self::DynamicBoundary { state, .. } => Some(ForceOntoBoundary {
                id: 0, // dummy entry, value is not used
                force: state.mass() * acceleration,
                force_location: state.center_of_mass(),
            }),
        };

        if let Some(force) = force {
            self.add_force_onto_boundary(force);
        }
    }

    fn center_of_mass(&self) -> Option<Point3<f64>> {
        match self {
            Self::StaticBoundary { .. } => None,
            Self::DynamicBoundary { state, .. } => Some(state.center_of_mass()),
        }
    }
}

impl BoundaryType {
    /// Current pose mapping body/local frame -> world frame.
    /// Identity for static boundaries, since their fields are already
    /// baked into world space at construction time.
    fn pose(&self) -> Isometry3<f64> {
        match self {
            Self::StaticBoundary { .. } => Isometry3::identity(),
            Self::DynamicBoundary { state, .. } => state.pose(),
        }
    }

    fn motion(&self) -> Option<&RigidBodyMotion> {
        match self {
            Self::StaticBoundary { .. } => None,
            Self::DynamicBoundary { state, .. } => Some(state),
        }
    }

    fn motion_mut(&mut self) -> Option<&mut RigidBodyMotion> {
        match self {
            Self::StaticBoundary { .. } => None,
            Self::DynamicBoundary { state, .. } => Some(state),
        }
    }

    /// Signed distance at a WORLD-space point.
    fn signed_distance(&self, p_world: &Point3<f64>) -> Result<f64, EvaluationError> {
        match self {
            Self::StaticBoundary {
                signed_distance_field,
                ..
            } => signed_distance_field.function(p_world),
            Self::DynamicBoundary {
                signed_distance_field,
                state,
                ..
            } => {
                let p_local = state.world_to_local(p_world);
                signed_distance_field.function(&p_local)
            }
        }
    }

    /// Signed distance gradient at a WORLD-space point, returned in WORLD coords.
    fn signed_distance_gradient(
        &self,
        p_world: &Point3<f64>,
    ) -> Result<Vector3<f64>, EvaluationError> {
        match self {
            Self::StaticBoundary {
                signed_distance_field,
                ..
            } => signed_distance_field.gradient(p_world),
            Self::DynamicBoundary {
                signed_distance_field,
                state,
                ..
            } => {
                let p_local = state.world_to_local(p_world);
                let grad_local = signed_distance_field.gradient(&p_local)?;
                Ok(state.local_to_world_vector(&grad_local))
            }
        }
    }

    /// Volume-map value at a WORLD-space point.
    fn volume_map_value(&self, p_world: &Point3<f64>) -> Result<f64, EvaluationError> {
        match self {
            Self::StaticBoundary { volume_map, .. } => volume_map.function(p_world),
            Self::DynamicBoundary {
                volume_map, state, ..
            } => {
                let p_local = state.world_to_local(p_world);
                volume_map.function(&p_local)
            }
        }
    }

    /// Velocity at a point on the boundary.
    fn velocity_at_point_on_boundary(&self, p_world: &Point3<f64>) -> Vector3<f64> {
        match self {
            Self::StaticBoundary { .. } => Vector3::zero(),
            Self::DynamicBoundary { state, .. } => state.velocity_at_point(p_world),
        }
    }

    /// Render mesh in body-frame (for static boundaries == world space).
    fn render_mesh_local(&self) -> &RenderMesh {
        match self {
            Self::StaticBoundary { render_mesh, .. } => render_mesh,
            Self::DynamicBoundary { render_mesh, .. } => render_mesh,
        }
    }

    fn render_pose(&self) -> RenderPose {
        RenderPose::from(self.pose())
    }

    fn render_mesh_id(&self) -> u32 {
        match self {
            Self::StaticBoundary { render_mesh_id, .. } => *render_mesh_id,
            Self::DynamicBoundary { render_mesh_id, .. } => *render_mesh_id,
        }
    }

    fn render_mesh_id_mut(&mut self) -> &mut u32 {
        match self {
            Self::StaticBoundary { render_mesh_id, .. } => render_mesh_id,
            Self::DynamicBoundary { render_mesh_id, .. } => render_mesh_id,
        }
    }

    fn boundary_neighbor_list(&self) -> &NeighborList {
        match self {
            Self::StaticBoundary {
                boundary_neighbor_list,
                ..
            } => boundary_neighbor_list,
            Self::DynamicBoundary {
                boundary_neighbor_list,
                ..
            } => boundary_neighbor_list,
        }
    }

    fn boundary_neighbor_list_mut(&mut self) -> &mut NeighborList {
        match self {
            Self::StaticBoundary {
                boundary_neighbor_list,
                ..
            } => boundary_neighbor_list,
            Self::DynamicBoundary {
                boundary_neighbor_list,
                ..
            } => boundary_neighbor_list,
        }
    }

    fn boundary_neighbor_list_viscosity(&self) -> &NeighborList {
        match self {
            Self::StaticBoundary {
                boundary_neighbor_list_viscosity,
                ..
            } => boundary_neighbor_list_viscosity,
            Self::DynamicBoundary {
                boundary_neighbor_list_viscosity,
                ..
            } => boundary_neighbor_list_viscosity,
        }
    }

    fn boundary_neighbor_list_viscosity_mut(&mut self) -> &mut NeighborList {
        match self {
            Self::StaticBoundary {
                boundary_neighbor_list_viscosity,
                ..
            } => boundary_neighbor_list_viscosity,
            Self::DynamicBoundary {
                boundary_neighbor_list_viscosity,
                ..
            } => boundary_neighbor_list_viscosity,
        }
    }

    fn add_force_onto_boundary(&mut self, force: ForceOntoBoundary) {
        match self {
            Self::StaticBoundary { .. } => {}
            Self::DynamicBoundary { state, .. } => {
                state.add_force(force);
            }
        }
    }

    fn step_forward_in_time(&mut self, dt: f64) {
        match self {
            Self::StaticBoundary { .. } => {}
            Self::DynamicBoundary { state, .. } => {
                state.step_forward_in_time(dt);
            }
        }
    }

    fn checkpoint_state(&self) -> Option<RigidBodyMotionCheckpoint> {
        match self {
            Self::StaticBoundary { .. } => None,
            Self::DynamicBoundary { state, .. } => Some(state.get_checkpoint()),
        }
    }

    fn restore_from_checkpoint(&mut self, saved: &Option<RigidBodyMotionCheckpoint>) {
        match (self, saved) {
            (Self::DynamicBoundary { state, .. }, Some(saved)) => {
                state.restore_from_checkpoint(saved);
                // No cached `position`/`velocity` to refresh here: `find_boundary_samples`
                // recomputes boundary samples fresh from `signed_distance_field` /
                // `volume_map` via `state.world_to_local(...)` on every call, so the
                // restored `state` alone is sufficient.
            }
            (Self::StaticBoundary { .. }, None) => {}
            // Mismatch between the checkpoint and the current boundary setup
            // (e.g. scene changed between saving and resuming): ignore rather
            // than panic, but this indicates a stale checkpoint.
            _ => {
                #[cfg(feature = "logging")]
                tracing::warn!(
                    "Boundary checkpoint entry does not match boundary type \
                         (static vs. dynamic); skipping restore for this boundary."
                );
            }
        }
    }

    pub fn world_aabb(&self) -> WorldAABB {
        match self {
            BoundaryType::StaticBoundary { bounding_box, .. } => WorldAABB {
                min: bounding_box.center - (bounding_box.half_extents),
                max: bounding_box.center + (bounding_box.half_extents),
            },
            BoundaryType::DynamicBoundary {
                state,
                bounding_box,
                ..
            } => {
                let pose = state.pose(); // Isometry3: aktuelle Rotation + Translation
                let r = pose.rotation.to_rotation_matrix();

                // Komponentenweises |R| angewendet auf die lokalen Halbextents (Arvos Methode)
                let abs_r = r.matrix().abs();
                let world_half_extents = abs_r * (bounding_box.half_extents);

                let world_center = pose.transform_point(&Point3::from(bounding_box.center));

                WorldAABB {
                    min: world_center - world_half_extents,
                    max: world_center + world_half_extents,
                }
            }
        }
    }
}

// ─── NeighborList ───────────────────────────────────────────────

use neighbor_list::NeighborList;

mod neighbor_list {
    use super::*;

    /// Per-boundary list of fluid samples treated as neighbors, alongside
    /// their interpolated position/velocity/volume on the boundary surface.
    ///
    /// The only way to (re)populate this list is [`NeighborList::rebuild`],
    /// which resizes, clears, fills and flattens atomically — `resize`,
    /// `clear` and `flatten` are private to this module.
    #[derive(Debug, Clone, Default)]
    pub struct NeighborList {
        positions: Vec<Point3<f64>>,
        velocities: Vec<Vector3<f64>>,
        volumes: Vec<f64>,
        /// Flat neighbor list
        indices: Vec<usize>,
        /// Index list to point to start of the neighbor list of each sample
        offsets: Vec<usize>,
        /// Unflattened neighbor list which is necessary since there can exist many boundaries
        unflattened_positions: Vec<Vec<Point3<f64>>>,
        unflattened_velocities: Vec<Vec<Vector3<f64>>>,
        unflattened_volumes: Vec<Vec<f64>>,
        unflattened_indices: Vec<Vec<usize>>,
    }

    /// Transient handle granting mutable access to a [`NeighborList`]'s
    /// unflattened per-sample buffers — see [`super::NeighborList`]'s
    /// module-level counterpart in `neighbor_search` for the rationale.
    pub struct NeighborListBuilder<'a> {
        positions: &'a mut Vec<Vec<Point3<f64>>>,
        velocities: &'a mut Vec<Vec<Vector3<f64>>>,
        volumes: &'a mut Vec<Vec<f64>>,
        indices: &'a mut Vec<Vec<usize>>,
    }

    impl<'a> NeighborListBuilder<'a> {
        pub fn neighbors_mut(
            &mut self,
        ) -> (
            &mut Vec<Vec<Point3<f64>>>,
            &mut Vec<Vec<Vector3<f64>>>,
            &mut Vec<Vec<f64>>,
            &mut Vec<Vec<usize>>,
        ) {
            (self.positions, self.velocities, self.volumes, self.indices)
        }
    }

    impl NeighborList {
        /// Total number of flattened neighbor entries across all samples in
        /// this list — i.e. the length of `positions`/`velocities`/`volumes`/
        /// `indices`, NOT the number of samples. Used by `BoundaryType::pos_now`/
        /// `vel_now`/`volume` as the offset separating this list's entries from
        /// a second list's (`boundary_neighbor_list` vs.
        /// `boundary_neighbor_list_viscosity`) in a combined global index space.
        ///
        /// Not to be confused with `neighbor_search::NeighborList::num_samples`,
        /// which counts samples rather than flattened entries.
        pub fn len(&self) -> usize {
            self.positions.len()
        }

        fn resize(&mut self, len: usize) {
            self.unflattened_positions.resize(len, Vec::new());
            self.unflattened_velocities.resize(len, Vec::new());
            self.unflattened_volumes.resize(len, Vec::new());
            self.unflattened_indices.resize(len, Vec::new());
        }

        fn clear(&mut self) {
            self.positions.clear();
            self.velocities.clear();
            self.volumes.clear();
            self.indices.clear();
            self.offsets.clear();
            for_each_mut!(
                mut [self.unflattened_indices, self.unflattened_positions, self.unflattened_velocities, self.unflattened_volumes],
                ref [],
                |_id, id_index, id_pos, id_vel, id_vol| {
                    id_index.clear();
                    id_pos.clear();
                    id_vel.clear();
                    id_vol.clear();
                }
            );
        }

        /// Flatten neighbor list
        fn flatten(&mut self, global_offset: usize) {
            self.positions.clear();
            self.velocities.clear();
            self.volumes.clear();
            self.indices.clear();
            self.offsets.clear();

            let total_neighbors: usize = self.unflattened_indices.iter().map(|v| v.len()).sum();
            let num_particles = self.unflattened_indices.len();

            self.positions.reserve(total_neighbors);
            self.velocities.reserve(total_neighbors);
            self.volumes.reserve(total_neighbors);
            self.indices.reserve(total_neighbors);
            self.offsets.reserve(num_particles + 1);

            self.offsets.push(0);
            for (((nbr_pos, nbr_vel), nbr_vol), idcs) in self
                .unflattened_positions
                .iter()
                .zip(&self.unflattened_velocities)
                .zip(&self.unflattened_volumes)
                .zip(&self.unflattened_indices)
            {
                let offset_bc_of_previous_samples_neighbors = self.positions.len();
                self.positions.extend_from_slice(nbr_pos);
                self.velocities.extend_from_slice(nbr_vel);
                self.volumes.extend_from_slice(nbr_vol);
                for &local_idx in idcs {
                    self.indices
                        .push(global_offset + offset_bc_of_previous_samples_neighbors + local_idx);
                }
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
        pub fn rebuild(
            &mut self,
            num_samples: usize,
            global_offset: usize,
            fill: impl FnOnce(&mut NeighborListBuilder<'_>),
        ) {
            self.resize(num_samples);
            self.clear();
            let mut builder = NeighborListBuilder {
                positions: &mut self.unflattened_positions,
                velocities: &mut self.unflattened_velocities,
                volumes: &mut self.unflattened_volumes,
                indices: &mut self.unflattened_indices,
            };
            fill(&mut builder);
            self.flatten(global_offset);
        }

        /// Get indices of neighbor of sample with identifier 'id'
        pub fn get_neighbors(&self, id: usize) -> &[usize] {
            &self.indices[self.offsets[id]..self.offsets[id + 1]]
        }

        pub fn pos_now<I>(&self, id: I) -> &I::Output
        where
            I: SliceIndex<[Point3<f64>]>,
        {
            &self.positions[id]
        }

        pub fn vel_now<I>(&self, id: I) -> &I::Output
        where
            I: SliceIndex<[Vector3<f64>]>,
        {
            &self.velocities[id]
        }

        pub fn volume<I>(&self, id: I) -> &I::Output
        where
            I: SliceIndex<[f64]>,
        {
            &self.volumes[id]
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        // ─── Helper functions ───────────────────────────────────────────────

        fn pos(x: f64, y: f64, z: f64) -> Point3<f64> {
            Point3::new(x, y, z)
        }

        fn vel(x: f64, y: f64, z: f64) -> Vector3<f64> {
            Vector3::new(x, y, z)
        }

        // ─── Construction / empty-state tests ────────────────────────────────

        #[test]
        fn default_is_empty() {
            let nl = NeighborList::default();
            assert_eq!(nl.len(), 0);
        }

        #[test]
        fn rebuild_from_default_grows_correctly() {
            // Mirrors actual production usage in `find_boundary_samples`:
            // `NeighborList::default()` is held across time steps, and the
            // first real call is `rebuild` growing it to the current sample
            // count.
            let mut nl = NeighborList::default();
            nl.rebuild(2, 0, |b| {
                let (positions, velocities, volumes, indices) = b.neighbors_mut();
                positions[0] = vec![pos(1., 0., 0.)];
                velocities[0] = vec![vel(0.1, 0., 0.)];
                volumes[0] = vec![0.5];
                indices[0] = vec![0];
            });

            assert_eq!(nl.len(), 1);
            assert_eq!(nl.get_neighbors(0), &[0]);
            assert_eq!(*nl.pos_now(0), pos(1., 0., 0.));
            assert_eq!(*nl.vel_now(0), vel(0.1, 0., 0.));
            assert_eq!(*nl.volume(0), 0.5);
            // Particle 1 has no neighbors
            assert_eq!(nl.get_neighbors(1), &[]);
        }

        // ─── flatten: ordering / consistency ─────────────────────────────────

        #[test]
        fn flatten_empty_list() {
            let mut nl = NeighborList::default();
            nl.rebuild(3, 0, |_| {
                // fill does nothing — no sample has any neighbors
            });

            assert_eq!(nl.len(), 0);
            for i in 0..3 {
                assert_eq!(nl.get_neighbors(i), &[]);
            }
        }

        #[test]
        fn rebuild_single_particle_with_neighbor() {
            let mut nl = NeighborList::default();
            nl.rebuild(1, 0, |b| {
                let (positions, velocities, volumes, indices) = b.neighbors_mut();
                positions[0] = vec![pos(2., 3., 4.)];
                velocities[0] = vec![vel(0.5, 0.6, 0.7)];
                volumes[0] = vec![1.25];
                indices[0] = vec![0];
            });

            assert_eq!(nl.get_neighbors(0), &[0]);
            assert_eq!(*nl.pos_now(0), pos(2., 3., 4.));
            assert_eq!(*nl.vel_now(0), vel(0.5, 0.6, 0.7));
            assert_eq!(*nl.volume(0), 1.25);
        }

        #[test]
        fn rebuild_multiple_particles_preserves_order() {
            let mut nl = NeighborList::default();
            nl.rebuild(2, 0, |b| {
                let (positions, velocities, volumes, indices) = b.neighbors_mut();
                positions[0] = vec![pos(1., 0., 0.), pos(2., 0., 0.)];
                velocities[0] = vec![vel(0.1, 0., 0.), vel(0.2, 0., 0.)];
                volumes[0] = vec![0.1, 0.2];
                indices[0] = vec![0, 1];

                positions[1] = vec![pos(9., 0., 0.)];
                velocities[1] = vec![vel(0.9, 0., 0.)];
                volumes[1] = vec![0.9];
                indices[1] = vec![0];
            });

            // Particle 0's two neighbors keep their relative order
            assert_eq!(*nl.pos_now(0), pos(1., 0., 0.));
            assert_eq!(*nl.pos_now(1), pos(2., 0., 0.));
            // Particle 1's single neighbor is appended after particle 0's data
            assert_eq!(*nl.pos_now(2), pos(9., 0., 0.));
            assert_eq!(*nl.volume(2), 0.9);
        }

        // ─── flatten: global_offset semantics (unique to volume_map_boundary) ───────

        #[test]
        fn flatten_applies_global_offset_uniformly() {
            // `global_offset` is added uniformly to every stored index, so
            // `id - global_offset` must always yield a valid index into this
            // list's own `pos_now`/`vel_now`/`volume` — this is exactly the
            // invariant `BoundaryType::pos_now`/`vel_now`/`volume` rely on when
            // dispatching between `boundary_neighbor_list` and
            // `boundary_neighbor_list_viscosity` by comparing `id` against the
            // first list's `len()`.
            let global_offset = 100;
            let mut nl = NeighborList::default();
            nl.rebuild(1, global_offset, |b| {
                let (positions, velocities, volumes, indices) = b.neighbors_mut();
                positions[0] = vec![pos(5., 5., 5.)];
                velocities[0] = vec![vel(1., 1., 1.)];
                volumes[0] = vec![0.42];
                indices[0] = vec![0];
            });

            let stored_id = nl.get_neighbors(0)[0];
            assert_eq!(stored_id, global_offset);

            let local_id = stored_id - global_offset;
            assert_eq!(*nl.pos_now(local_id), pos(5., 5., 5.));
            assert_eq!(*nl.vel_now(local_id), vel(1., 1., 1.));
            assert_eq!(*nl.volume(local_id), 0.42);
        }

        #[test]
        fn flatten_accumulates_offset_across_particles() {
            // The trickiest part of `flatten`'s bookkeeping: a later particle's
            // local neighbor index (0, 1, 2, ...) must be shifted by the number
            // of neighbor entries already flattened for *earlier* particles in
            // this same list — not just by `global_offset`. Particle 0 has 2
            // neighbors (occupying flat slots 0 and 1), so particle 1's first
            // neighbor (local index 0) must land at flat slot 2.
            let mut nl = NeighborList::default();
            nl.rebuild(2, 0, |b| {
                let (positions, velocities, volumes, indices) = b.neighbors_mut();
                positions[0] = vec![pos(1., 0., 0.), pos(2., 0., 0.)];
                velocities[0] = vec![vel(0., 0., 0.), vel(0., 0., 0.)];
                volumes[0] = vec![0., 0.];
                indices[0] = vec![0, 1];

                positions[1] = vec![pos(9., 9., 9.)];
                velocities[1] = vec![vel(0., 0., 0.)];
                volumes[1] = vec![0.];
                indices[1] = vec![0]; // local index 0 within particle 1's own data
            });

            assert_eq!(nl.get_neighbors(0), &[0, 1]);
            assert_eq!(nl.get_neighbors(1), &[2]);
            assert_eq!(*nl.pos_now(2), pos(9., 9., 9.));
        }

        // ─── resize / clear (direct access, same module as production code) ─

        #[test]
        fn resize_grows() {
            let mut nl = NeighborList::default();
            nl.rebuild(2, 0, |b| {
                let (positions, _, _, indices) = b.neighbors_mut();
                positions[0] = vec![pos(1., 0., 0.)];
                indices[0] = vec![0];
            });

            nl.resize(5);

            assert_eq!(nl.unflattened_indices.len(), 5);
            // Existing data preserved
            assert_eq!(nl.unflattened_indices[0], vec![0]);
            // New entries are empty
            for i in 2..5 {
                assert!(nl.unflattened_indices[i].is_empty());
                assert!(nl.unflattened_positions[i].is_empty());
            }
        }

        #[test]
        fn resize_shrinks() {
            let mut nl = NeighborList::default();
            nl.rebuild(5, 0, |b| {
                let (positions, _, _, indices) = b.neighbors_mut();
                positions[0] = vec![pos(1., 0., 0.)];
                indices[0] = vec![0];
            });

            nl.resize(2);

            assert_eq!(nl.unflattened_indices.len(), 2);
            assert_eq!(nl.unflattened_positions[0], vec![pos(1., 0., 0.)]);
        }

        #[test]
        fn clear_resets_all_unflattened_data() {
            let mut nl = NeighborList::default();
            nl.rebuild(2, 0, |b| {
                let (positions, velocities, volumes, indices) = b.neighbors_mut();
                positions[0] = vec![pos(1., 0., 0.)];
                velocities[0] = vec![vel(1., 0., 0.)];
                volumes[0] = vec![1.];
                indices[0] = vec![0];
            });

            nl.clear();

            assert!(nl.unflattened_positions.iter().all(|v| v.is_empty()));
            assert!(nl.unflattened_velocities.iter().all(|v| v.is_empty()));
            assert!(nl.unflattened_volumes.iter().all(|v| v.is_empty()));
            assert!(nl.unflattened_indices.iter().all(|v| v.is_empty()));
        }

        // ─── rebuild: shrink/grow and no-stale-data guarantees ──────────────

        #[test]
        fn rebuild_shrink_then_grow_has_no_stale_data() {
            // `resize` truncates when shrinking rather than caching removed
            // elements, so growing back must yield fresh, empty slots — not
            // stale data from before the shrink.
            let mut nl = NeighborList::default();
            nl.rebuild(3, 0, |b| {
                let (positions, _, _, indices) = b.neighbors_mut();
                for i in 0..3 {
                    positions[i] = vec![pos(i as f64, 0., 0.)];
                    indices[i] = vec![0];
                }
            });

            nl.rebuild(1, 0, |b| {
                let (positions, _, _, indices) = b.neighbors_mut();
                positions[0] = vec![pos(99., 0., 0.)];
                indices[0] = vec![0];
            });

            nl.rebuild(3, 0, |b| {
                let (positions, _, _, indices) = b.neighbors_mut();
                positions[0] = vec![pos(42., 0., 0.)];
                indices[0] = vec![0];
                // slots 1, 2 intentionally left untouched by `fill`
            });

            assert_eq!(nl.get_neighbors(0), &[0]);
            assert_eq!(*nl.pos_now(0), pos(42., 0., 0.));
            for i in 1..3 {
                assert_eq!(
                    nl.get_neighbors(i),
                    &[],
                    "slot {i} should be empty, not stale"
                );
            }
        }

        #[test]
        fn rebuild_fill_noop_clears_previous_data() {
            let mut nl = NeighborList::default();
            nl.rebuild(2, 0, |b| {
                let (positions, _, _, indices) = b.neighbors_mut();
                positions[0] = vec![pos(1., 0., 0.)];
                indices[0] = vec![0];
            });

            nl.rebuild(2, 0, |_| {
                // fill does nothing
            });

            assert_eq!(nl.get_neighbors(0), &[]);
            assert_eq!(nl.get_neighbors(1), &[]);
            assert_eq!(nl.len(), 0);
        }

        #[test]
        fn rebuild_called_twice_replaces_data() {
            let mut nl = NeighborList::default();
            nl.rebuild(1, 0, |b| {
                let (positions, velocities, volumes, indices) = b.neighbors_mut();
                positions[0] = vec![pos(1., 0., 0.)];
                velocities[0] = vec![vel(1., 0., 0.)];
                volumes[0] = vec![1.];
                indices[0] = vec![0];
            });

            nl.rebuild(1, 0, |b| {
                let (positions, velocities, volumes, indices) = b.neighbors_mut();
                positions[0] = vec![pos(2., 0., 0.), pos(3., 0., 0.)];
                velocities[0] = vec![vel(2., 0., 0.), vel(3., 0., 0.)];
                volumes[0] = vec![2., 3.];
                indices[0] = vec![0, 1];
            });

            assert_eq!(nl.get_neighbors(0), &[0, 1]);
            assert_eq!(*nl.pos_now(0), pos(2., 0., 0.));
            assert_eq!(*nl.pos_now(1), pos(3., 0., 0.));
        }

        // ─── data-consistency & range-indexing tests ─────────────────────────

        #[test]
        fn get_data_length_consistency() {
            let mut nl = NeighborList::default();
            nl.rebuild(4, 0, |b| {
                let (positions, velocities, volumes, indices) = b.neighbors_mut();
                positions[0] = vec![pos(1., 0., 0.), pos(2., 0., 0.), pos(3., 0., 0.)];
                velocities[0] = vec![vel(0., 0., 0.); 3];
                volumes[0] = vec![0.; 3];
                indices[0] = vec![0, 1, 2];

                positions[2] = vec![pos(4., 0., 0.); 4];
                velocities[2] = vec![vel(0., 0., 0.); 4];
                volumes[2] = vec![0.; 4];
                indices[2] = vec![0, 1, 2, 3];
            });

            let total: usize = (0..4).map(|i| nl.get_neighbors(i).len()).sum();
            assert_eq!(total, nl.positions.len());
            assert_eq!(total, nl.velocities.len());
            assert_eq!(total, nl.volumes.len());
            assert_eq!(total, nl.indices.len());
            assert_eq!(nl.offsets.len(), 5); // num_particles + 1
        }

        #[test]
        fn pos_now_vel_now_volume_support_range_indexing() {
            // `pos_now`/`vel_now`/`volume` are generic over `SliceIndex`, not
            // just `usize` — used by callers that want a contiguous slice
            // rather than a single element.
            let mut nl = NeighborList::default();
            nl.rebuild(1, 0, |b| {
                let (positions, velocities, volumes, indices) = b.neighbors_mut();
                positions[0] = vec![pos(1., 0., 0.), pos(2., 0., 0.), pos(3., 0., 0.)];
                velocities[0] = vec![vel(1., 0., 0.), vel(2., 0., 0.), vel(3., 0., 0.)];
                volumes[0] = vec![1., 2., 3.];
                indices[0] = vec![0, 1, 2];
            });

            assert_eq!(nl.pos_now(0..2), &[pos(1., 0., 0.), pos(2., 0., 0.)]);
            assert_eq!(nl.vel_now(1..3), &[vel(2., 0., 0.), vel(3., 0., 0.)]);
            assert_eq!(nl.volume(..), &[1., 2., 3.]);
        }

        #[test]
        fn large_neighbor_list() {
            let n = 1000;
            let mut nl = NeighborList::default();
            nl.rebuild(n, 0, |b| {
                let (positions, velocities, volumes, indices) = b.neighbors_mut();
                for i in 0..n {
                    positions[i] = vec![pos(i as f64, 0., 0.)];
                    velocities[i] = vec![vel(0., 0., 0.)];
                    volumes[i] = vec![1.0];
                    indices[i] = vec![0];
                }
            });

            assert_eq!(nl.len(), n);
            for i in 0..n {
                assert_eq!(nl.get_neighbors(i), &[i]);
                assert_eq!(*nl.pos_now(i), pos(i as f64, 0., 0.));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::UnitQuaternion;
    use parry3d_f64::{math::Vec3, shape::TriMeshFlags};

    // ─── ball_volume ────────────────────────────────────────────────

    #[test]
    fn ball_volume_zero_radius_is_zero() {
        assert_eq!(ball_volume(0.0), 0.0);
    }

    #[test]
    fn ball_volume_unit_radius() {
        let expected = 4.0 / 3.0 * std::f64::consts::PI;
        assert!((ball_volume(1.0) - expected).abs() < 1e-12);
    }

    #[test]
    fn ball_volume_scales_with_cube_of_radius() {
        // V(2r) = 8 * V(r), a basic sanity check on the r^3 scaling.
        let r = 0.7;
        assert!((ball_volume(2.0 * r) - 8.0 * ball_volume(r)).abs() < 1e-9);
    }

    // ─── cubic_extension_fn ─────────────────────────────────────────
    //
    // Deliberately tests only qualitative, kernel-formula-independent
    // properties (boundary values, monotonicity, continuity) rather than
    // exact numeric values — the precise `CubicBSpline3D::kernel_function`
    // polynomial isn't something I've verified in this conversation, so
    // hardcoding expected magnitudes here would risk asserting values I
    // can't actually justify.

    #[test]
    fn cubic_extension_fn_is_one_at_and_below_zero() {
        assert_eq!(cubic_extension_fn(0.0, 1.0), 1.0);
        assert_eq!(cubic_extension_fn(-0.5, 1.0), 1.0);
    }

    #[test]
    fn cubic_extension_fn_is_zero_at_and_beyond_support_radius() {
        assert_eq!(cubic_extension_fn(1.0, 1.0), 0.0);
        assert_eq!(cubic_extension_fn(1.5, 1.0), 0.0);
    }

    #[test]
    fn cubic_extension_fn_is_monotonically_decreasing_in_between() {
        let h = 1.0;
        let v1 = cubic_extension_fn(0.1 * h, h);
        let v2 = cubic_extension_fn(0.5 * h, h);
        let v3 = cubic_extension_fn(0.9 * h, h);
        assert!(v1 > v2, "{v1} should be > {v2}");
        assert!(v2 > v3, "{v2} should be > {v3}");
        assert!(v3 > 0.0 && v3 < 1.0);
    }

    #[test]
    fn cubic_extension_fn_approaches_zero_continuously_near_support_radius() {
        // Standard cubic B-spline SPH kernels vanish smoothly at their
        // compact support radius, so the ratio-based value just inside `h`
        // should already be close to the `0.0` returned for `sd >= h`.
        let h = 1.0;
        let just_inside = cubic_extension_fn(h - 1e-6, h);
        assert!(
            just_inside < 1e-3,
            "expected near-zero continuity at h, got {just_inside}"
        );
    }

    // ─── TriangleMeshWrapper::signed_distance ───────────────────────

    /// Cube of side length 2 centered at the origin, outward-facing
    /// winding — the same fixture used for `SampleBoundary` tests.
    fn cube_trimesh() -> TriMesh {
        let positions = vec![
            Vec3::new(1., 1., 1.),
            Vec3::new(1., 1., -1.),
            Vec3::new(1., -1., 1.),
            Vec3::new(1., -1., -1.),
            Vec3::new(-1., 1., 1.),
            Vec3::new(-1., 1., -1.),
            Vec3::new(-1., -1., 1.),
            Vec3::new(-1., -1., -1.),
        ];
        let indices: Vec<[u32; 3]> = vec![
            [4, 2, 0],
            [2, 7, 3],
            [6, 5, 7],
            [1, 7, 5],
            [0, 3, 1],
            [4, 1, 5],
            [4, 6, 2],
            [2, 6, 7],
            [6, 4, 5],
            [1, 3, 7],
            [0, 2, 3],
            [4, 0, 1],
        ];
        TriMesh::with_flags(
            positions,
            indices,
            TriMeshFlags::ORIENTED
                | TriMeshFlags::MERGE_DUPLICATE_VERTICES
                | TriMeshFlags::FIX_INTERNAL_EDGES,
        )
        .expect("valid cube mesh")
    }

    #[test]
    fn signed_distance_outside_is_positive_and_matches_face_distance() {
        let mesh = cube_trimesh();
        let wrapper = TriangleMeshWrapper::new(&mesh);
        let sd = wrapper.signed_distance(&Point3::new(2., 0., 0.)).unwrap();
        assert!((sd - 1.0).abs() < 1e-9);
    }

    #[test]
    fn signed_distance_inside_is_negative() {
        let mesh = cube_trimesh();
        let wrapper = TriangleMeshWrapper::new(&mesh);
        let sd = wrapper.signed_distance(&Point3::origin()).unwrap();
        assert!((sd - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn signed_distance_on_surface_is_near_zero() {
        let mesh = cube_trimesh();
        let wrapper = TriangleMeshWrapper::new(&mesh);
        let sd = wrapper.signed_distance(&Point3::new(1., 0., 0.)).unwrap();
        assert!(sd.abs() < 1e-9);
    }
    // ─── SDFnWrapper::volume ──────────────────────────────────────────
    //
    // Built directly via `CubicSerendipityDiscretization::new`'s public API
    // with a single cell (dx == domain size) and a CONSTANT signed-distance
    // function, rather than through a real mesh + AABB + padding pipeline.
    // This is deliberately fast (32 node evaluations to build the field) and
    // makes `volume()`'s three branches exactly predictable: a constant
    // field is reproduced EXACTLY everywhere by the discretization's
    // partition-of-unity property, so `sd_center` is known precisely.

    fn constant_sdfn(value: f64) -> CubicSerendipityDiscretization {
        CubicSerendipityDiscretization::new(
            Point3::new(-1., -1., -1.),
            Point3::new(1., 1., 1.),
            None,
            None,
            2.0, // dx == domain size -> single cell
            &move |_p: &Point3<f64>| Ok(value),
            None,
        )
        .unwrap()
    }

    #[test]
    fn sdfn_wrapper_volume_returns_zero_fully_outside_support() {
        let sdfn = constant_sdfn(0.5); // signed distance uniformly 0.5 (far outside)
        let wrapper = SDFnWrapper::new(&sdfn);
        let h = 0.1; // 2h = 0.2 <= 0.5 -> fast "fully outside" path
        let v = wrapper.volume(&Point3::origin(), h, 5).unwrap();
        assert_eq!(v, 0.0);
    }

    #[test]
    fn sdfn_wrapper_volume_returns_full_ball_when_fully_inside_support() {
        let sdfn = constant_sdfn(-0.5); // signed distance uniformly -0.5 (well inside)
        let wrapper = SDFnWrapper::new(&sdfn);
        let h = 0.1; // -h = -0.1 >= -0.5 -> fast "fully inside" path
        let v = wrapper.volume(&Point3::origin(), h, 5).unwrap();
        let expected = ball_volume(h);
        assert!((v - expected).abs() < 1e-12);
    }

    #[test]
    fn sdfn_wrapper_volume_integrates_via_quadrature_on_surface() {
        // sd = 0.0 uniformly: `cubic_extension_fn` is exactly 1.0 everywhere
        // (the `sd <= 0.` branch), so this exercises the real quadrature
        // code path while the expected result is still fully predictable —
        // the full ball volume, same as the "fully inside" fast path, just
        // computed the expensive way.
        let sdfn = constant_sdfn(0.0);
        let wrapper = SDFnWrapper::new(&sdfn);
        let h = 0.1; // 0.0 is neither >= 2h (0.2) nor <= -h (-0.1) -> quadrature
        let v = wrapper.volume(&Point3::origin(), h, 6).unwrap();
        let expected = ball_volume(h);
        assert!(
            (v - expected).abs() / expected < 1e-4,
            "expected ≈{expected}, got {v}"
        );
    }

    #[test]
    fn sdfn_wrapper_volume_treats_out_of_bounds_center_as_zero() {
        // A point far outside the sdfn's domain ([-1,1]^3) makes the initial
        // `sdfn.function(point)` lookup for `sd_center` return
        // `Err(OutOfBounds)`. Per the `// pruned cell → outside band → treat
        // as 0` comment, `volume()` treats this as "outside the support
        // band" and returns `Ok(0.0)` — it does NOT propagate the error.
        let sdfn = constant_sdfn(0.0);
        let wrapper = SDFnWrapper::new(&sdfn);
        let v = wrapper.volume(&Point3::new(100., 0., 0.), 0.1, 5).unwrap();
        assert_eq!(v, 0.0);
    }

    #[test]
    fn sdfn_wrapper_volume_propagates_error_when_integration_sphere_exceeds_domain() {
        // Contrasts with the previous test: here the CENTER lookup succeeds
        // (point is inside the domain), but the quadrature samples points
        // within radius `h` of it — some of which fall outside the domain
        // because the point is close to its boundary. Unlike the center
        // lookup, this error IS propagated, not converted to `Ok(0.0)`.
        let sdfn = constant_sdfn(0.0);
        let wrapper = SDFnWrapper::new(&sdfn);
        let h = 0.5; // large enough that the sphere exceeds the [-1,1]^3 domain
        let result = wrapper.volume(&Point3::new(0.95, 0., 0.), h, 5);
        assert!(matches!(result, Err(EvaluationError::OutOfBounds)));
    }

    // ─── BoundaryType: Normal-/Viscosity-Index-Offset-Konsistenz ────

    #[test]
    fn boundary_index_offset_is_consistent() {
        // Weißbox-Test: baut ein `BoundaryType::StaticBoundary` direkt
        // (ohne teure `add_static_boundary`-Diskretisierung) mit einem
        // linearen SDF (von der Serendipity-Interpolation nahezu exakt
        // reproduziert) und einem konstanten, positiven Volume-Map-Feld,
        // sodass ein einzelner Fluidpartikel nahe der Ebene sd(x)=0
        // zuverlässig sowohl Normal- als auch Viscosity-Nachbarn erzeugt.
        let sd_field = CubicSerendipityDiscretization::new(
            Point3::new(-1., -1., -1.),
            Point3::new(1., 1., 1.),
            None,
            None,
            2.0,                        // einzelne Zelle
            &|p: &Point3<f64>| Ok(p.x), // Ebene bei x = 0
            None,
        )
        .unwrap();
        let volume_map = CubicSerendipityDiscretization::new(
            Point3::new(-1., -1., -1.),
            Point3::new(1., 1., 1.),
            Some(0.),
            None,
            2.0,
            &|_p: &Point3<f64>| Ok(1.0),
            None,
        )
        .unwrap();

        let mut vm = VolumeMapBoundary {
            boundaries: vec![BoundaryType::StaticBoundary {
                signed_distance_field: sd_field,
                volume_map,
                render_mesh: RenderMesh::default(), // ggf. anpassen, s. Hinweis unten
                render_mesh_id: 0,
                boundary_neighbor_list: NeighborList::default(),
                boundary_neighbor_list_viscosity: NeighborList::default(),
                bounding_box: LocalAABB {
                    center: Point3::origin(),
                    half_extents: Vector3::zero(),
                },
            }],
            rest_density_grid_spacing: 0.0,
            kernel_support_radius: 0.0,
            p: vec![],
            v_err: vec![],
            lambda: vec![],
            v_tang: vec![],
            s_p: vec![],
            s_lambda: vec![],
            extra_force: vec![],
            extra_torque: vec![],
            v_trial: vec![],
            omega_trial: vec![],
        };

        let within_range = 0.5;
        let rest_density_grid_spacing = 0.1;
        let mut ns = crate::neighbor_search::SpatialHashing::new(within_range);
        let positions = vec![Point3::new(0.1, 0.0, 0.0)];

        vm.find_boundary_samples(&mut ns, within_range, &positions, rest_density_grid_spacing);

        let boundary = &vm.boundaries[0];
        let n_normal = boundary.boundary_neighbor_list().len();
        let n_visc = boundary.boundary_neighbor_list_viscosity().len();
        assert!(n_normal > 0, "expected at least one normal neighbor sample");
        assert!(
            n_visc > 0,
            "expected at least one viscosity neighbor sample"
        );

        for id in 0..positions.len() {
            for &nb in boundary.get_neighbors(id, RequestMode::Normal) {
                assert!(nb < n_normal, "Normal index {nb} >= {n_normal}");
                let _ = boundary.volume(nb);
                let _ = boundary.position(nb);
            }
            for &nb in boundary.get_neighbors(id, RequestMode::ViscosityAcceleration) {
                assert!(
                    (n_normal..n_normal + n_visc).contains(&nb),
                    "Viscosity index {nb} not in [{n_normal}, {})",
                    n_normal + n_visc
                );
                let _ = boundary.volume(nb);
                let _ = boundary.position(nb);
            }
        }
    }

    #[test]
    fn viscosity_neighbor_velocities_are_computed_per_point_not_hardcoded_to_zero() {
        // Regression test for a bug where the four viscosity-neighbor sample
        // points (`p_visc_1..4`) were correctly positioned via named `let`
        // bindings, but `r.v_vel` was populated with
        // `std::iter::repeat_n(Vector3::zero(), 4)` instead of actually
        // evaluating `b.velocity_at_point_on_boundary(&p_visc_i)` at each point
        // -- silently zeroing every boundary-viscosity velocity, regardless of
        // the boundary's actual motion.
        //
        // Requires a `DynamicBoundary`: for `StaticBoundary`,
        // `velocity_at_point_on_boundary` always returns `Vector3::zero()`
        // unconditionally, so the CORRECT result there is indistinguishable
        // from the bug -- a static fixture would pass under both the buggy and
        // fixed code.
        //
        // Uses BOTH nonzero linear and angular velocity so that (a) an
        // all-zero regression is caught immediately by the per-point equality
        // check below, and (b) a more subtle variant -- e.g. accidentally
        // evaluating `velocity_at_point_on_boundary(&point_on_boundary)` once
        // and reusing it for all four samples instead of each point's own --
        // is also caught, since the four sample points sit at geometrically
        // distinct offsets and nonzero angular velocity makes at least some of
        // their expected velocities differ.
        let sd_field = CubicSerendipityDiscretization::new(
            Point3::new(-1., -1., -1.),
            Point3::new(1., 1., 1.),
            None,
            None,
            2.0,
            &|p: &Point3<f64>| Ok(p.x), // plane at x = 0
            None,
        )
        .unwrap();
        let volume_map = CubicSerendipityDiscretization::new(
            Point3::new(-1., -1., -1.),
            Point3::new(1., 1., 1.),
            Some(0.),
            None,
            2.0,
            &|_p: &Point3<f64>| Ok(1.0),
            None,
        )
        .unwrap();

        let linear_velocity = Vector3::new(1.0, 0.0, 0.0);
        let angular_velocity = Vector3::new(0.0, 0.0, 2.0);
        let center_of_mass = Point3::origin();

        // Identity inertia tensors + identity orientation make the DERIVED
        // `angular_velocity()` (used internally by `velocity_at_point`) equal
        // the raw input `angular_velocity` exactly -- see
        // `RigidBodyMotion::update_derived_variables`: with R = I and
        // I_body = I_body_inv = identity, `inertia_tensor_inv_world` collapses
        // to identity too, so no distortion needs to be accounted for below.
        let state = RigidBodyMotion::new(
            1.0,
            nalgebra::Matrix3::identity(),
            nalgebra::Matrix3::identity(),
            center_of_mass,
            UnitQuaternion::identity(),
            linear_velocity,
            angular_velocity,
        );

        let mut vm = VolumeMapBoundary {
            boundaries: vec![BoundaryType::DynamicBoundary {
                signed_distance_field: sd_field,
                volume_map,
                render_mesh: RenderMesh::default(),
                render_mesh_id: 0,
                state,
                boundary_neighbor_list: NeighborList::default(),
                boundary_neighbor_list_viscosity: NeighborList::default(),
                bounding_box: LocalAABB {
                    center: Point3::new(0.0, 0.0, 0.0),
                    half_extents: Vector3::new(0.0, 0.0, 0.0),
                },
            }],
            rest_density_grid_spacing: 0.0,
            kernel_support_radius: 0.0,
            p: vec![],
            v_err: vec![],
            lambda: vec![],
            v_tang: vec![],
            s_p: vec![],
            s_lambda: vec![],
            extra_force: vec![],
            extra_torque: vec![],
            v_trial: vec![],
            omega_trial: vec![],
        };

        let within_range = 0.5;
        let rest_density_grid_spacing = 0.1;
        let mut ns = crate::neighbor_search::SpatialHashing::new(within_range);
        let positions = vec![Point3::new(0.1, 0.0, 0.0)];

        vm.find_boundary_samples(&mut ns, within_range, &positions, rest_density_grid_spacing);

        let boundary = &vm.boundaries[0];
        let n_normal = boundary.boundary_neighbor_list().len();
        let n_visc = boundary.boundary_neighbor_list_viscosity().len();
        assert_eq!(
            n_visc, 4,
            "expected exactly 4 viscosity sample points (p_visc_1..4) for the \
             single fluid particle in range"
        );

        // Independently recompute the expected velocity at each viscosity
        // sample's ACTUAL reported position, via the standard rigid-body
        // kinematics formula `v_p = v_cm + omega x (p - r_cm)` -- exactly what
        // `velocity_at_point_on_boundary(&p_visc_i)` should evaluate to for
        // each point. Comparing against the ACTUAL returned `pos` (rather than
        // hand-derived expected positions) makes this assertion independent of
        // any signed-distance-field interpolation imprecision -- both sides of
        // the comparison use the same `pos`, so only the VELOCITY computation
        // is actually being tested.
        let expected_velocity_at =
            |p: &Point3<f64>| linear_velocity + angular_velocity.cross(&(p - center_of_mass));

        let mut velocities = Vec::with_capacity(n_visc);
        for local_id in 0..n_visc {
            let global_id = n_normal + local_id;
            let pos = *boundary.position(global_id);
            let vel = *boundary.velocity(global_id);
            let expected = expected_velocity_at(&pos);

            assert!(
                (vel - expected).norm() < 1e-9,
                "viscosity sample {local_id} at {pos:?}: expected velocity {expected:?}, got {vel:?} \
                 -- this is exactly the symptom of hardcoding v_vel to Vector3::zero()"
            );
            velocities.push(vel);
        }

        // Direct symptom of the original bug: every sample's velocity was
        // hardcoded to exactly zero, regardless of the boundary's actual motion.
        assert!(
            velocities.iter().any(|v| v.norm() > 1e-6),
            "all viscosity sample velocities are ~zero -- looks like the \
             std::iter::repeat_n(Vector3::zero(), 4) regression"
        );

        // With nonzero angular velocity and geometrically distinct sample
        // points, the velocities must not all coincide -- guards against a
        // subtler variant that assigns one point's velocity to all four
        // samples instead of each one evaluating its own position.
        let first = velocities[0];
        assert!(
            velocities.iter().any(|v| (v - first).norm() > 1e-9),
            "all four viscosity sample velocities are identical despite nonzero \
             angular velocity and distinct sample positions -- expected each \
             p_visc_i to get its OWN velocity_at_point_on_boundary(&p_visc_i)"
        );
    }
}
