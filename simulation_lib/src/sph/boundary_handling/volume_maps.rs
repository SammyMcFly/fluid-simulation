//! Implicit frictional boundary handling via volume maps
use crate::for_each;
use crate::neighbor_search::NeighborSearch;
use crate::render_info::{BoundaryMeshColoring, BoundaryVisualization};
use crate::setup::input::{DynamicBoundaryDef, StaticBoundaryDef};
use crate::sph::boundary_handling::{
    Boundary, BoundaryHandling, ForceOntoBoundary, RequestMode, RigidBodyMotion,
};
use crate::sph::kernel::{CubicBSpline3D, KernelFn};
use crate::utilities::discretization::{
    CubicSerendipityDiscretization, EvaluationError, gauss_legendre_integrate,
};
use crate::utilities::euler_deg_to_quaternion;
use crate::utilities::triangle_mesh::{MeshContainer, RenderMesh, RenderVertex};

use nalgebra::{Isometry3, Point3, Vector3};
use num_traits::Zero;
use parry3d_f64::mass_properties::MassProperties;
use parry3d_f64::math::Pose;
use parry3d_f64::query::PointQuery;
use parry3d_f64::shape::{Shape, TriMesh};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::slice::SliceIndex;
#[cfg(feature = "logging")]
use tracing::{debug, info, warn};

#[derive(Debug, Default, Clone)]
pub struct VolumeMaps {
    boundaries: Vec<BoundaryType>,
    // boundary_neighbor_list: NeighborList,
    // boundary_neighbor_list_viscosity: NeighborList,
    // pub static_boundaries: Vec<StaticBoundary>,
    // pub dynamic_boundaries: Vec<DynamicBoundary>,
}

impl VolumeMaps {
    const INTEGRATION_ORDER: usize = 30;
}

impl BoundaryHandling for VolumeMaps {
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
    ) {
        // apply transformation
        mesh.transform(
            &boundary.translation,
            &boundary.rotation_euler_deg,
            &boundary.scale,
        );
        let trimesh = mesh.trimesh();
        let identity = Pose::identity();
        let dx = rest_density_grid_spacing * 4.; // TODO
        let aabb_min = Point3::new(
            trimesh.aabb(&identity).mins.x,
            trimesh.aabb(&identity).mins.y,
            trimesh.aabb(&identity).mins.z,
        );
        let aabb_max = Point3::new(
            trimesh.aabb(&identity).maxs.x,
            trimesh.aabb(&identity).maxs.y,
            trimesh.aabb(&identity).maxs.z,
        );

        let padding_sd = 3.1 * kernel_support_radius;
        let sd_field = TriangleMeshWrapper::new(trimesh);

        #[cfg(feature = "logging")]
        info!("Start cubic serendipity discretization.");

        let sdfn = CubicSerendipityDiscretization::new(
            aabb_min - Vector3::new(padding_sd, padding_sd, padding_sd),
            aabb_max + Vector3::new(padding_sd, padding_sd, padding_sd),
            Some(-padding_sd),
            Some(padding_sd),
            dx,
            &|p| sd_field.signed_distance(p),
        );
        let sd_field = SDFnWrapper::new(&sdfn);

        #[cfg(feature = "logging")]
        info!("Finished cubic serendipity discretization.");
        #[cfg(feature = "logging")]
        info!("Start volume integration.");

        // cubic serendipity gradient padding
        // let padding_vm = 2. * kernel_support_radius;
        // central difference padding
        let padding_vm = 2. * kernel_support_radius + dx / 6.0;
        let v_max = 4.0 / 3.0 * std::f64::consts::PI * kernel_support_radius.powi(3);
        let vm = CubicSerendipityDiscretization::new(
            aabb_min - Vector3::new(padding_vm, padding_vm, padding_vm),
            aabb_max + Vector3::new(padding_vm, padding_vm, padding_vm),
            Some(0.),
            None,
            dx,
            &|p| {
                sd_field
                    .volume(p, kernel_support_radius, Self::INTEGRATION_ORDER)
                    .map(|v| v.clamp(0.0, v_max))
            },
        );

        #[cfg(feature = "logging")]
        info!("Finished volume integration.");

        let boundary = BoundaryType::StaticBoundary {
            signed_distance_field: sdfn,
            volume_map: vm,
            render_mesh: mesh.render_mesh(boundary.render_vertex_normals).clone(),
            render_mesh_id: boundary.boundary_id,
            boundary_neighbor_list: NeighborList::default(),
            boundary_neighbor_list_viscosity: NeighborList::default(),
        };
        self.boundaries.push(boundary);
    }

    fn add_dynamic_boundary(
        &mut self,
        mesh: &mut MeshContainer,
        boundary: &DynamicBoundaryDef,
        rest_density_grid_spacing: f64,
        kernel_support_radius: f64,
    ) {
        // apply scale
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
        let global_center_of_mass = Point3::new(
            mass_props.local_com[0],
            mass_props.local_com[1],
            mass_props.local_com[2],
        ) + Vector3::from(boundary.translation);
        // discretize the mesh
        let identity = Pose::identity();
        let dx = rest_density_grid_spacing * 8.; // TODO
        let aabb_min = Point3::new(
            trimesh.aabb(&identity).mins.x,
            trimesh.aabb(&identity).mins.y,
            trimesh.aabb(&identity).mins.z,
        );
        let aabb_max = Point3::new(
            trimesh.aabb(&identity).maxs.x,
            trimesh.aabb(&identity).maxs.y,
            trimesh.aabb(&identity).maxs.z,
        );

        let padding_sd = 3.1 * kernel_support_radius;
        let sd_field = TriangleMeshWrapper::new(trimesh);

        #[cfg(feature = "logging")]
        info!("Start cubic serendipity discretization.");

        let sdfn = CubicSerendipityDiscretization::new(
            aabb_min - Vector3::new(padding_sd, padding_sd, padding_sd),
            aabb_max + Vector3::new(padding_sd, padding_sd, padding_sd),
            Some(-padding_sd),
            Some(padding_sd),
            dx,
            &|p| sd_field.signed_distance(p),
        );
        let sd_field = SDFnWrapper::new(&sdfn);

        #[cfg(feature = "logging")]
        info!("Finished cubic serendipity discretization.");
        #[cfg(feature = "logging")]
        info!("Start volume integration.");

        let padding_vm = 2. * kernel_support_radius;
        let v_max = 4.0 / 3.0 * std::f64::consts::PI * kernel_support_radius.powi(3);
        let vm = CubicSerendipityDiscretization::new(
            aabb_min - Vector3::new(padding_vm, padding_vm, padding_vm),
            aabb_max + Vector3::new(padding_vm, padding_vm, padding_vm),
            Some(0.),
            None,
            dx,
            &|p| {
                sd_field
                    .volume(p, kernel_support_radius, Self::INTEGRATION_ORDER)
                    .map(|v| v.clamp(0.0, v_max))
            },
        );

        #[cfg(feature = "logging")]
        info!("Finished volume integration.");

        let boundary = BoundaryType::DynamicBoundary {
            signed_distance_field: sdfn,
            volume_map: vm,
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
                euler_deg_to_quaternion(boundary.rotation_euler_deg),
                Vector3::from(boundary.velocity),
                Vector3::from(boundary.angular_velocity),
            ),
            boundary_neighbor_list: NeighborList::default(),
            boundary_neighbor_list_viscosity: NeighborList::default(),
        };
        self.boundaries.push(boundary);
    }

    fn initialize<K: KernelFn>(
        &mut self,
        _neighbor_search: &mut impl NeighborSearch,
        _kernel_support_radius: f64,
        _boundary_rest_volume_weighting: f64,
    ) {
    }

    // fn find_boundary_samples(
    //     &mut self,
    //     _neighbor_search: &mut impl NeighborSearch,
    //     within_range: f64,
    //     positions: &[Point3<f64>],
    //     rest_density_grid_spacing: f64,
    // ) {
    //     let num_samples = positions.len();
    //     self.boundary_neighbor_list.resize(num_samples);
    //     self.boundary_neighbor_list_viscosity.resize(num_samples);
    //     self.boundary_neighbor_list.clear();
    //     self.boundary_neighbor_list_viscosity.clear();

    //     struct PerParticleNeighbors {
    //         pos: Vec<Point3<f64>>,
    //         vel: Vec<Vector3<f64>>,
    //         vol: Vec<f64>,
    //         v_pos: Vec<Point3<f64>>,
    //         v_vel: Vec<Vector3<f64>>,
    //         v_vol: Vec<f64>,
    //     }

    //     let boundaries = &self.boundaries;

    //     #[cfg(not(feature = "parallel"))]
    //     let pos_iter = positions.iter();
    //     #[cfg(feature = "parallel")]
    //     let pos_iter = positions.par_iter();
    //     let results: Vec<PerParticleNeighbors> = pos_iter
    //         .map(|pos| {
    //             let mut r = PerParticleNeighbors {
    //                 pos: Vec::new(),
    //                 vel: Vec::new(),
    //                 vol: Vec::new(),
    //                 v_pos: Vec::new(),
    //                 v_vel: Vec::new(),
    //                 v_vol: Vec::new(),
    //             };

    //             for b in boundaries {

    //                 let signed_distance = if let Ok(sd) = b.signed_distance_field.function(pos)
    //                     && sd < within_range
    //                 {
    //                     sd
    //                 } else {
    //                     continue;
    //                 };

    //                 let signed_distance_gradient = if let Ok(sdg) =
    //                     b.signed_distance_field.gradient(pos)
    //                 { // Skip particles with unphysical of signed distance gradient: For SDF the eikonal condition should hold: ‖∇d‖ ≈ 1
    //                     if sdg.norm() > 0.5 && sdg.norm() < 2.0 {
    //                         sdg
    //                     } else {
    //                         #[cfg(feature = "logging")]
    //                         warn!(
    //                             "Skipping particle with unphysical signed distance gradient: ‖∇d‖ = {}",
    //                             sdg.norm()
    //                         );
    //                         continue;
    //                     }
    //                 } else {
    //                     continue;
    //                 };

    //                 let volume = if let Ok(v) = b.volume_map.function(pos)
    //                     && v > 0.
    //                 { // Skip particles with very small volume or negative volume due to cubic serendipity interpolation
    //                     v
    //                 } else {
    //                     continue;
    //                 };

    //                 // Clamp volume to v_max range to avoid excessive volumes due to cubic serendipity interpolation
    //                 let v_max = 4.0 / 3.0 * std::f64::consts::PI * within_range.powi(3);
    //                 let volume = volume.min(v_max);

    //                 // let d = signed_distance.abs();
    //                 let d = (signed_distance + 0.25 * rest_density_grid_spacing)
    //                     .max(rest_density_grid_spacing);
    //                 let point_on_boundary =
    //                     pos - signed_distance_gradient / signed_distance_gradient.norm() * d;

    //                 r.pos.push(point_on_boundary);
    //                 r.vel.push(Vector3::zero());
    //                 r.vol.push(volume);

    //                 #[cfg(feature = "logging")]
    //                 debug!(
    //                     "pos: {}, sd: {}, sdg: {:?}, vol: {}, point_ob: {:?}",
    //                     pos, signed_distance, signed_distance_gradient, volume, point_on_boundary
    //                 );

    //                 let vec_p_to_boundary = point_on_boundary - pos;
    //                 let mut vec_temp = Vector3::new(1., 0., 0.);
    //                 if (vec_p_to_boundary / vec_p_to_boundary.norm())
    //                     .dot(&vec_temp)
    //                     .abs()
    //                     > 0.9
    //                 {
    //                     vec_temp = Vector3::new(0., 1., 0.);
    //                 }
    //                 let t1 = vec_p_to_boundary.cross(&vec_temp);
    //                 let t2 = vec_p_to_boundary.cross(&t1);
    //                 let t1 = t1 / t1.norm();
    //                 let t2 = t2 / t2.norm();
    //                 let d = 0.5 * within_range;

    //                 r.v_pos.extend_from_slice(&[
    //                     point_on_boundary + d * t1,
    //                     point_on_boundary - d * t1,
    //                     point_on_boundary + d * t2,
    //                     point_on_boundary - d * t2,
    //                 ]);
    //                 r.v_vel.extend(std::iter::repeat_n(Vector3::zero(), 4));
    //                 r.v_vol.extend_from_slice(&[0.25 * volume; 4]);
    //             }
    //             r
    //         })
    //         .collect();

    //     // Sequential write-back
    //     let (neighbor_pos, neighbor_vel, neighbor_vol, neighbor_indices) =
    //         self.boundary_neighbor_list.neighbors_mut();
    //     let (neighbor_v_pos, neighbor_v_vel, neighbor_v_vol, neighbor_v_indices) =
    //         self.boundary_neighbor_list_viscosity.neighbors_mut();

    //     for (i, r) in results.iter().enumerate() {
    //         for (j, ((pos, vel), vol)) in r.pos.iter().zip(&r.vel).zip(&r.vol).enumerate() {
    //             neighbor_pos[i].push(*pos);
    //             neighbor_vel[i].push(*vel);
    //             neighbor_vol[i].push(*vol);
    //             neighbor_indices[i].push(j);
    //         }

    //         for j in 0..(r.v_pos.len() / 4) {
    //             let local_start = 4 * j;
    //             let base = local_start;
    //             for k in 0..4 {
    //                 neighbor_v_pos[i].push(r.v_pos[4 * j + k]);
    //                 neighbor_v_vel[i].push(r.v_vel[4 * j + k]);
    //                 neighbor_v_vol[i].push(r.v_vol[4 * j + k]);
    //             }
    //             neighbor_v_indices[i].extend_from_slice(&[base, base + 1, base + 2, base + 3]);
    //         }
    //     }

    //     self.boundary_neighbor_list.flatten(0);
    //     self.boundary_neighbor_list_viscosity
    //         .flatten(self.boundary_neighbor_list.len());
    // }
    fn find_boundary_samples(
        &mut self,
        _neighbor_search: &mut impl NeighborSearch,
        within_range: f64,
        positions: &[Point3<f64>],
        rest_density_grid_spacing: f64,
    ) {
        let num_samples = positions.len();

        for b in &mut self.boundaries {
            b.boundary_neighbor_list_mut().resize(num_samples);
            b.boundary_neighbor_list_viscosity_mut().resize(num_samples);
            b.boundary_neighbor_list_mut().clear();
            b.boundary_neighbor_list_viscosity_mut().clear();

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
                        && sd < within_range
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
                            warn!(
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
                    let v_max = 4.0 / 3.0 * std::f64::consts::PI * within_range.powi(3);
                    let volume = volume.min(v_max);

                    // let d = signed_distance.abs();
                    let d = (signed_distance + 0.25 * rest_density_grid_spacing)
                        .max(rest_density_grid_spacing);
                    let point_on_boundary =
                        pos - signed_distance_gradient / signed_distance_gradient.norm() * d;
                    let velocity_at_boundary = b.velocity_at_point_on_boundary(&point_on_boundary);

                    r.pos.push(point_on_boundary);
                    r.vel.push(velocity_at_boundary);
                    r.vol.push(volume);

                    #[cfg(feature = "logging")]
                    debug!(
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
                    let d = 0.5 * within_range;

                    r.v_pos.extend_from_slice(&[
                        point_on_boundary + d * t1,
                        point_on_boundary - d * t1,
                        point_on_boundary + d * t2,
                        point_on_boundary - d * t2,
                    ]);
                    r.v_vel.extend(std::iter::repeat_n(Vector3::zero(), 4));
                    r.v_vol.extend_from_slice(&[0.25 * volume; 4]);

                r
            })
            .collect();

            // Sequential write-back
            {
                let (neighbor_pos, neighbor_vel, neighbor_vol, neighbor_indices) =
                    b.boundary_neighbor_list_mut().neighbors_mut();
                for (i, r) in results.iter().enumerate() {
                    for (j, ((pos, vel), vol)) in r.pos.iter().zip(&r.vel).zip(&r.vol).enumerate() {
                        neighbor_pos[i].push(*pos);
                        neighbor_vel[i].push(*vel);
                        neighbor_vol[i].push(*vol);
                        neighbor_indices[i].push(j);
                    }
                }
            }
            {
                let (neighbor_v_pos, neighbor_v_vel, neighbor_v_vol, neighbor_v_indices) =
                    b.boundary_neighbor_list_viscosity_mut().neighbors_mut();

                for (i, r) in results.iter().enumerate() {
                    for j in 0..(r.v_pos.len() / 4) {
                        let local_start = 4 * j;
                        let base = local_start;
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
            }

            b.boundary_neighbor_list_mut().flatten(0);
            let len = b.boundary_neighbor_list().len();
            b.boundary_neighbor_list_viscosity_mut().flatten(len);
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
                        .map(|b| b.render_mesh_world())
                        .collect(),
                    coloring,
                }
            }
            BoundaryVisualization::Samples { .. } => {
                panic!("Cannot provide samples as there are none.")
            }
        }
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
        integration_order: usize,
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

        gauss_legendre_integrate(
            &|p| match self.sdfn.function(p) {
                Ok(sd) => Ok(cubic_extension_fn(sd, kernel_support_radius)),
                Err(e) => Err(e),
            },
            point,
            kernel_support_radius,
            integration_order,
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

#[derive(Debug, Clone)]
struct NeighborList {
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

impl Default for NeighborList {
    fn default() -> Self {
        Self::new(0)
    }
}

impl NeighborList {
    fn new(len: usize) -> Self {
        Self {
            positions: Vec::new(),
            velocities: Vec::new(),
            volumes: Vec::new(),
            indices: vec![usize::default(); len],
            offsets: vec![usize::default(); len + 1],
            unflattened_positions: vec![Vec::new(); len],
            unflattened_velocities: vec![Vec::new(); len],
            unflattened_volumes: vec![Vec::new(); len],
            unflattened_indices: vec![Vec::new(); len],
        }
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn resize(&mut self, len: usize) {
        // self.positions.resize(len, Point3::origin());
        // self.velocities.resize(len, Vector3::zero());
        // self.volumes.resize(len, 0.);
        // self.indices.resize(len, usize::default());
        // self.offsets.resize(len + 1, usize::default());
        self.unflattened_positions.resize(len, Vec::new());
        self.unflattened_velocities.resize(len, Vec::new());
        self.unflattened_volumes.resize(len, Vec::new());
        self.unflattened_indices.resize(len, Vec::new());
    }

    pub fn clear(&mut self) {
        self.positions.clear();
        self.velocities.clear();
        self.volumes.clear();
        self.indices.clear();
        self.offsets.clear();
        for_each!(
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

    /// Get mutable reference to unflattened neighbor list: one Vec<usize> per sample
    ///
    /// Contract: Always call flatten after updating data in unflattened array
    pub fn neighbors_mut(
        &mut self,
    ) -> (
        &mut Vec<Vec<Point3<f64>>>,
        &mut Vec<Vec<Vector3<f64>>>,
        &mut Vec<Vec<f64>>,
        &mut Vec<Vec<usize>>,
    ) {
        (
            &mut self.unflattened_positions,
            &mut self.unflattened_velocities,
            &mut self.unflattened_volumes,
            &mut self.unflattened_indices,
        )
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

    /// Get indices of neighbor of sample with identifier 'id'
    pub fn get_neighbors(&self, id: usize) -> &[usize] {
        &self.indices[self.offsets[id]..self.offsets[id + 1]]
    }

    fn pos_now<I>(&self, id: I) -> &I::Output
    where
        I: SliceIndex<[Point3<f64>]>,
    {
        &self.positions[id]
    }

    fn vel_now<I>(&self, id: I) -> &I::Output
    where
        I: SliceIndex<[Vector3<f64>]>,
    {
        &self.velocities[id]
    }

    fn volume<I>(&self, id: I) -> &I::Output
    where
        I: SliceIndex<[f64]>,
    {
        &self.volumes[id]
    }
}

// // ─── Boundaries ───────────────────────────────────────────────

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
    },
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

    fn pos_now(&self, id: usize) -> &Point3<f64> {
        let num_neighbors = self.boundary_neighbor_list().len();
        if id < num_neighbors {
            self.boundary_neighbor_list().pos_now(id)
        } else {
            self.boundary_neighbor_list_viscosity()
                .pos_now(id - num_neighbors)
        }
    }

    fn vel_now(&self, id: usize) -> &Vector3<f64> {
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
                force: state.mass * acceleration,
                force_location: state.center_of_mass,
            }),
        };

        if let Some(force) = force {
            self.add_force_onto_boundary(force);
        }
    }

    fn center_of_mass(&self) -> Option<Point3<f64>> {
        match self {
            Self::StaticBoundary { .. } => None,
            Self::DynamicBoundary { state, .. } => Some(state.center_of_mass),
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

    /// Render mesh transformed into the current WORLD-space pose.
    ///
    /// Static boundaries already have their geometry baked into world space
    /// at construction time. Dynamic boundaries store their render mesh in
    /// the body/local frame (centered at the center of mass), so the
    /// current rigid-body pose must be reapplied every time it's requested.
    fn render_mesh_world(&self) -> RenderMesh {
        match self {
            Self::StaticBoundary { render_mesh, .. } => render_mesh.clone(),
            Self::DynamicBoundary {
                render_mesh, state, ..
            } => {
                let pose = state.pose();
                let vertices = render_mesh
                    .vertices
                    .iter()
                    .map(|v| {
                        let p_local = Point3::new(v.position[0], v.position[1], v.position[2]);
                        let n_local = Vector3::new(v.normal[0], v.normal[1], v.normal[2]);
                        let p_world = pose.transform_point(&p_local);
                        let n_world = pose.rotation.transform_vector(&n_local);
                        RenderVertex {
                            position: [p_world.x, p_world.y, p_world.z],
                            normal: [n_world.x, n_world.y, n_world.z],
                        }
                    })
                    .collect();
                RenderMesh {
                    vertices,
                    indices: render_mesh.indices.clone(),
                }
            }
        }
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
}

// #[test]
// fn boundary_index_offset_is_consistent() {
//     let mut vm = /* VolumeMaps mit mind. 1 Boundary aufbauen */;
//     let positions = /* ein paar Testpartikel nahe der Wand */;
//     vm.find_boundary_samples(&mut ns, within_range, &positions, dx);

//     let n_normal = vm.boundary_neighbor_list.len();
//     let n_visc   = vm.boundary_neighbor_list_viscosity.len();

//     for id in 0..positions.len() {
//         // Normal-Indizes müssen in [0, n_normal) liegen
//         for &nb in vm.get_neighbors(id, RequestMode::Normal) {
//             assert!(nb < n_normal, "Normal-Index {nb} >= {n_normal}");
//             let _ = vm.volume(nb);   // darf nicht paniken
//             let _ = vm.pos_now(nb);
//         }
//         // Viskositäts-Indizes müssen in [n_normal, n_normal + n_visc) liegen
//         for &nb in vm.get_neighbors(id, RequestMode::ViscosityAcceleration) {
//             assert!(
//                 (n_normal..n_normal + n_visc).contains(&nb),
//                 "Visc-Index {nb} nicht in [{n_normal}, {})",
//                 n_normal + n_visc
//             );
//             let _ = vm.volume(nb);
//             let _ = vm.pos_now(nb);
//         }
//     }
// }
