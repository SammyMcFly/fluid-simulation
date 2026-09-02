//! Implicit frictional boundary handling via volume maps
use crate::for_each;
use crate::neighbor_search::NeighborSearch;
use crate::render_info::{
    BoundaryMeshColoring, BoundarySampleColoring, BoundaryVisualization, RenderPose,
};
use crate::setup::input::{DynamicBoundaryDef, StaticBoundaryDef};
use crate::sph::boundary_handling::{
    Boundary, BoundaryCheckpoint, BoundaryHandling, ForceOntoBoundary, RequestMode,
    RigidBodyMotion, RigidBodyMotionCheckpoint,
};
use crate::sph::kernel::{CubicBSpline3D, KernelFn};
use crate::utilities::discretization::{
    CubicSerendipityDiscretization, EvaluationError, gauss_legendre_integrate,
};
use crate::utilities::euler_deg_to_quaternion;
use crate::utilities::triangle_mesh::{MeshContainer, RenderMesh};

use nalgebra::{Isometry3, Point3, Vector3};
use num_traits::Zero;
use parry3d_f64::mass_properties::MassProperties;
use parry3d_f64::math::Pose;
use parry3d_f64::query::PointQuery;
use parry3d_f64::shape::{Shape, TriMesh};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::slice::SliceIndex;

fn ball_volume(radius: f64) -> f64 {
    4.0 / 3.0 * std::f64::consts::PI * radius.powi(3)
}

#[derive(Debug, Default, Clone)]
pub struct VolumeMaps {
    boundaries: Vec<BoundaryType>,
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
        let dx = rest_density_grid_spacing * 4.; // TODO
        let fields = DiscretizedBoundaryFields::new(trimesh, dx, kernel_support_radius);

        let boundary = BoundaryType::StaticBoundary {
            signed_distance_field: fields.signed_distance_field,
            volume_map: fields.volume_map,
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
        let fields = DiscretizedBoundaryFields::new(trimesh, dx, kernel_support_radius);

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

    fn find_boundary_samples(
        &mut self,
        _neighbor_search: &mut impl NeighborSearch,
        within_range: f64,
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
                    let volume = volume.min(ball_volume(within_range));

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
    fn new(trimesh: &TriMesh, dx: f64, kernel_support_radius: f64) -> DiscretizedBoundaryFields {
        #[cfg(feature = "logging")]
        tracing::info!("Start cubic serendipity discretization.");
        let identity = Pose::identity();
        let aabb = trimesh.aabb(&identity);
        let aabb_min = Point3::new(aabb.mins.x, aabb.mins.y, aabb.mins.z);
        let aabb_max = Point3::new(aabb.maxs.x, aabb.maxs.y, aabb.maxs.z);

        let padding_sd = 3.1 * kernel_support_radius;
        let sd_field = TriangleMeshWrapper::new(trimesh);

        let sdfn = CubicSerendipityDiscretization::new(
            aabb_min - Vector3::new(padding_sd, padding_sd, padding_sd),
            aabb_max + Vector3::new(padding_sd, padding_sd, padding_sd),
            Some(-padding_sd),
            Some(padding_sd),
            dx,
            &|p| sd_field.signed_distance(p),
        );
        let sd_field_wrapped = SDFnWrapper::new(&sdfn);

        #[cfg(feature = "logging")]
        tracing::info!("Finished cubic serendipity discretization.");
        #[cfg(feature = "logging")]
        tracing::info!("Start volume integration.");

        let padding_vm = 2. * kernel_support_radius + dx / 6.0; // add dx / 6.0 so that the central difference can be evaluated with h = dx / 6.0
        let vm = CubicSerendipityDiscretization::new(
            aabb_min - Vector3::new(padding_vm, padding_vm, padding_vm),
            aabb_max + Vector3::new(padding_vm, padding_vm, padding_vm),
            Some(0.),
            None,
            dx,
            &|p| {
                sd_field_wrapped
                    .volume(p, kernel_support_radius, VolumeMaps::INTEGRATION_ORDER)
                    .map(|v| v.clamp(0.0, ball_volume(kernel_support_radius)))
            },
        );

        #[cfg(feature = "logging")]
        tracing::info!("Finished volume integration.");

        DiscretizedBoundaryFields {
            signed_distance_field: sdfn,
            volume_map: vm,
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
