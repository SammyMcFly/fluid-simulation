//! Implicit frictional boundary handling via volume maps
use crate::for_each;
use crate::neighbor_search::NeighborSearch;
use crate::render_info::{
    BoundaryMeshColoring, BoundarySampleColoring, BoundaryVisualization, RenderPose,
};
use crate::sph::boundary_handling::{
    Boundary, BoundaryCheckpoint, BoundaryHandling, ForceOntoBoundary, RequestMode,
    RigidBodyMotion, RigidBodyMotionCheckpoint,
};
use crate::sph::kernel::{CubicBSpline3D, KernelFn};
use crate::sph::setup::input::{DynamicBoundaryDef, StaticBoundaryDef};
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
use std::panic;
use std::slice::SliceIndex;

fn ball_volume(radius: f64) -> f64 {
    4.0 / 3.0 * std::f64::consts::PI * radius.powi(3)
}

#[derive(Debug, Default, Clone)]
pub struct VolumeMapBoundary {
    boundaries: Vec<BoundaryType>,
}

impl VolumeMapBoundary {
    const INTEGRATION_ORDER: usize = 30;
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
    fn new(trimesh: &TriMesh, dx: f64, kernel_support_radius: f64) -> DiscretizedBoundaryFields {
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
        );
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

        let vm = CubicSerendipityDiscretization::new(x_min, x_max, Some(0.), None, dx, &|p| {
            sd_field_wrapped
                .volume(
                    p,
                    kernel_support_radius,
                    VolumeMapBoundary::INTEGRATION_ORDER,
                )
                .map(|v| v.clamp(0.0, ball_volume(kernel_support_radius)))
        });
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
        )
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
        );
        let volume_map = CubicSerendipityDiscretization::new(
            Point3::new(-1., -1., -1.),
            Point3::new(1., 1., 1.),
            Some(0.),
            None,
            2.0,
            &|_p: &Point3<f64>| Ok(1.0),
        );

        let mut vm = VolumeMapBoundary {
            boundaries: vec![BoundaryType::StaticBoundary {
                signed_distance_field: sd_field,
                volume_map,
                render_mesh: RenderMesh::default(), // ggf. anpassen, s. Hinweis unten
                render_mesh_id: 0,
                boundary_neighbor_list: NeighborList::default(),
                boundary_neighbor_list_viscosity: NeighborList::default(),
            }],
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
}
