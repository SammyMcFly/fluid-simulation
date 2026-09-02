//! Implicit frictional boundary handling via volume maps
use crate::fluid::Len;
use crate::fluid::Positional;
use crate::for_each;
use crate::neighbor_search::NeighborList;
use crate::neighbor_search::NeighborSearch;
use crate::render_info::RenderPose;
use crate::render_info::{BoundaryMeshColoring, BoundarySampleColoring, BoundaryVisualization};
use crate::setup::input::DynamicBoundaryDef;
use crate::setup::input::StaticBoundaryDef;
use crate::sph::boundary_handling::Boundary;
use crate::sph::boundary_handling::BoundaryCheckpoint;
use crate::sph::boundary_handling::BoundaryHandling;
use crate::sph::boundary_handling::ForceOntoBoundary;
use crate::sph::boundary_handling::RequestMode;
use crate::sph::boundary_handling::RigidBodyMotion;
use crate::sph::boundary_handling::RigidBodyMotionCheckpoint;
// use crate::sph::boundary_handling::BoundaryParameters;
use crate::sph::kernel::KernelFn;
use crate::utilities::euler_deg_to_quaternion;
use crate::utilities::sampling::sample_triangle_mesh_surface;
use crate::utilities::triangle_mesh::MeshContainer;
use crate::utilities::triangle_mesh::RenderMesh;
use crate::utilities::vector;

use bincode::Decode;
use bincode::Encode;
use nalgebra::Isometry3;
use nalgebra::{Point3, Vector3};
use parry3d_f64::mass_properties::MassProperties;
use parry3d_f64::shape::Shape;
use parry3d_f64::shape::TriMesh;
use rayon::prelude::*;
use serde::Deserialize;
use serde::Serialize;
use std::slice::SliceIndex;

#[derive(Debug, Default, Clone)]
pub struct SampleBoundary {
    /// Boundary samples
    boundaries: Vec<BoundaryType>,
    // List of boundary neighbors
    // boundary_neighbor_list: NeighborList,
    // /// Boundary parameters
    // params: BoundaryParameters,
}

impl BoundaryHandling for SampleBoundary {
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
        _kernel_support_radius: f64,
    ) {
        // apply transformation
        mesh.transform(
            &boundary.translation,
            &boundary.rotation_euler_deg,
            &boundary.scale,
        );
        let trimesh = mesh.trimesh();
        let position = sample_triangle_mesh_surface(trimesh, rest_density_grid_spacing);
        let len = position.len();
        let boundary = BoundaryType::StaticBoundary {
            // boundary_id: boundary.boundary_id,
            position,
            velocity: vec![Vector3::zeros(); len],
            volume: vec![0.; len],
            render_mesh: mesh.render_mesh(boundary.render_vertex_normals).clone(),
            render_mesh_id: boundary.boundary_id,
            boundary_neighbor_list: NeighborList::default(),
        };

        self.boundaries.push(boundary);
    }

    fn add_dynamic_boundary(
        &mut self,
        mesh: &mut MeshContainer,
        boundary: &DynamicBoundaryDef,
        rest_density_grid_spacing: f64,
        _kernel_support_radius: f64,
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
        // sample the mesh
        let position_body = sample_triangle_mesh_surface(trimesh, rest_density_grid_spacing);
        let len = position_body.len();
        let boundary = BoundaryType::DynamicBoundary {
            position_body,
            position: vec![Point3::origin(); len],
            velocity: vec![Vector3::zeros(); len],
            volume: vec![0.; len],
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
        };

        self.boundaries.push(boundary);
    }

    fn initialize<K: KernelFn>(
        &mut self,
        neighbor_search: &mut impl NeighborSearch,
        kernel_support_radius: f64,
        boundary_rest_volume_weighting: f64,
    ) {
        for b in &mut self.boundaries {
            b.initialize::<K>(
                neighbor_search,
                kernel_support_radius,
                boundary_rest_volume_weighting,
            );
        }
    }

    fn find_boundary_samples(
        &mut self,
        neighbor_search: &mut impl NeighborSearch,
        kernel_support_radius: f64,
        positions: &[Point3<f64>],
        _rest_density_grid_spacing: f64,
    ) {
        for b in &mut self.boundaries {
            let (boundary_position, neighbor_list) = b.position_and_boundary_neighbor_list_mut();
            neighbor_search.find_samples(
                kernel_support_radius,
                positions,
                boundary_position,
                neighbor_list,
            );
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
                        let ids: Vec<u32> = self
                            .boundaries
                            .iter()
                            .flat_map(|b| {
                                std::iter::repeat_n(b.render_mesh_id(), b.position().len())
                            })
                            .collect();
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
            BoundaryVisualization::Samples { coloring, .. } => {
                let render_mesh_ids: Vec<u32> = self
                    .boundaries
                    .iter()
                    .flat_map(|b| std::iter::repeat_n(b.render_mesh_id(), b.position().len()))
                    .collect();
                let max_id = *render_mesh_ids.iter().max().unwrap_or(&0);
                let coloring = match coloring {
                    BoundarySampleColoring::Uniform => BoundarySampleColoring::Uniform,
                    BoundarySampleColoring::BoundaryId { .. } => {
                        BoundarySampleColoring::BoundaryId {
                            ids: render_mesh_ids,
                            max_id,
                        }
                    }
                };
                BoundaryVisualization::Samples {
                    positions: self
                        .boundaries
                        .iter()
                        .flat_map(|b| {
                            b.position()
                                .iter()
                                .map(|pos| [pos.x as f32, pos.y as f32, pos.z as f32])
                        })
                        .collect(),
                    coloring,
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

// // ─── Boundaries ───────────────────────────────────────────────

/// Boundary represented by samples, which are identified by an ID (usize)
#[derive(Debug, Clone)]
pub enum BoundaryType {
    /// Static boundary
    StaticBoundary {
        position: Vec<Point3<f64>>,
        velocity: Vec<Vector3<f64>>,
        /// volume (necessary for sph fluid)
        volume: Vec<f64>,
        render_mesh: RenderMesh,
        render_mesh_id: u32,
        /// List of boundary neighbors
        boundary_neighbor_list: NeighborList,
    },
    /// Dynamic boundary: performs rigid-body motion with two-way coupling with fluid
    DynamicBoundary {
        position_body: Vec<Point3<f64>>,
        position: Vec<Point3<f64>>,
        velocity: Vec<Vector3<f64>>,
        /// volume (necessary for sph fluid)
        volume: Vec<f64>,
        render_mesh: RenderMesh,
        render_mesh_id: u32,
        state: RigidBodyMotion,
        /// List of boundary neighbors
        boundary_neighbor_list: NeighborList,
    },
}

impl Boundary for BoundaryType {
    fn get_neighbors(&self, id: usize, _mode: RequestMode) -> &[usize] {
        self.boundary_neighbor_list().get_neighbors(id)
    }

    fn pos_now(&self, id: usize) -> &Point3<f64> {
        &self.position()[id]
    }

    fn vel_now(&self, id: usize) -> &Vector3<f64> {
        &self.velocity()[id]
    }

    fn volume(&self, id: usize) -> f64 {
        self.volume()[id]
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
    /// Initialize boundary particles with global position, velocity and pseudo volume
    /// taking into account transformations.
    fn initialize<K: KernelFn>(
        &mut self,
        neighbor_search: &mut impl NeighborSearch,
        kernel_support_radius: f64,
        boundary_rest_volume_weighting: f64,
    ) {
        self.update_positions_and_velocities();

        let mut boundary_boundary_neighbor_list = NeighborList::new(self.len());
        neighbor_search.find_samples(
            kernel_support_radius,
            self.position(),
            self.position(),
            &mut boundary_boundary_neighbor_list,
        );
        for boundary_particle_index in 0..self.len() {
            // add inverse volume for every boundary neighbor
            let mut inverse_volume = 0.;
            // get boundary neighbors of boundary particles
            for boundary_neighbor in
                boundary_boundary_neighbor_list.get_neighbors(boundary_particle_index)
            {
                let r_vec = vector(
                    self.pos_now(boundary_particle_index),
                    self.pos_now(*boundary_neighbor),
                );
                inverse_volume += K::kernel_function(&r_vec, kernel_support_radius);
            }
            // calculate pseudo volume
            let pseudo_volume = boundary_rest_volume_weighting / inverse_volume;
            self.volume_mut()[boundary_particle_index] = pseudo_volume;
            // #[cfg(feature = "logging")]
            // tracing::debug!("boundary particle {} has position: {}", boundary_particle_index, self.boundary_particles[boundary_particle_index].pos());
            // #[cfg(feature = "logging")]
            // tracing::debug!("boundary particle {} has mass: {}", boundary_particle_index, self.boundary_particles[boundary_particle_index].mass());
        }
    }

    fn update_positions_and_velocities(&mut self) {
        match self {
            Self::StaticBoundary { .. } => {}
            Self::DynamicBoundary {
                position,
                velocity,
                position_body,
                state,
                ..
            } => {
                for_each!(
                    mut [position, velocity],
                    ref [position_body = position_body],
                    |id, id_pos, id_vel| {
                        *id_pos = state.local_to_world(&position_body[id]);
                        *id_vel = state.velocity_at_cm()
                            + state.angular_velocity().cross(&state.local_to_world_vector(
                                &Vector3::new(
                                    position_body[id].x,
                                    position_body[id].y,
                                    position_body[id].z,
                                ))
                            );
                    }
                );
            }
        }
    }

    /// Current pose mapping body/local frame -> world frame.
    /// Identity for static boundaries, since their fields are already
    /// baked into world space at construction time.
    fn pose(&self) -> Isometry3<f64> {
        match self {
            Self::StaticBoundary { .. } => Isometry3::identity(),
            Self::DynamicBoundary { state, .. } => state.pose(),
        }
    }

    fn position(&self) -> &Vec<Point3<f64>> {
        match self {
            Self::StaticBoundary { position, .. } => position,
            Self::DynamicBoundary { position, .. } => position,
        }
    }

    fn position_mut(&mut self) -> &mut Vec<Point3<f64>> {
        match self {
            Self::StaticBoundary { position, .. } => position,
            Self::DynamicBoundary { position, .. } => position,
        }
    }

    fn velocity(&self) -> &Vec<Vector3<f64>> {
        match self {
            Self::StaticBoundary { velocity, .. } => velocity,
            Self::DynamicBoundary { velocity, .. } => velocity,
        }
    }

    fn velocity_mut(&mut self) -> &mut Vec<Vector3<f64>> {
        match self {
            Self::StaticBoundary { velocity, .. } => velocity,
            Self::DynamicBoundary { velocity, .. } => velocity,
        }
    }

    fn volume(&self) -> &Vec<f64> {
        match self {
            Self::StaticBoundary { volume, .. } => volume,
            Self::DynamicBoundary { volume, .. } => volume,
        }
    }

    fn volume_mut(&mut self) -> &mut Vec<f64> {
        match self {
            Self::StaticBoundary { volume, .. } => volume,
            Self::DynamicBoundary { volume, .. } => volume,
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

    fn position_and_boundary_neighbor_list_mut(
        &mut self,
    ) -> (&Vec<Point3<f64>>, &mut NeighborList) {
        match self {
            Self::StaticBoundary {
                position,
                boundary_neighbor_list,
                ..
            } => (position, boundary_neighbor_list),
            Self::DynamicBoundary {
                position,
                boundary_neighbor_list,
                ..
            } => (position, boundary_neighbor_list),
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
                self.update_positions_and_velocities();
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
        match (&mut *self, saved) {
            (Self::DynamicBoundary { state, .. }, Some(saved)) => {
                state.restore_from_checkpoint(saved);
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
        // Refresh cached position/velocity from the restored rigid-body pose.
        // No-op for `StaticBoundary`.
        self.update_positions_and_velocities();
    }
}

impl Len for BoundaryType {
    fn len(&self) -> usize {
        self.position().len()
    }
}

// impl Positional for StaticSampleBoundary {
//     fn pos_now<I>(&self, id: I) -> &I::Output
//     where
//         I: SliceIndex<[Point3<f64>]>,
//     {
//         &self.position[id]
//     }
// }

// pub fn vel_now<I>(&self, id: I) -> &I::Output
// where
//     I: SliceIndex<[Vector3<f64>]>,
// {
//     &self.velocity[id]
// }

// pub fn set_volume(&mut self, id: usize, volume: f64) {
//     self.volume[id] = volume;
// }

// pub fn volume<I>(&self, id: I) -> &I::Output
// where
//     I: SliceIndex<[f64]>,
// {
//     &self.volume[id]
// }
// }

// impl From<SerBoundary3D> for StaticSampleBoundary {
//     fn from(ser_boundary: SerBoundary3D) -> Self {
//         let len = ser_boundary.position.len();
//         Self {
//             boundary_id: ser_boundary.boundary_id,
//             position: ser_boundary
//                 .position
//                 .iter()
//                 .map(|pos| (*pos).into())
//                 .collect(),
//             velocity: ser_boundary
//                 .velocity
//                 .iter()
//                 .map(|vel| (*vel).into())
//                 .collect(),
//             volume: vec![0.; len],
//             render_meshes: ser_boundary.render_mesh,
//             render_mesh_ids: ser_boundary.render_mesh_id,
//         }
//     }
// }

// /// Compressed and serializable particle in a 3-dimensional context
// #[derive(Debug, Clone, Default, Serialize, Deserialize, Encode, Decode)]
// pub struct SerBoundary3D {
//     pub boundary_id: Vec<u32>,
//     pub position: Vec<[f64; 3]>,
//     pub velocity: Vec<[f64; 3]>,
//     render_mesh: Vec<RenderMesh>,
//     render_mesh_id: Vec<u32>,
// }

// impl From<StaticSampleBoundary> for SerBoundary3D {
//     fn from(boundary: StaticSampleBoundary) -> Self {
//         Self {
//             boundary_id: boundary.boundary_id,
//             position: boundary.position.iter().map(|pos| (*pos).into()).collect(),
//             velocity: boundary.velocity.iter().map(|vel| (*vel).into()).collect(),
//             render_mesh: boundary.render_meshes,
//             render_mesh_id: boundary.render_mesh_ids,
//         }
//     }
// }
