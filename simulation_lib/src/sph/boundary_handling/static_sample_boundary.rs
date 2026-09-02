//! Implicit frictional boundary handling via volume maps
use crate::for_each;
use crate::neighbor_search::NeighborList;
use crate::neighbor_search::NeighborSearch;
use crate::render_info::RenderPose;
use crate::render_info::{BoundaryMeshColoring, BoundarySampleColoring, BoundaryVisualization};
use crate::sph::boundary_handling::{
    Boundary, BoundaryCheckpoint, BoundaryHandling, ForceOntoBoundary, RequestMode,
    RigidBodyMotion, RigidBodyMotionCheckpoint,
};
use crate::sph::fluid::Len;
use crate::sph::setup::input::DynamicBoundaryDef;
use crate::sph::setup::input::StaticBoundaryDef;
// use crate::sph::boundary_handling::BoundaryParameters;
use crate::sph::kernel::KernelFn;
use crate::utilities::euler_deg_to_quaternion;
use crate::utilities::sampling::sample_triangle_mesh_surface;
use crate::utilities::triangle_mesh::MeshContainer;
use crate::utilities::triangle_mesh::RenderMesh;
use crate::utilities::vector;

use nalgebra::Isometry3;
use nalgebra::{Point3, Vector3};
use parry3d_f64::mass_properties::MassProperties;
use parry3d_f64::shape::Shape;
use rayon::prelude::*;

#[derive(Debug, Default, Clone)]
pub struct StaticSampleBoundary {
    /// Boundary samples
    boundaries: Vec<BoundaryType>,
    // List of boundary neighbors
    // boundary_neighbor_list: NeighborList,
    // /// Boundary parameters
    // params: BoundaryParameters,
}

impl BoundaryHandling for StaticSampleBoundary {
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
        let orientation = euler_deg_to_quaternion(boundary.rotation_euler_deg);
        let local_com_vec = Vector3::new(
            mass_props.local_com[0],
            mass_props.local_com[1],
            mass_props.local_com[2],
        );
        let global_center_of_mass = Point3::from(Vector3::from(boundary.translation))
            + orientation.transform_vector(&local_com_vec);
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
                                std::iter::repeat_n(b.render_mesh_id(), b.positions().len())
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
                    .flat_map(|b| std::iter::repeat_n(b.render_mesh_id(), b.positions().len()))
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
                            b.positions()
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

    fn position(&self, id: usize) -> &Point3<f64> {
        &self.positions()[id]
    }

    fn velocity(&self, id: usize) -> &Vector3<f64> {
        &self.velocities()[id]
    }

    fn volume(&self, id: usize) -> f64 {
        self.volumes()[id]
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
            self.positions(),
            self.positions(),
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
                    self.position(boundary_particle_index),
                    self.position(*boundary_neighbor),
                );
                inverse_volume += K::kernel_function(&r_vec, kernel_support_radius);
            }
            // calculate pseudo volume
            let pseudo_volume = boundary_rest_volume_weighting / inverse_volume;
            self.volumes_mut()[boundary_particle_index] = pseudo_volume;
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

    fn positions(&self) -> &Vec<Point3<f64>> {
        match self {
            Self::StaticBoundary { position, .. } => position,
            Self::DynamicBoundary { position, .. } => position,
        }
    }

    fn positions_mut(&mut self) -> &mut Vec<Point3<f64>> {
        match self {
            Self::StaticBoundary { position, .. } => position,
            Self::DynamicBoundary { position, .. } => position,
        }
    }

    fn velocities(&self) -> &Vec<Vector3<f64>> {
        match self {
            Self::StaticBoundary { velocity, .. } => velocity,
            Self::DynamicBoundary { velocity, .. } => velocity,
        }
    }

    fn velocities_mut(&mut self) -> &mut Vec<Vector3<f64>> {
        match self {
            Self::StaticBoundary { velocity, .. } => velocity,
            Self::DynamicBoundary { velocity, .. } => velocity,
        }
    }

    fn volumes(&self) -> &Vec<f64> {
        match self {
            Self::StaticBoundary { volume, .. } => volume,
            Self::DynamicBoundary { volume, .. } => volume,
        }
    }

    fn volumes_mut(&mut self) -> &mut Vec<f64> {
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
        self.positions().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, UnitQuaternion};

    fn make_dynamic_boundary(position_body: Vec<Point3<f64>>, com: Point3<f64>) -> BoundaryType {
        let len = position_body.len();
        BoundaryType::DynamicBoundary {
            position_body,
            position: vec![Point3::origin(); len],
            velocity: vec![Vector3::zeros(); len],
            volume: vec![0.; len],
            render_mesh: RenderMesh::default(),
            render_mesh_id: 0,
            state: RigidBodyMotion::new(
                1.0,
                Matrix3::identity(),
                Matrix3::identity(),
                com,
                UnitQuaternion::identity(),
                Vector3::zeros(),
                Vector3::zeros(),
            ),
            boundary_neighbor_list: NeighborList::default(),
        }
    }

    fn make_static_boundary(position: Vec<Point3<f64>>) -> BoundaryType {
        let len = position.len();
        BoundaryType::StaticBoundary {
            position,
            velocity: vec![Vector3::zeros(); len],
            volume: vec![0.; len],
            render_mesh: RenderMesh::default(),
            render_mesh_id: 0,
            boundary_neighbor_list: NeighborList::default(),
        }
    }

    // ─── pose / update_positions_and_velocities ────────────────────

    #[test]
    fn static_boundary_pose_is_identity() {
        let boundary = make_static_boundary(vec![Point3::new(1., 2., 3.)]);
        assert_eq!(boundary.pose(), Isometry3::identity());
    }

    #[test]
    fn update_positions_and_velocities_is_noop_for_static() {
        let mut boundary = make_static_boundary(vec![Point3::new(1., 2., 3.)]);
        let before = boundary.positions().clone();
        boundary.update_positions_and_velocities();
        assert_eq!(*boundary.positions(), before);
    }

    #[test]
    fn update_positions_and_velocities_transforms_body_frame_to_world() {
        let com = Point3::new(5., 0., 0.);
        let mut boundary = make_dynamic_boundary(vec![Point3::new(1., 0., 0.)], com);

        boundary.update_positions_and_velocities();

        // Identity orientation, so world position = position_body + com.
        assert_eq!(boundary.positions()[0], Point3::new(6., 0., 0.));
    }

    #[test]
    fn update_positions_and_velocities_computes_rigid_body_velocity() {
        // v(p) = v_cm + omega x (p - com)
        let mut boundary = make_dynamic_boundary(vec![Point3::new(1., 0., 0.)], Point3::origin());
        if let BoundaryType::DynamicBoundary { state, .. } = &mut boundary {
            *state = RigidBodyMotion::new(
                1.0,
                Matrix3::identity(),
                Matrix3::identity(),
                Point3::origin(),
                UnitQuaternion::identity(),
                Vector3::new(1., 0., 0.),
                Vector3::new(0., 0., 2.),
            );
        }

        boundary.update_positions_and_velocities();

        // world position of the sample is (1,0,0); velocity = (1,0,0) + (0,0,2) x (1,0,0) = (1,2,0)
        assert_eq!(boundary.velocities()[0], Vector3::new(1., 2., 0.));
    }

    // ─── Boundary trait impl ────────────────────────────────────────

    #[test]
    fn pos_now_vel_now_volume_delegate_to_fields() {
        let mut boundary =
            make_static_boundary(vec![Point3::new(1., 2., 3.), Point3::new(4., 5., 6.)]);
        boundary.velocities_mut()[0] = Vector3::new(0.1, 0.2, 0.3);
        boundary.volumes_mut()[0] = 0.42;

        assert_eq!(*boundary.position(0), Point3::new(1., 2., 3.));
        assert_eq!(*boundary.velocity(0), Vector3::new(0.1, 0.2, 0.3));
        assert_eq!(boundary.volume(0), 0.42);
    }

    #[test]
    fn static_boundary_is_not_dynamic() {
        let boundary = make_static_boundary(vec![Point3::origin()]);
        assert!(!boundary.is_dynamic());
        assert!(boundary.center_of_mass().is_none());
    }

    #[test]
    fn dynamic_boundary_is_dynamic() {
        let boundary = make_dynamic_boundary(vec![Point3::origin()], Point3::new(1., 1., 1.));
        assert!(boundary.is_dynamic());
        assert_eq!(boundary.center_of_mass(), Some(Point3::new(1., 1., 1.)));
    }

    #[test]
    fn add_acceleration_is_noop_for_static() {
        let mut boundary = make_static_boundary(vec![Point3::origin()]);
        // Should not panic even though there is no rigid-body state.
        boundary.add_acceleration(Vector3::new(0., -9.81, 0.));
    }

    #[test]
    fn add_acceleration_applies_mass_scaled_force_at_center_of_mass() {
        let mut boundary = make_dynamic_boundary(vec![Point3::origin()], Point3::origin());
        boundary.add_acceleration(Vector3::new(0., -9.81, 0.));
        boundary.step_forward_in_time(0.1);

        if let BoundaryType::DynamicBoundary { state, .. } = &boundary {
            assert!(state.velocity_at_cm().y < 0.);
        } else {
            panic!("expected DynamicBoundary");
        }
    }

    // ─── checkpoint / restore ────────────────────────────────────────

    #[test]
    fn checkpoint_state_is_none_for_static() {
        let boundary = make_static_boundary(vec![Point3::origin()]);
        assert!(boundary.checkpoint_state().is_none());
    }

    #[test]
    fn checkpoint_state_is_some_for_dynamic() {
        let boundary = make_dynamic_boundary(vec![Point3::origin()], Point3::new(1., 2., 3.));
        let checkpoint = boundary
            .checkpoint_state()
            .expect("dynamic boundary must checkpoint");
        assert_eq!(checkpoint.center_of_mass, Point3::new(1., 2., 3.));
    }

    #[test]
    fn restore_from_checkpoint_updates_world_position() {
        let mut boundary = make_dynamic_boundary(vec![Point3::new(1., 0., 0.)], Point3::origin());
        boundary.update_positions_and_velocities();
        assert_eq!(boundary.positions()[0], Point3::new(1., 0., 0.));

        let saved = Some(RigidBodyMotionCheckpoint {
            center_of_mass: Point3::new(10., 0., 0.),
            orientation: UnitQuaternion::identity(),
            linear_velocity: Vector3::zeros(),
            angular_momentum: Vector3::zeros(),
            force: Vector3::zeros(),
            torque: Vector3::zeros(),
        });

        boundary.restore_from_checkpoint(&saved);

        // `restore_from_checkpoint` must recompute `position` from the new
        // pose, not just overwrite `state` and leave stale cached data.
        assert_eq!(boundary.positions()[0], Point3::new(11., 0., 0.));
    }

    #[test]
    fn restore_from_checkpoint_is_noop_for_matching_static() {
        let mut boundary = make_static_boundary(vec![Point3::new(1., 2., 3.)]);
        let before = boundary.positions().clone();

        boundary.restore_from_checkpoint(&None);

        assert_eq!(*boundary.positions(), before);
    }

    #[test]
    fn restore_from_checkpoint_type_mismatch_does_not_panic() {
        // A stale checkpoint (e.g. scene changed between save and resume)
        // should be diagnosed, not crash the worker thread.
        let mut static_boundary = make_static_boundary(vec![Point3::origin()]);
        let saved = Some(RigidBodyMotionCheckpoint {
            center_of_mass: Point3::origin(),
            orientation: UnitQuaternion::identity(),
            linear_velocity: Vector3::zeros(),
            angular_momentum: Vector3::zeros(),
            force: Vector3::zeros(),
            torque: Vector3::zeros(),
        });
        static_boundary.restore_from_checkpoint(&saved); // Some for a static boundary

        let mut dynamic_boundary = make_dynamic_boundary(vec![Point3::origin()], Point3::origin());
        dynamic_boundary.restore_from_checkpoint(&None); // None for a dynamic boundary
        // Neither call should panic.
    }

    // ─── position_and_boundary_neighbor_list_mut ────────────────────

    #[test]
    fn position_and_boundary_neighbor_list_mut_returns_consistent_refs() {
        let mut boundary = make_static_boundary(vec![Point3::new(1., 2., 3.)]);
        let (position, neighbor_list) = boundary.position_and_boundary_neighbor_list_mut();
        assert_eq!(position[0], Point3::new(1., 2., 3.));
        assert_eq!(neighbor_list.num_samples(), 0); // freshly default-constructed
    }

    // ─── initialize (volume computation via real neighbor search) ───

    #[test]
    fn initialize_computes_nonzero_volume_for_isolated_samples() {
        // Four samples far enough apart that each is its own only neighbor
        // (distance 0 to itself) — makes the expected pseudo-volume exactly
        // predictable: inverse_volume = W(0, h), so volume = weighting / W(0, h).
        let spacing = 10.0; // far apart relative to kernel_support_radius
        let mut boundary = make_static_boundary(vec![
            Point3::new(0., 0., 0.),
            Point3::new(spacing, 0., 0.),
            Point3::new(0., spacing, 0.),
            Point3::new(0., 0., spacing),
        ]);

        let mut neighbor_search = crate::neighbor_search::SpatialHashing::new(1.0);
        let kernel_support_radius = 1.0;
        let weighting = 2.0;

        boundary.initialize::<crate::sph::kernel::CubicBSpline3D>(
            &mut neighbor_search,
            kernel_support_radius,
            weighting,
        );

        let w0 = crate::sph::kernel::CubicBSpline3D::kernel_function(
            &Vector3::zeros(),
            kernel_support_radius,
        );
        let expected_volume = weighting / w0;

        for i in 0..4 {
            assert!(
                (boundary.volumes()[i] - expected_volume).abs() < 1e-9,
                "sample {i}: expected {expected_volume}, got {}",
                boundary.volumes()[i]
            );
        }
    }

    #[test]
    fn initialize_updates_position_before_computing_volume_for_dynamic() {
        // `initialize` must call `update_positions_and_velocities` BEFORE
        // running the neighbor search / volume computation, so that both
        // operate on correct WORLD-space positions, not the
        // `Point3::origin()` placeholders every `DynamicBoundary` is
        // constructed with — regression test for the "dynamic boundary
        // particles don't collide" bug fixed earlier.
        let com = Point3::new(100., 0., 0.);
        let mut boundary = make_dynamic_boundary(vec![Point3::origin()], com);
        assert_eq!(boundary.positions()[0], Point3::origin()); // placeholder before initialize

        let mut neighbor_search = crate::neighbor_search::SpatialHashing::new(1.0);
        boundary.initialize::<crate::sph::kernel::CubicBSpline3D>(&mut neighbor_search, 1.0, 1.0);

        assert_eq!(boundary.positions()[0], com);
    }
}
