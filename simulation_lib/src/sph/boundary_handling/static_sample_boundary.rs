use bincode::Decode;
use bincode::Encode;
/// Implicit frictional boundary handling via volume maps
use nalgebra::{Point3, Vector3};
use parry3d_f64::shape::TriMesh;
use serde::Deserialize;
use serde::Serialize;
use std::slice::SliceIndex;

use crate::fluid::Len;
use crate::fluid::Positional;
use crate::neighbor_search::NeighborList;
use crate::neighbor_search::NeighborSearch;
use crate::render_info::{BoundaryMeshColoring, BoundarySampleColoring, BoundaryVisualization};
use crate::sph::boundary_handling::BoundaryHandling;
use crate::sph::boundary_handling::RequestMode;
// use crate::sph::boundary_handling::BoundaryParameters;
use crate::sph::kernel::KernelFn;
use crate::utilities::sampling::sample_triangle_mesh_surface;
use crate::utilities::triangle_mesh::RenderMesh;
use crate::utilities::vector;

#[derive(Debug, Default)]
pub struct StaticSampleBoundary {
    /// Boundary samples
    boundary: SampleBoundary3D,
    /// List of boundary neighbors
    boundary_neighbor_list: NeighborList,
    // /// Boundary parameters
    // params: BoundaryParameters,
}

impl BoundaryHandling for StaticSampleBoundary {
    fn new() -> Self {
        Self::default()
    }

    fn is_empty(&self) -> bool {
        self.boundary.is_empty()
    }

    fn add_boundary(
        &mut self,
        boundary: &TriMesh,
        boundary_id: u32,
        rest_density_grid_spacing: f64,
        kernel_support_radius: f64,
    ) {
        self.boundary
            .add_boundary(boundary, boundary_id, rest_density_grid_spacing);
        self.boundary_neighbor_list.resize(self.boundary.len());
    }

    fn initialize<K: KernelFn>(
        &mut self,
        neighbor_search: &mut impl NeighborSearch,
        kernel_support_radius: f64,
        boundary_rest_volume_weighting: f64,
    ) {
        self.init_boundary_volume::<K>(
            neighbor_search,
            kernel_support_radius,
            boundary_rest_volume_weighting,
        );
    }

    fn find_boundary_samples(
        &mut self,
        neighbor_search: &mut impl NeighborSearch,
        kernel_support_radius: f64,
        positions: &[Point3<f64>],
    ) {
        neighbor_search.find_samples(
            kernel_support_radius,
            positions,
            &self.boundary.position,
            &mut self.boundary_neighbor_list,
        );
    }

    fn get_neighbors(&self, id: usize, _mode: RequestMode) -> &[usize] {
        self.boundary_neighbor_list.get_neighbors(id)
    }

    fn pos_now(&self, id: usize) -> &Point3<f64> {
        self.boundary.pos_now(id)
    }

    fn vel_now(&self, id: usize) -> &Vector3<f64> {
        self.boundary.vel_now(id)
    }

    fn volume(&self, id: usize) -> f64 {
        *self.boundary.volume(id)
    }

    // fn density(&self, id: usize) -> &f64 {
    //     &self.params.density[self.boundary.boundary_id[id] as usize]
    // }

    fn get_fluid_depth(&self, fluid_volume: f64) -> f64 {
        0.
    }

    fn get_visualization(&self, selector: &BoundaryVisualization) -> BoundaryVisualization {
        match selector {
            BoundaryVisualization::TriangleMesh { coloring, .. } => {
                let coloring = match coloring {
                    BoundaryMeshColoring::Original => BoundaryMeshColoring::Original,
                    BoundaryMeshColoring::Uniform => BoundaryMeshColoring::Uniform,
                    BoundaryMeshColoring::BoundaryId { .. } => BoundaryMeshColoring::BoundaryId {
                        ids: self.boundary.render_mesh_ids.clone(),
                        max_id: *self.boundary.render_mesh_ids.iter().max().unwrap_or(&0),
                    },
                };
                BoundaryVisualization::TriangleMesh {
                    meshes: self.boundary.render_meshes.clone(),
                    coloring,
                }
            }
            BoundaryVisualization::Samples { coloring, .. } => {
                let coloring = match coloring {
                    BoundarySampleColoring::Uniform => BoundarySampleColoring::Uniform,
                    BoundarySampleColoring::BoundaryId { .. } => {
                        BoundarySampleColoring::BoundaryId {
                            ids: self.boundary.boundary_id.clone(),
                            max_id: *self.boundary.boundary_id.iter().max().unwrap_or(&0),
                        }
                    }
                };
                BoundaryVisualization::Samples {
                    positions: self
                        .boundary
                        .position
                        .iter()
                        .map(|pos| [pos.x as f32, pos.y as f32, pos.z as f32])
                        .collect(),
                    coloring,
                }
            }
        }
    }
}

impl StaticSampleBoundary {
    /// Calculate and set pseudo mass of all boundary particles
    fn init_boundary_volume<K: KernelFn>(
        &mut self,
        neighbor_search: &mut impl NeighborSearch,
        kernel_support_radius: f64,
        boundary_rest_volume_weighting: f64,
    ) {
        let mut boundary_boundary_neighbor_list = NeighborList::new(self.boundary.len());
        neighbor_search.find_samples(
            kernel_support_radius,
            &self.boundary.position,
            &self.boundary.position,
            &mut boundary_boundary_neighbor_list,
        );
        for boundary_particle_index in 0..self.boundary.len() {
            // add inverse volume for every boundary neighbor
            let mut inverse_volume = 0.;
            // get boundary neighbors of boundary particles
            for boundary_neighbor in
                boundary_boundary_neighbor_list.get_neighbors(boundary_particle_index)
            {
                let r_vec = vector(
                    self.boundary.pos_now(boundary_particle_index),
                    self.boundary.pos_now(*boundary_neighbor),
                );
                inverse_volume += K::kernel_function(&r_vec, kernel_support_radius);
            }
            // calculate mass with rest density of fluid
            let pseudo_volume = boundary_rest_volume_weighting / inverse_volume;
            self.boundary
                .set_volume(boundary_particle_index, pseudo_volume);
            // #[cfg(feature = "logging")]
            // debug!("boundary particle {} has position: {}", boundary_particle_index, self.boundary_particles[boundary_particle_index].pos());
            // #[cfg(feature = "logging")]
            // debug!("boundary particle {} has mass: {}", boundary_particle_index, self.boundary_particles[boundary_particle_index].mass());
        }
    }
}

/// Boundary represented by samples, which are identified by an ID (usize)
#[derive(Debug, Clone, Default)]
pub struct SampleBoundary3D {
    boundary_id: Vec<u32>,
    position: Vec<Point3<f64>>,
    velocity: Vec<Vector3<f64>>,
    /// volume (necessary for sph fluid)
    volume: Vec<f64>,
    render_meshes: Vec<RenderMesh>,
    render_mesh_ids: Vec<u32>,
}

impl Len for SampleBoundary3D {
    fn len(&self) -> usize {
        self.position.len()
    }
}

impl Positional for SampleBoundary3D {
    fn pos_now<I>(&self, id: I) -> &I::Output
    where
        I: SliceIndex<[Point3<f64>]>,
    {
        &self.position[id]
    }
}

impl SampleBoundary3D {
    pub fn add_boundary(
        &mut self,
        boundary: &TriMesh,
        boundary_id: u32,
        rest_density_grid_spacing: f64,
    ) {
        let position = sample_triangle_mesh_surface(boundary, rest_density_grid_spacing);
        let len = position.len();
        let boundary = Self {
            boundary_id: vec![boundary_id; len],
            position,
            velocity: vec![Vector3::zeros(); len],
            volume: vec![0.; len],
            render_meshes: vec![RenderMesh::from_trimesh(boundary, boundary_id)],
            render_mesh_ids: vec![boundary_id],
        };
        self.extend(boundary);
    }

    fn extend(&mut self, other: Self) {
        self.boundary_id.extend(other.boundary_id);
        self.position.extend(other.position);
        self.velocity.extend(other.velocity);
        self.volume.extend(other.volume);
        self.render_meshes.extend(other.render_meshes);
        self.render_mesh_ids.extend(other.render_mesh_ids);
    }

    pub fn vel_now<I>(&self, id: I) -> &I::Output
    where
        I: SliceIndex<[Vector3<f64>]>,
    {
        &self.velocity[id]
    }

    pub fn set_volume(&mut self, id: usize, volume: f64) {
        self.volume[id] = volume;
    }

    pub fn volume<I>(&self, id: I) -> &I::Output
    where
        I: SliceIndex<[f64]>,
    {
        &self.volume[id]
    }
}

impl From<SerBoundary3D> for SampleBoundary3D {
    fn from(ser_boundary: SerBoundary3D) -> Self {
        let len = ser_boundary.position.len();
        Self {
            boundary_id: ser_boundary.boundary_id,
            position: ser_boundary
                .position
                .iter()
                .map(|pos| (*pos).into())
                .collect(),
            velocity: ser_boundary
                .velocity
                .iter()
                .map(|vel| (*vel).into())
                .collect(),
            volume: vec![0.; len],
            render_meshes: ser_boundary.render_mesh,
            render_mesh_ids: ser_boundary.render_mesh_id,
        }
    }
}

/// Compressed and serializable particle in a 3-dimensional context
#[derive(Debug, Clone, Default, Serialize, Deserialize, Encode, Decode)]
pub struct SerBoundary3D {
    pub boundary_id: Vec<u32>,
    pub position: Vec<[f64; 3]>,
    pub velocity: Vec<[f64; 3]>,
    render_mesh: Vec<RenderMesh>,
    render_mesh_id: Vec<u32>,
}

impl From<SampleBoundary3D> for SerBoundary3D {
    fn from(boundary: SampleBoundary3D) -> Self {
        Self {
            boundary_id: boundary.boundary_id,
            position: boundary.position.iter().map(|pos| (*pos).into()).collect(),
            velocity: boundary.velocity.iter().map(|vel| (*vel).into()).collect(),
            render_mesh: boundary.render_meshes,
            render_mesh_id: boundary.render_mesh_ids,
        }
    }
}
