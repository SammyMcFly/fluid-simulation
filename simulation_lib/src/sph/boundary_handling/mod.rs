//! Boundary handling module
use nalgebra::{Point3, Vector3};
use serde::Deserialize;

use crate::{
    neighbor_search::NeighborSearch, render_info::BoundaryVisualization,
    setup::input::VertexNormalRenderOption, sph::kernel::KernelFn,
    utilities::triangle_mesh::MeshContainer,
};

mod static_sample_boundary;
mod volume_maps;

pub use static_sample_boundary::StaticSampleBoundary;
pub use volume_maps::VolumeMaps;

#[derive(Debug, Deserialize)]
pub enum BoundaryHandlingVariant {
    StaticSampleBoundary,
    VolumeMaps,
}

pub trait BoundaryHandling: Send + Sync + Clone {
    fn new() -> Self;

    fn is_empty(&self) -> bool;

    fn add_boundary(
        &mut self,
        mesh: &mut MeshContainer,
        id: u32,
        rest_density_grid_spacing: f64,
        kernel_support_radius: f64,
        render_vertex_normals: VertexNormalRenderOption,
    );

    fn initialize<K: KernelFn>(
        &mut self,
        neighbor_search: &mut impl NeighborSearch,
        kernel_support_radius: f64,
        boundary_rest_volume_weighting: f64,
    );

    fn find_boundary_samples(
        &mut self,
        neighbor_search: &mut impl NeighborSearch,
        within_range: f64,
        positions: &[Point3<f64>],
        rest_density_grid_spacing: f64,
    );

    fn get_neighbors(&self, id: usize, mode: RequestMode) -> &[usize];

    fn pos_now(&self, id: usize) -> &Point3<f64>;

    fn vel_now(&self, id: usize) -> &Vector3<f64>;

    fn volume(&self, id: usize) -> f64;

    // fn density(&self, id: usize) -> &f64;

    fn get_fluid_depth(&self, fluid_volume: f64) -> f64;

    fn get_visualization(&self, selector: &BoundaryVisualization) -> BoundaryVisualization;
}

#[derive(Debug, Clone, Copy, Default)]
pub enum RequestMode {
    #[default]
    Normal,
    ViscosityAcceleration,
}
