/// Boundary handling module
use nalgebra::{Point3, Vector3};
use parry3d_f64::shape::TriMesh;
use serde::Deserialize;
use std::slice::SliceIndex;

use crate::{
    neighbor_search::NeighborSearch, render_info::BoundaryVisualization, sph::kernel::KernelFn,
};

mod static_sample_boundary;
// mod volume_maps;

pub use static_sample_boundary::StaticSampleBoundary;
// pub use volume_maps::VolumeMaps;

#[derive(Debug, Deserialize)]
pub enum BoundaryHandlingVariant {
    StaticSampleBoundary,
    // VolumeMaps,
}

pub trait BoundaryHandling: Send + Sync {
    fn new() -> Self;

    fn is_empty(&self) -> bool;

    fn add_boundary(
        &mut self,
        mesh: &TriMesh,
        id: u32,
        rest_density_grid_spacing: f64,
        // boundary_params: BoundaryParameters,
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
    );

    fn get_neighbors(&self, id: usize) -> &[usize];

    fn pos_now<I>(&self, id: I) -> &I::Output
    where
        I: SliceIndex<[Point3<f64>]>;

    fn vel_now<I>(&self, id: I) -> &I::Output
    where
        I: SliceIndex<[Vector3<f64>]>;

    fn volume<I>(&self, id: I) -> &I::Output
    where
        I: SliceIndex<[f64]>;

    // fn density(&self, id: usize) -> &f64;

    fn get_fluid_depth(&self, fluid_volume: f64) -> f64;

    fn get_visualization(&self, selector: &BoundaryVisualization) -> BoundaryVisualization;
}

// #[derive(Debug, Default)]
// struct BoundaryParameters {
//     density: Vec<f64>,
// }
