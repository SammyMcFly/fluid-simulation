//! simulation info module
//!
//! Contains info structs about a simulation and a time step
use crate::measurement::Measurement;
use crate::sph::SPHSystem;
use crate::sph::boundary_handling::BoundaryHandlingVariant;
use crate::sph::setup::input::Parameters;
use crate::sph::setup::input::Procedures;
use crate::utilities::triangle_mesh::RenderMesh;

use bincode::{Decode, Encode};
use nalgebra::Isometry3;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct SimulationParameters {
    /// Particle size
    pub particle_diameter: f32,
    /// Light position
    pub light_position: [f32; 3],
    /// maximum buffer length
    pub buffer_length_limit: usize,
    /// Flag that is true if a measurement is taken in simulation, else false
    pub is_measured: bool,
    /// Flag that is true if simulation state are stored in a file (recorded), else false
    pub is_recorded: bool,
    /// Boundary type: explicitly sampled boundary or implicit boundary
    pub explicitly_sampled_boundary: bool,
}

impl SimulationParameters {
    pub fn new(
        procedures: &Procedures,
        params: &Parameters,
        light_position: [f32; 3],
        is_measured: bool,
        is_recorded: bool,
    ) -> Self {
        Self {
            particle_diameter: params.rest_density_grid_spacing as f32,
            light_position,
            buffer_length_limit: params.buffer_length_limit,
            is_measured,
            is_recorded,
            explicitly_sampled_boundary: matches!(
                procedures.boundary_handling,
                BoundaryHandlingVariant::StaticSampleBoundary
            ),
        }
    }
}

impl TryFrom<&[u8]> for SimulationParameters {
    type Error = bincode::error::DecodeError;

    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        let cfg = bincode::config::standard();
        let (decoded, _len): (Self, usize) = bincode::decode_from_slice(bytes, cfg)?;
        Ok(decoded)
    }
}

impl From<SimulationParameters> for Vec<u8> {
    fn from(time_step_info: SimulationParameters) -> Self {
        let cfg = bincode::config::standard();
        bincode::encode_to_vec(time_step_info, cfg).unwrap()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct TimeStepInfo {
    pub time_step_number: u64,
    /// measurement at this point in time
    pub measurement: Measurement,
    /// fluid
    pub fluid: FluidVisualization,
    /// boundary
    pub boundary: BoundaryVisualization,
}

impl TimeStepInfo {
    pub fn from_system(system: &mut dyn SPHSystem, selector: &Self) -> Self {
        Self {
            time_step_number: system.time_steps_propagated(),
            measurement: system.take_measurement(),
            fluid: FluidVisualization::from_system(system, &selector.fluid),
            boundary: BoundaryVisualization::from_system(system, &selector.boundary),
        }
    }
}

impl TryFrom<&[u8]> for TimeStepInfo {
    type Error = bincode::error::DecodeError;

    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        let cfg = bincode::config::standard();
        let (decoded, _len): (Self, usize) = bincode::decode_from_slice(bytes, cfg)?;
        Ok(decoded)
    }
}

impl From<TimeStepInfo> for Vec<u8> {
    fn from(time_step_info: TimeStepInfo) -> Self {
        let cfg = bincode::config::standard();
        bincode::encode_to_vec(time_step_info, cfg).unwrap()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Encode, Decode)]
pub enum FluidVisualization {
    TriangleMesh {
        meshes: Vec<(u32, RenderMesh)>,
        max_fluid_id: u32,
        coloring: FluidMeshColoring,
    },
    Samples {
        positions: Vec<[f32; 3]>,
        coloring: FluidSampleColoring,
    },
    SensorPlane {
        quantity: ScalarQuantity,
        planes: Vec<SensorPlaneData>,
    },
}

impl FluidVisualization {
    fn from_system(system: &mut dyn SPHSystem, selector: &Self) -> Self {
        match selector {
            Self::TriangleMesh { coloring, .. } => Self::TriangleMesh {
                meshes: system.get_fluid_surface(),
                max_fluid_id: system.get_fluid_ids().into_iter().max().unwrap_or(0),
                coloring: coloring.clone(),
            },
            Self::Samples { coloring, .. } => Self::Samples {
                positions: system.get_fluid_pos(),
                coloring: FluidSampleColoring::from_system(system, coloring),
            },
            Self::SensorPlane { planes, quantity } => {
                let mut planes_acc = Vec::new();
                for plane in planes {
                    let data = system.get_quantity_at_positions(quantity, &plane.positions);
                    planes_acc.push(SensorPlaneData {
                        positions: plane.positions.clone(),
                        data,
                        rows: plane.rows,
                        cols: plane.cols,
                    });
                }
                Self::SensorPlane {
                    quantity: quantity.clone(),
                    planes: planes_acc,
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Encode, Decode)]
pub enum FluidMeshColoring {
    /// Uniform, halbtransparentes Grau
    Uniform,
    /// Einfärben nach fluid_id
    FluidId,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Encode, Decode)]
pub enum FluidSampleColoring {
    Uniform,
    FluidId {
        id: Vec<u32>,
        max_id: u32,
    },
    QuantityGraded {
        quantity: ScalarQuantity,
        data: Vec<f32>,
    },
}

impl FluidSampleColoring {
    fn from_system(system: &dyn SPHSystem, selector: &Self) -> Self {
        match selector {
            Self::Uniform => Self::Uniform,
            Self::FluidId { .. } => {
                let val = system.get_fluid_ids();
                let max_id = val.iter().copied().max().unwrap_or(0);
                Self::FluidId { id: val, max_id }
            }
            Self::QuantityGraded { quantity, .. } => Self::QuantityGraded {
                quantity: quantity.clone(),
                data: system.get_quantity_of_fluid_samples(quantity),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Encode, Decode)]
pub struct SensorPlaneData {
    pub positions: Vec<[f32; 3]>,
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Encode, Decode)]
pub enum ScalarQuantity {
    Speed,
    Volume,
    Density,
    DensityError,
    Pressure,
    KineticEnergy,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Encode, Decode)]
pub enum BoundaryVisualization {
    TriangleMesh {
        /// Mesh in local/body frame and its current world pose.
        /// Static boundaries always use `RenderPose::IDENTITY`.
        meshes: Vec<(RenderMesh, RenderPose)>,
        coloring: BoundaryMeshColoring,
    },
    Samples {
        positions: Vec<[f32; 3]>,
        coloring: BoundarySampleColoring,
    },
}

impl BoundaryVisualization {
    fn from_system(system: &dyn SPHSystem, selector: &Self) -> Self {
        system.get_boundary_visualization(selector)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Encode, Decode)]
pub struct RenderPose {
    pub translation: [f32; 3],
    /// Quaternion as (i, j, k, w)
    pub rotation: [f32; 4],
}

impl RenderPose {
    pub const IDENTITY: Self = Self {
        translation: [0.0, 0.0, 0.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
    };
}

impl From<Isometry3<f64>> for RenderPose {
    fn from(isometry: Isometry3<f64>) -> Self {
        let t = isometry.translation.vector;
        let q = isometry.rotation.into_inner();
        Self {
            translation: [t.x as f32, t.y as f32, t.z as f32],
            rotation: [q.i as f32, q.j as f32, q.k as f32, q.w as f32],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Encode, Decode)]
pub enum BoundaryMeshColoring {
    Original,
    Uniform,
    BoundaryId { ids: Vec<u32>, max_id: u32 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Encode, Decode)]
pub enum BoundarySampleColoring {
    Uniform,
    BoundaryId { ids: Vec<u32>, max_id: u32 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measurement::Measurement;
    use crate::sph::boundary_handling::BoundaryCheckpoint;
    use crate::sph::fluid::FluidCheckpoint;
    use crate::sph::{SPHSystem, SystemCheckpoint};
    use std::rc::Rc;

    // ─── Mock SPHSystem ─────────────────────────────────────────────────

    #[derive(Clone)]
    struct MockSystem {
        fluid_ids: Vec<u32>,
        fluid_pos: Vec<[f32; 3]>,
        fluid_surface: Vec<(u32, RenderMesh)>,
        quantity_of_samples: Vec<f32>,
        quantity_at_positions: Vec<f32>,
        boundary_visualization_result: BoundaryVisualization,
        time_steps_propagated: u64,
    }

    impl Default for MockSystem {
        fn default() -> Self {
            Self {
                fluid_ids: vec![],
                fluid_pos: vec![],
                fluid_surface: vec![],
                quantity_of_samples: vec![],
                quantity_at_positions: vec![],
                boundary_visualization_result: BoundaryVisualization::Samples {
                    positions: vec![],
                    coloring: BoundarySampleColoring::Uniform,
                },
                time_steps_propagated: 0,
            }
        }
    }

    impl SPHSystem for MockSystem {
        fn time(&self) -> f64 {
            unimplemented!("not exercised by render_info tests")
        }
        fn time_steps_propagated(&self) -> u64 {
            self.time_steps_propagated
        }
        fn step_forward_in_time(&mut self) {
            unimplemented!("not exercised by render_info tests")
        }
        fn take_measurement(&self) -> Measurement {
            Measurement::default()
        }
        fn get_fluid_ids(&self) -> Vec<u32> {
            self.fluid_ids.clone()
        }
        fn get_fluid_pos(&self) -> Vec<[f32; 3]> {
            self.fluid_pos.clone()
        }
        fn get_fluid_checkpoint(&self) -> FluidCheckpoint {
            unimplemented!("not exercised by render_info tests")
        }
        fn get_quantity_of_fluid_samples(&self, _quantity: &ScalarQuantity) -> Vec<f32> {
            self.quantity_of_samples.clone()
        }
        fn get_quantity_at_positions(
            &mut self,
            _quantity: &ScalarQuantity,
            _positions: &[[f32; 3]],
        ) -> Vec<f32> {
            self.quantity_at_positions.clone()
        }
        fn get_fluid_surface(&self) -> Vec<(u32, RenderMesh)> {
            self.fluid_surface.clone()
        }
        fn get_boundary_visualization(
            &self,
            _selector: &BoundaryVisualization,
        ) -> BoundaryVisualization {
            self.boundary_visualization_result.clone()
        }
        fn get_boundary_checkpoint(&self) -> BoundaryCheckpoint {
            unimplemented!("not exercised by render_info tests")
        }
        fn continue_from_checkpoint(&mut self, _checkpoint: Rc<SystemCheckpoint>) {
            unimplemented!("not exercised by render_info tests")
        }
    }

    // ─── FluidVisualization::from_system: TriangleMesh ───────────────────

    #[test]
    fn fluid_visualization_triangle_mesh_uses_fluid_surface_and_max_id() {
        let mut mock = MockSystem {
            fluid_ids: vec![2, 5, 1],
            fluid_surface: vec![(2, RenderMesh::default()), (5, RenderMesh::default())],
            ..Default::default()
        };
        let selector = FluidVisualization::TriangleMesh {
            meshes: vec![],
            max_fluid_id: 0,
            coloring: FluidMeshColoring::FluidId,
        };

        let result = FluidVisualization::from_system(&mut mock, &selector);

        match result {
            FluidVisualization::TriangleMesh {
                meshes,
                max_fluid_id,
                coloring,
            } => {
                assert_eq!(meshes.len(), 2);
                // NOTE: `max_fluid_id` comes from `get_fluid_ids()` (every
                // active particle), while `meshes` comes from
                // `get_fluid_surface()`, which (per `Fluid::reconstruct_
                // surfaces`'s doc comment) skips fluid ids whose surface
                // reconstruction came out empty. So `max_fluid_id` can, in
                // principle, reference an id that has no corresponding
                // entry in `meshes` — harmless for a color-range upper
                // bound, but worth documenting rather than assuming
                // they're always in sync.
                assert_eq!(max_fluid_id, 5);
                assert_eq!(coloring, FluidMeshColoring::FluidId);
            }
            _ => panic!("expected TriangleMesh variant"),
        }
    }

    #[test]
    fn fluid_visualization_triangle_mesh_max_id_defaults_to_zero_with_no_fluid() {
        let mut mock = MockSystem::default(); // fluid_ids empty
        let selector = FluidVisualization::TriangleMesh {
            meshes: vec![],
            max_fluid_id: 0,
            coloring: FluidMeshColoring::Uniform,
        };

        let result = FluidVisualization::from_system(&mut mock, &selector);

        match result {
            FluidVisualization::TriangleMesh { max_fluid_id, .. } => {
                assert_eq!(max_fluid_id, 0);
            }
            _ => panic!("expected TriangleMesh variant"),
        }
    }

    // ─── FluidVisualization::from_system: Samples ────────────────────────

    #[test]
    fn fluid_visualization_samples_uses_fluid_positions_and_delegates_coloring() {
        let mut mock = MockSystem {
            fluid_pos: vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            ..Default::default()
        };
        let selector = FluidVisualization::Samples {
            positions: vec![],
            coloring: FluidSampleColoring::Uniform,
        };

        let result = FluidVisualization::from_system(&mut mock, &selector);

        match result {
            FluidVisualization::Samples {
                positions,
                coloring,
            } => {
                assert_eq!(positions, vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
                assert_eq!(coloring, FluidSampleColoring::Uniform);
            }
            _ => panic!("expected Samples variant"),
        }
    }

    // ─── FluidVisualization::from_system: SensorPlane ────────────────────

    #[test]
    fn fluid_visualization_sensor_plane_fills_data_for_every_plane_preserving_metadata() {
        let mut mock = MockSystem {
            quantity_at_positions: vec![10.0, 20.0],
            ..Default::default()
        };
        let plane_a = SensorPlaneData {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            data: vec![], // deliberately stale/empty, must be overwritten
            rows: 1,
            cols: 2,
        };
        let plane_b = SensorPlaneData {
            positions: vec![[0.0, 1.0, 0.0]],
            data: vec![999.0], // deliberately WRONG, must be overwritten
            rows: 1,
            cols: 1,
        };
        let selector = FluidVisualization::SensorPlane {
            quantity: ScalarQuantity::Pressure,
            planes: vec![plane_a.clone(), plane_b.clone()],
        };

        let result = FluidVisualization::from_system(&mut mock, &selector);

        match result {
            FluidVisualization::SensorPlane { quantity, planes } => {
                assert_eq!(quantity, ScalarQuantity::Pressure);
                assert_eq!(planes.len(), 2);
                for plane in &planes {
                    assert_eq!(plane.data, vec![10.0, 20.0]);
                }
                assert_eq!(planes[0].positions, plane_a.positions);
                assert_eq!(planes[0].rows, plane_a.rows);
                assert_eq!(planes[0].cols, plane_a.cols);
                assert_eq!(planes[1].positions, plane_b.positions);
            }
            _ => panic!("expected SensorPlane variant"),
        }
    }

    #[test]
    fn fluid_visualization_sensor_plane_with_no_planes_yields_no_planes() {
        let mut mock = MockSystem::default();
        let selector = FluidVisualization::SensorPlane {
            quantity: ScalarQuantity::Speed,
            planes: vec![],
        };

        let result = FluidVisualization::from_system(&mut mock, &selector);

        match result {
            FluidVisualization::SensorPlane { planes, .. } => assert!(planes.is_empty()),
            _ => panic!("expected SensorPlane variant"),
        }
    }

    // ─── FluidSampleColoring::from_system ─────────────────────────────────

    #[test]
    fn fluid_sample_coloring_uniform_stays_uniform() {
        let mock = MockSystem {
            fluid_ids: vec![1, 2, 3],
            ..Default::default()
        };
        let result = FluidSampleColoring::from_system(&mock, &FluidSampleColoring::Uniform);
        assert_eq!(result, FluidSampleColoring::Uniform);
    }

    #[test]
    fn fluid_sample_coloring_fluid_id_reports_ids_and_max() {
        let mock = MockSystem {
            fluid_ids: vec![3, 1, 4, 1, 5],
            ..Default::default()
        };
        let selector = FluidSampleColoring::FluidId {
            id: vec![],
            max_id: 0,
        };

        let result = FluidSampleColoring::from_system(&mock, &selector);

        match result {
            FluidSampleColoring::FluidId { id, max_id } => {
                assert_eq!(id, vec![3, 1, 4, 1, 5]);
                assert_eq!(max_id, 5);
            }
            _ => panic!("expected FluidId variant"),
        }
    }

    #[test]
    fn fluid_sample_coloring_fluid_id_max_defaults_to_zero_when_empty() {
        let mock = MockSystem::default();
        let selector = FluidSampleColoring::FluidId {
            id: vec![],
            max_id: 0,
        };
        let result = FluidSampleColoring::from_system(&mock, &selector);
        match result {
            FluidSampleColoring::FluidId { max_id, .. } => assert_eq!(max_id, 0),
            _ => panic!("expected FluidId variant"),
        }
    }

    #[test]
    fn fluid_sample_coloring_quantity_graded_preserves_quantity_and_replaces_data() {
        let mock = MockSystem {
            quantity_of_samples: vec![0.1, 0.2, 0.3],
            ..Default::default()
        };
        let selector = FluidSampleColoring::QuantityGraded {
            quantity: ScalarQuantity::KineticEnergy,
            data: vec![999.0], // stale, must be overwritten
        };

        let result = FluidSampleColoring::from_system(&mock, &selector);

        match result {
            FluidSampleColoring::QuantityGraded { quantity, data } => {
                assert_eq!(quantity, ScalarQuantity::KineticEnergy);
                assert_eq!(data, vec![0.1, 0.2, 0.3]);
            }
            _ => panic!("expected QuantityGraded variant"),
        }
    }

    // ─── BoundaryVisualization::from_system ───────────────────────────────

    #[test]
    fn boundary_visualization_from_system_delegates_to_the_system() {
        let canned = BoundaryVisualization::TriangleMesh {
            meshes: vec![],
            coloring: BoundaryMeshColoring::Uniform,
        };
        let mock = MockSystem {
            boundary_visualization_result: canned.clone(),
            ..Default::default()
        };
        let selector = BoundaryVisualization::Samples {
            positions: vec![],
            coloring: BoundarySampleColoring::Uniform,
        };

        let result = BoundaryVisualization::from_system(&mock, &selector);

        assert_eq!(result, canned);
    }
}
