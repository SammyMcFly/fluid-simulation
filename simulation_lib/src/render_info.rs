//! simulation info module
//!
//! Contains info structs about a simulation and a time step
use crate::measurement::Measurement;
use crate::setup::input::Parameters;
use crate::setup::input::Procedures;
use crate::sph::SPHSystem;
use crate::sph::boundary_handling::BoundaryHandlingVariant;
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
