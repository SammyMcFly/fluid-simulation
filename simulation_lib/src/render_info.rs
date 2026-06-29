use crate::measurement::Measurement;
use crate::setup::input::Parameters;
use crate::sph::SPHSystem;
use crate::utilities::triangle_mesh::RenderMesh;
use bincode::{Decode, Encode};
/// simulation info module
///
/// Contains info structs about a simulation and a time step
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
}

impl SimulationParameters {
    pub fn new(
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
        }
    }
}

impl From<&[u8]> for SimulationParameters {
    fn from(bytes: &[u8]) -> Self {
        let cfg = bincode::config::standard();
        let (decoded, _len): (Self, usize) = bincode::decode_from_slice(bytes, cfg).unwrap();
        decoded
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
            measurement: system.take_measurement(),
            fluid: FluidVisualization::from_system(system, &selector.fluid),
            boundary: BoundaryVisualization::from_system(system, &selector.boundary),
        }
    }
}

impl From<&[u8]> for TimeStepInfo {
    fn from(bytes: &[u8]) -> Self {
        let cfg = bincode::config::standard();
        let (decoded, _len): (Self, usize) = bincode::decode_from_slice(bytes, cfg).unwrap();
        decoded
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
        mesh: RenderMesh,
    },
    Samples {
        positions: Vec<[f32; 3]>,
        coloring: FluidColoring,
    },
    SensorPlane {
        planes: Vec<SensorPlaneData>,
    },
}

impl FluidVisualization {
    fn from_system(system: &mut dyn SPHSystem, selector: &Self) -> Self {
        match selector {
            Self::TriangleMesh { .. } => Self::TriangleMesh {
                mesh: system.get_fluid_surface(),
            },
            Self::Samples { coloring, .. } => Self::Samples {
                positions: system.get_fluid_pos(),
                coloring: FluidColoring::from_system(system, coloring),
            },
            Self::SensorPlane { planes } => {
                let mut planes_acc = Vec::new();
                for plane in planes {
                    let quantity =
                        system.get_quantity_at_positions(&plane.quantity, &plane.positions);
                    planes_acc.push(SensorPlaneData {
                        positions: plane.positions.clone(),
                        quantity,
                        rows: plane.rows,
                        cols: plane.cols,
                    });
                }
                Self::SensorPlane { planes: planes_acc }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Encode, Decode)]
pub enum FluidColoring {
    Uniform,
    FluidId { val: Vec<u32>, max_id: u32 },
    QuantityGraded { quantity: ScalarQuantity },
}

impl FluidColoring {
    fn from_system(system: &dyn SPHSystem, selector: &Self) -> Self {
        match selector {
            Self::Uniform => Self::Uniform,
            Self::FluidId { .. } => {
                let val = system.get_fluid_ids();
                let max_id = val.iter().copied().max().unwrap_or(0);
                Self::FluidId { val, max_id }
            }
            Self::QuantityGraded { quantity } => Self::QuantityGraded {
                quantity: system.get_quantity_of_fluid_samples(quantity),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Encode, Decode)]
pub struct SensorPlaneData {
    pub positions: Vec<[f32; 3]>,
    pub quantity: ScalarQuantity,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Encode, Decode)]
pub enum ScalarQuantity {
    SpeedGraded(Vec<f32>),
    VolumeGraded(Vec<f32>),
    DensityGraded(Vec<f32>),
    DensityErrorGraded(Vec<f32>),
    PressureGraded(Vec<f32>),
    KineticEnergyGraded(Vec<f32>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Encode, Decode)]
pub enum BoundaryVisualization {
    TriangleMesh {
        mesh: RenderMesh,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Encode, Decode)]
pub enum BoundaryMeshColoring {
    Original,
    Uniform,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Encode, Decode)]
pub enum BoundarySampleColoring {
    Uniform,
    BoundaryId { val: Vec<u32>, max_id: u32 },
}
