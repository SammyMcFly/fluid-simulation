//! Rendering library for SPH fluid simulation visualization.
//!
//! Provides a shader widget (`FluidViewport`) that can be embedded
//! in a cosmic/iced application to render particles, meshes, and sensor planes.
//!
//! # Architecture
//!
//! - `camera` – Camera math, projection, controller (no GPU)
//! - `lighting` – Light state and uniform (no GPU)
//! - `pipeline` – `FluidRenderer`: all GPU resources (impl `shader::Pipeline`)
//! - `primitive` – `FluidFrame`: per-frame render data (impl `shader::Primitive`)
//! - `viewport` – `FluidViewport`: widget program + interaction (impl `shader::Program`)
//! - `scene_builder` – Converts simulation data to CPU-side scene representation
//! - `colormap` – Scalar-to-color mapping
//! - `model` – Vertex types and buffer layouts

pub mod camera;
pub mod colormap;
pub mod cut;
pub mod lighting;
pub mod model;
pub mod pipeline;
pub mod primitive;
pub mod scene_builder;
pub mod viewport;

// ─── Public re-exports for convenience ────────────────────────

// The three shader widget types
pub use pipeline::SimulationRenderer;
pub use primitive::SimulationFrame;
pub use viewport::SimulationViewport;

// State types used by the application
pub use camera::CameraState;
pub use lighting::LightState;
pub use viewport::ViewportEvent;

// Scene building
pub use pipeline::BillboardInstance;
pub use primitive::{BoundarySceneData, FluidSceneData, SceneData};
pub use scene_builder::build_scene_data;

// Vertex types (needed if app builds custom geometry)
pub use model::{ColoredMeshVertex, ModelVertex};
