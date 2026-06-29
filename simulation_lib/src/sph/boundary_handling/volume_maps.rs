/// Implicit frictional boundary handling via volume maps
// use approx::assert_abs_diff_eq;
use crate::utilities::integrate_sphere_volume;
use gauss_quad::GaussLegendre;
use parry3d_f64::shape::Triangle;
use nalgebra::{Isometry3, Point3, UnitQuaternion, Vector3};
use parry3d_f64::shape::{TriMesh, TriMeshFlags};
use parry3d_f64::query::PointQuery;

use crate::sph::boundary_handling::BoundaryHandling;
use crate::utilities::triangle_mesh::{MeshHandle, MeshLibrary};


#[derive(Debug, Clone)]
pub struct VolumeMaps {
    pub library: MeshLibrary,
    pub static_boundaries: Vec<StaticBoundary>,
    pub dynamic_boundaries: Vec<DynamicBoundary>,
}

impl BoundaryRepresentation for VolumeMaps {
    fn add_boundary(&mut self, mesh: TriMesh, rest_density_grid_spacing: f64,) {
        let position = sample_mesh_surface_grid(&mesh, rest_density_grid_spacing);
        let boundary = Boundary3D::default();
        for position in position {
            boundary.push(position, Vector3::zeros(), 0.);
        }
        self.boundaries.push(boundary);
    }

    fn initialize() {

    }

    fn add_viscosity_acceleration() {

    }

    fn add_pressure_acceleration() {

    }

    fn get_fluid_depth(&self, fluid_volume: f64) -> f64 {
        0.
    }
}

impl VolumeMaps {
    fn new() -> Self {
        Self {
            library: MeshLibrary::new(),
            static_boundaries: Vec::new(),
            dynamic_boundaries: Vec::new(),
        }
    }

    // pub fn add_fluid(&mut self, mesh: MeshHandle, isometry: Isometry3<f64>, scale: Vector3<f64>) {
    //     self.fluid_instances.push(FluidInstance { mesh, isometry, scale });
    // }

    pub fn add_static_boundary(
        &mut self,
        mesh: MeshHandle,
        isometry: Isometry3<f64>,
        scale: Vector3<f64>,
    ) {
        let world_trimesh = self.build_world_trimesh(mesh, &isometry, &scale);
        self.static_boundaries.push(StaticBoundary {
            mesh,
            isometry,
            scale,
            world_trimesh,
        });
    }

    pub fn add_dynamic_boundary(
        &mut self,
        mesh: MeshHandle,
        isometry: Isometry3<f64>,
        scale: Vector3<f64>,
        // motion: Box<dyn BoundaryMotion>,
    ) {
        // Ensure trimesh is built for local-space queries
        let _ = self.library.trimesh(mesh);
        self.dynamic_boundaries.push(DynamicBoundary {
            mesh,
            isometry,
            scale,
            // motion,
        });
    }

    fn build_world_trimesh(
        &mut self,
        mesh: MeshHandle,
        isometry: &Isometry3<f64>,
        scale: &Vector3<f64>,
    ) -> TriMesh {
        let raw = &self.library.meshes[mesh.0].raw;
        let positions: Vec<Point3<f64>> = raw
            .positions
            .iter()
            .map(|v| {
                let scaled = Point3::new(v.x * scale.x, v.y * scale.y, v.z * scale.z);
                isometry.transform_point(&scaled)
            })
            .collect();

        TriMesh::with_flags(
            positions,
            raw.indices.clone(),
            TriMeshFlags::ORIENTED
                | TriMeshFlags::MERGE_DUPLICATE_VERTICES
                | TriMeshFlags::FIX_INTERNAL_EDGES,
        ).expect("Could not build TriMesh in world coordinates.")
    }

    /// Signed distance to nearest boundary (negative = inside boundary)
    pub fn boundary_signed_distance(&self, point: &Point3<f64>) -> f64 {
        let mut min_dist = f64::MAX;

        // Static boundaries: query world-space TriMesh directly
        for boundary in &self.static_boundaries {
            let proj = boundary.world_trimesh.project_local_point(point, true);
            let dist = (point - proj.point).norm();
            let signed = if proj.is_inside { -dist } else { dist };
            min_dist = min_dist.min(signed);
        }

        // Dynamic boundaries: transform point to local space
        for boundary in &self.dynamic_boundaries {
            let local_point = boundary.isometry.inverse_transform_point(point);
            // Apply inverse scale
            let local_point = Point3::new(
                local_point.x / boundary.scale.x,
                local_point.y / boundary.scale.y,
                local_point.z / boundary.scale.z,
            );
            let trimesh = self.library.meshes[boundary.mesh.0]
                .trimesh
                .as_ref()
                .expect("TriMesh not built for dynamic boundary");
            let proj = trimesh.project_local_point(&local_point, true);
            let dist = (local_point - proj.point).norm();
            // Scale distance back to world space (approximation for uniform scale)
            let avg_scale = (boundary.scale.x + boundary.scale.y + boundary.scale.z) / 3.0;
            let signed = if proj.is_inside { -dist * avg_scale } else { dist * avg_scale };
            min_dist = min_dist.min(signed);
        }

        min_dist
    }
}

// ─── Boundaries ───────────────────────────────────────────────

pub enum BoundaryType {
    /// Static boundary: never moves. World-space TriMesh precomputed.
    /// Render data sent to GPU once.
    StaticBoundary {
        mesh: MeshHandle,
        isometry: Isometry3<f64>,
        scale: Vector3<f64>,
        /// Precomputed world-space TriMesh for fast SDF queries
        world_trimesh: TriMesh,
    },
    /// Dynamic boundary: moves/rotates during simulation.
    /// SDF queries transform point to local space.
    /// Renderer receives updated transform each frame.
    DynamicBoundary {
        mesh: MeshHandle,
        isometry: Isometry3<f64>,
        scale: Vector3<f64>,
        // pub motion: Box<dyn BoundaryMotion>,
    }
}

// // ─── Motion trait ─────────────────────────────────────────────

// pub struct MotionState {
//     pub isometry: Isometry3<f64>,
//     pub linear_velocity: Vector3<f64>,
//     pub angular_velocity: Vector3<f64>,
// }

// pub trait BoundaryMotion: Send + Sync + Sized {
//     fn update(&mut self, time: f64, dt: f64) -> MotionState;
// }

// // ─── Example motion implementations ──────────────────────────

// pub struct OscillateX {
//     pub amplitude: f64,
//     pub frequency: f64,
//     pub base_isometry: Isometry3<f64>,
// }

// impl BoundaryMotion for OscillateX {
//     fn update(&mut self, time: f64, _dt: f64) -> MotionState {
//         let offset = self.amplitude * (self.frequency * time * std::f64::consts::TAU).sin();
//         let velocity_x = self.amplitude * self.frequency * std::f64::consts::TAU
//             * (self.frequency * time * std::f64::consts::TAU).cos();

//         let mut iso = self.base_isometry;
//         iso.translation.x += offset;

//         MotionState {
//             isometry: iso,
//             linear_velocity: Vector3::new(velocity_x, 0.0, 0.0),
//             angular_velocity: Vector3::zeros(),
//         }
//     }
// }

// pub struct Rotate {
//     pub axis: Vector3<f64>,
//     pub angular_speed: f64, // rad/s
//     pub base_isometry: Isometry3<f64>,
// }

// impl BoundaryMotion for Rotate {
//     fn update(&mut self, time: f64, _dt: f64) -> MotionState {
//         let angle = self.angular_speed * time;
//         let rotation = UnitQuaternion::from_axis_angle(
//             &nalgebra::Unit::new_normalize(self.axis),
//             angle,
//         );
//         let iso = Isometry3::from_parts(
//             self.base_isometry.translation,
//             rotation * self.base_isometry.rotation,
//         );
//         MotionState {
//             isometry: iso,
//             linear_velocity: Vector3::zeros(),
//             angular_velocity: self.axis.normalize() * self.angular_speed,
//         }
//     }
// }

fn load_mesh_from_file(file_path: &str) -> Result<TriMesh, Box<dyn std::error::Error>> {
    let (models, _materials) = tobj::load_obj(file_path, &tobj::GPU_LOAD_OPTIONS)?;

    let mut vertices: Vec<Vector> = Vec::new();
    let mut indices: Vec<[u32; 3]> = Vec::new();

    for model in &models {
        let mesh = &model.mesh;
        let vertex_offset = vertices.len() as u32;

        // Collect vertices (positions come as a flat [x, y, z, x, y, z, ...])
        for chunk in mesh.positions.chunks_exact(3) {
            vertices.push(Vector::new(chunk[0], chunk[1], chunk[2]));
        }

        // Collect triangle indices (already triangulated with GPU_LOAD_OPTIONS)
        for tri in mesh.indices.chunks_exact(3) {
            indices.push([
                tri[0] + vertex_offset,
                tri[1] + vertex_offset,
                tri[2] + vertex_offset,
            ]);
        }
    }
    Ok(TriMesh::new(vertices, indices)?)
}

pub fn load_scene(path: &Path) -> Simulation {
    let content = std::fs::read_to_string(path).expect("Failed to read scene file");
    let scene_file: input::Scene = toml::from_str(&content).expect("Failed to parse scene TOML");

    let mut sim = Simulation::new();

    // Load meshes
    let mut name_to_handle: HashMap<String, input::MeshHandle> = HashMap::new();
    for (name, obj_path) in &scene_file.meshes {
        let handle = sim.library.load_obj(obj_path);
        name_to_handle.insert(name.clone(), handle);
    }

    // Create instances
    for inst in &scene_file.instances {
        let handle = name_to_handle
            .get(&inst.mesh)
            .copied()
            .unwrap_or_else(|| panic!("Unknown mesh name: '{}'", inst.mesh));

        let position = Vector3::new(inst.position[0], inst.position[1], inst.position[2]);
        let [r, p, y] = inst.rotation_euler_deg.map(|d| d.to_radians());
        let rotation = UnitQuaternion::from_euler_angles(r, p, y);
        let scale = Vector3::new(inst.scale[0], inst.scale[1], inst.scale[2]);

        sim.spawn(handle, position, rotation, scale);
    }

    sim
}
