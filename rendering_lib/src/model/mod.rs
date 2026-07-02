use crate::instances::BillboardInstanceRaw;
use iced_wgpu::wgpu;
use iced_wgpu::wgpu::util::DeviceExt;
use std::ops::Range;

// Include the .obj file inside the binary so there is no dependency at runtime
const SPHERE_DATA: &str = include_str!("sphere.obj");

pub struct ModelAssets {
    pub sphere_mesh: Model,
    // pub light_mesh: Mesh,
    // pub bind_group_layout: wgpu::BindGroupLayout,
}

impl ModelAssets {
    pub fn new(
        gpu_context: &super::gpu_context::GpuContext,
        particle_diameter: f32,
    ) -> Result<Self, tobj::LoadError> {
        Ok(Self {
            sphere_mesh: Model::load_model(gpu_context, particle_diameter)?,
        })
    }
}

pub trait VertexBufferLayout {
    fn desc() -> wgpu::VertexBufferLayout<'static>;
}

impl VertexBufferLayout for BillboardInstanceRaw {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<BillboardInstanceRaw>() as wgpu::BufferAddress,
            // We need to switch from using a step mode of Vertex to Instance
            // This means that our shaders will only change to use the next
            // instance when the shader starts processing a new instance
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // center: vec3<f32>
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // radius: f32
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32,
                },
                // color: vec4<f32>
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelVertex {
    pub position: [f32; 3],
    // pub tex_coords: [f32; 2],
    pub normal: [f32; 3],
}

impl VertexBufferLayout for ModelVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<ModelVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // wgpu::VertexAttribute {
                //     offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                //     shader_location: 1,
                //     format: wgpu::VertexFormat::Float32x2,
                // },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

/// Per-vertex data for colored meshes (boundary mesh, fluid surface, sensor plane)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ColoredMeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
}

impl VertexBufferLayout for ColoredMeshVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<ColoredMeshVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

pub struct Model {
    pub meshes: Vec<Mesh>,
    // pub materials: Vec<Material>,
}

impl Model {
    pub fn load_model(
        gpu_context: &super::gpu_context::GpuContext,
        scaling: f32,
    ) -> Result<Model, tobj::LoadError> {
        let mut cursor = std::io::Cursor::new(SPHERE_DATA.as_bytes());
        let (models, _) =
            tobj::load_obj_buf(
                &mut cursor,
                &tobj::LoadOptions {
                    triangulate: true,
                    single_index: true,
                    ..Default::default()
                },
                |_mtl_path: &std::path::Path| -> Result<
                    (Vec<tobj::Material>, ahash::AHashMap<String, usize>),
                    tobj::LoadError,
                > { Ok((Vec::new(), ahash::AHashMap::new())) },
            )?;

        let meshes = models
            .into_iter()
            .map(|m| {
                let vertices = (0..m.mesh.positions.len() / 3)
                    .map(|i| {
                        if m.mesh.normals.is_empty() {
                            ModelVertex {
                                position: [
                                    m.mesh.positions[i * 3] / 1.78 * scaling,
                                    (m.mesh.positions[i * 3 + 1] - 0.89) / 1.78 * scaling,
                                    m.mesh.positions[i * 3 + 2] / 1.78 * scaling,
                                ],
                                // tex_coords: [m.mesh.texcoords[i * 2], 1.0 - m.mesh.texcoords[i * 2 + 1]],
                                normal: [0.0, 0.0, 0.0],
                            }
                        } else {
                            ModelVertex {
                                position: [
                                    m.mesh.positions[i * 3] / 1.78 * scaling,
                                    (m.mesh.positions[i * 3 + 1] - 0.89) / 1.78 * scaling,
                                    m.mesh.positions[i * 3 + 2] / 1.78 * scaling,
                                ],
                                // tex_coords: [m.mesh.texcoords[i * 2], 1.0 - m.mesh.texcoords[i * 2 + 1]],
                                normal: [
                                    m.mesh.normals[i * 3],
                                    m.mesh.normals[i * 3 + 1],
                                    m.mesh.normals[i * 3 + 2],
                                ],
                            }
                        }
                    })
                    .collect::<Vec<_>>();

                let vertex_buffer =
                    gpu_context
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Sphere Index Buffer"),
                            contents: bytemuck::cast_slice(&vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                let index_buffer =
                    gpu_context
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Sphere Index Buffer"),
                            contents: bytemuck::cast_slice(&m.mesh.indices),
                            usage: wgpu::BufferUsages::INDEX,
                        });

                Mesh {
                    name: "sphere".to_string(),
                    vertex_buffer,
                    index_buffer,
                    num_elements: m.mesh.indices.len() as u32,
                    // material: m.mesh.material_id.unwrap_or(0),
                }
            })
            .collect::<Vec<_>>();

        Ok(Model { meshes }) //  materials })
    }
}

// pub struct Material {
//     pub name: String,
//     // pub diffuse_texture: texture::Texture,
//     pub bind_group: wgpu::BindGroup,
// }

pub struct Mesh {
    pub name: String,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_elements: u32,
    // pub material: usize,
}

// pub trait DrawModel<'a> {
//     fn draw_mesh(
//         &mut self,
//         mesh: &'a Mesh,
//         // material: &'a Material,
//         camera_bind_group: &'a wgpu::BindGroup,
//         light_bind_group: &'a wgpu::BindGroup,
//     );
//     fn draw_mesh_instanced(
//         &mut self,
//         mesh: &'a Mesh,
//         // material: &'a Material,
//         instances: Range<u32>,
//         camera_bind_group: &'a wgpu::BindGroup,
//         light_bind_group: &'a wgpu::BindGroup,
//     );

//     fn draw_model(
//         &mut self,
//         model: &'a Model,
//         camera_bind_group: &'a wgpu::BindGroup,
//         light_bind_group: &'a wgpu::BindGroup,
//     );
//     fn draw_model_instanced(
//         &mut self,
//         model: &'a Model,
//         instances: Range<u32>,
//         camera_bind_group: &'a wgpu::BindGroup,
//         light_bind_group: &'a wgpu::BindGroup,
//     );
// }

// impl<'a, 'b> DrawModel<'b> for wgpu::RenderPass<'a>
// where
//     'b: 'a,
// {
//     fn draw_mesh(
//         &mut self,
//         mesh: &'b Mesh,
//         // material: &'b Material,
//         camera_bind_group: &'b wgpu::BindGroup,
//         light_bind_group: &'b wgpu::BindGroup,
//     ) {
//         self.draw_mesh_instanced(mesh, 0..1, camera_bind_group, light_bind_group); //material
//     }

//     fn draw_mesh_instanced(
//         &mut self,
//         mesh: &'b Mesh,
//         // material: &'b Material,
//         instances: Range<u32>,
//         camera_bind_group: &'b wgpu::BindGroup,
//         light_bind_group: &'b wgpu::BindGroup,
//     ) {
//         self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
//         self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
//         self.set_bind_group(0, camera_bind_group, &[]);
//         self.set_bind_group(1, light_bind_group, &[]);
//         self.draw_indexed(0..mesh.num_elements, 0, instances);
//     }

//     fn draw_model(
//         &mut self,
//         model: &'b Model,
//         camera_bind_group: &'b wgpu::BindGroup,
//         light_bind_group: &'b wgpu::BindGroup,
//     ) {
//         self.draw_model_instanced(model, 0..1, camera_bind_group, light_bind_group);
//     }

//     fn draw_model_instanced(
//         &mut self,
//         model: &'b Model,
//         instances: Range<u32>,
//         camera_bind_group: &'b wgpu::BindGroup,
//         light_bind_group: &'b wgpu::BindGroup,
//     ) {
//         for mesh in &model.meshes {
//             // let material = &model.materials[mesh.material];
//             self.draw_mesh_instanced(mesh, instances.clone(), camera_bind_group, light_bind_group); // material
//         }
//     }
// }

pub trait DrawLight<'a> {
    fn draw_light_mesh(
        &mut self,
        mesh: &'a Mesh,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );
    fn draw_light_mesh_instanced(
        &mut self,
        mesh: &'a Mesh,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );

    fn draw_light_model(
        &mut self,
        model: &'a Model,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );
    fn draw_light_model_instanced(
        &mut self,
        model: &'a Model,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
        light_bind_group: &'a wgpu::BindGroup,
    );
}

impl<'a, 'b> DrawLight<'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_light_mesh(
        &mut self,
        mesh: &'b Mesh,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
    ) {
        self.draw_light_mesh_instanced(mesh, 0..1, camera_bind_group, light_bind_group);
    }

    fn draw_light_mesh_instanced(
        &mut self,
        mesh: &'b Mesh,
        instances: Range<u32>,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
    ) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.set_bind_group(0, camera_bind_group, &[]);
        self.set_bind_group(1, light_bind_group, &[]);
        self.draw_indexed(0..mesh.num_elements, 0, instances);
    }

    fn draw_light_model(
        &mut self,
        model: &'b Model,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
    ) {
        self.draw_light_model_instanced(model, 0..1, camera_bind_group, light_bind_group);
    }
    fn draw_light_model_instanced(
        &mut self,
        model: &'b Model,
        instances: Range<u32>,
        camera_bind_group: &'b wgpu::BindGroup,
        light_bind_group: &'b wgpu::BindGroup,
    ) {
        for mesh in &model.meshes {
            self.draw_light_mesh_instanced(
                mesh,
                instances.clone(),
                camera_bind_group,
                light_bind_group,
            );
        }
    }
}
