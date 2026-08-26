struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
};
@group(1) @binding(0) var<uniform> light: Light;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

struct InstanceInput {
    @location(3) translation: vec3<f32>,
    @location(4) rotation: vec4<f32>, // quaternion (i, j, k, w)
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = q.xyz;
    let qw = q.w;
    let c1 = cross(qv, v);
    let c2 = cross(qv, c1);
    return v + 2.0 * (qw * c1 + c2);
}

/// Sim frame (x, y, z) -> render frame (x, z, -y). Orthogonal (det = +1),
/// so it's valid for normals too.
fn swizzle(v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(v.x, v.z, -v.y);
}

@vertex
fn vs_main(in: VertexInput, inst: InstanceInput) -> VertexOutput {
    var out: VertexOutput;

    let world_pos_sim = quat_rotate(inst.rotation, in.position) + inst.translation;
    let world_normal_sim = quat_rotate(inst.rotation, in.normal);

    out.world_position = swizzle(world_pos_sim);
    out.world_normal = swizzle(world_normal_sim);
    out.color = in.color;
    out.clip_position = camera.view_proj * vec4(out.world_position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(in.world_normal);
    let light_dir = normalize(swizzle(light.position) - in.world_position);

    // Simple diffuse lighting
    let ambient = 0.2;
    let diffuse = max(dot(normal, light_dir), 0.0) * 0.8;
    let lit = ambient + diffuse;

    let result = in.color.rgb * lit;

    // Alpha from vertex color
    return vec4(result, in.color.a);
}
