

struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: Camera;

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
};
@group(1) @binding(0)
var<uniform> light: Light;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

/// Sim frame (x, y, z) -> render frame (x, z, -y). Orthogonal (det = +1),
/// so it's valid for normals too.
fn swizzle(v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(v.x, v.z, -v.y);
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.world_position = swizzle(in.position);
    out.world_normal = swizzle(in.normal);
    out.color = in.color;
    out.clip_position = camera.view_proj * vec4(in.position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var normal = normalize(in.world_normal);

    // Flip normal if facing away from camera (back-face)
    let view_dir = normalize(camera.view_pos.xyz - in.world_position);
    if dot(normal, view_dir) < 0.0 {
        normal = -normal;
    }

    let ambient_strength = 0.15;
    let ambient = light.color * ambient_strength;

    let light_dir = normalize(swizzle(light.position) - in.world_position);
    let diffuse = max(dot(normal, light_dir), 0.0) * light.color;

    let half_dir = normalize(view_dir + light_dir);
    let specular = pow(max(dot(normal, half_dir), 0.0), 32.0) * light.color * 0.3;

    let result = (ambient + diffuse) * in.color.rgb + specular;
    return vec4(result, in.color.a);
}
