struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: Camera;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

/// Sim frame (x, y, z) -> render frame (x, z, -y). Orthogonal (det = +1),
/// so it's valid for normals too.
fn swizzle(v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(v.x, v.z, -v.y);
}


@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.color = in.color;
    out.clip_position = camera.view_proj * vec4(swizzle(in.position), 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Farbe unverändert, keine Lichtberechnung
    return in.color;
}
