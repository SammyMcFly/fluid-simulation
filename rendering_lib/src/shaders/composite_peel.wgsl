@group(0) @binding(0) var accum_tex: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

/// Fullscreen triangle covering the entire viewport (no vertex buffer needed).
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var out: VertexOutput;
    out.clip_position = vec4(positions[idx], 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(in.clip_position.xy);
    return textureLoad(accum_tex, pixel, 0);
}
