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
@group(1) @binding(1) var prev_depth: texture_depth_2d;
@group(1) @binding(2) var opaque_depth: texture_depth_2d;
@group(1) @binding(3) var prev_id: texture_2d<u32>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

struct InstanceInput {
    @location(3) translation: vec3<f32>,
    @location(4) rotation: vec4<f32>, // quaternion (i, j, k, w)
    @location(5) object_id: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) @interpolate(flat) object_id: u32,
};

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = q.xyz;
    let qw = q.w;
    let c1 = cross(qv, v);
    let c2 = cross(qv, c1);
    return v + 2.0 * (qw * c1 + c2);
}

/// Sim frame (x, y, z) -> render frame (x, z, -y).
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
    out.object_id = inst.object_id;
    out.clip_position = camera.view_proj * vec4(out.world_position, 1.0);
    return out;
}

struct FragOutput {
    @location(0) color: vec4<f32>,
    @location(1) id: u32,
};

// Blinn-Phong shading, premultiplied by alpha (required for the "under"
// compositing operator used while peeling).
fn shade(in: VertexOutput) -> vec4<f32> {
    var normal = normalize(in.world_normal);
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
    return vec4(result * in.color.a, in.color.a);
}

// Small bias against floating-point precision: without it, two triangles
// meeting at a shared mesh edge (e.g. adjacent facets of a faceted body)
// can produce depths that are mathematically distinct but numerically
// equal (or in the wrong order) at a given pixel, due to interpolation
// rounding differences between the two triangles. This causes exactly
// one layer to wrongly discard the other at that pixel -- visible as a
// thin dark seam running along internal mesh edges. See e.g. GPU Gems 2,
// "Depth Peeling for Order-Independent Transparency", for this being a
// standard, expected requirement of the technique, not a one-off bug.
const DEPTH_EPSILON: f32 = 1e-7;

fn occluded_by_opaque(pixel: vec2<i32>, depth: f32) -> bool {
    let opaque_d = textureLoad(opaque_depth, pixel, 0);
    return depth >= opaque_d - DEPTH_EPSILON;
}

/// First peel layer: no previous-layer depth to compare against yet.
@fragment
fn fs_first(in: VertexOutput) -> FragOutput {
    let pixel = vec2<i32>(in.clip_position.xy);
    if occluded_by_opaque(pixel, in.clip_position.z) {
        discard;
    }
    var out: FragOutput;
    out.color = shade(in);
    out.id = in.object_id;
    return out;
}

/// Subsequent peel layers: discard anything not strictly farther than the
/// previously peeled layer at this pixel.
@fragment
fn fs_next(in: VertexOutput) -> FragOutput {
    let pixel = vec2<i32>(in.clip_position.xy);
    let depth = in.clip_position.z;
    if occluded_by_opaque(pixel, depth) {
        discard;
    }
    let prev_d = textureLoad(prev_depth, pixel, 0);
    let prev_id_at_pixel = textureLoad(prev_id, pixel, 0).r;

    // Only tolerate DEPTH_EPSILON's floating-point rounding slop when the
    // previous layer's surviving fragment at this pixel was the SAME
    // surface (same object_id) -- that's the shared-mesh-edge seam this
    // epsilon exists for. For a genuinely different surface (e.g. two
    // meshes intersecting), require strictly-greater depth with zero
    // tolerance, so neither side of the intersection gets wrongly
    // discarded.
    let same_surface = prev_id_at_pixel == in.object_id;
    let eps = select(0.0, DEPTH_EPSILON, same_surface);
    if depth <= prev_d + eps {
        discard;
    }

    var out: FragOutput;
    out.color = shade(in);
    out.id = in.object_id;
    return out;
}
