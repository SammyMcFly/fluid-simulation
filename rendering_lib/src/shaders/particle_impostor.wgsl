// Camera uniform — must match Rust struct layout exactly
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

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) view_pos: vec3<f32>,   // actual quad-corner position in view space
    @location(1) view_center: vec3<f32>,
    @location(2) world_center: vec3<f32>,
    @location(3) radius: f32,
    @location(4) color: vec4<f32>,
};

/// Sim frame (x, y, z) -> render frame (x, z, -y). Orthogonal (det = +1),
/// so it's valid for normals too.
fn swizzle(v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(v.x, v.z, -v.y);
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) center: vec3<f32>,
    @location(1) radius: f32,
    @location(2) color: vec4<f32>,
) -> VertexOutput {
    let world_center = swizzle(center);

    // Generate camera-facing quad from vertex index (2 triangles = 6 vertices)
    let quad = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
        vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0),
    );
    let uv = quad[vertex_index];

    // Sphere center in view space
    let view_center = (camera.view * vec4(world_center, 1.0)).xyz;

    // Expand quad in view space (always faces camera)
    let expand = radius * 1.05; // slight oversizing to avoid edge clipping
    let view_pos = view_center + vec3(uv * expand, 0.0);

    var out: VertexOutput;
    out.clip_position = camera.proj * vec4(view_pos, 1.0);
    out.view_pos = view_pos;
    out.view_center = view_center;
    out.world_center = world_center;
    out.radius = radius;
    out.color = color;
    return out;
}

struct FragOutput {
    @builtin(frag_depth) depth: f32,
    @location(0) color: vec4<f32>,
};

@fragment
fn fs_main(in: VertexOutput) -> FragOutput {
    // True perspective ray-sphere intersection: the camera sits at the origin
    // in view space, and the ray passes through the actual (interpolated)
    // view-space position of this fragment.
    let ray_dir = normalize(in.view_pos);

    let oc = -in.view_center; // origin (0,0,0) minus sphere center
    let b = dot(ray_dir, oc);
    let c = dot(oc, oc) - in.radius * in.radius;
    let discriminant = b * b - c;

    if discriminant < 0.0 {
        discard;
    }

    let t = -b - sqrt(discriminant); // nearer intersection
    if t < 0.0 {
        discard;
    }

    let frag_view_pos = ray_dir * t;
    let view_normal = normalize(frag_view_pos - in.view_center);

    // Transform normal and position to world space for lighting
    let world_normal = normalize((camera.inv_view * vec4(view_normal, 0.0)).xyz);
    let world_pos = in.world_center + world_normal * in.radius;

    // Blinn-Phong lighting
    let ambient_strength = 0.1;
    let ambient_color = light.color * ambient_strength;

    let light_dir = normalize(swizzle(light.position) - world_pos);
    let diffuse_strength = max(dot(world_normal, light_dir), 0.0);
    let diffuse_color = light.color * diffuse_strength;

    let view_dir = normalize(camera.view_pos.xyz - world_pos);
    let half_dir = normalize(view_dir + light_dir);
    let specular_strength = pow(max(dot(world_normal, half_dir), 0.0), 32.0);
    let specular_color = specular_strength * light.color;

    let result = (ambient_color + diffuse_color) * in.color.rgb + specular_color * 0.2;

    // Correct depth from the actual (perspective) intersection point
    let clip = camera.proj * vec4(frag_view_pos, 1.0);

    var out: FragOutput;
    out.depth = clip.z / clip.w;
    out.color = vec4(result, in.color.a);
    return out;
}
