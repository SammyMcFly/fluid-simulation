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

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.world_position = in.position;
    out.world_normal = in.normal;
    out.color = in.color;
    out.clip_position = camera.view_proj * vec4(in.position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(in.world_normal);
    let view_dir = normalize(camera.view_pos.xyz - in.world_position);
    let light_dir = normalize(light.position - in.world_position);

    // Fresnel (Schlick, IOR ~1.33 → R0 ≈ 0.02)
    let r0 = 0.02;
    let cos_theta = max(dot(normal, view_dir), 0.0);
    let fresnel = r0 + (1.0 - r0) * pow(1.0 - cos_theta, 5.0);

    // Diffuse (subtil für Glas-Look)
    let diffuse = max(dot(normal, light_dir), 0.0) * 0.3;

    // Specular (stark, glasig)
    let half_dir = normalize(view_dir + light_dir);
    let specular = pow(max(dot(normal, half_dir), 0.0), 64.0) * 1.2;

    // Farbe mit Lichtbrechungs-Effekt
    let base = in.color.rgb * (0.1 + diffuse);
    let result = base + light.color * (specular + fresnel * 0.3);

    // Alpha: Fresnel-gesteuert (Kanten opaker, Mitte transparenter)
    let alpha = clamp(in.color.a * (0.3 + fresnel * 0.7), 0.1, 0.85);

    return vec4(result, alpha);
}
