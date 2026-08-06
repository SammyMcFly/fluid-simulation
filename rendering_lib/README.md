# rendering_lib

GPU rendering library for SPH fluid simulation visualization, built on `wgpu` via `libcosmic`/`iced`'s shader widget API. Provides an embeddable 3D viewport (`SimulationViewport`) that renders fluid particles, reconstructed surfaces, sensor planes, and boundary geometry, plus offscreen screenshot capture.

## Overview

This crate implements the three-part `iced::widget::shader` abstraction:

| `iced` trait | Type here | Lifetime |
|---|---|---|
| `shader::Program` | `SimulationViewport<W>` | Owned by the application; holds camera/light/scene state and handles input |
| `shader::Primitive` | `SimulationFrame<W>` | Created every frame by `draw()`; carries CPU-side data + upload/draw logic |
| `shader::Pipeline` | `SimulationRenderer` | Created once by `iced`; owns all persistent GPU resources |

It consumes `simulation_lib::render_info::TimeStepInfo` and turns it into GPU-ready geometry, so the frontend crates (`sci-phi`, `sci-phi-player`) only need to forward simulation data and react to `ViewportEvent`s.

## Module structure

| Module | Purpose |
|--------|---------|
| `camera` | `Camera`, `Projection`, `CameraController`, `CameraUniform`, `CameraState`, abstract `Key` enum — camera math and input handling (no GPU) |
| `lighting` | `Light`, `LightUniform`, `LightState` — orbiting point light, resettable (no GPU) |
| `pipeline` | `SimulationRenderer` (`shader::Pipeline`): bind group layouts, uniform buffers, all render pipelines, `DepthTexture`, `SceneGpuBuffers`, `LightMesh`, screenshot state machine, `BillboardInstance`, `ScreenshotCommand` trait |
| `primitive` | `SimulationFrame` (`shader::Primitive`): `SceneData`/`FluidSceneData`/`BoundarySceneData`, per-frame upload (`prepare`), render passes (`render`), screenshot readback, `ScreenshotRequest`/`ScreenshotTarget` |
| `viewport` | `SimulationViewport` (`shader::Program`): `ViewportState`, `ViewportEvent`, mouse/keyboard/scroll handling, key mapping |
| `scene_builder` | `build_scene_data` — converts `TimeStepInfo` into CPU-side `SceneData` (colormapping, cutting, coordinate swap, sensor-plane normals) |
| `colormap` | `Colormap` enum (7 `colorous` gradients), `values_to_colors`, `ids_to_colors` |
| `cut` | `Cut` — axis-aligned clipping planes (per-axis active/bound/inverse) and `sensor_plane_samples` grid generation |
| `model` | `ModelVertex`, `ColoredMeshVertex`, `VertexBufferLayout` trait; embedded `sphere.obj` for the light indicator |
| `shaders/*.wgsl` | WGSL shaders (see below), compiled in via `include_str!` |

## Architecture

### Frame lifecycle

```
Application::update()
    └─ viewport.set_scene(build_scene_data(&time_step, ...))
    └─ viewport.camera.tick(dt) / viewport.light.tick(dt)

Application::view()  ->  shader(&viewport)
    │
    ▼  SimulationViewport::draw()   (shader::Program)
SimulationFrame { camera_uniform, light_uniform, scene: SceneData,
                  background_color, readback_request, worker_sender,
                  screenshot_consumed }
    │
    ▼  SimulationFrame::prepare()   (shader::Primitive)

    1. advance screenshot state machine (map / read back / dispatch)
    2. (re)create DepthTexture if viewport size changed

    3. queue.write_buffer(camera_buffer), queue.write_buffer(light_buffer)
    4. upload_scene()  -> create_buffer_init for particles / meshes / sensor plane

    5. if screenshot requested: setup_offscreen()

    │
    ▼  SimulationFrame::render()
    Pass 1: on-screen render pass (LoadOp::Load, depth cleared) -> draw_scene()
    Pass 2 (optional): offscreen render pass (background cleared) -> draw_scene()

            + copy_texture_to_buffer(offscreen -> staging)
            + state: Idle -> CopyIssued

```

`draw()` returns `false` from `shader::Primitive::draw`, so all drawing goes through the custom `render()` implementation (needed for the extra offscreen pass and its own depth attachment).

`build_scene_data` performs three transformations in one pass:

1. **Coordinate swap** — simulation space is z-up, graphics space is y-up: `[x, y, z] → [x, z, -y]` (applied to positions *and* normals).
2. **Cutting** — particles are filtered by `Cut::cut()`; boundary particles optionally exempt via `cut_boundary`.
3. **Coloring** — `FluidSampleColoring` / `FluidMeshColoring` / `BoundarySampleColoring` / `BoundaryMeshColoring` are resolved to `[f32; 4]` RGBA via `colormap::values_to_colors` (normalized to `[0, max_mapping]`) or `ids_to_colors`.

Sensor planes additionally get per-vertex normals accumulated from triangle cross products and a row-major triangle index grid.

### Render pipelines and draw order

`draw_scene()` issues draws in a fixed order (opaque first, transparent last, since there is no depth sorting):

| Order | Geometry | Pipeline | Shader | Notes |
|---|---|---|---|---|
| 1 | Light indicator sphere | `light_pipeline` | `light.wgsl` | Embedded `sphere.obj`, scaled 0.25 |
| 2 | Fluid particles | `particle_pipeline` | `particle_impostor.wgsl` | Opaque, `BlendState::REPLACE` |
| 3 | Sensor plane | `mesh_unlit_pipeline` | `mesh_unlit.wgsl` | Camera BGL only, `cull_mode: None` (double-sided) |
| 4 | Fluid mesh (transparent) | `mesh_transparent_fluid_backface_pipeline` → `mesh_transparent_fluid_pipeline` | `mesh_transparent_fluid.wgsl` | Fresnel/glass look, depth writes off |
| 4' | Fluid mesh (opaque) | `mesh_opaque_pipeline` | `mesh_opaque.wgsl` | Normal flipped toward camera |
| 5 | Boundary particles | `particle_transparent_pipeline` | `particle_impostor.wgsl` | Alpha blending |
| 6 | Boundary mesh | `mesh_transparent_backface_pipeline` (cull front) → `mesh_transparent_pipeline` (cull back) | `mesh_transparent.wgsl` | Two-pass back-then-front for correct transparency |

All pipelines share a `Depth32Float` depth attachment (`CompareFunction::Less`); transparent pipelines set `depth_write_enabled: false`.

**Particle impostors** — particles are drawn as 6-vertex camera-facing quads (`draw(0..6, 0..count)`) with one `BillboardInstance { center, radius, color }` per particle. The fragment shader discards fragments outside the unit disc, reconstructs the sphere normal, applies Blinn-Phong lighting, and writes a corrected `frag_depth` so impostors intersect correctly with meshes.

**Bind groups** — group 0 = `Camera` uniform, group 1 = `Light` uniform. All shaders except `mesh_unlit.wgsl` use both.

### Camera & lighting

`CameraController` accumulates WASD/Space/Shift (as abstract `Key` values), mouse drag, and scroll deltas; `CameraState::tick(dt)` applies them and refreshes the uniform. Pitch is clamped to ±(π/2 − ε). `LightState::tick(dt)` rotates the light about the z-axis at `movement_speed`; both states support `reset()` to their initial values.

### Input handling

`SimulationViewport::update` translates `iced` events into `ViewportEvent`s published to the app's `Message` type (`Message: From<ViewportEvent>`):

| Input | `ViewportEvent` | Capture |
|---|---|---|
| Bounds change | `Resized { width, height }` | – |
| Middle mouse press/release | *(state only)* | ✅ |
| Cursor moved while middle held | `CameraRotated { dx, dy }` | ✅ |
| Wheel scrolled (cursor in bounds) | `CameraScrolled { delta }` | ✅ |
| `W`/`A`/`S`/`D`/`Space`/`Shift`, arrow keys | `CameraKey { key, pressed }` | ✅ on press |

Mouse and key events are only accepted while the cursor is over the viewport bounds (key *release* is always forwarded to avoid stuck keys). `mouse_interaction` shows a grabbing cursor while rotating.

### Screenshot capture

Screenshots are rendered to a dedicated offscreen texture (so the UI chrome and background color are excluded) and read back over three frames via an explicit state machine:

```
app: viewport.request_screenshot(ScreenshotRequest { target })
     screenshot_consumed = false
        │
Frame N   render(): offscreen pass + copy_texture_to_buffer
                    Idle -> CopyIssued { pending }
Frame N+1 prepare(): buffer.slice(..).map_async(Read, ..)
                    CopyIssued -> MapPending { pending }
Frame N+2 prepare(): staging_mapped == true

                    - de-pad rows (padded_bpr -> width*4)
                    - swizzle BGRA -> RGBA if surface_format is Bgra8Unorm[Srgb]

                    - send W::write_rendering(..) | W::save_screenshot_to_file(..)

                    MapPending -> Idle,  screenshot_consumed = true
        │
app: viewport.is_screenshot_done() == true   // safe to request the next frame
```

Actual PNG encoding happens **outside** this crate: the RGBA buffer is sent over a `crossbeam::channel::Sender<W>` to a worker crate. `W` is any type implementing the `ScreenshotCommand` trait:

```rust
pub trait ScreenshotCommand: Debug + Send + 'static {
    fn write_rendering(data: Vec<u8>, width: u32, height: u32,
                       frame_index: usize, directory: PathBuf) -> Self;
    fn save_screenshot_to_file(data: Vec<u8>, width: u32, height: u32,
                               file_path: PathBuf) -> Self;
}
```

This is implemented by `sci-phi-backend::WorkerCommand` and `sci-phi-player-backend::WorkerCommand`, which keeps `rendering_lib` independent of any specific backend.

### Cutting planes

`Cut` holds three independent axis planes (`{x,y,z}_active`, `_bound`, `_inverse`/`_inv` sign). `cut(&position)` returns whether a point is on the kept side of *all* active planes. `sensor_plane_samples(dx, min, max)` generates row-major `SensorPlaneData` grids on each active plane, which the simulation fills with interpolated quantities and which are then rendered with `mesh_unlit.wgsl`.

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `logging` | ❌ | Enables `tracing` log statements (`dep:tracing`) |

## Dependencies

| Crate | Purpose |
|-------|---------|
| `simulation_lib` | Source of `TimeStepInfo`, `FluidVisualization`, `BoundaryVisualization`, `RenderMesh`, `SensorPlaneData` |
| `libcosmic` (`wgpu` feature) | `iced`/`cosmic` shader widget API and re-exported `wgpu` |
| `cgmath` | Camera/projection matrix math (`Matrix4`, `Point3`, `Rad`, `Quaternion`) |
| `nalgebra` | Linear algebra interop with `simulation_lib` |
| `bytemuck` | `Pod`/`Zeroable` derives and casting vertex/uniform data to GPU bytes |
| `tobj` | Loading the embedded `sphere.obj` light indicator mesh |
| `ahash` | Hash map required by `tobj`'s material loader callback |
| `colorous` | Perceptual colormap gradients (Viridis, Magma, Inferno, Plasma, Turbo, Cividis, Blues) |
| `crossbeam` | `Sender<W>` channel for dispatching screenshot buffers to a worker |
| `serde`, `bincode` | (De)serializing `Colormap` in persisted UI settings / recordings |
| `pollster` | Blocking on async `wgpu` operations where needed |
| `image` | Image type support for screenshot handling |
| `num-traits` | Generic numeric helpers |
| `rustc-hash`, `ron`, `csv`, `toml` | Shared workspace utility dependencies |
| `tracing` *(optional, `logging`)* | Structured logging |

## Usage

```rust
use rendering_lib::{
    CameraState, LightState, SimulationViewport, ViewportEvent, build_scene_data,
};
use rendering_lib::colormap::Colormap;
use rendering_lib::cut::Cut;
use cosmic::iced::widget::shader;

// 1. Construct once, in your Application::new()
let mut viewport = SimulationViewport::new(
    CameraState::new(
        (0.0, -3.0, 1.0),               // position (z-up)
        cgmath::Deg(-90.0),             // yaw
        cgmath::Deg(0.0),               // pitch
        1.0, 0.4, 4.0,                  // speed, sensitivity, scroll_speed
        cgmath::Deg(45.0), 0.1, 100.0,  // fovy, znear, zfar
        800, 600,                       // initial size
    ),
    LightState::new([2.0, 2.0, 4.0], [1.0, 1.0, 1.0], 0.1),
    [0.05, 0.05, 0.08, 1.0],            // background color
    worker_sender,                      // Sender<WorkerCommand>
);

// 2. Feed new simulation data in Application::update()
viewport.set_scene(build_scene_data(
    &time_step_info,
    &Cut::default(),
    /* cut_boundary */ false,
    /* boundary_hidden */ false,
    /* boundary_alpha */ 0.3,
    /* particle_radius */ 0.01,
    /* max_mapping */ 5.0,
    Colormap::Viridis,
));

// 3. Advance camera & light on every tick
viewport.camera.tick(dt);
viewport.light.tick(dt);

// 4. Embed in Application::view()
let element = shader(&viewport).width(Fill).height(Fill);

// 5. Handle emitted events (Message: From<ViewportEvent>)
match event {
    ViewportEvent::CameraRotated { dx, dy } => viewport.camera.controller.process_mouse_motion(dx, dy),
    ViewportEvent::CameraScrolled { delta }  => viewport.camera.controller.process_scroll(delta),
    ViewportEvent::CameraKey { key, pressed } => { viewport.camera.controller.process_key(key, pressed); }
    ViewportEvent::Resized { width, height }  => viewport.resize(width, height),
    ViewportEvent::RequestRedraw => {}
}
```

## Notes & caveats

- **Coordinate systems**: everything public (camera position, light position, `Cut` bounds) uses the simulation's **z-up** convention; the swap to graphics **y-up** happens internally in `position_in_graphics_coordinates()` and `scene_builder`.
- **Screenshots**: always check `is_screenshot_done()` before calling `request_screenshot()` again, otherwise frames will be dropped.
<!--- **Transparency**: rendering relies on a fixed draw order plus a two-pass back-face/front-face split rather than per-triangle depth sorting. Overlapping transparent surfaces from unusual angles can still show artifacts.
- **Buffer churn**: `upload_scene()` recreates vertex/index buffers every frame via `create_buffer_init`. This is simple and correct for changing particle counts, but is a candidate for a reusable growable-buffer optimization.-->
