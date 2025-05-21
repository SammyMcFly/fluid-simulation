//! Rusty fluid solver
//!
//! Simple fluid solver that calculates position of fluid particles and renders them into a separate window
//!
//!
use clap::Parser;

use tracing::level_filters::LevelFilter;
use tracing_subscriber::FmtSubscriber;
use tracing::info; // , error, info, span, trace, warn, debug};

use nalgebra::Vector3;

use std::f32::consts::PI;

#[cfg(not(target_arch = "wasm32"))]
use bevy::pbr::wireframe::{WireframeConfig, WireframePlugin};
use bevy::{
    color::palettes::basic::SILVER,
    prelude::*,
    render::{
        render_asset::RenderAssetUsages,
        render_resource::{Extent3d, TextureDimension, TextureFormat},
    },
};



/// Simple fluid solver for simulating a fluid (or any other particle based simulation)
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    // /// File path to input .obj file
    // #[arg(short, long, default_value = "./test_scene.obj")]
    // object: String,
    // /// Horizontal resolution
    // /// Vertical resolution
    /// Log severity level (Options: TRACE, DEBUG, INFO, WARN, ERROR, OFF)
    #[arg(short, long, default_value_t=String::from("OFF"))]
    log: String,
}



fn main() {
    // parse args
    let args = Args::parse();
    // init logging
    let severity_level = match &args.log[..] {
        "TRACE" => LevelFilter::TRACE,
        "DEBUG" => LevelFilter::DEBUG,
        "INFO" => LevelFilter::INFO,
        "WARN" => LevelFilter::WARN,
        "ERROR" => LevelFilter::ERROR,
        _ => LevelFilter::OFF, // String::from("OFF")
    };
    let subscriber = FmtSubscriber::builder()
        .with_max_level(severity_level)
        .with_writer(std::io::stdout)
        .with_line_number(true)
        // .with_ansi(false)
        // .pretty()
        .finish();
        // .with(debug_log);
    tracing::subscriber::set_global_default(subscriber).expect("Failed to set subscriber");
    info!("Start rusty fluid solver");

    println!("Hello, world!");

    App::new()
        .add_plugins((
            DefaultPlugins.set(ImagePlugin::default_nearest()),
            #[cfg(not(target_arch = "wasm32"))]
            WireframePlugin::default(),
        ))
        // .insert_resource(GreetTimer(Timer::from_seconds(2.0, TimerMode::Repeating)))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                sine_wave,
                #[cfg(not(target_arch = "wasm32"))]
                toggle_wireframe,
            ),
        )
        .run();
}


/// A marker component for our shapes so we can query them separately from the ground plane
#[derive(Component)]
struct Shape;

const SHAPES_X_EXTENT: f32 = 14.0;
const Z_EXTENT: f32 = 5.0;

#[derive(Resource)]
struct GreetTimer(Timer);


fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let debug_material = materials.add(StandardMaterial {
        base_color: Color::LinearRgba(LinearRgba::new(1.0, 0.0, 0.0, 0.5)),
        ..default()
    });

    // let shapes = [
    //     // meshes.add(Cuboid::default()),
    //     // meshes.add(Tetrahedron::default()),
    //     // meshes.add(Capsule3d::default()),
    //     // meshes.add(Torus::default()),
    //     // meshes.add(Cylinder::default()),
    //     // meshes.add(Cone::default()),
    //     // meshes.add(ConicalFrustum::default()),
    //     // meshes.add(Sphere::default().mesh().ico(5).unwrap()),
    //     meshes.add(Sphere::default().mesh().uv(32, 18)),
    // ];

    for i in 0..500000 {
        commands.spawn((
            Mesh3d(meshes.add(Sphere::new(0.0045).mesh().uv(4, 3)),),
            MeshMaterial3d(debug_material.clone()),
            Transform::from_xyz(
                ((i%1000) as f32)*0.01 - 5.0,
                2.0,
                ((i/1000) as f32)*0.01 - 5.0,
            ),
            // .with_rotation(Quat::from_rotation_x(-PI / 4.)),
            Shape,
        ));
    }

    // let num_shapes = shapes.len();

    // for (i, shape) in shapes.into_iter().enumerate() {
    //     commands.spawn((
    //         Mesh3d(shape),
    //         MeshMaterial3d(debug_material.clone()),
    //         Transform::from_xyz(
    //             -SHAPES_X_EXTENT / 2. + i as f32 / (num_shapes - 1) as f32 * SHAPES_X_EXTENT,
    //             2.0,
    //             Z_EXTENT / 2.,
    //         )
    //         .with_rotation(Quat::from_rotation_x(-PI / 4.)),
    //         Shape,
    //     ));
    // }

    commands.spawn((
        PointLight {
            shadows_enabled: true,
            intensity: 10_000_000.,
            range: 100.0,
            shadow_depth_bias: 0.2,
            ..default()
        },
        Transform::from_xyz(8.0, 16.0, 8.0),
    ));

    // ground plane
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(50.0, 50.0).subdivisions(10))),
        MeshMaterial3d(materials.add(Color::from(SILVER))),
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 7., 14.0).looking_at(Vec3::new(0., 1., 0.), Vec3::Y),
    ));

    #[cfg(not(target_arch = "wasm32"))]
    commands.spawn((
        Text::new("Press space to toggle wireframes"),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(12.0),
            left: Val::Px(12.0),
            ..default()
        },
    ));
}

fn sine_wave(mut query: Query<&mut Transform, With<Shape>>, time: Res<Time>) {
    for mut transform in &mut query {
        let time = time.elapsed_secs();

        let d = (transform.translation.x.powi(2) + transform.translation.z.powi(2)).sqrt();
        transform.translation.y = (time-d).sin()*0.5+1.1;
    }
}

/// Creates a colorful test pattern
fn uv_debug_texture() -> Image {
    const TEXTURE_SIZE: usize = 8;

    let mut palette: [u8; 32] = [
        255, 102, 159, 255, 255, 159, 102, 255, 236, 255, 102, 255, 121, 255, 102, 255, 102, 255,
        198, 255, 102, 198, 255, 255, 121, 102, 255, 255, 236, 102, 255, 255,
    ];

    let mut texture_data = [0; TEXTURE_SIZE * TEXTURE_SIZE * 4];
    for y in 0..TEXTURE_SIZE {
        let offset = TEXTURE_SIZE * y * 4;
        texture_data[offset..(offset + TEXTURE_SIZE * 4)].copy_from_slice(&palette);
        palette.rotate_right(4);
    }

    Image::new_fill(
        Extent3d {
            width: TEXTURE_SIZE as u32,
            height: TEXTURE_SIZE as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &texture_data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn toggle_wireframe(
    mut wireframe_config: ResMut<WireframeConfig>,
    keyboard: Res<ButtonInput<KeyCode>>,
) {
    if keyboard.just_pressed(KeyCode::Space) {
        wireframe_config.global = !wireframe_config.global;
    }
}
