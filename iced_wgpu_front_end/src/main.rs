//! # Rusty fluid solver
//!
//! Sketched fluid solver aiming to be powerful
//!
//! BUGS:
//! -
//!
//! IDEAS:
//! - Implement time incremation by user (display one frame at a time)
//! - Multithreading
//!
//! DONE:
//! - Split particle_diameter up into particle_diameter and smoothing length
//! - Display 'filled buffer length' or precumputed timesteps or precomputed time in [s]
//! - Exchanged std::collections::HashMap for rustc_hash::FxHashMap for performance reasons:
//! -   When there were massive hash collisions, this change would cause a performance penalty.
//! -   Since hash collisions happen when there all multiple particles in one grid cell or different grid cells
//! -   map to the same hash value, this is a non-problem, because of pressure forces prevent
//! -   too many particles in one grid cell. And in general, not too many grid cells are used at all.
//! - Implemented skipping of queue elements when time between frames is greater than twice the time increment.
//! -   This should better syncronize the system time with real user time.
//!
//! flamegraph profiling:
//! - cargo flamegraph -- ./scene_config.toml
use std::thread;
use std::sync::{Arc, Mutex};

use tracing::level_filters::LevelFilter;
use tracing_subscriber::FmtSubscriber;
// use tracing::debug; // , error, info, span, trace, warn};
use clap::Parser;

use iced_winit::winit::{
    // event::WindowEvent,
    event_loop::{ControlFlow, EventLoop},
    // keyboard::ModifiersState,
};

pub mod physics;
mod gui;
mod mediation;
mod setup;



fn fetch_file_info(
    config_file: &str,
    queue: Arc<Mutex<mediation::IntermediateQueue>>,
    controls: Arc<Mutex<mediation::IntermediateControls>>
) -> Result<(physics::System3D, u32), Box<dyn std::error::Error>> {
    // Read the scene config file
    let config_file_content = std::fs::read_to_string(config_file)?;
    // Parse the content into the Config struct
    let config: setup::Config = toml::from_str(&config_file_content)?;

    // hand over time_inc
    controls.lock().unwrap().set_time_inc(config.params.time_inc as f32);
    // hand over particle size
    controls.lock().unwrap().set_particle_size(config.params.particle_diameter as f32);
    // hand over rest density
    controls.lock().unwrap().set_rest_density((config.params.particle_mass/config.params.particle_diameter.powi(3)) as f32);
    // hand over light position
    controls.lock().unwrap().set_light_position(config.scene.light.position);
    // init system properties
    let system_properties = physics::SystemProperties::new(
        config.params.time_inc,
        config.params.particle_mass,
        config.params.particle_diameter,
        config.params.viscosity,
        config.params.stiffness,
        physics::cubic_spline_3d,
        physics::cubic_spline_3d_gradient
    );
    // create simulation system
    Ok((physics::System3D::from_config(
        &config,
        system_properties,
        queue,
        controls,
    ), config.params.buffer_length_limit))
}



/// Simple fluid solver written in rust
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// File path to input .toml file with scene info
    #[arg(default_value = "./scene_config.toml")] // short, long,
    scene: String,
    /// Log severity level (Options: TRACE, DEBUG, INFO, WARN, ERROR, OFF)
    #[arg(short, long, default_value_t=String::from("DEBUG"))]
    log: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // parse args
    let args = Args::parse();
    // init logging
    let severity_level = match &args.log[..] {
        "TRACE" => LevelFilter::TRACE,
        "DEBUG" => LevelFilter::DEBUG,
        "INFO" => LevelFilter::INFO,
        "WARN" => LevelFilter::WARN,
        "ERROR" => LevelFilter::ERROR,
        _ => LevelFilter::OFF,
    };
    // init logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(severity_level)
        .with_writer(std::io::stdout)
        .with_line_number(true)
        // .with_ansi(false)
        // .pretty()
        .finish();
        // .with(debug_log);
    tracing::subscriber::set_global_default(subscriber).expect("Failed to set subscriber");


    // init queue and controls connecting simulation backend with graphics front end
    let queue = Arc::new(Mutex::new(mediation::IntermediateQueue::default()));
    let controls = Arc::new(Mutex::new(mediation::IntermediateControls::default()));

    let (mut system_at_time_0, mut buffer_length) = fetch_file_info(&args.scene, queue.clone(), controls.clone())
        .expect("Invalid scene file!");
    system_at_time_0.queue_for_visualization();

    // run simulation in separate thread: Calculate new positions if queue not full
    let handle = thread::spawn(move || {
        // clone simulation system for time propagation (keep clone of initial state)
        let mut system = system_at_time_0.clone();
        loop {
            if system.controls.lock().unwrap().is_connection_terminated() {
                break;
            }
            {
                if system.controls.lock().unwrap().is_reset_requested() {
                    // reread config file to update simulation system
                    match fetch_file_info(&args.scene, system.queue.clone(), system.controls.clone()) {
                        Ok(system_buffer_length) => {
                            (system_at_time_0, buffer_length) = system_buffer_length;
                            system = system_at_time_0.clone();
                        },
                        _ => {
                            system = system_at_time_0.clone();
                            println!("Invalid scene file!");
                        },
                    };
                    system.queue.lock().unwrap().clear();
                    system.queue_for_visualization();
                    system.controls.lock().unwrap().reset_done();
                } else if system.queue.lock().unwrap().len() >= buffer_length {
                    // thread::sleep(std::time::Duration::from_millis(300));
                    continue;
                // if system.queue.lock().unwrap().len() >= 100 { // for profiling
                    // break;
                } else {
                    system.inc_time(physics::PropagationMethod::EulerCromer);
                    system.queue_for_visualization();
                }
            }
        }
    });

    let event_loop = EventLoop::new().unwrap();

    // ControlFlow::Poll continuously runs the event loop, even if the OS hasn't
    // dispatched any events. This is ideal for games and similar applications.
    event_loop.set_control_flow(ControlFlow::Poll);

    // ControlFlow::Wait pauses the event loop if no events are available to process.
    // This is ideal for non-game applications that only update in response to user
    // input, and uses significantly less power/CPU time than ControlFlow::Poll.
    // event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = gui::StateApplication::new(queue.clone(), controls.clone());

    let _ = event_loop.run_app(&mut app);

    handle.join().expect("Couldn't join simulation thread");
    Ok(())
}
