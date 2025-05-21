//! # Rusty fluid solver
//!
//! Sketched fluid solver aiming to be powerful
//!
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
) -> Result<physics::System3D, Box<dyn std::error::Error>> {
    // Read the scene config file
    let config_file_content = std::fs::read_to_string(config_file)?;
    // Parse the content into the Config struct
    let config: setup::Config = toml::from_str(&config_file_content)?;

    // hand over time_inc
    controls.lock().unwrap().set_time_inc(config.params.time_inc as f32);
    // hand over particle size
    controls.lock().unwrap().set_particle_size(config.params.particle_size as f32);
    // hand over light position
    controls.lock().unwrap().set_light_position(config.scene.light.position);
    // init system properties
    let system_properties = physics::SystemProperties::new(
        config.params.time_inc,
        config.params.particle_mass,
        config.params.particle_size,
        config.params.viscosity,
        config.params.stiffness,
        physics::cubic_spline_3d,
        physics::cubic_spline_3d_gradient
    );
    // create simulation system
    Ok(physics::System3D::from_config(
        &config,
        system_properties,
        queue,
        controls,
    ))
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

    let system = fetch_file_info(&args.scene, queue.clone(), controls.clone()).unwrap();

    let event_loop = EventLoop::new().unwrap();

    // ControlFlow::Poll continuously runs the event loop, even if the OS hasn't
    // dispatched any events. This is ideal for games and similar applications.
    event_loop.set_control_flow(ControlFlow::Poll);

    // ControlFlow::Wait pauses the event loop if no events are available to process.
    // This is ideal for non-game applications that only update in response to user
    // input, and uses significantly less power/CPU time than ControlFlow::Poll.
    // event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = gui::StateApplication::new(queue.clone(), controls.clone());

    // run simulation in separate thread: Calculate new positions if queue not full
    let handle = thread::spawn(move || {
        // move simulation system
        let mut system = system;
        loop {
            {
                let reset_requested = system.controls.lock().unwrap().is_reset_requested();
                if reset_requested {
                    // reread config file to update simulation system
                    system = fetch_file_info(&args.scene, system.queue.clone(), system.controls.clone()).unwrap();
                    system.controls.lock().unwrap().reset_done();
                }
            }
            {
                let controls = system.controls.lock().unwrap();
                if controls.is_connection_terminated() {
                    break;
                }
            }
            {
                let queue = system.queue.lock().unwrap();
                if queue.len() >= 300 {
                    // debug!("len: {}", q.len());
                    // thread::sleep(std::time::Duration::from_millis(300));
                    continue;
                }
            }
            system.inc_time(physics::PropagationMethod::EulerCromer);
        }
    });

    let _ = event_loop.run_app(&mut app);
    handle.join().expect("Couldn't join simulation thread");

    Ok(())
}
