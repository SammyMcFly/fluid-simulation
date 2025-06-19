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

use crate::measure::MeasurementSeries;
use crate::physics::System3D;

pub mod physics;
mod gui;
mod mediation;
mod setup;
mod measure;



/// Simple fluid solver written in rust
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// File path to input .toml file with scene info
    #[arg(default_value = "./scene_config.toml")] // short, long,
    config: String,
    /// File path to file with state of all particles of a system, where to start simulating from
    #[arg(short, long, default_value_t=String::from(""))]
    state: String,
    /// Log severity level (Options: TRACE, DEBUG, INFO, WARN, ERROR, OFF)
    #[arg(short, long, default_value_t=String::from("DEBUG"))]
    log: String,
    /// Store measurements to .csv file
    #[arg(short, long, default_value_t=String::from(""))]
    measurement_file: String,
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

    let measurement_series = if !args.measurement_file.is_empty() {
        Some(Arc::new(Mutex::new(MeasurementSeries::default())))
    } else {
        None
    };
    let moved_measurement_series = measurement_series.clone();

    // init queue and controls connecting simulation backend with graphics front end
    let controls = Arc::new(Mutex::new(mediation::IntermediateControls::default()));

    // load simulation system
    let (mut system_at_time_0, mut buffer_length) = if !args.state.is_empty() {
        if let Ok(res) = System3D::load_state(&args.state, &args.config, controls.clone(), measurement_series.clone()) {
            res
        } else {
            println!("Invalid state file!");
            System3D::from_config(&args.config, controls.clone(), measurement_series.clone())
            .expect("Invalid scene file!")
        }
    } else {
        System3D::from_config(&args.config, controls.clone(), measurement_series.clone())
        .expect("Invalid scene file!")
    };
    // pass on initial position for visualization
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
                if system.controls.lock().unwrap().is_saving_requested() {
                    if system.save_state("state.ron").is_ok() {
                        println!("Successfully saved state!");
                    } else {
                        println!("Failed to save state!");
                    }
                    system.controls.lock().unwrap().saving_done();
                }
                if system.controls.lock().unwrap().is_reset_requested() {
                    // reload simulation system
                    if !args.state.is_empty() {
                        match System3D::load_state(&args.state, &args.config, system.controls.clone(), moved_measurement_series.clone()) {
                            Ok(new_system) => {
                                (system_at_time_0, buffer_length) = new_system;
                                system = system_at_time_0.clone();
                            },
                            _ => {
                                system = system_at_time_0.clone();
                                println!("Invalid state or scene file!");
                            },
                        }
                    } else {
                        match System3D::from_config(&args.config, system.controls.clone(), moved_measurement_series.clone()) {
                            Ok(new_system) => {
                                (system_at_time_0, buffer_length) = new_system;
                                system = system_at_time_0.clone();
                            },
                            _ => {
                                system = system_at_time_0.clone();
                                println!("Invalid scene file!");
                            },
                        };
                    }
                    system.controls.lock().unwrap().particle_positions.clear();
                    system.queue_for_visualization();
                    system.controls.lock().unwrap().reset_done();
                } else if system.controls.lock().unwrap().particle_positions.len() >= buffer_length {
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

    let mut app = gui::StateApplication::new(controls.clone());

    let _ = event_loop.run_app(&mut app);

    handle.join().expect("Couldn't join simulation thread");

    // save measurements
    if let Some(ms) = measurement_series {
        ms.lock().unwrap().save(&args.measurement_file)?;
    }
    Ok(())
}
