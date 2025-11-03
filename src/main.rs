//! # Rusty fluid solver
//!
//! Sketched fluid solver aiming to be powerful
//!
//! BUGS:
//! -
//!
//! IDEAS:
//! -
//!
//! DONE:
//! -
//!
//! flamegraph profiling:
//! - cargo flamegraph -- ./scene_config.toml
use std::thread;
use std::sync::{Arc, Mutex};
use clap::Parser;
use iced_winit::winit::{
    event_loop::{ControlFlow, EventLoop},
};

#[cfg(feature = "logging")]
use tracing::info;
#[cfg(feature = "logging")]
use tracing::level_filters::LevelFilter;
#[cfg(feature = "logging")]
use tracing_subscriber::FmtSubscriber;
// #[cfg(feature = "logging")]
// use tracing::debug; // , error, info, span, trace, warn};

use crate::measure::MeasurementSeries;
use crate::physics::System3D;

pub mod physics;
mod gui;
mod mediation;
mod setup;
mod measure;


#[cfg(all(feature = "local_pressure", feature = "global_pressure"))]
compile_error!("Features `local_pressure` und `global_pressure` schließen sich gegenseitig aus.");


/// Simple fluid solver written in rust
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// File path to input .toml file with scene info
    // #[arg(default_value = "./scene_config.toml")] // short, long,
    config: String,
    /// File path to file with state of all particles of a system, where to start simulating from
    #[arg(short, long, default_value_t=String::from(""))]
    state: String,
    /// Log severity level (Options: TRACE, DEBUG, INFO, WARN, ERROR, OFF)
    #[arg(short, long, default_value_t=String::from("OFF"))]
    log: String,
    /// Store measurements to .csv file
    #[arg(short, long, default_value_t=String::from(""))]
    measurement_file: String,
}

/// Init logging
#[cfg(feature = "logging")]
fn init_logging(args: &Args) {
    let severity_level = match &args.log[..] {
        "TRACE" => LevelFilter::TRACE,
        "DEBUG" => LevelFilter::DEBUG,
        "INFO" => LevelFilter::INFO,
        "WARN" => LevelFilter::WARN,
        "ERROR" => LevelFilter::ERROR,
        _ => LevelFilter::OFF,
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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // parse args
    let args = Args::parse();

    if cfg!(feature = "logging") {
        init_logging(&args);
    }

    let measurement_series = if !args.measurement_file.is_empty() {
        Some(Arc::new(Mutex::new(MeasurementSeries::default())))
    } else {
        None
    };
    let moved_measurement_series = measurement_series.clone();

    // init queue and controls connecting simulation backend with graphics front end
    let controls = Arc::new(Mutex::new(mediation::IntermediateControls::default()));
    // clone controls for graphics front end
    let controls_front_end = controls.clone();

    // load simulation system
    let (mut system_at_time_0, mut buffer_length, mut int_scheme) = if !args.state.is_empty() {
        if let Ok((sys_conf, buf_len, int_scheme)) = setup::System3DConfigConstructor::new(&args.config, Some(&args.state), controls.clone(), measurement_series.clone()) {
            (System3D::new(sys_conf.finish()), buf_len, int_scheme)
        } else {
            println!("Invalid state file!");
            let (sys_conf, buf_len, int_scheme) = setup::System3DConfigConstructor::new(&args.config, None, controls.clone(), measurement_series.clone())
                .expect("Invalid scene file!");
            (System3D::new(sys_conf.finish()), buf_len, int_scheme)
        }
    } else {
        let (sys_conf, buf_len, int_scheme) = setup::System3DConfigConstructor::new(&args.config, None, controls.clone(), measurement_series.clone())
            .expect("Invalid scene file!");
        (System3D::new(sys_conf.finish()), buf_len, int_scheme)
    };
    // pass on initial position for visualization
    {
        controls.lock().unwrap().queue_for_visualization(&system_at_time_0.particles, &system_at_time_0.boundary_particles, system_at_time_0.get_average_mass_density());
    }
    // system_at_time_0.queue_for_visualization();

    // run simulation in separate thread: Calculate new positions if queue not full
    let handle = thread::spawn(move || {
        // clone simulation system for time propagation (keep clone of initial state)
        let mut system = system_at_time_0.clone();
        loop {
            if controls.lock().unwrap().is_connection_terminated() {
                break;
            }
            {
                if controls.lock().unwrap().is_saving_requested() {
                    if system.save_state("state.ron").is_ok() {
                        println!("Successfully saved state!");
                    } else {
                        println!("Failed to save state!");
                    }
                    controls.lock().unwrap().saving_done();
                }
                if controls.lock().unwrap().is_reset_requested() {
                    // reload simulation system
                    if !args.state.is_empty() {
                        match setup::System3DConfigConstructor::new(&args.config, Some(&args.state), controls.clone(), moved_measurement_series.clone()) {
                            Ok((sys_conf, buf_len, scheme)) => {
                                (system_at_time_0, buffer_length, int_scheme) = (System3D::new(sys_conf.finish()), buf_len, scheme);
                                system = system_at_time_0.clone();
                            },
                            _ => {
                                system = system_at_time_0.clone();
                                println!("Invalid state or scene file!");
                            },
                        }
                    } else {
                        match setup::System3DConfigConstructor::new(&args.config, None, controls.clone(), moved_measurement_series.clone()) {
                            Ok((sys_conf, buf_len, scheme)) => {
                                (system_at_time_0, buffer_length, int_scheme) = (System3D::new(sys_conf.finish()), buf_len, scheme);
                                system = system_at_time_0.clone();
                            },
                            _ => {
                                system = system_at_time_0.clone();
                                println!("Invalid scene file!");
                            },
                        };
                    }
                    controls.lock().unwrap().particle_positions.clear();
                    controls.lock().unwrap().boundary_particle_positions.clear();
                    controls.lock().unwrap().queue_for_visualization(&system.particles, &system.boundary_particles, system.get_average_mass_density());
                    controls.lock().unwrap().reset_done();
                } else if controls.lock().unwrap().particle_positions.len() >= buffer_length {
                    // thread::sleep(std::time::Duration::from_millis(300));
                    continue;
                // } else if system.controls.lock().unwrap().particle_positions.len() >= 300 { // for profiling
                //     break;
                } else {
                    system.inc_time(&int_scheme);
                    controls.lock().unwrap().queue_for_visualization(&system.particles, &system.boundary_particles, system.get_average_mass_density());
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

    let mut app = gui::StateApplication::new(controls_front_end);

    let _ = event_loop.run_app(&mut app);

    handle.join().expect("Couldn't join simulation thread");

    // save measurements
    if let Some(ms) = measurement_series {
        ms.lock().unwrap().save(&args.measurement_file)?;
        info!("Saved measurements to {}", &args.measurement_file);
        println!("Saved measurements to {}", &args.measurement_file);
    }
    Ok(())
}
