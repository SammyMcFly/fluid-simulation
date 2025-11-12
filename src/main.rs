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

pub mod physics;
mod gui;
mod mediation;
mod setup;
mod measure;


#[cfg(all(feature = "local_pressure", feature = "global_pressure"))]
compile_error!("Only one of the features `local_pressure` and `global_pressure` can be activated at the same time.");
#[cfg(all(not(feature = "local_pressure"), not(feature = "global_pressure")))]
compile_error!("One of the features `local_pressure` and `global_pressure` must be activated.");



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
    #[arg(short, long, default_value_t=String::from("DEBUG"))]
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

    #[cfg(feature = "logging")]
    init_logging(&args);

    let measurement_series = if !args.measurement_file.is_empty() {
        Some(Arc::new(Mutex::new(measure::MeasurementSeries::default())))
    } else {
        None
    };
    let moved_measurement_series = measurement_series.clone();

    // init queue and controls connecting simulation backend with graphics front end
    let controls = Arc::new(Mutex::new(mediation::IntermediateControls::default()));
    // clone controls for graphics front end
    let controls_front_end = controls.clone();

    // load simulation system
    let (system_at_time_0, buffer_length, int_scheme) = if !args.state.is_empty() {
        if let Ok((sys_conf, buf_len, int_scheme)) = setup::System3DConfigConstructor::new(&args.config, Some(&args.state), controls.clone(), measurement_series.clone()) {
            (physics::System3D::new(sys_conf.finish()), buf_len, int_scheme)
        } else {
            println!("Invalid state file!");
            let (sys_conf, buf_len, int_scheme) = setup::System3DConfigConstructor::new(&args.config, None, controls.clone(), measurement_series.clone())
                .expect("Invalid scene file!");
            (physics::System3D::new(sys_conf.finish()), buf_len, int_scheme)
        }
    } else {
        let (sys_conf, buf_len, int_scheme) = setup::System3DConfigConstructor::new(&args.config, None, controls.clone(), measurement_series.clone())
            .expect("Invalid scene file!");
        (physics::System3D::new(sys_conf.finish()), buf_len, int_scheme)
    };
    // pass on initial position for visualization
    {
        controls.lock().unwrap().queue_for_visualization(&system_at_time_0.particles, &system_at_time_0.boundary_particles, system_at_time_0.get_average_mass_density());
    }

    // run simulation in separate thread: Calculate new positions if queue not full
    let handle = physics::run_system_in_thread(
        system_at_time_0,
        buffer_length,
        int_scheme,
        controls,
        args.state,
        args.config,
        moved_measurement_series,
    );

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
        #[cfg(feature = "logging")]
        info!("Saved measurements to {}", &args.measurement_file);
        println!("Saved measurements to {}", &args.measurement_file);
    }
    Ok(())
}
