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
use clap::Parser;
use iced_winit::winit::event_loop::{ControlFlow, EventLoop};

// #[cfg(feature = "logging")]
// use tracing::info; // debug, error, info, span, trace, warn};
#[cfg(feature = "logging")]
use tracing::level_filters::LevelFilter;
#[cfg(feature = "logging")]
use tracing_subscriber::FmtSubscriber;

mod app;

use app::messages::WorkerMessage;


#[cfg(all(feature = "local_pressure", feature = "global_pressure"))]
compile_error!("Only one of the features `local_pressure` and `global_pressure` can be activated at the same time.");
#[cfg(all(not(feature = "local_pressure"), not(feature = "global_pressure")))]
compile_error!("One of the features `local_pressure` and `global_pressure` must be activated.");



/// Simple fluid solver written in rust
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// File path to input .toml file with scene info
    // #[arg(default_value_t=String::from("scene_config.toml"))]
    config: Option<String>,
    /// File path to file with state of all particles of a system, where to start simulating from
    #[arg(short, long)]
    state: Option<String>,
    /// File path to store measurements to (.csv file)
    #[arg(short, long,)]
    measurement_file: Option<String>,
    /// Time, which the first measurement/recording is taken at
    #[arg(short, long,)]
    start_time: Option<f64>,
    /// Time, which the final measurement/recording is taken at
    ///
    /// At the same time the simulation is paused. Currently there is no possibility to resume the simulation.
    #[arg(short, long,)]
    finish_time: Option<f64>,
    /// File path to store time step info between start_time and end_time to
    #[arg(long,)]
    recording_file: Option<String>,
    /// Resume playback at start
    #[arg(short, long,)]
    resume: bool,
    /// Exit when finished
    #[arg(short, long,)]
    exit: bool,
    /// Log severity level (Options: TRACE, DEBUG, INFO, WARN, ERROR, OFF)
    #[arg(short, long, default_value_t=String::from("INFO"))]
    log: String,
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

    let event_loop = EventLoop::<WorkerMessage>::with_user_event()
        .build()
        .expect("Failed to build event loop!");

    // ControlFlow::Poll continuously runs the event loop, even if the OS hasn't
    // dispatched any events. This is ideal for games and similar applications.
    event_loop.set_control_flow(ControlFlow::Poll);

    // ControlFlow::Wait pauses the event loop if no events are available to process.
    // This is ideal for non-game applications that only update in response to user
    // input, and uses significantly less power/CPU time than ControlFlow::Poll.
    // event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = app::StateApplication::new(&event_loop, args);

    let _ = event_loop.run_app(&mut app);

    Ok(())
}
