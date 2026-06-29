/// # Rusty fluid solver
///
/// Fluid solver aiming to be powerful
///
/// flamegraph profiling:
/// - cargo flamegraph -- ./scene_config.toml
use clap::Parser;
use iced_winit::winit::event_loop::{ControlFlow, EventLoop};

// use tracing::info; // debug, error, info, span, trace, warn};
use tracing::level_filters::LevelFilter;
use tracing_subscriber::FmtSubscriber;

mod app;

use app::messages::WorkerMessage;

/// Simple fluid solver written in rust.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// File path to input .toml file with scene info
    // #[arg(default_value_t=String::from("scene_config.toml"))]
    params: String,
    /// File path to file, which the to be simulated scene is defined in
    #[arg(long)]
    scene: String,
    /// File path to file with state of all fluid particles of a system, where to start simulating from
    ///
    /// If this argument is set the fluid defined in --scene is ignored.
    #[arg(long)]
    state: Option<String>,
    /// File path to store measurements to (.csv file)
    #[arg(short, long)]
    measurement_file: Option<String>,
    /// File path to store time step info between start_time and end_time to
    #[arg(long)]
    recording_file: Option<String>,
    /// File path to store rendered images to (.png files)
    #[arg(long)]
    rendering_dir: Option<String>,
    /// Time, which measurement/recording/rendering is started at
    #[arg(short, long)]
    start_time: Option<f64>,
    /// Time, which measurement/recording/rendering is terminated at
    ///
    /// At the same time the simulation is paused. Currently there is no possibility to resume the simulation.
    #[arg(short, long)]
    finish_time: Option<f64>,
    /// Resume playback at start
    #[arg(short, long)]
    resume: bool,
    /// Exit when finished
    #[arg(short, long)]
    exit: bool,
    /// Log severity level (Options: TRACE, DEBUG, INFO, WARN, ERROR, OFF)
    #[arg(short, long, default_value_t=String::from("INFO"))]
    log: String,
}

/// Init logging
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
