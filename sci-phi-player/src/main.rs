//! # Sci-Phi
//!
//! Advanced fluid solver
//!
//! flamegraph profiling:
//! - cargo flamegraph -- ./scene_config.toml
//!
//! SPDX-License-Identifier: MPL-2.0
mod app;
mod config;
mod i18n;

use clap::Parser;

/// Program for playing back a recording of an SPH fluid simulation create by Sci-PHi.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// File path to recording of a rusty fluid solver simulation binary file
    // #[arg(default_value_t=String::from("scene_config.toml"))]
    recording: String,
    /// Resume playback at start
    #[arg(short, long)]
    resume: bool,
    /// File path to store rendered images to (.png files)
    #[arg(long)]
    rendering_dir: Option<String>,
    /// Time, which the first measurement/recording is taken at
    #[arg(short, long)]
    start_time: Option<f64>,
    /// Time, which the final measurement/recording is taken at
    ///
    /// At the same time the simulation is paused. Currently there is no possibility to resume the simulation.
    #[arg(short, long)]
    finish_time: Option<f64>,
    /// Log severity level (Options: TRACE, DEBUG, INFO, WARN, ERROR, OFF)
    #[arg(short, long, default_value_t=String::from("INFO"))]
    log: String,
}

fn main() -> cosmic::iced::Result {
    let args = Args::parse();
    init_logging(&args);

    // Get the system's preferred languages.
    let requested_languages = i18n_embed::DesktopLanguageRequester::requested_languages();

    // Enable localizations to be applied.
    i18n::init(&requested_languages);

    // Settings for configuring the application window and iced runtime.
    let settings = cosmic::app::Settings::default()
        .size_limits(
            cosmic::iced::Limits::NONE
                .min_width(1000.0)
                .min_height(700.0),
        )
        .exit_on_close(false);

    // Starts the application's event loop with `()` as the application's flags.
    cosmic::app::run::<app::AppModel>(settings, args)
}

fn init_logging(args: &Args) {
    use tracing::level_filters::LevelFilter;
    use tracing_subscriber::FmtSubscriber;

    let level = match args.log.as_str() {
        "TRACE" => LevelFilter::TRACE,
        "DEBUG" => LevelFilter::DEBUG,
        "INFO" => LevelFilter::INFO,
        "WARN" => LevelFilter::WARN,
        "ERROR" => LevelFilter::ERROR,
        _ => LevelFilter::OFF,
    };

    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .with_writer(std::io::stdout)
        .with_line_number(true)
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("Failed to set subscriber");
}
