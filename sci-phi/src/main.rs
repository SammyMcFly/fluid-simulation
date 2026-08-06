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

/// Advanced fluid solver written in rust.
#[derive(Parser, Debug, Clone)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// File path to input .toml file with simulation parameters
    pub params: String,
    /// File path to scene definition
    #[arg(long)]
    pub scene: String,
    /// File path to initial state (overrides fluid in scene)
    #[arg(long)]
    pub state: Option<String>,
    /// File path for measurement output (.csv)
    #[arg(short, long)]
    pub measurement_file: Option<String>,
    /// File path to store recorded timesteps to
    #[arg(long)]
    pub recording_file: Option<String>,
    /// Directory for rendered image output
    #[arg(long)]
    pub rendering_dir: Option<String>,
    /// Start time for measurement/recording/rendering
    #[arg(short, long)]
    pub start_time: Option<f64>,
    /// Finish time for measurement/recording/rendering
    #[arg(short, long)]
    pub finish_time: Option<f64>,
    /// Resume playback at start
    #[arg(short, long)]
    pub resume: bool,
    /// Exit when finished
    #[arg(short, long)]
    pub exit: bool,
    /// Log severity level (TRACE, DEBUG, INFO, WARN, ERROR, OFF)
    #[arg(short, long, default_value_t = String::from("INFO"))]
    pub log: String,
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
