// SPDX-License-Identifier: MPL-2.0
mod inspector;
mod pages;
mod playback;

use crate::app::pages::simulation::{
    BoundaryVisOption, FluidVisOption, QuantityOption, SensorField,
};
use crate::app::pages::{ContextPage, Page};
use crate::app::playback::InsertionResult;
use crate::config::Config;
use crate::fl;
use pages::plotting::PlottingSettings;
use pages::simulation::SimulationSettings;
use playback::{FrameControl, InstanceStore, PlaybackControls, StagingResult};

use cosmic::app::context_drawer;
use cosmic::cosmic_config::{self, CosmicConfigEntry};
use cosmic::iced::alignment::{Horizontal, Vertical};
use cosmic::iced::widget as iced_widget;
use cosmic::iced::{Alignment, Length, Subscription};
use cosmic::widget::segmented_button::Entity;
use cosmic::widget::{self, about::About, icon, menu, nav_bar, row};
use cosmic::{prelude::*, theme};
use rendering_lib::colormap::Colormap;
use rendering_lib::primitive::{ScreenshotRequest, ScreenshotTarget};
use rendering_lib::{CameraState, LightState, SimulationViewport, ViewportEvent, build_scene_data};
use sci_phi_backend::commands::WorkerCommand;
use sci_phi_backend::messages::WorkerMessage;
use sci_phi_backend::worker_loop;
use simulation_lib::measurement::{MeasurementSeries, RecordingStatus};
use simulation_lib::render_info::{
    FluidSampleColoring, FluidVisualization, ScalarQuantity, TimeStepInfo,
};
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::PathBuf;
use tracing::{error, info, warn};

const REPOSITORY: &str = env!("CARGO_PKG_REPOSITORY");
const APP_ICON: &[u8] = include_bytes!("../../resources/icons/hicolor/scalable/apps/icon.svg");

/// The application model stores app-specific state used to describe its interface and
/// drive its logic.
pub struct AppModel {
    /// Application state which is managed by the COSMIC runtime.
    core: cosmic::Core,
    /// Contains items assigned to the nav bar panel.
    nav: nav_bar::Model,
    /// Viewport for the fluid simulation.
    simulation_page: SimulationViewport<WorkerCommand>,
    /// Plotting
    plotting_page: pages::plotting::PlottingViewport,
    /// Display a context drawer with the designated page if defined.
    context_page: ContextPage,
    /// The simulation setting context page.
    sim_settings: SimulationSettings,
    render_template: TimeStepInfo,
    /// The plotting settings context page.
    plot_settings: PlottingSettings,
    /// The about page for this app.
    about: About,
    /// Inspector: info and logs
    inspector: inspector::Inspector,
    // in AppModel
    dialog_page: Option<PendingAction>,
    /// Key bindings for the application's menu bar.
    key_binds: HashMap<menu::KeyBind, MenuAction>,
    /// Configuration data that persists between application runs.
    config: Config,
    // Playback
    instances: InstanceStore,
    frame: FrameControl,
    playback: PlaybackControls,
    last_tick: std::time::Instant,
    // Backend communication
    to_worker: crossbeam::channel::Sender<WorkerCommand>,
    from_worker: crossbeam::channel::Receiver<WorkerMessage>,
    worker_handle: Option<std::thread::JoinHandle<()>>,
    /// Measurement series used for plotting
    plotting_measurement_series: MeasurementSeries,
    /// CLI rendering state (render every frame to disk)
    rendering: Option<RenderingState>,
    /// Close program when finish time is reached
    exit_when_finished: bool,
}

/// Messages emitted by the application and its widgets.
#[derive(Debug, Clone)]
pub enum Message {
    ClosePressed(cosmic::iced::window::Id),
    RequestClose,
    ToggleInspector,
    InspectorTabSelected(Entity),
    ToggleContextPage(ContextPage),
    LaunchUrl(String),
    UpdateConfig(Config),
    DialogConfirm,
    DialogCancel,
    Reload,
    ResetCamera,
    TakeScreenshot,
    /// File dialog returned a path for single screenshot
    ScreenshotPathChosen(PathBuf, bool), // true = unpause after file saved, false = do not unpause after file saved
    SaveCurrentState,
    /// File dialog returned a path for saving the current state
    StatePathChosen(PathBuf, bool),
    /// File dialog was cancelled
    Cancelled,
    Play,
    Pause,
    StepForward,
    StepBackward,
    // Viewport events
    Viewport(ViewportEvent),
    // Camera tick (from subscription)
    CameraTick,
    SetFluidVisualization(usize),
    SetFluidQuantity(usize),
    SensorPlaneInput(SensorField, String),
    SensorPlaneStep(SensorField, bool), // true = +, false = −
    ApplySensorPlaneConfig,
    SetColormap(usize),
    ColorMappingMaxInput(String),
    ColorMappingMaxStep(f32, bool), // true = +, false = −
    ApplyColorMappingMax,
    SetBoundaryVisualization(usize),
    ToggleHideBoundary,
    SetBoundaryAlpha(f32),
    ToggleCutX,
    ToggleCutZ,
    ToggleCutY,
    FlipCutY,
    FlipCutX,
    FlipCutZ,
    CutXBoundChanged(f32),
    CutYBoundChanged(f32),
    CutZBoundChanged(f32),
    CutXBoundInput(String),
    CutYBoundInput(String),
    CutZBoundInput(String),
    ToggleCutBoundary,
    ToggleDiscardPast,
    DiscardNow,
    ToggleLoop,
    ToggleInvertTime,
}

// Enable conversion from ViewportEvent → Message
impl From<ViewportEvent> for Message {
    fn from(event: ViewportEvent) -> Self {
        Message::Viewport(event)
    }
}

/// Create a COSMIC application from the app model
impl cosmic::Application for AppModel {
    /// The async executor that will be used to run your application's commands.
    type Executor = cosmic::executor::Default;

    /// Data that your application receives to its init method.
    type Flags = crate::Args;

    /// Messages which the application and its widgets will emit.
    type Message = Message;

    /// Unique identifier in RDNN (reverse domain name notation) format.
    const APP_ID: &'static str = "dev.sammymcfly.SciPhi";

    fn core(&self) -> &cosmic::Core {
        &self.core
    }

    fn core_mut(&mut self) -> &mut cosmic::Core {
        &mut self.core
    }

    /// Initializes the application with any given flags and startup commands.
    fn init(
        mut core: cosmic::Core,
        args: Self::Flags,
    ) -> (Self, Task<cosmic::Action<Self::Message>>) {
        // set the context drawer to not be drawn over the content
        core.window.context_is_overlay = false;
        // Create a nav bar with page items.
        let mut nav = nav_bar::Model::default();

        nav.insert()
            .text(fl!("simulation"))
            .data::<Page>(Page::Simulation)
            .icon(icon::from_name("applications-science-symbolic"))
            .activate();

        nav.insert()
            .text(fl!("plotting"))
            .data::<Page>(Page::Measurements)
            .icon(icon::from_name("applications-system-symbolic"));

        // Create the about widget
        let about = About::default()
            .name(fl!("app-title"))
            .icon(widget::icon::from_svg_bytes(APP_ICON))
            .version(env!("CARGO_PKG_VERSION"))
            .links([(fl!("repository"), REPOSITORY)])
            .license(env!("CARGO_PKG_LICENSE"));

        let camera = CameraState::new(
            (0.0, -60.0, 40.0),
            cgmath::Deg(-90.0),
            cgmath::Deg(-30.0),
            0.25, // speed
            1.25, // sensitivity
            3.0,  // scroll_speed
            cgmath::Deg(45.0),
            0.1,   // znear
            100.0, // zfar
            800,
            600,
        );

        let light = LightState::new([2.0, 2.0, 100.0], [1.0, 1.0, 1.0], 5.0);

        // Backend channels
        let (to_worker_tx, to_worker_rx) = crossbeam::channel::unbounded::<WorkerCommand>();
        let (from_worker_tx, from_worker_rx) = crossbeam::channel::unbounded::<WorkerMessage>();

        let viewport =
            SimulationViewport::new(camera, light, [0.15, 0.15, 0.15, 1.0], to_worker_tx.clone());

        // Start worker thread
        let worker_handle = std::thread::spawn(move || {
            worker_loop(to_worker_rx, from_worker_tx);
        });

        // Send initial simulation command
        let with_info = TimeStepInfo {
            time_step_number: 0,
            measurement: simulation_lib::measurement::Measurement::default(),
            fluid: FluidVisualization::Samples {
                positions: vec![],
                coloring: FluidSampleColoring::QuantityGraded {
                    data: vec![],
                    quantity: ScalarQuantity::Speed,
                },
            },
            boundary: pages::simulation::BoundaryVisOption::MeshOriginal.to_template(),
        };

        to_worker_tx
            .send(WorkerCommand::Simulate {
                params_file_path: args.params.clone(),
                scene_file_path: args.scene.clone(),
                state_file_path: args.state.clone(),
                measurement_file_path: args.measurement_file.clone(),
                start_time: args.start_time,
                finish_time: args.finish_time,
                recording_file: args.recording_file.as_ref().map(std::path::PathBuf::from),
                rendering_dir: args.rendering_dir.as_ref().map(std::path::PathBuf::from),
                with_info: Box::new(with_info.clone()),
            })
            .unwrap();

        // Measurement series
        let measurement_series = MeasurementSeries::default();

        let playback = PlaybackControls::new(args.resume, true);

        // Construct the app model with the runtime's core.
        let mut app = AppModel {
            core,
            nav,
            simulation_page: viewport,
            plotting_page: pages::plotting::PlottingViewport::default(),
            context_page: ContextPage::default(),
            sim_settings: SimulationSettings::from(&with_info),
            render_template: with_info,
            plot_settings: PlottingSettings::default(),
            about,
            inspector: inspector::Inspector::new(
                true,
                args.measurement_file.is_some(),
                args.recording_file.is_some(),
                args.rendering_dir.is_some(),
            ),
            dialog_page: None,
            key_binds: HashMap::new(),
            // Optional configuration file for an application.
            config: cosmic_config::Config::new(Self::APP_ID, Config::VERSION)
                .map(|context| match Config::get_entry(&context) {
                    Ok(config) => config,
                    Err((_errors, config)) => config,
                })
                .unwrap_or_default(),
            frame: FrameControl::default(),
            instances: InstanceStore::default(),
            playback,
            last_tick: std::time::Instant::now(),
            to_worker: to_worker_tx,
            from_worker: from_worker_rx,
            worker_handle: Some(worker_handle),
            // recording_status: RecordingStatus::NotStarted,
            plotting_measurement_series: measurement_series,
            rendering: if args.rendering_dir.is_some() {
                Some(RenderingState::new(args.start_time, args.finish_time))
            } else {
                None
            },
            exit_when_finished: args.exit,
        };

        // Hide navigation bar
        app.core.nav_bar_set_toggled(false);

        // Create a startup command that sets the window title.
        let command = app.update_title();

        (app, command)
    }

    /// Elements to pack at the start of the header bar.
    fn header_start(&self) -> Vec<Element<'_, Self::Message>> {
        vec![] // leer
    }

    fn header_end(&self) -> Vec<Element<'_, Self::Message>> {
        let hamburger_menu = menu::bar(vec![menu::Tree::with_children(
            widget::icon::from_name("open-menu-symbolic")
                .size(20)
                .symbolic(true)
                .apply(widget::container)
                .padding(4)
                .apply(Element::from),
            menu::items(
                &self.key_binds,
                vec![menu::Item::Button(fl!("about"), None, MenuAction::About)],
            ),
        )]);

        vec![hamburger_menu.into()]
    }

    /// Enables the COSMIC application to create a nav bar with this model.
    fn nav_model(&self) -> Option<&nav_bar::Model> {
        Some(&self.nav)
    }

    fn footer(&self) -> Option<Element<'_, Message>> {
        self.inspector.view()
    }

    /// Display a context drawer if the context page is requested.
    fn context_drawer(&self) -> Option<context_drawer::ContextDrawer<'_, Self::Message>> {
        if !self.core.window.show_context {
            return None;
        }

        Some(match self.context_page {
            ContextPage::SimulationSettings => context_drawer::context_drawer(
                &self.sim_settings,
                Message::ToggleContextPage(ContextPage::SimulationSettings),
            ),
            ContextPage::PlottingSettings => context_drawer::context_drawer(
                &self.plot_settings,
                Message::ToggleContextPage(ContextPage::PlottingSettings),
            ),
            ContextPage::About => context_drawer::about(
                &self.about,
                |url| Message::LaunchUrl(url.to_string()),
                Message::ToggleContextPage(ContextPage::About),
            ),
        })
    }

    /// Describes the interface based on the current state of the application model.
    ///
    /// Application events will be processed through the view. Any messages emitted by
    /// events received by widgets will be passed to the update method.
    fn view(&self) -> Element<'_, Self::Message> {
        let space_s = cosmic::theme::spacing().space_s;

        let content: Element<_> = match self.nav.active_data::<Page>().unwrap() {
            Page::Simulation => {
                let top_bar = self.top_bar(Page::Simulation);
                let shader_widget = iced_widget::Shader::new(&self.simulation_page)
                    .width(Length::Fill)
                    .height(Length::Fill);

                widget::column::with_capacity(2)
                    .push(top_bar)
                    .push(shader_widget)
                    .spacing(space_s)
                    .height(Length::Fill)
                    .into()
            }

            Page::Measurements => {
                let top_bar = self.top_bar(Page::Measurements);
                // let viewport = Shader::new(&self.fluid_viewport)
                //     .width(Length::Fill)
                //     .height(Length::Fill);

                widget::column::with_capacity(2)
                    .push(top_bar)
                    // .push(viewport)
                    .spacing(space_s)
                    .height(Length::Fill)
                    .into()
            }
        };

        widget::container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .apply(widget::container)
            .width(Length::Fill)
            .align_x(Horizontal::Center)
            .align_y(Vertical::Center)
            .into()
    }

    /// Register subscriptions for this application.
    ///
    /// Subscriptions are long-running async tasks running in the background which
    /// emit messages to the application through a channel. They can be dynamically
    /// stopped and started conditionally based on application state, or persist
    /// indefinitely.
    fn subscription(&self) -> Subscription<Self::Message> {
        // Camera tick already polls the worker channel
        Subscription::batch(vec![
            cosmic::iced::event::listen_with(|event, _status, window_id| match event {
                cosmic::iced::event::Event::Window(cosmic::iced::window::Event::CloseRequested) => {
                    Some(Message::ClosePressed(window_id))
                }
                _ => None,
            }),
            cosmic::iced::time::every(std::time::Duration::from_millis(16))
                .map(|_| Message::CameraTick),
        ])
    }

    fn on_app_exit(&mut self) -> Option<Message> {
        Some(Message::RequestClose)
    }

    /// Handles messages emitted by the application and its widgets.
    ///
    /// Tasks may be returned for asynchronous execution of code in the background
    /// on the application's async runtime.
    fn update(&mut self, message: Self::Message) -> Task<cosmic::Action<Self::Message>> {
        match message {
            Message::ClosePressed(window_id) => {
                if Some(window_id) == self.core.main_window_id() {
                    return self.update(Message::RequestClose); // your existing path
                }
            }
            Message::RequestClose => {
                return self.request_action(PendingAction::Close);
            }
            Message::ToggleInspector => {
                self.inspector.toggle_show();
            }
            Message::InspectorTabSelected(tab) => {
                self.inspector.activate(tab);
            }
            Message::ToggleContextPage(context_page) => {
                if self.context_page == context_page {
                    // Close the context drawer if the toggled context page is the same.
                    self.core.window.show_context = !self.core.window.show_context;
                } else {
                    // Open the context drawer to display the requested context page.
                    self.context_page = context_page;
                    self.core.window.show_context = true;
                }
            }
            Message::UpdateConfig(config) => {
                self.config = config;
            }
            Message::DialogConfirm => {
                if let Some(action) = self.dialog_page.take() {
                    // recording is stopped by the action itself; advance status if you track it
                    return self.perform(action);
                }
            }
            Message::DialogCancel => {
                self.dialog_page = None;
            }
            Message::LaunchUrl(url) => match open::that_detached(&url) {
                Ok(()) => {}
                Err(err) => {
                    error!("failed to open {url:?}: {err}");
                }
            },
            Message::Reload => {
                return self.request_action(PendingAction::Reload);
            }
            Message::ResetCamera => {
                self.simulation_page.reset_camera();
            }
            Message::TakeScreenshot => {
                let was_playing = self.playback.is_playing();
                self.playback.pause();

                // Open native file dialog asynchronously
                return cosmic::task::future(async move {
                    let title = fl!("file-dialog", "save-state");
                    let filter = fl!("file-dialog", "filter-ron");
                    let dialog = rfd::AsyncFileDialog::new()
                        .set_title(title)
                        .add_filter(filter, &["ron"])
                        .set_file_name("state.ron")
                        .save_file()
                        .await;

                    match dialog {
                        Some(handle) => {
                            Message::ScreenshotPathChosen(handle.path().to_path_buf(), was_playing)
                        }
                        None => Message::Cancelled,
                    }
                })
                .map(cosmic::Action::App);
            }
            Message::ScreenshotPathChosen(path, unpause) => {
                self.simulation_page.request_screenshot(ScreenshotRequest {
                    target: ScreenshotTarget::SingleFile { path },
                });
                if unpause {
                    self.playback.play();
                }
            }
            Message::Cancelled => {
                // Do nothing, user cancelled
            }
            Message::SaveCurrentState => {
                if self.instances.is_active() {
                    let was_playing = self.playback.is_playing();
                    self.playback.pause();

                    // Open native file dialog asynchronously
                    return cosmic::task::future(async move {
                        let title = fl!("file-dialog", "save-screenshot");
                        let filter = fl!("file-dialog", "filter-png");
                        let dialog = rfd::AsyncFileDialog::new()
                            .set_title(title)
                            .add_filter(filter, &["png"])
                            .set_file_name("screenshot.png")
                            .save_file()
                            .await;

                        match dialog {
                            Some(handle) => {
                                Message::StatePathChosen(handle.path().to_path_buf(), was_playing)
                            }
                            None => Message::Cancelled,
                        }
                    })
                    .map(cosmic::Action::App);
                }
            }
            Message::StatePathChosen(path, unpause) => {
                if let Some(info) = self.instances.get_current_time_step_info() {
                    let _ = self.to_worker.send(WorkerCommand::SaveState {
                        time_step_number: info.time_step_number,
                        file_path: path,
                    });
                }
                if unpause {
                    self.playback.play();
                }
            }
            Message::Play => {
                self.frame.reset_steps();
                self.playback.play();
                if self
                    .instances
                    .finished_loop(self.playback.is_playing_forward())
                    && !self.rendering.as_ref().is_some_and(|r| r.active)
                {
                    self.instances.allow_looping_once(self.playback.is_looped());
                }
            }
            Message::Pause => {
                self.playback.pause();
                self.instances.reset_allow_looping_once();
            }
            Message::StepForward => {
                self.frame.step_forward();
            }
            Message::StepBackward => {
                self.frame.step_backward();
            }
            Message::Viewport(event) => match event {
                ViewportEvent::CameraRotated { dx, dy } => {
                    self.simulation_page
                        .camera
                        .controller
                        .process_mouse_motion(dx, dy);
                }
                ViewportEvent::CameraScrolled { delta } => {
                    self.simulation_page.camera.controller.process_scroll(delta);
                }
                ViewportEvent::CameraKey { key, pressed } => {
                    self.simulation_page
                        .camera
                        .controller
                        .process_key(key, pressed);
                }
                ViewportEvent::Resized { width, height } => {
                    self.simulation_page.camera.resize(width, height);
                }
                ViewportEvent::RequestRedraw => {}
            },

            Message::CameraTick => {
                // Clear screenshot request (was consumed by this frame's draw())
                // Only clear screenshot request once the pipeline has consumed it
                if self.simulation_page.is_screenshot_done() {
                    self.simulation_page.screenshot_request = None;
                }

                // 1. Camera + Light tick
                let now = std::time::Instant::now();
                let dt = (now - self.last_tick).as_secs_f32();
                self.last_tick = now;
                self.simulation_page.camera.tick(dt);
                self.simulation_page.light.tick(dt);

                // 2. Poll worker channel (drain all available messages)
                let mut task: Option<Task<cosmic::Action<Message>>> = None;
                while let Ok(msg) = self.from_worker.try_recv() {
                    if let Some(t) = self.handle_worker_message(msg) {
                        task = Some(t);
                        break; // window::close ends everything anyway
                    }
                }

                // 3. Check if rendering mode is waiting for screenshot completion
                let rendering_blocked = if let Some(ref rendering) = self.rendering {
                    rendering.active
                        && rendering.awaiting_capture
                        && !self.simulation_page.is_screenshot_done()
                } else {
                    false
                };

                // If rendering completed capture, mark ready for next frame
                if let Some(ref mut rendering) = self.rendering {
                    if rendering.active
                        && rendering.awaiting_capture
                        && self.simulation_page.is_screenshot_done()
                    {
                        rendering.awaiting_capture = false;
                        rendering.frame_counter += 1;
                    }
                }

                // 4. Playback tick
                let mut frame_new = false;
                if !rendering_blocked {
                    let action = self.frame.get_next_action(
                        self.playback.is_playing(),
                        self.playback.is_playing_forward(),
                        self.rendering.as_ref().is_some_and(|r| r.active),
                    );
                    let result = self.instances.stage_next(
                        action,
                        self.playback.is_playing_forward(),
                        self.playback.is_looped(),
                        self.playback.discard_past,
                    );
                    match result {
                        StagingResult::Initialized => {
                            self.frame.rendering_new_sim_state_now();
                            self.frame.set_time_increment(self.instances.get_time_inc());
                            frame_new = true;
                        }
                        StagingResult::SteppedInTime {
                            direction,
                            discarded,
                        } => {
                            self.frame.rendering_new_sim_state_now();
                            if !self.rendering.as_ref().is_some_and(|r| r.active) {
                                self.frame.stepped_in_time(direction);
                            }
                            self.frame
                                .count_discarded_time_steps(discarded, self.playback.discard_past);
                            self.frame.set_time_increment(self.instances.get_time_inc());
                            frame_new = true;
                        }
                        StagingResult::SomeTaken { discarded } => {
                            self.frame.rendering_new_sim_state_now();
                            self.frame
                                .count_discarded_time_steps(discarded, self.playback.discard_past);
                            self.frame.set_time_increment(self.instances.get_time_inc());
                            frame_new = true;
                        }
                        StagingResult::StoppedAtLoopEndWithSomeTaken { discarded } => {
                            self.frame.rendering_new_sim_state_now();
                            self.frame
                                .count_discarded_time_steps(discarded, self.playback.discard_past);
                            if !self.sim_settings.discard_past
                                || !self.sim_settings.wait_for_timesteps
                            {
                                self.playback.pause();
                            }
                            self.frame.set_time_increment(self.instances.get_time_inc());
                            frame_new = true;
                        }
                        StagingResult::StoppedAtLoopEndWithNoneTaken => {
                            if !self.sim_settings.discard_past
                                || !self.sim_settings.wait_for_timesteps
                            {
                                self.playback.pause();
                            }
                            self.frame.rendering_new_sim_state_now();
                        }
                        StagingResult::NoneTaken | StagingResult::NothingToStage => {
                            if !self.playback.is_playing() {
                                self.frame.rendering_new_sim_state_now();
                            }
                        }
                        StagingResult::Uninitialized => {}
                    }
                }

                // 5. Rebuild scene if new frame
                if frame_new {
                    self.rebuild_scene();

                    // In rendering mode: request screenshot for every new frame
                    if let Some(ref mut rendering) = self.rendering
                        && let Some(ts_info) = self.instances.get_current_time_step_info()
                        && !rendering.finished_once
                    {
                        if !rendering.active && self.playback.is_playing_forward() {
                            let should_activate = match rendering.start_time {
                                Some(start) => ts_info.measurement.time >= start,
                                None => true, // no start_time → start immediately
                            };
                            if should_activate {
                                rendering.active = true;
                                rendering.frame_counter = 0;
                                self.inspector.info.rendering_status.advance_to_next_state();
                                // deactivate looping
                                self.playback.play_looped = false;
                                info!("Rendering mode activated at t={}", ts_info.measurement.time);
                            }
                        }
                        if rendering.active && !rendering.awaiting_capture {
                            self.simulation_page.request_screenshot(ScreenshotRequest {
                                target: ScreenshotTarget::RenderingFrame {
                                    frame_index: rendering.frame_counter,
                                    output_dir: PathBuf::default(),
                                },
                            });
                            rendering.awaiting_capture = true;
                        }
                        if let Some(finish_time) = rendering.finish_time
                            && finish_time < ts_info.measurement.time
                            && rendering.active
                        {
                            rendering.active = false;
                            rendering.finished_once = true;
                            self.inspector.info.rendering_status.advance_to_next_state();
                            self.playback.pause();
                            info!(
                                "Rendering complete: {} frames captured",
                                rendering.frame_counter
                            );
                            if self.exit_when_finished {
                                let _ = self.to_worker.send(WorkerCommand::Stop);
                                if let Some(id) = self.core.main_window_id() {
                                    task = Some(cosmic::iced::window::close(id));
                                }
                            }
                        }
                    }
                }

                self.inspector.update_info(
                    self.instances.get_current_time_step_info(),
                    self.instances.remaining_buffer_len(),
                );

                // 6. Request more timesteps when discarding past
                let discarded = self.frame.get_and_reset_time_steps_discarded();
                if discarded > 0 {
                    let _ = self
                        .to_worker
                        .send(WorkerCommand::AddTimeStepsToCompute(discarded));
                }

                // Return task if one was produced (e.g. window::close)
                if let Some(t) = task {
                    return t;
                }
            }
            Message::SetFluidVisualization(idx) => {
                if let Some(&opt) = FluidVisOption::ALL.get(idx)
                    && self.sim_settings.fluid_vis != opt
                {
                    self.sim_settings.fluid_vis = opt;
                    return self.request_action(PendingAction::ReloadWithCurrentVisualization);
                }
            }
            Message::SetFluidQuantity(idx) => {
                if let Some(&q) = QuantityOption::ALL.get(idx)
                    && self.sim_settings.fluid_quantity != q
                {
                    self.sim_settings.fluid_quantity = q;
                    return self.request_action(PendingAction::ReloadWithCurrentVisualization);
                }
            }
            Message::SensorPlaneInput(field, value) => {
                let cfg = &mut self.sim_settings.sensor_plane;
                match field {
                    SensorField::Min(i) => cfg.min[i] = value,
                    SensorField::Max(i) => cfg.max[i] = value,
                    SensorField::Dx => cfg.dx = value,
                }
            }
            Message::SensorPlaneStep(field, inc) => {
                self.step_sensor_field(field, inc);
            }
            Message::ApplySensorPlaneConfig => {
                if self.sim_settings.sensor_plane.changed() {
                    self.sim_settings.sensor_plane.clamp_min_max();
                    self.sim_settings.sensor_plane.min_prev =
                        self.sim_settings.sensor_plane.min.clone();
                    self.sim_settings.sensor_plane.max_prev =
                        self.sim_settings.sensor_plane.max.clone();
                    self.sim_settings.sensor_plane.dx_prev =
                        self.sim_settings.sensor_plane.dx.clone();
                    return self.request_action(PendingAction::ReloadWithCurrentVisualization);
                }
            }
            Message::SetColormap(idx) => {
                if let Some(&cmap) = Colormap::ALL.get(idx) {
                    self.sim_settings.colormap = cmap;
                    self.rebuild_scene();
                }
            }
            Message::ColorMappingMaxInput(s) => {
                self.sim_settings.color_mapping_max_input = s;
            }
            Message::ColorMappingMaxStep(step, increment) => {
                if increment {
                    self.sim_settings.color_mapping_max += step;
                } else {
                    self.sim_settings.color_mapping_max -= step;
                }
                self.sim_settings.color_mapping_max_input =
                    self.sim_settings.color_mapping_max.to_string();
                self.rebuild_scene();
            }
            Message::ApplyColorMappingMax => {
                if let Ok(v) = self
                    .sim_settings
                    .color_mapping_max_input
                    .trim()
                    .parse::<f32>()
                {
                    // strikt positiv, 0 ist Untergrenze → max muss > 0 sein
                    self.sim_settings.color_mapping_max = v.max(1e-6);
                    self.sim_settings.color_mapping_max_input =
                        self.sim_settings.color_mapping_max.to_string();
                    self.rebuild_scene();
                } else {
                    // reset if parse fails
                    self.sim_settings.color_mapping_max_input =
                        self.sim_settings.color_mapping_max.to_string();
                }
            }
            Message::SetBoundaryVisualization(idx) => {
                if let Some(&opt) = BoundaryVisOption::ALL.get(idx)
                    && self.sim_settings.boundary_vis != opt
                {
                    self.sim_settings.boundary_vis = opt;
                    return self.request_action(PendingAction::ReloadWithCurrentVisualization);
                }
            }
            Message::ToggleHideBoundary => {
                self.sim_settings.boundary_hidden = !self.sim_settings.boundary_hidden;
                self.rebuild_scene();
            }
            Message::SetBoundaryAlpha(a) => {
                self.sim_settings.boundary_alpha = a;
                self.rebuild_scene();
            }
            Message::ToggleCutX => {
                self.sim_settings.cut.x_active = !self.sim_settings.cut.x_active;
                if self.sim_settings.fluid_vis == FluidVisOption::SensorPlane {
                    return self.request_action(PendingAction::ReloadWithCurrentVisualization);
                } else {
                    self.rebuild_scene();
                }
            }
            Message::ToggleCutY => {
                self.sim_settings.cut.y_active = !self.sim_settings.cut.y_active;
                if self.sim_settings.fluid_vis == FluidVisOption::SensorPlane {
                    return self.request_action(PendingAction::ReloadWithCurrentVisualization);
                } else {
                    self.rebuild_scene();
                }
            }
            Message::ToggleCutZ => {
                self.sim_settings.cut.z_active = !self.sim_settings.cut.z_active;
                if self.sim_settings.fluid_vis == FluidVisOption::SensorPlane {
                    return self.request_action(PendingAction::ReloadWithCurrentVisualization);
                } else {
                    self.rebuild_scene();
                }
            }
            Message::FlipCutX => {
                self.sim_settings.cut.x_flip();
                self.rebuild_scene();
            }
            Message::FlipCutY => {
                self.sim_settings.cut.y_flip();
                self.rebuild_scene();
            }
            Message::FlipCutZ => {
                self.sim_settings.cut.z_flip();
                self.rebuild_scene();
            }
            Message::CutXBoundChanged(delta) => {
                self.sim_settings.cut.x_bound += delta;
                self.sim_settings.cut_x_input = format!("{:.1}", self.sim_settings.cut.x_bound);
                if self.sim_settings.fluid_vis == FluidVisOption::SensorPlane {
                    return self.request_action(PendingAction::ReloadWithCurrentVisualization);
                }
                self.rebuild_scene();
            }
            Message::CutYBoundChanged(delta) => {
                self.sim_settings.cut.y_bound += delta;
                self.sim_settings.cut_y_input = format!("{:.1}", self.sim_settings.cut.y_bound);
                if self.sim_settings.fluid_vis == FluidVisOption::SensorPlane {
                    return self.request_action(PendingAction::ReloadWithCurrentVisualization);
                }
                self.rebuild_scene();
            }
            Message::CutZBoundChanged(delta) => {
                self.sim_settings.cut.z_bound += delta;
                self.sim_settings.cut_z_input = format!("{:.1}", self.sim_settings.cut.z_bound);
                if self.sim_settings.fluid_vis == FluidVisOption::SensorPlane {
                    return self.request_action(PendingAction::ReloadWithCurrentVisualization);
                }
                self.rebuild_scene();
            }
            Message::CutXBoundInput(value) => {
                self.sim_settings.cut_x_input = value.clone();
                if let Ok(v) = value.parse::<f32>() {
                    self.sim_settings.cut.x_bound = v;
                    if self.sim_settings.fluid_vis == FluidVisOption::SensorPlane {
                        return self.request_action(PendingAction::ReloadWithCurrentVisualization);
                    }
                    self.rebuild_scene();
                }
            }
            Message::CutYBoundInput(value) => {
                self.sim_settings.cut_y_input = value.clone();
                if let Ok(v) = value.parse::<f32>() {
                    self.sim_settings.cut.y_bound = v;
                    if self.sim_settings.fluid_vis == FluidVisOption::SensorPlane {
                        return self.request_action(PendingAction::ReloadWithCurrentVisualization);
                    }
                    self.rebuild_scene();
                }
            }
            Message::CutZBoundInput(value) => {
                self.sim_settings.cut_z_input = value.clone();
                if let Ok(v) = value.parse::<f32>() {
                    self.sim_settings.cut.z_bound = v;
                    if self.sim_settings.fluid_vis == FluidVisOption::SensorPlane {
                        return self.request_action(PendingAction::ReloadWithCurrentVisualization);
                    }
                    self.rebuild_scene();
                }
            }
            Message::ToggleCutBoundary => {
                self.sim_settings.cut_boundary = !self.sim_settings.cut_boundary;
                self.rebuild_scene();
            }

            Message::ToggleDiscardPast => {
                self.sim_settings.discard_past = !self.sim_settings.discard_past;
                self.playback.discard_past = self.sim_settings.discard_past;
                if self.sim_settings.discard_past {
                    self.playback.direction = playback::PlaybackDirection::Forward;
                    self.playback.play_looped = false;
                } else {
                    self.playback.direction = if self.sim_settings.invert_time {
                        playback::PlaybackDirection::Backward
                    } else {
                        playback::PlaybackDirection::Forward
                    };
                    self.playback.play_looped = self.sim_settings.play_looped;
                }
            }
            Message::DiscardNow => {
                let discarded = self.instances.discard_past();
                self.frame.count_discarded_time_steps(discarded, true);
            }
            Message::ToggleLoop => {
                if self.rendering.as_ref().is_none_or(|r| !r.active) {
                    self.sim_settings.play_looped = !self.sim_settings.play_looped;
                    self.playback.play_looped = self.sim_settings.play_looped;
                }
            }
            Message::ToggleInvertTime => {
                if self.rendering.as_ref().is_none_or(|r| !r.active) {
                    self.sim_settings.invert_time = !self.sim_settings.invert_time;
                    self.playback.direction = if self.sim_settings.invert_time {
                        playback::PlaybackDirection::Backward
                    } else {
                        playback::PlaybackDirection::Forward
                    };
                }
            }
        }
        Task::none()
    }

    /// Called when a nav item is selected.
    fn on_nav_select(&mut self, id: nav_bar::Id) -> Task<cosmic::Action<Self::Message>> {
        // Activate the page in the model.
        self.nav.activate(id);

        self.update_title()
    }

    fn dialog(&self) -> Option<Element<Self::Message>> {
        let dialog_page = self.dialog_page.as_ref()?;

        // Hilfs-Closure, damit's kompakt bleibt.
        let args = |what: &str| {
            let mut m = HashMap::new();
            m.insert("what", what.to_string());
            m
        };

        let what = if (self.inspector.info.is_measurement_saved || self.inspector.info.is_recorded)
            && self.inspector.info.is_rendered_to_file
        {
            fl!("dialog", "what-recording-and-rendering")
        } else if self.inspector.info.is_measurement_saved || self.inspector.info.is_recorded {
            fl!("dialog", "what-recording")
        } else {
            fl!("dialog", "what-rendering")
        };

        let (title, body, action) = match dialog_page {
            PendingAction::Close => (
                fl!("dialog", "title-stop", args(&what)),
                fl!("dialog", "body-close", args(&what)),
                fl!("dialog", "stop-close"),
            ),
            PendingAction::Reload => (
                fl!("dialog", "title-stop", args(&what)),
                fl!("dialog", "body-reload", args(&what)),
                fl!("dialog", "stop-continue"),
            ),
            PendingAction::ReloadWithCurrentVisualization => (
                fl!("dialog", "title-stop", args(&what)),
                fl!("dialog", "body-visualization", args(&what)),
                fl!("dialog", "stop-continue"),
            ),
        };

        let dialog = widget::dialog()
            .title(title)
            .body(body)
            .primary_action(widget::button::destructive(action).on_press(Message::DialogConfirm))
            .secondary_action(
                widget::button::standard(fl!("dialog", "cancel")).on_press(Message::DialogCancel),
            );

        Some(dialog.into())
    }
}

impl AppModel {
    fn handle_worker_message(
        &mut self,
        msg: WorkerMessage,
    ) -> Option<Task<cosmic::Action<Message>>> {
        match msg {
            WorkerMessage::TimeStepReady(ts_info) => {
                // Check if boundary visualization option request is met
                if self.sim_settings.boundary_vis
                    != BoundaryVisOption::from_template(&ts_info.boundary)
                {
                    return Some(
                        self.request_action(PendingAction::ReloadWithCurrentVisualization),
                    );
                }
                // Accumulate measurement
                self.plotting_measurement_series
                    .push_back(ts_info.measurement.clone());

                match self.instances.insert(*ts_info) {
                    InsertionResult::TooOld => {
                        self.frame.count_discarded_time_steps(1, true);
                    }
                    InsertionResult::ReplacedCurrent => {
                        self.rebuild_scene();
                    }
                    _ => {}
                }
                None
            }
            WorkerMessage::SimulationLoaded(sim_info) => {
                self.simulation_page
                    .light
                    .set_position(sim_info.light_position);
                self.simulation_page.reset_camera();
                self.instances = InstanceStore::default();
                self.instances
                    .set_length_limit(sim_info.buffer_length_limit);
                self.frame = FrameControl::default();

                let boundary_viz_option = self
                    .sim_settings
                    .update_boundary_viz_option(sim_info.explicitly_sampled_boundary);
                if self.sim_settings.boundary_vis != boundary_viz_option {
                    self.sim_settings.boundary_vis = boundary_viz_option;
                }

                // Request buffer fill
                let _ = self.to_worker.send(WorkerCommand::AddTimeStepsToCompute(
                    sim_info.buffer_length_limit,
                ));
                None
            }
            WorkerMessage::FinishedReloading(sim_info) => {
                self.instances.reset(true);
                self.instances
                    .set_length_limit(sim_info.buffer_length_limit);
                self.frame.reset();

                let boundary_viz_option = self
                    .sim_settings
                    .update_boundary_viz_option(sim_info.explicitly_sampled_boundary);
                if self.sim_settings.boundary_vis != boundary_viz_option {
                    self.sim_settings.boundary_vis = boundary_viz_option;
                }

                let _ = self.to_worker.send(WorkerCommand::AddTimeStepsToCompute(
                    sim_info.buffer_length_limit,
                ));
                None
            }
            WorkerMessage::ContinuedFromCheckpoint => {
                let _ = self.to_worker.send(WorkerCommand::AddTimeStepsToCompute(
                    self.instances.buffer_length_limit(),
                ));
                None
            }
            WorkerMessage::ReachedStartTime => {
                self.inspector.info.recording_status.advance_to_next_state();
                None
            }
            WorkerMessage::ReachedFinishTime => {
                self.inspector.info.recording_status.advance_to_next_state();
                let rendering_active = self.rendering.as_ref().is_some_and(|r| r.active);
                if !rendering_active {
                    self.playback.pause();
                }
                if self.exit_when_finished && !rendering_active {
                    let _ = self.to_worker.send(WorkerCommand::Stop);
                    return Some(Task::batch([
                        cosmic::iced::window::close(self.core.main_window_id().unwrap()),
                        cosmic::iced::exit(),
                    ]));
                }
                None
            }
            WorkerMessage::SavedState => {
                // Show notification or update status
                None
            }
            WorkerMessage::SavedMeasurement => {
                // Show notification or update status
                None
            }
            WorkerMessage::Error(e) => {
                error!("Backend error: {e}");
                // Could show in inspector logs tab
                None
            }
        }
    }

    fn request_action(&mut self, action: PendingAction) -> Task<cosmic::Action<Message>> {
        if self.is_recording() || self.is_rendering() {
            tracing::info!("recording");
            self.dialog_page = Some(action);
            Task::none()
        } else {
            tracing::info!("not recording");
            return self.perform(action);
        }
        // }
        Task::none()
    }

    fn is_recording(&self) -> bool {
        // "in progress" = started but not finished. Adjust to your real variants.
        !matches!(
            self.inspector.info.recording_status,
            RecordingStatus::None | RecordingStatus::NotStarted
        ) && !self.inspector.info.recording_status.is_finished()
    }

    fn is_rendering(&self) -> bool {
        // "in progress" = started but not finished. Adjust to your real variants.
        !matches!(
            self.inspector.info.rendering_status,
            RecordingStatus::None | RecordingStatus::NotStarted | RecordingStatus::Finished
        )
    }

    fn perform(&mut self, action: PendingAction) -> Task<cosmic::Action<Message>> {
        match action {
            PendingAction::Close => Task::batch([
                cosmic::iced::window::close(self.core.main_window_id().unwrap()),
                cosmic::iced::exit(),
            ]),
            PendingAction::ReloadWithCurrentVisualization => {
                self.reload_with_current_visualization()
            }
            PendingAction::Reload => self.reload(),
        }
    }

    fn reload(&mut self) -> Task<cosmic::Action<Message>> {
        let _ = self.to_worker.send(WorkerCommand::Reload);
        Task::none()
    }

    fn reload_with_current_visualization(&mut self) -> Task<cosmic::Action<Message>> {
        // Stop rendering
        if let Some(rendering) = &self.rendering
            && rendering.active
        {
            warn!("Rendering interrupted");
        }
        self.rendering = None;
        // discard future instances
        self.instances.discard_future();
        // continue from first time step in instance buffer
        let with_info = TimeStepInfo {
            time_step_number: self
                .instances
                .get_first_time_step_info()
                .map(|i| i.time_step_number)
                .unwrap_or(0),
            measurement: self.render_template.measurement.clone(),
            fluid: self.sim_settings.build_fluid_template(),
            boundary: self.sim_settings.boundary_vis.to_template(),
        };
        self.render_template = with_info.clone();
        let _ = self.to_worker.send(WorkerCommand::ContinueFromTimeStep {
            with_info: Box::new(with_info),
        });
        Task::none()
    }

    fn step_sensor_field(&mut self, field: SensorField, inc: bool) {
        let cfg = &mut self.sim_settings.sensor_plane;
        let step = match field {
            SensorField::Dx => cfg.dx_step,
            _ => cfg.step,
        };
        let buf = match field {
            SensorField::Min(i) => &mut cfg.min[i],
            SensorField::Max(i) => &mut cfg.max[i],
            SensorField::Dx => &mut cfg.dx,
        };
        let mut v: f32 = buf.parse().unwrap_or(0.);
        v += if inc { step } else { -step };
        if matches!(field, SensorField::Dx) {
            v = v.max(step);
        }
        *buf = format!("{v:.3}");
    }

    /// Updates the header and window titles.
    pub fn update_title(&mut self) -> Task<cosmic::Action<Message>> {
        let mut window_title = fl!("app-title");

        if let Some(page) = self.nav.text(self.nav.active()) {
            window_title.push_str(" — ");
            window_title.push_str(page);
        }

        if let Some(id) = self.core.main_window_id() {
            self.set_window_title(window_title, id)
        } else {
            Task::none()
        }
    }
    fn top_bar(&self, page: Page) -> cosmic::Element<'_, Message> {
        let spacing = theme::active().cosmic().spacing;
        let simulation_present = self.instances.is_active();

        // Localized tooltip strings — kept alive for the borrows below.
        let s_reload = fl!("tooltip", "reload-simulation");
        let s_camera = fl!("tooltip", "reset-camera");
        let s_screen = fl!("tooltip", "screenshot");
        let s_state = fl!("tooltip", "save-state");
        let s_plot = fl!("tooltip", "save-plot");
        let s_back = fl!("tooltip", "step-back");
        let s_fwd = fl!("tooltip", "step-forward");
        let s_pause = fl!("tooltip", "pause");
        let s_play = fl!("tooltip", "play");
        let s_inspect = fl!("tooltip", "inspector");
        let s_settings = fl!("tooltip", "settings");

        let left = match page {
            Page::Simulation => row![
                icon_button(
                    "view-refresh-symbolic",
                    s_reload,
                    Message::Reload,
                    simulation_present,
                ),
                icon_button(
                    "view-restore-symbolic",
                    s_camera,
                    Message::ResetCamera,
                    simulation_present,
                ),
                icon_button(
                    "camera-photo-symbolic",
                    s_screen,
                    Message::TakeScreenshot,
                    simulation_present && !self.rendering.as_ref().is_some_and(|s| s.active),
                ),
                icon_button(
                    "document-save-symbolic",
                    s_state,
                    Message::SaveCurrentState,
                    simulation_present,
                ),
            ]
            .spacing(spacing.space_xxs),
            Page::Measurements => row![
                icon_button(
                    "view-refresh-symbolic",
                    s_reload,
                    Message::Reload,
                    simulation_present,
                ),
                icon_button(
                    "document-save-symbolic",
                    s_plot,
                    Message::TakeScreenshot,
                    simulation_present,
                ),
            ]
            .spacing(spacing.space_xxs),
        };

        let play_pause_icon = if self.playback.is_playing() {
            "media-playback-pause-symbolic"
        } else {
            "media-playback-start-symbolic"
        };
        let play_pause_label = if self.playback.is_playing() {
            s_pause
        } else {
            s_play
        };
        let play_pause_message = if self.playback.is_playing() {
            Message::Pause
        } else {
            Message::Play
        };

        let center = row![
            icon_button(
                "media-skip-backward-symbolic",
                s_back,
                Message::StepBackward,
                simulation_present
                    && !self.playback.is_playing()
                    && !self.rendering.as_ref().is_some_and(|s| s.active),
            ),
            icon_button(
                play_pause_icon,
                play_pause_label,
                play_pause_message,
                simulation_present,
            ),
            icon_button(
                "media-skip-forward-symbolic",
                s_fwd,
                Message::StepForward,
                simulation_present && !self.playback.is_playing(),
            ),
        ]
        .spacing(spacing.space_xxs);

        let context_drawer = match page {
            Page::Simulation => ContextPage::SimulationSettings,
            Page::Measurements => ContextPage::PlottingSettings,
        };
        let right = row![
            icon_button(
                "dialog-information-symbolic",
                s_inspect,
                Message::ToggleInspector,
                true,
            ),
            icon_button(
                "emblem-system-symbolic",
                s_settings,
                Message::ToggleContextPage(context_drawer),
                true,
            ),
        ];

        row![
            left,
            widget::space::horizontal(),
            center,
            widget::space::horizontal(),
            right,
        ]
        .align_y(Alignment::Center)
        .padding([spacing.space_xxs, spacing.space_none])
        .spacing(spacing.space_xs)
        .width(Length::Fill)
        .into()
    }

    fn rebuild_scene(&mut self) {
        if let Some(info) = self.instances.get_current_time_step_info() {
            self.sim_settings
                .set_radius(0.5 * (info.measurement.rest_density_grid_spacing as f32));
            let scene = build_scene_data(
                info,
                &self.sim_settings.cut,
                self.sim_settings.cut_boundary,
                self.sim_settings.boundary_hidden,
                self.sim_settings.boundary_alpha,
                self.sim_settings.particle_radius,
                self.sim_settings.color_mapping_max,
                self.sim_settings.colormap,
            );
            self.simulation_page.set_scene(scene);
        }
    }
}

impl Drop for AppModel {
    fn drop(&mut self) {
        if self
            .rendering
            .as_ref()
            .is_some_and(|r| r.finish_time.is_some())
            && !matches!(self.inspector.info.recording_status, RecordingStatus::None)
            && !matches!(self.inspector.info.rendering_status, RecordingStatus::None)
            && (!self.inspector.info.recording_status.is_finished()
                || !self.inspector.info.rendering_status.is_finished())
        {
            warn!("Finish time was not reached!");
        }
        // Stop worker
        let _ = self.to_worker.send(WorkerCommand::Stop);
        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }
    }
}

/// Helper function: create icon button with tooltip
fn icon_button<'a>(
    icon_name: &'a str,
    tooltip: impl Into<Cow<'static, str>>,
    message: Message,
    is_active: bool,
) -> cosmic::Element<'a, Message> {
    let spacing = theme::active().cosmic().spacing;
    let mut btn = widget::button::icon(widget::icon::from_name(icon_name).size(spacing.space_m))
        .icon_size(spacing.space_m)
        .padding(spacing.space_xs);

    if is_active {
        btn = btn.on_press(message);
    }

    let tooltip: Cow<'static, str> = tooltip.into();

    widget::tooltip(
        btn,
        widget::text(tooltip.into_owned()),
        widget::tooltip::Position::Bottom,
    )
    .into()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MenuAction {
    About,
}

impl menu::action::MenuAction for MenuAction {
    type Message = Message;

    fn message(&self) -> Self::Message {
        match self {
            MenuAction::About => Message::ToggleContextPage(ContextPage::About),
        }
    }
}

#[derive(Clone, Debug)]
pub enum PendingAction {
    Close,
    ReloadWithCurrentVisualization,
    Reload,
}

/// Tracks the state of CLI rendering mode
#[derive(Debug, Clone, Default)]
pub struct RenderingState {
    /// Whether rendering mode is active (frames are being captured)
    pub active: bool,
    /// Current frame counter
    pub frame_counter: usize,
    /// Whether we're waiting for the current frame's screenshot to complete
    pub awaiting_capture: bool,
    /// Start time
    pub start_time: Option<f64>,
    /// Finish time
    pub finish_time: Option<f64>,
    pub finished_once: bool,
}

impl RenderingState {
    fn new(start_time: Option<f64>, finish_time: Option<f64>) -> Self {
        Self {
            start_time,
            finish_time,
            ..Default::default()
        }
    }
}
