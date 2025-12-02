//! Main application
//!
//!
//! Is based on wgpu and winit.
use crossbeam::{channel, channel::Sender};
use iced_winit::winit;
use iced_winit::winit::event_loop::EventLoop;
use iced_winit::winit::event::{WindowEvent, DeviceEvent};

#[cfg(feature = "logging")]
use tracing::{
    info,
    error,
}; // error, trace, warn, debug, info,

mod backend;
pub mod rendering;
pub mod messages;

use backend::{worker_loop, commands::WorkerCommand};
use rendering::AppState;
use messages::WorkerMessage;


/// Application does:
/// - handles the event loop
/// - passes event to state
pub struct StateApplication {
    state: Option<AppState>,
    worker_handle: Option<std::thread::JoinHandle<()>>,
    to_worker: Sender<WorkerCommand>,
    /// Start simulation with resumed playback
    start_resumed: bool,
    /// Exit upon reaching the simulations finish time
    exit_when_finished: bool,
}

impl StateApplication {
    pub fn new(event_loop: &EventLoop<WorkerMessage>, args: crate::Args) -> Self {
        // init channels connecting simulation backend with graphics/ui front end
        let (to_worker, from_ui) = channel::unbounded::<WorkerCommand>();
        let event_loop_proxy = event_loop.create_proxy();

        // run backend
        let handle = Some(std::thread::spawn(move || {
            worker_loop(from_ui, event_loop_proxy);
        }));

        // send commands to backend depending on user input (args)
        if args.config.is_some() {
            to_worker.send(WorkerCommand::Simulate {
                config: args.config.unwrap(),
                // config: args.config,
                state: args.state.clone(),
                measure: args.measurement_file.clone(),
                start_time: args.start_time,
                finish_time: args.finish_time,
                recording_file: args.recording_file,
            }).unwrap();
        }

        Self {
            state: Option::default(),
            worker_handle: handle,
            to_worker,
            start_resumed: args.resume,
            exit_when_finished: args.exit,
        }
    }
}

impl winit::application::ApplicationHandler<WorkerMessage> for StateApplication {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop.create_window(winit::window::Window::default_attributes()
            .with_visible(true).with_title("Rusty Fluid Solver")).unwrap();

        match AppState::new(window, self.start_resumed) {
            Ok(state) => self.state = Some(state),
            Err(e) => panic!("Failed to load sphere: {}", e),
        }
    }

    fn window_event(
        &mut self,
        event_loop:
        &winit::event_loop::ActiveEventLoop,
        id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let window = self.state.as_ref().unwrap().window();
        // if window.id() == id && !self.state.as_mut().unwrap().input(&event) {
        if window.id() == id {
            match event {
                WindowEvent::CloseRequested => {
                    #[cfg(feature = "logging")]
                    info!("The close button was pressed; stopping...");
                    event_loop.exit();
                },
                WindowEvent::Resized(physical_size) => {
                    self.state.as_mut().unwrap().resize(physical_size);
                }
                WindowEvent::RedrawRequested => {
                    self.state.as_mut().unwrap().update(&self.to_worker);
                    self.state.as_mut().unwrap().render().unwrap();
                    // even smoother with this instead of in function about_to_wait
                    // self.state.as_mut().unwrap().window.request_redraw();
                }
                _ => {
                    self.state.as_mut().unwrap().process_window_event(&event);
                    self.state.as_mut().unwrap().update(&self.to_worker);
                },
            }
        }
        // self.state.as_mut().unwrap().window.request_redraw();
    }

    fn device_event(
            &mut self,
            _event_loop: &winit::event_loop::ActiveEventLoop,
            _device_id: winit::event::DeviceId,
            event: DeviceEvent,
    ) {
        self.state.as_mut().unwrap().process_device_event(&event);
    }

    fn user_event(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, event: WorkerMessage) {
        match event {
            WorkerMessage::TimeIncFinished(ts_info) => {
                self.state.as_mut().unwrap().received_content(ts_info);
            },
            WorkerMessage::SimulationLoaded(sim_info) => {
                self.state.as_mut().unwrap().new_simulation(sim_info.clone());
                // tell backend to simulate and return "buffer_length_limit" number
                // of states in time,
                // minus 1 for the initial state that is sent immediately, anyway,
                // any will be registered as new state:
                // it will be dequeued and a replacement will be requested automatically
                self.to_worker.send(WorkerCommand::AddTimeStepsToCompute(
                    sim_info.buffer_length_limit-1
                )).unwrap();
            },
            WorkerMessage::SavedState => (),
            WorkerMessage::SavedMeasurement => (),
            WorkerMessage::FinishedResetting(sim_info) => {
                if let Some(state) = &mut self.state {
                    state.continue_after_reset(sim_info);
                    // tell backend to simulate and return "buffer_length_limit" number
                    // of states in time,
                    // minus 1 for the initial state that is sent immediately, anyway,
                    // any will be registered as new state:
                    // it will be dequeued and a replacement will be requested automatically
                    self.to_worker.send(WorkerCommand::AddTimeStepsToCompute(
                        state.instances.length_limit-1
                    )).unwrap();
                }
            },
            WorkerMessage::ReachedStartTime => {
                self.state.as_mut().unwrap().ui.advance_to_next_measurement_state();
            },
            WorkerMessage::ReachedFinishTime => {
                self.state.as_mut().unwrap().ui.advance_to_next_measurement_state();
                if self.exit_when_finished {
                    event_loop.exit();
                }
            },
            WorkerMessage::Error(e) => {
                #[cfg(feature = "logging")]
                error!("Backend error: {e}");
            }, // todo: handle/print error in ui
        }
    }

    // show mouse drift
    // is probably more efficient
    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        self.state.as_mut().unwrap().window.request_redraw();
    }

    // fn suspended(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {}

    fn exiting(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        self.to_worker.send(WorkerCommand::Stop).unwrap();
        self.worker_handle.take().unwrap().join().expect("Couldn't join simulation thread");
        #[cfg(feature = "logging")]
        info!("Terminating frontend!");
    }
}







