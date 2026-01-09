//! Main application
//!
//!
//! Is based on wgpu and winit.
use crossbeam::{channel, channel::Sender};
use iced_winit::winit;
use iced_winit::winit::event_loop::EventLoop;
use iced_winit::winit::event::{WindowEvent, DeviceEvent};

use tracing::{
    info,
    error,
}; // error, trace, warn, debug, info,

use rendering_lib::AppState;

mod backend;
pub mod rendering;
pub mod messages;

use backend::{worker_loop, commands::WorkerCommand};
use messages::WorkerMessage;
use rendering::Player;



const DISCARD_PAST: bool = false;
const WAIT_FOR_TIMESTEPS: bool = false;


/// Application does:
/// - handles the event loop
/// - passes event to state
pub struct StateApplication {
    state: Option<AppState>,
    worker_handle: Option<std::thread::JoinHandle<()>>,
    to_worker: Sender<WorkerCommand>,
    /// Command line arguments
    args: crate::Args,
}

impl StateApplication {
    pub fn new(event_loop: &EventLoop<WorkerMessage>, args: crate::Args) -> Self {
        // init channels connecting simulation backend with graphics/ui front end
        let (to_worker, from_ui) = channel::bounded::<WorkerCommand>(15);
        let event_loop_proxy = event_loop.create_proxy();

        // run backend
        let handle = Some(std::thread::spawn(move || {
            worker_loop(from_ui, event_loop_proxy);
        }));

        Self {
            state: Option::default(),
            worker_handle: handle,
            to_worker,
            args,
        }
    }
}

impl winit::application::ApplicationHandler<WorkerMessage> for StateApplication {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop.create_window(winit::window::Window::default_attributes()
            .with_visible(true).with_title("Rusty Fluid Solver")).unwrap();

        if let Some(fp) = &self.args.recording {
            self.to_worker.send(WorkerCommand::ReadRecording(fp.clone())).unwrap();
        }

        match AppState::new(
            window,
            self.args.resume,
            self.args.rendering_dir.clone(),
            self.args.start_time,
            self.args.finish_time,
            DISCARD_PAST,
            WAIT_FOR_TIMESTEPS,
        ) {
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

    fn user_event(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop, event: WorkerMessage) {
        match event {
            WorkerMessage::FinishedReading(sim_info, time_steps) => {
                self.state.as_mut().unwrap().received_content(sim_info, time_steps);
            },
            WorkerMessage::SavedScreenshot => {

            },
            WorkerMessage::SavedState => {

            },
            WorkerMessage::Error(e) => {
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
        info!("Terminating frontend!");
    }
}







