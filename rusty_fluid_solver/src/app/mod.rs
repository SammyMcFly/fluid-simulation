//! Main application
//!
//!
//! Is based on wgpu and winit.
use crossbeam::{channel, channel::Sender};
use iced_winit::winit;
use iced_winit::winit::event::{DeviceEvent, WindowEvent};
use iced_winit::winit::event_loop::EventLoop;

use rendering_lib::ui::controls::cut::Cut;
use simulation_lib::measurement::Measurement;
use simulation_lib::render_info::*;
use tracing::{error, info}; // error, trace, warn, debug, info,

use rendering_lib::AppState;

mod backend;
pub mod messages;
pub mod rendering;

use backend::{commands::WorkerCommand, worker_loop};
use messages::WorkerMessage;
use rendering::Simulator;

const DISCARD_PAST: bool = true;
const WAIT_FOR_TIMESTEPS: bool = true;

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
        let (to_worker, from_ui) = channel::unbounded::<WorkerCommand>();
        let event_loop_proxy = event_loop.create_proxy();

        // run backend
        let handle = Some(std::thread::spawn(move || {
            worker_loop(from_ui, event_loop_proxy);
        }));

        // send commands to backend depending on user input (args)
        to_worker
            .send(WorkerCommand::Simulate {
                params_file_path: args.params.clone(),
                scene_file_path: args.scene.clone(),
                state_file_path: args.state.clone(),
                measurement_file_path: args.measurement_file.clone(),
                start_time: args.start_time,
                finish_time: args.finish_time,
                recording_file: args.recording_file.clone(),
                with_info: Box::new(TimeStepInfo {
                    measurement: Measurement::default(),
                    fluid: FluidVisualization::SensorPlane {
                        planes: Cut {
                            x_active: false,
                            x_bound: 0.0,
                            x_inverse: false,
                            x_inv: 0.0,
                            y_active: true,
                            y_bound: 15.0,
                            y_inverse: false,
                            y_inv: 0.0,
                            z_active: false,
                            z_bound: 0.0,
                            z_inverse: false,
                            z_inv: 0.0,
                        }
                        .sensor_plane_samples(
                            0.1,
                            [-10.0, 0.0, -15.0],
                            [30.0, 20.0, 20.0],
                            &ScalarQuantity::PressureGraded(vec![]),
                        ),
                    },
                    // fluid: FluidVisualization::Samples {
                    //     positions: Vec::new(),
                    //     coloring: FluidColoring::QuantityGraded {
                    //         quantity: ScalarQuantity::SpeedGraded(Vec::new()),
                    //     },
                    // },
                    // fluid: FluidVisualization::Samples {
                    //     positions: Vec::new(),
                    //     coloring:  FluidColoring::FluidId { val: vec![], max_id: 0 },
                    // },
                    // fluid: FluidVisualization::TriangleMesh { mesh: RenderMesh::default() },
                    boundary: BoundaryVisualization::Samples {
                        positions: Vec::new(),
                        coloring: BoundarySampleColoring::Uniform,
                    },
                    // boundary: BoundaryVisualization::TriangleMesh { mesh: RenderMesh::default(), coloring: BoundaryMeshColoring::Original },
                }),
            })
            .unwrap();

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
        let window = event_loop
            .create_window(
                winit::window::Window::default_attributes()
                    .with_visible(true)
                    .with_title("Rusty Fluid Solver"),
            )
            .unwrap();

        match AppState::new(
            window,
            self.args.resume,
            self.args.rendering_dir.clone(),
            self.args.start_time,
            self.args.finish_time,
            self.args.measurement_file.clone(),
            DISCARD_PAST,
            WAIT_FOR_TIMESTEPS,
        ) {
            Ok(state) => self.state = Some(state),
            Err(e) => panic!("Failed to load sphere: {}", e),
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
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
                }
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
                }
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

    fn user_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        event: WorkerMessage,
    ) {
        match event {
            WorkerMessage::TimeIncFinished(ts_info) => {
                self.state.as_mut().unwrap().received_content(*ts_info);
            }
            WorkerMessage::SimulationLoaded(sim_info) => {
                self.state
                    .as_mut()
                    .unwrap()
                    .new_simulation(sim_info.clone());
                // tell backend to simulate and return "buffer_length_limit" number
                // of states in time,
                // // minus 1 for the initial state that is sent immediately, anyway,
                // // any will be registered as new state:
                // // it will be dequeued and a replacement will be requested automatically
                self.to_worker
                    .send(WorkerCommand::AddTimeStepsToCompute(
                        sim_info.buffer_length_limit,
                    ))
                    .unwrap();
            }
            WorkerMessage::SavedScreenshot => (),
            WorkerMessage::SavedState => (),
            WorkerMessage::SavedMeasurement => (),
            WorkerMessage::FinishedResetting(sim_info) => {
                if let Some(state) = &mut self.state {
                    state.continue_after_reset(sim_info.clone());
                    // tell backend to simulate and return "buffer_length_limit" number
                    // of states in time,
                    // // minus 1 for the initial state that is sent immediately, anyway,
                    // // any will be registered as new state:
                    // // it will be dequeued and a replacement will be requested automatically
                    self.to_worker
                        .send(WorkerCommand::AddTimeStepsToCompute(
                            sim_info.buffer_length_limit,
                        ))
                        .unwrap();
                }
            }
            WorkerMessage::ReachedStartTime => {
                self.state
                    .as_mut()
                    .unwrap()
                    .ui
                    .advance_to_next_measurement_state();
                self.state
                    .as_mut()
                    .unwrap()
                    .ui
                    .advance_to_next_recording_state();
            }
            WorkerMessage::ReachedFinishTime => {
                self.state
                    .as_mut()
                    .unwrap()
                    .ui
                    .advance_to_next_measurement_state();
                self.state
                    .as_mut()
                    .unwrap()
                    .ui
                    .advance_to_next_recording_state();
                if self.args.exit {
                    event_loop.exit();
                }
            }
            WorkerMessage::Error(e) => {
                error!("Backend error: {e}");
            } // todo: handle/print error in ui
        }
    }

    // show mouse drift
    // is probably more efficient
    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        self.state.as_mut().unwrap().window.request_redraw();
    }

    // fn suspended(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {}

    fn exiting(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(state) = &self.state {
            if let Some(ms) = &state.measurement_series {
                self.to_worker
                    .send(WorkerCommand::SaveMeasurement {
                        measurement_series: ms.clone(),
                    })
                    .unwrap();
            }
        }
        self.to_worker.send(WorkerCommand::Stop).unwrap();
        self.worker_handle
            .take()
            .unwrap()
            .join()
            .expect("Couldn't join simulation thread");
        info!("Terminating frontend!");
    }
}
