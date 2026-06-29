/// Backend module
use crossbeam::channel::Receiver;
use iced_wgpu::wgpu;
use iced_winit::winit::event_loop::EventLoopProxy;
use simulation_lib::render_info::{SimulationParameters, TimeStepInfo};
use simulation_lib::setup::input::{Parameters, Procedures, Scene};
use simulation_lib::setup::new_boxed_system3d;
use simulation_lib::sph::SPHSystem;
use std::time::Duration;
use tracing::{error, info, warn}; // debug, error, info, span, trace, warn,

use simulation_lib::measurement::RecordingStatus;
use simulation_lib::*;

use crate::app::messages::WorkerMessage;

pub mod commands;
pub mod recording;

use commands::WorkerCommand;

/// Struct that does:
/// - holds initial state of a system
/// - develops the system in time
/// - allows resetting the system to the initial state
/// - optionally: memorizes all taken measurements, stores them at the end
struct Simulation {
    // initial_system: sph::System3D,
    system: Box<dyn SPHSystem>,
    parameters: SimulationParameters,
    render_preset: TimeStepInfo,
    state_appender: Option<recording::StateAppender>,
    recording_status: RecordingStatus,
    start_time: Option<f64>,
    finish_time: Option<f64>,
}

impl Simulation {
    /// Try to load simulation and return initial state
    fn load(
        simulation_load_info: &SimulationLoadInfo,
    ) -> Result<(TimeStepInfo, Simulation), Box<dyn std::error::Error + Send + Sync>> {
        let procedures = Procedures::from_file(&simulation_load_info.params_file_path)?;
        let params = Parameters::from_file(&simulation_load_info.params_file_path)?;
        let scene = Scene::from_file(&simulation_load_info.scene_file_path)?;

        let initial_system = new_boxed_system3d(
            &procedures,
            &params,
            &scene,
            simulation_load_info.state_file_path.as_deref(),
        )?;

        let sim_info = SimulationParameters::new(
            &params,
            [
                scene.light.position[0] as f32,
                scene.light.position[1] as f32,
                scene.light.position[2] as f32,
            ],
            simulation_load_info.measurement_file_path.is_some(),
            simulation_load_info.recording_file_path.is_some(),
        );
        let measurement_series = match simulation_load_info
            .measurement_file_path
            .as_deref()
            .map(measurement::MeasurementSeries::new)
        {
            Some(Ok(ms)) => Some(ms),
            Some(Err(e)) => {
                error!("Failed to handle measurement file: {}", e);
                return Err(format!("Failed to handle measurement file: {}", e).into());
            }
            None => None,
        };
        let state_appender = match simulation_load_info
            .recording_file_path
            .as_deref()
            .map(|file_path| recording::StateAppender::new(file_path, &sim_info))
        {
            Some(Ok(ms)) => Some(ms),
            Some(Err(e)) => {
                error!("Failed to handle recording file: {}", e);
                return Err(format!("Failed to handle recording file: {}", e).into());
            }
            None => None,
        };
        let recording_status = if measurement_series.is_some() || state_appender.is_some() {
            RecordingStatus::NotStarted
        } else {
            RecordingStatus::None
        };
        let mut sim: Simulation = Self {
            system: initial_system,
            parameters: sim_info,
            render_preset: simulation_load_info.with_info.clone(),
            state_appender,
            recording_status,
            start_time: simulation_load_info.start_time,
            finish_time: simulation_load_info.finish_time,
        };
        info!("Loaded new simulation!");

        Ok((sim.get_time_step_info(), sim))
    }
    fn get_next_time_step(&mut self) -> TimeStepInfo {
        self.system.step_forward_in_time();
        self.get_time_step_info()
    }
    fn get_time_step_info(&mut self) -> TimeStepInfo {
        let time_step_info = TimeStepInfo::from_system(&mut *self.system, &self.render_preset);
        self.record(&time_step_info);
        time_step_info
    }
    fn time(&self) -> f64 {
        self.system.time()
    }
    /// Updates measurement status
    fn get_recording_status(&mut self) -> bool {
        if let RecordingStatus::NotStarted = self.recording_status {
            if let Some(st) = self.start_time
                && self.time() >= st
            {
                self.recording_status.advance_to_next_state();
            } else if self.start_time.is_none() {
                self.recording_status.advance_to_next_state();
            }
        }
        let mut final_recording = false;
        if let RecordingStatus::InProgress = self.recording_status
            && let Some(ft) = self.finish_time
            && self.time() >= ft
        {
            self.recording_status.advance_to_next_state();
            final_recording = true;
        }
        self.recording_status.is_active() || final_recording
    }
    fn record(&mut self, time_step_info: &TimeStepInfo) {
        if self.get_recording_status()
            && let Some(rec) = &self.state_appender
        {
            let _ = rec.append_time_step_info_to_file(time_step_info.clone());
        }
    }
    fn started_recording(&self) -> bool {
        self.recording_status.is_active() || self.recording_status.is_finished()
    }
    fn finished_recording(&self) -> bool {
        self.recording_status.is_finished()
    }
    fn has_finish_time(&self) -> bool {
        self.finish_time.is_some()
    }
    // fn stop(&mut self) -> Result<(), String> {
    //     if let Some(meas) = &mut self.measurement_series {
    //         match meas.save() {
    //             Err(e) => {
    //                 error!("Failed saving measurement: {}", e);
    //                 return Err(format!("Failed saving measurement: {}", e));
    //             }
    //             Ok(_) => {
    //                 info!(
    //                     "Successfully saved measurement: {}",
    //                     meas.get_path().as_path().display()
    //                 );
    //                 return Ok(());
    //             }
    //         }
    //     }
    //     Ok(())
    // }
}

#[derive(Default, PartialEq, Eq)]
enum ComputationState {
    Computing,
    #[default]
    Paused,
}

#[derive(Clone)]
struct SimulationLoadInfo {
    params_file_path: String,
    scene_file_path: String,
    state_file_path: Option<String>,
    measurement_file_path: Option<String>,
    start_time: Option<f64>,
    finish_time: Option<f64>,
    recording_file_path: Option<String>,
    with_info: TimeStepInfo,
}

/// Struct that does:
/// - wraps the [[Simulation]], in case one is loaded
/// - holds the info if new time steps are currently computed or not
#[derive(Default)]
struct SimulationController {
    state: ComputationState,
    timesteps_to_compute: usize,
    start_registered: bool,
    finish_registered: bool,

    simulation_load_info: Option<SimulationLoadInfo>,
    simulation: Option<Simulation>,
}

impl SimulationController {
    /// Try to load simulation and return initial state
    fn load_simulation(
        &mut self,
        simulation_load_info: SimulationLoadInfo,
    ) -> Result<TimeStepInfo, Box<dyn std::error::Error + Send + Sync>> {
        self.timesteps_to_compute = 0;
        self.simulation_load_info = Some(simulation_load_info);
        match Simulation::load(self.simulation_load_info.as_ref().unwrap()) {
            Ok((initial_state, sim)) => {
                self.simulation = Some(sim);
                Ok(initial_state)
            }
            Err(e) => Err(e),
        }
    }
    fn compute_more_timesteps(&mut self, num: usize) {
        self.timesteps_to_compute += num;
    }
    fn compute(&mut self) {
        self.state = ComputationState::Computing;
    }
    fn pause(&mut self) {
        self.state = ComputationState::Paused;
    }
    /// Calculate next time step depending on state
    fn get_next_time_step(&mut self) -> Option<TimeStepInfo> {
        if self.state == ComputationState::Computing && self.timesteps_to_compute > 0 {
            self.simulation.as_mut().map(|sim| {
                self.timesteps_to_compute -= 1;
                sim.get_next_time_step()
            })
        } else {
            None
        }
    }
    fn just_started_recording(&mut self) -> bool {
        if let Some(sim) = &self.simulation
            && sim.started_recording()
            && !self.start_registered
        {
            self.start_registered = true;
            return true;
        }
        false
    }
    fn just_finished_recording(&mut self) -> bool {
        if let Some(sim) = &self.simulation
            && sim.finished_recording()
            && !self.finish_registered
        {
            self.finish_registered = true;
            return true;
        }
        false
    }
    fn not_reached_existing_finish_time(&self) -> bool {
        if let Some(sim) = &self.simulation
            && sim.has_finish_time()
            && !sim.finished_recording()
        {
            true
        } else {
            false
        }
    }
    // fn stop(&mut self) -> Result<(), String> {
    //     if let Some(sim) = &mut self.simulation {
    //         return sim.stop();
    //     }
    //     Ok(())
    // }
}

/// Function that does:
/// - receives [[WorkerCommand]] from front-end
/// - passes [[WorkerCommand]] to [[SimulationController]]
/// - sends [[WorkerMessage]] back to front-end
pub fn worker_loop(from_ui: Receiver<WorkerCommand>, to_ui: EventLoopProxy<WorkerMessage>) {
    let mut simulation_controller = SimulationController::default();
    'worker: loop {
        for cmd in from_ui.try_iter() {
            match cmd {
                WorkerCommand::Simulate {
                    params_file_path,
                    scene_file_path,
                    state_file_path,
                    measurement_file_path,
                    start_time,
                    finish_time,
                    recording_file,
                    with_info,
                } => {
                    // load system
                    match simulation_controller.load_simulation(SimulationLoadInfo {
                        params_file_path,
                        scene_file_path,
                        state_file_path,
                        measurement_file_path,
                        start_time,
                        finish_time,
                        recording_file_path: recording_file,
                        with_info: *with_info,
                    }) {
                        Ok(initial_state) => {
                            let _ = to_ui.send_event(WorkerMessage::SimulationLoaded(
                                simulation_controller
                                    .simulation
                                    .as_ref()
                                    .unwrap()
                                    .parameters
                                    .clone(),
                            ));
                            // send initial state
                            let _ = to_ui.send_event(WorkerMessage::TimeIncFinished(Box::new(
                                initial_state,
                            )));
                        }
                        Err(e) => {
                            let _ = to_ui.send_event(WorkerMessage::Error(e));
                        }
                    }
                    simulation_controller.compute();
                }
                WorkerCommand::AddTimeStepsToCompute(num) => {
                    // debug!("compute: {}", num);
                    simulation_controller.compute_more_timesteps(num);
                }
                WorkerCommand::SaveState { fluid, filepath } => {
                    let save_message = if recording::save_system_state(fluid, &filepath).is_ok() {
                        info!("Successfully saved state: {}", filepath);
                        WorkerMessage::SavedState
                    } else {
                        error!("Failed to save state!");
                        WorkerMessage::Error("Failed to save state!".to_string().into())
                    };
                    let _ = to_ui.send_event(save_message);
                }
                WorkerCommand::SaveScreenshot(rbr) => {
                    let buffer = rbr.buffer.lock().unwrap();
                    let buffer_slice = buffer.buffer.slice(..);
                    let (tx, rx) = crossbeam::channel::bounded::<()>(1);
                    // rbr.buffer.lock().unwrap().mapping_started = true;
                    buffer_slice.map_async(wgpu::MapMode::Read, move |_| {
                        tx.send(()).ok();
                    });
                    // Drive the future to completion
                    rbr.device.poll(wgpu::Maintain::Wait);

                    // Wait for callback
                    rx.recv().unwrap();

                    let data = {
                        let slice = buffer_slice.get_mapped_range();
                        slice.to_vec()
                    };

                    // Free buffer for next use
                    buffer.buffer.unmap();
                    // buffer.mapping_started = false;

                    match recording::save_screenshot(&data, &rbr, &buffer, &rbr.output_dir) {
                        Ok(_) => {
                            let _ = to_ui.send_event(WorkerMessage::SavedScreenshot);
                        }
                        Err(e) => {
                            let _ = to_ui.send_event(WorkerMessage::Error(e.to_string().into()));
                        }
                    }
                }
                WorkerCommand::SaveMeasurement { measurement_series } => {
                    match measurement_series.save() {
                        Ok(_) => {
                            info!(
                                "Successfully saved measurement: {}",
                                measurement_series.get_path().as_path().display()
                            );
                            let _ = to_ui.send_event(WorkerMessage::SavedMeasurement);
                        }
                        Err(e) => {
                            error!("Failed saving measurement: {}", e);
                            let _ = to_ui.send_event(WorkerMessage::Error(e.to_string().into()));
                        }
                    }
                }
                // WorkerCommand::Resume => {
                //     info!("Run simulation!");
                //     simulation_controller.compute();
                // },
                // WorkerCommand::Pause => {
                //     info!("Paused simulation!");
                //     simulation_controller.pause();
                // },
                WorkerCommand::Reset => {
                    info!("Reset simulation!");
                    // reload system
                    if let Some(info) = &simulation_controller.simulation_load_info {
                        match simulation_controller.load_simulation(info.clone()) {
                            Ok(initial_state) => {
                                let _ = to_ui.send_event(WorkerMessage::FinishedResetting(
                                    simulation_controller
                                        .simulation
                                        .as_ref()
                                        .unwrap()
                                        .parameters
                                        .clone(),
                                ));
                                // send initial state
                                let _ = to_ui.send_event(WorkerMessage::TimeIncFinished(Box::new(
                                    initial_state,
                                )));
                            }
                            Err(e) => {
                                let _ = to_ui.send_event(WorkerMessage::Error(e));
                            }
                        }
                        simulation_controller.compute();
                    }
                }
                WorkerCommand::Stop => {
                    if simulation_controller.not_reached_existing_finish_time() {
                        warn!("Finish time was not reached!");
                    }
                    // let save_message = if let Err(e) = simulation_controller.stop() {
                    //     WorkerMessage::Error(e)
                    // } else {
                    //     info!("Successfully saved measurement!");
                    //     WorkerMessage::SavedMeasurement
                    // };
                    // let _ = to_ui.send_event(save_message);
                    info!("Stopped backend!");
                    break 'worker;
                }
            }
        }
        // check if start time is reached
        if simulation_controller.just_started_recording() {
            info!("Reached start time");
            let _ = to_ui.send_event(WorkerMessage::ReachedStartTime);
        }
        // check if finish time is reached
        if simulation_controller.just_finished_recording() {
            info!("Reached finish time");
            simulation_controller.pause();
            let _ = to_ui.send_event(WorkerMessage::ReachedFinishTime);
        }
        // progress simulation
        if let Some(res) = simulation_controller.get_next_time_step() {
            let _ = to_ui.send_event(WorkerMessage::TimeIncFinished(Box::new(res)));
        } else {
            std::thread::sleep(Duration::from_millis(16));
        }
    }
}
