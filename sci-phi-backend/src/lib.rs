//! Backend for the Sci-Phi fluid simulation application.
pub mod commands;
pub mod messages;
pub mod recording;

use crate::recording::{FileIoError, save_screenshot_into_directory, save_screenshot_to_file};
use commands::WorkerCommand;
use messages::WorkerMessage;

use crossbeam::channel::{Receiver, Sender};
use simulation_lib::measurement::{MeasurementError, MeasurementSeries, RecordingStatus};
use simulation_lib::render_info::{SimulationParameters, TimeStepInfo};
use simulation_lib::setup::SetupError;
use simulation_lib::setup::input::{ConfigError, ParameterFile, Scene};
use simulation_lib::setup::new_boxed_system3d;
use simulation_lib::sph::{SPHSystem, SystemCheckpoint};
use std::rc::Rc;
use std::time::Duration;

/// Errors that can occur while loading, controlling, or recording a
/// simulation on the worker thread.
#[derive(Debug, thiserror::Error)]
pub enum SimulationError {
    #[error("failed to load configuration: {0}")]
    Config(#[from] ConfigError),

    #[error("failed to construct simulation system: {0}")]
    Setup(#[from] SetupError),

    #[error("failed to open measurement file: {0}")]
    Measurement(#[from] MeasurementError),

    #[error(transparent)]
    Recording(#[from] FileIoError),

    #[error(
        "cannot continue from checkpoint at time step {requested}: only \
         {available} time steps have been computed so far"
    )]
    CheckpointNotReady { requested: u64, available: u64 },
}

/// Struct that does:
/// - holds initial state of a system
/// - develops the system in time
/// - allows resetting the system to the initial state
/// - optionally: memorizes all taken measurements, stores them at the end
struct Simulation {
    // initial_system: sph::System,
    system: Box<dyn SPHSystem>,
    checkpoints: Vec<Rc<SystemCheckpoint>>,
    parameters: SimulationParameters,
    render_preset: TimeStepInfo,
    measurement_series: Option<MeasurementSeries>,
    state_appender: Option<recording::TSInfoAppender>,
    recording_status: RecordingStatus,
    start_time: Option<f64>,
    finish_time: Option<f64>,
    /// Highest `time_step_number` passed to `record()` so far. Persisted
    /// across checkpoint rewinds (never reset by `continue_from_checkpoint`)
    /// so a visualization change only affects the recording at exactly this
    /// one "tip" step:
    /// - steps strictly older than this are skipped entirely — already
    ///   recorded, frozen with whatever visualization was active back then,
    /// - this exact step is overwritten in place with the now-current
    ///   visualization once the rewind catches back up to it,
    /// - anything beyond it is appended normally as a genuinely new step.
    last_recorded_step: Option<u64>,

    state_saver_system: Box<dyn SPHSystem>,
}

impl Simulation {
    const N: u64 = 25;

    /// Try to load simulation and return initial state
    fn load(
        simulation_load_info: &SimulationLoadInfo,
    ) -> Result<(TimeStepInfo, Simulation), SimulationError> {
        let ParameterFile {
            procedures,
            parameters,
        } = ParameterFile::from_file(&simulation_load_info.params_file_path)?;
        let scene = Scene::from_file(&simulation_load_info.scene_file_path)?;

        let initial_system = new_boxed_system3d(
            &procedures,
            &parameters,
            &scene,
            simulation_load_info.state_file_path.as_deref(),
        )?;

        let state_saver_system = dyn_clone::clone_box(&*initial_system);

        let checkpoints = vec![Rc::new(SystemCheckpoint::from_sph_system(&*initial_system))];

        let sim_info = SimulationParameters::new(
            &procedures,
            &parameters,
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
            .map(MeasurementSeries::new)
        {
            Some(Ok(ms)) => Some(ms),
            Some(Err(e)) => {
                tracing::error!("Failed to handle measurement file: {}", e);
                return Err(SimulationError::Measurement(e));
            }
            None => None,
        };
        let state_appender = match simulation_load_info
            .recording_file_path
            .as_deref()
            .map(|file_path| recording::TSInfoAppender::new(file_path, &sim_info))
        {
            Some(Ok(ms)) => Some(ms),
            Some(Err(e)) => {
                tracing::error!("Failed to handle recording file: {}", e);
                return Err(SimulationError::Recording(e));
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
            checkpoints,
            parameters: sim_info,
            render_preset: simulation_load_info.with_info.clone(),
            measurement_series,
            state_appender,
            recording_status,
            start_time: simulation_load_info.start_time,
            finish_time: simulation_load_info.finish_time,
            last_recorded_step: None,
            state_saver_system,
        };
        tracing::info!("Loaded new simulation!");

        Ok((sim.get_time_step_info(), sim))
    }

    fn continue_from_checkpoint(
        &mut self,
        with_info: TimeStepInfo,
    ) -> Result<TimeStepInfo, SimulationError> {
        self.render_preset = with_info.clone();

        if self.system.time_steps_propagated() < with_info.time_step_number {
            tracing::warn!(
                "Could not cater to request of continuing from checkpoint. Checkpoint has not yet been created."
            );
            return Err(SimulationError::CheckpointNotReady {
                requested: with_info.time_step_number,
                available: self.system.time_steps_propagated(),
            });
        }
        let last_checkpoint =
            self.checkpoints[usize::try_from(with_info.time_step_number / Self::N)
                .expect("Value too large for usize")]
            .clone();
        self.system.continue_from_checkpoint(last_checkpoint);
        Ok(self.get_time_step_info())
    }

    fn get_next_time_step(&mut self) -> TimeStepInfo {
        self.system.step_forward_in_time();
        self.get_time_step_info()
    }

    fn get_time_step_info(&mut self) -> TimeStepInfo {
        let time_step_info = TimeStepInfo::from_system(&mut *self.system, &self.render_preset);
        self.record(&time_step_info);

        if time_step_info.time_step_number.is_multiple_of(Self::N) {
            let idx = usize::try_from(time_step_info.time_step_number / Self::N)
                .expect("Value too large for usize");
            if idx == self.checkpoints.len() {
                self.checkpoints
                    .push(Rc::new(SystemCheckpoint::from_sph_system(&*self.system)));
            }
        }
        time_step_info
    }

    fn time(&self) -> f64 {
        self.system.time()
    }

    /// Updates recording status and returns whether the a recording should be taken
    fn record_now(&mut self) -> bool {
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
        if !self.record_now() {
            return;
        }

        let ts = time_step_info.time_step_number;
        let is_new_step = self.last_recorded_step.is_none_or(|last| ts > last);
        // Exactly the one step re-produced right at the point a checkpoint
        // rewind caught back up to where recording had already progressed to
        // — the "tip" of the recording at the moment the visualization was
        // changed. Everything strictly older is left untouched.
        let is_same_step = self.last_recorded_step == Some(ts);

        // Measurement values don't depend on visualization, so the seam step
        // already holds correct data; only genuinely new steps get pushed.
        if is_new_step && let Some(meas) = &mut self.measurement_series {
            meas.push_back(time_step_info.measurement.clone());
        }

        // Binary recording's payload DOES depend on visualization: write for
        // new steps (plain append) and for the seam step (overwrite its
        // stale record in place); skip everything strictly older, which
        // stays frozen with whichever visualization was active when it was
        // originally recorded.
        if (is_new_step || is_same_step)
            && let Some(rec) = &mut self.state_appender
        {
            let _ = rec.append_time_step(time_step_info.clone());
        }

        if is_new_step {
            self.last_recorded_step = Some(ts);
        }
    }

    fn started_recording(&self) -> bool {
        self.recording_status.is_active() || self.recording_status.is_finished()
    }

    fn finished_recording(&self) -> bool {
        self.recording_status.is_finished()
    }

    // fn has_finish_time(&self) -> bool {
    //     self.finish_time.is_some()
    // }

    fn save_measurement(&mut self) -> Result<(), SimulationError> {
        if let Some(meas) = &mut self.measurement_series {
            meas.save()
                .inspect_err(|e| tracing::error!("Failed saving measurement: {}", e))?;
            tracing::info!(
                "Successfully saved measurement: {}",
                meas.get_path().display()
            );
        }
        Ok(())
    }

    fn save_state(
        &mut self,
        time_step_number: u64,
        file_path: &std::path::Path,
    ) -> Result<(), SimulationError> {
        if self.system.time_steps_propagated() < time_step_number {
            tracing::warn!("Cannot save state at time step {time_step_number}: not yet computed.");
            return Err(SimulationError::CheckpointNotReady {
                requested: time_step_number,
                available: self.system.time_steps_propagated(),
            });
        }
        let last_checkpoint = self.checkpoints
            [usize::try_from(time_step_number / Self::N).expect("Value too large for usize")]
        .clone();
        self.state_saver_system
            .continue_from_checkpoint(last_checkpoint);
        let mut step = self.state_saver_system.time_steps_propagated();
        while step < time_step_number {
            self.state_saver_system.step_forward_in_time();
            step += 1;
        }
        let checkpoint = SystemCheckpoint::from_sph_system(&*self.state_saver_system);
        recording::save_system_state(checkpoint.into(), file_path)?;
        Ok(())
    }
}

#[derive(Default, PartialEq, Eq)]
enum ComputationState {
    Computing,
    #[default]
    Paused,
}

#[derive(Debug, Clone)]
struct SimulationLoadInfo {
    params_file_path: String,
    scene_file_path: String,
    state_file_path: Option<String>,
    measurement_file_path: Option<String>,
    start_time: Option<f64>,
    finish_time: Option<f64>,
    recording_file_path: Option<std::path::PathBuf>,
    rendering_dir: Option<std::path::PathBuf>,
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
    ) -> Result<TimeStepInfo, SimulationError> {
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

    /// Return to last checkpoint and continue simulation
    fn continue_from_checkpoint(
        &mut self,
        time_step_info: TimeStepInfo,
    ) -> Result<TimeStepInfo, SimulationError> {
        self.simulation_load_info.as_mut().unwrap().with_info = time_step_info;
        self.timesteps_to_compute = 0;
        match self.simulation.as_mut().unwrap().continue_from_checkpoint(
            self.simulation_load_info
                .as_mut()
                .unwrap()
                .with_info
                .clone(),
        ) {
            Ok(initial_state) => Ok(initial_state),
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

    // fn not_reached_existing_finish_time(&self) -> bool {
    //     if let Some(sim) = &self.simulation
    //         && sim.has_finish_time()
    //         && !sim.finished_recording()
    //     {
    //         true
    //     } else {
    //         false
    //     }
    // }

    fn save_measurement(&mut self) -> Result<(), SimulationError> {
        if let Some(sim) = &mut self.simulation {
            return sim.save_measurement();
        }
        Ok(())
    }

    fn stop(&mut self) -> Result<(), SimulationError> {
        if !self.finish_registered {
            self.save_measurement()?;
        }
        Ok(())
    }
}

/// Function that does:
/// - receives [[WorkerCommand]] from front-end
/// - passes [[WorkerCommand]] to [[SimulationController]]
/// - sends [[WorkerMessage]] back to front-end
pub fn worker_loop(from_ui: Receiver<WorkerCommand>, to_ui: Sender<WorkerMessage>) {
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
                    rendering_dir,
                    with_info,
                } => {
                    match simulation_controller.load_simulation(SimulationLoadInfo {
                        params_file_path,
                        scene_file_path,
                        state_file_path,
                        measurement_file_path,
                        start_time,
                        finish_time,
                        recording_file_path: recording_file,
                        rendering_dir,
                        with_info: *with_info,
                    }) {
                        Ok(initial_state) => {
                            let _ = to_ui.send(WorkerMessage::SimulationLoaded(
                                simulation_controller
                                    .simulation
                                    .as_ref()
                                    .unwrap()
                                    .parameters
                                    .clone(),
                            ));
                            let _ =
                                to_ui.send(WorkerMessage::TimeStepReady(Box::new(initial_state)));
                        }
                        Err(e) => {
                            let _ = to_ui.send(WorkerMessage::Error(e.to_string()));
                        }
                    }
                    simulation_controller.compute();
                }
                WorkerCommand::AddTimeStepsToCompute(num) => {
                    simulation_controller.compute_more_timesteps(num);
                }
                WorkerCommand::SaveState {
                    time_step_number,
                    file_path,
                } => {
                    if let Some(simulation) = &mut simulation_controller.simulation {
                        match simulation.save_state(time_step_number, &file_path) {
                            Ok(_) => {
                                tracing::info!("Successfully saved state: {}", file_path.display());
                                let _ = to_ui.send(WorkerMessage::SavedState);
                            }
                            Err(e) => {
                                tracing::error!("Failed to save state: {}", e);
                                let _ = to_ui.send(WorkerMessage::Error(e.to_string()));
                            }
                        }
                    }
                }
                WorkerCommand::WriteRendering {
                    data,
                    width,
                    height,
                    frame_index,
                    overwrite,
                } => {
                    if let Some(info) = &simulation_controller.simulation_load_info
                        && let Some(rendering_dir) = &info.rendering_dir
                    {
                        if let Err(e) = save_screenshot_into_directory(
                            &data,
                            width,
                            height,
                            frame_index,
                            rendering_dir,
                            overwrite,
                        ) {
                            tracing::error!("Screenshot failed: {e}");
                            let _ =
                                to_ui.send(WorkerMessage::Error(format!("Screenshot failed: {e}")));
                        } else {
                            tracing::info!("Saved screenshot frame {frame_index}");
                        }
                    }
                }
                WorkerCommand::SaveScreenshotToFile {
                    data,
                    width,
                    height,
                    file_path,
                } => {
                    if let Err(e) = save_screenshot_to_file(&data, width, height, &file_path, false)
                    {
                        tracing::error!("Screenshot failed: {e}");
                        let _ = to_ui.send(WorkerMessage::Error(format!("Screenshot failed: {e}")));
                    } else {
                        tracing::info!("Saved manual screenshot frame");
                    }
                }
                WorkerCommand::Reload => {
                    tracing::info!("Reloading simulation!");
                    if simulation_controller.simulation_load_info.is_some() {
                        let load_info = simulation_controller.simulation_load_info.clone().unwrap();
                        match simulation_controller.load_simulation(load_info) {
                            Ok(initial_state) => {
                                let _ = to_ui.send(WorkerMessage::FinishedReloading(
                                    simulation_controller
                                        .simulation
                                        .as_ref()
                                        .unwrap()
                                        .parameters
                                        .clone(),
                                ));
                                let _ = to_ui
                                    .send(WorkerMessage::TimeStepReady(Box::new(initial_state)));
                            }
                            Err(e) => {
                                let _ = to_ui.send(WorkerMessage::Error(e.to_string()));
                            }
                        }
                        simulation_controller.compute();
                    }
                }
                WorkerCommand::ContinueFromTimeStep { with_info } => {
                    tracing::info!("Continuing simulation from closest checkpoint!");
                    if simulation_controller.simulation_load_info.is_some() {
                        match simulation_controller.continue_from_checkpoint(*with_info) {
                            Ok(initial_state) => {
                                let _ = to_ui.send(WorkerMessage::ContinuedFromCheckpoint);
                                let _ = to_ui
                                    .send(WorkerMessage::TimeStepReady(Box::new(initial_state)));
                            }
                            Err(e) => {
                                let _ = to_ui.send(WorkerMessage::Error(e.to_string()));
                            }
                        }
                        simulation_controller.compute();
                    }
                }
                WorkerCommand::Stop => {
                    match simulation_controller.stop() {
                        Ok(_) => {}
                        Err(e) => {
                            let _ = to_ui.send(WorkerMessage::Error(e.to_string()));
                        }
                    }
                    tracing::info!("Stopped backend!");
                    break 'worker;
                }
            }
        }

        if simulation_controller.just_started_recording() {
            tracing::info!("Reached start time");
            let _ = to_ui.send(WorkerMessage::ReachedStartTime);
        }
        if simulation_controller.just_finished_recording() {
            tracing::info!("Reached finish time");
            simulation_controller.pause();
            let _ = to_ui.send(WorkerMessage::ReachedFinishTime);
            let save_message = if let Err(e) = simulation_controller.save_measurement() {
                WorkerMessage::Error(e.to_string())
            } else {
                WorkerMessage::SavedMeasurement
            };
            let _ = to_ui.send(save_message);
        }

        if let Some(res) = simulation_controller.get_next_time_step() {
            let _ = to_ui.send(WorkerMessage::TimeStepReady(Box::new(res)));
        } else {
            std::thread::sleep(Duration::from_millis(16));
        }
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn it_works() {
//         let result = add(2, 2);
//         assert_eq!(result, 4);
//     }
// }
