//! Backend module
use std::time::Duration;
use std::io::Write;

use iced_winit::winit::event_loop::EventLoopProxy;
use crossbeam::channel::Receiver;

#[cfg(feature = "logging")]
use tracing::{error, warn, info}; // debug, error, info, span, trace, warn,

use commands::WorkerCommand;
use sph::particle::{SerParticle3D, BoundaryParticle3D};
use crate::app::rendering::ui::controls::ParticleColor;
use crate::app::messages::WorkerMessage;
use measure::MeasurementStatus;

pub mod commands;
pub mod sph;
pub mod setup;
pub mod measure;




/// Store the current state of all fluid particles to a file
fn save_system_state(particles: Vec<SerParticle3D>, file_path: &str) -> std::io::Result<()> {
    let file_path = std::path::Path::new(file_path);
    // convert to global path
    let file_path_parent = std::fs::canonicalize(
        file_path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(std::path::Path::new("."))
    )?;
    // Create the parent directory if it does not exist
    if !file_path_parent.exists() {
        std::fs::create_dir_all(file_path_parent.clone())?;
        #[cfg(feature = "logging")]
        info!("Created directories: {}", file_path_parent.display());
    }
    let global_file_path = file_path_parent.join(file_path.file_name().expect("No final component found."));
    let ron_string = ron::to_string(&particles).unwrap();
    let mut file = std::fs::File::create(global_file_path)?;
    file.write_all(ron_string.as_bytes())?;
    Ok(())
}

#[derive(Debug, Clone)]
pub struct SimulationParameters {
    /// Particle size
    pub particle_diameter: f32,
    /// Rest density
    pub rest_density: f32,
    /// Light position
    pub light_position: [f32; 3],
    /// Particle color
    pub particle_color: ParticleColor,
    /// Boundary particle color
    pub boundary_particle_color: ParticleColor,
    /// Integration Scheme
    pub integration_scheme: sph::PropagationMethod,
    /// maximum buffer length
    pub buffer_length_limit: usize,
    /// Flag that is true if a measurement is taken in simulation, else false
    pub is_measured: bool,
}

#[derive(Debug, Clone, Default)]
pub struct TimeStepInfo {
    // system time
    pub time: f32,
    // time increment
    pub time_increment: f32,
    // average density
    pub average_density: f32,
    // particles
    pub fluid: Vec<SerParticle3D>,
    pub boundary: Vec<BoundaryParticle3D>
}

/// Struct that does:
/// - holds initial state of a system
/// - develops the system in time
/// - allows resetting the system to the initial state
/// - optionally: memorizes all taken measurements, stores them at the end
struct Simulation {
    // initial_system: sph::System3D,
    system: sph::System3D,
    parameters: SimulationParameters,
    measurements: Option<measure::MeasurementSeries>,
    measurement_status: MeasurementStatus,
    start_time: Option<f64>,
    finish_time: Option<f64>,
}

impl Simulation {
    /// Try to load simulation and return initial state
    fn load(
        simulation_load_info: &SimulationLoadInfo,
    ) -> Result<(TimeStepInfo, Simulation), String> {
        match setup::System3DConfigConstructor::new(
            &simulation_load_info.config_file_path,
            simulation_load_info.state_file_path.as_deref(),
            simulation_load_info.measurement_file_path.is_some(),
        ) {
            Ok((sys_conf, sim_info)) => {
                let initial_system = sph::System3D::new(sys_conf.finish());
                let measurement_series = simulation_load_info.measurement_file_path.as_deref().map(measure::MeasurementSeries::new);
                let measurement_status = if measurement_series.is_some() {
                    MeasurementStatus::NotStarted
                } else {
                    MeasurementStatus::None
                };
                let mut sim: Simulation = Self {
                    system: initial_system,
                    parameters: sim_info,
                    measurements: measurement_series,
                    measurement_status,
                    start_time: simulation_load_info.start_time,
                    finish_time: simulation_load_info.finish_time,
                };
                #[cfg(feature = "logging")]
                info!("Loaded new simulation!");
                sim.update_measurement_status();
                if let Some(meas) = &mut sim.measurements && sim.measurement_status.is_active() {
                    sim.system.push_back_measurement(meas);
                }
                Ok((sim.system.get_time_step_info(), sim))
        },
            _ => {
                #[cfg(feature = "logging")]
                error!("Invalid state or scene file!");
                Err("Invalid state or scene file!".to_string())
            },
        }
    }
    fn get_next_time_step(&mut self) -> TimeStepInfo {
        self.system.step_forward_in_time(&self.parameters.integration_scheme);
        self.update_measurement_status();
        if let Some(meas) = &mut self.measurements && self.measurement_status.is_active() {
            self.system.push_back_measurement(meas);
        }
        self.system.get_time_step_info()
    }
    fn time(&self) -> f64 {
        self.system.time()
    }
    /// Updates measurement status
    fn update_measurement_status(&mut self) {
        if let MeasurementStatus::NotStarted = self.measurement_status {
            if let Some(st) = self.start_time && self.time() >= st {
                self.measurement_status.advance_to_next_state();
            } else if self.start_time.is_none() {
                self.measurement_status.advance_to_next_state();
            }
        }
        if let MeasurementStatus::Measuring = self.measurement_status
                && let Some(ft) = self.finish_time && self.time() > ft {
            self.measurement_status.advance_to_next_state();
        }
    }
    fn started_measurement(&self) -> bool {
        self.measurement_status.is_active() || self.measurement_status.is_finished()
    }
    fn finished_measurement(&self) -> bool {
        self.measurement_status.is_finished()
    }
    fn has_finish_time(&self) -> bool {
        self.finish_time.is_some()
    }
    fn stop(&mut self) -> Result<(), String> {
        if let Some(meas) = &mut self.measurements {
            match meas.save() {
                Err(e) => {
                    #[cfg(feature = "logging")]
                    error!("Failed saving measurement: {}", e);
                    return Err(format!("Failed saving measurement: {}", e))
                },
                Ok(_) => {
                    #[cfg(feature = "logging")]
                    info!("Successfully saved measurement: {}", meas.get_path());
                    return Ok(())
                },
            }
        }
        Ok(())
    }
}

#[derive(Default, PartialEq, Eq)]
enum SimulationState {
    Computing,
    #[default]
    Paused,
}

#[derive(Clone)]
struct SimulationLoadInfo {
    config_file_path: String,
    state_file_path: Option<String>,
    measurement_file_path: Option<String>,
    start_time: Option<f64>,
    finish_time: Option<f64>,
}

/// Struct that does:
/// - wraps the [[Simulation]], in case one is loaded
/// - holds the info if new time steps are currently computed or not
#[derive(Default)]
struct SimulationController {
    state: SimulationState,
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
    ) -> Result<TimeStepInfo, String>{
        self.timesteps_to_compute = 0;
        self.simulation_load_info = Some(simulation_load_info);
        match Simulation::load(
            self.simulation_load_info.as_ref().unwrap(),
        ) {
            Ok((initial_state, sim)) => {
                self.simulation = Some(sim);
                Ok(initial_state)
            },
            Err(e) => { Err(e) },
        }
    }
    fn compute_more_timesteps(&mut self, num: usize) {
        self.timesteps_to_compute += num;
    }
    fn compute(&mut self) {
        self.state = SimulationState::Computing;
    }
    fn pause(&mut self) {
        self.state = SimulationState::Paused;
    }
    /// Calculate next time step depending on state
    fn get_next_time_step(&mut self) -> Option<TimeStepInfo> {
        if self.state == SimulationState::Computing && self.timesteps_to_compute > 0 {
            self.simulation.as_mut().map(|sim| {
                self.timesteps_to_compute -= 1;
                sim.get_next_time_step()
            })
        } else {
            None
        }
    }
    fn just_started_measurement(&mut self) -> bool {
        if let Some(sim) = &self.simulation && sim.started_measurement() && !self.start_registered {
            self.start_registered = true;
            return true
        }
        false
    }
    fn just_finished_measurement(&mut self) -> bool {
        if let Some(sim) = &self.simulation && sim.finished_measurement() && !self.finish_registered {
            self.finish_registered = true;
            return true
        }
        false
    }
    fn not_reached_existing_finish_time(&self) -> bool {
        if let Some(sim) = &self.simulation && sim.has_finish_time() && !sim.finished_measurement(){
            true
        } else {
            false
        }
    }
    fn stop(&mut self) -> Result<(), String> {
        if let Some(sim) = &mut self.simulation {
            return sim.stop()
        }
        Ok(())
    }
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
                    config,
                    state,
                    measure,
                    start_time,
                    finish_time,
                } => {
                    // load system
                    match simulation_controller.load_simulation(
                        SimulationLoadInfo {
                            config_file_path: config,
                            state_file_path: state,
                            measurement_file_path: measure,
                            start_time,
                            finish_time,
                        },
                    ) {
                        Ok(initial_state) => {
                            let _ = to_ui.send_event(WorkerMessage::SimulationLoaded(simulation_controller.simulation.as_ref().unwrap().parameters.clone()));
                            // send initial state
                            let _ = to_ui.send_event(WorkerMessage::TimeIncFinished(initial_state));
                        },
                        Err(e) => {
                            let _ = to_ui.send_event(WorkerMessage::Error(e));
                        },
                    }
                    simulation_controller.compute();
                }
                WorkerCommand::AddTimeStepsToCompute(num) => {
                    // #[cfg(feature = "logging")]
                    // debug!("compute: {}", num);
                    simulation_controller.compute_more_timesteps(num);
                },
                WorkerCommand::SaveState { particles, filepath } => {
                    let save_message = if save_system_state(particles, &filepath).is_ok() {
                        #[cfg(feature = "logging")]
                        info!("Successfully saved state: {}", filepath);
                        WorkerMessage::SavedState
                    } else {
                        #[cfg(feature = "logging")]
                        error!("Failed to save state!");
                        WorkerMessage::Error("Failed to save state!".to_string())
                    };
                    let _ = to_ui.send_event(save_message);
                },
                // WorkerCommand::Resume => {
                //     #[cfg(feature = "logging")]
                //     info!("Run simulation!");
                //     simulation_controller.compute();
                // },
                // WorkerCommand::Pause => {
                //     #[cfg(feature = "logging")]
                //     info!("Paused simulation!");
                //     simulation_controller.pause();
                // },
                WorkerCommand::Reset => {
                    #[cfg(feature = "logging")]
                    info!("Reset simulation!");
                    // reload system
                    if let Some(info) = &simulation_controller.simulation_load_info {
                        match simulation_controller.load_simulation(
                            info.clone(),
                        ) {
                            Ok(initial_state) => {
                                let _ = to_ui.send_event(WorkerMessage::FinishedResetting(simulation_controller.simulation.as_ref().unwrap().parameters.clone()));
                                // send initial state
                                let _ = to_ui.send_event(WorkerMessage::TimeIncFinished(initial_state));
                            },
                            Err(e) => {
                                let _ = to_ui.send_event(WorkerMessage::Error(e));
                            },
                        }
                        simulation_controller.compute();
                    }
                },
                WorkerCommand::Stop => {
                    if simulation_controller.not_reached_existing_finish_time() {
                        #[cfg(feature = "logging")]
                        warn!("Finish time was not reached!");
                    }
                    let save_message = if let Err(e) = simulation_controller.stop() {
                        WorkerMessage::Error(e)
                    } else {
                        #[cfg(feature = "logging")]
                        info!("Successfully saved measurement!");
                        WorkerMessage::SavedMeasurement
                    };
                    let _ = to_ui.send_event(save_message);
                    #[cfg(feature = "logging")]
                    info!("Stopped backend!");
                    break 'worker;
                },
            }
        }
        // check if start time is reached
        if simulation_controller.just_started_measurement() {
            #[cfg(feature = "logging")]
            info!("Reached start time");
            let _ = to_ui.send_event(WorkerMessage::ReachedStartTime);
        }
        // check if finish time is reached
        if simulation_controller.just_finished_measurement() {
            #[cfg(feature = "logging")]
            info!("Reached finish time");
            simulation_controller.pause();
            let _ = to_ui.send_event(WorkerMessage::ReachedFinishTime);
        }
        // progress simulation
        if let Some(res) = simulation_controller.get_next_time_step() {
            let _ = to_ui.send_event(WorkerMessage::TimeIncFinished(res));
        } else {
            std::thread::sleep(Duration::from_millis(16));
        }
    }
}