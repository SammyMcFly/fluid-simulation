//! Backend module
use std::time::Duration;
use std::io::Write;

use iced_winit::winit::event_loop::EventLoopProxy;
use crossbeam::channel::Receiver;

#[cfg(feature = "logging")]
use tracing::{error, info, debug}; // debug, error, info, span, trace, warn,

use commands::WorkerCommand;
use sph::particle::{SerParticle3D, BoundaryParticle3D};
use crate::app::rendering::ui::controls::ParticleColor;
use crate::app::messages::WorkerMessage;

pub mod sph;
pub mod setup;
pub mod measure;
pub mod commands;



/// Store the current state of all fluid particles to a file
fn save_system_state(particles: Vec<SerParticle3D>, filepath: &str) -> std::io::Result<()> {
    let ron_string = ron::to_string(&particles).unwrap();
    let mut file = std::fs::File::create(filepath)?;
    file.write_all(ron_string.as_bytes())?;
    Ok(())
}

#[derive(Debug, Clone)]
pub struct SimulationInfo {
    // particle size
    pub particle_diameter: f32,
    // rest density
    pub rest_density: f32,
    // light position
    pub light_position: [f32; 3],
    // particle color
    pub particle_color: ParticleColor,
    // boundary particle color
    pub boundary_particle_color: ParticleColor,
    /// Integration Scheme
    pub integration_scheme: sph::PropagationMethod,
    /// maximum buffer length
    pub buffer_length_limit: usize,
}

#[derive(Clone, Default)]
pub struct TimeStepInfo {
    // hand over time_inc
    pub time_inc: f32,
    // average density
    pub average_density: f32,
    //
    pub fluid: Vec<SerParticle3D>,
    pub boundary: Vec<BoundaryParticle3D>
}

/// Struct that does:
/// - holds initial state of a system
/// - develops the system in time
/// - allows resetting the system to the initial state
struct Simulation {
    initial_system: sph::System3D,
    system: sph::System3D,
    info: SimulationInfo,
}

impl Simulation {
    /// Try to load simulation and return initial state
    fn load(
        config: &str,
        state: Option<&str>,
        _measure: Option<&str>,
        _finish_time: Option<f32>
    ) -> Result<(TimeStepInfo, Simulation), String> {
        match setup::System3DConfigConstructor::new(
            config,
            state,
        ) {
            Ok((sys_conf, sim_info)) => {
                let initial_system = sph::System3D::new(sys_conf.finish());
                #[cfg(feature = "logging")]
                info!("Loaded new simulation!");
                let sim: Simulation = Self {
                    initial_system: initial_system.clone(),
                    system: initial_system,
                    info: sim_info,
                };
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
        self.system.step_forward_in_time(&self.info.integration_scheme);
        self.system.get_time_step_info()
    }
    fn reset(&mut self) {
        self.system = self.initial_system.clone();
    }
}

#[derive(Default, PartialEq, Eq)]
enum SimulationState {
    Computing,
    #[default]
    Paused,
}

/// Struct that does:
/// - wraps the [[Simulation]], in case one is loaded
/// - holds the info if new time steps are currently computed or not
#[derive(Default)]
struct SimulationController {
    state: SimulationState,
    compute_timesteps: usize,
    simulation: Option<Simulation>,
}

impl SimulationController {
    /// Try to load simulation and return initial state
    fn load_simulation(&mut self, config: &str, state: Option<&str>, measure: Option<&str>, finish_time: Option<f32>,
    ) -> Result<TimeStepInfo, String>{
        self.compute_timesteps = 0;
        match Simulation::load(config, state, measure, finish_time) {
            Ok((initial_state, sim)) => {
                self.simulation = Some(sim);
                Ok(initial_state)
            },
            Err(e) => { Err(e) },
        }
    }
    fn compute_more_timesteps(&mut self, num: usize) {
        self.compute_timesteps += num;
    }
    fn compute(&mut self) {
        self.state = SimulationState::Computing;
    }
    // fn pause(&mut self) {
    //     self.state = SimulationState::Paused;
    // }
    fn get_next_time_step(&mut self) -> Option<TimeStepInfo> {
        if self.state == SimulationState::Computing && self.compute_timesteps > 0 {
            self.simulation.as_mut().map(|sim| {
                self.compute_timesteps -= 1;
                sim.get_next_time_step()
            })
        } else {
            None
        }
    }
    fn reset(&mut self) {
        self.compute_timesteps = 0;
        if let Some(sim) = &mut self.simulation {
            sim.reset();
        }
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
                WorkerCommand::Simulate { config, state, measure, finish_time } => {
                    // load system
                    match simulation_controller.load_simulation(
                        &config,
                        state.as_deref(),
                        measure.as_deref(),
                        finish_time
                    ) {
                        Ok(initial_state) => {
                            let _ = to_ui.send_event(WorkerMessage::SimulationLoaded(simulation_controller.simulation.as_ref().unwrap().info.clone()));
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
                    #[cfg(feature = "logging")]
                    debug!("compute: {}", num);
                    simulation_controller.compute_more_timesteps(num);
                },
                WorkerCommand::Save { particles, filepath } => {
                    let save_message = if save_system_state(particles, &filepath).is_ok() {
                        #[cfg(feature = "logging")]
                        info!("Successfully saved state!");
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
                    simulation_controller.reset();
                    let _ = to_ui.send_event(WorkerMessage::FinishedResetting);
                },
                WorkerCommand::Stop => {
                    #[cfg(feature = "logging")]
                    info!("Stopped simulation!");
                    break 'worker;
                },
            }
        }
        if let Some(res) = simulation_controller.get_next_time_step() {
            let _ = to_ui.send_event(WorkerMessage::TimeIncFinished(res));
        } else {
            std::thread::sleep(Duration::from_millis(16));
        }
    }
}