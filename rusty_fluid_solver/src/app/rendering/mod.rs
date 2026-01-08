//! AppState
//!
//!
use crossbeam::channel::Sender;

use tracing::{
    debug,
}; // error, trace, warn, debug, info,

use rendering_lib::*;
use rendering_lib::ui::UserInput;
use simulation_lib::{SimulationParameters, TimeStepInfo};

use crate::app::backend::commands::WorkerCommand;





pub trait Simulator {
    type Command;

    fn update(&mut self, to_worker: &Sender<Self::Command>,);

    fn new_simulation(&mut self, info: SimulationParameters,);

    fn received_content(&mut self, time_steps: TimeStepInfo);

    fn continue_after_reset(&mut self, info: SimulationParameters,);
}

impl Simulator for AppState {
    type Command = WorkerCommand;

    /// Update buffer for next rendering step (consider new state of State)
    fn update(
        &mut self,
        to_worker: &Sender<WorkerCommand>,
    ) {
        // send new timesteps to compute
        let steps = self.frame.get_time_steps_discarded();
        if steps > 0 {
            to_worker.send(WorkerCommand::AddTimeStepsToCompute(steps)).unwrap();
        }

        // handle user input messages
        for message in self.messages.drain(..) {
            match &message {
                // send commands to worker thread
                UserInput::RequestCameraReset => {
                    self.camera.reset(&self.gpu);
                    // self.light.reset(&self.gpu);
                },
                UserInput::StepForward => {
                    self.frame.step_forward();
                    debug!("Pressed Step!");
                },
                UserInput::StepBackward => {
                    self.frame.step_backward();
                    debug!("Pressed Step!");
                },
                UserInput::RequestReset => {
                    to_worker.send(WorkerCommand::Reset).unwrap();
                },
                UserInput::RequestSaving => {
                    if self.instances.is_active() {
                        to_worker.send(
                            WorkerCommand::SaveState {
                                particles: self.instances.get_info().unwrap().fluid.clone(),
                                filepath: "./state.ron".to_string()
                            }
                        ).unwrap()
                    }
                },
                UserInput::RequestScreenshot => {
                    self.screenshot.queue_screenshot("./screenshots".to_string());
                },
                UserInput::PlayForward => {
                    self.frame.reset_steps();
                    if self.instances.finished_loop(true) {
                        self.instances.allow_looping_once(self.ui.controls.loop_control.play_looped());
                    }
                    debug!("Play forward!");
                    // control is update in ui.update not here
                },
                UserInput::PlayBackward => {
                    self.frame.reset_steps();
                    if self.instances.finished_loop(false) {
                        self.instances.allow_looping_once(self.ui.controls.loop_control.play_looped());
                    }
                    debug!("Play backward!");
                    // control is update in ui.update not here
                },
                UserInput::Pause => {
                    assert!(self.frame.steps_to_do == 0);
                    self.instances.reset_allow_looping_once();
                    debug!("Pause!");
                    // control is update in ui.update not here
                },
                UserInput::RequestReadback(req) => {
                    to_worker.send(WorkerCommand::SaveScreenshot(req.clone())).unwrap()
                },
                UserInput::DiscardPast => {
                    let discarded = self.instances.discard_past();
                    self.frame.steps_discarded(discarded, self.ui.controls.discard_past);
                },
                _ => (),
            }
            // also update ui
            self.ui.update(message);
        }

        // Update camera
        self.camera.update(&self.gpu, self.frame.time_since_last_update());

        // Update the light
        self.light.update(&self.gpu, self.frame.time_since_last_update());

        // set time since last update
        self.frame.updating_now();
    }




    fn new_simulation(&mut self, sim_info: SimulationParameters,) {
        match model::ModelAssets::new(&self.gpu, sim_info.particle_diameter) {
            Ok(model) => self.model = model,
            Err(e) => panic!("Failed to load sphere: {}", e),
        }
        self.camera.reset(&self.gpu);
        self.light.set_light( &self.gpu, sim_info.light_position);
        self.instances = instances::InstanceStore::new(&self.gpu);
        self.ui.new_simulation(sim_info);
        self.frame.reset();
    }

    // might panic
    fn received_content(&mut self, info: TimeStepInfo) {
        self.instances.push(info);
    }

    fn continue_after_reset(&mut self, info: SimulationParameters,) {
        match model::ModelAssets::new(&self.gpu, info.particle_diameter) {
            Ok(model) => self.model = model,
            Err(e) => panic!("Failed to load sphere: {}", e),
        }
        // self.camera.reset(&self.gpu);
        // self.light.set_light(&self.gpu, sim_info.light_position);
        self.instances.reset(&self.gpu);
        self.ui.new_simulation(info);
        self.frame.reset();
    }
}
