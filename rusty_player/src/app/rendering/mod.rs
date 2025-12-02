//! AppState
//!
//!
use crossbeam::channel::Sender;

#[cfg(feature = "logging")]
use tracing::{
    debug,
}; // error, trace, warn, debug, info,

use rendering_lib::*;

use crate::app::backend::commands::WorkerCommand;
use simulation_lib::{SimulationParameters, TimeStepInfo};
use ui::UserInput;




// const LIGHT_COLOR: Option<[f32; 3]> = Some([1., 0.5, 0.5]);
const LIGHT_COLOR: Option<[f32; 3]> = Some([1.; 3]);


pub trait Player {
    type Command;

    fn update(&mut self, to_worker: &Sender<Self::Command>,);

    fn received_content(&mut self, sim_info: SimulationParameters, time_steps: Vec<TimeStepInfo>);
}


impl Player for AppState {
    type Command = WorkerCommand;

    /// Update buffer for next rendering step (consider new state of State)
    fn update(
        &mut self,
        to_worker: &Sender<WorkerCommand>,
    ) {
        // handle user input messages
        for message in self.messages.drain(..) {
            match message {
                // send commands to worker thread
                UserInput::RequestCameraReset => {
                    self.camera.reset(&self.gpu);
                    // self.light.reset(&self.gpu);
                },
                UserInput::StepForward => {
                    self.frame.step_forward();
                    #[cfg(feature = "logging")]
                    debug!("Pressed Step!");
                },
                UserInput::StepBackward => {
                    self.frame.step_backward();
                    #[cfg(feature = "logging")]
                    debug!("Pressed Step!");
                },
                UserInput::RequestReset => {
                    self.instances.reset(&self.gpu);
                    self.frame.reset();
                },
                UserInput::RequestSaving => {
                    if !self.instances.is_empty() {
                        to_worker.send(
                            WorkerCommand::SaveState {
                                particles: self.instances.get_info().unwrap().fluid.clone(),
                                file_path: "./state.ron".to_string()
                            }
                        ).unwrap()
                    }
                },
                UserInput::PlayForward => {
                    self.frame.reset_steps();
                    if self.instances.finished_loop(true) {
                        self.instances.allow_looping_once(self.ui.controls.loop_control.play_looped());
                    }
                    #[cfg(feature = "logging")]
                    debug!("Play forward!");
                    // control is update in ui.update not here
                },
                UserInput::PlayBackward => {
                    self.frame.reset_steps();
                    if self.instances.finished_loop(false) {
                        self.instances.allow_looping_once(self.ui.controls.loop_control.play_looped());
                    }
                    #[cfg(feature = "logging")]
                    debug!("Play backward!");
                    // control is update in ui.update not here
                },
                UserInput::Pause => {
                    assert!(self.frame.steps_to_do == 0);
                    self.instances.reset_allow_looping_once();
                    #[cfg(feature = "logging")]
                    debug!("Pause!");
                    // control is update in ui.update not here
                },
                _ => (),
            }
            // also update ui
            self.ui.update(message);
        }

        // Update camera
        self.camera.update(&self.gpu, self.frame.time_since_last_render());

        // Update the light
        self.light.update(&self.gpu, self.frame.time_since_last_render());
    }




    // might panic
    fn received_content(&mut self, sim_info: SimulationParameters, time_steps: Vec<TimeStepInfo>) {
        match model::ModelAssets::new(&self.gpu, sim_info.particle_diameter) {
            Ok(model) => self.model = model,
            Err(e) => panic!("Failed to load sphere: {}", e),
        }
        self.camera.reset(&self.gpu);
        self.light.set_light(&self.gpu, sim_info.light_position, LIGHT_COLOR);
        self.instances.store(time_steps);
        self.ui.new_simulation(sim_info);
        self.frame.reset();
    }
}
