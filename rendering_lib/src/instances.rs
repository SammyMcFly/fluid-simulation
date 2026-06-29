//! Instance definition and store for instances
//!
use simulation_lib::render_info::TimeStepInfo;
// #[cfg(feature = "logging")]
// use tracing::{
//     debug,
// }; // error, trace, warn, debug, info,

use crate::frame_control::Action;
use crate::ui::controls::cut::Cut;


// #[derive(Debug, Clone, Default)]
// pub struct Instance {
//     /// Position of instance with order: x, y, z
//     pub position: nalgebra::Vector3<f32>,
//     pub radius: f32,
//     /// Color of instance
//     pub color: [f32; 4],
// }

/// Compact instance data for billboard impostors
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BillboardInstanceRaw {
    pub center: [f32; 3],
    pub radius: f32,
    pub color: [f32; 4],
}

impl BillboardInstanceRaw {
    pub fn new(center: [f32; 3], radius: f32, color: [f32; 4]) -> Self {
        Self { center, radius, color }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct StagingSettings {
    cut: Cut,
    is_boundary_hidden: bool,
}

impl StagingSettings {
    pub fn new(cut: Cut, is_boundary_hidden: bool) -> Self {
        Self {
            cut,
            is_boundary_hidden,
        }
    }
}

pub enum StagingResult {
    Initialized,
    SteppedInTime,
    SomeTaken(usize),
    StoppedAtLoopEndWithSomeTaken(usize),
    StoppedAtLoopEndWithNoneTaken,
    NoneTaken,
    NothingToStage,
    Uninitialized,
}

pub struct InstanceStore {
    // staging_settings: Option<StagingSettings>,
    // pub rendered_instances: Option<Vec<Instance>>,
    // pub buffer: wgpu::Buffer,

    pub info_buffer: Vec<TimeStepInfo>,
    current_index: usize,
    active: bool,
    allow_looping_once: bool,
}

impl InstanceStore {
    pub fn new() -> Self {
        // let rendered_instances: Option<Vec<Instance>> = None;
        // let instance_buffer =
        //     Self::create_instance_buffer(gpu_context, rendered_instances.as_deref());

        Self {
            // staging_settings: None,
            // rendered_instances,
            // buffer: instance_buffer,
            info_buffer: Vec::default(),
            current_index: 0,
            active: false,
            allow_looping_once: false,
        }
    }

    // fn create_instance_buffer(
    //     gpu_context: &super::gpu_context::GpuContext,
    //     instances: Option<&[Instance]>,
    // ) -> wgpu::Buffer {
    //     let instance_data = if let Some(inst) = instances
    //         && !inst.is_empty()
    //     {
    //         inst.iter().map(Instance::to_raw).collect::<Vec<_>>()
    //     } else {
    //         // Empty placeholder — one invisible particle
    //         vec![BillboardInstanceRaw::new(
    //             [0.0, 0.0, 0.0],
    //             0.0,
    //             [0., 1., 0., 1.],
    //         )]
    //     };

    //     gpu_context
    //         .device
    //         .create_buffer_init(&wgpu::util::BufferInitDescriptor {
    //             label: Some("Instance Buffer"),
    //             contents: bytemuck::cast_slice(&instance_data),
    //             usage: wgpu::BufferUsages::VERTEX,
    //         })
    // }

    // pub fn is_active(&self) -> bool {
    //     self.rendered_instances.is_some()
    // }
    pub fn is_active(&self) -> bool {
        self.active
    }

    // pub fn store(&mut self, time_steps: Vec<TimeStepInfo>) {
    //     self.staging_settings = None;
    //     self.rendered_instances = None;
    //     self.info_buffer = time_steps;
    //     self.current_index = 0;
    // }
    /// Replace the entire buffer (e.g. loading from file)
    pub fn store(&mut self, time_steps: Vec<TimeStepInfo>) {
        // self.staging_settings = None;
        self.info_buffer = time_steps;
        self.current_index = 0;
        self.active = false;
    }

    pub fn push(&mut self, time_step_info: TimeStepInfo) {
        self.info_buffer.push(time_step_info);
    }

    pub fn get_time_step_info(&self) -> Option<&TimeStepInfo> {
        if self.active {
            Some(&self.info_buffer[self.current_index])
        } else {
            None
        }
    }

    /// Get time increment
    pub fn get_time_inc(&self) -> f32 {
        if self.is_active() {
            self.info_buffer[self.current_index].measurement.time_step_size as f32
        } else {
            0.
        }
    }

    // pub fn particle_count(&self) -> u32 {
    //     self.rendered_instances
    //         .as_ref()
    //         .map(|v| v.len() as u32)
    //         .unwrap_or(0)
    // }

    // /// Filter particles and pass selected to rendered instances
    // fn info_to_instances(&mut self) {
    //     let settings = self.staging_settings.as_ref().unwrap().clone();
    //     self.rendered_instances = Some(
    //         self.info_buffer[self.current_index]
    //             .fluid
    //             .position
    //             .iter()
    //             .zip(&self.info_buffer[self.current_index].fluid.velocity)
    //             .filter(|(id_position, _id_velocity)| settings.cut.cut(id_position))
    //             .map(|(id_position, id_velocity)| {
    //                 let color = match settings.particle_color {
    //                     ParticleColor::VelocityGraded => {
    //                         let whiteness = f64::min(
    //                             (id_velocity[0].powi(2)
    //                                 + id_velocity[1].powi(2)
    //                                 + id_velocity[2].powi(2))
    //                             .powf(0.5)
    //                                 / 10.,
    //                             1.,
    //                         );
    //                         [whiteness as f32, whiteness as f32, 1.]
    //                     }
    //                     ParticleColor::FixedColor(color) => color,
    //                 };
    //                 Instance {
    //                     position: nalgebra::Vector3::new(
    //                         id_position[0] as f32,
    //                         id_position[1] as f32,
    //                         id_position[2] as f32,
    //                     ),
    //                     radius: RADIUS,
    //                     color: [color[0], color[1], color[2], ALPHA],
    //                 }
    //             })
    //             .collect(),
    //     );
    //     if !settings.is_boundary_hidden {
    //         self.rendered_instances.as_mut().unwrap().extend(
    //             self.info_buffer[self.current_index]
    //                 .boundary
    //                 .position
    //                 .iter()
    //                 .zip(&self.info_buffer[self.current_index].boundary.velocity)
    //                 .filter(|(id_position, _id_velocity)| settings.cut.cut(id_position))
    //                 .map(|(id_position, id_velocity)| {
    //                     let color = match settings.boundary_particle_color {
    //                         ParticleColor::VelocityGraded => {
    //                             let vel = id_velocity;
    //                             let whiteness = f64::min(
    //                                 (vel[0].powi(2) + vel[1].powi(2) + vel[2].powi(2)).powf(0.5)
    //                                     / 10.,
    //                                 1.,
    //                             );
    //                             [whiteness as f32, whiteness as f32, 1.]
    //                         }
    //                         ParticleColor::FixedColor(color) => color,
    //                     };
    //                     Instance {
    //                         position: nalgebra::Vector3::new(
    //                             id_position[0] as f32,
    //                             id_position[1] as f32,
    //                             id_position[2] as f32,
    //                         ),
    //                         radius: RADIUS,
    //                         color: [color[0], color[1], color[2], ALPHA],
    //                     }
    //                 })
    //                 .collect::<Vec<Instance>>(),
    //         );
    //     }
    // }

    pub fn remaining_buffer_len(&self) -> usize {
        if self.info_buffer.is_empty() {
            0
        } else {
            self.info_buffer.len() - (self.current_index + 1)
        }
    }

    pub fn finished_loop(&self, forward: bool) -> bool {
        if self.info_buffer.is_empty() {
            return true;
        }
        if forward {
            self.current_index == self.info_buffer.len() - 1
        } else {
            self.current_index == 0
        }
    }

    pub fn allow_looping_once(&mut self, looped_playback: bool) {
        if !looped_playback {
            self.allow_looping_once = true;
        }
    }

    pub fn reset_allow_looping_once(&mut self) {
        self.allow_looping_once = false;
    }

    /// Advances to the next frame depending on the direction and looping behavior
    ///
    /// Returns true if it tried unallowed loop
    fn next_index(&mut self, forward: bool, looped: bool) -> bool {
        if forward {
            if self.current_index + 1 < self.info_buffer.len() {
                self.current_index += 1;
                false
            } else if self.current_index + 1 >= self.info_buffer.len() && looped {
                self.current_index = 0;
                false
            } else if self.current_index + 1 >= self.info_buffer.len()
                && !looped
                && self.allow_looping_once
            {
                self.current_index = 0;
                self.allow_looping_once = false;
                false
            } else {
                true
            }
        } else if self.current_index > 0 {
            self.current_index -= 1;
            false
        } else if self.current_index == 0 && looped {
            self.current_index = self.info_buffer.len() - 1;
            false
        } else if self.current_index == 0 && !looped && self.allow_looping_once {
            self.current_index = self.info_buffer.len() - 1;
            self.allow_looping_once = false;
            false
        } else {
            true
        }
    }

    pub fn discard_past(&mut self) -> usize {
        if self.current_index == 0 {
            return 0;
        }
        let discarded = self.current_index;
        self.info_buffer.drain(0..self.current_index);
        self.current_index = 0;
        discarded
    }

    fn activate(&mut self, discard_past: bool) -> usize {
        self.active = true;
        if discard_past {
            self.discard_past()
        } else {
            0
        }
    }

    // fn stage(
    //     &mut self,
    //     gpu_context: &super::gpu_context::GpuContext,
    //     staging_settings: &StagingSettings,
    //     discard_past: bool,
    // ) -> usize {
    //     let discarded = if discard_past { self.discard_past() } else { 0 };
    //     // self.staging_settings = Some(staging_settings.clone());
    //     // self.info_to_instances();
    //     // self.buffer = Self::create_instance_buffer(gpu_context, self.rendered_instances.as_deref());
    //     discarded
    // }

    /// Determine what to stage next based on playback action.
    /// Returns a `StagingResult` indicating what happened.
    pub fn stage_next(
        &mut self,
        action: Action,
        forward: bool,
        looped_playback: bool,
        discard_past: bool,
    ) -> StagingResult {
        if self.info_buffer.is_empty() && !self.is_active() {
            return StagingResult::Uninitialized;
        }
        if self.info_buffer.is_empty() {
            // && self.staged_info.is_some()
            return StagingResult::NothingToStage;
        }
        if !self.is_active() {
            assert!(self.current_index == 0);
            // self.stage(gpu_context, staging_settings, false);
            self.activate(false);
            return StagingResult::Initialized;
        }
        match action {
            Action::PlayTimeInterval(interval) => {
                let mut taken = 0;
                let mut interval =
                    interval - self.info_buffer[self.current_index].measurement.time_step_size as f32;
                while interval >= 0. {
                    if self.next_index(forward, looped_playback) {
                        if taken > 0 {
                            // let discarded =
                            //     self.stage(gpu_context, staging_settings, discard_past);
                            let discarded = self.activate(discard_past);
                            return StagingResult::StoppedAtLoopEndWithSomeTaken(discarded);
                        }
                        return StagingResult::StoppedAtLoopEndWithNoneTaken;
                    }
                    taken += 1;
                    interval -= self.info_buffer[self.current_index].measurement.time_step_size as f32;
                }
                if taken > 0 {
                    // let discarded = self.stage(gpu_context, staging_settings, discard_past);
                    let discarded = self.activate(discard_past);
                    StagingResult::SomeTaken(discarded)
                } else {
                    StagingResult::NoneTaken
                }
            }
            Action::StepInTime => {
                self.next_index(forward, true);
                // self.stage(gpu_context, staging_settings, discard_past);
                self.activate(discard_past);
                StagingResult::SteppedInTime
            }
            Action::Wait => StagingResult::NoneTaken,
        }
    }

    // pub fn update_staged(
    //     &mut self,
    //     gpu_context: &super::gpu_context::GpuContext,
    //     staging_settings: &StagingSettings,
    // ) {
    //     if let Some(sta_set) = &self.staging_settings
    //         && *staging_settings != *sta_set
    //     {
    //         // println!("not eq \n{:?}, \n{:?}", *staging_settings, *sta_set);
    //         self.staging_settings = Some(staging_settings.clone());
    //         self.info_to_instances();
    //         self.buffer =
    //             Self::create_instance_buffer(gpu_context, self.rendered_instances.as_deref());
    //     }
    // }

    pub fn reset(&mut self, clear_buffer: bool) {
        if clear_buffer {
            self.info_buffer.clear();
        }
        // self.staging_settings = None;
        // self.rendered_instances = None;
        // self.buffer = Self::create_instance_buffer(gpu_context, self.rendered_instances.as_deref());
        self.current_index = 0;
        self.active = false;
        self.allow_looping_once = false;
    }
}
