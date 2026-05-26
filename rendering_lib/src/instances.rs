//! Instance definition and store for instances
//!
use iced_wgpu::wgpu;
use iced_wgpu::wgpu::util::DeviceExt;

// #[cfg(feature = "logging")]
// use tracing::{
//     debug,
// }; // error, trace, warn, debug, info,

use crate::frame_control::Action;
use crate::model::ToRaw;
use crate::ui::controls::cut::Cut;
use simulation_lib::sph::sample::Positional;
use simulation_lib::{ParticleColor, TimeStepInfo};

#[derive(Debug, Clone, Default)]
pub struct Instance {
    /// Position of instance with order: x, y, z
    pub position: nalgebra::Vector3<f32>,
    /// Color of instance
    pub color: [f32; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub struct StagingSettings {
    cut: Cut,
    is_boundary_hidden: bool,
    particle_color: ParticleColor,
    boundary_particle_color: ParticleColor,
}

impl StagingSettings {
    pub fn new(
        cut: Cut,
        is_boundary_hidden: bool,
        particle_color: ParticleColor,
        boundary_particle_color: ParticleColor,
    ) -> Self {
        Self {
            cut,
            is_boundary_hidden,
            particle_color,
            boundary_particle_color,
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
    staging_settings: Option<StagingSettings>,
    pub rendered_instances: Option<Vec<Instance>>,
    pub buffer: wgpu::Buffer,

    pub info_buffer: Vec<TimeStepInfo>,
    current_index: usize,
    allow_looping_once: bool,
}

impl InstanceStore {
    pub fn new(gpu_context: &super::gpu_context::GpuContext) -> Self {
        let rendered_instances: Option<Vec<Instance>> = None;
        let instance_buffer =
            Self::create_instance_buffer(gpu_context, rendered_instances.as_deref());

        Self {
            staging_settings: None,
            rendered_instances,
            buffer: instance_buffer,
            info_buffer: Vec::default(),
            current_index: 0,
            allow_looping_once: false,
        }
    }

    fn create_instance_buffer(
        gpu_context: &super::gpu_context::GpuContext,
        instances: Option<&[Instance]>,
    ) -> wgpu::Buffer {
        let instance_data = if let Some(inst) = instances
            && !inst.is_empty()
        {
            inst.iter().map(Instance::to_raw).collect::<Vec<_>>()
        } else {
            // println!("is none or empty!");
            vec![super::model::InstanceRaw::new(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [0., 1., 0.],
            )]
        };

        gpu_context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&instance_data),
                usage: wgpu::BufferUsages::VERTEX,
            })
    }

    pub fn is_active(&self) -> bool {
        self.rendered_instances.is_some()
    }

    pub fn store(&mut self, time_steps: Vec<TimeStepInfo>) {
        self.staging_settings = None;
        self.rendered_instances = None;
        self.info_buffer = time_steps;
        self.current_index = 0;
    }

    pub fn push(&mut self, time_step_info: TimeStepInfo) {
        self.info_buffer.push(time_step_info);
    }

    pub fn get_info(&self) -> Option<&TimeStepInfo> {
        if self.is_active() {
            Some(&self.info_buffer[self.current_index])
        } else {
            None
        }
    }

    /// Get time increment
    pub fn get_time_inc(&self) -> f32 {
        if self.is_active() {
            self.info_buffer[self.current_index].time_increment
        } else {
            0.
        }
    }

    /// Filter particles and pass selected to rendered instances
    fn info_to_instances(&mut self) {
        let settings = self.staging_settings.as_ref().unwrap().clone();
        self.rendered_instances = Some(
            self.info_buffer[self.current_index]
                .fluid
                .position
                .iter()
                .zip(&self.info_buffer[self.current_index].fluid.velocity)
                .zip(&self.info_buffer[self.current_index].fluid.enabled)
                .filter(|((id_position, _id_velocity), _id_enabled)| settings.cut.cut(id_position))
                .filter(|((_id_position, _id_velocity), id_enabled)| **id_enabled)
                .map(|((_id_position, id_velocity), _id_enabled)| {
                    let color = match settings.particle_color {
                        ParticleColor::VelocityGraded => {
                            let whiteness = f64::min(
                                (id_velocity[0].powi(2)
                                    + id_velocity[1].powi(2)
                                    + id_velocity[2].powi(2))
                                .powf(0.5)
                                    / 10.,
                                1.,
                            );
                            [whiteness as f32, whiteness as f32, 1.]
                        }
                        ParticleColor::FixedColor(color) => color,
                    };
                    Instance {
                        position: nalgebra::Vector3::new(
                            _id_position[0] as f32,
                            _id_position[1] as f32,
                            _id_position[2] as f32,
                        ),
                        color,
                    }
                })
                .collect(),
        );
        if !settings.is_boundary_hidden {
            self.rendered_instances.as_mut().unwrap().extend(
                self.info_buffer[self.current_index]
                    .boundary
                    .position
                    .iter()
                    .zip(&self.info_buffer[self.current_index].boundary.velocity)
                    .filter(|(id_position, _id_velocity)| settings.cut.cut(id_position))
                    .map(|(id_position, id_velocity)| {
                        let color = match settings.boundary_particle_color {
                            ParticleColor::VelocityGraded => {
                                let vel = id_velocity;
                                let whiteness = f64::min(
                                    (vel[0].powi(2) + vel[1].powi(2) + vel[2].powi(2)).powf(0.5)
                                        / 10.,
                                    1.,
                                );
                                [whiteness as f32, whiteness as f32, 1.]
                            }
                            ParticleColor::FixedColor(color) => color,
                        };
                        Instance {
                            position: nalgebra::Vector3::new(
                                id_position[0] as f32,
                                id_position[1] as f32,
                                id_position[2] as f32,
                            ),
                            color,
                        }
                    })
                    .collect::<Vec<Instance>>(),
            );
        }
    }

    pub fn finished_loop(&self, forward: bool) -> bool {
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

    /// Advances index to next index depending on the direction and looping behavior
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
        let discarded: Vec<TimeStepInfo> = self.info_buffer.drain(0..self.current_index).collect();
        self.current_index = 0;
        discarded.len()
    }

    fn stage(
        &mut self,
        gpu_context: &super::gpu_context::GpuContext,
        staging_settings: &StagingSettings,
        discard_past: bool,
    ) -> usize {
        let discarded = if discard_past { self.discard_past() } else { 0 };
        self.staging_settings = Some(staging_settings.clone());
        self.info_to_instances();
        self.buffer = Self::create_instance_buffer(gpu_context, self.rendered_instances.as_deref());
        discarded
    }

    pub fn stage_next(
        &mut self,
        gpu_context: &super::gpu_context::GpuContext,
        staging_settings: &StagingSettings,
        action: Action,
        forward: bool,
        looped_playback: bool,
        discard_past: bool,
    ) -> StagingResult {
        if self.info_buffer.is_empty() && !self.is_active() {
            StagingResult::Uninitialized
        } else if self.info_buffer.is_empty() {
            // && self.staged_info.is_some()
            StagingResult::NothingToStage
        } else if !self.is_active() {
            assert!(self.current_index == 0);
            self.stage(gpu_context, staging_settings, false);
            StagingResult::Initialized
        } else {
            match action {
                Action::PlayTimeInterval(interval) => {
                    let mut taken = 0;
                    let mut interval =
                        interval - self.info_buffer[self.current_index].time_increment;
                    while interval >= 0. {
                        if self.next_index(forward, looped_playback) {
                            if taken > 0 {
                                let discarded =
                                    self.stage(gpu_context, staging_settings, discard_past);
                                return StagingResult::StoppedAtLoopEndWithSomeTaken(discarded);
                            }
                            return StagingResult::StoppedAtLoopEndWithNoneTaken;
                        }
                        taken += 1;
                        interval -= self.info_buffer[self.current_index].time_increment;
                    }
                    if taken > 0 {
                        let discarded = self.stage(gpu_context, staging_settings, discard_past);
                        StagingResult::SomeTaken(discarded)
                    } else {
                        StagingResult::NoneTaken
                    }
                }
                Action::StepInTime => {
                    self.next_index(forward, true);
                    self.stage(gpu_context, staging_settings, discard_past);
                    StagingResult::SteppedInTime
                }
                Action::Wait => StagingResult::NoneTaken,
            }
        }
    }

    pub fn update_staged(
        &mut self,
        gpu_context: &super::gpu_context::GpuContext,
        staging_settings: &StagingSettings,
    ) {
        if let Some(sta_set) = &self.staging_settings
            && *staging_settings != *sta_set
        {
            // println!("not eq \n{:?}, \n{:?}", *staging_settings, *sta_set);
            self.staging_settings = Some(staging_settings.clone());
            self.info_to_instances();
            self.buffer =
                Self::create_instance_buffer(gpu_context, self.rendered_instances.as_deref());
        }
    }

    pub fn reset(&mut self, gpu_context: &super::gpu_context::GpuContext, clear_buffer: bool) {
        if clear_buffer {
            self.info_buffer.clear();
        }
        self.staging_settings = None;
        self.rendered_instances = None;
        self.buffer = Self::create_instance_buffer(gpu_context, self.rendered_instances.as_deref());
        self.current_index = 0;
        self.allow_looping_once = false;
    }

    pub fn remaining_buffer_len(&self) -> usize {
        self.info_buffer.len() - (self.current_index + 1)
    }
}
