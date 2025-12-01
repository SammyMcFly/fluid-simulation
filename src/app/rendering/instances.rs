//! Instance definition and store for instances
//!
use iced_wgpu::wgpu;
use iced_wgpu::wgpu::util::DeviceExt;

// #[cfg(feature = "logging")]
// use tracing::{
//     debug,
// }; // error, trace, warn, debug, info,

use crate::app::rendering::frame_control::NextAction;
use crate::app::rendering::model::ToRaw;
use crate::app::backend::rusty_fluid_solver::TimeStepInfo;
use crate::app::backend::rusty_fluid_solver::Positional;
use crate::app::rendering::ui::controls::{ParticleColor, cut::Cut};




#[derive(Debug, Clone, Default)]
pub struct Instance {
    pub position: nalgebra::Vector3<f32>,
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
        Self { cut, is_boundary_hidden, particle_color, boundary_particle_color, }
    }
}

pub enum StagingResult {
    Initialized,
    SomeTaken,
    StoppedAtLoopEnd,
    NoneTaken,
    NothingToStage,
    Uninitialized,
}

pub struct InstanceStore {
    pub staged_info: Option<TimeStepInfo>,
    staging_settings: Option<StagingSettings>,
    pub rendered_instances: Option<Vec<Instance>>,
    pub buffer: wgpu::Buffer,

    pub info_queue: Vec<TimeStepInfo>,
    current_index: usize,
    allow_looping_once: bool,
}

impl InstanceStore {
    pub fn new(gpu_context: &super::gpu_context::GpuContext) -> Self {
        let rendered_instances: Option<Vec<Instance>> = None;
        let instance_buffer = Self::create_instance_buffer(gpu_context, rendered_instances.as_deref());

        Self {
            staged_info: None,
            staging_settings: None,
            rendered_instances,
            buffer: instance_buffer,
            info_queue: Vec::default(),
            current_index: 0,
            allow_looping_once: false,
        }
    }

    fn create_instance_buffer(
        gpu_context: &super::gpu_context::GpuContext,
        instances: Option<&[Instance]>,
    ) -> wgpu::Buffer {
        let instance_data = if let Some(inst) = instances && !inst.is_empty() {
            inst.iter().map(Instance::to_raw).collect::<Vec<_>>()
        } else {
            // println!("is none or empty!");
            vec![super::model::InstanceRaw::new(
                [
                    [1.0,0.0,0.0,0.0],
                    [0.0,1.0,0.0,0.0],
                    [0.0,0.0,1.0,0.0],
                    [0.0, 0.0, 0.0, 1.0]
                ],
                [0., 1., 0.,],
            )]
        };

        gpu_context.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&instance_data),
                usage: wgpu::BufferUsages::VERTEX,
            }
        )
    }

    pub fn store(&mut self, time_steps: Vec<TimeStepInfo>) {
        self.info_queue = time_steps;
        self.current_index = 0;
    }

    pub fn get_info(&self) -> Option<&TimeStepInfo> {
        if let Some(info) = &self.staged_info {
            Some(info)
        } else {
            None
        }
    }

    /// Get time increment
    pub fn get_time_inc(&self) -> f32 {
        if let Some(info) = &self.staged_info {
            info.time_increment
        } else {
            0.
        }
    }

    /// Filter particles and pass selected to rendered instances
    fn info_to_instances(&mut self,) {
        if let Some(info) = &self.staged_info {
            let info = info.clone();
            let settings = self.staging_settings.as_ref().unwrap().clone();
            self.rendered_instances = Some(info.fluid.into_iter().filter(|particle| {
                settings.cut.cut(particle)
            }).filter(|particle| {
                particle.is_enabled()
            }).map(|particle| {
                let color = match settings.particle_color {
                    ParticleColor::VelocityGraded => {
                        let whiteness = f64::min(
                            (particle.vel_now()[0].powi(2)+particle.vel_now()[1].powi(2)+particle.vel_now()[2].powi(2)).powf(0.5)/10.,
                            1.,
                        );
                        [ whiteness as f32, whiteness as f32, 1. ]
                    },
                    ParticleColor::FixedColor(color) => color,
                };
                Instance {
                    // flip y and z coordinate
                    position: nalgebra::Vector3::new(-particle.pos_now()[0] as f32, particle.pos_now()[2] as f32, particle.pos_now()[1] as f32),
                    color,
                }
            }).collect());
            if !settings.is_boundary_hidden {
                self.rendered_instances.as_mut().unwrap().extend(info.boundary.into_iter().filter(|particle| {
                    settings.cut.cut(particle)
                }).map(|particle| {
                    let color = match settings.boundary_particle_color {
                    ParticleColor::VelocityGraded => {
                            #[cfg(not(feature = "global_pressure"))]
                            let vel = 0.;
                            #[cfg(feature = "global_pressure")]
                            let vel = particle.vel_now();
                            let whiteness = f64::min(
                                (vel[0].powi(2)+vel[1].powi(2)+vel[2].powi(2)).powf(0.5)/10.,
                                1.,
                            );
                            [ whiteness as f32, whiteness as f32, 1. ]
                        },
                        ParticleColor::FixedColor(color) => color,
                    };
                    Instance {
                        // flip y and z coordinate
                        position: nalgebra::Vector3::new(-particle.pos_now()[0] as f32, particle.pos_now()[2] as f32, particle.pos_now()[1] as f32),
                        color,
                    }
                }).collect::<Vec<Instance>>());
            }
        }
    }

    pub fn finished_loop(&self, forward: bool) -> bool {
        if forward {
            self.current_index == self.info_queue.len()-1
        } else {
            self.current_index == 0
        }

    }

    pub fn allow_looping_once(&mut self, looped_playback: bool,) {
        if !looped_playback {
            self.allow_looping_once = true;
        }
    }

    pub fn reset_allow_looping_once(&mut self) {
        self.allow_looping_once = false;
    }

    /// Advances index to next index depending on the direction and looping behavior
    ///
    /// Returns if it tried unallowed loop
    fn next_index(&mut self, forward: bool, looped: bool,) -> bool {
        if forward {
            if self.current_index+1 < self.info_queue.len() {
                self.current_index += 1;
                false
            } else if self.current_index+1 >= self.info_queue.len() && looped {
                self.current_index = 0;
                false
            } else if self.current_index+1 >= self.info_queue.len() && !looped && self.allow_looping_once {
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
            self.current_index = self.info_queue.len()-1;
            false
        } else if self.current_index == 0 && !looped && self.allow_looping_once {
            self.current_index = self.info_queue.len()-1;
            self.allow_looping_once = false;
            false
        } else {
            true
        }
    }

    fn prestage(&mut self) {
        let ts_info = &self.info_queue[self.current_index];
        self.staged_info = Some(ts_info.clone());
    }

    fn stage(
        &mut self,
        gpu_context: &super::gpu_context::GpuContext,
        staging_settings: &StagingSettings,
    ) {
        self.staging_settings = Some(staging_settings.clone());
        self.info_to_instances();
        self.buffer = Self::create_instance_buffer(gpu_context, self.rendered_instances.as_deref());
    }

    pub fn stage_next(
        &mut self,
        gpu_context: &super::gpu_context::GpuContext,
        staging_settings: &StagingSettings,
        action: NextAction,
        forward: bool,
        looped_playback: bool,
    ) -> StagingResult {

        if self.info_queue.is_empty() && self.staged_info.is_none() {
            StagingResult::Uninitialized
        } else if self.info_queue.is_empty() { // && self.staged_info.is_some()
            StagingResult::NothingToStage
        } else if self.staged_info.is_none() {
            assert!(self.current_index == 0);
            self.prestage();
            self.stage(gpu_context, staging_settings);
            StagingResult::Initialized
        } else {
            match action {
                NextAction::PlayTimeInterval(interval) => {
                    let mut taken = 0;
                    let mut interval = interval-self.staged_info.as_ref().unwrap().time_increment;
                    while interval >= 0. {
                        if self.next_index(forward, looped_playback) {
                            if taken > 0 {
                                self.stage(gpu_context, staging_settings);
                            }
                            return StagingResult::StoppedAtLoopEnd;
                        }
                        taken += 1;
                        self.prestage();
                        interval -= self.staged_info.as_ref().unwrap().time_increment;
                    }
                    self.stage(gpu_context, staging_settings);
                    StagingResult::SomeTaken
                },
                NextAction::StepInTime => {
                    self.next_index(forward, true);
                    self.prestage();
                    self.stage(gpu_context, staging_settings);
                    StagingResult::SomeTaken
                },
                NextAction::Wait => StagingResult::NoneTaken,
            }
        }
    }

    pub fn update_staged(
        &mut self,
        gpu_context: &super::gpu_context::GpuContext,
        staging_settings: &StagingSettings,
    ) {
        if let Some(sta_set) = &self.staging_settings && *staging_settings != *sta_set {
            // println!("not eq \n{:?}, \n{:?}", *staging_settings, *sta_set);
            self.staging_settings = Some(staging_settings.clone());
            self.info_to_instances();
            self.buffer = Self::create_instance_buffer(gpu_context, self.rendered_instances.as_deref());
        }
    }

    pub fn reset(&mut self, gpu_context: &super::gpu_context::GpuContext,) {
        self.staged_info = None;
        self.staging_settings = None;
        self.rendered_instances = None;
        self.buffer = Self::create_instance_buffer(gpu_context, self.rendered_instances.as_deref());
        self.current_index = 0;
        self.allow_looping_once = false;
    }

    pub fn is_empty(&self) -> bool {
        self.staged_info.is_none()
    }

    pub fn queue_len(&self) -> usize {
        self.info_queue.len()-(self.current_index+1)
    }
}
