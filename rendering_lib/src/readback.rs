//! Readback from GPU to store in image file
//!
use simulation_lib::measurement::RecordingStatus;

use iced_wgpu::wgpu;
use iced_winit::winit;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub struct ReadbackBuffer {
    pub buffer: wgpu::Buffer,
    pub padded_bytes_per_row: u32,
    /// Invoked map_async for this buffer
    pub mapping_started: bool,
}

#[derive(Debug, Clone, Default)]
pub struct BufferCycle {
    pub number_of_buffers: usize,
    pub staging_buffers: Vec<Arc<Mutex<ReadbackBuffer>>>,
    pub padded_bytes_per_row: u32,
    pub next_frame_index: usize,
}

impl BufferCycle {
    pub fn new(
        gpu_context: &super::gpu_context::GpuContext,
        size: winit::dpi::PhysicalSize<u32>,
    ) -> Self {
        let number_of_buffers = 3;

        let bytes_per_pixel = 4u32;
        let unpadded_bytes_per_row = size.width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT; // 256
        let padded_bytes_per_row = ((unpadded_bytes_per_row + align - 1) / align) * align;

        let buffer_size =
            padded_bytes_per_row as wgpu::BufferAddress * size.height as wgpu::BufferAddress;

        let mut staging_buffers = Vec::new();
        for _ in 0..number_of_buffers {
            staging_buffers.push(Arc::new(Mutex::new(ReadbackBuffer {
                buffer: gpu_context.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("readback staging buffer"),
                    size: buffer_size,
                    mapped_at_creation: false,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                }),
                padded_bytes_per_row,
                mapping_started: false,
            })));
        }

        Self {
            number_of_buffers,
            staging_buffers,
            padded_bytes_per_row,
            next_frame_index: 0,
        }
    }

    pub fn get_next_buffer_and_info(&mut self) -> (&Arc<Mutex<ReadbackBuffer>>, usize, u32) {
        let next_frame_index = self.next_frame_index;
        let buffer = &self.staging_buffers[next_frame_index % self.number_of_buffers];
        self.next_frame_index += 1;
        (buffer, next_frame_index, self.padded_bytes_per_row)
    }
}

#[derive(Debug, Clone)]
pub struct ReadbackRequest {
    pub buffer: Arc<Mutex<ReadbackBuffer>>,
    pub width: u32,
    pub height: u32,
    pub frame_index: usize,
    pub device: Arc<wgpu::Device>,
    pub output_dir: PathBuf,
}

#[derive(Debug, Clone, Default)]
pub struct ReadbackController {
    pub buffers: BufferCycle,

    render_directory: Option<String>,
    start_time: Option<f64>,
    finish_time: Option<f64>,
    started: bool,
    finished: bool,

    take_single_screenshot: bool,
    screenshot_directory: Option<String>,
}

pub enum ReadbackAction {
    Read(PathBuf),
    None,
}

impl ReadbackController {
    pub fn new(
        gpu_context: &super::gpu_context::GpuContext,
        size: winit::dpi::PhysicalSize<u32>,
        render_directory: Option<String>,
        start_time: Option<f64>,
        finish_time: Option<f64>,
    ) -> Self {
        Self {
            buffers: BufferCycle::new(gpu_context, size),
            render_directory,
            start_time,
            finish_time,
            started: false,
            finished: false,
            take_single_screenshot: false,
            screenshot_directory: None,
        }
    }

    pub fn resize(
        &mut self,
        gpu_context: &super::gpu_context::GpuContext,
        size: winit::dpi::PhysicalSize<u32>,
    ) {
        self.buffers = BufferCycle::new(gpu_context, size);
    }

    pub fn queue_screenshot(&mut self, screenshot_directory: String) {
        self.take_single_screenshot = true;
        self.screenshot_directory = Some(screenshot_directory);
    }

    /// update rendering status
    pub fn update_rendering_status(&mut self, time: f32, rendering_status: &mut RecordingStatus) {
        if self.render_directory.is_some() {
            if let Some(start) = self.start_time
                && time as f64 >= start
                && !self.started
            {
                rendering_status.advance_to_next_state();
                self.started = true;
            }
            if let Some(finish) = self.finish_time
                && time as f64 >= finish
                && !self.finished
            {
                rendering_status.advance_to_next_state();
                self.finished = true;
            }
        }
    }

    /// Take screenshot now?
    pub fn screenshot_this(
        &mut self,
        frame_new: bool,
        rendering_status: RecordingStatus,
    ) -> ReadbackAction {
        if self.render_directory.is_some()
            && frame_new
            && matches!(rendering_status, RecordingStatus::InProgress)
        {
            ReadbackAction::Read(PathBuf::from(self.render_directory.as_ref().unwrap()))
        } else if self.take_single_screenshot {
            self.take_single_screenshot = false;
            ReadbackAction::Read(PathBuf::from(self.screenshot_directory.as_ref().unwrap()))
        } else {
            ReadbackAction::None
        }
    }
}
