//! FluidViewport – the shader widget program.
//! Connects camera/light/scene state to the rendering primitive.
//! Handles mouse/keyboard interaction within the viewport area.
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cosmic::iced::Rectangle;
use cosmic::iced::event::Event;
use cosmic::iced::keyboard;
use cosmic::iced::mouse;
use cosmic::iced::widget::shader;
use crossbeam::channel::Sender;

use crate::camera::{CameraState, Key};
use crate::lighting::LightState;
use crate::pipeline::ScreenshotCommand;
use crate::primitive::{SceneData, SimulationFrame};
use crate::primitive::{ScreenshotRequest, ScreenshotTarget};

// ─── Messages from Viewport ──────────────────────────────────

/// Messages that the viewport emits to the application.
/// Generic over the app's Message type.
#[derive(Debug, Clone)]
pub enum ViewportEvent {
    /// Camera was rotated by mouse drag
    CameraRotated { dx: f32, dy: f32 },
    /// Camera was zoomed by scroll
    CameraScrolled { delta: f32 },
    /// A camera movement key was pressed/released
    CameraKey { key: Key, pressed: bool },
    /// The viewport was resized
    Resized { width: u32, height: u32 },
    /// Request a redraw (camera is moving)
    RequestRedraw,
}

// ─── Viewport State (widget-local, persists between frames) ──

/// Tracks interaction state within the shader widget.
#[derive(Debug, Default)]
pub struct ViewportState {
    pub middle_mouse_pressed: bool,
    pub last_cursor_position: Option<(f32, f32)>,
    /// Whether camera is currently moving (keys held)
    pub camera_active: bool,
    pub last_size: Option<(u32, u32)>,
}

// ─── FluidViewport ───────────────────────────────────────────

/// The shader widget program. Holds all state needed to produce a FluidFrame.
pub struct SimulationViewport<W: ScreenshotCommand> {
    pub camera: CameraState,
    pub light: LightState,
    pub scene: SceneData,
    pub background_color: [f32; 4],
    /// Time of last draw (for dt calculation)
    last_draw: std::time::Instant,
    pub screenshot_request: Option<ScreenshotRequest>,
    pub worker_sender: Option<Sender<W>>,
    /// Signals that the last screenshot readback was consumed and sent to the worker.
    /// The app sets this to `false` when requesting a new screenshot,
    /// and the primitive sets it to `true` after the data is read back and dispatched.
    pub screenshot_consumed: Arc<AtomicBool>,
    next_screenshot_id: std::sync::atomic::AtomicU64,
}

impl<W: ScreenshotCommand> SimulationViewport<W> {
    pub fn new(
        camera: CameraState,
        light: LightState,
        background_color: [f32; 4],
        worker_sender: Sender<W>,
    ) -> Self {
        Self {
            camera,
            light,
            scene: SceneData::default(),
            background_color,
            last_draw: std::time::Instant::now(),
            screenshot_request: None,
            worker_sender: Some(worker_sender),
            screenshot_consumed: Arc::new(AtomicBool::new(true)),
            next_screenshot_id: std::sync::atomic::AtomicU64::new(0),
        }
    }

    pub fn set_background_color(&mut self, color: [f32; 4]) {
        self.background_color = color;
    }

    /// Called by Application::update() when new simulation data arrives
    pub fn set_scene(&mut self, scene: SceneData) {
        self.scene = scene;
    }

    /// Called by Application::update() when camera should reset
    pub fn reset_camera(&mut self) {
        self.camera.reset();
    }

    /// Called by Application::update() when light should reset
    pub fn reset_light(&mut self) {
        self.light.reset();
    }

    /// Resize projection (called on window resize)
    pub fn resize(&mut self, width: u32, height: u32) {
        self.camera.resize(width, height);
    }

    pub fn request_screenshot(&mut self, target: ScreenshotTarget) {
        let id = self.next_screenshot_id.fetch_add(1, Ordering::Relaxed);
        self.screenshot_consumed.store(false, Ordering::Release);
        self.screenshot_request = Some(ScreenshotRequest { target, id });
    }

    /// Returns true if the last screenshot has been fully captured and dispatched
    pub fn is_screenshot_done(&self) -> bool {
        self.screenshot_consumed.load(Ordering::Acquire)
    }
}

impl<W: ScreenshotCommand, Message> shader::Program<Message> for SimulationViewport<W>
where
    Message: Clone + From<ViewportEvent>,
{
    type State = ViewportState;
    type Primitive = SimulationFrame<W>;

    fn update(
        &self,
        state: &mut ViewportState,
        event: &Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<shader::Action<Message>> {
        // Track size changes
        if bounds.width > 0.0 && bounds.height > 0.0 {
            let new_size = (bounds.width as u32, bounds.height as u32);
            if state.last_size != Some(new_size) {
                state.last_size = Some(new_size);
                return Some(shader::Action::publish(Message::from(
                    ViewportEvent::Resized {
                        width: new_size.0,
                        height: new_size.1,
                    },
                )));
            }
        }

        // Only handle events when cursor is within bounds
        let cursor_in_bounds = cursor.position_over(bounds).is_some();

        match event {
            // ─── Mouse button ─────────────────────────────────
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Middle)) => {
                if cursor_in_bounds {
                    state.middle_mouse_pressed = true;
                    if let Some(pos) = cursor.position() {
                        state.last_cursor_position = Some((pos.x, pos.y));
                    }
                    return Some(shader::Action::capture());
                }
            }
            Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Middle)) => {
                if state.middle_mouse_pressed {
                    state.middle_mouse_pressed = false;
                    state.last_cursor_position = None;
                    return Some(shader::Action::capture());
                }
            }

            // ─── Mouse motion ─────────────────────────────────
            Event::Mouse(mouse::Event::CursorMoved { position }) => {
                if state.middle_mouse_pressed {
                    if let Some((last_x, last_y)) = state.last_cursor_position {
                        let dx = position.x - last_x;
                        let dy = position.y - last_y;
                        state.last_cursor_position = Some((position.x, position.y));

                        let msg = ViewportEvent::CameraRotated { dx, dy };
                        return Some(shader::Action::publish(Message::from(msg)).and_capture());
                    } else {
                        state.last_cursor_position = Some((position.x, position.y));
                    }
                }
            }

            // ─── Scroll ───────────────────────────────────────
            Event::Mouse(mouse::Event::WheelScrolled { delta }) => {
                if cursor_in_bounds {
                    let scroll = match delta {
                        mouse::ScrollDelta::Lines { y, .. } => *y * 100.0,
                        mouse::ScrollDelta::Pixels { y, .. } => *y,
                    };
                    let msg = ViewportEvent::CameraScrolled { delta: -scroll };
                    return Some(shader::Action::publish(Message::from(msg)).and_capture());
                }
            }

            // ─── Keyboard ─────────────────────────────────────
            Event::Keyboard(keyboard::Event::KeyPressed { key, .. }) => {
                if cursor_in_bounds {
                    if let Some(cam_key) = map_keyboard_key(key) {
                        state.camera_active = true;
                        let msg = ViewportEvent::CameraKey {
                            key: cam_key,
                            pressed: true,
                        };
                        return Some(shader::Action::publish(Message::from(msg)).and_capture());
                    }
                }
            }
            Event::Keyboard(keyboard::Event::KeyReleased { key, .. }) => {
                if let Some(cam_key) = map_keyboard_key(key) {
                    let msg = ViewportEvent::CameraKey {
                        key: cam_key,
                        pressed: false,
                    };
                    // Check if any movement key is still held
                    // (simplified: just mark inactive, tick will handle it)
                    return Some(shader::Action::publish(Message::from(msg)));
                }
            }

            _ => {}
        }

        None
    }

    fn draw(
        &self,
        _state: &Self::State,
        _cursor: mouse::Cursor,
        bounds: Rectangle,
    ) -> Self::Primitive {
        SimulationFrame {
            camera_uniform: self.camera.uniform,
            light_uniform: self.light.uniform,
            scene: self.scene.clone(),
            background_color: self.background_color,
            readback_request: self.screenshot_request.clone(),
            worker_sender: self.worker_sender.clone(),
            screenshot_consumed: self.screenshot_consumed.clone(),
        }
    }

    fn mouse_interaction(
        &self,
        state: &Self::State,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> mouse::Interaction {
        if state.middle_mouse_pressed {
            mouse::Interaction::Grabbing
        } else if cursor.position_over(bounds).is_some() {
            mouse::Interaction::default()
        } else {
            mouse::Interaction::default()
        }
    }
}

// ─── Key mapping ──────────────────────────────────────────────

fn map_keyboard_key(key: &keyboard::Key) -> Option<Key> {
    match key {
        keyboard::Key::Character(c) => match c.as_str() {
            "w" => Some(Key::Forward),
            "s" => Some(Key::Backward),
            "a" => Some(Key::Left),
            "d" => Some(Key::Right),
            " " => Some(Key::Up),
            _ => None,
        },
        keyboard::Key::Named(keyboard::key::Named::Shift) => Some(Key::Down),
        keyboard::Key::Named(keyboard::key::Named::ArrowUp) => Some(Key::Forward),
        keyboard::Key::Named(keyboard::key::Named::ArrowDown) => Some(Key::Backward),
        keyboard::Key::Named(keyboard::key::Named::ArrowLeft) => Some(Key::Left),
        keyboard::Key::Named(keyboard::key::Named::ArrowRight) => Some(Key::Right),
        _ => None,
    }
}
