use std::collections::VecDeque;


#[derive(Debug, Clone, Default)]
pub struct IntermediateQueue<T> {
    length: u32,
    queue: VecDeque<T>,
}

impl<T> IntermediateQueue<T> {
    pub fn push_back(&mut self, value: T) {
        self.length += 1;
        self.queue.push_back(value);
    }
    pub fn pop_front(&mut self) -> Option<T> {
        self.length -= 1;
        self.queue.pop_front()
    }
    pub fn clear(&mut self) {
        self.length = 0;
        self.queue.clear();
    }
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
    pub fn len(&self) -> u32 {
        self.length
    }
}


#[derive(Debug, Clone)]
pub struct Instance {
    pub position: nalgebra::Vector3<f32>,
    // rotation: cgmath::Quaternion<f32>,
    pub color: [f32; 3],
}

impl Default for Instance {
    fn default() -> Self {
        Self {
            position: nalgebra::Vector3::new(0.0, 0.0, 0.0),
            color: [0.0, 0.0, 0.0],
        }
    }
}



#[derive(Debug, Clone)]
pub struct IntermediateControls {
    connection_terminated: bool,
    reset_requested: bool,
    saving_requested: bool,
    time_inc: f32,
    particle_size: f32,
    rest_density: f32,
    light_position: [f32; 3],

    pub particle_positions: IntermediateQueue<Vec<Instance>>,
    pub boundary_particle_positions: IntermediateQueue<Vec<Instance>>,
    pub average_density: IntermediateQueue<f32>,
}

impl Default for IntermediateControls {
    fn default() -> Self {
        Self {
            connection_terminated: false,
            reset_requested: false,
            saving_requested: false,
            time_inc: 0.01,
            particle_size: 1.0,
            rest_density: 0.,
            light_position: [ 2.0, 20.0, 2.0 ],
            particle_positions: IntermediateQueue::default(),
            boundary_particle_positions: IntermediateQueue::default(),
            average_density: IntermediateQueue::default(),
        }
    }
}

impl IntermediateControls {

    pub fn is_connection_terminated(&self) -> bool {
        self.connection_terminated
    }
    pub fn terminate_connection(&mut self) {
        self.connection_terminated = true;
    }
    pub fn is_reset_requested(&self) -> bool {
        self.reset_requested
    }
    pub fn request_reset(&mut self) {
        self.reset_requested = true;
    }
    pub fn reset_done(&mut self) {
        self.reset_requested = false;
    }
    pub fn is_saving_requested(&self) -> bool {
        self.saving_requested
    }
    pub fn request_saving(&mut self) {
        self.saving_requested = true;
    }
    pub fn saving_done(&mut self) {
        self.saving_requested = false;
    }
    pub fn time_inc(&self) -> f32 {
        self.time_inc
    }
    pub fn set_time_inc(&mut self, time_inc: f32) {
        self.time_inc = time_inc;
    }
    pub fn particle_size(&self) -> f32 {
        self.particle_size
    }
    pub fn set_particle_size(&mut self, particle_size: f32) {
        self.particle_size = particle_size;
    }
    pub fn light_position(&self) -> [f32; 3] {
        self.light_position
    }
    pub fn set_light_position(&mut self, position: [f32; 3]) {
        self.light_position = position;
    }
    pub fn get_rest_density(&self) -> f32 {
        self.rest_density
    }
    pub fn set_rest_density(&mut self, density: f32) {
        self.rest_density = density;
    }
}


#[cfg(test)]
mod tests {
    // use super::*;


}
