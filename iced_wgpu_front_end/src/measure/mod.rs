use std::collections::VecDeque;

use serde::Serialize;


#[derive(Debug, Clone, Default, Serialize)]
pub struct Measurement {
    pub time: f64,
    pub density: f64,
    pub kinetic_energy: f64,
    pub stiffness: f64,
    pub viscosity: f64,
    pub fluid_depth: f64,
}


#[derive(Debug, Clone, Default)]
pub struct MeasurementSeries {
    length: u32,
    queue: VecDeque<Measurement>,
}

impl MeasurementSeries {
    pub fn push_back(&mut self, value: Measurement) {
        self.length += 1;
        self.queue.push_back(value);
    }
    pub fn pop_front(&mut self) -> Option<Measurement> {
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
    pub fn save(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file = std::fs::File::create(path)?;
        let mut wtr = csv::Writer::from_writer(file);

        for measurement in &self.queue {
            wtr.serialize(measurement)?;
        }

        wtr.flush()?;
        Ok(())
    }
}