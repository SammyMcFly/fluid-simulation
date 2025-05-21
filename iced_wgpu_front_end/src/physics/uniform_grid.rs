//! Module provides the necessary entities for a uniform grid implementation
//!
use nalgebra::Vector3;
use std::collections::HashMap;

use super::particle::{ParticleQ3, GridParticle};


type UniformGridCell = Vector3<i64>;


#[derive(Debug, Clone)]
pub struct UniformGrid {
    hash_map: HashMap<u64, Vec<usize>>,
    edge_length: f64,
}

impl UniformGrid {
    pub fn new(edge_length: f64) -> Self {
        let hash_map: HashMap<u64, Vec<usize>> = HashMap::new();
        Self {
            hash_map,
            edge_length,
        }

    }

    pub fn populate(&mut self, particles: &[super::particle::Particle3D]) {
        for (i, particle) in particles.iter().enumerate() {
            let cell = self.get_cell(&particle.pos().now());
            let cell_hash = Self::hash(cell.x, cell.y, cell.z);
            self.hash_map.entry(cell_hash).or_default().push(i);
        }
    }

    pub fn populate_boundary_particles(&mut self, particles: &[super::particle::BoundaryParticle3D]) {
        for (i, particle) in particles.iter().enumerate() {
            let cell = self.get_cell(&particle.pos());
            let cell_hash = Self::hash(cell.x, cell.y, cell.z);
            self.hash_map.entry(cell_hash).or_default().push(i);
        }
    }

    fn hash(x: i64, y: i64, z: i64) -> u64 {
        const P1: u64 = 73856093;
        const P2: u64 = 19349663;
        const P3: u64 = 83492791;
        (x as u64 * P1) ^ (y as u64 * P2) ^ (z as u64 * P3)
    }

    fn get_cell(&self, position: &Vector3<f64>) -> UniformGridCell {
        UniformGridCell::new(
            (position.x / self.edge_length).floor() as i64,
            (position.y / self.edge_length).floor() as i64,
            (position.z / self.edge_length).floor() as i64,
        )
    }

    pub fn get_particles_in_kernel_range(
        &self,
        position: &Vector3<f64>,
        particles: &[impl GridParticle],
        // distance: fn(&Particle, &Particle) -> f64,
    ) -> Vec<usize> {
        let mut particles_in_kernel_range = Vec::new();
        let cell = self.get_cell(position);

        for dx in -2..=2 {
            for dy in -2..=2 {
                for dz in -2..=2 {
                    let neighbor_cell = (cell.x + dx, cell.y + dy, cell.z + dz);
                    let hash = Self::hash(neighbor_cell.0, neighbor_cell.1, neighbor_cell.2);
                    if let Some(indices) = self.hash_map.get(&hash) {
                        for &j in indices {
                            // Distance check
                            if particles[j].get_distance(position) < 2.*self.edge_length {
                                particles_in_kernel_range.push(j);
                            }
                        }
                    }
                }
            }
        }
        particles_in_kernel_range
    }

    pub fn clear(&mut self) {
        self.hash_map.clear();
    }
}


