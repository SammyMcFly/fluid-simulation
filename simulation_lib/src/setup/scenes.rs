use cgmath::Zero;
use serde::Deserialize;
use nalgebra::Vector3;
use super::super::sph::particle::Initializable;

use super::super::sph::particle::{Particle3D, BoundaryParticle3D};
#[cfg(feature = "springs")]
use crate::sph::spring::Spring;
use super::Scene;



#[derive(Debug, Deserialize)]
pub struct NoLidCube {
    pub particles: ParticleSetup,
    pub boundary_particles: BoundaryParticleSetup,
    // pub springs: SpringConfig,
}

#[derive(Debug, Deserialize)]
pub struct ParticleSetup {
    pub n_particles_x: usize,
    pub n_particles_y: usize,
    pub n_particles_z: usize,
    pub x_offset: f64,
    pub y_offset: f64,
    pub z_offset: f64,
    pub particle_spacing: f64,
}

#[derive(Debug, Deserialize)]
pub struct BoundaryParticleSetup {
    pub n_floor_particles_x: usize,
    pub n_floor_particles_y: usize,
    pub n_floor_particles_z: usize,
    pub wall_height: usize,
    pub x_wall_thickness: usize,
    pub y_wall_thickness: usize,
    pub x_offset: f64,
    pub y_offset: f64,
    pub z_offset: f64,
    pub particle_spacing: f64,
}

// #[derive(Debug, Deserialize)]
// pub struct SpringConfig {

// }

impl Scene for NoLidCube {
    fn get_boundary(&self, _: f64) -> Vec<BoundaryParticle3D> {
        // init boundary particles
        let mut boundary_particles = vec![];
        // init floor
        for i in 0..self.boundary_particles.n_floor_particles_x {
            for j in 0..self.boundary_particles.n_floor_particles_y {
                for k in 0..self.boundary_particles.n_floor_particles_z {
                    let x = (i as f64)*self.boundary_particles.particle_spacing+self.boundary_particles.x_offset;
                    let y = (j as f64)*self.boundary_particles.particle_spacing+self.boundary_particles.y_offset;
                    let z = (k as f64)*self.boundary_particles.particle_spacing+self.boundary_particles.z_offset;
                    let boundary_particle = BoundaryParticle3D::new(
                        [Vector3::new(x, y, z), Vector3::zeros()],
                        Vector3::zeros(),
                        0., // still needs to be initialized
                        );
                    boundary_particles.push(boundary_particle);
                }
            }
        }
        // init walls ()
        for i in 0..self.boundary_particles.n_floor_particles_x {
            for j in 0..self.boundary_particles.n_floor_particles_y {
                for k in self.boundary_particles.n_floor_particles_z..(self.boundary_particles.n_floor_particles_z+self.boundary_particles.wall_height) {
                    let x = (i as f64)*self.boundary_particles.particle_spacing+self.boundary_particles.x_offset;
                    let y = (j as f64)*self.boundary_particles.particle_spacing+self.boundary_particles.y_offset;
                    let z = (k as f64)*self.boundary_particles.particle_spacing+self.boundary_particles.z_offset;
                    // filter for particles at the edge of the floor
                    if x < (self.boundary_particles.x_offset + self.boundary_particles.x_wall_thickness as f64 * self.boundary_particles.particle_spacing)
                            || x > (self.boundary_particles.x_offset + ((self.boundary_particles.n_floor_particles_x as f64 - 1.) - self.boundary_particles.x_wall_thickness as f64) * self.boundary_particles.particle_spacing)
                            || y < (self.boundary_particles.y_offset + self.boundary_particles.y_wall_thickness as f64 * self.boundary_particles.particle_spacing)
                            || y > (self.boundary_particles.y_offset + ((self.boundary_particles.n_floor_particles_y as f64 - 1.) - self.boundary_particles.y_wall_thickness as f64) * self.boundary_particles.particle_spacing) {
                        let boundary_particle = BoundaryParticle3D::new(
                            [Vector3::new(x, y, z), Vector3::zeros()],
                            Vector3::zero(),
                            0., // still needs to be initialized
                        );
                        boundary_particles.push(boundary_particle);
                    }
                }
            }
        }
        boundary_particles
    }

    fn get_fluid(&self, rest_density: f64, rest_density_grid_spacing: f64) -> Vec<Particle3D> {
        // init particles
        let mut particles = vec![];
        for i in 0..self.particles.n_particles_x {
            for j in 0..self.particles.n_particles_y {
                for k in 0..self.particles.n_particles_z {
                    // do shift every second level
                    let shift = if k % 2 == 1 {
                        self.particles.particle_spacing/2.
                    } else {
                        0.
                    };
                    let x = (i as f64)*self.particles.particle_spacing+self.particles.x_offset+shift;
                    let y = (j as f64)*self.particles.particle_spacing+self.particles.y_offset;
                    let z = (k as f64)*self.particles.particle_spacing+self.particles.z_offset;

                    let mass = rest_density*rest_density_grid_spacing.powi(3);
                    let particle = Particle3D::new(
                        [Vector3::new(x, y, z), Vector3::new(x, y, z)],
                        Vector3::new(0., 0., 0.),
                        mass);
                    particles.push(particle);
                }
            }
        }
        particles
    }

    #[cfg(feature = "springs")]
    fn get_springs(&self) -> Vec<Spring> {
        // init springs (Note: Consider not disabling particles)
        let mut springs = vec![];
        // add springs configured in config file here
        springs
    }

    fn calc_fluid_depth(&self, rest_density_grid_spacing: f64) -> f64 {
        // estimate fluid depth
        let floor_area = (self.boundary_particles.n_floor_particles_x as f64 - 2.*self.boundary_particles.x_wall_thickness as f64)
            *(self.boundary_particles.n_floor_particles_y as f64 - 2.*self.boundary_particles.y_wall_thickness as f64)
            *self.boundary_particles.particle_spacing.powi(2);
        let total_particle_volume = self.particles.n_particles_x as f64
            *self.particles.n_particles_y as f64
            *self.particles.n_particles_z as f64
            *rest_density_grid_spacing.powi(3);
        total_particle_volume/floor_area/rest_density_grid_spacing
    }
}



#[derive(Debug, Deserialize)]
pub struct Spiral {
    base: [f64; 3],
    fluid_particles: SpiralFluid,
    boundary: SpiralBoundary,
}

#[derive(Debug, Deserialize)]
struct SpiralFluid {
    number_of_particles: u64,
}

#[derive(Debug, Deserialize)]
struct SpiralBoundary {
    length: u64,
    width: u64,
    minimum_heigth: u64,
    inner_width1: u64,
    inner_length1: u64,
    inner_length2: u64,
    whole_height: u64,
    inner_length3: u64,
    barrier_height: u64,
}

impl Scene for Spiral {
    fn get_boundary(&self, rest_density_grid_spacing: f64) -> Vec<BoundaryParticle3D> {
        let fluid_body_width = self.boundary.inner_width1-2;
        let fluid_body_length = self.boundary.inner_length1-1;
        let fluid_body_height = self.fluid_particles.number_of_particles/fluid_body_width/fluid_body_length;
        let height = self.boundary.minimum_heigth.max(fluid_body_height+3);
        // init boundary particles
        let mut boundary_particles = vec![];
        // init floor and lid
        boundary_particles.extend(Square::new(
            Vector3::from(self.base),
            Vector3::new(1., 0., 0.),
            Vector3::new(0., 1., 0.),
            self.boundary.length,
            self.boundary.width
        )
        .fetch::<BoundaryParticle3D>(0.0, rest_density_grid_spacing));
        boundary_particles.extend(Square::new(
            Vector3::from(self.base)+Vector3::new(0.0, 0.0, ((height-1) as f64)*rest_density_grid_spacing),
            Vector3::new(1., 0., 0.),
            Vector3::new(0., 1., 0.),
            self.boundary.length,
            self.boundary.width,
        )
        .fetch::<BoundaryParticle3D>(0.0, rest_density_grid_spacing));
        // init walls
        boundary_particles.extend(Square::new(
            Vector3::from(self.base)+Vector3::new(0.0, 0.0, rest_density_grid_spacing),
            Vector3::new(1., 0., 0.),
            Vector3::new(0., 0., 1.),
            self.boundary.length-1,
            height-1,
        )
        .fetch::<BoundaryParticle3D>(0.0, rest_density_grid_spacing));
        boundary_particles.extend(Square::new(
            Vector3::from(self.base)+Vector3::new(0.0, 0.0, rest_density_grid_spacing)
                +Vector3::new(((self.boundary.length-1) as f64)*rest_density_grid_spacing, 0., 0.),
            Vector3::new(0., 1., 0.),
            Vector3::new(0., 0., 1.),
            self.boundary.length-1,
            height-1,
        )
        .fetch::<BoundaryParticle3D>(0.0, rest_density_grid_spacing));
        boundary_particles.extend(Square::new(
            Vector3::from(self.base)+Vector3::new(0.0, 0.0, rest_density_grid_spacing)
                +Vector3::new(((self.boundary.length-1) as f64)*rest_density_grid_spacing, 0., 0.)
                +Vector3::new(0., ((self.boundary.width-1) as f64)*rest_density_grid_spacing, 0.),
            Vector3::new(-1., 0., 0.),
            Vector3::new(0., 0., 1.),
            self.boundary.length-1,
            height-1,
        )
        .fetch::<BoundaryParticle3D>(0.0, rest_density_grid_spacing));
        boundary_particles.extend(Square::new(
            Vector3::from(self.base)+Vector3::new(0.0, 0.0, rest_density_grid_spacing)
                +Vector3::new(0., ((self.boundary.width-1) as f64)*rest_density_grid_spacing, 0.),
            Vector3::new(0., -1., 0.),
            Vector3::new(0., 0., 1.),
            self.boundary.length-1,
            height-1,
        )
        .fetch::<BoundaryParticle3D>(0.0, rest_density_grid_spacing));
        // init inner walls
        boundary_particles.extend(Square::new(
            Vector3::from(self.base)+Vector3::new(0.0, 0.0, rest_density_grid_spacing)
                +Vector3::new(0.0, rest_density_grid_spacing, 0.0)
                +Vector3::new(((self.boundary.inner_width1-1) as f64)*rest_density_grid_spacing, 0., 0.),
            Vector3::new(0., 1., 0.),
            Vector3::new(0., 0., 1.),
            self.boundary.inner_length1-2,
            height-2,
        )
        .fetch::<BoundaryParticle3D>(0.0, rest_density_grid_spacing));
        boundary_particles.extend(Square::new(
            Vector3::from(self.base)+Vector3::new(0.0, 0.0, rest_density_grid_spacing)
                +Vector3::new(((self.boundary.inner_width1-1) as f64)*rest_density_grid_spacing, 0., 0.)
                +Vector3::new(0.,((self.boundary.inner_length1-1) as f64)*rest_density_grid_spacing, 0.),
            Vector3::new(1., 0., 0.),
            Vector3::new(0., 0., 1.),
            self.boundary.inner_length2-1,
            height-2,
        )
        .fetch::<BoundaryParticle3D>(0.0, rest_density_grid_spacing));
        boundary_particles.extend(Square::new(
            Vector3::from(self.base)+Vector3::new(0.0, 0.0, rest_density_grid_spacing)
                +Vector3::new(((self.boundary.inner_width1-1) as f64)*rest_density_grid_spacing, 0., 0.)
                +Vector3::new(0., ((self.boundary.inner_length1-1) as f64)*rest_density_grid_spacing, 0.)
                +Vector3::new(((self.boundary.inner_length2-1) as f64)*rest_density_grid_spacing, 0., 0.),
            Vector3::new(0., -1., 0.),
            Vector3::new(0., 0., 1.),
            self.boundary.inner_length3,
            height-2,
        )
        .fetch::<BoundaryParticle3D>(0.0, rest_density_grid_spacing));
        // init obstacles
        boundary_particles.extend(Square::new(
            Vector3::from(self.base)+Vector3::new(0.0, 0.0, rest_density_grid_spacing)
                +Vector3::new(((self.boundary.inner_width1-1+self.boundary.inner_length2/2) as f64)*rest_density_grid_spacing, 0., 0.)
                +Vector3::new(0., ((self.boundary.inner_length1) as f64)*rest_density_grid_spacing, 0.)
                +Vector3::new(0., 0., (self.boundary.whole_height as f64)*rest_density_grid_spacing),
            Vector3::new(0., 1., 0.),
            Vector3::new(0., 0., 1.),
            self.boundary.width-self.boundary.inner_length1-1,
            height-self.boundary.whole_height-2,
        )
        .fetch::<BoundaryParticle3D>(0.0, rest_density_grid_spacing));
        boundary_particles.extend(Square::new(
            Vector3::from(self.base)+Vector3::new(0.0, 0.0, rest_density_grid_spacing)
                +Vector3::new(((self.boundary.inner_width1-1) as f64)*rest_density_grid_spacing, 0., 0.)
                +Vector3::new(0., ((self.boundary.inner_length1-1) as f64)*rest_density_grid_spacing, 0.)
                +Vector3::new(((self.boundary.inner_length2) as f64)*rest_density_grid_spacing, 0., 0.)
                +Vector3::new(0., -((self.boundary.inner_length3-1) as f64)*rest_density_grid_spacing, 0.),
            Vector3::new(1., 0., 0.),
            Vector3::new(0., 0., 1.),
            self.boundary.length-self.boundary.inner_width1-self.boundary.inner_length2,
            self.boundary.barrier_height,
        )
        .fetch::<BoundaryParticle3D>(0.0, rest_density_grid_spacing));
        boundary_particles
    }

    fn get_fluid(&self, rest_density: f64, rest_density_grid_spacing: f64) -> Vec<Particle3D> {
        let fluid_body_width = self.boundary.inner_width1-3;
        let fluid_body_length = self.boundary.inner_length1-1;
        let fluid_body_base = Vector3::from(self.base)+Vector3::new(1.1*rest_density_grid_spacing, 1.1*rest_density_grid_spacing, 1.1*rest_density_grid_spacing);

        let fluid_body_height = self.fluid_particles.number_of_particles/fluid_body_width/fluid_body_length;
        // init particles
        Cube::new(
            fluid_body_base,
            Vector3::new(1., 0., 0.),
            Vector3::new(0., 1., 0.),
            Vector3::new(0., 0., 1.),
            fluid_body_width,
            fluid_body_length,
            fluid_body_height,
        ).fetch::<Particle3D>(rest_density, rest_density_grid_spacing)
    }

    #[cfg(feature = "springs")]
    fn get_springs(&self) -> Vec<Spring> {
        // init springs (Note: Consider not disabling particles)
        let mut springs = vec![];
        // add springs configured in config file here
        springs
    }

    fn calc_fluid_depth(&self, rest_density_grid_spacing: f64) -> f64 {
        // estimate fluid depth
        let floor_area = 1.0;
        let total_particle_volume = self.fluid_particles.number_of_particles as f64
            *rest_density_grid_spacing.powi(3);
        total_particle_volume/floor_area/rest_density_grid_spacing
    }
}


struct Line {
    base: Vector3<f64>,
    direction: Vector3<f64>,
    length: u64,
}

impl Line {
    fn new(
        base: Vector3<f64>,
        direction: Vector3<f64>,
        length: u64,
    ) -> Self {
        Self { base, direction, length, }
    }

    fn fetch<T: Initializable>(&self, rest_density: f64, rest_density_grid_spacing: f64) -> Vec<T> {
        let mut particles = vec![];
        for i in 0..self.length {
            particles.push(T::new(
                [self.base+self.direction*((i as f64)*rest_density_grid_spacing), Vector3::zeros()],
                Vector3::zeros(),
                rest_density*rest_density_grid_spacing.powi(3),
            ));
        }
        particles
    }
}

struct Square {
    base: Vector3<f64>,
    direction1: Vector3<f64>,
    direction2: Vector3<f64>,
    length1: u64,
    length2: u64,
}

impl Square {
    fn new(
        base: Vector3<f64>,
        direction1: Vector3<f64>,
        direction2: Vector3<f64>,
        length1: u64,
        length2: u64,
    ) -> Self {
        Self { base, direction1, direction2, length1, length2 }
    }

    fn fetch<T: Initializable>(&self, rest_density: f64, rest_density_grid_spacing: f64) -> Vec<T> {
        let mut particles = vec![];
        for i in 0..self.length2 {
            particles.extend(Line::new(
                self.base+self.direction2*((i as f64)*rest_density_grid_spacing),
                self.direction1,
                self.length1)
                .fetch::<T>(rest_density, rest_density_grid_spacing));
        }
        particles
    }
}

struct Cube {
    base: Vector3<f64>,
    direction1: Vector3<f64>,
    direction2: Vector3<f64>,
    direction3: Vector3<f64>,
    length1: u64,
    length2: u64,
    length3: u64,
}

impl Cube {
    fn new(
        base: Vector3<f64>,
        direction1: Vector3<f64>,
        direction2: Vector3<f64>,
        direction3: Vector3<f64>,
        length1: u64,
        length2: u64,
        length3: u64,
    ) -> Self {
        Self { base, direction1, direction2, direction3, length1, length2, length3, }
    }

    fn fetch<T: Initializable>(&self, rest_density: f64, rest_density_grid_spacing: f64) -> Vec<T> {
        let mut particles = vec![];
        for i in 0..self.length3 {
            // do shift every second level
            let shift = if i % 2 == 1 {
                Vector3::new(rest_density_grid_spacing/2., 0., 0.)
            } else {
                Vector3::zeros()
            };
            particles.extend(Square::new(
                self.base+shift+self.direction3*((i as f64)*rest_density_grid_spacing),
                self.direction1,
                self.direction2,
                self.length1,
                self.length2)
                .fetch::<T>(rest_density, rest_density_grid_spacing));
        }
        particles
    }
}