use nalgebra::Vector3;
use std::collections::HashMap;

fn hash(x: i64, y: i64, z: i64) -> u64 {
    const P1: u64 = 73856093;
    const P2: u64 = 19349663;
    const P3: u64 = 83492791;
    (x as u64 * P1) ^ (y as u64 * P2) ^ (z as u64 * P3)
}

#[derive(Debug)]
struct Particle {
    position: Vector3<f64>,
}

type UniformGridCell = Vector3<i64>;

fn get_cell(particle: &Particle, kernel_support: f64) -> UniformGridCell {
    UniformGridCell::new(
        (particle.position.x / kernel_support).floor() as i64,
        (particle.position.y / kernel_support).floor() as i64,
        (particle.position.z / kernel_support).floor() as i64,
    )
}

fn get_neighbors(
        particle: &Particle,
        grid: &HashMap<u64, Vec<usize>>,
        kernel_support: f64,
        cell: fn(&Particle, f64) -> UniformGridCell,
        // distance: fn(&Particle, &Particle) -> f64,
) -> Vec<usize> {
    let mut neighbors = Vec::new();
    let cell = cell(particle, kernel_support);

    for dx in -2..=2 {
        for dy in -2..=2 {
            for dz in -2..=2 {
                let neighbor_cell = (cell.x + dx, cell.y + dy, cell.z + dz);
                let hash = hash(neighbor_cell.0, neighbor_cell.1, neighbor_cell.2);
                if let Some(indices) = grid.get(&hash) {
                    for &j in indices {
                        // Distance check
                        neighbors.push(j);
                    }
                }
            }
        }
    }
    neighbors
}

fn get_distance(particle1: &Particle, particle2: &Particle) -> f64 {
    ((particle1.position.x-particle2.position.x).powi(2)
        +(particle1.position.y-particle2.position.y).powi(2)
        +(particle1.position.z-particle2.position.z).powi(2)).sqrt()
}

/// Direction from particle1 towards particle2
fn get_direction(particle1: &Particle, particle2: &Particle) -> Vector3<f64> {
    particle2.position-particle1.position
}

/// Cubic spline kernel function
/// 
/// Control flow is ordered in a way to minimize calculations
/// (this assumes that normalized_distance >= 2. for many function calls)
/// Since the cubic spline function is continuous, the points:
/// normalized_distance == 2. and normalized_distance == 1. can be chosen
/// to be calculated in the earlier, simpler and thus faster branch
fn cubic_spline_3d(distance: f64, kernel_support: f64) -> f64 {
    let normalized_distance = distance/kernel_support;
    if normalized_distance >= 2. {
        0.
    } else if normalized_distance >= 1. {
        let prefactor = 1./4./std::f64::consts::PI/kernel_support.powi(3);
        prefactor*(2.-normalized_distance).powi(3)
    } else {
        let prefactor = 1./4./std::f64::consts::PI/kernel_support.powi(3);
        prefactor*((2.-normalized_distance).powi(3)-4.*(1.-normalized_distance).powi(3))
    }
}

fn cubic_spline_3d_gradient(distance: f64, kernel_support: f64, direction: Vector3<f64>) -> Vector3<f64> {
    let normalized_distance = distance/kernel_support;
    if normalized_distance >= 2. {
        Vector3::zeros()
    } else if normalized_distance >= 1. {
        let prefactor = 1./4./std::f64::consts::PI/kernel_support.powi(5);
        direction/normalized_distance*prefactor*(-3.*(2.-normalized_distance).powi(2))
    } else if normalized_distance > 0. {
        let prefactor = 1./4./std::f64::consts::PI/kernel_support.powi(5);
        direction/normalized_distance*prefactor*(-3.*(2.-normalized_distance).powi(2)+12.*(1.-normalized_distance).powi(2))
    } else {
        Vector3::zeros()
    }
}

fn main() {
    let kernel_support = 1.0; //h
    let particle_spacing = 0.9;
    let particle_mass = 3.;
    let mut grid: HashMap<u64, Vec<usize>> = HashMap::new();
    let mut particles: Vec<Particle> = vec![];
    for i in 0..10 {
        for j in 0..10 {
            for k in 0..10 {
                particles.push(Particle { position: Vector3::new(
                    (i as f64)*particle_spacing, 
                    (j as f64)*particle_spacing, 
                    (k as f64)*particle_spacing
                ) });
            }
        }
    }
    for (i, particle) in particles.iter().enumerate() {
        let cell = get_cell(particle, kernel_support);
        let cell_hash = hash(cell.x, cell.y, cell.z);
        grid.entry(cell_hash).or_default().push(i);
    }

    let ref_particle = Particle { position: Vector3::new(2.5, 2.5, 2.5) };
    let neighbors_candidates = get_neighbors(&ref_particle, &grid, kernel_support, get_cell);
    let mut neighbors: Vec<usize> = vec![];
    let mut count_real_neighbors = 0;
    for n in &neighbors_candidates {
        if get_distance(&particles[*n], &ref_particle) < 2.*kernel_support {
            // println!("{:?}", particles[*n]);
            count_real_neighbors += 1;
            neighbors.push(*n);
        }
    }
    println!("Neighbors within two grid cells: {}", &neighbors_candidates.len());
    println!("Neighbors within distance 2.0: {}", &count_real_neighbors);

    let mut sum_over_kernel_values = 0.;
    let mut sum_over_kernel_gradient_values = Vector3::zeros();
    for n in &neighbors {
        let distance = get_distance(&ref_particle, &particles[*n]);
        let direction = get_direction(&ref_particle, &particles[*n]);
        // println!("{}, {}", &distance, &direction);
        // println!("{}", cubic_spline_3d(distance, kernel_support));
        // println!("{}", cubic_spline_3d_gradient(distance, kernel_support, direction));
        sum_over_kernel_values += cubic_spline_3d(distance, kernel_support);
        sum_over_kernel_gradient_values += cubic_spline_3d_gradient(distance, kernel_support, direction);
    }
    println!("Sum over kernel values: 1/area = {}", &sum_over_kernel_values);
    println!("Sum over kernel values times mass: density = {}", particle_mass*sum_over_kernel_values);
    println!("sum over kernel gradient values: {}", &sum_over_kernel_gradient_values);
}

// Optimization Tips

//     Avoid allocations: Pre-allocate cell lists and neighbor lists if particle count is known.

//     Memory layout: Use SoA (Structure of Arrays) for better cache efficiency.

//     Parallelism: Build grid and query neighbors in parallel (e.g., with Rayon in Rust).

//     Cell reuse: Clear and reuse grid each timestep to avoid reallocation.

//     Limit checks: Only include particles within hh radius after checking distances.
