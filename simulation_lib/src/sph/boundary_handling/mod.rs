/// Boundary handling module
use gauss_quad::GaussLegendre;
// use approx::assert_abs_diff_eq;
use std::f64::consts::PI;

pub trait BoundaryRepresentation {
    fn initialize();
    fn add_viscosity_acceleration();
    fn add_pressure_acceleration();
}


// let integrator = GaussLegendre::new(30.try_into()?);

// let integral = integrator.integrate(0.0, 1.0, |x| {x * x});

// /// Integriert f(x, y, z) über eine Kugel mit Radius R.
// /// Verwendet n_r × n_mu × n_phi Quadraturpunkte.
// fn integrate_sphere_volume<F>(
//     f: F,
//     radius: f64,
//     n_r: usize,
//     n_mu: usize,
//     n_phi: usize,
// ) -> f64
// where
//     F: Fn(f64, f64, f64) -> f64,
// {
//     let quad_r = GaussLegendre::new(n_r.try_into().unwrap());
//     let quad_mu = GaussLegendre::new(n_mu.try_into().unwrap());
//     let quad_phi = GaussLegendre::new(n_phi.try_into().unwrap());

//     let (nodes_r, weights_r) = (quad_r.nodes(), quad_r.weights());
//     let (nodes_mu, weights_mu) = (quad_mu.nodes(), quad_mu.weights());
//     let (nodes_phi, weights_phi) = (quad_phi.nodes(), quad_phi.weights());

//     // Transformation: r ∈ [0, R], μ ∈ [-1, 1], φ ∈ [0, 2π]
//     let half_r = 0.5 * radius;
//     let mid_r = 0.5 * radius;
//     let half_phi = PI; // (2π - 0) / 2
//     let mid_phi = PI;  // (2π + 0) / 2

//     let mut sum = 0.0;

//     for i in 0..n_r {
//         let r = half_r * nodes_r[i] + mid_r;
//         let w_r = half_r * weights_r[i];

//         for j in 0..n_mu {
//             // μ ∈ [-1, 1] → keine Transformation nötig
//             let mu = nodes_mu[j];
//             let w_mu = weights_mu[j];

//             let theta = mu.acos();
//             let sin_theta = (1.0 - mu * mu).sqrt();

//             for k in 0..n_phi {
//                 let phi = half_phi * nodes_phi[k] + mid_phi;
//                 let w_phi = half_phi * weights_phi[k];

//                 // Kartesische Koordinaten
//                 let x = r * sin_theta * phi.cos();
//                 let y = r * sin_theta * phi.sin();
//                 let z = r * mu; // r * cos(θ)

//                 // Jacobian: r² (sin θ ist durch μ-Substitution absorbiert)
//                 let jacobian = r * r;

//                 sum += w_r * w_mu * w_phi * jacobian * f(x, y, z);
//             }
//         }
//     }

//     sum
// }
