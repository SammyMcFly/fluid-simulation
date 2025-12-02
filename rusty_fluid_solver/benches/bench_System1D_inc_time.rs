//! only works in libraries: rename main.rs to lib.rs


// use rusty_fluid_solver::physics::{System1D, Vertex1D, PropagationMethod};
// use criterion::{criterion_group, criterion_main, Criterion};

// pub fn criterion_benchmark(c: &mut Criterion) {
//     let time_inc = 0.01;
//     let x1 = Vertex1D::new(
//         [-5.0, -5.0],
//         0.0,
//         0.0,
//         1.0,
//         [1.0,0.0,0.0]);
//     let x2  = Vertex1D::new(
//         [5.0, 5.0],
//         0.0,
//         0.0,
//         1.0,
//         [0.0,0.0,1.0]);

//     let mut system = System1D::new(
//         vec![x1, x2],
//         vec![vec![(1, 0.1, 8.0)], vec![(0, 0.1, 8.0)]],
//         time_inc);

//     c.bench_function("bench System1D::inc_time explicit Euler with circular buffer", |b| b.iter(|| system.inc_time(PropagationMethod::ExplicitEuler)));
// }

// criterion_group!(benches, criterion_benchmark);
// criterion_main!(benches);
