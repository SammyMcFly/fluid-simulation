//! Acceleration module
use crate::for_each;
use crate::iteration::for_each_collect;
use crate::neighbor_search::NeighborList;
use crate::sph::SystemParameters;
use crate::sph::boundary_handling::BoundaryHandling;
use crate::sph::boundary_handling::ForceOntoBoundary;
use crate::sph::boundary_handling::RequestMode;
use crate::sph::fluid::Fluid;
use crate::sph::kernel::KernelFn;
use crate::sph::vector;

use nalgebra::Vector3;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// reset acceleration, i. e. set it to 0.
pub fn reset_acceleration(fluid: &mut Fluid) {
    for_each!(
        mut [fluid.acceleration],
        ref [],
        |_id, id_acceleration| {
            *id_acceleration = Vector3::zeros();
        }
    );
}

/// Add gravity acceleration to all not boundary particles
pub fn add_gravity_acceleration<B: BoundaryHandling>(fluid: &mut Fluid, boundary: &mut B) {
    let strength_of_gravity = 9.81;
    // // gravitate downwards
    // for_each!(
    //     mut [fluid.acceleration],
    //     ref [],
    //     |_id, id_acceleration| {
    //         *id_acceleration +=  Vector3::new(0.0, 0.0, -strength_of_gravity);
    //     }
    // );
    // for b in boundary.iter_mut() {
    //     b.add_acceleration(Vector3::new(0.0, 0.0, -strength_of_gravity));
    // }

    // gravitate around point
    //
    /// Small regularization term (relative to the squared distance) to avoid a
    /// singular/unstable acceleration for particles very close to the
    /// gravitation center — same style of softening already used for the
    /// denominator in `add_viscosity_acceleration`.
    const GRAVITATION_SOFTENING: f64 = 1e-4;
    for_each!(
        mut [fluid.acceleration],
        ref [position = fluid.position],
        |id, id_acceleration| {
            use nalgebra::Point3;
            let gravitation_center = Point3::new(0.0, 0.0, 0.0);
            let direction = vector(&position[id], &gravitation_center);
            let dist = (direction.norm_squared() + GRAVITATION_SOFTENING).sqrt();
            *id_acceleration += strength_of_gravity * direction / dist;
        }
    );
    for b in boundary.iter_mut() {
        use nalgebra::Point3;
        let gravitation_center = Point3::new(0.0, 0.0, 0.0);
        if let Some(cm) = b.center_of_mass() {
            let direction = vector(&cm, &gravitation_center);
            let dist = (direction.norm_squared() + GRAVITATION_SOFTENING).sqrt();
            b.add_acceleration(strength_of_gravity * direction / dist);
        }
    }
}

/// Calculate viscosity acceleration at current time and add it to respective particles
pub fn add_viscosity_acceleration<K: KernelFn>(
    fluid: &mut Fluid,
    boundary: &mut impl BoundaryHandling,
    neighbors: &NeighborList,
    params: &SystemParameters,
) {
    let forces_onto_boundary: Vec<ForceOntoBoundary> = for_each_collect!(
        mut [fluid.acceleration],
        ref [
            pos_now = fluid.position,
            vel_now = fluid.velocity,
            mass = fluid.mass,
            volume = fluid.volume,
            neighbors = neighbors,
            boundary = boundary
        ],
        |id, id_acceleration, local_forces| {
            let mut accu = Vector3::zeros();
            // add viscostiy acceleration from other moving particles
            for &neighbor in neighbors.get_neighbors(id) {
                let r_vec = vector(
                    &pos_now[neighbor],
                    &pos_now[id],
                );
                accu += params.fluid_viscosity
                    * 2.
                    * (3. + 2.)
                    * volume[neighbor]
                    * (vel_now[id] - vel_now[neighbor])
                        .dot(&(pos_now[id] - pos_now[neighbor]))
                    / ((pos_now[id] - pos_now[neighbor])
                        .norm_squared()
                        + 0.01 * params.rest_density_grid_spacing.powi(2))
                    * K::kernel_gradient(
                        &r_vec,
                        params.kernel_support_radius,
                    );
            }
            // add viscostiy acceleration contribution from boundary
            for (i, b) in boundary.iter().enumerate() {
                for &boundary_neighbor in b.get_neighbors(id, RequestMode::ViscosityAcceleration) {
                    let r_vec = vector(
                        b.position(boundary_neighbor),
                        &pos_now[id],
                    );
                    let acceleration = params.boundary_viscosity
                        * 2.
                        * (3. + 2.)
                        * b.volume(boundary_neighbor)
                        * (vel_now[id] - *b.velocity(boundary_neighbor))
                            .dot(
                                &(pos_now[id]
                                    - *b.position(boundary_neighbor)),
                            )
                        / ((pos_now[id]
                            - *b.position(boundary_neighbor))
                        .norm_squared()
                            + 0.01 * params.rest_density_grid_spacing.powi(2))
                        * K::kernel_gradient(
                            &r_vec,
                            params.kernel_support_radius,
                        );
                    let force = mass[id] * acceleration;
                    if b.is_dynamic() {
                        local_forces.push(ForceOntoBoundary {
                            id: i,
                            force: -force,
                            force_location: *b.position(boundary_neighbor),
                        });
                    }
                    accu += acceleration;
                }
            }
            *id_acceleration += accu;
        }
    );
    for force in forces_onto_boundary {
        boundary.add_force_onto_boundary(force);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neighbor_search::{NeighborList, NeighborSearch, SpatialHashing};
    use crate::render_info::BoundaryVisualization;
    use crate::sph::boundary_handling::{Boundary, BoundaryCheckpoint, VolumeMapBoundary};
    use crate::sph::fluid::Len;
    use crate::sph::kernel::CubicBSpline3D;
    use crate::sph::setup::input::{DynamicBoundaryDef, StaticBoundaryDef};
    use crate::utilities::triangle_mesh::MeshContainer;
    use nalgebra::Point3;
    use parry3d_f64::math::Vec3;
    use parry3d_f64::shape::TriMesh;

    // ─── Fixtures / helpers ─────────────────────────────────────────────

    fn make_params(
        fluid_viscosity: f64,
        boundary_viscosity: f64,
        kernel_support_radius: f64,
        rest_density_grid_spacing: f64,
    ) -> SystemParameters {
        #[cfg(not(feature = "cfl_time_step"))]
        {
            SystemParameters::new(
                0.001,
                rest_density_grid_spacing,
                kernel_support_radius,
                -1e9,
                fluid_viscosity,
                boundary_viscosity,
                0.0,
            )
        }
        #[cfg(feature = "cfl_time_step")]
        {
            SystemParameters::new(
                0.01,
                0.4,
                rest_density_grid_spacing,
                kernel_support_radius,
                -1e9,
                fluid_viscosity,
                boundary_viscosity,
                0.0,
            )
        }
    }

    fn cube_trimesh(side: f64) -> TriMesh {
        let h = side / 2.0;
        let positions = vec![
            Vec3::new(h, h, h),
            Vec3::new(h, h, -h),
            Vec3::new(h, -h, h),
            Vec3::new(h, -h, -h),
            Vec3::new(-h, h, h),
            Vec3::new(-h, h, -h),
            Vec3::new(-h, -h, h),
            Vec3::new(-h, -h, -h),
        ];
        let indices: Vec<[u32; 3]> = vec![
            [4, 2, 0],
            [2, 7, 3],
            [6, 5, 7],
            [1, 7, 5],
            [0, 3, 1],
            [4, 1, 5],
            [4, 6, 2],
            [2, 6, 7],
            [6, 4, 5],
            [1, 3, 7],
            [0, 2, 3],
            [4, 0, 1],
        ];
        TriMesh::new(positions, indices).expect("valid cube mesh")
    }

    /// Builds a `Fluid` with at least `min_n` sampled particles via the
    /// public `add_samples` API. The sampled positions/velocities/etc. are
    /// irrelevant — every test overwrites the fields it cares about by
    /// index before calling the function under test.
    fn fluid_with_at_least(min_n: usize) -> Fluid {
        let mesh = cube_trimesh(4.0);
        let mut fluid = Fluid::new();
        fluid.add_samples(&mesh, 0, 1000.0, 0.5);
        assert!(
            fluid.len() >= min_n,
            "expected at least {min_n} sampled particles, got {}",
            fluid.len()
        );
        fluid
    }

    fn build_fluid_neighbor_list(positions: &[Point3<f64>], radius: f64) -> NeighborList {
        let mut ns = SpatialHashing::new(radius);
        let mut neighbor_list = NeighborList::new(positions.len());
        ns.find_samples(radius, positions, positions, &mut neighbor_list);
        neighbor_list
    }

    // ─── Mock boundary: full manual control over samples/neighbors, and
    // captures forces/accelerations registered onto it for assertions.
    // Needed because `VolumeMapBoundary`'s internal `boundaries` field is
    // private, and `add_dynamic_boundary` requires the full (expensive)
    // mesh-discretization pipeline just to get a dynamic boundary at all.

    #[derive(Clone)]
    struct MockSample {
        position: Point3<f64>,
        velocity: Vector3<f64>,
        volume: f64,
    }

    #[derive(Clone)]
    struct MockBoundaryEntry {
        samples: Vec<MockSample>,
        neighbors_normal: Vec<Vec<usize>>,
        neighbors_viscosity: Vec<Vec<usize>>,
        /// `None` => static boundary (no reaction forces expected).
        /// `Some(..)` => dynamic boundary (`Boundary::is_dynamic`'s default
        /// impl derives directly from this).
        center_of_mass: Option<Point3<f64>>,
        /// Recorded via `Boundary::add_acceleration`, purely for test
        /// assertions — the real `RigidBodyMotion`'s force accumulator
        /// isn't inspectable from outside `boundary_handling`.
        accumulated_acceleration: Vector3<f64>,
    }

    impl Default for MockBoundaryEntry {
        fn default() -> Self {
            Self {
                samples: Vec::new(),
                neighbors_normal: Vec::new(),
                neighbors_viscosity: Vec::new(),
                center_of_mass: None,
                accumulated_acceleration: Vector3::zeros(),
            }
        }
    }

    impl Boundary for MockBoundaryEntry {
        fn get_neighbors(&self, id: usize, mode: RequestMode) -> &[usize] {
            let list = match mode {
                RequestMode::Normal => &self.neighbors_normal,
                RequestMode::ViscosityAcceleration => &self.neighbors_viscosity,
            };
            list.get(id).map(|v| v.as_slice()).unwrap_or(&[])
        }

        fn position(&self, id: usize) -> &Point3<f64> {
            &self.samples[id].position
        }

        fn velocity(&self, id: usize) -> &Vector3<f64> {
            &self.samples[id].velocity
        }

        fn volume(&self, id: usize) -> f64 {
            self.samples[id].volume
        }

        fn add_acceleration(&mut self, acceleration: Vector3<f64>) {
            self.accumulated_acceleration += acceleration;
        }

        fn center_of_mass(&self) -> Option<Point3<f64>> {
            self.center_of_mass
        }
    }

    /// Verbatim copy of a `ForceOntoBoundary`'s fields (which itself has no
    /// `Clone`/`Debug`), captured for test assertions.
    #[derive(Debug, Clone, Copy)]
    struct RecordedForce {
        id: usize,
        force: Vector3<f64>,
        force_location: Point3<f64>,
    }

    #[derive(Clone, Default)]
    struct MockBoundary {
        entries: Vec<MockBoundaryEntry>,
        recorded_forces: Vec<RecordedForce>,
    }

    impl BoundaryHandling for MockBoundary {
        fn new() -> Self {
            Self::default()
        }

        fn is_empty(&self) -> bool {
            self.entries.is_empty()
        }

        fn add_static_boundary(
            &mut self,
            _mesh: &mut MeshContainer,
            _boundary: &StaticBoundaryDef,
            _rest_density_grid_spacing: f64,
            _kernel_support_radius: f64,
        ) {
            unimplemented!("not exercised by non_pressure_accelerations tests")
        }

        fn add_dynamic_boundary(
            &mut self,
            _mesh: &mut MeshContainer,
            _boundary: &DynamicBoundaryDef,
            _rest_density_grid_spacing: f64,
            _kernel_support_radius: f64,
        ) {
            unimplemented!("not exercised by non_pressure_accelerations tests")
        }

        fn initialize<K: KernelFn>(
            &mut self,
            _neighbor_search: &mut impl NeighborSearch,
            _kernel_support_radius: f64,
            _boundary_rest_volume_weighting: f64,
        ) {
        }

        fn find_boundary_samples(
            &mut self,
            _neighbor_search: &mut impl NeighborSearch,
            _within_range: f64,
            _positions: &[Point3<f64>],
            _rest_density_grid_spacing: f64,
        ) {
            unimplemented!("test fixtures set up neighbors directly, not via search")
        }

        fn iter(&self) -> impl Iterator<Item = &dyn Boundary> + '_ {
            self.entries.iter().map(|b| b as &dyn Boundary)
        }

        fn iter_mut(&mut self) -> impl Iterator<Item = &mut dyn Boundary> + '_ {
            self.entries.iter_mut().map(|b| b as &mut dyn Boundary)
        }

        fn add_force_onto_boundary(&mut self, force: ForceOntoBoundary) {
            self.recorded_forces.push(RecordedForce {
                id: force.id,
                force: force.force,
                force_location: force.force_location,
            });
        }

        fn step_forward_in_time(&mut self, _dt: f64) {}

        fn get_fluid_depth(&self, _fluid_volume: f64) -> f64 {
            0.0
        }

        fn get_visualization(&self, _selector: &BoundaryVisualization) -> BoundaryVisualization {
            unimplemented!("not exercised by non_pressure_accelerations tests")
        }

        fn get_checkpoint(&self) -> BoundaryCheckpoint {
            BoundaryCheckpoint::default()
        }

        fn restore_from_checkpoint(&mut self, _state: &BoundaryCheckpoint) {}
    }

    // ─── reset_acceleration ─────────────────────────────────────────────

    #[test]
    fn reset_acceleration_zeroes_all_particles() {
        let mut fluid = fluid_with_at_least(3);
        fluid.acceleration[0] = Vector3::new(1.0, 2.0, 3.0);
        fluid.acceleration[1] = Vector3::new(-1.0, 0.0, 5.0);
        fluid.acceleration[2] = Vector3::new(0.5, 0.5, 0.5);

        reset_acceleration(&mut fluid);

        for a in &fluid.acceleration {
            assert_eq!(*a, Vector3::zeros());
        }
    }

    #[test]
    fn reset_acceleration_on_empty_fluid_does_not_panic() {
        let mut fluid = Fluid::new();
        reset_acceleration(&mut fluid);
        assert!(fluid.acceleration.is_empty());
    }

    // ─── add_gravity_acceleration: fluid particles ─────────────────────

    #[test]
    fn gravity_pulls_a_single_particle_toward_the_origin() {
        // Mirrors the softened attraction formula: `direction` points from
        // the particle toward the gravitation center, magnitude
        // regularized by a small softening term (hard-coded here as
        // `1e-4`, matching the private `GRAVITATION_SOFTENING` constant —
        // update this test if that constant's value ever changes).
        let mut fluid = fluid_with_at_least(1);
        let pos = Point3::new(2.0, 0.0, 0.0);
        fluid.position[0] = pos;
        fluid.acceleration[0] = Vector3::zeros();

        let mut boundary = VolumeMapBoundary::default();
        add_gravity_acceleration(&mut fluid, &mut boundary);

        let direction = vector(&pos, &Point3::origin());
        let dist = (direction.norm_squared() + 1e-4).sqrt();
        let expected = 9.81 * direction / dist;

        assert!((fluid.acceleration[0] - expected).norm() < 1e-9);
    }

    #[test]
    fn add_gravity_acceleration_adds_to_existing_acceleration_rather_than_overwriting() {
        let mut fluid_a = fluid_with_at_least(1);
        fluid_a.position[0] = Point3::new(1.0, 0.0, 0.0);
        fluid_a.acceleration[0] = Vector3::zeros();
        let mut boundary_a = VolumeMapBoundary::default();
        add_gravity_acceleration(&mut fluid_a, &mut boundary_a);
        let gravity_only = fluid_a.acceleration[0];

        let mut fluid_b = fluid_with_at_least(1);
        fluid_b.position[0] = Point3::new(1.0, 0.0, 0.0);
        let pre_existing = Vector3::new(3.0, -2.0, 1.0);
        fluid_b.acceleration[0] = pre_existing;
        let mut boundary_b = VolumeMapBoundary::default();
        add_gravity_acceleration(&mut fluid_b, &mut boundary_b);

        assert!((fluid_b.acceleration[0] - (pre_existing + gravity_only)).norm() < 1e-9);
    }

    #[test]
    fn gravity_acceleration_stays_finite_exactly_at_the_gravitation_center() {
        // Regression test for exactly the hazard the softening term
        // guards against: a particle placed exactly at the gravitation
        // center must not produce NaN/inf.
        let mut fluid = fluid_with_at_least(1);
        fluid.position[0] = Point3::origin();
        fluid.acceleration[0] = Vector3::zeros();

        let mut boundary = VolumeMapBoundary::default();
        add_gravity_acceleration(&mut fluid, &mut boundary);

        assert!(fluid.acceleration[0].iter().all(|c| c.is_finite()));
        // direction == 0 exactly, so 0/dist == 0 regardless of dist.
        assert_eq!(fluid.acceleration[0], Vector3::zeros());
    }

    #[test]
    fn gravity_acts_independently_on_each_particle() {
        let mut fluid = fluid_with_at_least(2);
        fluid.position[0] = Point3::new(1.0, 0.0, 0.0);
        fluid.position[1] = Point3::new(0.0, 2.0, 0.0);
        fluid.acceleration[0] = Vector3::zeros();
        fluid.acceleration[1] = Vector3::zeros();

        let mut boundary = VolumeMapBoundary::default();
        add_gravity_acceleration(&mut fluid, &mut boundary);

        assert!(fluid.acceleration[0].dot(&Vector3::new(1.0, 0.0, 0.0)) < 0.0);
        assert!(fluid.acceleration[1].dot(&Vector3::new(0.0, 1.0, 0.0)) < 0.0);
    }

    // ─── add_gravity_acceleration: boundaries ───────────────────────────

    #[test]
    fn gravity_adds_acceleration_onto_dynamic_boundaries_toward_the_origin() {
        let mut fluid = Fluid::new(); // no fluid particles needed for this check
        let mut boundary = MockBoundary::default();
        boundary.entries.push(MockBoundaryEntry {
            center_of_mass: Some(Point3::new(3.0, 0.0, 0.0)),
            ..Default::default()
        });

        add_gravity_acceleration(&mut fluid, &mut boundary);

        let acc = boundary.entries[0].accumulated_acceleration;
        assert!(acc.x < 0.0, "expected pull toward the origin, got {acc:?}");
        assert!(acc.y.abs() < 1e-12);
        assert!(acc.z.abs() < 1e-12);
    }

    #[test]
    fn gravity_does_not_call_add_acceleration_on_boundaries_without_a_center_of_mass() {
        // Static boundaries report `center_of_mass() == None`; the loop
        // must skip them entirely rather than calling `add_acceleration`
        // with some default/zero center of mass.
        let mut fluid = Fluid::new();
        let mut boundary = MockBoundary::default();
        boundary.entries.push(MockBoundaryEntry {
            center_of_mass: None,
            ..Default::default()
        });

        add_gravity_acceleration(&mut fluid, &mut boundary);

        assert_eq!(
            boundary.entries[0].accumulated_acceleration,
            Vector3::zeros()
        );
    }

    // ─── add_viscosity_acceleration: fluid-fluid ────────────────────────

    #[test]
    fn add_viscosity_acceleration_matches_manual_formula_for_fluid_fluid_pairs() {
        let h = 1.0;
        let dx = 0.3;
        let params = make_params(5.0, 5.0, h, dx);

        let mut fluid = fluid_with_at_least(3);
        let positions = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.3, 0.0, 0.0),
            Point3::new(0.0, 0.3, 0.0),
        ];
        let velocities = [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];
        let volumes = [0.02, 0.025, 0.03];
        for i in 0..3 {
            fluid.position[i] = positions[i];
            fluid.velocity[i] = velocities[i];
            fluid.volume[i] = volumes[i];
            fluid.mass[i] = 0.5;
            fluid.acceleration[i] = Vector3::zeros();
        }

        let neighbor_list = build_fluid_neighbor_list(&fluid.position, h);
        let mut boundary = VolumeMapBoundary::default();
        add_viscosity_acceleration::<CubicBSpline3D>(
            &mut fluid,
            &mut boundary,
            &neighbor_list,
            &params,
        );

        for id in 0..3 {
            let mut expected = Vector3::zeros();
            for &j in neighbor_list.get_neighbors(id) {
                let r_vec = vector(&positions[j], &positions[id]);
                expected += params.fluid_viscosity
                    * 2.0
                    * (3.0 + 2.0)
                    * volumes[j]
                    * (velocities[id] - velocities[j]).dot(&(positions[id] - positions[j]))
                    / ((positions[id] - positions[j]).norm_squared() + 0.01 * dx.powi(2))
                    * CubicBSpline3D::kernel_gradient(&r_vec, h);
            }
            assert!((fluid.acceleration[id] - expected).norm() < 1e-9, "id={id}");
        }
    }

    #[test]
    fn identical_velocities_produce_zero_fluid_fluid_viscosity_acceleration() {
        // No relative motion between neighbors -> no shear -> no viscous
        // force, regardless of proximity or viscosity strength.
        let h = 1.0;
        let params = make_params(5.0, 5.0, h, 0.3);
        let mut fluid = fluid_with_at_least(2);
        fluid.position[0] = Point3::new(0.0, 0.0, 0.0);
        fluid.position[1] = Point3::new(0.3, 0.0, 0.0);
        let v = Vector3::new(2.0, -1.0, 0.5);
        fluid.velocity[0] = v;
        fluid.velocity[1] = v;
        fluid.volume[0] = 0.02;
        fluid.volume[1] = 0.02;
        fluid.acceleration[0] = Vector3::zeros();
        fluid.acceleration[1] = Vector3::zeros();

        let neighbor_list = build_fluid_neighbor_list(&fluid.position, h);
        let mut boundary = VolumeMapBoundary::default();
        add_viscosity_acceleration::<CubicBSpline3D>(
            &mut fluid,
            &mut boundary,
            &neighbor_list,
            &params,
        );

        assert!(fluid.acceleration[0].norm() < 1e-12);
        assert!(fluid.acceleration[1].norm() < 1e-12);
    }

    #[test]
    fn viscosity_acceleration_scales_linearly_with_fluid_viscosity_parameter() {
        let h = 1.0;
        let dx = 0.3;

        let run = |mu: f64| {
            let params = make_params(mu, mu, h, dx);
            let mut fluid = fluid_with_at_least(2);
            fluid.position[0] = Point3::new(0.0, 0.0, 0.0);
            fluid.position[1] = Point3::new(0.3, 0.0, 0.0);
            fluid.velocity[0] = Vector3::new(1.0, 0.0, 0.0);
            fluid.velocity[1] = Vector3::new(-1.0, 0.0, 0.0);
            fluid.volume[0] = 0.02;
            fluid.volume[1] = 0.02;
            fluid.acceleration[0] = Vector3::zeros();
            fluid.acceleration[1] = Vector3::zeros();

            let neighbor_list = build_fluid_neighbor_list(&fluid.position, h);
            let mut boundary = VolumeMapBoundary::default();
            add_viscosity_acceleration::<CubicBSpline3D>(
                &mut fluid,
                &mut boundary,
                &neighbor_list,
                &params,
            );
            fluid.acceleration[0]
        };

        let a1 = run(2.0);
        let a2 = run(4.0);
        assert!((a2 - 2.0 * a1).norm() < 1e-9);
    }

    #[test]
    fn add_viscosity_acceleration_with_no_boundaries_only_applies_fluid_fluid_terms() {
        let h = 1.0;
        let params = make_params(5.0, 5.0, h, 0.3);
        let mut fluid = fluid_with_at_least(1);
        fluid.position[0] = Point3::origin();
        fluid.velocity[0] = Vector3::new(1.0, 0.0, 0.0);
        fluid.volume[0] = 0.02;
        fluid.acceleration[0] = Vector3::zeros();

        let neighbor_list = NeighborList::new(fluid.len()); // no fluid neighbors registered
        let mut boundary = VolumeMapBoundary::default();
        add_viscosity_acceleration::<CubicBSpline3D>(
            &mut fluid,
            &mut boundary,
            &neighbor_list,
            &params,
        );

        assert_eq!(fluid.acceleration[0], Vector3::zeros());
    }

    // ─── add_viscosity_acceleration: boundary contribution ──────────────

    #[test]
    fn add_viscosity_acceleration_includes_boundary_contribution_and_skips_reaction_force_for_static_boundaries()
     {
        let h = 1.0;
        let dx = 0.3;
        let params = make_params(0.0, 5.0, h, dx); // isolate the boundary term

        let mut fluid = fluid_with_at_least(1);
        fluid.position[0] = Point3::new(0.0, 0.0, 0.0);
        fluid.velocity[0] = Vector3::new(1.0, 0.0, 0.0);
        fluid.volume[0] = 0.02;
        fluid.mass[0] = 0.5;
        fluid.acceleration[0] = Vector3::zeros();

        let neighbor_list = NeighborList::new(fluid.len()); // no fluid-fluid neighbors

        let boundary_pos = Point3::new(0.2, 0.0, 0.0);
        let boundary_vol = 0.01;
        let mut boundary = MockBoundary::default();
        boundary.entries.push(MockBoundaryEntry {
            samples: vec![MockSample {
                position: boundary_pos,
                velocity: Vector3::zeros(),
                volume: boundary_vol,
            }],
            neighbors_viscosity: vec![vec![0]],
            neighbors_normal: vec![vec![]],
            center_of_mass: None, // static -> no reaction force expected
            ..Default::default()
        });

        add_viscosity_acceleration::<CubicBSpline3D>(
            &mut fluid,
            &mut boundary,
            &neighbor_list,
            &params,
        );

        let r_vec = vector(&boundary_pos, &fluid.position[0]);
        let expected = params.boundary_viscosity
            * 2.0
            * (3.0 + 2.0)
            * boundary_vol
            * (fluid.velocity[0] - Vector3::zeros()).dot(&(fluid.position[0] - boundary_pos))
            / ((fluid.position[0] - boundary_pos).norm_squared() + 0.01 * dx.powi(2))
            * CubicBSpline3D::kernel_gradient(&r_vec, h);

        assert!((fluid.acceleration[0] - expected).norm() < 1e-9);
        assert!(
            boundary.recorded_forces.is_empty(),
            "a static boundary must not receive a reaction force"
        );
    }

    #[test]
    fn add_viscosity_acceleration_registers_newtons_third_law_reaction_force_on_dynamic_boundaries()
    {
        let h = 1.0;
        let dx = 0.3;
        let params = make_params(0.0, 5.0, h, dx);

        let mut fluid = fluid_with_at_least(1);
        fluid.position[0] = Point3::new(0.0, 0.0, 0.0);
        fluid.velocity[0] = Vector3::new(1.0, 0.0, 0.0);
        fluid.volume[0] = 0.02;
        fluid.mass[0] = 0.5;
        fluid.acceleration[0] = Vector3::zeros();

        let neighbor_list = NeighborList::new(fluid.len());

        let boundary_pos = Point3::new(0.2, 0.0, 0.0);
        let mut boundary = MockBoundary::default();
        boundary.entries.push(MockBoundaryEntry {
            samples: vec![MockSample {
                position: boundary_pos,
                velocity: Vector3::zeros(),
                volume: 0.01,
            }],
            neighbors_viscosity: vec![vec![0]],
            neighbors_normal: vec![vec![]],
            center_of_mass: Some(Point3::new(5.0, 0.0, 0.0)), // dynamic
            ..Default::default()
        });

        add_viscosity_acceleration::<CubicBSpline3D>(
            &mut fluid,
            &mut boundary,
            &neighbor_list,
            &params,
        );

        // Force felt by the fluid particle (boundary-only, since
        // fluid_viscosity == 0.0 here). Newton's third law demands the
        // registered reaction force be exactly its negation.
        let force_on_fluid = fluid.mass[0] * fluid.acceleration[0];

        assert_eq!(boundary.recorded_forces.len(), 1);
        let recorded = boundary.recorded_forces[0];
        assert_eq!(recorded.id, 0);
        assert!((recorded.force - (-force_on_fluid)).norm() < 1e-9);
        assert_eq!(recorded.force_location, boundary_pos);
    }

    #[test]
    fn add_viscosity_acceleration_handles_multiple_fluid_particles_independently() {
        let h = 1.0;
        let dx = 0.3;
        let params = make_params(0.0, 5.0, h, dx);

        let mut fluid = fluid_with_at_least(2);
        fluid.position[0] = Point3::new(0.0, 0.0, 0.0);
        fluid.position[1] = Point3::new(0.0, 1.0, 0.0);
        fluid.velocity[0] = Vector3::new(1.0, 0.0, 0.0);
        fluid.velocity[1] = Vector3::new(0.0, 1.0, 0.0);
        fluid.volume[0] = 0.02;
        fluid.volume[1] = 0.02;
        fluid.mass[0] = 0.5;
        fluid.mass[1] = 0.5;
        fluid.acceleration[0] = Vector3::zeros();
        fluid.acceleration[1] = Vector3::zeros();

        let neighbor_list = NeighborList::new(fluid.len()); // no fluid-fluid coupling

        let mut boundary = MockBoundary::default();
        boundary.entries.push(MockBoundaryEntry {
            samples: vec![
                MockSample {
                    position: Point3::new(0.2, 0.0, 0.0),
                    velocity: Vector3::zeros(),
                    volume: 0.01,
                },
                MockSample {
                    position: Point3::new(0.0, 1.2, 0.0),
                    velocity: Vector3::zeros(),
                    volume: 0.01,
                },
            ],
            neighbors_viscosity: vec![vec![0], vec![1]],
            neighbors_normal: vec![vec![], vec![]],
            center_of_mass: Some(Point3::new(10.0, 0.0, 0.0)),
            ..Default::default()
        });

        add_viscosity_acceleration::<CubicBSpline3D>(
            &mut fluid,
            &mut boundary,
            &neighbor_list,
            &params,
        );

        assert!(
            fluid.acceleration[0].x.abs() > 1e-12,
            "particle 0 should feel a boundary viscosity effect along x"
        );
        assert!(
            fluid.acceleration[1].y.abs() > 1e-12,
            "particle 1 should feel a boundary viscosity effect along y"
        );
        assert_eq!(boundary.recorded_forces.len(), 2);
    }
}
