//! Integration tests for `pressure_solver`'s public API: the
//! `PressureSolverVariant` enum, the `PressureSolver` trait's default
//! `measurement_info` method, and `SolverMeasurementInfo`.
//!
//! The concrete solvers (`IISPH`, `IISPHwOST`, `SESPH`, `SESPHwSplitting`)
//! are only checked here for actually implementing the `PressureSolver`
//! trait (a compile-time smoke test) — their internal convergence
//! behavior is not covered, since their source wasn't available to derive
//! meaningful behavioral tests from.
mod iisph;
mod iisph_optimized_source_term;
mod sesph;
mod sesph_with_splitting;

use serde::de::value::{Error as ValueError, StrDeserializer};
use serde::de::{Deserialize, IntoDeserializer};

use simulation_lib::neighbor_search::NeighborList;
use simulation_lib::sph::CurrentSystemProperties;
use simulation_lib::sph::SystemParameters;
use simulation_lib::sph::boundary_handling::BoundaryHandling;
use simulation_lib::sph::fluid::Fluid;
use simulation_lib::sph::kernel::KernelFn;
use simulation_lib::sph::pressure_solver::{
    IISPH, IISPHwOST, PressureSolver, PressureSolverVariant, SESPH, SESPHwSplitting,
    SolverMeasurementInfo,
};
use simulation_lib::sph::setup::input::Parameters;

// ─── SolverMeasurementInfo ─────────────────────────────────────────────

#[test]
fn solver_measurement_info_default_is_all_zero() {
    let info = SolverMeasurementInfo::default();
    assert_eq!(info.stiffness, 0.0);
    assert_eq!(info.target_density_error, 0.0);
    assert_eq!(info.solver_iterations, 0);
    assert_eq!(info.relaxation_factor, 0.0);
    assert_eq!(info.predicted_density_error, 0.0);
}

// ─── PressureSolverVariant: Deserialize contract ───────────────────────

fn deserialize_variant(name: &str) -> Result<PressureSolverVariant, ValueError> {
    let de: StrDeserializer<'_, ValueError> = name.into_deserializer();
    PressureSolverVariant::deserialize(de)
}

#[test]
fn pressure_solver_variant_deserializes_every_documented_name() {
    assert!(matches!(
        deserialize_variant("SESPH"),
        Ok(PressureSolverVariant::SESPH)
    ));
    assert!(matches!(
        deserialize_variant("SESPHwSplitting"),
        Ok(PressureSolverVariant::SESPHwSplitting)
    ));
    assert!(matches!(
        deserialize_variant("IISPH"),
        Ok(PressureSolverVariant::IISPH)
    ));
    assert!(matches!(
        deserialize_variant("IISPHwOST"),
        Ok(PressureSolverVariant::IISPHwOST)
    ));
}

#[test]
fn pressure_solver_variant_rejects_an_unknown_name() {
    assert!(deserialize_variant("NotARealSolver").is_err());
}

// ─── PressureSolver trait: default `measurement_info` ──────────────────

/// Minimal trait implementer used purely to exercise `PressureSolver`'s
/// default `measurement_info` method in isolation, without depending on
/// any concrete solver's (unknown-to-this-test) internal logic.
#[derive(Clone)]
struct NoOpSolver;

impl PressureSolver for NoOpSolver {
    fn new(_params: &Parameters) -> Self {
        NoOpSolver
    }

    fn solve_and_add_acceleration<K: KernelFn>(
        &mut self,
        _fluid: &mut Fluid,
        _boundary: &mut impl BoundaryHandling,
        _neighbors: &NeighborList,
        _params: &SystemParameters,
        _properties: &mut CurrentSystemProperties,
    ) {
        // Intentionally does nothing: only `measurement_info`'s default
        // implementation is under test here.
    }
}

#[test]
fn pressure_solver_default_measurement_info_matches_solver_measurement_info_default() {
    let solver = NoOpSolver;
    let info = solver.measurement_info();
    let default_info = SolverMeasurementInfo::default();

    assert_eq!(info.stiffness, default_info.stiffness);
    assert_eq!(info.target_density_error, default_info.target_density_error);
    assert_eq!(info.solver_iterations, default_info.solver_iterations);
    assert_eq!(info.relaxation_factor, default_info.relaxation_factor);
    assert_eq!(
        info.predicted_density_error,
        default_info.predicted_density_error
    );
}

// ─── Concrete solvers actually implement the trait (compile-time smoke
// tests) ─────────────────────────────────────────────────────────────────

fn assert_implements_pressure_solver<T: PressureSolver>() {}

#[test]
fn concrete_solvers_implement_pressure_solver_and_its_supertraits() {
    // `PressureSolver: Send + Sync + Clone` — this also transitively
    // confirms those bounds hold for each concrete type.
    assert_implements_pressure_solver::<IISPH>();
    assert_implements_pressure_solver::<IISPHwOST>();
    assert_implements_pressure_solver::<SESPH>();
    assert_implements_pressure_solver::<SESPHwSplitting>();
}
