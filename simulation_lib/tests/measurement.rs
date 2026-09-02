//! Integration tests for `measurement`, exercising only its public API.

use std::sync::atomic::{AtomicUsize, Ordering};

use simulation_lib::measurement::{
    Measurement, MeasurementError, MeasurementSeries, RecordingStatus,
};

static TEMP_DIR_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Creates a fresh, uniquely named temp directory for a single test, so
/// concurrently running tests never interfere with each other's files.
/// Cleanup is best-effort (ignored if it fails).
fn unique_temp_dir() -> std::path::PathBuf {
    let n = TEMP_DIR_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("measurement_test_{}_{n}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("failed to create temp dir");
    dir
}

fn sample_measurement(time: f64) -> Measurement {
    Measurement {
        time,
        density: 998.2,
        density_error: 0.5,
        kinetic_energy: 12.3,
        stiffness: 500.0,
        fluid_viscosity: 0.01,
        boundary_viscosity: 0.02,
        fluid_depth: 3.0,
        rest_density_grid_spacing: 0.025,
        kernel_support_radius: 0.05,
        time_step_size: 0.001,
        target_density_error: 0.1,
        solver_iterations: 7,
        relaxation_factor: 0.5,
        time_step_wall_clock_time: 0.003,
        predicted_density_error: 0.09,
    }
}

/// `Measurement` derives neither `PartialEq` nor `Copy`, so comparisons are
/// done field-by-field.
fn assert_measurement_eq(actual: &Measurement, expected: &Measurement) {
    assert!((actual.time - expected.time).abs() < 1e-12);
    assert!((actual.density - expected.density).abs() < 1e-12);
    assert!((actual.density_error - expected.density_error).abs() < 1e-12);
    assert!((actual.kinetic_energy - expected.kinetic_energy).abs() < 1e-12);
    assert!((actual.stiffness - expected.stiffness).abs() < 1e-12);
    assert!((actual.fluid_viscosity - expected.fluid_viscosity).abs() < 1e-12);
    assert!((actual.boundary_viscosity - expected.boundary_viscosity).abs() < 1e-12);
    assert!((actual.fluid_depth - expected.fluid_depth).abs() < 1e-12);
    assert!((actual.rest_density_grid_spacing - expected.rest_density_grid_spacing).abs() < 1e-12);
    assert!((actual.kernel_support_radius - expected.kernel_support_radius).abs() < 1e-12);
    assert!((actual.time_step_size - expected.time_step_size).abs() < 1e-12);
    assert!((actual.target_density_error - expected.target_density_error).abs() < 1e-12);
    assert_eq!(actual.solver_iterations, expected.solver_iterations);
    assert!((actual.relaxation_factor - expected.relaxation_factor).abs() < 1e-12);
    assert!((actual.time_step_wall_clock_time - expected.time_step_wall_clock_time).abs() < 1e-12);
    assert!((actual.predicted_density_error - expected.predicted_density_error).abs() < 1e-12);
}

// ─── RecordingStatus ────────────────────────────────────────────────────

#[test]
fn recording_status_default_is_none() {
    let status = RecordingStatus::default();
    assert!(!status.is_active());
    assert!(!status.is_finished());
}

#[test]
fn recording_status_advances_through_the_documented_sequence() {
    let mut status = RecordingStatus::NotStarted;
    assert!(!status.is_active());
    assert!(!status.is_finished());

    status.advance_to_next_state();
    assert!(matches!(status, RecordingStatus::InProgress));
    assert!(status.is_active());
    assert!(!status.is_finished());

    status.advance_to_next_state();
    assert!(matches!(status, RecordingStatus::Finished));
    assert!(!status.is_active());
    assert!(status.is_finished());
}

#[test]
#[should_panic]
fn recording_status_advance_from_finished_panics() {
    let mut status = RecordingStatus::Finished;
    status.advance_to_next_state();
}

#[test]
#[should_panic]
fn recording_status_advance_from_none_panics() {
    let mut status = RecordingStatus::None;
    status.advance_to_next_state();
}

#[test]
fn recording_status_is_active_is_true_only_for_in_progress() {
    assert!(!RecordingStatus::None.is_active());
    assert!(!RecordingStatus::NotStarted.is_active());
    assert!(RecordingStatus::InProgress.is_active());
    assert!(!RecordingStatus::Finished.is_active());
}

#[test]
fn recording_status_is_finished_is_true_only_for_finished() {
    assert!(!RecordingStatus::None.is_finished());
    assert!(!RecordingStatus::NotStarted.is_finished());
    assert!(!RecordingStatus::InProgress.is_finished());
    assert!(RecordingStatus::Finished.is_finished());
}

// ─── Measurement ────────────────────────────────────────────────────────

#[test]
fn measurement_default_has_all_zero_fields() {
    let m = Measurement::default();
    assert_eq!(m.time, 0.0);
    assert_eq!(m.density, 0.0);
    assert_eq!(m.density_error, 0.0);
    assert_eq!(m.kinetic_energy, 0.0);
    assert_eq!(m.stiffness, 0.0);
    assert_eq!(m.fluid_viscosity, 0.0);
    assert_eq!(m.boundary_viscosity, 0.0);
    assert_eq!(m.fluid_depth, 0.0);
    assert_eq!(m.rest_density_grid_spacing, 0.0);
    assert_eq!(m.kernel_support_radius, 0.0);
    assert_eq!(m.time_step_size, 0.0);
    assert_eq!(m.target_density_error, 0.0);
    assert_eq!(m.solver_iterations, 0);
    assert_eq!(m.relaxation_factor, 0.0);
    assert_eq!(m.time_step_wall_clock_time, 0.0);
    assert_eq!(m.predicted_density_error, 0.0);
}

#[test]
fn measurement_survives_a_csv_round_trip() {
    // Exercises the exact mechanism `MeasurementSeries::save` relies on
    // (`csv::Writer::serialize`), independent of file I/O.
    let original = sample_measurement(1.5);

    let mut wtr = csv::Writer::from_writer(Vec::new());
    wtr.serialize(&original)
        .expect("failed to serialize Measurement to CSV");
    let bytes = wtr.into_inner().expect("failed to finalize CSV writer");

    let mut rdr = csv::Reader::from_reader(bytes.as_slice());
    let mut records = rdr.deserialize::<Measurement>();
    let roundtripped: Measurement = records
        .next()
        .expect("expected exactly one CSV record")
        .expect("failed to deserialize Measurement from CSV");
    assert!(records.next().is_none(), "expected exactly one CSV record");

    assert_measurement_eq(&roundtripped, &original);
}

// ─── MeasurementSeries::new: path handling ───────────────────────────────

#[test]
fn new_creates_missing_parent_directories() {
    let base = unique_temp_dir();
    let nested_path = base.join("nested/subdir/measurements.csv");

    let series = MeasurementSeries::new(nested_path.to_str().unwrap())
        .expect("expected directory creation to succeed");

    assert!(nested_path.parent().unwrap().exists());
    assert_eq!(series.get_path().file_name().unwrap(), "measurements.csv");

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn new_returns_an_absolute_canonicalized_path() {
    let base = unique_temp_dir();
    let path = base.join("measurements.csv");

    let series = MeasurementSeries::new(path.to_str().unwrap()).unwrap();

    assert!(series.get_path().is_absolute());

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn new_with_existing_parent_and_no_name_collision_keeps_the_original_name() {
    let base = unique_temp_dir(); // parent already exists
    let path = base.join("fresh_measurements.csv");

    let series = MeasurementSeries::new(path.to_str().unwrap()).unwrap();

    assert_eq!(
        series.get_path().file_name().unwrap(),
        "fresh_measurements.csv"
    );

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn new_appends_a_numbered_suffix_when_the_file_already_exists() {
    let base = unique_temp_dir();
    let path = base.join("measurements.csv");
    std::fs::write(&path, "pretend existing content").unwrap();

    let series = MeasurementSeries::new(path.to_str().unwrap()).unwrap();

    let name = series
        .get_path()
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    assert_ne!(name, "measurements.csv");
    assert!(
        name.contains("_#2") && name.ends_with(".csv"),
        "expected a '_#2'-suffixed variant, got '{name}'"
    );

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn new_increments_the_suffix_across_repeated_collisions() {
    let base = unique_temp_dir();
    let path = base.join("measurements.csv");
    std::fs::write(&path, "original").unwrap();
    std::fs::write(base.join("measurements_#2.csv"), "collision 1").unwrap();
    std::fs::write(base.join("measurements_#3.csv"), "collision 2").unwrap();

    let series = MeasurementSeries::new(path.to_str().unwrap()).unwrap();

    assert_eq!(
        series.get_path().file_name().unwrap(),
        "measurements_#4.csv"
    );

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn new_called_twice_in_a_row_yields_two_distinct_paths() {
    // Mirrors realistic usage: creating several `MeasurementSeries` for the
    // same base name within the same run must never silently collide.
    let base = unique_temp_dir();
    let path = base.join("run.csv");

    let series_a = MeasurementSeries::new(path.to_str().unwrap()).unwrap();
    series_a.save().unwrap(); // actually create the file on disk
    let series_b = MeasurementSeries::new(path.to_str().unwrap()).unwrap();

    assert_ne!(series_a.get_path(), series_b.get_path());

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn new_with_a_bare_filename_and_no_directory_component_succeeds() {
    // Exercises the `.filter(|p| !p.as_os_str().is_empty())` fallback to
    // "." for a path with no directory component at all.
    let previous_dir = std::env::current_dir().unwrap();
    let base = unique_temp_dir();
    std::env::set_current_dir(&base).expect("failed to change to temp dir");

    let result = MeasurementSeries::new("bare_name.csv");

    std::env::set_current_dir(previous_dir).expect("failed to restore working directory");
    let _ = std::fs::remove_dir_all(&base);

    let series = result.expect("expected a bare filename (no directory component) to work");
    assert_eq!(series.get_path().file_name().unwrap(), "bare_name.csv");
}

// ─── MeasurementSeries::push_back / save ─────────────────────────────────

#[test]
fn save_writes_every_pushed_measurement_in_order() {
    let base = unique_temp_dir();
    let path = base.join("series.csv");
    let mut series = MeasurementSeries::new(path.to_str().unwrap()).unwrap();

    let m1 = sample_measurement(0.0);
    let m2 = sample_measurement(0.1);
    let m3 = sample_measurement(0.2);
    series.push_back(m1.clone());
    series.push_back(m2.clone());
    series.push_back(m3.clone());

    series.save().expect("save should succeed");

    let mut rdr = csv::Reader::from_path(series.get_path()).unwrap();
    let records: Vec<Measurement> = rdr
        .deserialize()
        .map(|r| r.expect("failed to deserialize row"))
        .collect();

    assert_eq!(records.len(), 3);
    assert_measurement_eq(&records[0], &m1);
    assert_measurement_eq(&records[1], &m2);
    assert_measurement_eq(&records[2], &m3);

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn save_on_an_empty_series_creates_an_essentially_empty_file() {
    // No measurements were pushed, so `csv::Writer::serialize` is never
    // called at all -> the file is created but has no data rows (nor even
    // a header row, since headers are only written on the first
    // `serialize` call).
    let base = unique_temp_dir();
    let path = base.join("empty.csv");
    let series = MeasurementSeries::new(path.to_str().unwrap()).unwrap();

    series
        .save()
        .expect("saving an empty series should still succeed");

    let contents = std::fs::read_to_string(series.get_path()).unwrap();
    assert!(
        contents.trim().is_empty(),
        "expected an empty file, got: {contents:?}"
    );

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn save_called_twice_overwrites_rather_than_appends() {
    let base = unique_temp_dir();
    let path = base.join("overwrite.csv");
    let mut series = MeasurementSeries::new(path.to_str().unwrap()).unwrap();

    series.push_back(sample_measurement(0.0));
    series.save().unwrap();

    series.push_back(sample_measurement(0.1));
    series.save().unwrap(); // queue now holds 2 measurements

    let mut rdr = csv::Reader::from_path(series.get_path()).unwrap();
    let records: Vec<Measurement> = rdr
        .deserialize()
        .map(|r| r.expect("failed to deserialize row"))
        .collect();

    // Since `File::create` truncates, the second `save()` must fully
    // replace the file's contents (2 rows), not append to the first
    // save's single row (which would incorrectly total 3).
    assert_eq!(records.len(), 2);

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn get_path_is_stable_across_multiple_calls() {
    let base = unique_temp_dir();
    let path = base.join("stable.csv");
    let series = MeasurementSeries::new(path.to_str().unwrap()).unwrap();

    assert_eq!(series.get_path(), series.get_path());

    let _ = std::fs::remove_dir_all(&base);
}

// ─── MeasurementError ────────────────────────────────────────────────────

#[test]
fn measurement_error_io_variant_wraps_and_displays_the_source() {
    fn trigger() -> Result<(), MeasurementError> {
        let _ = std::fs::File::open("/definitely/does/not/exist/measurement_test.csv")?;
        Ok(())
    }
    let err = trigger().expect_err("expected file open to fail");
    assert!(matches!(err, MeasurementError::Io(_)));
    assert!(format!("{err}").starts_with("I/O error:"));
}

#[test]
fn measurement_error_csv_variant_wraps_and_displays_the_source() {
    fn trigger() -> Result<(), MeasurementError> {
        // A CSV reader with `flexible(false)` (the default) errors on a
        // record with an inconsistent number of fields once headers have
        // established the expected column count.
        let data = "a,b,c\n1,2\n";
        let mut rdr = csv::Reader::from_reader(data.as_bytes());
        for result in rdr.records() {
            result?;
        }
        Ok(())
    }
    let err = trigger().expect_err("expected a CSV field-count mismatch to fail");
    assert!(matches!(err, MeasurementError::Csv(_)));
    assert!(format!("{err}").starts_with("CSV serialization error:"));
}

#[test]
fn measurement_error_no_unique_file_name_displays_the_offending_path() {
    let path = std::path::PathBuf::from("/some/example/measurements.csv");
    let err = MeasurementError::NoUniqueFileName(path.clone());
    let message = format!("{err}");
    assert!(message.contains(&path.display().to_string()));
    assert!(message.contains("already exist"));
}

// ─── MeasurementSeries::new: exhausting the suffix range (documented,
// not exercised — see comment) ────────────────────────────────────────────

// A literal test of the `NoUniqueFileName` error path via the public API
// would require pre-creating on the order of 65,000 colliding files on
// disk (the full `u16` suffix range) before the first call to `new` even
// begins — likely taking minutes and creating significant I/O load for a
// single assertion. So it is intentionally not included
// here rather than included as an impractically slow or flaky test.
