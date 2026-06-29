pub fn viridis(t: f32) -> [f32; 3] {
    let c = colorous::VIRIDIS.eval_continuous(t as f64);
    [c.r as f32 / 255.0, c.g as f32 / 255.0, c.b as f32 / 255.0]
}



/// Normalizes a slice of values to [0, 1] and maps through colormap
pub fn values_to_colors(values: &[f32], alpha: f32) -> Vec<[f32; 4]> {
    if values.is_empty() {
        return Vec::new();
    }
    // let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    // let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min: f32 = 0.;
    let max: f32 = 10.;
    let range = (max - min).max(1e-10);

    values
        .iter()
        .map(|&v| {
            let t = ((v - min) / range) as f64;
            let c = colorous::VIRIDIS.eval_continuous(t);
            [c.r as f32 / 255.0, c.g as f32 / 255.0, c.b as f32 / 255.0, alpha]
        })
        .collect()
}

/// Assign distinct colors based on integer IDs
pub fn id_to_colors(ids: &[u32], max_id: u32, alpha: f32) -> Vec<[f32; 4]> {
    ids.iter()
        .map(|&id| {
            let t = if max_id > 0 { id as f64 / max_id as f64 } else { 0.0 };
            let c = colorous::VIRIDIS.eval_continuous(t);
            [c.r as f32 / 255.0, c.g as f32 / 255.0, c.b as f32 / 255.0, alpha]
        })
        .collect()
}