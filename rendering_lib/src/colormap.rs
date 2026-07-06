use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Encode, Decode)]
pub enum Colormap {
    #[default]
    Viridis,
    Magma,
    Inferno,
    Plasma,
    Turbo,
    Cividis,
    Blues,
}

impl Colormap {
    pub const LABELS: [&'static str; 7] = [
        "Viridis", "Magma", "Inferno", "Plasma", "Turbo", "Cividis", "Blues",
    ];

    pub const ALL: [Colormap; 7] = [
        Colormap::Viridis,
        Colormap::Magma,
        Colormap::Inferno,
        Colormap::Plasma,
        Colormap::Turbo,
        Colormap::Cividis,
        Colormap::Blues,
    ];

    pub fn gradient(self) -> colorous::Gradient {
        match self {
            Colormap::Viridis => colorous::VIRIDIS,
            Colormap::Magma => colorous::MAGMA,
            Colormap::Inferno => colorous::INFERNO,
            Colormap::Plasma => colorous::PLASMA,
            Colormap::Turbo => colorous::TURBO,
            Colormap::Cividis => colorous::CIVIDIS,
            Colormap::Blues => colorous::BLUES,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Colormap::Viridis => "Viridis",
            Colormap::Magma => "Magma",
            Colormap::Inferno => "Inferno",
            Colormap::Plasma => "Plasma",
            Colormap::Turbo => "Turbo",
            Colormap::Cividis => "Cividis",
            Colormap::Blues => "Blues",
        }
    }

    #[inline]
    pub fn eval(self, t: f32) -> [f32; 3] {
        let c = if self == Colormap::Blues {
            self.gradient()
                .eval_continuous((1. - t).clamp(0.0, 1.0) as f64)
        } else {
            self.gradient().eval_continuous(t.clamp(0.0, 1.0) as f64)
        };
        [c.r as f32 / 255.0, c.g as f32 / 255.0, c.b as f32 / 255.0]
    }
}

/// Normalizes a slice of values to [0, 1] and maps through colormap
pub fn values_to_colors(values: &[f32], max: f32, cmap: Colormap, alpha: f32) -> Vec<[f32; 4]> {
    if values.is_empty() {
        return Vec::new();
    }
    let min = 0.0_f32; // strikt positiv → 0 ist fix
    let range = (max - min).max(1e-10);

    values
        .iter()
        .map(|&v| {
            let t = (v - min) / range;
            let [r, g, b] = cmap.eval(t);
            [r, g, b, alpha]
        })
        .collect()
}

/// Assign distinct colors based on integer IDs
pub fn ids_to_colors(ids: &[u32], max_id: u32, cmap: Colormap, alpha: f32) -> Vec<[f32; 4]> {
    ids.iter()
        .map(|&id| {
            let t = if max_id > 0 {
                id as f32 / max_id as f32
            } else {
                0.0
            };
            let [r, g, b] = cmap.eval(t);
            [r, g, b, alpha]
        })
        .collect()
}
