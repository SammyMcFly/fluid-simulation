use simulation_lib::render_info::SensorPlaneData;

#[derive(Debug, Clone, PartialEq)]
pub struct Cut {
    pub x_active: bool,
    pub x_bound: f32,
    pub x_inverse: bool,
    pub x_inv: f32,
    pub y_active: bool,
    pub y_bound: f32,
    pub y_inverse: bool,
    pub y_inv: f32,
    pub z_active: bool,
    pub z_bound: f32,
    pub z_inverse: bool,
    pub z_inv: f32,
}

impl Default for Cut {
    fn default() -> Self {
        Self {
            x_active: false,
            x_bound: 0.,
            x_inverse: false,
            x_inv: 1.,
            y_active: false,
            y_bound: 0.,
            y_inverse: false,
            y_inv: 1.,
            z_active: false,
            z_bound: 0.,
            z_inverse: false,
            z_inv: 1.,
        }
    }
}

impl Cut {
    /// Returns `true` if the given position is inside the cut plane.
    pub fn cut(&self, position: &[f32; 3]) -> bool {
        let x_ok = !self.x_active || self.x_inv * (position[0] - self.x_bound) >= 0.;
        let y_ok = !self.y_active || self.y_inv * (position[1] - self.y_bound) >= 0.;
        let z_ok = !self.z_active || self.z_inv * (position[2] - self.z_bound) >= 0.;
        x_ok && y_ok && z_ok
    }

    pub fn x_flip(&mut self) {
        self.x_inverse = !self.x_inverse;
        self.x_inv *= -1.;
    }

    pub fn y_flip(&mut self) {
        self.y_inverse = !self.y_inverse;
        self.y_inv *= -1.;
    }

    pub fn z_flip(&mut self) {
        self.z_inverse = !self.z_inverse;
        self.z_inv *= -1.;
    }

    /// Returns sample grids for each active cut plane.
    /// Each plane is a complete rectangular grid (row-major) with known dimensions.
    /// No clipping by other cuts – that's handled at the rendering level.
    pub fn sensor_plane_samples(
        &self,
        dx: f32,
        interval_min: [f32; 3],
        interval_max: [f32; 3],
    ) -> Vec<SensorPlaneData> {
        let mut planes = Vec::new();

        // Plane x = x_bound: grid over y × z
        if self.x_active {
            let mut positions = Vec::new();
            let mut rows = 0usize;
            let mut cols = 0usize;

            let mut y = interval_min[1];
            while y <= interval_max[1] {
                let mut col_count = 0usize;
                let mut z = interval_min[2];
                while z <= interval_max[2] {
                    positions.push([self.x_bound, y, z]);
                    col_count += 1;
                    z += dx;
                }
                if rows == 0 {
                    cols = col_count;
                }
                rows += 1;
                y += dx;
            }

            if rows >= 2 && cols >= 2 {
                planes.push(SensorPlaneData {
                    positions,
                    data: Vec::new(),
                    rows,
                    cols,
                });
            }
        }

        // Plane y = y_bound: grid over x × z
        if self.y_active {
            let mut positions = Vec::new();
            let mut rows = 0usize;
            let mut cols = 0usize;

            let mut x = interval_min[0];
            while x <= interval_max[0] {
                let mut col_count = 0usize;
                let mut z = interval_min[2];
                while z <= interval_max[2] {
                    positions.push([x, self.y_bound, z]);
                    col_count += 1;
                    z += dx;
                }
                if rows == 0 {
                    cols = col_count;
                }
                rows += 1;
                x += dx;
            }

            if rows >= 2 && cols >= 2 {
                planes.push(SensorPlaneData {
                    positions,
                    data: Vec::new(),
                    rows,
                    cols,
                });
            }
        }

        // Plane z = z_bound: grid over x × y
        if self.z_active {
            let mut positions = Vec::new();
            let mut rows = 0usize;
            let mut cols = 0usize;

            let mut x = interval_min[0];
            while x <= interval_max[0] {
                let mut col_count = 0usize;
                let mut y = interval_min[1];
                while y <= interval_max[1] {
                    positions.push([x, y, self.z_bound]);
                    col_count += 1;
                    y += dx;
                }
                if rows == 0 {
                    cols = col_count;
                }
                rows += 1;
                x += dx;
            }

            if rows >= 2 && cols >= 2 {
                planes.push(SensorPlaneData {
                    positions,
                    data: Vec::new(),
                    rows,
                    cols,
                });
            }
        }

        planes
    }
}
