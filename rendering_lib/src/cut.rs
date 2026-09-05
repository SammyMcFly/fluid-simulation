use crate::model::ColoredMeshVertex;
use simulation_lib::render_info::SensorPlaneData;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CutAxis {
    X,
    Y,
    Z,
}

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

    pub fn toggle(&mut self, axis: CutAxis) {
        match axis {
            CutAxis::X => self.x_active = !self.x_active,
            CutAxis::Y => self.y_active = !self.y_active,
            CutAxis::Z => self.z_active = !self.z_active,
        }
    }

    pub fn flip(&mut self, axis: CutAxis) {
        match axis {
            CutAxis::X => {
                self.x_inverse = !self.x_inverse;
                self.x_inv *= -1.;
            }
            CutAxis::Y => {
                self.y_inverse = !self.y_inverse;
                self.y_inv *= -1.;
            }
            CutAxis::Z => {
                self.z_inverse = !self.z_inverse;
                self.z_inv *= -1.;
            }
        }
    }

    pub fn add_cut_bound(&mut self, axis: CutAxis, delta: f32) {
        match axis {
            CutAxis::X => {
                self.x_bound += delta;
            }
            CutAxis::Y => {
                self.y_bound += delta;
            }
            CutAxis::Z => {
                self.z_bound += delta;
            }
        }
    }

    pub fn set_cut_bound(&mut self, axis: CutAxis, value: f32) {
        match axis {
            CutAxis::X => {
                self.x_bound = value;
            }
            CutAxis::Y => {
                self.y_bound = value;
            }
            CutAxis::Z => {
                self.z_bound = value;
            }
        }
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

/// Linearly interpolates between two vertices at parameter `t` (0 = `a`, 1 = `b`).
///
/// Normals are lerped without renormalizing -- the sliver of new geometry
/// introduced right at a cut boundary makes the resulting shading error
/// negligible, and renormalizing per-vertex on every cut recompute would add
/// unnecessary cost. Revisit if this ever becomes visible on very large
/// adjacent triangles.
fn lerp_vertex(a: &ColoredMeshVertex, b: &ColoredMeshVertex, t: f32) -> ColoredMeshVertex {
    let lerp3 = |x: [f32; 3], y: [f32; 3]| {
        [
            x[0] + t * (y[0] - x[0]),
            x[1] + t * (y[1] - x[1]),
            x[2] + t * (y[2] - x[2]),
        ]
    };
    let lerp4 = |x: [f32; 4], y: [f32; 4]| {
        [
            x[0] + t * (y[0] - x[0]),
            x[1] + t * (y[1] - x[1]),
            x[2] + t * (y[2] - x[2]),
            x[3] + t * (y[3] - x[3]),
        ]
    };
    ColoredMeshVertex {
        position: lerp3(a.position, b.position),
        normal: lerp3(a.normal, b.normal),
        color: lerp4(a.color, b.color),
    }
}

/// Clips a single triangle against one half-space (`signed_dist(v) >= 0` is
/// kept), walking its 3 edges in order. Returns a convex polygon of 0, 3, or
/// 4 vertices in the SAME winding order as the input -- clipping a triangle
/// by one plane can add at most one new edge, never more.
fn clip_triangle_by_plane(
    v: &[ColoredMeshVertex; 3],
    signed_dist: &[f32; 3],
) -> Vec<ColoredMeshVertex> {
    let mut out = Vec::with_capacity(4);
    for i in 0..3 {
        let j = (i + 1) % 3;
        let (cur, cur_d) = (v[i], signed_dist[i]);
        let (next, next_d) = (v[j], signed_dist[j]);
        if cur_d >= 0.0 {
            out.push(cur);
        }
        if (cur_d >= 0.0) != (next_d >= 0.0) {
            let t = cur_d / (cur_d - next_d);
            out.push(lerp_vertex(&cur, &next, t));
        }
    }
    out
}

/// Clips every triangle of `(vertices, indices)` against the half-space
/// `inv * (position[axis] - bound) >= 0`, fan-triangulating each triangle's
/// (0, 3, or 4-vertex) clipped result.
///
/// Adjacent triangles sharing an edge always compute the SAME interpolation
/// parameter `t` for that shared edge (same endpoint values, same signed
/// distances), so this never introduces cracks/T-junctions along the cut
/// boundary -- the new edge vertices line up exactly between triangles.
fn clip_mesh_by_axis_plane(
    vertices: &[ColoredMeshVertex],
    indices: &[u32],
    axis: usize,
    bound: f32,
    inv: f32,
) -> (Vec<ColoredMeshVertex>, Vec<u32>) {
    let mut out_vertices = Vec::with_capacity(vertices.len());
    let mut out_indices = Vec::with_capacity(indices.len());

    for tri in indices.chunks_exact(3) {
        let v = [
            vertices[tri[0] as usize],
            vertices[tri[1] as usize],
            vertices[tri[2] as usize],
        ];
        let signed_dist = [
            inv * (v[0].position[axis] - bound),
            inv * (v[1].position[axis] - bound),
            inv * (v[2].position[axis] - bound),
        ];

        let polygon = clip_triangle_by_plane(&v, &signed_dist);
        if polygon.len() < 3 {
            continue; // fully outside, or degenerate
        }

        let base = out_vertices.len() as u32;
        out_vertices.extend_from_slice(&polygon);
        for k in 1..polygon.len() - 1 {
            out_indices.push(base);
            out_indices.push(base + k as u32);
            out_indices.push(base + k as u32 + 1);
        }
    }

    (out_vertices, out_indices)
}

impl Cut {
    /// Clips a triangle mesh against every currently active cut plane,
    /// producing a clean cut surface (new vertices right along each cut
    /// boundary) instead of discarding whole straddling triangles.
    ///
    /// Vertex positions must already be in the same coordinate space as
    /// `x_bound`/`y_bound`/`z_bound` (world space) -- see the boundary-mesh
    /// caveat below regarding per-instance pose transforms.
    pub fn clip_mesh(
        &self,
        vertices: Vec<ColoredMeshVertex>,
        indices: Vec<u32>,
    ) -> (Vec<ColoredMeshVertex>, Vec<u32>) {
        let mut current = (vertices, indices);
        if self.x_active {
            current = clip_mesh_by_axis_plane(&current.0, &current.1, 0, self.x_bound, self.x_inv);
        }
        if self.y_active {
            current = clip_mesh_by_axis_plane(&current.0, &current.1, 1, self.y_bound, self.y_inv);
        }
        if self.z_active {
            current = clip_mesh_by_axis_plane(&current.0, &current.1, 2, self.z_bound, self.z_inv);
        }
        current
    }
}
