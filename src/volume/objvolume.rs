use super::{Image, PaintVolume, SurfaceVolume, Volume, VoxelVolume};
use crate::volume::{AffineTransform, CompositingMode, VoxelPaintVolume};
use libm::{pow, sqrt};
use std::sync::Arc;
use wavefront_obj::obj::{self, Object, Primitive, Vertex};

// TODO: create a single AABB index for both XYZ and UV index
struct XYZIndex {
    n: usize, // same in every direction
    min: [f64; 3],
    max: [f64; 3],
    grid: Vec<Vec<usize>>, // n * n * n cell with indices to the faces
}
impl XYZIndex {
    fn new(object: &Object, n: usize) -> Self {
        let mut grid = vec![vec![]; n * n * n];

        fn minmax(vertices: &[Vertex], coord: impl Fn(&Vertex) -> f64) -> (f64, f64) {
            (
                vertices.iter().map(|v| coord(v)).fold(f64::INFINITY, f64::min),
                vertices.iter().map(|v| coord(v)).fold(f64::NEG_INFINITY, f64::max),
            )
        }

        let (min_x, max_x) = minmax(&object.vertices, |v| v.x);
        let (min_y, max_y) = minmax(&object.vertices, |v| v.y);
        let (min_z, max_z) = minmax(&object.vertices, |v| v.z);

        let dx = (max_x - min_x) / n as f64;
        let dy = (max_y - min_y) / n as f64;
        let dz = (max_z - min_z) / n as f64;

        for (i, s) in object.geometry[0].shapes.iter().enumerate() {
            match s.primitive {
                Primitive::Triangle(i1, i2, i3) => {
                    let v1 = &i1.0;
                    let v2 = &i2.0;
                    let v3 = &i3.0;

                    let vert1 = object.vertices[*v1];
                    let vert2 = object.vertices[*v2];
                    let vert3 = object.vertices[*v3];

                    let min_x_t = vert1.x.min(vert2.x).min(vert3.x);
                    let max_x_t = vert1.x.max(vert2.x).max(vert3.x);

                    let min_y_t = vert1.y.min(vert2.y).min(vert3.y);
                    let max_y_t = vert1.y.max(vert2.y).max(vert3.y);

                    let min_z_t = vert1.z.min(vert2.z).min(vert3.z);
                    let max_z_t = vert1.z.max(vert2.z).max(vert3.z);

                    let min_x_cell = (((min_x_t - min_x) / dx) as usize).max(0).min(n - 1);
                    let max_x_cell = (((max_x_t - min_x) / dx) as usize).max(0).min(n - 1);

                    let min_y_cell = (((min_y_t - min_y) / dy) as usize).max(0).min(n - 1);
                    let max_y_cell = (((max_y_t - min_y) / dy) as usize).max(0).min(n - 1);

                    let min_z_cell = (((min_z_t - min_z) / dz) as usize).max(0).min(n - 1);
                    let max_z_cell = (((max_z_t - min_z) / dz) as usize).max(0).min(n - 1);

                    for x in min_x_cell..=max_x_cell {
                        for y in min_y_cell..=max_y_cell {
                            for z in min_z_cell..=max_z_cell {
                                let idx = x * n * n + y * n + z;
                                grid[idx].push(i);
                            }
                        }
                    }
                }
                _ => todo!(),
            }
        }

        /* let num_cells = rows * cols;
        let max_occ = grid.iter().map(|g| g.len()).max().unwrap();
        println!("FaceIndex: {} cells, max occ: {}", num_cells, max_occ); */

        let min = [min_x, min_y, min_z];
        let max = [max_x, max_y, max_z];

        Self { n, min, max, grid }
    }

    fn in_bounds(&self, min: [f64; 3], max: [f64; 3]) -> Vec<usize> {
        let min_x_cell = (((min[0] - self.min[0]) / (self.max[0] - self.min[0]) * self.n as f64) as usize)
            .max(0)
            .min(self.n - 1);
        let max_x_cell = (((max[0] - self.min[0]) / (self.max[0] - self.min[0]) * self.n as f64) as usize)
            .max(0)
            .min(self.n - 1);

        let min_y_cell = (((min[1] - self.min[1]) / (self.max[1] - self.min[1]) * self.n as f64) as usize)
            .max(0)
            .min(self.n - 1);

        let max_y_cell = (((max[1] - self.min[1]) / (self.max[1] - self.min[1]) * self.n as f64) as usize)
            .max(0)
            .min(self.n - 1);

        let min_z_cell = (((min[2] - self.min[2]) / (self.max[2] - self.min[2]) * self.n as f64) as usize)
            .max(0)
            .min(self.n - 1);
        let max_z_cell = (((max[2] - self.min[2]) / (self.max[2] - self.min[2]) * self.n as f64) as usize)
            .max(0)
            .min(self.n - 1);

        let mut indices = vec![];
        for x in min_x_cell..=max_x_cell {
            for y in min_y_cell..=max_y_cell {
                for z in min_z_cell..=max_z_cell {
                    let idx = x * self.n * self.n + y * self.n + z;
                    indices.extend(self.grid[idx].iter());
                }
            }
        }
        indices
    }
}

struct UVIndex {
    rows: usize,
    cols: usize,
    min_u: f64,
    max_u: f64,
    min_v: f64,
    max_v: f64,
    grid: Vec<Vec<usize>>, // rows * cols cell with indices to the faces
}
impl UVIndex {
    fn new(object: &Object, rows: usize, cols: usize) -> Self {
        let mut grid = vec![vec![]; rows * cols];

        let min_u = object.tex_vertices.iter().map(|v| v.u).fold(f64::INFINITY, f64::min);
        let max_u = object
            .tex_vertices
            .iter()
            .map(|v| v.u)
            .fold(f64::NEG_INFINITY, f64::max);

        let min_v = object.tex_vertices.iter().map(|v| v.v).fold(f64::INFINITY, f64::min);
        let max_v = object
            .tex_vertices
            .iter()
            .map(|v| v.v)
            .fold(f64::NEG_INFINITY, f64::max);

        let du = (max_u - min_u) / cols as f64;
        let dv = (max_v - min_v) / rows as f64;

        for (i, s) in object.geometry[0].shapes.iter().enumerate() {
            match s.primitive {
                Primitive::Triangle(i1, i2, i3) => {
                    let v1 = &i1.1.unwrap();
                    let v2 = &i2.1.unwrap();
                    let v3 = &i3.1.unwrap();

                    let vert1 = object.tex_vertices[*v1];
                    let vert2 = object.tex_vertices[*v2];
                    let vert3 = object.tex_vertices[*v3];

                    let min_u_t = vert1.u.min(vert2.u).min(vert3.u);
                    let max_u_t = vert1.u.max(vert2.u).max(vert3.u);

                    let min_v_t = vert1.v.min(vert2.v).min(vert3.v);
                    let max_v_t = vert1.v.max(vert2.v).max(vert3.v);

                    let min_col = (((min_u_t - min_u) / du) as usize).max(0).min(cols - 1);
                    let max_col = (((max_u_t - min_u) / du) as usize).max(0).min(cols - 1);

                    let min_row = (((min_v_t - min_v) / dv) as usize).max(0).min(rows - 1);
                    let max_row = (((max_v_t - min_v) / dv) as usize).max(0).min(rows - 1);

                    for row in min_row..=max_row {
                        for col in min_col..=max_col {
                            let idx = row * cols + col;
                            grid[idx].push(i);
                        }
                    }
                }
                _ => todo!(),
            }
        }

        /* let num_cells = rows * cols;
        let max_occ = grid.iter().map(|g| g.len()).max().unwrap();
        println!("FaceIndex: {} cells, max occ: {}", num_cells, max_occ); */

        Self {
            rows,
            cols,
            min_u,
            max_u,
            min_v,
            max_v,
            grid,
        }
    }

    fn in_bounds(&self, min_u: f64, max_u: f64, min_v: f64, max_v: f64) -> Vec<usize> {
        let min_col = (((min_u - self.min_u) / (self.max_u - self.min_u) * self.cols as f64) as usize)
            .max(0)
            .min(self.cols - 1);
        let max_col = (((max_u - self.min_u) / (self.max_u - self.min_u) * self.cols as f64) as usize)
            .max(0)
            .min(self.cols - 1);

        let min_row = (((min_v - self.min_v) / (self.max_v - self.min_v) * self.rows as f64) as usize)
            .max(0)
            .min(self.rows - 1);
        let max_row = (((max_v - self.min_v) / (self.max_v - self.min_v) * self.rows as f64) as usize)
            .max(0)
            .min(self.rows - 1);

        let mut indices = vec![];
        for row in min_row..=max_row {
            for col in min_col..=max_col {
                let idx = row * self.cols + col;
                indices.extend(self.grid[idx].iter());
            }
        }

        indices
    }
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum ProjectionKind {
    None,
    OrthographicXZ,
}

pub struct ObjFile {
    object: Object,
    has_inverted_uv_tris: bool,
    uv_index: UVIndex,
    xyz_index: XYZIndex,
}
impl ObjFile {
    pub fn new(mut object: Object, transform: &Option<AffineTransform>, projection: ProjectionKind) -> Self {
        if let Some(AffineTransform { matrix: affine }) = transform {
            object.vertices.iter_mut().for_each(|v| {
                let x = affine[0][0] * v.x + affine[0][1] * v.y + affine[0][2] * v.z + affine[0][3];
                let y = affine[1][0] * v.x + affine[1][1] * v.y + affine[1][2] * v.z + affine[1][3];
                let z = affine[2][0] * v.x + affine[2][1] * v.y + affine[2][2] * v.z + affine[2][3];
                v.x = x;
                v.y = y;
                v.z = z;
            });
            object.normals.iter_mut().for_each(|n| {
                let nx = affine[0][0] * n.x + affine[0][1] * n.y + affine[0][2] * n.z;
                let ny = affine[1][0] * n.x + affine[1][1] * n.y + affine[1][2] * n.z;
                let nz = affine[2][0] * n.x + affine[2][1] * n.y + affine[2][2] * n.z;

                let norm = sqrt(nx * nx + ny * ny + nz * nz);
                n.x = nx / norm;
                n.y = ny / norm;
                n.z = nz / norm;
            });
        }

        if projection == ProjectionKind::OrthographicXZ {
            object.normals.iter_mut().for_each(|n| {
                n.x = 0.0;
                n.y = 1.0;
                n.z = 0.0;
            });

            let vert_clone = object.vertices.clone();
            let min_x = vert_clone.iter().map(|v| v.x).fold(f64::INFINITY, f64::min);
            let max_x = vert_clone.iter().map(|v| v.x).fold(f64::NEG_INFINITY, f64::max);

            let dx = max_x - min_x;

            let min_z = vert_clone.iter().map(|v| v.z).fold(f64::INFINITY, f64::min);
            let max_z = vert_clone.iter().map(|v| v.z).fold(f64::NEG_INFINITY, f64::max);
            let dz = max_z - min_z;

            object.tex_vertices.iter_mut().enumerate().for_each(|(idx, tex)| {
                // replace the existing uv by an orthographic projection into the xz plane
                let v = vert_clone[idx];
                let x = v.x;
                let z = v.z;

                tex.u = (x - min_x) / dx;
                tex.v = (z - min_z) / dz;
            });
        } else {
            // rescale tex to 0..1
            let min_u = object.tex_vertices.iter().map(|v| v.u).fold(f64::INFINITY, f64::min);
            let max_u = object
                .tex_vertices
                .iter()
                .map(|v| v.u)
                .fold(f64::NEG_INFINITY, f64::max);
            let du = max_u - min_u;
            let min_v = object.tex_vertices.iter().map(|v| v.v).fold(f64::INFINITY, f64::min);
            let max_v = object
                .tex_vertices
                .iter()
                .map(|v| v.v)
                .fold(f64::NEG_INFINITY, f64::max);
            let dv = max_v - min_v;
            object.tex_vertices.iter_mut().for_each(|tex| {
                tex.u = (tex.u - min_u) / du;
                tex.v = (tex.v - min_v) / dv;
            });
        }

        let has_inverted_uv_tris = Self::has_inverted_uv_tris(object.clone());
        let target_cell_num = 100.;
        let num_tris = object.geometry[0].shapes.len() as f64;
        let n = sqrt(num_tris / target_cell_num) as usize;
        let uv_index = UVIndex::new(&object, n, n);

        let n = pow(num_tris / target_cell_num, 1. / 3.) as usize;
        let xyz_index = XYZIndex::new(&object, n);

        Self {
            object,
            has_inverted_uv_tris,
            uv_index,
            xyz_index,
        }
    }
    fn has_inverted_uv_tris(obj: Object) -> bool {
        for s in obj.geometry[0].shapes.iter().skip(1) {
            match s.primitive {
                Primitive::Triangle(i1, i2, i3) => {
                    let v1 = &obj.tex_vertices[i1.1.unwrap()];
                    let v2 = &obj.tex_vertices[i2.1.unwrap()];
                    let v3 = &obj.tex_vertices[i3.1.unwrap()];

                    let u1 = (v1.u * 100000.) as i32;
                    let v1 = (v1.v * 100000.) as i32;

                    let u2 = (v2.u * 100000.) as i32;
                    let v2 = (v2.v * 100000.) as i32;

                    let u3 = (v3.u * 100000.) as i32;
                    let v3 = (v3.v * 100000.) as i32;

                    let u = (u1 + u2 + u3) / 3;
                    let v = (v1 + v2 + v3) / 3;

                    let w0 = orient2d(u2, v2, u3, v3, u, v);
                    let w1 = orient2d(u3, v3, u1, v1, u, v);
                    let w2 = orient2d(u1, v1, u2, v2, u, v);

                    return w0 >= 0 && w1 >= 0 && w2 >= 0;
                }
                _ => (),
            }
        }
        false
    }
}

#[derive(Clone)]
pub struct ObjVolume {
    volume: Volume,
    obj: Arc<ObjFile>,
    width: usize,
    height: usize,
}
impl ObjVolume {
    pub fn load_from_obj(
        obj_file_path: &str,
        base_volume: Volume,
        width: usize,
        height: usize,
        transform: &Option<AffineTransform>,
        projection: ProjectionKind,
    ) -> Self {
        Self::new(
            Arc::new(Self::load_obj(obj_file_path, transform, projection)),
            base_volume,
            width,
            height,
        )
    }
    pub fn new(obj: Arc<ObjFile>, base_volume: Volume, width: usize, height: usize) -> Self {
        Self {
            volume: base_volume,
            obj,
            width,
            height,
        }
    }

    pub fn load_obj(file_path: &str, transform: &Option<AffineTransform>, projection: ProjectionKind) -> ObjFile {
        let obj_file = std::fs::read_to_string(file_path).unwrap();
        // filter out opacity definitions that wavefront_obj does not cope well with
        let (not_faces, faces): (Vec<_>, Vec<_>) = obj_file
            .lines()
            .filter(|line| !line.starts_with("mtllib"))
            .filter(|line| !line.starts_with("usemtl"))
            .partition(|line| !line.starts_with("f"));

        let obj_file = not_faces
            .into_iter()
            .chain(faces.into_iter())
            .collect::<Vec<_>>()
            .join("\n");

        let obj_set = obj::parse(obj_file).unwrap();

        let mut objects = obj_set.objects;
        /* println!("Loaded obj file with {} objects", objects.len());
        for o in objects.iter() {
            println!(
                "Object: {}, geometries: {} vertices: {}",
                o.name,
                o.geometry.len(),
                o.vertices.len()
            );
        } */

        let object = objects.remove(0);
        ObjFile::new(object, transform, projection)
    }

    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }
    pub fn convert_to_volume_coords(&self, coord: [i32; 3]) -> [i32; 3] {
        let u = coord[0];
        let v = coord[1];
        let w = coord[2] as f64;

        let ut = u as f64 / self.width() as f64;
        let vt = self.v(v as f64 / self.height() as f64);

        let obj = &self.obj.object;
        for i in self.obj.uv_index.in_bounds(ut, ut, vt, vt) {
            let s = &obj.geometry[0].shapes[i];
            match s.primitive {
                Primitive::Triangle(i1, i2, i3) => {
                    let v1 = &obj.tex_vertices[i1.1.unwrap()];
                    let v2 = &obj.tex_vertices[i2.1.unwrap()];
                    let v3 = &obj.tex_vertices[i3.1.unwrap()];

                    let u1 = (v1.u * self.width() as f64) as i32;
                    let v1 = (self.v(v1.v) * self.height() as f64) as i32;

                    let u2 = (v2.u * self.width() as f64) as i32;
                    let v2 = (self.v(v2.v) * self.height() as f64) as i32;

                    let u3 = (v3.u * self.width() as f64) as i32;
                    let v3 = (self.v(v3.v) * self.height() as f64) as i32;

                    let min_u_t = u1.min(u2).min(u3);
                    let max_u_t = u1.max(u2).max(u3);

                    let min_v_t = v1.min(v2).min(v3);
                    let max_v_t = v1.max(v2).max(v3);

                    // broad pre-check against triangle bounding box
                    if min_u_t <= u && u <= max_u_t && min_v_t <= v && v <= max_v_t {
                        let w0 = orient2d(u2, v2, u3, v3, u, v);
                        let w1 = orient2d(u3, v3, u1, v1, u, v);
                        let w2 = orient2d(u1, v1, u2, v2, u, v);

                        if w0 >= 0 && w1 >= 0 && w2 >= 0 {
                            let xyz1 = &obj.vertices[i1.0];
                            let xyz2 = &obj.vertices[i2.0];
                            let xyz3 = &obj.vertices[i3.0];

                            // barymetric interpolation
                            let invwsum = 1. / (w0 + w1 + w2) as f64;
                            let x = (w0 as f64 * xyz1.x + w1 as f64 * xyz2.x + w2 as f64 * xyz3.x) * invwsum;
                            let y = (w0 as f64 * xyz1.y + w1 as f64 * xyz2.y + w2 as f64 * xyz3.y) * invwsum;
                            let z = (w0 as f64 * xyz1.z + w1 as f64 * xyz2.z + w2 as f64 * xyz3.z) * invwsum;

                            let (nx, ny, nz) = if coord[2] == 0 {
                                (0.0, 0.0, 0.0)
                            } else {
                                let nxyz1 = &obj.normals[i1.2.unwrap()];
                                let nxyz2 = &obj.normals[i2.2.unwrap()];
                                let nxyz3 = &obj.normals[i3.2.unwrap()];

                                let nx = (w0 as f64 * nxyz1.x + w1 as f64 * nxyz2.x + w2 as f64 * nxyz3.x) * invwsum;
                                let ny = (w0 as f64 * nxyz1.y + w1 as f64 * nxyz2.y + w2 as f64 * nxyz3.y) * invwsum;
                                let nz = (w0 as f64 * nxyz1.z + w1 as f64 * nxyz2.z + w2 as f64 * nxyz3.z) * invwsum;

                                (nx, ny, nz)
                            };

                            return [(x + w * nx) as i32, (y + w * ny) as i32, (z + w * nz) as i32];
                        }
                    }
                }
                _ => todo!(),
            }
        }

        [-1, -1, -1]
    }

    fn v(&self, v: f64) -> f64 {
        if self.obj.has_inverted_uv_tris {
            v
        } else {
            1.0 - v
        }
    }
}

fn orient2d(u1: i32, v1: i32, u2: i32, v2: i32, u3: i32, v3: i32) -> i32 {
    (u2 - u1) * (v3 - v1) - (v2 - v1) * (u3 - u1)
}

trait CompositionState {
    fn update(&mut self, a: u8) -> bool;
    fn result(&self, num_layers: u32) -> u8;
    fn reset(&mut self);
}
struct MaxCompositionState {
    value: u8,
}
impl MaxCompositionState {
    fn new() -> Self {
        Self { value: 0 }
    }
}
impl CompositionState for MaxCompositionState {
    fn update(&mut self, a: u8) -> bool {
        self.value = self.value.max(a);
        true
    }
    fn result(&self, _num_layers: u32) -> u8 {
        self.value
    }
    fn reset(&mut self) {
        self.value = 0;
    }
}
struct NoCompositionState;
impl CompositionState for NoCompositionState {
    fn update(&mut self, _a: u8) -> bool {
        false
    }
    fn result(&self, _num_layers: u32) -> u8 {
        0
    }
    fn reset(&mut self) {}
}

struct AlphaCompositionState {
    min: f32,
    max: f32,
    alpha_cutoff: f32,
    opacity: f32,
    value: f32,
    alpha: f32,
}
impl AlphaCompositionState {
    fn new(min: f32, max: f32, alpha_cutoff: f32, opacity: f32) -> Self {
        Self {
            min,
            max,
            alpha_cutoff,
            opacity: opacity,
            value: 0.0,
            alpha: 0.0,
        }
    }
}
impl CompositionState for AlphaCompositionState {
    fn update(&mut self, a: u8) -> bool {
        let value = ((a as f32 / 255.0 - self.min) / (self.max - self.min)).clamp(0.0, 1.0);

        if value == 0.0 {
            // speed through empty area
            return true;
        }

        let weight = (1.0 - self.alpha) * (value * self.opacity).min(1.0);
        self.value += weight * value;
        self.alpha += weight;

        return self.alpha < self.alpha_cutoff;
    }
    fn result(&self, _num_layers: u32) -> u8 {
        (self.value * 255.0).clamp(0.0, 255.0) as u8
    }
    fn reset(&mut self) {
        self.value = 0.0;
        self.alpha = 0.0;
    }
}

struct AlphaHeightMapCompositionState {
    min: f32,
    max: f32,
    alpha_cutoff: f32,
    opacity: f32,
    alpha: f32,
    depth: f32,
    weighted_depth: f32,
}
impl AlphaHeightMapCompositionState {
    fn new(min: f32, max: f32, alpha_cutoff: f32, opacity: f32) -> Self {
        Self {
            min,
            max,
            alpha_cutoff,
            opacity: opacity,
            alpha: 0.0,
            depth: 0.0,
            weighted_depth: 0.0,
        }
    }
}
impl CompositionState for AlphaHeightMapCompositionState {
    fn update(&mut self, a: u8) -> bool {
        let value = ((a as f32 / 255.0 - self.min) / (self.max - self.min)).clamp(0.0, 1.0);

        if value == 0.0 {
            // speed through empty area
            self.depth += 1.0;
            return true;
        }

        let weight = (1.0 - self.alpha) * (value * self.opacity).min(1.0);
        self.alpha += weight;
        self.weighted_depth += weight * self.depth;
        self.depth += 1.0;

        return self.alpha < self.alpha_cutoff;
    }
    fn result(&self, num_layers: u32) -> u8 {
        //((1.0 - self.weighted_depth * 4.0 / self.alpha / num_layers as f32) * 255.0).clamp(0.0, 255.0) as u8
        (255.0 - self.weighted_depth / self.alpha * 255.0 / num_layers as f32).clamp(0.0, 255.0) as u8
    }
    fn reset(&mut self) {
        self.depth = 0.0;
        self.weighted_depth = 0.0;
        self.alpha = 0.0;
    }
}

impl PaintVolume for ObjVolume {
    fn paint(
        &self,
        xyz: [i32; 3],
        u_coord: usize,
        v_coord: usize,
        plane_coord: usize,
        width: usize,
        height: usize,
        sfactor: u8,
        paint_zoom: u8,
        config: &super::DrawingConfig,
        buffer: &mut Image,
    ) {
        assert!(u_coord == 0);
        assert!(v_coord == 1);
        assert!(plane_coord == 2);

        let draw_outlines = config.draw_xyz_outlines;
        let composite = config.compositing.mode != CompositingMode::None;
        let composite_layers_in_front = config.compositing.layers_in_front as i32;
        let composite_layers_behind = config.compositing.layers_behind as i32;
        let composite_total_layers = composite_layers_in_front + composite_layers_behind + 1; // +1 for the current layer
        let mut composition: Box<dyn CompositionState> = match config.compositing.mode {
            CompositingMode::Max => Box::new(MaxCompositionState::new()),
            CompositingMode::Alpha => Box::new(AlphaCompositionState::new(
                config.compositing.alpha_min as f32 / 255.0,
                config.compositing.alpha_max as f32 / 255.0,
                config.compositing.alpha_threshold as f32 / 10000.0,
                config.compositing.opacity as f32 / 100.0,
            )),
            CompositingMode::AlphaHeightMap => Box::new(AlphaHeightMapCompositionState::new(
                config.compositing.alpha_min as f32 / 255.0,
                config.compositing.alpha_max as f32 / 255.0,
                config.compositing.alpha_threshold as f32 / 10000.0,
                config.compositing.opacity as f32 / 100.0,
            )),
            CompositingMode::None => Box::new(NoCompositionState {}),
        };
        let composite_direction = if config.compositing.reverse_direction { -1 } else { 1 };

        let real_xyz = if draw_outlines {
            self.convert_to_volume_coords(xyz)
        } else {
            [0, 0, 0]
        };

        let volume = self.volume.clone();

        let ffactor = sfactor as f64;

        let w_factor = xyz[2] as f64;

        let min_u = xyz[0] - width as i32 / 2 * paint_zoom as i32;
        let max_u = xyz[0] + width as i32 / 2 * paint_zoom as i32;
        let min_v = xyz[1] - height as i32 / 2 * paint_zoom as i32;
        let max_v = xyz[1] + height as i32 / 2 * paint_zoom as i32;

        let min_u_vt = min_u as f64 / self.width() as f64;
        let max_u_vt = max_u as f64 / self.width() as f64;

        let min_v_vt = self.v(min_v as f64 / self.height() as f64);
        let max_v_vt = self.v(max_v as f64 / self.height() as f64);

        let (min_v_vt, max_v_vt) = if self.obj.has_inverted_uv_tris {
            (min_v_vt, max_v_vt)
        } else {
            (max_v_vt, min_v_vt)
        };

        /* println!(
            "xyz: {:?}, width: {}, height: {}, sfactor: {}, paint_zoom: {}",
            xyz, width, height, sfactor, paint_zoom
        );
        println!("min_u: {}, max_u: {}, min_v: {}, max_v: {}", min_u, max_u, min_v, max_v); */

        let obj = &self.obj.object;
        //for s in obj.geometry[0].shapes.iter() {
        for i in self.obj.uv_index.in_bounds(min_u_vt, max_u_vt, min_v_vt, max_v_vt) {
            let s = &obj.geometry[0].shapes[i];
            match s.primitive {
                Primitive::Triangle(i1, i2, i3) => {
                    let vert1 = &obj.tex_vertices[i1.1.unwrap()];
                    let vert2 = &obj.tex_vertices[i2.1.unwrap()];
                    let vert3 = &obj.tex_vertices[i3.1.unwrap()];

                    let min_tri_u_t = vert1.u.min(vert2.u).min(vert3.u);
                    let max_tri_u_t = vert1.u.max(vert2.u).max(vert3.u);

                    let min_tri_v_t = vert1.v.min(vert2.v).min(vert3.v);
                    let max_tri_v_t = vert1.v.max(vert2.v).max(vert3.v);

                    // clip to paint area in texture coordinates
                    //if !(min_u_t > max_u || max_u_t < min_u || min_v_t > max_v || max_v_t < min_v) {
                    if !(min_tri_u_t > max_u_vt
                        || max_tri_u_t < min_u_vt
                        || min_tri_v_t > max_v_vt
                        || max_tri_v_t < min_v_vt)
                    {
                        /*
                        Implement the simple tri rasterization algorithm from https://fgiesen.wordpress.com/2013/02/08/triangle-rasterization-in-practice/

                        // Compute triangle bounding box
                        int minX = min3(v0.x, v1.x, v2.x);
                        int minY = min3(v0.y, v1.y, v2.y);
                        int maxX = max3(v0.x, v1.x, v2.x);
                        int maxY = max3(v0.y, v1.y, v2.y);

                        // Clip against screen bounds
                        minX = max(minX, 0);
                        minY = max(minY, 0);
                        maxX = min(maxX, screenWidth - 1);
                        maxY = min(maxY, screenHeight - 1);

                        // Rasterize
                        Point2D p;
                        for (p.y = minY; p.y <= maxY; p.y++) {
                            for (p.x = minX; p.x <= maxX; p.x++) {
                                // Determine barycentric coordinates
                                int w0 = orient2d(v1, v2, p);
                                int w1 = orient2d(v2, v0, p);
                                int w2 = orient2d(v0, v1, p);

                                // If p is on or inside all edges, render pixel.
                                if (w0 >= 0 && w1 >= 0 && w2 >= 0)
                                    renderPixel(p, w0, w1, w2);
                            }
                        } */

                        let u1 = (vert1.u * self.width() as f64) as i32;
                        let v1 = (self.v(vert1.v) * self.height() as f64) as i32;

                        let u2 = (vert2.u * self.width() as f64) as i32;
                        let v2 = (self.v(vert2.v) * self.height() as f64) as i32;

                        let u3 = (vert3.u * self.width() as f64) as i32;
                        let v3 = (self.v(vert3.v) * self.height() as f64) as i32;

                        /* println!(
                            "i1.1: {}, i2.1: {}, i3.1: {}",
                            i1.1.unwrap(),
                            i2.1.unwrap(),
                            i3.1.unwrap()
                        );
                        println!("vert1: {:?} vert2: {:?} vert3: {:?}", vert1, vert2, vert3);
                        println!("u1: {}, v1: {}, u2: {}, v2: {}, u3: {}, v3: {}", u1, v1, u2, v2, u3, v3); */

                        let u1i = u1 - min_u;
                        let v1i = v1 - min_v;

                        let u2i = u2 - min_u;
                        let v2i = v2 - min_v;

                        let u3i = u3 - min_u;
                        let v3i = v3 - min_v;

                        /* println!(
                            "u1i: {}, v1i: {}, u2i: {}, v2i: {}, u3i: {}, v3i: {}",
                            u1i, v1i, u2i, v2i, u3i, v3i
                        ); */

                        // calculate triangle bounding box in paint coordinates, make sure to slightly widen to align
                        // to paint_zoom to avoid gaps at the edges because of integer processing
                        let tmin_u = paint_zoom_align(u1i.min(u2i).min(u3i), paint_zoom).max(0);
                        let tmax_u = paint_zoom_align_up(u1i.max(u2i).max(u3i), paint_zoom)
                            .min(width as i32 * paint_zoom as i32 - 1);

                        let tmin_v = paint_zoom_align(v1i.min(v2i).min(v3i), paint_zoom).max(0);
                        let tmax_v = paint_zoom_align_up(v1i.max(v2i).max(v3i), paint_zoom)
                            .min(height as i32 * paint_zoom as i32 - 1);

                        /* println!(
                            "tmin_u: {}, tmax_u: {}, tmin_v: {}, tmax_v: {}",
                            tmin_u, tmax_u, tmin_v, tmax_v
                        ); */
                        // TODO: would probably better to operate in paint coordinates instead
                        for v in (tmin_v..=tmax_v).step_by(paint_zoom as usize) {
                            for u in (tmin_u..=tmax_u).step_by(paint_zoom as usize) {
                                let w0 = orient2d(u2i, v2i, u3i, v3i, u, v);
                                let w1 = orient2d(u3i, v3i, u1i, v1i, u, v);
                                let w2 = orient2d(u1i, v1i, u2i, v2i, u, v);

                                //println!("At u:{} v:{} w0: {}, w1: {}, w2: {}", u, v, w0, w1, w2);

                                if w0 >= 0 && w1 >= 0 && w2 >= 0 {
                                    if u >= 0
                                        && u < width as i32 * paint_zoom as i32
                                        && v >= 0
                                        && v < height as i32 * paint_zoom as i32
                                    {
                                        let xyz1 = &obj.vertices[i1.0];
                                        let xyz2 = &obj.vertices[i2.0];
                                        let xyz3 = &obj.vertices[i3.0];

                                        // barymetric interpolation
                                        let invwsum = 1. / (w0 + w1 + w2) as f64;
                                        let x =
                                            (w0 as f64 * xyz1.x + w1 as f64 * xyz2.x + w2 as f64 * xyz3.x) * invwsum;
                                        let y =
                                            (w0 as f64 * xyz1.y + w1 as f64 * xyz2.y + w2 as f64 * xyz3.y) * invwsum;
                                        let z =
                                            (w0 as f64 * xyz1.z + w1 as f64 * xyz2.z + w2 as f64 * xyz3.z) * invwsum;

                                        let (nx, ny, nz) = if xyz[2] == 0 && !composite {
                                            (0.0, 0.0, 0.0)
                                        } else {
                                            let nxyz1 = &obj.normals[i1.2.unwrap()];
                                            let nxyz2 = &obj.normals[i2.2.unwrap()];
                                            let nxyz3 = &obj.normals[i3.2.unwrap()];

                                            let nx = (w0 as f64 * nxyz1.x + w1 as f64 * nxyz2.x + w2 as f64 * nxyz3.x)
                                                * invwsum;
                                            let ny = (w0 as f64 * nxyz1.y + w1 as f64 * nxyz2.y + w2 as f64 * nxyz3.y)
                                                * invwsum;
                                            let nz = (w0 as f64 * nxyz1.z + w1 as f64 * nxyz2.z + w2 as f64 * nxyz3.z)
                                                * invwsum;

                                            (nx, ny, nz)
                                        };

                                        let value = if !composite {
                                            let x = x + w_factor * nx;
                                            let y = y + w_factor * ny;
                                            let z = z + w_factor * nz;

                                            if config.trilinear_interpolation {
                                                volume.get_interpolated(
                                                    [x / ffactor, y / ffactor, z / ffactor],
                                                    sfactor as i32,
                                                )
                                            } else {
                                                volume.get([x / ffactor, y / ffactor, z / ffactor], sfactor as i32)
                                            }
                                        } else {
                                            composition.reset();
                                            let start = xyz[2] + composite_direction * composite_layers_in_front;
                                            let end = xyz[2] - composite_direction * (composite_layers_behind + 1);

                                            let step = if start < end { 1 } else { -1 };

                                            let mut w = start;
                                            while w != end {
                                                let w_factor = w as f64;

                                                let x = x + w_factor * nx;
                                                let y = y + w_factor * ny;
                                                let z = z + w_factor * nz;

                                                let new_value = if config.trilinear_interpolation {
                                                    volume.get_interpolated(
                                                        [x / ffactor, y / ffactor, z / ffactor],
                                                        sfactor as i32,
                                                    )
                                                } else {
                                                    volume.get([x / ffactor, y / ffactor, z / ffactor], sfactor as i32)
                                                };

                                                if !composition.update(new_value) {
                                                    break;
                                                }

                                                w += step;
                                            }
                                            composition.result(composite_total_layers as u32)
                                        };

                                        buffer.set_gray(
                                            u as usize / paint_zoom as usize,
                                            v as usize / paint_zoom as usize,
                                            value,
                                        );

                                        if draw_outlines {
                                            if (x - real_xyz[0] as f64).abs() < 2.0 {
                                                buffer.set_rgb(
                                                    u as usize / paint_zoom as usize,
                                                    v as usize / paint_zoom as usize,
                                                    0,
                                                    0,
                                                    255,
                                                );
                                            } else if (y - real_xyz[1] as f64).abs() < 2.0 {
                                                buffer.set_rgb(
                                                    u as usize / paint_zoom as usize,
                                                    v as usize / paint_zoom as usize,
                                                    255,
                                                    0,
                                                    0,
                                                );
                                            } else if (z - real_xyz[2] as f64).abs() < 2.0 {
                                                buffer.set_rgb(
                                                    u as usize / paint_zoom as usize,
                                                    v as usize / paint_zoom as usize,
                                                    0,
                                                    255,
                                                    0,
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                _ => todo!(),
            }
        }
    }
    fn shared(&self) -> super::VolumeCons {
        let obj = self.obj.clone();
        let width = self.width;
        let height = self.height;
        let volume = self.volume.shared();

        Box::new(move || {
            ObjVolume {
                volume: volume(),
                obj,
                width,
                height,
            }
            .into_volume()
        })
    }
}

impl SurfaceVolume for ObjVolume {
    fn paint_plane_intersection(
        &self,
        xyz: [i32; 3],
        u_coord: usize,
        v_coord: usize,
        plane_coord: usize,
        width: usize,
        height: usize,
        _sfactor: u8,
        paint_zoom: u8,
        highlight_uv_section: Option<[i32; 3]>,
        config: &super::DrawingConfig,
        image: &mut Image,
    ) {
        let u = xyz[u_coord];
        let v = xyz[v_coord];
        let w = xyz[plane_coord];

        let min_u = u - width as i32 / 2 * paint_zoom as i32;
        let max_u = u + width as i32 / 2 * paint_zoom as i32;
        let min_v = v - height as i32 / 2 * paint_zoom as i32;
        let max_v = v + height as i32 / 2 * paint_zoom as i32;

        let (uv_section_min, uv_section_max) = if let Some(h) = highlight_uv_section {
            (
                [
                    (h[0] as f64 - width as f64 / 2. * paint_zoom as f64) / self.width as f64,
                    (h[1] as f64 - height as f64 / 2. * paint_zoom as f64) / self.height as f64,
                ],
                [
                    (h[0] as f64 + width as f64 / 2. * paint_zoom as f64) / self.width as f64,
                    (h[1] as f64 + height as f64 / 2. * paint_zoom as f64) / self.height as f64,
                ],
            )
        } else {
            ([0f64, 0f64], [0f64, 0f64])
        };

        let draw_outline_vertices = config.draw_outline_vertices;

        let mut mins = [0.0, 0.0, 0.0];
        let mut maxs = [0.0, 0.0, 0.0];

        mins[u_coord] = min_u as f64;
        maxs[u_coord] = max_u as f64;

        mins[v_coord] = min_v as f64;
        maxs[v_coord] = max_v as f64;

        mins[plane_coord] = w as f64;
        maxs[plane_coord] = w as f64;

        let obj = &self.obj.object;
        for i in self.obj.xyz_index.in_bounds(mins, maxs) {
            let s = &obj.geometry[0].shapes[i];
            match s.primitive {
                Primitive::Triangle(i1, i2, i3) => {
                    let v1 = &obj.vertices[i1.0];
                    let v2 = &obj.vertices[i2.0];
                    let v3 = &obj.vertices[i3.0];

                    let minx = v1.x.min(v2.x).min(v3.x);
                    let maxx = v1.x.max(v2.x).max(v3.x);

                    let miny = v1.y.min(v2.y).min(v3.y);
                    let maxy = v1.y.max(v2.y).max(v3.y);

                    let minz = v1.z.min(v2.z).min(v3.z);
                    let maxz = v1.z.max(v2.z).max(v3.z);

                    // clip to paint area
                    if !(minx > maxs[0]
                        || maxx < mins[0]
                        || miny > maxs[1]
                        || maxy < mins[1]
                        || minz > maxs[2]
                        || maxz < mins[2])
                    {
                        fn coord(v: &Vertex) -> [f64; 3] {
                            [v.x, v.y, v.z]
                        }
                        fn intersects(v1: &Vertex, v2: &Vertex, w: i32, plane_coord: usize) -> bool {
                            (coord(v1)[plane_coord] - w as f64).signum() != (coord(v2)[plane_coord] - w as f64).signum()
                        }

                        let mut points: Vec<[f64; 3]> = vec![];

                        let mut add_intersection_points = |v1: &Vertex, v2: &Vertex| {
                            if intersects(v1, v2, w, plane_coord) {
                                let x1 = coord(v1)[u_coord];
                                let y1 = coord(v1)[v_coord];

                                let x2 = coord(v2)[u_coord];
                                let y2 = coord(v2)[v_coord];

                                let d1 = w as f64 - coord(v1)[plane_coord];
                                let d2 = w as f64 - coord(v2)[plane_coord];

                                let t = d1 / (d1 - d2);

                                let mut coords = [0.0, 0.0, 0.0];
                                coords[u_coord] = x1 + t * (x2 - x1);
                                coords[v_coord] = y1 + t * (y2 - y1);
                                coords[plane_coord] = w as f64;

                                points.push(coords);
                            }
                        };

                        add_intersection_points(v1, v2);
                        add_intersection_points(v2, v3);
                        add_intersection_points(v3, v1);

                        if points.len() == 2 {
                            let should_highlight = {
                                if highlight_uv_section.is_some() {
                                    let v1t = &obj.tex_vertices[i1.1.unwrap()];
                                    let v2t = &obj.tex_vertices[i2.1.unwrap()];
                                    let v3t = &obj.tex_vertices[i3.1.unwrap()];

                                    let min_u_t = v1t.u.min(v2t.u).min(v3t.u);
                                    let max_u_t = v1t.u.max(v2t.u).max(v3t.u);

                                    let min_v_t = self.v(v1t.v).min(self.v(v2t.v)).min(self.v(v3t.v));
                                    let max_v_t = self.v(v1t.v).max(self.v(v2t.v)).max(self.v(v3t.v));

                                    min_u_t <= uv_section_max[0] as f64
                                        && max_u_t >= uv_section_min[0] as f64
                                        && min_v_t <= uv_section_max[1] as f64
                                        && max_v_t >= uv_section_min[1] as f64
                                } else {
                                    false
                                }
                            };

                            let p1 = points[0];
                            let p2 = points[1];

                            // convert to image coordinates
                            let x0 = ((p1[u_coord] - min_u as f64) / paint_zoom as f64) as i32;
                            let y0 = ((p1[v_coord] - min_v as f64) / paint_zoom as f64) as i32;

                            let x1 = ((p2[u_coord] - min_u as f64) / paint_zoom as f64) as i32;
                            let y1 = ((p2[v_coord] - min_v as f64) / paint_zoom as f64) as i32;

                            let (r, g, b, rp, gp, bp) = if should_highlight {
                                (0xff, 0xaa, 0, 0, 0xff, 0)
                            } else {
                                (0xff, 0, 0xff, 0, 0, 0xff)
                            };

                            line(x0, y0, x1, y1, image, width, height, r, g, b);
                            if draw_outline_vertices {
                                point(x0, y0, image, 4, rp, gp, bp);
                                point(x1, y1, image, 4, rp, gp, bp);
                            }
                        }
                    }
                }
                _ => todo!(),
            }
        }
    }
}

// align values to paint_zoom to avoid gaps because of integer processing
fn paint_zoom_align(v: i32, paint_zoom: u8) -> i32 {
    let v = v / paint_zoom as i32;
    v * paint_zoom as i32
}
fn paint_zoom_align_up(v: i32, paint_zoom: u8) -> i32 {
    paint_zoom_align(v + paint_zoom as i32 - 1, paint_zoom)
}

#[allow(dead_code)] // useful for debugging triangle shapes
fn line(x0: i32, y0: i32, x1: i32, y1: i32, buffer: &mut Image, width: usize, height: usize, r: u8, g: u8, b: u8) {
    // simple bresenham algorithm from https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm

    let dx = (x1 as i32 - x0 as i32).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 as i32 - y0 as i32).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut error = dx + dy;

    let mut x = x0 as i32;
    let mut y = y0 as i32;

    while x != x1 || y != y1 {
        //println!("x: {}, y: {}", x, y);
        if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
            buffer.set_rgb(x as usize, y as usize, r, g, b);
        }

        let e2 = 2 * error;
        if e2 >= dy {
            error += dy;
            x += sx;
        }
        if e2 <= dx {
            error += dx;
            y += sy;
        }
    }
}

fn point(x0: i32, y0: i32, buffer: &mut Image, width: usize, r: u8, g: u8, b: u8) {
    let halfw = width as i32 / 2;
    for x in x0 - halfw..x0 + halfw {
        for y in y0 - halfw..y0 + halfw {
            if x >= 0 && x < buffer.width as i32 && y >= 0 && y < buffer.height as i32 {
                buffer.set_rgb(x as usize, y as usize, r, g, b);
            }
        }
    }
}

impl VoxelVolume for ObjVolume {
    fn get(&self, _xyz: [f64; 3], _downsampling: i32) -> u8 {
        todo!()
    }
    fn reset_for_painting(&self) {
        self.volume.reset_for_painting();
    }
}
