use super::{Image, OutlineSegment, PaintVolume, SurfaceVolume, Volume, VoxelVolume};
use crate::volume::composition::{
    AlphaCompositionState, AlphaHeightMapCompositionState, Compositor, MaxCompositionState, NoCompositionState,
    OverlayCombineParams,
};
use crate::volume::{AffineTransform, CompositingMode, VoxelPaintVolume};
use libm::{pow, sqrt};
use std::sync::Arc;
use std::time::Instant;
use wavefront_obj::obj::{self, Object, Primitive, TVertex, Vertex};

/// Compact per-triangle indices: vertex/tex/normal for each of the 3 corners.
/// Replaces wavefront_obj's `Shape` (~176 B with allocator-pressure inner Vecs)
/// with 36 B of plain POD that clones as a single memcpy.
#[derive(Copy, Clone, Debug)]
pub struct Triangle {
    pub v: [u32; 3],
    pub t: [u32; 3],
    pub n: [u32; 3],
}

// TODO: create a single AABB index for both XYZ and UV index
struct XYZIndex {
    n: usize, // same in every direction
    min: [f64; 3],
    max: [f64; 3],
    grid: Vec<Vec<usize>>, // n * n * n cell with indices to the faces
}
impl XYZIndex {
    fn new(vertices: &[Vertex], triangles: &[Triangle], n: usize) -> Self {
        let mut grid = vec![vec![]; n * n * n];

        fn minmax(vertices: &[Vertex], coord: impl Fn(&Vertex) -> f64) -> (f64, f64) {
            (
                vertices.iter().map(|v| coord(v)).fold(f64::INFINITY, f64::min),
                vertices.iter().map(|v| coord(v)).fold(f64::NEG_INFINITY, f64::max),
            )
        }

        let (min_x, max_x) = minmax(vertices, |v| v.x);
        let (min_y, max_y) = minmax(vertices, |v| v.y);
        let (min_z, max_z) = minmax(vertices, |v| v.z);

        let dx = (max_x - min_x) / n as f64;
        let dy = (max_y - min_y) / n as f64;
        let dz = (max_z - min_z) / n as f64;

        for (i, tri) in triangles.iter().enumerate() {
            let vert1 = vertices[tri.v[0] as usize];
            let vert2 = vertices[tri.v[1] as usize];
            let vert3 = vertices[tri.v[2] as usize];

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
    fn new(tex_vertices: &[TVertex], triangles: &[Triangle], rows: usize, cols: usize) -> Self {
        let mut grid = vec![vec![]; rows * cols];

        let min_u = tex_vertices.iter().map(|v| v.u).fold(f64::INFINITY, f64::min);
        let max_u = tex_vertices.iter().map(|v| v.u).fold(f64::NEG_INFINITY, f64::max);

        let min_v = tex_vertices.iter().map(|v| v.v).fold(f64::INFINITY, f64::min);
        let max_v = tex_vertices.iter().map(|v| v.v).fold(f64::NEG_INFINITY, f64::max);

        let du = (max_u - min_u) / cols as f64;
        let dv = (max_v - min_v) / rows as f64;

        for (i, tri) in triangles.iter().enumerate() {
            let vert1 = tex_vertices[tri.t[0] as usize];
            let vert2 = tex_vertices[tri.t[1] as usize];
            let vert3 = tex_vertices[tri.t[2] as usize];

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

#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
pub enum ProjectionKind {
    None,
    OrthographicXZ,
}

/// Affine-independent state shared by all `ObjFile`s built from the same source mesh.
/// `pre_vertices`/`pre_normals` are kept so that switching the base volume only needs to
/// clone them, apply the new affine, and rebuild the per-affine `XYZIndex`. Cacheable
/// across base volume switches for `ProjectionKind::None`; `OrthographicXZ` constructs
/// a private base inline since its projection reads post-affine vertex positions.
struct ObjFileBase {
    pre_vertices: Vec<Vertex>,
    pre_normals: Vec<Vertex>,
    tex_vertices: Vec<TVertex>,
    triangles: Vec<Triangle>,
    has_inverted_uv_tris: bool,
    uv_index: Arc<UVIndex>,
    projection_dimensions: Option<(usize, usize)>,
}

impl ObjFileBase {
    /// Build a cacheable base for `ProjectionKind::None`. UV rescale + uv_index +
    /// inversion check are all UV-only so the result is independent of any later affine.
    fn new_none(object: Object) -> Self {
        let t_extract = Instant::now();
        let (pre_vertices, pre_normals, mut tex_vertices, triangles) = extract_compact(object);
        log::info!(
            "ObjFileBase::new_none extract took {:?} (tris={}, verts={}, tex={})",
            t_extract.elapsed(),
            triangles.len(),
            pre_vertices.len(),
            tex_vertices.len()
        );

        let t_proj = Instant::now();
        let min_u = tex_vertices.iter().map(|v| v.u).fold(f64::INFINITY, f64::min);
        let max_u = tex_vertices.iter().map(|v| v.u).fold(f64::NEG_INFINITY, f64::max);
        let du = max_u - min_u;
        let min_v = tex_vertices.iter().map(|v| v.v).fold(f64::INFINITY, f64::min);
        let max_v = tex_vertices.iter().map(|v| v.v).fold(f64::NEG_INFINITY, f64::max);
        let dv = max_v - min_v;
        tex_vertices.iter_mut().for_each(|tex| {
            tex.u = (tex.u - min_u) / du;
            tex.v = (tex.v - min_v) / dv;
        });
        log::info!("ObjFileBase::new_none uv_rescale took {:?}", t_proj.elapsed());

        let t_inv = Instant::now();
        let has_inverted_uv_tris = compute_has_inverted_uv_tris(&tex_vertices, &triangles);
        log::info!("ObjFileBase::new_none inversion_check took {:?}", t_inv.elapsed());

        let target_cell_num = 100.;
        let num_tris = triangles.len() as f64;

        let t_uv = Instant::now();
        let n = sqrt(num_tris / target_cell_num) as usize;
        let uv_index = Arc::new(UVIndex::new(&tex_vertices, &triangles, n, n));
        log::info!("ObjFileBase::new_none uv_index took {:?}", t_uv.elapsed());

        Self {
            pre_vertices,
            pre_normals,
            tex_vertices,
            triangles,
            has_inverted_uv_tris,
            uv_index,
            projection_dimensions: None,
        }
    }
}

pub struct ObjFile {
    base: Arc<ObjFileBase>,
    vertices: Vec<Vertex>, // post-affine
    normals: Vec<Vertex>,  // post-affine
    xyz_index: XYZIndex,
}
impl ObjFile {
    pub fn new(object: Object, transform: &Option<AffineTransform>, projection: ProjectionKind) -> Self {
        match projection {
            ProjectionKind::None => {
                let base = Arc::new(ObjFileBase::new_none(object));
                Self::from_base(&base, transform)
            }
            ProjectionKind::OrthographicXZ => Self::new_orthographic(object, transform),
        }
    }

    /// Build an ObjFile from a (typically cached) affine-independent base. Clones the
    /// pre-affine vertex/normal arrays, applies the affine in-place, then builds xyz_index.
    /// `tex_vertices` and `triangles` stay shared with the base via `Arc`.
    fn from_base(base: &Arc<ObjFileBase>, transform: &Option<AffineTransform>) -> Self {
        let t_clone = Instant::now();
        let mut vertices = base.pre_vertices.clone();
        let mut normals = base.pre_normals.clone();
        log::info!("ObjFile::from_base v/n clone took {:?}", t_clone.elapsed());

        let t_affine = Instant::now();
        Self::apply_affine(&mut vertices, &mut normals, transform);
        log::info!("ObjFile::from_base affine took {:?}", t_affine.elapsed());

        let t_xyz = Instant::now();
        let target_cell_num = 100.;
        let num_tris = base.triangles.len() as f64;
        let n = pow(num_tris / target_cell_num, 1. / 3.) as usize;
        let xyz_index = XYZIndex::new(&vertices, &base.triangles, n);
        log::info!("ObjFile::from_base xyz_index took {:?}", t_xyz.elapsed());

        Self {
            base: Arc::clone(base),
            vertices,
            normals,
            xyz_index,
        }
    }

    /// Non-cacheable construction for OrthographicXZ. The projection reads post-affine
    /// vertices so the base is not reusable across affines — built once and wrapped in an
    /// Arc only for layout uniformity.
    fn new_orthographic(object: Object, transform: &Option<AffineTransform>) -> Self {
        let t_extract = Instant::now();
        let (mut vertices, mut normals, mut tex_vertices, triangles) = extract_compact(object);
        log::info!("ObjFile::new_orthographic extract took {:?}", t_extract.elapsed());

        let t_affine = Instant::now();
        Self::apply_affine(&mut vertices, &mut normals, transform);
        log::info!("ObjFile::new_orthographic affine took {:?}", t_affine.elapsed());

        let t_proj = Instant::now();
        normals.iter_mut().for_each(|n| {
            n.x = 0.0;
            n.y = 1.0;
            n.z = 0.0;
        });
        let min_x = vertices.iter().map(|v| v.x).fold(f64::INFINITY, f64::min);
        let max_x = vertices.iter().map(|v| v.x).fold(f64::NEG_INFINITY, f64::max);
        let dx = max_x - min_x;
        let min_z = vertices.iter().map(|v| v.z).fold(f64::INFINITY, f64::min);
        let max_z = vertices.iter().map(|v| v.z).fold(f64::NEG_INFINITY, f64::max);
        let dz = max_z - min_z;
        for (idx, tex) in tex_vertices.iter_mut().enumerate() {
            let v = vertices[idx];
            tex.u = (v.x - min_x) / dx;
            tex.v = (v.z - min_z) / dz;
        }
        let projection_dimensions = Some((dx.ceil() as usize, dz.ceil() as usize));
        log::info!("ObjFile::new_orthographic projection took {:?}", t_proj.elapsed());

        let t_inv = Instant::now();
        let has_inverted_uv_tris = compute_has_inverted_uv_tris(&tex_vertices, &triangles);
        log::info!("ObjFile::new_orthographic inversion_check took {:?}", t_inv.elapsed());

        let target_cell_num = 100.;
        let num_tris = triangles.len() as f64;

        let t_uv = Instant::now();
        let n = sqrt(num_tris / target_cell_num) as usize;
        let uv_index = Arc::new(UVIndex::new(&tex_vertices, &triangles, n, n));
        log::info!("ObjFile::new_orthographic uv_index took {:?}", t_uv.elapsed());

        let t_xyz = Instant::now();
        let n = pow(num_tris / target_cell_num, 1. / 3.) as usize;
        let xyz_index = XYZIndex::new(&vertices, &triangles, n);
        log::info!("ObjFile::new_orthographic xyz_index took {:?}", t_xyz.elapsed());

        // pre_* fields are unused for orthographic (no caching) but populated for layout
        // uniformity. The cost is one extra clone — only paid on the one-shot ortho path.
        let base = Arc::new(ObjFileBase {
            pre_vertices: vertices.clone(),
            pre_normals: normals.clone(),
            tex_vertices,
            triangles,
            has_inverted_uv_tris,
            uv_index,
            projection_dimensions,
        });

        Self {
            base,
            vertices,
            normals,
            xyz_index,
        }
    }

    fn apply_affine(vertices: &mut [Vertex], normals: &mut [Vertex], transform: &Option<AffineTransform>) {
        if let Some(AffineTransform { matrix: affine }) = transform {
            vertices.iter_mut().for_each(|v| {
                let x = affine[0][0] * v.x + affine[0][1] * v.y + affine[0][2] * v.z + affine[0][3];
                let y = affine[1][0] * v.x + affine[1][1] * v.y + affine[1][2] * v.z + affine[1][3];
                let z = affine[2][0] * v.x + affine[2][1] * v.y + affine[2][2] * v.z + affine[2][3];
                v.x = x;
                v.y = y;
                v.z = z;
            });
            normals.iter_mut().for_each(|n| {
                let nx = affine[0][0] * n.x + affine[0][1] * n.y + affine[0][2] * n.z;
                let ny = affine[1][0] * n.x + affine[1][1] * n.y + affine[1][2] * n.z;
                let nz = affine[2][0] * n.x + affine[2][1] * n.y + affine[2][2] * n.z;

                let norm = sqrt(nx * nx + ny * ny + nz * nz);
                n.x = nx / norm;
                n.y = ny / norm;
                n.z = nz / norm;
            });
        }
    }
}

/// Consume a wavefront_obj `Object` and produce the compact representation we actually
/// use. Drops `groups`/`smoothing_groups` per shape (always empty in our files), drops
/// non-triangle primitives, and packs VTN indices into a 36-byte `Triangle` (vs ~176 B
/// for `wavefront_obj::Shape`). The original `Object` is consumed and freed at the end.
fn extract_compact(object: Object) -> (Vec<Vertex>, Vec<Vertex>, Vec<TVertex>, Vec<Triangle>) {
    let Object {
        vertices,
        normals,
        tex_vertices,
        geometry,
        ..
    } = object;
    let shapes = geometry.into_iter().next().map(|g| g.shapes).unwrap_or_default();
    let mut triangles = Vec::with_capacity(shapes.len());
    for s in shapes {
        if let Primitive::Triangle(i1, i2, i3) = s.primitive {
            triangles.push(Triangle {
                v: [i1.0 as u32, i2.0 as u32, i3.0 as u32],
                t: [
                    i1.1.unwrap_or(0) as u32,
                    i2.1.unwrap_or(0) as u32,
                    i3.1.unwrap_or(0) as u32,
                ],
                n: [
                    i1.2.unwrap_or(0) as u32,
                    i2.2.unwrap_or(0) as u32,
                    i3.2.unwrap_or(0) as u32,
                ],
            });
        }
    }
    (vertices, normals, tex_vertices, triangles)
}

fn compute_has_inverted_uv_tris(tex_vertices: &[TVertex], triangles: &[Triangle]) -> bool {
    for tri in triangles.iter().skip(1) {
        let v1 = &tex_vertices[tri.t[0] as usize];
        let v2 = &tex_vertices[tri.t[1] as usize];
        let v3 = &tex_vertices[tri.t[2] as usize];

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
    false
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
        let t_total = Instant::now();
        let obj_file = Self::load_obj(obj_file_path, transform, projection);
        let result = Self::from_obj_file(obj_file, base_volume, width, height);
        log::info!("ObjVolume::load_from_obj total took {:?}", t_total.elapsed());
        result
    }

    /// Produce a new `ObjVolume` sharing this one's `Arc<ObjFileBase>` (parsed mesh + UV
    /// index + inversion check + tex vertices) but with a fresh affine applied to a clone
    /// of the pre-affine vertices/normals, and a fresh `XYZIndex` built against them.
    /// Used to switch the base volume under an obj segment (Ctrl+1/2/3) without re-parsing
    /// the obj file. Only valid for `ProjectionKind::None` bases — orthographic projections
    /// embed post-affine positions into their UVs and must go through `load_from_obj`.
    pub fn with_base(
        &self,
        base_volume: Volume,
        width: usize,
        height: usize,
        transform: &Option<AffineTransform>,
    ) -> Self {
        let t_total = Instant::now();
        let obj_file = ObjFile::from_base(&self.obj.base, transform);
        let result = Self::from_obj_file(obj_file, base_volume, width, height);
        log::info!("ObjVolume::with_base total took {:?}", t_total.elapsed());
        result
    }

    /// Wrap an already-built ObjFile in an ObjVolume, picking the right final width/height
    /// (orthographic projections override the requested dims with the projected extents).
    pub fn from_obj_file(obj_file: ObjFile, base_volume: Volume, width: usize, height: usize) -> Self {
        let obj_file = Arc::new(obj_file);
        let (final_width, final_height) = if let Some((proj_width, proj_height)) = obj_file.base.projection_dimensions {
            if proj_width != width || proj_height != height {
                println!(
                    "Note: Adjusting dimensions from {}x{} to {}x{} to maintain 1:1 scale for orthographic projection",
                    width, height, proj_width, proj_height
                );
            }
            (proj_width, proj_height)
        } else {
            (width, height)
        };
        Self::new(obj_file, base_volume, final_width, final_height)
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
        let object = Self::parse_object(file_path);
        ObjFile::new(object, transform, projection)
    }

    fn parse_object(file_path: &str) -> Object {
        log::info!("ObjVolume::parse_object path={}", file_path);
        let t_read = Instant::now();
        let obj_file = std::fs::read_to_string(file_path).unwrap();
        log::info!(
            "ObjVolume::parse_object read took {:?} ({} bytes)",
            t_read.elapsed(),
            obj_file.len()
        );

        let t_part = Instant::now();
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
        log::info!("ObjVolume::parse_object partition took {:?}", t_part.elapsed());

        let t_parse = Instant::now();
        let obj_set = obj::parse(obj_file).unwrap();
        log::info!("ObjVolume::parse_object parse took {:?}", t_parse.elapsed());

        let mut objects = obj_set.objects;
        objects.remove(0)
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

        let obj = &self.obj;
        for i in obj.base.uv_index.in_bounds(ut, ut, vt, vt) {
            let tri = &obj.base.triangles[i];
            let v1 = &obj.base.tex_vertices[tri.t[0] as usize];
            let v2 = &obj.base.tex_vertices[tri.t[1] as usize];
            let v3 = &obj.base.tex_vertices[tri.t[2] as usize];

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
                    let xyz1 = &obj.vertices[tri.v[0] as usize];
                    let xyz2 = &obj.vertices[tri.v[1] as usize];
                    let xyz3 = &obj.vertices[tri.v[2] as usize];

                    // barymetric interpolation
                    let invwsum = 1. / (w0 + w1 + w2) as f64;
                    let x = (w0 as f64 * xyz1.x + w1 as f64 * xyz2.x + w2 as f64 * xyz3.x) * invwsum;
                    let y = (w0 as f64 * xyz1.y + w1 as f64 * xyz2.y + w2 as f64 * xyz3.y) * invwsum;
                    let z = (w0 as f64 * xyz1.z + w1 as f64 * xyz2.z + w2 as f64 * xyz3.z) * invwsum;

                    let (nx, ny, nz) = if coord[2] == 0 {
                        (0.0, 0.0, 0.0)
                    } else {
                        let nxyz1 = &obj.normals[tri.n[0] as usize];
                        let nxyz2 = &obj.normals[tri.n[1] as usize];
                        let nxyz3 = &obj.normals[tri.n[2] as usize];

                        let nx = (w0 as f64 * nxyz1.x + w1 as f64 * nxyz2.x + w2 as f64 * nxyz3.x) * invwsum;
                        let ny = (w0 as f64 * nxyz1.y + w1 as f64 * nxyz2.y + w2 as f64 * nxyz3.y) * invwsum;
                        let nz = (w0 as f64 * nxyz1.z + w1 as f64 * nxyz2.z + w2 as f64 * nxyz3.z) * invwsum;

                        (nx, ny, nz)
                    };

                    return [(x + w * nx) as i32, (y + w * ny) as i32, (z + w * nz) as i32];
                }
            }
        }

        [-1, -1, -1]
    }

    fn v(&self, v: f64) -> f64 {
        if self.obj.base.has_inverted_uv_tris {
            v
        } else {
            1.0 - v
        }
    }

    /// Generalized cross-section paint for the UW/VW panes.
    ///
    /// One screen axis is `w` (offset along the surface normal); the other is a
    /// surface coordinate (`u` for UW, `v` for VW). The third surface coordinate
    /// is held fixed at `xyz[plane_coord]`. We follow the user's plan: find the
    /// triangles crossing the fixed surface line (a degenerate uv-index slab),
    /// rasterize each restricted to that line to build the surface profile
    /// `(pos, normal)` per line pixel, then expand each profile point along its
    /// normal across the whole `w` extent via the backend's
    /// `gather_along_normal` fast path. `w` is a spatial screen axis here, so
    /// this is always single-sample — compositing only applies to the UV pane.
    fn paint_cross_section(
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
        let volume = self.volume.clone();
        let inv_f = 1.0 / sfactor as f64;
        let pz = paint_zoom as i32;

        // `w` is the screen axis whose coord index is 2. For UW it is vertical
        // (v_coord == 2); for VW it is horizontal (u_coord == 2).
        let w_is_vertical = v_coord == 2;
        let (line_dim, w_dim) = if w_is_vertical { (width, height) } else { (height, width) };
        let line_axis = if w_is_vertical { u_coord } else { v_coord }; // surface axis in {0,1}
        let fixed_axis = plane_coord; // the other surface axis in {0,1}

        let origin_line = xyz[line_axis] - line_dim as i32 / 2 * pz;
        let line_hi = origin_line + (line_dim as i32 - 1) * pz;
        let origin_w = (xyz[2] - w_dim as i32 / 2 * pz) as f64;
        let w_hi = origin_w + (w_dim as i32 * pz) as f64;
        let fixed_val = xyz[fixed_axis];

        let fw = self.width() as f64;
        let fh = self.height() as f64;

        // Texture-space slab covering the slice line for the uv index. Segment
        // voxel coords map to texture coords as u_tex = u/width and
        // v_tex = self.v(v/height) (self.v is its own inverse, matching the UV
        // path's `min_v_vt = self.v(min_v / height)`).
        let (ut_min, ut_max, vt_min, vt_max) = if line_axis == 0 {
            let vt = self.v(fixed_val as f64 / fh);
            (origin_line as f64 / fw, line_hi as f64 / fw, vt, vt)
        } else {
            let vt_a = self.v(origin_line as f64 / fh);
            let vt_b = self.v(line_hi as f64 / fh);
            (fixed_val as f64 / fw, fixed_val as f64 / fw, vt_a.min(vt_b), vt_a.max(vt_b))
        };

        // Surface profile: (pos, normal) at each line pixel a triangle covers.
        let mut line: Vec<Option<([f64; 3], [f64; 3])>> = vec![None; line_dim];

        let obj = &self.obj;
        for i in obj.base.uv_index.in_bounds(ut_min, ut_max, vt_min, vt_max) {
            let tri = &obj.base.triangles[i];
            let vt1 = &obj.base.tex_vertices[tri.t[0] as usize];
            let vt2 = &obj.base.tex_vertices[tri.t[1] as usize];
            let vt3 = &obj.base.tex_vertices[tri.t[2] as usize];

            let u1 = (vt1.u * fw) as i32;
            let sv1 = (self.v(vt1.v) * fh) as i32;
            let u2 = (vt2.u * fw) as i32;
            let sv2 = (self.v(vt2.v) * fh) as i32;
            let u3 = (vt3.u * fw) as i32;
            let sv3 = (self.v(vt3.v) * fh) as i32;

            // Triangle extent along the line axis → covered line-pixel range.
            let (l1, l2, l3) = if line_axis == 0 { (u1, u2, u3) } else { (sv1, sv2, sv3) };
            let line_min = l1.min(l2).min(l3);
            let line_max = l1.max(l2).max(l3);
            let li_start = (((line_min - origin_line) as f64 / pz as f64).ceil() as i32).max(0);
            let li_end = (((line_max - origin_line) as f64 / pz as f64).floor() as i32).min(line_dim as i32 - 1);
            if li_end < li_start {
                continue;
            }

            let xyz1 = &obj.vertices[tri.v[0] as usize];
            let xyz2 = &obj.vertices[tri.v[1] as usize];
            let xyz3 = &obj.vertices[tri.v[2] as usize];
            let n1 = &obj.normals[tri.n[0] as usize];
            let n2 = &obj.normals[tri.n[1] as usize];
            let n3 = &obj.normals[tri.n[2] as usize];

            // Pre-dispatch chunk fetches for this triangle's slab of rays — the
            // gather fast path reads the shard mmap unconditionally (see
            // UnifiedVolume::composite_along_normal_inner). Bound the box over
            // the 3 verts × the strip's w extremes, as the UV composite path does.
            if config.trilinear_interpolation {
                let mut tmin = [f64::INFINITY; 3];
                let mut tmax = [f64::NEG_INFINITY; 3];
                for (p, n) in [(xyz1, n1), (xyz2, n2), (xyz3, n3)] {
                    for &ww in &[origin_w, w_hi] {
                        let q = [(p.x + ww * n.x) * inv_f, (p.y + ww * n.y) * inv_f, (p.z + ww * n.z) * inv_f];
                        for d in 0..3 {
                            if q[d] < tmin[d] {
                                tmin[d] = q[d];
                            }
                            if q[d] > tmax[d] {
                                tmax[d] = q[d];
                            }
                        }
                    }
                }
                volume.touch_aabb(tmin, tmax, sfactor as i32);
            }

            for li in li_start..=li_end {
                let line_coord = origin_line + li * pz;
                let (qu, qv) = if line_axis == 0 { (line_coord, fixed_val) } else { (fixed_val, line_coord) };

                let w0 = orient2d(u2, sv2, u3, sv3, qu, qv);
                let w1 = orient2d(u3, sv3, u1, sv1, qu, qv);
                let w2 = orient2d(u1, sv1, u2, sv2, qu, qv);
                if w0 >= 0 && w1 >= 0 && w2 >= 0 {
                    let invwsum = 1. / (w0 + w1 + w2) as f64;
                    let pos = [
                        (w0 as f64 * xyz1.x + w1 as f64 * xyz2.x + w2 as f64 * xyz3.x) * invwsum,
                        (w0 as f64 * xyz1.y + w1 as f64 * xyz2.y + w2 as f64 * xyz3.y) * invwsum,
                        (w0 as f64 * xyz1.z + w1 as f64 * xyz2.z + w2 as f64 * xyz3.z) * invwsum,
                    ];
                    let nrm = [
                        (w0 as f64 * n1.x + w1 as f64 * n2.x + w2 as f64 * n3.x) * invwsum,
                        (w0 as f64 * n1.y + w1 as f64 * n2.y + w2 as f64 * n3.y) * invwsum,
                        (w0 as f64 * n1.z + w1 as f64 * n2.z + w2 as f64 * n3.z) * invwsum,
                    ];
                    line[li as usize] = Some((pos, nrm));
                }
            }
        }

        // Expand each surface profile point along its normal across the w
        // extent. Color path so the (inside-out) overlay tint shows, matching
        // the UV pane; `gather_color_along_normal` keeps the cache fast path.
        let mut col = vec![ecolor::Color32::from_gray(0); w_dim];
        for li in 0..line_dim {
            let Some((pos, n)) = line[li] else {
                continue;
            };
            let write = |buffer: &mut Image, wi: usize, color: ecolor::Color32| {
                let (bx, by) = if w_is_vertical { (li, wi) } else { (wi, li) };
                buffer.set(bx, by, color);
            };
            if config.trilinear_interpolation {
                let base = [
                    (pos[0] + origin_w * n[0]) * inv_f,
                    (pos[1] + origin_w * n[1]) * inv_f,
                    (pos[2] + origin_w * n[2]) * inv_f,
                ];
                let dir = [pz as f64 * n[0] * inv_f, pz as f64 * n[1] * inv_f, pz as f64 * n[2] * inv_f];
                volume.gather_color_along_normal(base, dir, sfactor as i32, &mut col);
                for wi in 0..w_dim {
                    write(buffer, wi, col[wi]);
                }
            } else {
                for wi in 0..w_dim {
                    let ww = origin_w + (wi as i32 * pz) as f64;
                    let p = [(pos[0] + ww * n[0]) * inv_f, (pos[1] + ww * n[1]) * inv_f, (pos[2] + ww * n[2]) * inv_f];
                    write(buffer, wi, volume.get_color(p, sfactor as i32));
                }
            }
        }
    }
}

fn orient2d(u1: i32, v1: i32, u2: i32, v2: i32, u3: i32, v3: i32) -> i32 {
    (u2 - u1) * (v3 - v1) - (v2 - v1) * (u3 - u1)
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
        // UV keeps its dedicated triangle-raster path (compositing, outlines,
        // depth scrub) below. UW/VW cross-sections take the generalized
        // slice-line + normal-walk path.
        if !(u_coord == 0 && v_coord == 1 && plane_coord == 2) {
            self.paint_cross_section(
                xyz, u_coord, v_coord, plane_coord, width, height, sfactor, paint_zoom, config, buffer,
            );
            return;
        }

        let draw_outlines = config.draw_xyz_outlines;
        let composite = config.compositing.mode != CompositingMode::None;
        let composite_layers_in_front = config.compositing.layers_in_front as i32;
        let composite_layers_behind = config.compositing.layers_behind as i32;
        let composite_total_layers = composite_layers_in_front + composite_layers_behind + 1; // +1 for the current layer
        let mut composition: Compositor = match config.compositing.mode {
            CompositingMode::Max => Compositor::Max(MaxCompositionState::new()),
            CompositingMode::Alpha => Compositor::Alpha(AlphaCompositionState::new(
                config.compositing.alpha_min as f32 / 255.0,
                config.compositing.alpha_max as f32 / 255.0,
                config.compositing.alpha_threshold as f32 / 10000.0,
                config.compositing.opacity as f32 / 100.0,
            )),
            CompositingMode::AlphaHeightMap => Compositor::HeightMap(AlphaHeightMapCompositionState::new(
                config.compositing.alpha_min as f32 / 255.0,
                config.compositing.alpha_max as f32 / 255.0,
                config.compositing.alpha_threshold as f32 / 10000.0,
                config.compositing.opacity as f32 / 100.0,
            )),
            CompositingMode::AlphaOverlay => Compositor::AlphaOverlay(AlphaCompositionState::new(
                config.compositing.alpha_min as f32 / 255.0,
                config.compositing.alpha_max as f32 / 255.0,
                config.compositing.alpha_threshold as f32 / 10000.0,
                config.compositing.opacity as f32 / 100.0,
            )),
            CompositingMode::AlphaOverlayStart => Compositor::AlphaOverlayStart(AlphaCompositionState::new(
                config.compositing.alpha_min as f32 / 255.0,
                config.compositing.alpha_max as f32 / 255.0,
                config.compositing.alpha_threshold as f32 / 10000.0,
                config.compositing.opacity as f32 / 100.0,
            )),
            CompositingMode::AlphaOverlayCombined => Compositor::AlphaOverlayCombined(
                AlphaCompositionState::new(
                    config.compositing.alpha_min as f32 / 255.0,
                    config.compositing.alpha_max as f32 / 255.0,
                    config.compositing.alpha_threshold as f32 / 10000.0,
                    config.compositing.opacity as f32 / 100.0,
                ),
                OverlayCombineParams {
                    background: config.compositing.overlay_background as f32 / 100.0,
                    value_norm: 1.0 - config.compositing.overlay_value_norm as f32 / 100.0,
                },
            ),
            CompositingMode::None => Compositor::None(NoCompositionState),
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

        let (min_v_vt, max_v_vt) = if self.obj.base.has_inverted_uv_tris {
            (min_v_vt, max_v_vt)
        } else {
            (max_v_vt, min_v_vt)
        };

        /* println!(
            "xyz: {:?}, width: {}, height: {}, sfactor: {}, paint_zoom: {}",
            xyz, width, height, sfactor, paint_zoom
        );
        println!("min_u: {}, max_u: {}, min_v: {}, max_v: {}", min_u, max_u, min_v, max_v); */

        let obj = &self.obj;
        for i in obj.base.uv_index.in_bounds(min_u_vt, max_u_vt, min_v_vt, max_v_vt) {
            let tri = &obj.base.triangles[i];
            {
                {
                    let vert1 = &obj.base.tex_vertices[tri.t[0] as usize];
                    let vert2 = &obj.base.tex_vertices[tri.t[1] as usize];
                    let vert3 = &obj.base.tex_vertices[tri.t[2] as usize];

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

                        // Per-tri pre-dispatch. For composite rays we
                        // conservatively bound the box of target-LOD
                        // voxels every ray within this triangle could
                        // touch (3 vertex bases × {start, end} ray
                        // extents), and ask the backend to kick fetches
                        // + parent-LOD upscale fills for those chunks
                        // before the pixel loop drives the per-voxel
                        // composite. The composite inner loop reads the
                        // shard mmap unconditionally — without this
                        // pre-touch, the only thing it would see for
                        // unloaded chunks is the kernel zero page.
                        if composite && config.trilinear_interpolation {
                            let inv_f = 1.0 / ffactor;
                            let xyz1 = &obj.vertices[tri.v[0] as usize];
                            let xyz2 = &obj.vertices[tri.v[1] as usize];
                            let xyz3 = &obj.vertices[tri.v[2] as usize];
                            let nxyz1 = &obj.normals[tri.n[0] as usize];
                            let nxyz2 = &obj.normals[tri.n[1] as usize];
                            let nxyz3 = &obj.normals[tri.n[2] as usize];
                            let start_w = (xyz[2] + composite_direction * composite_layers_in_front) as f64;
                            let end_w = (xyz[2] - composite_direction * (composite_layers_behind + 1)) as f64;
                            let verts = [(xyz1, nxyz1), (xyz2, nxyz2), (xyz3, nxyz3)];
                            let mut tmin = [f64::INFINITY; 3];
                            let mut tmax = [f64::NEG_INFINITY; 3];
                            for (p, n) in &verts {
                                for &w in &[start_w, end_w] {
                                    let q = [
                                        (p.x + w * n.x) * inv_f,
                                        (p.y + w * n.y) * inv_f,
                                        (p.z + w * n.z) * inv_f,
                                    ];
                                    for d in 0..3 {
                                        if q[d] < tmin[d] {
                                            tmin[d] = q[d];
                                        }
                                        if q[d] > tmax[d] {
                                            tmax[d] = q[d];
                                        }
                                    }
                                }
                            }
                            volume.touch_aabb(tmin, tmax, sfactor as i32);
                        }

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
                                        let xyz1 = &obj.vertices[tri.v[0] as usize];
                                        let xyz2 = &obj.vertices[tri.v[1] as usize];
                                        let xyz3 = &obj.vertices[tri.v[2] as usize];

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
                                            let nxyz1 = &obj.normals[tri.n[0] as usize];
                                            let nxyz2 = &obj.normals[tri.n[1] as usize];
                                            let nxyz3 = &obj.normals[tri.n[2] as usize];

                                            let nx = (w0 as f64 * nxyz1.x + w1 as f64 * nxyz2.x + w2 as f64 * nxyz3.x)
                                                * invwsum;
                                            let ny = (w0 as f64 * nxyz1.y + w1 as f64 * nxyz2.y + w2 as f64 * nxyz3.y)
                                                * invwsum;
                                            let nz = (w0 as f64 * nxyz1.z + w1 as f64 * nxyz2.z + w2 as f64 * nxyz3.z)
                                                * invwsum;

                                            (nx, ny, nz)
                                        };

                                        if !composite {
                                            let x = x + w_factor * nx;
                                            let y = y + w_factor * ny;
                                            let z = z + w_factor * nz;

                                            let color = if config.trilinear_interpolation {
                                                volume.get_color_interpolated(
                                                    [x / ffactor, y / ffactor, z / ffactor],
                                                    sfactor as i32,
                                                )
                                            } else {
                                                volume
                                                    .get_color([x / ffactor, y / ffactor, z / ffactor], sfactor as i32)
                                            };

                                            buffer.set(
                                                u as usize / paint_zoom as usize,
                                                v as usize / paint_zoom as usize,
                                                color,
                                            );
                                        } else {
                                            let start = xyz[2] + composite_direction * composite_layers_in_front;
                                            let end = xyz[2] - composite_direction * (composite_layers_behind + 1);

                                            let step = if start < end { 1 } else { -1 };
                                            let n_samples = (end - start) / step;

                                            if config.trilinear_interpolation {
                                                // Walk via the backend's composite_color_along_normal
                                                // hook. The default impl wraps the gray fast path in
                                                // Color32::from_gray; OverlayVolume overrides it to
                                                // composite base and overlay separately (both hitting
                                                // the cache fast path) and blend, so the cache
                                                // optimizations survive overlay-on rendering.
                                                let inv_f = 1.0 / ffactor;
                                                let base = [
                                                    (x + start as f64 * nx) * inv_f,
                                                    (y + start as f64 * ny) * inv_f,
                                                    (z + start as f64 * nz) * inv_f,
                                                ];
                                                let dir = [
                                                    step as f64 * nx * inv_f,
                                                    step as f64 * ny * inv_f,
                                                    step as f64 * nz * inv_f,
                                                ];
                                                let color = volume.composite_color_along_normal(
                                                    base,
                                                    dir,
                                                    0.0,
                                                    n_samples as f64,
                                                    sfactor as i32,
                                                    &mut composition,
                                                    composite_total_layers as u32,
                                                );
                                                buffer.set(
                                                    u as usize / paint_zoom as usize,
                                                    v as usize / paint_zoom as usize,
                                                    color,
                                                );
                                            } else {
                                                composition.reset();
                                                let mut compositor = composition.as_ref_mut();
                                                let mut w = start;
                                                while w != end {
                                                    let w_factor = w as f64;
                                                    let x = x + w_factor * nx;
                                                    let y = y + w_factor * ny;
                                                    let z = z + w_factor * nz;
                                                    let new_value = volume
                                                        .get([x / ffactor, y / ffactor, z / ffactor], sfactor as i32);
                                                    if !compositor.update(new_value) {
                                                        break;
                                                    }
                                                    w += step;
                                                }
                                                let value = composition.result(composite_total_layers as u32);
                                                buffer.set_gray(
                                                    u as usize / paint_zoom as usize,
                                                    v as usize / paint_zoom as usize,
                                                    value,
                                                );
                                            }
                                        }

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
    fn plane_intersection(&self, u_coord: usize, v_coord: usize, plane_coord: usize, w: i32) -> Vec<OutlineSegment> {
        let obj = &self.obj;
        let w = w as f64;

        let mut mins = obj.xyz_index.min;
        let mut maxs = obj.xyz_index.max;
        mins[plane_coord] = w;
        maxs[plane_coord] = w;

        // the index lists a triangle once for every cell it overlaps
        let mut candidates = obj.xyz_index.in_bounds(mins, maxs);
        candidates.sort_unstable();
        candidates.dedup();

        let (tex_w, tex_h) = (self.width as f64, self.height as f64);
        let mut segments = Vec::new();
        for i in candidates {
            let tri = &obj.base.triangles[i];
            let vs = tri.v.map(|v| {
                let v = &obj.vertices[v as usize];
                [v.x, v.y, v.z]
            });
            let Some((a, b)) = super::triangle_plane_segment(vs, u_coord, v_coord, plane_coord, w) else {
                continue;
            };

            let ts = tri.t.map(|t| {
                let t = &obj.base.tex_vertices[t as usize];
                [t.u * tex_w, self.v(t.v) * tex_h]
            });
            let tex_min = [
                ts[0][0].min(ts[1][0]).min(ts[2][0]) as f32,
                ts[0][1].min(ts[1][1]).min(ts[2][1]) as f32,
            ];
            let tex_max = [
                ts[0][0].max(ts[1][0]).max(ts[2][0]) as f32,
                ts[0][1].max(ts[1][1]).max(ts[2][1]) as f32,
            ];
            segments.push(OutlineSegment { a, b, tex_min, tex_max });
        }
        segments
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
pub(super) fn line(x0: i32, y0: i32, x1: i32, y1: i32, buffer: &mut Image, width: usize, height: usize, r: u8, g: u8, b: u8) {
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

impl VoxelVolume for ObjVolume {
    fn get(&self, _xyz: [f64; 3], _downsampling: i32) -> u8 {
        todo!()
    }
    fn reset_for_painting(&self) {
        self.volume.reset_for_painting();
    }
}
