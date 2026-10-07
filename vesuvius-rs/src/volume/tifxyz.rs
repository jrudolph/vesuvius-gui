//! vc3d tifxyz segmentation volume — alternative to `ObjVolume` for surfaces
//! distributed as a directory of `meta.json` + `x.tif` + `y.tif` + `z.tif` (the
//! same on-disk format volume-cartographer's `QuadSurface` reads/writes).
//!
//! Because tifxyz already is a regular UV→XYZ grid we never need triangle
//! rasterization: painting is a direct bilinear lookup per output pixel. The
//! grid implicitly triangulates as (r,c)-(r,c+1)-(r+1,c) + (r,c+1)-(r+1,c+1)-(r+1,c)
//! — the same triangulation `paint_plane_intersection` walks to draw the
//! segment outline on the orthogonal XYZ panes.

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use libm::modf;
use serde::Deserialize;
use tiff::decoder::{Decoder, DecodingResult};

use super::composition::{
    AlphaCompositionState, AlphaHeightMapCompositionState, Compositor, MaxCompositionState, NoCompositionState,
    OverlayCombineParams,
};
use super::{
    AffineTransform, CompositingMode, DrawingConfig, Image, PaintVolume, SurfaceVolume, Volume, VolumeCons,
    VoxelPaintVolume, VoxelVolume,
};

// ---------- on-disk metadata ----------

#[derive(Deserialize)]
struct TifXyzMeta {
    uuid: String,
    #[serde(rename = "type")]
    seg_type: String,
    format: String,
    scale: [f64; 2],
    #[serde(default)]
    bbox: Option<[[f64; 3]; 2]>,
}

// ---------- data ----------

/// Parsed tifxyz directory, independent of base volume and affine transform.
/// Shared via `Arc` so `with_base()` doesn't re-decode TIFFs.
pub struct TifXyzBase {
    pub rows: usize,
    pub cols: usize,
    pub uuid: String,
    pub scale: [f64; 2],
    pub bbox: Option<[[f64; 3]; 2]>,
    /// `rows * cols`; sentinel `[-1, -1, -1]` for invalid cells.
    pub pre_xyz: Vec<[f32; 3]>,
    /// `rows * cols`; central-difference normal. Zero where any neighbor is
    /// invalid or where the cell is on the grid border.
    pub pre_normals: Vec<[f32; 3]>,
    pub valid: Vec<bool>,
}

/// Per-transform projection of `TifXyzBase`. Holds the post-affine xyz/normals
/// so the inner paint loop can skip the matrix multiply per pixel.
pub struct TifXyzData {
    base: Arc<TifXyzBase>,
    xyz: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    /// `(rows-1) * (cols-1)` post-affine quad AABBs as `[min_x, min_y, min_z, max_x, max_y, max_z]`.
    /// Invalid quads (any corner invalid, or grid too small) use sentinel
    /// `[+inf, +inf, +inf, -inf, -inf, -inf]` so any plane test rejects them in one branch.
    quad_aabb: Vec<[f32; 6]>,
    applied_transform: Option<AffineTransform>,
}

#[derive(Clone)]
pub struct TifXyzVolume {
    volume: Volume,
    data: Arc<TifXyzData>,
    /// Grid-cells per output UV pixel, i.e. the factor `seg_to_grid` multiplies
    /// by. Defaults to `meta.json.scale` (→ natural `nominal_dims`), but a
    /// caller can render the unrolled segment at an arbitrary resolution by
    /// choosing `tex_width`/`tex_height` (see `from_data`); then it's
    /// `[cols/tex_width, rows/tex_height]`, the grid sampled finer/coarser.
    scale: [f64; 2],
    tex_width: usize,
    tex_height: usize,
}

// ---------- loading ----------

fn decode_float32_tif(path: &Path) -> Result<(usize, usize, Vec<f32>)> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut decoder = Decoder::new(BufReader::new(file)).with_context(|| format!("tiff header for {}", path.display()))?;
    let (w, h) = decoder.dimensions().with_context(|| format!("tiff dims for {}", path.display()))?;
    let img = decoder
        .read_image()
        .with_context(|| format!("tiff read for {}", path.display()))?;
    let data = match img {
        DecodingResult::F32(v) => v,
        DecodingResult::F64(v) => v.into_iter().map(|x| x as f32).collect(),
        DecodingResult::U16(v) => v.into_iter().map(|x| x as f32).collect(),
        DecodingResult::U8(v) => v.into_iter().map(|x| x as f32).collect(),
        DecodingResult::I16(v) => v.into_iter().map(|x| x as f32).collect(),
        DecodingResult::I32(v) => v.into_iter().map(|x| x as f32).collect(),
        other => bail!("unsupported tiff pixel type for {}: {}", path.display(), tiff_kind(&other)),
    };
    if data.len() != (w as usize) * (h as usize) {
        bail!(
            "{}: expected {}x{} = {} samples, got {}",
            path.display(),
            w,
            h,
            (w as usize) * (h as usize),
            data.len()
        );
    }
    Ok((w as usize, h as usize, data))
}

fn tiff_kind(d: &DecodingResult) -> &'static str {
    match d {
        DecodingResult::U8(_) => "u8",
        DecodingResult::U16(_) => "u16",
        DecodingResult::U32(_) => "u32",
        DecodingResult::U64(_) => "u64",
        DecodingResult::I8(_) => "i8",
        DecodingResult::I16(_) => "i16",
        DecodingResult::I32(_) => "i32",
        DecodingResult::I64(_) => "i64",
        DecodingResult::F16(_) => "f16",
        DecodingResult::F32(_) => "f32",
        DecodingResult::F64(_) => "f64",
    }
}

fn transforms_equal(a: &Option<AffineTransform>, b: &Option<AffineTransform>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => x.matrix == y.matrix,
        _ => false,
    }
}

impl TifXyzBase {
    pub fn load(dir: &Path) -> Result<Arc<Self>> {
        let t = Instant::now();

        // meta.json
        let meta_path: PathBuf = dir.join("meta.json");
        let meta_raw = std::fs::read_to_string(&meta_path)
            .with_context(|| format!("reading {}", meta_path.display()))?;
        let meta: TifXyzMeta = serde_json::from_str(&meta_raw)
            .with_context(|| format!("parsing {}", meta_path.display()))?;
        if meta.format != "tifxyz" {
            bail!(
                "{}: expected format=\"tifxyz\", got {:?}",
                meta_path.display(),
                meta.format
            );
        }
        if meta.seg_type != "seg" {
            log::warn!("{}: unexpected type {:?} (expected \"seg\")", meta_path.display(), meta.seg_type);
        }

        // x/y/z tifs
        let (xw, xh, xs) = decode_float32_tif(&dir.join("x.tif"))?;
        let (yw, yh, ys) = decode_float32_tif(&dir.join("y.tif"))?;
        let (zw, zh, zs) = decode_float32_tif(&dir.join("z.tif"))?;
        if (xw, xh) != (yw, yh) || (xw, xh) != (zw, zh) {
            bail!(
                "x/y/z.tif dimension mismatch in {}: x={}x{} y={}x{} z={}x{}",
                dir.display(),
                xw,
                xh,
                yw,
                yh,
                zw,
                zh
            );
        }
        let cols = xw;
        let rows = xh;
        let n = rows * cols;

        // assemble xyz + validity (z <= 0 marks invalid, sentinel cleared to (-1,-1,-1))
        let mut pre_xyz = vec![[-1.0f32; 3]; n];
        let mut valid = vec![false; n];
        for i in 0..n {
            let z = zs[i];
            if z > 0.0 && z.is_finite() && xs[i].is_finite() && ys[i].is_finite() {
                pre_xyz[i] = [xs[i], ys[i], z];
                valid[i] = true;
            }
        }

        // central-difference normals; zero on border or where any neighbor is invalid
        let mut pre_normals = vec![[0.0f32; 3]; n];
        if rows >= 3 && cols >= 3 {
            for r in 1..rows - 1 {
                for c in 1..cols - 1 {
                    let i = r * cols + c;
                    let il = i - 1;
                    let ir = i + 1;
                    let iu = i - cols;
                    let id = i + cols;
                    if !(valid[i] && valid[il] && valid[ir] && valid[iu] && valid[id]) {
                        continue;
                    }
                    let du = sub3(pre_xyz[ir], pre_xyz[il]);
                    let dv = sub3(pre_xyz[id], pre_xyz[iu]);
                    let n = cross3(du, dv);
                    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                    if len > 1e-6 {
                        pre_normals[i] = [n[0] / len, n[1] / len, n[2] / len];
                    }
                }
            }
        }

        let valid_count: usize = valid.iter().filter(|&&b| b).count();
        log::info!(
            "TifXyzBase::load {}x{} ({} valid / {} total) in {:?}",
            cols,
            rows,
            valid_count,
            n,
            t.elapsed()
        );

        Ok(Arc::new(Self {
            rows,
            cols,
            uuid: meta.uuid,
            scale: meta.scale,
            bbox: meta.bbox,
            pre_xyz,
            pre_normals,
            valid,
        }))
    }
}

#[inline]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Build post-affine per-quad AABBs once per `TifXyzData::build`. Invalid quads
/// (any corner invalid, or grid smaller than 2x2) get sentinel `[+inf; 3, -inf; 3]`
/// so the plane-axis cull in `paint_plane_intersection` rejects them with a single
/// comparison instead of four `valid[]` loads + four `xyz[]` loads + six min/max.
fn build_quad_aabb(base: &TifXyzBase, xyz: &[[f32; 3]]) -> Vec<[f32; 6]> {
    let cols = base.cols;
    let rows = base.rows;
    if rows < 2 || cols < 2 {
        return Vec::new();
    }
    let qcols = cols - 1;
    let qrows = rows - 1;
    let sentinel = [f32::INFINITY, f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
    let mut aabb = vec![sentinel; qrows * qcols];
    let valid = &base.valid;
    for r in 0..qrows {
        let row_base = r * cols;
        let qrow_base = r * qcols;
        for c in 0..qcols {
            let i00 = row_base + c;
            let i10 = i00 + 1;
            let i01 = i00 + cols;
            let i11 = i01 + 1;
            if !(valid[i00] && valid[i10] && valid[i01] && valid[i11]) {
                continue;
            }
            let p00 = xyz[i00];
            let p10 = xyz[i10];
            let p01 = xyz[i01];
            let p11 = xyz[i11];
            aabb[qrow_base + c] = [
                p00[0].min(p10[0]).min(p01[0]).min(p11[0]),
                p00[1].min(p10[1]).min(p01[1]).min(p11[1]),
                p00[2].min(p10[2]).min(p01[2]).min(p11[2]),
                p00[0].max(p10[0]).max(p01[0]).max(p11[0]),
                p00[1].max(p10[1]).max(p01[1]).max(p11[1]),
                p00[2].max(p10[2]).max(p01[2]).max(p11[2]),
            ];
        }
    }
    aabb
}

impl TifXyzData {
    fn build(base: Arc<TifXyzBase>, transform: &Option<AffineTransform>) -> Arc<Self> {
        let t = Instant::now();
        let n = base.pre_xyz.len();

        let (xyz, normals) = if let Some(tf) = transform {
            let m = &tf.matrix;
            let mut xyz = vec![[-1.0f32; 3]; n];
            let mut normals = vec![[0.0f32; 3]; n];
            for i in 0..n {
                if !base.valid[i] {
                    continue;
                }
                let p = base.pre_xyz[i];
                xyz[i] = [
                    (m[0][0] * p[0] as f64 + m[0][1] * p[1] as f64 + m[0][2] * p[2] as f64 + m[0][3]) as f32,
                    (m[1][0] * p[0] as f64 + m[1][1] * p[1] as f64 + m[1][2] * p[2] as f64 + m[1][3]) as f32,
                    (m[2][0] * p[0] as f64 + m[2][1] * p[1] as f64 + m[2][2] * p[2] as f64 + m[2][3]) as f32,
                ];

                let pn = base.pre_normals[i];
                let nx = m[0][0] * pn[0] as f64 + m[0][1] * pn[1] as f64 + m[0][2] * pn[2] as f64;
                let ny = m[1][0] * pn[0] as f64 + m[1][1] * pn[1] as f64 + m[1][2] * pn[2] as f64;
                let nz = m[2][0] * pn[0] as f64 + m[2][1] * pn[1] as f64 + m[2][2] * pn[2] as f64;
                let len = (nx * nx + ny * ny + nz * nz).sqrt();
                if len > 1e-6 {
                    normals[i] = [(nx / len) as f32, (ny / len) as f32, (nz / len) as f32];
                }
            }
            (xyz, normals)
        } else {
            (base.pre_xyz.clone(), base.pre_normals.clone())
        };

        let quad_aabb = build_quad_aabb(&base, &xyz);

        log::info!(
            "TifXyzData::build (transform={}) in {:?}",
            transform.is_some(),
            t.elapsed()
        );

        Arc::new(Self {
            base,
            xyz,
            normals,
            quad_aabb,
            applied_transform: transform.clone(),
        })
    }

    #[inline]
    fn idx(&self, r: usize, c: usize) -> usize {
        r * self.base.cols + c
    }
}

// ---------- TifXyzVolume ----------

/// Default texture (== "nominal" / voxel-space) dimensions for a tifxyz base.
/// Matches `QuadSurface::scale()` semantics in volume-cartographer
/// (`core/src/QuadSurface.cpp:505`): `{cols / scale_x, rows / scale_y}`.
/// `meta.json.scale` is grid-cells-per-nominal-unit, so dividing the grid dims
/// by it yields the segment's extent in the same units ObjVolume's catalog
/// width/height use (voxel-space pixels), which is what downstream LOD
/// selection and zoom math assume.
fn nominal_dims(base: &TifXyzBase) -> (usize, usize) {
    let sx = base.scale[0].max(1e-9);
    let sy = base.scale[1].max(1e-9);
    let w = ((base.cols as f64) / sx).max(1.0).round() as usize;
    let h = ((base.rows as f64) / sy).max(1.0).round() as usize;
    (w, h)
}

impl TifXyzVolume {
    pub fn load_from_directory(
        dir: impl AsRef<Path>,
        base_volume: Volume,
        transform: &Option<AffineTransform>,
    ) -> Result<Self> {
        let dir = dir.as_ref();
        if !dir.is_dir() {
            bail!("tifxyz path is not a directory: {}", dir.display());
        }
        let base = TifXyzBase::load(dir)?;
        let (tex_width, tex_height) = nominal_dims(&base);
        let scale = base.scale;
        log::info!(
            "TifXyzVolume::load_from_directory grid {}x{} scale {:?} → nominal {}x{}",
            base.cols,
            base.rows,
            base.scale,
            tex_width,
            tex_height
        );
        let data = TifXyzData::build(base, transform);
        Ok(Self {
            volume: base_volume,
            data,
            scale,
            tex_width,
            tex_height,
        })
    }

    pub fn width(&self) -> usize {
        self.tex_width
    }
    pub fn height(&self) -> usize {
        self.tex_height
    }
    pub fn uuid(&self) -> &str {
        &self.data.base.uuid
    }

    /// The base-independent projection (parsed grid + baked transform). Cloning
    /// it is cheap (an `Arc`) and the result is `Send + Sync`, unlike the full
    /// `TifXyzVolume`, which embeds a (possibly `!Sync`) base `Volume`. Callers
    /// that render across threads (e.g. the offline renderer) keep one of these
    /// and rebuild a per-thread volume via [`from_data`](Self::from_data).
    pub fn data(&self) -> Arc<TifXyzData> {
        self.data.clone()
    }

    /// Build a volume from a pre-projected [`TifXyzData`] (see [`data`](Self::data)),
    /// swapping in `base_volume`. Avoids re-decoding the TIFFs and re-applying
    /// the affine.
    ///
    /// `dims` selects the output UV resolution: `None` uses the natural
    /// `grid ÷ scale` `nominal_dims` (and the exact `meta.json` scale, matching
    /// a fresh `load_from_directory` / `with_base`); `Some((w, h))` renders the
    /// unrolled segment at `w × h` instead, sampling the grid proportionally —
    /// the analog of choosing `--width`/`--height` for an obj segment. Pass `w`
    /// and `h` > 0.
    pub fn from_data(data: Arc<TifXyzData>, base_volume: Volume, dims: Option<(usize, usize)>) -> Self {
        let (tex_width, tex_height, scale) = match dims {
            Some((w, h)) => {
                let w = w.max(1);
                let h = h.max(1);
                (w, h, [data.base.cols as f64 / w as f64, data.base.rows as f64 / h as f64])
            }
            None => {
                let (w, h) = nominal_dims(&data.base);
                (w, h, data.base.scale)
            }
        };
        Self {
            volume: base_volume,
            data,
            scale,
            tex_width,
            tex_height,
        }
    }

    /// Reuse the parsed grid; only re-apply the affine if it changed.
    /// `width`/`height` are accepted for API parity with `ObjVolume::with_base`
    /// but ignored — tex dims are determined by `scale` + grid dims, not by
    /// caller-supplied values, so a fresh load and a `with_base` agree.
    pub fn with_base(
        &self,
        base_volume: Volume,
        _width: usize,
        _height: usize,
        transform: &Option<AffineTransform>,
    ) -> Self {
        let data = if transforms_equal(&self.data.applied_transform, transform) {
            self.data.clone()
        } else {
            TifXyzData::build(self.data.base.clone(), transform)
        };
        Self {
            volume: base_volume,
            data,
            scale: self.scale,
            tex_width: self.tex_width,
            tex_height: self.tex_height,
        }
    }

    /// Segment-space (output-UV) coords → grid-cell fractional coords.
    #[inline]
    fn seg_to_grid(&self, seg_u: f64, seg_v: f64) -> (f64, f64) {
        (seg_u * self.scale[0], seg_v * self.scale[1])
    }

    /// UV-pane segment-coord → world voxel coord lookup.
    pub fn convert_to_volume_coords(&self, coord: [i32; 3]) -> [i32; 3] {
        let u = coord[0];
        let v = coord[1];
        let w = coord[2] as f64;

        if u < 0 || v < 0 || u as usize >= self.tex_width || v as usize >= self.tex_height {
            return [-1, -1, -1];
        }
        let (gc, gr) = self.seg_to_grid(u as f64, v as f64);
        let Some(s) = self.sample_grid(gc, gr) else {
            return [-1, -1, -1];
        };
        [
            (s.xyz[0] as f64 + w * s.n[0] as f64) as i32,
            (s.xyz[1] as f64 + w * s.n[1] as f64) as i32,
            (s.xyz[2] as f64 + w * s.n[2] as f64) as i32,
        ]
    }

    /// Generalized cross-section paint for the UW/VW panes.
    ///
    /// One screen axis is `w` (the offset along the surface normal); the other
    /// is a surface coordinate (`u` for UW, `v` for VW). The third surface
    /// coordinate is held fixed at `xyz[plane_coord]`. So for each "line
    /// position" along the surface axis we sample the surface once and then
    /// walk the normal across the whole `w` extent using the backend's
    /// `gather_along_normal` fast path (which amortizes chunk lookups across
    /// the walk). Because `w` is a spatial screen axis here, this path is
    /// always single-sample — no compositing (that only applies to UV).
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
        config: &DrawingConfig,
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
        let origin_w = (xyz[2] - w_dim as i32 / 2 * pz) as f64;
        let fixed_val = xyz[fixed_axis];

        // Color path so the (inside-out) overlay tint shows, matching the UV
        // pane; `gather_color_along_normal` keeps the cache fast path.
        let mut col = vec![ecolor::Color32::from_gray(0); w_dim];
        for li in 0..line_dim {
            let mut seg = [0i32; 2];
            seg[line_axis] = origin_line + li as i32 * pz;
            seg[fixed_axis] = fixed_val;

            let (gc, gr) = self.seg_to_grid(seg[0] as f64, seg[1] as f64);
            let Some(sample) = self.sample_grid(gc, gr) else {
                continue;
            };
            let p = [sample.xyz[0] as f64, sample.xyz[1] as f64, sample.xyz[2] as f64];
            let n = [sample.n[0] as f64, sample.n[1] as f64, sample.n[2] as f64];

            let write = |buffer: &mut Image, wi: usize, color: ecolor::Color32| {
                let (bx, by) = if w_is_vertical { (li, wi) } else { (wi, li) };
                buffer.set(bx, by, color);
            };

            if config.trilinear_interpolation {
                let base = [
                    (p[0] + origin_w * n[0]) * inv_f,
                    (p[1] + origin_w * n[1]) * inv_f,
                    (p[2] + origin_w * n[2]) * inv_f,
                ];
                let dir = [pz as f64 * n[0] * inv_f, pz as f64 * n[1] * inv_f, pz as f64 * n[2] * inv_f];
                volume.gather_color_along_normal(base, dir, sfactor as i32, &mut col);
                for wi in 0..w_dim {
                    write(buffer, wi, col[wi]);
                }
            } else {
                for wi in 0..w_dim {
                    let w = origin_w + (wi as i32 * pz) as f64;
                    let p = [(p[0] + w * n[0]) * inv_f, (p[1] + w * n[1]) * inv_f, (p[2] + w * n[2]) * inv_f];
                    write(buffer, wi, volume.get_color(p, sfactor as i32));
                }
            }
        }
    }

    /// Bilinear sample of xyz + normal at fractional grid coords. Returns None
    /// when any of the four corners is invalid (mirrors ObjVolume's "no
    /// triangle covers this pixel" behavior).
    fn sample_grid(&self, gc: f64, gr: f64) -> Option<Sample> {
        let cols = self.data.base.cols;
        let rows = self.data.base.rows;
        if !(gc >= 0.0 && gr >= 0.0) {
            return None;
        }
        let (dc, c0_f) = modf(gc);
        let (dr, r0_f) = modf(gr);
        let c0 = c0_f as usize;
        let r0 = r0_f as usize;
        if c0 + 1 >= cols || r0 + 1 >= rows {
            return None;
        }
        let i00 = self.data.idx(r0, c0);
        let i10 = i00 + 1;
        let i01 = i00 + cols;
        let i11 = i01 + 1;
        let valid = &self.data.base.valid;
        if !(valid[i00] && valid[i10] && valid[i01] && valid[i11]) {
            return None;
        }
        let xyz = bilerp4(&self.data.xyz, [i00, i10, i01, i11], dc, dr);
        let n = bilerp4(&self.data.normals, [i00, i10, i01, i11], dc, dr);
        Some(Sample { xyz, n })
    }
}

struct Sample {
    xyz: [f32; 3],
    n: [f32; 3],
}

#[inline]
fn bilerp4(buf: &[[f32; 3]], idx: [usize; 4], dx: f64, dy: f64) -> [f32; 3] {
    let a = buf[idx[0]];
    let b = buf[idx[1]];
    let c = buf[idx[2]];
    let d = buf[idx[3]];
    let w00 = ((1.0 - dx) * (1.0 - dy)) as f32;
    let w10 = (dx * (1.0 - dy)) as f32;
    let w01 = ((1.0 - dx) * dy) as f32;
    let w11 = (dx * dy) as f32;
    [
        a[0] * w00 + b[0] * w10 + c[0] * w01 + d[0] * w11,
        a[1] * w00 + b[1] * w10 + c[1] * w01 + d[1] * w11,
        a[2] * w00 + b[2] * w10 + c[2] * w01 + d[2] * w11,
    ]
}

// ---------- trait impls ----------

impl PaintVolume for TifXyzVolume {
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
        config: &DrawingConfig,
        buffer: &mut Image,
    ) {
        // UV keeps its dedicated path (compositing, outlines, depth scrub) below.
        // UW/VW cross-sections take the generalized single-sample path.
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
        let composite_total_layers = composite_layers_in_front + composite_layers_behind + 1;
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
        let composite_direction: i32 = if config.compositing.reverse_direction { -1 } else { 1 };

        let real_xyz = if draw_outlines {
            self.convert_to_volume_coords(xyz)
        } else {
            [0, 0, 0]
        };

        let volume = self.volume.clone();
        let ffactor = sfactor as f64;
        let w_factor = xyz[2] as f64;

        let origin_u = xyz[0] - width as i32 / 2 * paint_zoom as i32;
        let origin_v = xyz[1] - height as i32 / 2 * paint_zoom as i32;

        for by in 0..height {
            for bx in 0..width {
                let seg_u = origin_u + (bx as i32) * paint_zoom as i32;
                let seg_v = origin_v + (by as i32) * paint_zoom as i32;
                let (gc, gr) = self.seg_to_grid(seg_u as f64, seg_v as f64);
                let Some(sample) = self.sample_grid(gc, gr) else {
                    continue;
                };
                let (x, y, z) = (sample.xyz[0] as f64, sample.xyz[1] as f64, sample.xyz[2] as f64);
                let (nx, ny, nz) = if xyz[2] == 0 && !composite {
                    (0.0, 0.0, 0.0)
                } else {
                    (sample.n[0] as f64, sample.n[1] as f64, sample.n[2] as f64)
                };

                if !composite {
                    let px = x + w_factor * nx;
                    let py = y + w_factor * ny;
                    let pz = z + w_factor * nz;

                    let raw = if config.trilinear_interpolation {
                        volume.get_interpolated([px / ffactor, py / ffactor, pz / ffactor], sfactor as i32)
                    } else {
                        volume.get([px / ffactor, py / ffactor, pz / ffactor], sfactor as i32)
                    };
                    let value = config.filter(raw);
                    buffer.set_gray(bx, by, value);

                    if draw_outlines {
                        if (px - real_xyz[0] as f64).abs() < 2.0 {
                            buffer.set_rgb(bx, by, 0, 0, 255);
                        } else if (py - real_xyz[1] as f64).abs() < 2.0 {
                            buffer.set_rgb(bx, by, 255, 0, 0);
                        } else if (pz - real_xyz[2] as f64).abs() < 2.0 {
                            buffer.set_rgb(bx, by, 0, 255, 0);
                        }
                    }
                } else {
                    let start = xyz[2] + composite_direction * composite_layers_in_front;
                    let end = xyz[2] - composite_direction * (composite_layers_behind + 1);
                    let step = if start < end { 1 } else { -1 };
                    let n_samples = (end - start) / step;

                    if config.trilinear_interpolation {
                        let inv_f = 1.0 / ffactor;
                        let base = [
                            (x + start as f64 * nx) * inv_f,
                            (y + start as f64 * ny) * inv_f,
                            (z + start as f64 * nz) * inv_f,
                        ];
                        let dir = [step as f64 * nx * inv_f, step as f64 * ny * inv_f, step as f64 * nz * inv_f];
                        let color = volume.composite_color_along_normal(
                            base,
                            dir,
                            0.0,
                            n_samples as f64,
                            sfactor as i32,
                            &mut composition,
                            composite_total_layers as u32,
                        );
                        buffer.set(bx, by, color);
                    } else {
                        composition.reset();
                        let mut compositor = composition.as_ref_mut();
                        let mut w = start;
                        while w != end {
                            let wf = w as f64;
                            let px = x + wf * nx;
                            let py = y + wf * ny;
                            let pz = z + wf * nz;
                            let v = volume.get([px / ffactor, py / ffactor, pz / ffactor], sfactor as i32);
                            if !compositor.update(v) {
                                break;
                            }
                            w += step;
                        }
                        let value = composition.result(composite_total_layers as u32);
                        buffer.set_gray(bx, by, config.filter(value));
                    }
                }
            }
        }
    }

    fn shared(&self) -> VolumeCons {
        let data = self.data.clone();
        let scale = self.scale;
        let tex_width = self.tex_width;
        let tex_height = self.tex_height;
        let volume = self.volume.shared();
        Box::new(move || {
            TifXyzVolume {
                volume: volume(),
                data,
                scale,
                tex_width,
                tex_height,
            }
            .into_volume()
        })
    }
}

impl VoxelVolume for TifXyzVolume {
    // Per-paint reset of the base (hot slots, and the unified cache's frame
    // tick that gates touch debouncing). `touch_aabb` is deliberately not
    // forwarded: its box is in base-volume voxel space, not segment space.
    fn reset_for_painting(&self) {
        self.volume.reset_for_painting();
    }

    fn get(&self, xyz: [f64; 3], downsampling: i32) -> u8 {
        // Treat input as (u, v, w) in segment space (this is how PPMVolume and
        // the UV pane traverse a surface volume).
        let u = xyz[0];
        let v = xyz[1];
        let w = xyz[2];

        if u <= 0.0 || v <= 0.0 || u >= self.tex_width as f64 || v >= self.tex_height as f64 || w.abs() > 45.0 {
            return 0;
        }
        let (gc, gr) = self.seg_to_grid(u, v);
        let Some(s) = self.sample_grid(gc, gr) else {
            return 0;
        };
        let px = s.xyz[0] as f64 + w * s.n[0] as f64;
        let py = s.xyz[1] as f64 + w * s.n[1] as f64;
        let pz = s.xyz[2] as f64 + w * s.n[2] as f64;
        self.volume.get(
            [
                px / downsampling as f64,
                py / downsampling as f64,
                pz / downsampling as f64,
            ],
            downsampling,
        )
    }
}

impl SurfaceVolume for TifXyzVolume {
    /// Walk the implicit grid triangulation and draw a line for every triangle
    /// that crosses the current plane. Mirrors `ObjVolume::paint_plane_intersection`
    /// pixel-for-pixel (same colors, same `highlight_uv_section` handling, same
    /// optional vertex markers) so the segment outline overlay is identical on
    /// the XYZ panes whether the segment came from .obj or .tifxyz.
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
        config: &DrawingConfig,
        buffer: &mut Image,
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
                    (h[0] as f64 - width as f64 / 2. * paint_zoom as f64) / self.tex_width as f64,
                    (h[1] as f64 - height as f64 / 2. * paint_zoom as f64) / self.tex_height as f64,
                ],
                [
                    (h[0] as f64 + width as f64 / 2. * paint_zoom as f64) / self.tex_width as f64,
                    (h[1] as f64 + height as f64 / 2. * paint_zoom as f64) / self.tex_height as f64,
                ],
            )
        } else {
            ([0f64, 0f64], [0f64, 0f64])
        };

        let draw_outline_vertices = config.draw_outline_vertices;

        let cols = self.data.base.cols;
        let rows = self.data.base.rows;
        if rows < 2 || cols < 2 {
            return;
        }
        let qcols = cols - 1;

        let xyz_grid = &self.data.xyz;
        let quad_aabb = &self.data.quad_aabb;
        let inv_cols = 1.0 / cols as f64;
        let inv_rows = 1.0 / rows as f64;

        // f32 versions of the paint-area bounds for the per-quad cull. Voxel coords
        // stay well within f32 precision, and the cache is f32, so this avoids
        // mixing widths in the hot loop.
        let w_f32 = w as f32;
        let min_u_f32 = min_u as f32;
        let max_u_f32 = max_u as f32;
        let min_v_f32 = min_v as f32;
        let max_v_f32 = max_v as f32;

        for r in 0..rows - 1 {
            let qrow_base = r * qcols;
            let row_base = r * cols;
            for c in 0..cols - 1 {
                let bb = quad_aabb[qrow_base + c];
                // Plane-axis first: invalid quads use a `+inf/-inf` sentinel that
                // also fails this test, so one branch covers both "invalid" and
                // "doesn't cross the plane" — by far the common reject paths.
                if bb[plane_coord] > w_f32 || bb[3 + plane_coord] < w_f32 {
                    continue;
                }
                if bb[3 + u_coord] < min_u_f32 || bb[u_coord] > max_u_f32 {
                    continue;
                }
                if bb[3 + v_coord] < min_v_f32 || bb[v_coord] > max_v_f32 {
                    continue;
                }

                let i00 = row_base + c;
                let i10 = i00 + 1;
                let i01 = i00 + cols;
                let i11 = i01 + 1;
                let p00 = xyz_grid[i00];
                let p10 = xyz_grid[i10];
                let p01 = xyz_grid[i01];
                let p11 = xyz_grid[i11];

                let should_highlight = if highlight_uv_section.is_some() {
                    // normalized UV bbox of this quad (one cell wide/tall)
                    let u_min = c as f64 * inv_cols;
                    let u_max = (c + 1) as f64 * inv_cols;
                    let v_min = r as f64 * inv_rows;
                    let v_max = (r + 1) as f64 * inv_rows;
                    u_min <= uv_section_max[0]
                        && u_max >= uv_section_min[0]
                        && v_min <= uv_section_max[1]
                        && v_max >= uv_section_min[1]
                } else {
                    false
                };

                // triangle 1: p00, p10, p01 ; triangle 2: p10, p11, p01
                draw_tri_plane_intersection(
                    p00, p10, p01,
                    u_coord, v_coord, plane_coord,
                    w, min_u, min_v, paint_zoom,
                    should_highlight, draw_outline_vertices,
                    width, height, buffer,
                );
                draw_tri_plane_intersection(
                    p10, p11, p01,
                    u_coord, v_coord, plane_coord,
                    w, min_u, min_v, paint_zoom,
                    should_highlight, draw_outline_vertices,
                    width, height, buffer,
                );
            }
        }
    }
}

#[inline]
fn draw_tri_plane_intersection(
    v1: [f32; 3],
    v2: [f32; 3],
    v3: [f32; 3],
    u_coord: usize,
    v_coord: usize,
    plane_coord: usize,
    w: i32,
    min_u: i32,
    min_v: i32,
    paint_zoom: u8,
    should_highlight: bool,
    draw_outline_vertices: bool,
    width: usize,
    height: usize,
    buffer: &mut Image,
) {
    let w_f = w as f64;
    let pc = plane_coord;

    let mut points: [[f64; 3]; 2] = [[0.0; 3]; 2];
    let mut n_points = 0usize;

    let add_intersection = |a: [f32; 3], b: [f32; 3], n: &mut usize, pts: &mut [[f64; 3]; 2]| {
        if *n >= 2 {
            return;
        }
        let da = a[pc] as f64 - w_f;
        let db = b[pc] as f64 - w_f;
        if da.signum() == db.signum() {
            return;
        }
        let t = da / (da - db);
        let mut coords = [0.0f64; 3];
        coords[u_coord] = a[u_coord] as f64 + t * (b[u_coord] as f64 - a[u_coord] as f64);
        coords[v_coord] = a[v_coord] as f64 + t * (b[v_coord] as f64 - a[v_coord] as f64);
        coords[plane_coord] = w_f;
        pts[*n] = coords;
        *n += 1;
    };

    add_intersection(v1, v2, &mut n_points, &mut points);
    add_intersection(v2, v3, &mut n_points, &mut points);
    add_intersection(v3, v1, &mut n_points, &mut points);

    if n_points != 2 {
        return;
    }

    let p1 = points[0];
    let p2 = points[1];
    let x0 = ((p1[u_coord] - min_u as f64) / paint_zoom as f64) as i32;
    let y0 = ((p1[v_coord] - min_v as f64) / paint_zoom as f64) as i32;
    let x1 = ((p2[u_coord] - min_u as f64) / paint_zoom as f64) as i32;
    let y1 = ((p2[v_coord] - min_v as f64) / paint_zoom as f64) as i32;

    let (r, g, b, rp, gp, bp) = if should_highlight {
        (0xff, 0xaa, 0, 0, 0xff, 0)
    } else {
        (0xff, 0, 0xff, 0, 0, 0xff)
    };

    super::objvolume::line(x0, y0, x1, y1, buffer, width, height, r, g, b);
    if draw_outline_vertices {
        super::objvolume::point(x0, y0, buffer, 4, rp, gp, bp);
        super::objvolume::point(x1, y1, buffer, 4, rp, gp, bp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::volume::EmptyVolume;

    const FIXTURE: &str =
        "/home/johannes/git/scrollprize/villa/volume-cartographer/core/test/data/segments/20241113070770";

    #[test]
    fn loads_fixture() {
        let Ok(base) = TifXyzBase::load(Path::new(FIXTURE)) else {
            eprintln!("skipping: fixture not present at {}", FIXTURE);
            return;
        };
        assert_eq!(base.cols, 129);
        assert_eq!(base.rows, 129);
        assert_eq!(base.uuid, "20241113070770");

        let valid_count = base.valid.iter().filter(|&&b| b).count();
        // C++ reader logs roughly 13902 valid cells; allow a little tolerance for
        // small format/sentinel-handling differences.
        assert!(valid_count > 13_000 && valid_count < 14_500, "valid_count={}", valid_count);

        // sentinel propagation
        for i in 0..base.pre_xyz.len() {
            if !base.valid[i] {
                assert_eq!(base.pre_xyz[i], [-1.0, -1.0, -1.0]);
            }
        }

        // some interior cell should have a non-zero normal
        let mid = base.rows / 2;
        let any_nonzero_normal = (1..base.cols - 1).any(|c| {
            let n = base.pre_normals[mid * base.cols + c];
            (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) > 0.5
        });
        assert!(any_nonzero_normal, "no non-zero normals in middle row");

        // border normals must be zero
        for c in 0..base.cols {
            assert_eq!(base.pre_normals[c], [0.0, 0.0, 0.0]);
            let i = (base.rows - 1) * base.cols + c;
            assert_eq!(base.pre_normals[i], [0.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn voxel_get_returns_zero_outside_grid() {
        let Ok(base) = TifXyzBase::load(Path::new(FIXTURE)) else {
            eprintln!("skipping: fixture not present");
            return;
        };
        let (tex_w, tex_h) = nominal_dims(&base);
        let data = TifXyzData::build(base, &None);
        let vol = TifXyzVolume::from_data(data, EmptyVolume {}.into_volume(), None);
        assert_eq!(vol.get([-1.0, 5.0, 0.0], 1), 0);
        assert_eq!(vol.get([5.0, -1.0, 0.0], 1), 0);
        let oob = (tex_w + 100) as f64;
        assert_eq!(vol.get([oob, oob, 0.0], 1), 0);
        // valid coords return 0 too because base is EmptyVolume — just exercise the path.
        let mid_u = (tex_w / 2) as f64;
        let mid_v = (tex_h / 2) as f64;
        let _ = vol.get([mid_u, mid_v, 0.0], 1);
    }

    #[test]
    fn paint_plane_intersection_draws_lines_through_segment() {
        let Ok(base) = TifXyzBase::load(Path::new(FIXTURE)) else {
            eprintln!("skipping: fixture not present");
            return;
        };
        // fixture bbox z range ≈ [4307, 5593]; pick a plane near the middle.
        let z_plane = 4950i32;
        let data = TifXyzData::build(base, &None);
        let vol = TifXyzVolume::from_data(data, EmptyVolume {}.into_volume(), None);
        let mut image = crate::volume::Image::new_from_color(256, 256, ecolor::Color32::TRANSPARENT);
        // XY pane: u_coord=0, v_coord=1, plane_coord=2. Center around the
        // approximate xy midpoint of the bbox, 32 voxels per pixel to span the
        // whole segment within 256 px.
        vol.paint_plane_intersection(
            [4500, 2800, z_plane],
            0,
            1,
            2,
            256,
            256,
            1,
            32,
            None,
            &DrawingConfig::default(),
            &mut image,
        );
        let drawn = image
            .data
            .iter()
            .filter(|c| **c != ecolor::Color32::TRANSPARENT)
            .count();
        assert!(drawn > 0, "expected paint_plane_intersection to draw at least one pixel");
    }

    #[test]
    fn nominal_dims_match_quadsurface_scale() {
        let Ok(base) = TifXyzBase::load(Path::new(FIXTURE)) else {
            eprintln!("skipping: fixture not present");
            return;
        };
        // Fixture: scale = 0.0078125 = 1/128. 129 cells / (1/128) = 16512.
        // Mirrors QuadSurface::scale() in volume-cartographer
        // (core/src/QuadSurface.cpp:505): {cols/scale_x, rows/scale_y}.
        let (w, h) = nominal_dims(&base);
        assert_eq!(w, 16512, "nominal width should be cols/scale_x");
        assert_eq!(h, 16512, "nominal height should be rows/scale_y");
    }
}
