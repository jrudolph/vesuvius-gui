//! Headless full-stack fixture for the download → cache → paint → tile path.
//!
//! Drives the real [`VolumePane::render`] inside a headless `egui::Context`
//! frame by frame — tile cache in egui memory, async tile futures, per-frame
//! poll budget, cross-mip placeholders, TTL re-renders — exactly as the GUI
//! does, minus the window. Textures egui would upload are kept in memory and
//! the frame's image shapes are composited on the CPU, so each frame yields
//! the pixels the user would have seen.
//!
//! Every frame is recorded ([`FrameRecord`]): tile-slot states, downloader
//! gauges/counters, render time, pane coverage and a content hash. Scripted
//! view changes (pan, drag, zoom, goto) and `run_until_settled` turn that into
//! per-phase reports ([`PhaseReport`]): time to full coverage, time to all
//! tiles ready, time to settle, bytes moved, aged-out downloads, frame cost.
//!
//! Interaction is applied by mutating coord/zoom between frames — the same
//! state `VolumePane::handle_drag`/`handle_scroll` would mutate — so no
//! synthetic pointer input is needed.
//!
//! The pane must be driven from inside a multi-threaded tokio runtime (tiles
//! render on `spawn_blocking` and are polled via `block_in_place`), like the
//! GUI's `#[tokio::main]`.

use crate::gui::{FrameBudget, PaneType, TileFrameStats, VolumePane};
use egui::epaint::{ClippedShape, ImageData, Shape};
use egui::{Color32, ColorImage, Pos2, Rect, TextureId, Vec2, ViewportId};
use std::collections::HashMap;
use std::fmt;
use std::hash::{BuildHasher, Hash, Hasher};
use std::io::Write;
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use vesuvius_rs::cache::{DownloaderStats, UnifiedCache, UnifiedVolume};
use vesuvius_rs::model::{NewVolumeReference, VolumeCreationParams};
use vesuvius_rs::volume::{DrawingConfig, TifXyzVolume, Volume};
use vesuvius_zarr::base_cache_dir;

/// The world a pane renders plus what the harness needs to observe it.
pub struct HarnessWorld {
    /// What the pane paints: the base volume, or a segment wrapping it.
    pub world: Volume,
    /// The unified-cache base volume, when the source is (ome-)zarr. Source
    /// of the downloader stats; `None` for legacy volume types.
    pub base: Option<Arc<UnifiedVolume>>,
    pub ranges: [RangeInclusive<i32>; 3],
    /// Center of the volume or segment.
    pub default_coord: [i32; 3],
    pub is_segment: bool,
}

impl HarnessWorld {
    /// Open a volume the way the GUI's `-v` does: an http(s) URL or local
    /// path to a zarr/ome-zarr. `cache_root` overrides the cache base dir
    /// (e.g. a fresh dir for cold-cache runs); default is the GUI's.
    pub fn open_volume(spec: &str, cache_root: Option<&Path>) -> Result<Self, String> {
        let vref = if spec.starts_with("http") {
            NewVolumeReference::from_url(spec)
        } else {
            NewVolumeReference::from_path(spec)
        }
        .map_err(|e| format!("cannot resolve volume {}: {}", spec, e))?;

        let root = cache_root.map(Path::to_path_buf).unwrap_or_else(base_cache_dir);
        // Same startup step as the GUI's main().
        UnifiedCache::for_cache_dir(&root).run_startup_maintenance();

        match &vref {
            NewVolumeReference::OmeZarr { id, location } => {
                let cache = NewVolumeReference::open_ome_zarr_cache(id, location, root, false);
                let [x, y, z] = cache.voxel_extent();
                let base = Arc::new(UnifiedVolume::new(cache));
                Ok(Self {
                    world: Volume::from_ref(base.clone()),
                    base: Some(base),
                    ranges: [0..=x as i32, 0..=y as i32, 0..=z as i32],
                    default_coord: [x as i32 / 2, y as i32 / 2, z as i32 / 2],
                    is_segment: false,
                })
            }
            other => {
                let params = VolumeCreationParams {
                    cache_dir: root.to_string_lossy().to_string(),
                };
                Ok(Self {
                    world: other.volume(&params),
                    base: None,
                    ranges: [0..=50000, 0..=50000, 0..=100000],
                    default_coord: [2800, 2500, 10852],
                    is_segment: false,
                })
            }
        }
    }

    /// Wrap the volume in a tifxyz segment, as the GUI's `--tifxyz` does
    /// (no transform, no overlay).
    pub fn with_tifxyz(self, dir: impl AsRef<Path>) -> Result<Self, String> {
        let dir = dir.as_ref();
        let seg = TifXyzVolume::load_from_directory(dir, self.world.clone(), &None)
            .map_err(|e| format!("cannot load tifxyz {}: {:#}", dir.display(), e))?;
        let (w, h) = (seg.width() as i32, seg.height() as i32);
        Ok(Self {
            world: Volume::from_ref(Arc::new(seg)),
            base: self.base,
            ranges: [0..=w, 0..=h, -40..=40],
            default_coord: [w / 2, h / 2, 0],
            is_segment: true,
        })
    }
}

#[derive(Clone)]
pub struct HarnessOptions {
    pub pane: PaneType,
    /// Pane size in pixels (pixels_per_point is fixed at 1).
    pub width: u32,
    pub height: u32,
    /// Target frame rate; also sets the per-frame poll budget (1 / fps), like
    /// the GUI's "Target FPS" slider.
    pub fps: u32,
    pub zoom: f32,
    pub coord: Option<[i32; 3]>,
    pub drawing_config: DrawingConfig,
    pub extra_resolutions: u32,
    /// Pace frames like an idle window: when egui didn't ask for a repaint,
    /// wait (up to 1 s) instead of rendering at `fps`. Default off: frames run
    /// continuously, like a user moving the mouse over the pane.
    pub honor_repaint: bool,
}

impl Default for HarnessOptions {
    fn default() -> Self {
        Self {
            pane: PaneType::UV,
            width: 1600,
            height: 1000,
            fps: 20,
            zoom: 1.0,
            coord: None,
            drawing_config: DrawingConfig::default(),
            extra_resolutions: 0,
            honor_repaint: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FrameRecord {
    pub frame: u64,
    /// Since harness start.
    pub t: Duration,
    /// Wall time of `ctx.run` (the pane's whole frame, incl. tile polling).
    pub render: Duration,
    pub coord: [i32; 3],
    pub zoom: f32,
    pub tiles: TileFrameStats,
    pub downloads: Option<DownloaderStats>,
    pub queued_tasks: Option<usize>,
    /// Chunks dispatched but not resident yet. Can stay > 0 with an idle
    /// downloader when fetches are stranded.
    pub pending_chunks: Option<usize>,
    /// Chunks whose fetch failed or was cancelled and not re-requested yet.
    /// They only come back when a tile re-render samples them again.
    pub cooldown_chunks: Option<usize>,
    /// Fraction of pane pixels covered by any tile texture this frame.
    pub coverage: f32,
    pub hash: u64,
    /// Pixels differ from the previous frame.
    pub changed: bool,
    pub repaint_requested: bool,
}

impl FrameRecord {
    fn loading(&self) -> u32 {
        self.tiles.loading_blank + self.tiles.loading_fallback
    }

    fn downloads_idle(&self) -> bool {
        self.downloads.map_or(true, |d| d.in_flight == 0 && d.queued == 0)
            && self.queued_tasks.map_or(true, |q| q == 0)
            && self.pending_chunks.map_or(true, |p| p == 0)
            && self.cooldown_chunks.map_or(true, |c| c == 0)
    }
}

/// What happened between a view change (or start) and the end of the wait
/// that followed it.
#[derive(Debug, Clone)]
pub struct PhaseReport {
    pub label: String,
    pub frames: usize,
    pub duration: Duration,
    /// First frame with every pane pixel covered (placeholders count).
    pub full_coverage: Option<Duration>,
    /// First frame with no tile slot still loading.
    pub all_tiles_ready: Option<Duration>,
    /// Time until downloads went idle and pixels stopped changing (only set
    /// by `run_until_settled`).
    pub settled: Option<Duration>,
    pub min_coverage: f32,
    pub max_blank_tiles: u32,
    pub downloads: Option<DownloaderStats>,
    pub max_in_flight: usize,
    pub max_queued: usize,
    /// Chunks still Pending at the end of the phase.
    pub pending_at_end: Option<usize>,
    /// Chunks in cooldown (failed / aged out, not re-requested) at the end.
    pub cooldown_at_end: Option<usize>,
    pub render_p50: Duration,
    pub render_p95: Duration,
    pub render_max: Duration,
    /// Frames whose render exceeded the frame interval.
    pub slow_frames: usize,
}

impl fmt::Display for PhaseReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let opt = |d: Option<Duration>| d.map_or("never".to_string(), |d| format!("{:.2}s", d.as_secs_f64()));
        writeln!(
            f,
            "== {} ({} frames, {:.2}s)",
            self.label,
            self.frames,
            self.duration.as_secs_f64()
        )?;
        writeln!(
            f,
            "  full coverage: {}   all tiles ready: {}   settled: {}",
            opt(self.full_coverage),
            opt(self.all_tiles_ready),
            opt(self.settled)
        )?;
        writeln!(
            f,
            "  min coverage: {:.1}%   max blank tiles: {}",
            self.min_coverage * 100.0,
            self.max_blank_tiles
        )?;
        if let Some(d) = self.downloads {
            let mb = d.bytes as f64 / 1e6;
            writeln!(
                f,
                "  downloads: {} submitted, {} ok, {} not-found, {} failed, {} aged-out, {:.1} MB ({:.2} MB/s)",
                d.submitted,
                d.completed,
                d.not_found,
                d.failed,
                d.aged_out,
                mb,
                mb / self.duration.as_secs_f64().max(1e-9)
            )?;
            writeln!(
                f,
                "  max in flight: {}   max queued: {}   chunks still pending: {}   in cooldown: {}",
                self.max_in_flight,
                self.max_queued,
                self.pending_at_end.map_or("-".to_string(), |p| p.to_string()),
                self.cooldown_at_end.map_or("-".to_string(), |c| c.to_string())
            )?;
        }
        write!(
            f,
            "  frame render p50 {:.1}ms  p95 {:.1}ms  max {:.1}ms  over budget: {}",
            self.render_p50.as_secs_f64() * 1e3,
            self.render_p95.as_secs_f64() * 1e3,
            self.render_max.as_secs_f64() * 1e3,
            self.slow_frames
        )
    }
}

pub struct PaneHarness {
    ctx: egui::Context,
    pane: VolumePane,
    world: Volume,
    base: Option<Arc<UnifiedVolume>>,
    ranges: [RangeInclusive<i32>; 3],
    opts: HarnessOptions,
    frame_interval: Duration,
    coord: [i32; 3],
    zoom: f32,
    start: Instant,
    textures: HashMap<TextureId, ColorImage>,
    pixels: Vec<Color32>,
    records: Vec<FrameRecord>,
    phase_start: usize,
    phase_label: String,
}

impl PaneHarness {
    pub fn new(world: HarnessWorld, opts: HarnessOptions) -> Self {
        let ctx = egui::Context::default();
        let coord = opts.coord.unwrap_or(world.default_coord);
        let fps = opts.fps.max(1);
        Self {
            ctx,
            pane: VolumePane::new(opts.pane, world.is_segment),
            world: world.world,
            base: world.base,
            ranges: world.ranges,
            frame_interval: Duration::from_secs_f64(1.0 / fps as f64),
            coord,
            zoom: opts.zoom,
            opts,
            start: Instant::now(),
            textures: HashMap::new(),
            pixels: Vec::new(),
            records: Vec::new(),
            phase_start: 0,
            phase_label: "start".to_string(),
        }
    }

    pub fn coord(&self) -> [i32; 3] {
        self.coord
    }

    pub fn zoom(&self) -> f32 {
        self.zoom
    }

    pub fn records(&self) -> &[FrameRecord] {
        &self.records
    }

    pub fn last(&self) -> Option<&FrameRecord> {
        self.records.last()
    }

    /// Render one frame and sleep out the rest of its interval.
    pub fn frame(&mut self) -> &FrameRecord {
        let frame_start = Instant::now();
        let budget = FrameBudget::new(self.frame_interval);
        let size = Vec2::new(self.opts.width as f32, self.opts.height as f32);

        let mut raw = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, size)),
            time: Some(self.start.elapsed().as_secs_f64()),
            ..Default::default()
        };
        raw.viewports.entry(ViewportId::ROOT).or_default().native_pixels_per_point = Some(1.0);

        let Self {
            ctx,
            pane,
            world,
            ranges,
            opts,
            frame_interval,
            coord,
            zoom,
            ..
        } = self;
        let output = ctx.run(raw, |ctx| {
            egui::CentralPanel::default().frame(egui::Frame::NONE).show(ctx, |ui| {
                // Single-pane layout: the pane gets the whole frame budget,
                // like GuiLayout::UV / XY in the app.
                pane.render(
                    ui,
                    coord,
                    world,
                    None,
                    None,
                    zoom,
                    &opts.drawing_config,
                    opts.extra_resolutions,
                    None,
                    ranges,
                    size,
                    &budget,
                    *frame_interval,
                );
            });
        });
        let render = frame_start.elapsed();

        for (id, delta) in &output.textures_delta.set {
            let ImageData::Color(image) = &delta.image;
            match delta.pos {
                None => {
                    self.textures.insert(*id, (**image).clone());
                }
                Some([x0, y0]) => {
                    if let Some(tex) = self.textures.get_mut(id) {
                        for y in 0..image.size[1] {
                            for x in 0..image.size[0] {
                                if x0 + x < tex.size[0] && y0 + y < tex.size[1] {
                                    tex.pixels[(y0 + y) * tex.size[0] + x0 + x] = image.pixels[y * image.size[0] + x];
                                }
                            }
                        }
                    }
                }
            }
        }
        let coverage = self.composite(&output.shapes);
        for id in &output.textures_delta.free {
            self.textures.remove(id);
        }

        let mut hasher = fxhash::FxBuildHasher::default().build_hasher();
        self.pixels.hash(&mut hasher);
        let hash = hasher.finish();
        let changed = self.records.last().map_or(true, |r| r.hash != hash);
        let repaint_delay = output
            .viewport_output
            .get(&ViewportId::ROOT)
            .map_or(Duration::MAX, |v| v.repaint_delay);

        let record = FrameRecord {
            frame: self.records.len() as u64,
            t: self.start.elapsed(),
            render,
            coord: self.coord,
            zoom: self.zoom,
            tiles: budget.tile_stats(),
            downloads: self.base.as_ref().map(|b| b.cache().download_stats()),
            queued_tasks: self.base.as_ref().map(|b| b.cache().queued_tasks()),
            pending_chunks: self.base.as_ref().map(|b| b.cache().pending_chunks()),
            cooldown_chunks: self.base.as_ref().map(|b| b.cache().cooldown_chunks()),
            coverage,
            hash,
            changed,
            repaint_requested: repaint_delay.is_zero(),
        };
        self.records.push(record);

        let mut next = frame_start + self.frame_interval;
        if self.opts.honor_repaint && repaint_delay > self.frame_interval {
            next = frame_start + repaint_delay.min(Duration::from_secs(1));
        }
        if let Some(rest) = next.checked_duration_since(Instant::now()) {
            std::thread::sleep(rest);
        }
        self.records.last().unwrap()
    }

    /// Composite the frame's textured meshes (each `painter.image` is an
    /// axis-aligned quad) into `self.pixels`, nearest-neighbour sampled.
    /// Returns the covered fraction of the pane.
    fn composite(&mut self, shapes: &[ClippedShape]) -> f32 {
        let (w, h) = (self.opts.width as usize, self.opts.height as usize);
        self.pixels.clear();
        self.pixels.resize(w * h, Color32::BLACK);
        let mut covered = vec![false; w * h];

        fn walk<'a>(shape: &'a Shape, clip: Rect, out: &mut Vec<(&'a egui::Mesh, Rect)>) {
            match shape {
                Shape::Vec(shapes) => shapes.iter().for_each(|s| walk(s, clip, out)),
                Shape::Mesh(mesh) => out.push((mesh, clip)),
                _ => {}
            }
        }
        let mut meshes = Vec::new();
        for cs in shapes {
            walk(&cs.shape, cs.clip_rect, &mut meshes);
        }

        for (mesh, clip) in meshes {
            let Some(tex) = self.textures.get(&mesh.texture_id) else {
                continue;
            };
            if mesh.vertices.is_empty() || tex.size[0] == 0 || tex.size[1] == 0 {
                continue;
            }
            let mut pos = Rect::NOTHING;
            let mut uv = Rect::NOTHING;
            for v in &mesh.vertices {
                pos.extend_with(v.pos);
                uv.extend_with(v.uv);
            }
            let area = pos.intersect(clip).intersect(Rect::from_min_size(Pos2::ZERO, Vec2::new(w as f32, h as f32)));
            if !area.is_positive() || pos.width() <= 0.0 || pos.height() <= 0.0 {
                continue;
            }
            let (tw, th) = (tex.size[0], tex.size[1]);
            let x0 = area.min.x.round().max(0.0) as usize;
            let x1 = (area.max.x.round() as usize).min(w);
            let y0 = area.min.y.round().max(0.0) as usize;
            let y1 = (area.max.y.round() as usize).min(h);
            for y in y0..y1 {
                let fv = (y as f32 + 0.5 - pos.min.y) / pos.height();
                let ty = (((uv.min.y + fv * uv.height()) * th as f32) as usize).min(th - 1);
                for x in x0..x1 {
                    let fu = (x as f32 + 0.5 - pos.min.x) / pos.width();
                    let tx = (((uv.min.x + fu * uv.width()) * tw as f32) as usize).min(tw - 1);
                    let src = tex.pixels[ty * tw + tx];
                    let i = y * w + x;
                    if src.a() == 255 {
                        self.pixels[i] = src;
                    } else if src.a() > 0 {
                        // Premultiplied alpha "over".
                        let dst = self.pixels[i];
                        let k = 255 - src.a() as u32;
                        let mix = |s: u8, d: u8| (s as u32 + d as u32 * k / 255).min(255) as u8;
                        self.pixels[i] = Color32::from_rgba_premultiplied(
                            mix(src.r(), dst.r()),
                            mix(src.g(), dst.g()),
                            mix(src.b(), dst.b()),
                            mix(src.a(), dst.a()),
                        );
                    } else {
                        continue;
                    }
                    covered[i] = true;
                }
            }
        }
        covered.iter().filter(|c| **c).count() as f32 / (w * h).max(1) as f32
    }

    /// Start a new phase. Closes the current one and returns its report.
    pub fn begin_phase(&mut self, label: impl Into<String>) -> PhaseReport {
        let report = self.phase_report(None);
        self.phase_start = self.records.len();
        self.phase_label = label.into();
        report
    }

    pub fn run_frames(&mut self, n: usize) {
        for _ in 0..n {
            self.frame();
        }
    }

    pub fn run_for(&mut self, d: Duration) {
        let end = Instant::now() + d;
        while Instant::now() < end {
            self.frame();
        }
    }

    /// Run until no tile slot is loading, downloads and cache tasks are idle,
    /// no chunk is pending or in cooldown, and the pixels haven't changed for
    /// `quiet`. Cooldown chunks count as unfinished even after their cooldown
    /// expires (they wait for a tile re-render to be requested again) — so
    /// chunks that failed and then scrolled off-screen keep a later phase
    /// from settling; `cooldown_at_end` in the report shows that. Returns the time from the
    /// start of the current phase to settling, or `None` on timeout.
    pub fn run_until_settled(&mut self, timeout: Duration, quiet: Duration) -> Option<Duration> {
        let deadline = Instant::now() + timeout;
        let mut last_change = self.start.elapsed();
        loop {
            let r = self.frame().clone();
            if r.changed || r.loading() > 0 || !r.downloads_idle() {
                last_change = r.t;
            } else if r.t.saturating_sub(last_change) >= quiet {
                return Some(last_change.saturating_sub(self.phase_t0()));
            }
            if Instant::now() >= deadline {
                return None;
            }
        }
    }

    fn phase_t0(&self) -> Duration {
        self.records
            .get(self.phase_start)
            .map_or(self.start.elapsed(), |r| r.t.saturating_sub(r.render))
    }

    /// Move the view by screen pixels (positive dx = content moves left, like
    /// dragging the mouse to the left).
    pub fn pan_screen(&mut self, dx: f32, dy: f32) {
        let (u, v, _) = self.opts.pane.coordinates();
        self.coord[u] = (self.coord[u] + (dx / self.zoom) as i32).clamp(*self.ranges[u].start(), *self.ranges[u].end());
        self.coord[v] = (self.coord[v] + (dy / self.zoom) as i32).clamp(*self.ranges[v].start(), *self.ranges[v].end());
    }

    /// Pan by (dx, dy) screen pixels spread evenly over `duration`, rendering
    /// every frame on the way — a continuous drag.
    pub fn drag(&mut self, dx: f32, dy: f32, duration: Duration) {
        let frames = (duration.as_secs_f64() * self.opts.fps.max(1) as f64).ceil().max(1.0) as usize;
        let (mut done_x, mut done_y) = (0.0f32, 0.0f32);
        for i in 1..=frames {
            let f = i as f32 / frames as f32;
            let (sx, sy) = ((dx * f).round() - done_x, (dy * f).round() - done_y);
            self.pan_screen(sx, sy);
            done_x += sx;
            done_y += sy;
            self.frame();
        }
    }

    pub fn set_zoom(&mut self, zoom: f32) {
        self.zoom = zoom;
    }

    pub fn set_coord(&mut self, coord: [i32; 3]) {
        self.coord = coord;
    }

    /// Report for the current phase so far. `settled` is filled in by the
    /// caller from `run_until_settled`.
    pub fn phase_report(&self, settled: Option<Duration>) -> PhaseReport {
        let recs = &self.records[self.phase_start.min(self.records.len())..];
        let t0 = self.phase_t0();
        let first = |pred: &dyn Fn(&FrameRecord) -> bool| recs.iter().find(|r| pred(r)).map(|r| r.t.saturating_sub(t0));
        let mut renders: Vec<Duration> = recs.iter().map(|r| r.render).collect();
        renders.sort();
        let pct = |p: f64| {
            renders
                .get(((renders.len() as f64 - 1.0) * p).round().max(0.0) as usize)
                .copied()
                .unwrap_or_default()
        };
        let before = self
            .phase_start
            .checked_sub(1)
            .and_then(|i| self.records.get(i))
            .and_then(|r| r.downloads)
            .unwrap_or_default();
        let downloads = recs.last().and_then(|r| r.downloads).map(|d| DownloaderStats {
            in_flight: d.in_flight,
            queued: d.queued,
            submitted: d.submitted - before.submitted,
            completed: d.completed - before.completed,
            not_found: d.not_found - before.not_found,
            failed: d.failed - before.failed,
            aged_out: d.aged_out - before.aged_out,
            bytes: d.bytes - before.bytes,
        });
        PhaseReport {
            label: self.phase_label.clone(),
            frames: recs.len(),
            duration: recs.last().map_or(Duration::ZERO, |r| r.t.saturating_sub(t0)),
            full_coverage: first(&|r| r.coverage >= 0.9999),
            all_tiles_ready: first(&|r| r.loading() == 0),
            settled,
            min_coverage: recs.iter().map(|r| r.coverage).fold(1.0, f32::min),
            max_blank_tiles: recs.iter().map(|r| r.tiles.loading_blank).max().unwrap_or(0),
            downloads,
            max_in_flight: recs.iter().filter_map(|r| r.downloads).map(|d| d.in_flight).max().unwrap_or(0),
            max_queued: recs.iter().filter_map(|r| r.downloads).map(|d| d.queued).max().unwrap_or(0),
            pending_at_end: recs.last().and_then(|r| r.pending_chunks),
            cooldown_at_end: recs.last().and_then(|r| r.cooldown_chunks),
            render_p50: pct(0.5),
            render_p95: pct(0.95),
            render_max: renders.last().copied().unwrap_or_default(),
            slow_frames: recs.iter().filter(|r| r.render > self.frame_interval).count(),
        }
    }

    /// The last composited frame as an RGBA image.
    pub fn snapshot(&self) -> image::RgbaImage {
        let (w, h) = (self.opts.width, self.opts.height);
        let mut img = image::RgbaImage::new(w, h);
        for (i, p) in self.pixels.iter().enumerate() {
            let [r, g, b, a] = p.to_srgba_unmultiplied();
            img.put_pixel(i as u32 % w, i as u32 / w, image::Rgba([r, g, b, a]));
        }
        img
    }

    pub fn save_png(&self, path: impl AsRef<Path>) -> Result<(), String> {
        self.snapshot().save(path.as_ref()).map_err(|e| e.to_string())
    }

    /// One row per frame.
    pub fn write_timeline_csv(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        writeln!(
            f,
            "frame,t_ms,render_ms,u,v,w,zoom,visible,ready,recalculating,loading_fallback,loading_blank,\
             budget_skipped,coverage,changed,repaint,dl_in_flight,dl_queued,dl_submitted,dl_completed,\
             dl_not_found,dl_failed,dl_aged_out,dl_bytes,queued_tasks,pending_chunks,cooldown_chunks"
        )?;
        for r in &self.records {
            let d = r.downloads.unwrap_or_default();
            writeln!(
                f,
                "{},{:.1},{:.2},{},{},{},{},{},{},{},{},{},{},{:.4},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                r.frame,
                r.t.as_secs_f64() * 1e3,
                r.render.as_secs_f64() * 1e3,
                r.coord[0],
                r.coord[1],
                r.coord[2],
                r.zoom,
                r.tiles.visible,
                r.tiles.ready,
                r.tiles.recalculating,
                r.tiles.loading_fallback,
                r.tiles.loading_blank,
                r.tiles.budget_skipped,
                r.coverage,
                r.changed as u8,
                r.repaint_requested as u8,
                d.in_flight,
                d.queued,
                d.submitted,
                d.completed,
                d.not_found,
                d.failed,
                d.aged_out,
                d.bytes,
                r.queued_tasks.unwrap_or(0),
                r.pending_chunks.unwrap_or(0),
                r.cooldown_chunks.unwrap_or(0)
            )?;
        }
        Ok(())
    }
}

/// One step of a harness script. Text form (`;`-separated, see
/// [`parse_script`]):
///
/// - `settle [timeout_s=300] [quiet_s=2]` — run until settled
/// - `wait <s>` — run frames for a fixed time
/// - `frames <n>`
/// - `pan <dx> <dy>` — jump by screen pixels (starts a phase)
/// - `drag <dx> <dy> <s>` — continuous pan over `s` seconds (starts a phase)
/// - `zoom <z>` (starts a phase)
/// - `goto <u> <v> <w>` (starts a phase)
/// - `png <name>` — save the current frame as `<name>.png`
#[derive(Debug, Clone, PartialEq)]
pub enum Step {
    Settle { timeout: Duration, quiet: Duration },
    Wait(Duration),
    Frames(usize),
    Pan(f32, f32),
    Drag(f32, f32, Duration),
    Zoom(f32),
    Goto([i32; 3]),
    Png(String),
}

pub fn parse_script(script: &str) -> Result<Vec<Step>, String> {
    let secs = |s: &str| {
        s.parse::<f64>()
            .map(Duration::from_secs_f64)
            .map_err(|e| format!("bad seconds `{}`: {}", s, e))
    };
    let num = |s: &str| s.parse::<f32>().map_err(|e| format!("bad number `{}`: {}", s, e));
    let int = |s: &str| s.parse::<i32>().map_err(|e| format!("bad integer `{}`: {}", s, e));
    let mut steps = Vec::new();
    for cmd in script.split(';').map(str::trim).filter(|c| !c.is_empty()) {
        let parts: Vec<&str> = cmd.split_whitespace().collect();
        let arity = |n: usize| {
            if parts.len() == n + 1 {
                Ok(())
            } else {
                Err(format!("`{}` takes {} argument(s)", parts[0], n))
            }
        };
        let step = match parts[0] {
            "settle" => Step::Settle {
                timeout: parts.get(1).map_or(Ok(Duration::from_secs(300)), |s| secs(s))?,
                quiet: parts.get(2).map_or(Ok(Duration::from_secs(2)), |s| secs(s))?,
            },
            "wait" => {
                arity(1)?;
                Step::Wait(secs(parts[1])?)
            }
            "frames" => {
                arity(1)?;
                Step::Frames(parts[1].parse().map_err(|e| format!("bad frame count: {}", e))?)
            }
            "pan" => {
                arity(2)?;
                Step::Pan(num(parts[1])?, num(parts[2])?)
            }
            "drag" => {
                arity(3)?;
                Step::Drag(num(parts[1])?, num(parts[2])?, secs(parts[3])?)
            }
            "zoom" => {
                arity(1)?;
                Step::Zoom(num(parts[1])?)
            }
            "goto" => {
                arity(3)?;
                Step::Goto([int(parts[1])?, int(parts[2])?, int(parts[3])?])
            }
            "png" => {
                arity(1)?;
                Step::Png(parts[1].to_string())
            }
            other => return Err(format!("unknown script command `{}`", other)),
        };
        steps.push(step);
    }
    Ok(steps)
}

/// Run a script, printing (via `report`) a [`PhaseReport`] after every
/// `settle`/`wait`/`frames` step. PNGs go to `out_dir`.
pub fn run_script(
    harness: &mut PaneHarness,
    steps: &[Step],
    out_dir: &Path,
    mut report: impl FnMut(&PhaseReport),
) -> Result<Vec<PhaseReport>, String> {
    let mut reports = Vec::new();
    for step in steps {
        match step {
            Step::Settle { timeout, quiet } => {
                let settled = harness.run_until_settled(*timeout, *quiet);
                let r = harness.phase_report(settled);
                report(&r);
                reports.push(r);
            }
            Step::Wait(d) => {
                harness.run_for(*d);
                let r = harness.phase_report(None);
                report(&r);
                reports.push(r);
            }
            Step::Frames(n) => {
                harness.run_frames(*n);
                let r = harness.phase_report(None);
                report(&r);
                reports.push(r);
            }
            Step::Pan(dx, dy) => {
                harness.begin_phase(format!("pan {} {}", dx, dy));
                harness.pan_screen(*dx, *dy);
            }
            Step::Drag(dx, dy, d) => {
                harness.begin_phase(format!("drag {} {} over {:.1}s", dx, dy, d.as_secs_f64()));
                harness.drag(*dx, *dy, *d);
            }
            Step::Zoom(z) => {
                harness.begin_phase(format!("zoom {}", z));
                harness.set_zoom(*z);
            }
            Step::Goto(c) => {
                harness.begin_phase(format!("goto {:?}", c));
                harness.set_coord(*c);
            }
            Step::Png(name) => {
                let path: PathBuf = out_dir.join(format!("{}.png", name));
                harness.save_png(&path)?;
            }
        }
    }
    Ok(reports)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_script() {
        let steps = parse_script("settle; drag 800 0 2.5; settle 60 1; zoom 0.45; png end").unwrap();
        assert_eq!(
            steps,
            vec![
                Step::Settle {
                    timeout: Duration::from_secs(300),
                    quiet: Duration::from_secs(2)
                },
                Step::Drag(800.0, 0.0, Duration::from_secs_f64(2.5)),
                Step::Settle {
                    timeout: Duration::from_secs(60),
                    quiet: Duration::from_secs(1)
                },
                Step::Zoom(0.45),
                Step::Png("end".to_string()),
            ]
        );
        assert!(parse_script("zoom").is_err());
        assert!(parse_script("fly 1").is_err());
    }
}
