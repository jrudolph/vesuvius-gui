use anyhow::{anyhow, Result};
use clap::Parser;
use futures::{stream, StreamExt};
use image::Luma;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::io::IsTerminal;
use std::ops::RangeInclusive;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use vesuvius_rs::cache::epoch::CAP_ENV_VAR;
use vesuvius_rs::cache::{ChunkCache, ChunkKey, ChunkState, UnifiedCache, UnifiedVolume};
use vesuvius_rs::model::NewVolumeReference;
use vesuvius_rs::volume::{
    AffineTransform, CompositingMode, CompositingSettings, DrawingConfig, EmptyVolume, Image, ObjFile, ObjVolume,
    OverlayColoring, OverlayVolume, PaintVolume, ProjectionKind, TifXyzData, TifXyzVolume, Volume, VolumeCons,
    VoxelPaintVolume, VoxelVolume,
};
use vesuvius_zarr::base_cache_dir;

mod metadata;
mod tiled_tiff;

/// Wall-clock budget for making one tile's chunks resident before we give up
/// and render with whatever is available. Chunk fetches are normally far
/// faster; this only guards against a permanently failing source so the run
/// can't hang forever.
const TILE_ENSURE_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Clone, Debug, serde::Serialize)]
pub struct Crop {
    pub top: usize,
    pub left: usize,
    pub width: usize,
    pub height: usize,
}
#[derive(Clone)]
struct CropParser;
impl clap::builder::TypedValueParser for CropParser {
    type Value = Crop;

    fn parse_ref(
        &self,
        _cmd: &clap::Command,
        _arg: Option<&clap::Arg>,
        value: &std::ffi::OsStr,
    ) -> std::result::Result<Self::Value, clap::Error> {
        // parse a value like 0+0-0x0 with a regexp

        let re = regex::Regex::new(r"(\d+)\+(\d+)-(\d+)x(\d+)").unwrap();
        let captures = re.captures(value.to_str().unwrap()).ok_or(clap::Error::raw(
            clap::error::ErrorKind::ValueValidation,
            "--crop argument could not be parsed. Use '--crop <left>+<top>-<width>x<height>', e.g. '--crop 1000+1000-500x500'.",
        ))?;
        let left = captures.get(1).unwrap().as_str().parse().unwrap();
        let top = captures.get(2).unwrap().as_str().parse().unwrap();
        let width = captures.get(3).unwrap().as_str().parse().unwrap();
        let height = captures.get(4).unwrap().as_str().parse().unwrap();
        Ok(Crop {
            top,
            left,
            width,
            height,
        })
    }
}

/// Compositing mode selectable on the command line. Mirrors
/// `volume::CompositingMode` one-to-one; kept separate so the CLI crate owns
/// the `clap::ValueEnum` derive. The `alpha-overlay*` modes additionally
/// require `--overlay` (the overlay supplies per-sample opacity).
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum, serde::Serialize)]
pub enum CompositeModeArg {
    None,
    Max,
    Alpha,
    AlphaHeightMap,
    AlphaOverlay,
    AlphaOverlayStart,
    AlphaOverlayCombined,
}
impl From<CompositeModeArg> for CompositingMode {
    fn from(m: CompositeModeArg) -> Self {
        match m {
            CompositeModeArg::None => CompositingMode::None,
            CompositeModeArg::Max => CompositingMode::Max,
            CompositeModeArg::Alpha => CompositingMode::Alpha,
            CompositeModeArg::AlphaHeightMap => CompositingMode::AlphaHeightMap,
            CompositeModeArg::AlphaOverlay => CompositingMode::AlphaOverlay,
            CompositeModeArg::AlphaOverlayStart => CompositingMode::AlphaOverlayStart,
            CompositeModeArg::AlphaOverlayCombined => CompositingMode::AlphaOverlayCombined,
        }
    }
}

/// Alpha / compositing options. By default no compositing is done (a single
/// plane is sampled per layer). `--composite-mode` (or the `--enable-alpha`
/// shortcut) switches to compositing along the surface normal across the
/// configured layer window.
#[derive(Parser, Debug, Clone, serde::Serialize)]
pub struct AlphaArgs {
    /// Compositing mode. Overrides `--enable-alpha` when set. The
    /// `alpha-overlay*` modes require `--overlay`.
    #[clap(long, value_enum)]
    composite_mode: Option<CompositeModeArg>,

    /// Shortcut for `--composite-mode alpha`. Ignored if `--composite-mode` is set.
    #[clap(long, default_value_t = false)]
    enable_alpha: bool,

    /// Number of layers to composite in front of the surface (default 6)
    #[clap(long)]
    composite_layers_in_front: Option<u8>,

    /// Number of layers to composite behind the surface (default 6)
    #[clap(long)]
    composite_layers_behind: Option<u8>,

    /// Alpha ramp lower bound, 0-255 (default 76)
    #[clap(long)]
    alpha_min: Option<u8>,

    /// Alpha ramp upper bound, 0-255 (default 178)
    #[clap(long)]
    alpha_max: Option<u8>,

    /// Alpha threshold, 0-10000 (default 9500)
    #[clap(long)]
    alpha_threshold: Option<u16>,

    /// Opacity multiplier (default 1)
    #[clap(long)]
    opacity: Option<u16>,

    /// AlphaOverlayCombined only: how much of the plain (unmasked) alpha
    /// result is crossfaded into the overlay-masked result, in percent
    /// (0-100). 0 shows the pure mask, 100 reproduces the regular alpha walk.
    #[clap(long)]
    overlay_background: Option<u8>,

    /// AlphaOverlayCombined only: how strongly the masked value is normalized
    /// by its accumulated coverage, in percent (0-100). 0 keeps the classic
    /// premultiplied look, 100 fully normalizes.
    #[clap(long)]
    overlay_value_norm: Option<u8>,

    /// Reverse the compositing direction along the normal
    #[clap(long, default_value_t = false)]
    composite_reverse_direction: bool,
}
impl AlphaArgs {
    fn to_settings(&self) -> CompositingSettings {
        let mut settings = DrawingConfig::default().compositing;
        settings.mode = match self.composite_mode {
            Some(mode) => mode.into(),
            None if self.enable_alpha => CompositingMode::Alpha,
            None => CompositingMode::None,
        };
        if let Some(v) = self.composite_layers_in_front {
            settings.layers_in_front = v;
        }
        if let Some(v) = self.composite_layers_behind {
            settings.layers_behind = v;
        }
        if let Some(v) = self.alpha_min {
            settings.alpha_min = v;
        }
        if let Some(v) = self.alpha_max {
            settings.alpha_max = v;
        }
        if let Some(v) = self.alpha_threshold {
            settings.alpha_threshold = v;
        }
        if let Some(v) = self.opacity {
            settings.opacity = v;
        }
        if let Some(v) = self.overlay_background {
            settings.overlay_background = v;
        }
        if let Some(v) = self.overlay_value_norm {
            settings.overlay_value_norm = v;
        }
        settings.reverse_direction = self.composite_reverse_direction;
        settings
    }
}

/// Vesuvius Renderer, a tool to render segments from obj files or vc3d tifxyz directories
#[derive(Parser, Debug, serde::Serialize)]
#[command(about, long_about = None)]
pub struct Args {
    /// Provide an obj segment file to render. Requires --width and --height.
    /// Mutually exclusive with --tifxyz.
    #[clap(long, conflicts_with = "tifxyz")]
    obj: Option<String>,

    /// Render a vc3d tifxyz directory (containing meta.json + x/y/z.tif). Grid
    /// dimensions are read from the TIFFs, so --width/--height are not required.
    /// Mutually exclusive with --obj.
    #[clap(long, conflicts_with = "obj")]
    tifxyz: Option<String>,

    /// Width of the segment file when browsing obj files
    #[clap(long)]
    width: Option<u32>,
    /// Height of the segment file when browsing obj files
    #[clap(long)]
    height: Option<u32>,

    /// Transform to apply to the segment (to map between different scans). You can either supply a filename to a transform json file
    /// (as defined in https://github.com/ScrollPrize/villa/blob/main/foundation/volume-registration/transform_schema.json) or supply
    /// a 4x3 affine transformation matrix as a json array string directly
    #[clap(long)]
    transform: Option<String>,

    /// Use orthographic projection along the Y axis (top-down view) when loading obj files (discarding existing texture coordinates).
    #[clap(long, default_value_t = false)]
    ortho_xz: bool,

    /// Invert the transform before applying it
    #[clap(long)]
    invert_transform: bool,

    /// Scale the output UV resolution by the transform's linear scale factor,
    /// so the unrolled segment is rendered at the *target* volume's resolution
    /// instead of the source's. The source UV extent is the catalog
    /// --width/--height for obj, or `grid ÷ scale` for tifxyz; when the
    /// transform magnifies the surface (e.g. mapping a coarse scan into a finer
    /// one) the source-resolution render is blurry and crop coords authored
    /// against the target volume fall out of bounds. Scales --width/--height
    /// (and thus --crop) too. No effect without --transform.
    #[clap(long, default_value_t = false)]
    scale_uv_by_transform: bool,

    /// The target directory to save the rendered images
    #[clap(long)]
    target_dir: String,

    /// Output layer id that corresponds to the segment surface (default 32)
    #[clap(long)]
    middle_layer: Option<u8>,

    /// Minimum layer id to render (default 25)
    #[clap(long)]
    min_layer: Option<u8>,

    /// Maximum layer id to render (default 41)
    #[clap(long)]
    max_layer: Option<u8>,

    /// Crop a specific region from the segment. The format is <left>+<top>-<width>x<height>.
    #[clap(long, value_parser = CropParser)]
    crop: Option<Crop>,

    /// File extension / image format to use for layers (default png)
    #[clap(long)]
    target_format: Option<String>,

    /// The ome-zarr volume to render against, given as an http(s) URL or a local path.
    #[clap(short, long)]
    volume: Option<String>,

    /// Optional ome-zarr overlay/label volume (e.g. an ink prediction) composited
    /// with the base volume, given as an http(s) URL or a local path. Required for
    /// the `alpha-overlay*` compositing modes, where the overlay supplies per-sample
    /// opacity while the base volume supplies the CT value.
    #[clap(long)]
    overlay: Option<String>,

    /// Override the data directory. By default, a directory in the user's cache is used
    #[clap(short, long)]
    data_directory: Option<String>,

    /// The tile size to split a segment into (for ergonomic reasons) (default 1024)
    #[clap(long)]
    tile_size: Option<u32>,

    /// Number of tiles to render in parallel (CPU-bound stage). Defaults to
    /// the number of CPU threads.
    #[clap(long)]
    render_concurrency: Option<usize>,

    /// How many tiles to fetch/ensure ahead of rendering. The ensure stage
    /// runs `render_concurrency + prefetch_depth` tiles concurrently, so the
    /// next tiles' chunks download while the current ones render. Together
    /// these bound the cache working set (~(render_concurrency +
    /// prefetch_depth) tiles); lower them if the purge log shows repeated
    /// evict/refetch cycles (default 4).
    #[clap(long)]
    prefetch_depth: Option<usize>,

    /// Override the unified cache size cap, in GB (sets VESUVIUS_CACHE_CAP_GB).
    /// Raise this for large renders so the working set fits without purging.
    #[clap(long)]
    cache_cap_gb: Option<u64>,

    /// CPU-bound blocking threads for the tokio runtime (default:
    /// render_concurrency + prefetch_depth + 2). Only the collect and render
    /// stages use blocking threads; the ensure stage waits asynchronously.
    #[clap(long)]
    worker_threads: Option<usize>,

    /// Emit a single plain progress line every N seconds (flushed) instead of
    /// the interactive progress bars. Useful for non-interactive logs (e.g.
    /// Kubernetes), where the bars draw to a hidden target and nothing appears.
    /// Auto-enabled (at 5s) when stderr is not a TTY; pass 0 to disable.
    #[clap(long)]
    progress_interval: Option<u64>,

    #[clap(flatten)]
    alpha: AlphaArgs,
}
impl Args {
    /// Path to the `--obj` segment file, if one was given.
    pub fn obj_path(&self) -> Option<&str> {
        self.obj.as_deref()
    }
    /// Path to the `--tifxyz` directory, if one was given.
    pub fn tifxyz_path(&self) -> Option<&str> {
        self.tifxyz.as_deref()
    }
}

fn main() -> Result<()> {
    env_logger::init();
    eprintln!(
        "vesuvius-render {} (git {}, built {})",
        env!("CARGO_PKG_VERSION"),
        metadata::GIT_REVISION,
        metadata::BUILD_TIME
    );
    let args = Args::parse();
    // The blocking pool runs the collect + render stages; the ensure stage's
    // wait is async and holds no blocking thread. Give the pool room for both
    // the renders and the prefetch tiles' collects so prefetch isn't starved.
    let render_conc = args.render_concurrency.unwrap_or(num_cpus::get());
    let prefetch = args.prefetch_depth.unwrap_or(4);
    let threads = args.worker_threads.unwrap_or(render_conc + prefetch + 2);

    // The cache cap is read once (cached) the first time a cache is built, so
    // it has to be set before any cache construction happens.
    if let Some(gb) = args.cache_cap_gb {
        std::env::set_var(CAP_ENV_VAR, gb.to_string());
    }

    let result = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .max_blocking_threads(threads)
        .build()
        .unwrap()
        .block_on(main_run(args));

    // Normal / error exit: durably record everything this run wrote. The
    // per-tile sync watchdog only fires every few seconds, so without this a
    // run that finishes between ticks loses its last batch of residency and
    // the next run re-decodes it. (Ctrl+C / SIGTERM are handled separately by
    // the signal task installed in `main_run`.)
    UnifiedCache::shutdown_all();

    result
}

/// Flush + persist all cache state on interruption. A partial render that's
/// Ctrl+C'd (or SIGTERM'd by a job scheduler) would otherwise never reach the
/// flush at the end of `Rendering::run`, so its decoded chunks never get
/// marked resident on disk and the next run starts cold. Mirrors the GUI's
/// `on_exit` hook.
fn install_shutdown_handler() {
    tokio::spawn(async {
        #[cfg(unix)]
        let mut sigterm = match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(s) => Some(s),
            Err(e) => {
                log::warn!("could not install SIGTERM handler: {}", e);
                None
            }
        };

        #[cfg(unix)]
        {
            let term = async {
                match sigterm.as_mut() {
                    Some(s) => {
                        s.recv().await;
                    }
                    None => std::future::pending::<()>().await,
                }
            };
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = term => {}
            }
        }
        #[cfg(not(unix))]
        {
            let _ = tokio::signal::ctrl_c().await;
        }

        log::warn!("interrupted: flushing cache before exit");
        UnifiedCache::shutdown_all();
        // 130 = 128 + SIGINT, the conventional "terminated by Ctrl+C" code.
        std::process::exit(130);
    });
}

async fn main_run(args: Args) -> Result<()> {
    install_shutdown_handler();

    let multi = MultiProgress::new();
    monitor_runtime_stats(&multi).await;

    // Plain-line progress for non-interactive logs. The indicatif bars draw to
    // a hidden target when stderr isn't a TTY (k8s, CI), so without this a
    // long render produces no output at all. Auto-enable at 5s off a TTY; an
    // explicit --progress-interval wins (0 disables it everywhere).
    let plain_progress = match args.progress_interval {
        Some(0) => None,
        Some(secs) => Some(Duration::from_secs(secs)),
        None if !std::io::stderr().is_terminal() => Some(Duration::from_secs(5)),
        None => None,
    };

    let params = (&args).into();
    let rendering = Rendering::new(params, &args)?;
    rendering.run(&multi, plain_progress).await?;
    Ok(())
}

/// Format a duration as `HH:MM:SS` for the plain progress line.
fn format_elapsed(d: Duration) -> String {
    let s = d.as_secs();
    format!("{:02}:{:02}:{:02}", s / 3600, (s % 3600) / 60, s % 60)
}

/// Spawn a task that prints one flushed progress line every `interval` until
/// aborted, summarizing each named bar's position/length. Used in place of the
/// interactive bars when stderr isn't a TTY. The returned handle is aborted
/// (and a final line printed) by the caller once the pipeline completes.
fn spawn_plain_progress_logger(
    bars: Vec<(&'static str, ProgressBar)>,
    interval: Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(interval).await;
            print_plain_progress(&bars);
        }
    })
}

fn print_plain_progress(bars: &[(&'static str, ProgressBar)]) {
    use std::io::Write;
    let elapsed = bars.first().map(|(_, b)| b.elapsed()).unwrap_or_default();
    let mut parts = vec![format!("elapsed={}", format_elapsed(elapsed))];
    for (label, bar) in bars {
        let len = bar.length().unwrap_or(0);
        let pos = bar.position();
        let pct = if len > 0 { pos as f64 / len as f64 * 100.0 } else { 0.0 };
        parts.push(format!("{}={}/{} ({:.0}%)", label, pos, len, pct));
    }
    if let Some(io) = format_io_stats(elapsed) {
        parts.push(io);
    }
    let mut stderr = std::io::stderr();
    let _ = writeln!(stderr, "[progress] {}", parts.join("  "));
    let _ = stderr.flush();
}

/// Compact local-fetch I/O segment for the periodic progress line. Returns
/// `None` until at least one chunk has been read. `read`/`decode` are per-chunk
/// means; `rd` is the share of busy time spent waiting on the store (vs CPU
/// decode); `MiB/s` is cumulative store read throughput over `elapsed`.
fn format_io_stats(elapsed: Duration) -> Option<String> {
    let s = vesuvius_zarr::metrics::snapshot();
    if s.chunk_count == 0 {
        return None;
    }
    let mib = s.store_bytes as f64 / (1024.0 * 1024.0);
    let mean_read_ms = s.read_ns as f64 / s.chunk_count as f64 / 1e6;
    let mean_decode_ms = s.decode_ns as f64 / s.chunk_count as f64 / 1e6;
    let secs = elapsed.as_secs_f64().max(1e-9);
    Some(format!(
        "io chunks={} MiB={:.1} read={:.2}ms decode={:.2}ms rd={:.0}% conc={:.1} peak={} MiB/s={:.1}",
        s.chunk_count,
        mib,
        mean_read_ms,
        mean_decode_ms,
        s.read_fraction() * 100.0,
        s.mean_concurrency(elapsed),
        s.peak_reads,
        mib / secs,
    ))
}

/// One-shot end-of-run summary of local chunk-fetch behaviour, written to
/// stderr. This is the line to record for each point of a worker-count sweep.
///
/// `conc` is the mean number of reads+decodes actually in flight (busy time /
/// wall) and `peak` the high-water mark — the real fetch width, independent of
/// the configured pool size, which `cache_workers` is shown alongside only for
/// reference. A high `read_frac` means store latency dominates (favor more
/// concurrency / fewer round-trips); a high decode share means CPU dominates.
/// If `conc` sits well below `cache_workers`, the limiter is upstream of the
/// fetch path (scheduling / in-order render head), not the pool width.
fn print_render_summary(elapsed: Duration, cache_workers: usize) {
    use std::io::Write;
    let s = vesuvius_zarr::metrics::snapshot();
    let mut stderr = std::io::stderr();
    if s.chunk_count == 0 {
        let _ = writeln!(stderr, "[summary] no local chunk reads recorded");
        let _ = stderr.flush();
        return;
    }
    let mib = s.store_bytes as f64 / (1024.0 * 1024.0);
    let secs = elapsed.as_secs_f64().max(1e-9);
    let mean_read_ms = s.read_ns as f64 / s.chunk_count as f64 / 1e6;
    let mean_decode_ms = s.decode_ns as f64 / s.chunk_count as f64 / 1e6;
    let _ = writeln!(
        stderr,
        "[summary] local-fetch chunks={} read={:.1}MiB wall={} \
         read_total={:.1}s (mean {:.2}ms) decode_total={:.1}s (mean {:.2}ms) read_frac={:.0}% \
         conc={:.1} peak={} cache_workers={} throughput={:.1}chunks/s {:.1}MiB/s",
        s.chunk_count,
        mib,
        format_elapsed(elapsed),
        s.read_ns as f64 / 1e9,
        mean_read_ms,
        s.decode_ns as f64 / 1e9,
        mean_decode_ms,
        s.read_fraction() * 100.0,
        s.mean_concurrency(elapsed),
        s.peak_reads,
        cache_workers,
        s.chunk_count as f64 / secs,
        mib / secs,
    );
    let _ = stderr.flush();
}

#[derive(Clone)]
struct RenderParams {
    width: usize,
    height: usize,
    projection: ProjectionKind,
    tile_size: usize,
    w_range: RangeInclusive<usize>,
    crop: Option<Crop>,
    mid_layer: usize,
    target_dir: String,
    target_format: String,
    compositing: CompositingSettings,
    render_concurrency: usize,
    prefetch_depth: usize,
    /// Provenance JSON (build version + invocation params) embedded into every
    /// rendered layer image. Built once from `Args` in `From<&Args>`.
    metadata_json: String,
}
impl RenderParams {
    fn render_left(&self) -> usize {
        self.crop.as_ref().map(|c| c.left).unwrap_or(0)
    }
    fn render_top(&self) -> usize {
        self.crop.as_ref().map(|c| c.top).unwrap_or(0)
    }
    fn render_width(&self) -> usize {
        self.crop.as_ref().map(|c| c.width).unwrap_or(self.width)
    }
    fn render_height(&self) -> usize {
        self.crop.as_ref().map(|c| c.height).unwrap_or(self.height)
    }
    /// The drawing config shared by the chunk-collection pass and the render
    /// pass — they MUST agree so the chunks we make resident are exactly the
    /// ones the render samples.
    fn drawing_config(&self) -> DrawingConfig {
        let mut config = DrawingConfig::default();
        config.trilinear_interpolation = true;
        config.compositing = self.compositing.clone();
        config
    }
}
impl From<&Args> for RenderParams {
    fn from(args: &Args) -> Self {
        Self {
            // For --obj these are the (required) catalog dims; for --tifxyz they
            // are placeholders overridden in `Rendering::new` from the TIFF grid.
            width: args.width.unwrap_or(0) as usize,
            height: args.height.unwrap_or(0) as usize,
            projection: if args.ortho_xz {
                ProjectionKind::OrthographicXZ
            } else {
                ProjectionKind::None
            },
            tile_size: args.tile_size.unwrap_or(1024) as usize,
            w_range: args.min_layer.unwrap_or(25) as usize..=args.max_layer.unwrap_or(41) as usize,
            crop: args.crop.clone(),
            mid_layer: args.middle_layer.unwrap_or(32) as usize,
            target_dir: args.target_dir.clone(),
            target_format: args.target_format.clone().unwrap_or("png".to_string()),
            compositing: args.alpha.to_settings(),
            render_concurrency: args.render_concurrency.unwrap_or(num_cpus::get()).max(1),
            prefetch_depth: args.prefetch_depth.unwrap_or(4),
            metadata_json: metadata::build_metadata_json(args),
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct UVTile {
    u: usize,
    v: usize,
    w: usize,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct VolumeChunk {
    x: usize,
    y: usize,
    z: usize,
}
impl From<(usize, usize, usize)> for VolumeChunk {
    fn from((x, y, z): (usize, usize, usize)) -> Self {
        VolumeChunk { x, y, z }
    }
}
impl VolumeChunk {
    fn key(&self) -> ChunkKey {
        ChunkKey::new(0, self.x as u32, self.y as u32, self.z as u32)
    }
}

/// The LOD-0 chunks one tile samples, split by the cache they live in. With no
/// overlay configured `overlay` is empty.
#[derive(Default)]
struct TileChunks {
    base: BTreeSet<VolumeChunk>,
    overlay: BTreeSet<VolumeChunk>,
}

/// The surface geometry being rendered. Both variants hold a base-independent,
/// `Send + Sync` projection of the surface; `Rendering::world` wraps it around
/// an inner (base) volume per tile. They must stay `Send` because `Rendering`
/// is moved into `spawn_blocking` — so neither embeds a base `Volume` (whose
/// trait object is `!Send`); the base is supplied per tile instead.
#[derive(Clone)]
enum Segment {
    /// Parsed obj mesh; wrapped fresh per tile by `ObjVolume::new`.
    Obj(Arc<ObjFile>),
    /// tifxyz grid + baked transform, loaded once; the real base volume is
    /// supplied per tile via `TifXyzVolume::from_data` (cheap — it reuses the
    /// `Arc`'d, parsed projection). `dims` is the output UV resolution: `None`
    /// uses the natural `grid ÷ scale` nominal dims; `Some((w, h))` renders the
    /// unrolled segment at `w × h` (set by `--scale-uv-by-transform`).
    TifXyz {
        data: Arc<TifXyzData>,
        dims: Option<(usize, usize)>,
    },
}

#[derive(Clone)]
struct Rendering {
    params: RenderParams,
    segment: Segment,
    cache: ChunkCache,
    /// Set when `--overlay` is given: a second cache over the overlay/label
    /// volume, wrapped together with the base via `OverlayVolume` at render
    /// time so the `alpha-overlay*` modes can read its per-sample opacity.
    overlay_cache: Option<ChunkCache>,
}

impl Rendering {
    fn new(mut params: RenderParams, args: &Args) -> Result<Self> {
        // The overlay-aware modes read opacity from the overlay volume; without
        // one they would silently degrade to a plain `Alpha` walk (that's how
        // `OverlayVolume` falls back for non-overlay backends, but here there's
        // no `OverlayVolume` at all), so fail fast — before the expensive obj
        // load — instead.
        if args.overlay.is_none()
            && matches!(
                params.compositing.mode,
                CompositingMode::AlphaOverlay
                    | CompositingMode::AlphaOverlayStart
                    | CompositingMode::AlphaOverlayCombined
            )
        {
            return Err(anyhow!(
                "compositing mode {:?} requires an overlay; pass --overlay <ome-zarr url or path>",
                params.compositing.mode
            ));
        }

        let transform = resolve_transform(args)?;

        // Optionally magnify the source UV dims by the transform's linear scale
        // factor so we render at the target volume's resolution rather than the
        // source's (see --scale-uv-by-transform). Identical for obj and tifxyz:
        // both have a source-resolution UV extent (catalog --width/--height for
        // obj, grid ÷ scale for tifxyz) that the transform may magnify.
        let scale_uv = |w: usize, h: usize| -> (usize, usize) {
            if !args.scale_uv_by_transform {
                return (w, h);
            }
            let Some(s) = transform.as_ref().map(|t| t.scale_factor()) else {
                log::warn!("--scale-uv-by-transform set but no --transform given; UV dims left unchanged");
                return (w, h);
            };
            let sw = (w as f64 * s).round().max(1.0) as usize;
            let sh = (h as f64 * s).round().max(1.0) as usize;
            log::info!("--scale-uv-by-transform: {}x{} × {:.4} → {}x{}", w, h, s, sw, sh);
            (sw, sh)
        };

        let segment = match (args.tifxyz.as_ref(), args.obj.as_ref()) {
            (Some(tifxyz_dir), _) => {
                // Load + project the grid once (with a throwaway base); the real
                // base is supplied per tile in `world`. Tex dims come from the
                // TIFFs (the --width/--height args are ignored for tifxyz).
                let tpl = TifXyzVolume::load_from_directory(tifxyz_dir, EmptyVolume {}.into_volume(), &transform)
                    .map_err(|e| anyhow!("Failed to load tifxyz from {}: {:#}", tifxyz_dir, e))?;
                let (nominal_w, nominal_h) = (tpl.width(), tpl.height());

                let (w, h) = scale_uv(nominal_w, nominal_h);
                params.width = w;
                params.height = h;
                // Only override the grid sampling when we actually scaled; `None`
                // keeps the exact `meta.json` scale (avoids round-trip rounding).
                let dims = if (w, h) == (nominal_w, nominal_h) {
                    None
                } else {
                    Some((w, h))
                };
                Segment::TifXyz { data: tpl.data(), dims }
            }
            (None, Some(obj_path)) => {
                let (Some(w), Some(h)) = (args.width, args.height) else {
                    return Err(anyhow!("--width and --height are required when using --obj"));
                };
                let (w, h) = scale_uv(w as usize, h as usize);
                params.width = w;
                params.height = h;
                Segment::Obj(Arc::new(ObjVolume::load_obj(obj_path, &transform, params.projection)))
            }
            (None, None) => return Err(anyhow!("one of --obj or --tifxyz is required")),
        };

        let cache_root = args
            .data_directory
            .clone()
            .map(PathBuf::from)
            .unwrap_or_else(base_cache_dir);

        let volume_arg = args
            .volume
            .clone()
            .ok_or_else(|| anyhow!("--volume <ome-zarr url or local path> is required"))?;
        let cache = build_cache(&volume_arg, cache_root.clone())?;

        let overlay_cache = match args.overlay.as_ref() {
            Some(overlay_arg) => Some(build_cache(overlay_arg, cache_root)?),
            None => None,
        };

        Ok(Self {
            params,
            segment,
            cache,
            overlay_cache,
        })
    }

    /// Build the per-tile "world" volume: maps the UV tile to world xyz and
    /// samples `inner` (the base, optionally overlay-wrapped). Shared by the
    /// chunk-collection pass and the render pass so both sample identically.
    fn world(&self, inner: Volume) -> Volume {
        match &self.segment {
            Segment::Obj(obj) => {
                ObjVolume::new(obj.clone(), inner, self.params.width, self.params.height).into_volume()
            }
            // Cheap: reuses the Arc'd projection, only the base differs per tile.
            Segment::TifXyz { data, dims } => TifXyzVolume::from_data(data.clone(), inner, *dims).into_volume(),
        }
    }

    async fn run(&self, multi: &MultiProgress, plain_progress: Option<Duration>) -> Result<()> {
        std::fs::create_dir_all(&self.params.target_dir)?;

        let count_style = ProgressStyle::with_template(
            "{spinner} {msg:25} {bar:80.cyan/blue} [{elapsed_precise}] ({eta:>4}) {pos}/{len}",
        )
        .unwrap()
        .tick_chars("→↘↓↙←↖↑↗");

        let tiles = self.uv_tiles();
        let tiles_per_layer = tiles.len() as u64 / self.params.w_range.clone().count() as u64;

        let ensure_bar = ProgressBar::new(tiles.len() as u64)
            .with_style(count_style.clone().tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈"))
            .with_message("Ensuring chunks");
        multi.add(ensure_bar.clone());
        ensure_bar.tick();

        let render_bar = ProgressBar::new(tiles.len() as u64)
            .with_style(count_style.clone().tick_chars("▪▫▨▧▦▩"))
            .with_message("Rendering tiles");
        multi.add(render_bar.clone());
        render_bar.tick();

        let layers_bar = ProgressBar::new(self.params.w_range.clone().count() as u64)
            .with_style(count_style.tick_chars("⌷⌸⌹⌺"))
            .with_message("Saving layers");
        multi.add(layers_bar.clone());
        layers_bar.tick();

        // In non-interactive mode replace the (hidden) bars with a periodic
        // flushed log line. Hide the interactive draw target so a forced
        // --progress-interval on a TTY doesn't render both.
        let progress_logger = plain_progress.map(|interval| {
            multi.set_draw_target(indicatif::ProgressDrawTarget::hidden());
            let bars = vec![
                ("ensure", ensure_bar.clone()),
                ("render", render_bar.clone()),
                ("layers", layers_bar.clone()),
            ];
            print_plain_progress(&bars);
            spawn_plain_progress_logger(bars, interval)
        });

        // Two bounded stages. The ensure stage runs ahead of rendering by
        // `prefetch_depth` tiles so the next tiles' chunks download (in the
        // cache's pool) while the current ones render — but only that far
        // ahead, so the resident working set stays small enough that the
        // background purge doesn't evict a tile's chunks before we read them.
        let render_conc = self.params.render_concurrency.max(1);
        let ensure_conc = (render_conc + self.params.prefetch_depth).max(1);

        // Two bounded, ordered stages produce `(tile, Image)` in `uv_tiles`
        // order (w-major, then row, then column). Boxed so either output path
        // below can consume the same stream.
        let rendered = stream::iter(tiles)
            .map(|tile| {
                let self_clone = self.clone();
                let ensure_bar = ensure_bar.clone();
                async move {
                    // Collect is CPU work (a paint pass); the ensure wait is
                    // async so it doesn't tie up a blocking thread.
                    let chunks = {
                        let s = self_clone.clone();
                        tokio::task::spawn_blocking(move || s.chunks_for(&tile)).await.unwrap()
                    };
                    self_clone.ensure_resident(&chunks).await;
                    ensure_bar.inc(1);
                    tile
                }
            })
            .buffered(ensure_conc)
            .map(|tile| {
                let self_clone = self.clone();
                let render_bar = render_bar.clone();
                async move {
                    let tile_image = tokio::task::spawn_blocking(move || self_clone.render_tile(&tile))
                        .await
                        .unwrap();
                    render_bar.inc(1);
                    (tile, tile_image)
                }
            })
            .buffered(render_conc) // ordered so each layer's tiles arrive contiguously
            .boxed();

        // `.tif`/`.tiff` streams each tile straight to a tiled BigTIFF as it's
        // rendered (O(a few tiles) of memory); every other format still
        // assembles a full `GrayImage` per layer and encodes it in one shot.
        let is_tiled_tiff = matches!(
            self.params.target_format.to_ascii_lowercase().as_str(),
            "tif" | "tiff"
        );
        let layer_results: Vec<Result<()>> = if is_tiled_tiff {
            self.stream_layers_to_tiled_tiff(rendered, layers_bar.clone()).await
        } else {
            rendered
                .chunks(tiles_per_layer as usize)
                .map(|tiles| {
                    let self_clone = self.clone();
                    let layers_bar = layers_bar.clone();
                    async move {
                        let res = tokio::task::spawn_blocking(move || self_clone.render_layer_from_tiles(tiles))
                            .await
                            .unwrap();
                        layers_bar.inc(1);
                        res
                    }
                })
                .buffer_unordered(render_conc)
                .collect::<Vec<_>>()
                .await
        };


        // Stop the periodic logger and emit a final 100% line so the log ends
        // on a complete state rather than whatever the last tick caught.
        if let Some(handle) = progress_logger {
            handle.abort();
            print_plain_progress(&[
                ("ensure", ensure_bar.clone()),
                ("render", render_bar.clone()),
                ("layers", layers_bar.clone()),
            ]);
        }

        // Per-pool worker count × number of pools (base + optional overlay).
        let cache_workers =
            vesuvius_rs::cache::configured_workers() * (1 + self.overlay_cache.is_some() as usize);
        print_render_summary(ensure_bar.elapsed(), cache_workers);

        // Make sure the chunks we fetched are durably recorded for the next run.
        self.cache.flush();

        // Propagate the first per-layer failure. Without this, a layer whose
        // save/encode errored (e.g. a TIFF that overflowed classic 32-bit
        // offsets) was collected into `layer_results` and silently dropped, so
        // the render exited 0 and downstream copied a half-written file.
        layer_results.into_iter().collect::<Result<Vec<()>>>()?;

        Ok(())
    }

    fn uv_tiles(&self) -> Vec<UVTile> {
        let top = self.params.crop.as_ref().map(|c| c.top).unwrap_or(0);
        let left = self.params.crop.as_ref().map(|c| c.left).unwrap_or(0);
        let width = self.params.render_width();
        let height = self.params.render_height();

        let mut res = Vec::new();
        for w in self.params.w_range.clone() {
            for v in (top..top + height).step_by(self.params.tile_size) {
                for u in (left..left + width).step_by(self.params.tile_size) {
                    let tile = UVTile { u, v, w };
                    res.push(tile);
                }
            }
        }
        res
    }

    /// Paint the tile once with a chunk-collecting backend (no data fetched)
    /// to learn exactly which LOD-0 chunks the render will sample. When an
    /// overlay is configured the collectors are wrapped in an `OverlayVolume`
    /// exactly like the render pass, so the overlay's sampled chunks are
    /// collected too (in their own cache's coordinate space).
    fn chunks_for(&self, UVTile { u, v, w }: &UVTile) -> TileChunks {
        let tile_width = self.params.tile_size;
        let tile_height = self.params.tile_size;

        // Keep handles to the *same* collector instances that `paint` samples.
        // (`TileCollectingVolume`'s state is a `RefCell` that deep-clones, so
        // wrapping a clone in the `Volume` and reading the original back would
        // observe an empty set — the painted data lives in the clone.)
        let base_collector = Arc::new(TileCollectingVolume::new());
        let overlay_collector = self
            .overlay_cache
            .as_ref()
            .map(|_| Arc::new(TileCollectingVolume::new()));

        let inner: Volume = match overlay_collector.as_ref() {
            Some(overlay_collector) => OverlayVolume::new(
                Volume::from_ref(base_collector.clone()),
                Volume::from_ref(overlay_collector.clone()),
                OverlayColoring::default(),
            )
            .into_volume(),
            None => Volume::from_ref(base_collector.clone()),
        };

        let world = self.world(inner);

        let mut image = Image::new(tile_width, tile_height);
        let xyz = [
            *u as i32 + tile_width as i32 / 2,
            *v as i32 + tile_height as i32 / 2,
            *w as i32 - self.params.mid_layer as i32,
        ];
        let config = self.params.drawing_config();
        world.paint(xyz, 0, 1, 2, tile_width, tile_height, 1, 1, &config, &mut image);

        let base = base_collector
            .state
            .replace(Default::default())
            .requested_tiles
            .into_iter()
            .map(Into::into)
            .collect();
        let overlay = overlay_collector
            .map(|c| {
                c.state
                    .replace(Default::default())
                    .requested_tiles
                    .into_iter()
                    .map(Into::into)
                    .collect()
            })
            .unwrap_or_default();
        TileChunks { base, overlay }
    }

    /// Dispatch every chunk the tile needs into the unified cache and block
    /// until each one is resident (or definitively empty). Re-touching via
    /// `state_or_fetch` on each poll keeps the chunks in the current LRU epoch,
    /// which the purge planner refuses to evict — so a tile's own chunks won't
    /// be purged out from under it between this phase and the render.
    /// Dispatch every chunk the tile needs and wait until each one settles
    /// (Resident/Empty/CooldownMiss — i.e. no longer Pending). The wait is
    /// async (`tokio::time::sleep`) so it occupies no blocking thread; the
    /// render stage's blocking threads stay free for tiles whose chunks are
    /// already resident.
    async fn ensure_resident(&self, chunks: &TileChunks) {
        // One (cache, keys) group per backing store. With no overlay the base
        // group is the only one; the overlay group reads from its own cache in
        // the same chunk-coordinate space.
        let mut groups: Vec<(ChunkCache, Vec<ChunkKey>)> =
            vec![(self.cache.clone(), chunks.base.iter().map(|c| c.key()).collect())];
        if let Some(overlay_cache) = self.overlay_cache.as_ref() {
            groups.push((overlay_cache.clone(), chunks.overlay.iter().map(|c| c.key()).collect()));
        }

        // Kick off all fetches first so each cache's downloader pool works them
        // concurrently. The initial dispatch can plan/queue a lot of work, so
        // run it on a blocking thread rather than the async executor.
        {
            let groups = groups.clone();
            tokio::task::spawn_blocking(move || {
                for (cache, keys) in &groups {
                    for key in keys {
                        let _ = cache.state_or_fetch(*key);
                    }
                }
            })
            .await
            .unwrap();
        }

        let deadline = Instant::now() + TILE_ENSURE_TIMEOUT;
        loop {
            // Keep only the chunks still in flight. Resident/Empty/CooldownMiss
            // are all settled and won't change by waiting longer — in
            // particular CooldownMiss covers out-of-bounds samples (routine at
            // surface edges and along composite normals), so we stop polling
            // those and let the read return 0 there rather than spin to the
            // timeout.
            let mut any_pending = false;
            for (cache, keys) in groups.iter_mut() {
                keys.retain(|key| matches!(cache.state_or_fetch(*key).as_ref(), ChunkState::Pending { .. }));
                any_pending |= !keys.is_empty();
            }
            if !any_pending {
                break;
            }
            if Instant::now() >= deadline {
                let pending: usize = groups.iter().map(|(_, keys)| keys.len()).sum();
                log::warn!(
                    "ensure_resident: {} chunk(s) still pending after {:?}; rendering with partial data",
                    pending,
                    TILE_ENSURE_TIMEOUT,
                );
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// Consume the ordered `(tile, Image)` stream and write one tiled BigTIFF
    /// per w-layer, flushing each tile to disk as it arrives instead of
    /// buffering a whole layer + a full `GrayImage`. Layers arrive contiguously
    /// (the stream is ordered), so a single active writer thread suffices: when
    /// `w` changes we close the current file (letting it finalize) and open the
    /// next. Returns one `Result` per layer, in completion order.
    async fn stream_layers_to_tiled_tiff(
        &self,
        mut rendered: impl futures::Stream<Item = (UVTile, Image)> + Unpin,
        layers_bar: ProgressBar,
    ) -> Vec<Result<()>> {
        use crate::tiled_tiff::{write_tiled_gray_tiff, TileGray};

        let width = self.params.render_width();
        let height = self.params.render_height();
        let left = self.params.render_left();
        let top = self.params.render_top();
        let tile_size = self.params.tile_size;
        let target_dir = self.params.target_dir.clone();
        let target_format = self.params.target_format.clone();
        let metadata = self.params.metadata_json.clone();
        // Bound the hand-off so a slow disk backpressures rendering rather than
        // letting tiles pile up in the channel.
        let cap = (self.params.render_concurrency * 2).max(1);

        // Surface the writer's outcome, mapping a task panic/cancel to an error.
        fn writer_result(joined: std::result::Result<Result<()>, tokio::task::JoinError>) -> Result<()> {
            joined.unwrap_or_else(|e| Err(anyhow!("tiled TIFF writer task failed: {e}")))
        }

        let mut results: Vec<Result<()>> = Vec::new();
        // The active layer's writer task + the channel feeding it. The tiff
        // encoder is synchronous and self-referential, so it runs as one
        // long-lived `spawn_blocking` task that pulls tiles from the channel;
        // its `JoinHandle` is awaitable, so rotating layers never blocks the
        // reactor.
        type Active = (usize, tokio::sync::mpsc::Sender<TileGray>, tokio::task::JoinHandle<Result<()>>);
        let mut active: Option<Active> = None;

        while let Some((UVTile { u, v, w }, image)) = rendered.next().await {
            // Rotate to a fresh writer when the layer changes.
            if active.as_ref().map(|(aw, _, _)| *aw) != Some(w) {
                if let Some((_, tx, handle)) = active.take() {
                    drop(tx); // close channel → writer drains + finalizes
                    results.push(writer_result(handle.await));
                    layers_bar.inc(1);
                }
                let (tx, mut rx) = tokio::sync::mpsc::channel::<TileGray>(cap);
                let path = format!("{}/{:02}.{}", target_dir, w, target_format);
                let (pw, ph, pts, pmeta) = (width, height, tile_size, metadata.clone());
                let handle = tokio::task::spawn_blocking(move || {
                    let tiles = std::iter::from_fn(move || rx.blocking_recv());
                    write_tiled_gray_tiff(&path, pw, ph, pts, &pmeta, tiles)
                });
                active = Some((w, tx, handle));
            }

            // Color32 → Gray8: drops the 4 MiB tile for a 1 MiB buffer before
            // the hand-off. Tiles are painted full-size (edge overhang
            // included), which is exactly what the tiled layout stores.
            let data: Vec<u8> = image.data.iter().map(|c| c.r()).collect();
            let col = (u - left) / tile_size;
            let row = (v - top) / tile_size;
            // A send error only happens if the writer already died; its actual
            // error is surfaced by `writer_result` when we drain below.
            let _ = active.as_ref().unwrap().1.send(TileGray { col, row, data }).await;
        }

        if let Some((_, tx, handle)) = active.take() {
            drop(tx);
            results.push(writer_result(handle.await));
            layers_bar.inc(1);
        }
        results
    }

    fn render_layer_from_tiles(&self, tiles: Vec<(UVTile, Image)>) -> Result<()> {
        let width = self.params.render_width();
        let height = self.params.render_height();
        let left = self.params.render_left();
        let top = self.params.render_top();

        let tile_size = self.params.tile_size;
        let w = tiles[0].0.w;
        let mut image = image::GrayImage::new(width as u32, height as u32);

        // copy in all the tile images
        for (UVTile { u, v, w: _ }, tile_image) in tiles {
            let tile_data = tile_image.data;
            // blit into the right area of image
            for lu in 0..tile_size {
                for lv in 0..tile_size {
                    let gu = u + lu - left;
                    let gv = v + lv - top;
                    if gu < width && gv < height {
                        // edge tiles may spill over boundaries of target image
                        image.put_pixel(gu as u32, gv as u32, Luma([tile_data[lv * tile_size + lu].r() as u8]));
                    }
                }
            }
        }

        let path = format!("{}/{:02}.{}", self.params.target_dir, w, self.params.target_format);
        metadata::save_image_with_metadata(&image, &path, &self.params.target_format, &self.params.metadata_json)?;

        Ok(())
    }

    fn render_tile(&self, UVTile { u, v, w }: &UVTile) -> Image {
        let paint_width = self.params.tile_size;
        let paint_height = self.params.tile_size;

        // Fresh per-tile reader over the shared cache: UnifiedVolume holds a
        // thread-local hot slot (!Sync), so each render thread needs its own.
        let base = UnifiedVolume::new(self.cache.clone()).into_volume();
        // Wrap with the overlay when configured so the overlay-aware modes can
        // read its per-sample opacity (the value still comes from the base).
        // Mirrors the GUI's `OverlayVolume` wiring; the chunks_for pass wraps
        // the collectors the same way so the ensured chunks match.
        let vol = match self.overlay_cache.as_ref() {
            Some(overlay_cache) => {
                let overlay = UnifiedVolume::new(overlay_cache.clone()).into_volume();
                OverlayVolume::new(base, overlay, OverlayColoring::default()).into_volume()
            }
            None => base,
        };
        let world = self.world(vol);
        let config = self.params.drawing_config();

        let mut image = Image::new(paint_width, paint_height);
        world.paint(
            [
                *u as i32 + paint_width as i32 / 2,
                *v as i32 + paint_height as i32 / 2,
                *w as i32 - self.params.mid_layer as i32,
            ],
            0,
            1,
            2,
            paint_width,
            paint_height,
            1,
            1,
            &config,
            &mut image,
        );
        image
    }
}

fn resolve_transform(args: &Args) -> Result<Option<AffineTransform>> {
    let Some(t) = args.transform.as_ref() else {
        return Ok(None);
    };
    let mut transform = AffineTransform::from_json_array_or_path(t).map_err(|e| anyhow!("Invalid transform: {}", e))?;
    if args.invert_transform {
        transform = transform
            .invert()
            .map_err(|e| anyhow!("Transform is not invertible: {}", e))?;
    }
    Ok(Some(transform))
}

/// Build a `ChunkCache` backed by an ome-zarr volume, mirroring the GUI's
/// construction chain (see `NewVolumeReference::volume`) but keeping the cache
/// handle so the renderer can dispatch/poll chunks and flush at the end.
fn build_cache(volume_arg: &str, cache_root: PathBuf) -> Result<ChunkCache> {
    let vref = if volume_arg.starts_with("http://") || volume_arg.starts_with("https://") {
        NewVolumeReference::from_url(volume_arg)
    } else {
        NewVolumeReference::from_path(volume_arg)
    }
    .map_err(|e| anyhow!("Could not resolve volume '{}': {}", volume_arg, e))?;

    let (id, location) = match vref {
        NewVolumeReference::OmeZarr { id, location } => (id, location),
        other => {
            return Err(anyhow!(
                "vesuvius-render only supports ome-zarr volumes via the unified cache; '{}' resolved to a different format",
                other.id()
            ))
        }
    };

    // Open the unified cache through the shared constructor so the on-disk
    // volume key matches the GUI's exactly (same source + id → same directory),
    // rather than re-deriving it here and risking divergence. Eager
    // materialization is on: the renderer reads essentially the whole 256³
    // sub-chunk over a run, so decode it once and persist all ~64 child cache
    // chunks rather than re-decoding from the raw store per sibling.
    let cache = NewVolumeReference::open_ome_zarr_cache(&id, &location, cache_root, true);
    // The renderer blocks until each chunk is resident before reading, so the
    // upscaled-from-parent preview is never read — skip the per-chunk upsample.
    cache.set_preview_synthesis(false);
    // For the same reason, don't prefetch the coarse preview-LOD pyramid in
    // `touch_aabb`: those chunks are only ever consumed by the GUI's
    // progressive preview / upscale-from-parent, never by the renderer, so
    // fetching+decoding them is pure wasted work (and an accidental LOD climb).
    cache.set_preview_prefetch(false);
    // And don't let the per-voxel sample paths climb to coarser LODs when a
    // target-LOD chunk isn't resident: the ensure stage pre-fetches exactly
    // the chunks each tile reads, so a climb here would only fetch+decode
    // coarse chunks that are never rendered (the stray high-LOD work).
    cache.set_lod_climb(false);
    // Finally, trust that the ensure stage made every sampled chunk resident:
    // the per-voxel interpolation then reads straight off the target-LOD shard
    // mmap with no chunk-state probe / DashMap lookup (the dominant per-voxel
    // cost in the render profile), like the composite-along-normal fast path.
    cache.set_assume_resident(true);
    // Queued fetches are never cancelled here: the renderer paints without
    // leases, so everything it requests is unleased (durable).
    Ok(cache)
}

async fn monitor_runtime_stats(multi: &MultiProgress) {
    let count_style = ProgressStyle::with_template(
        "{spinner} {msg:25} {bar:80.cyan/blue} [{elapsed_precise}] ({eta:>4}) {pos}/{len}",
    )
    .unwrap();
    let bar = ProgressBar::new(0)
        .with_style(count_style)
        .with_message("CPU threads active");

    multi.add(bar.clone());
    tokio::spawn(async move {
        loop {
            let metrics = tokio::runtime::Handle::current().metrics();

            bar.set_length(metrics.num_blocking_threads() as u64);
            bar.set_position((metrics.num_blocking_threads() - metrics.num_idle_blocking_threads()) as u64);

            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
    });
}

#[derive(Clone)]
struct TileCollectingVolumeState {
    requested_tiles: BTreeSet<(usize, usize, usize)>,
    last_requested: (usize, usize, usize),
}
impl Default for TileCollectingVolumeState {
    fn default() -> Self {
        Self {
            requested_tiles: BTreeSet::new(),
            last_requested: (0, 0, 0),
        }
    }
}

/// A VoxelVolume implementation that just collects needed tiles
#[derive(Clone)]
struct TileCollectingVolume {
    state: RefCell<TileCollectingVolumeState>,
}
impl TileCollectingVolume {
    fn new() -> Self {
        Self {
            state: TileCollectingVolumeState::default().into(),
        }
    }
    fn add_tile(&self, tile: (usize, usize, usize)) {
        let mut state = self.state.borrow_mut();
        if state.last_requested != tile {
            state.last_requested = tile;
            state.requested_tiles.insert(tile);
        }
    }
}
impl PaintVolume for TileCollectingVolume {
    fn paint(
        &self,
        _xyz: [i32; 3],
        _u_coord: usize,
        _v_coord: usize,
        _plane_coord: usize,
        _width: usize,
        _height: usize,
        _sfactor: u8,
        _paint_zoom: u8,
        _config: &DrawingConfig,
        _buffer: &mut Image,
    ) {
        panic!();
    }
    fn shared(&self) -> VolumeCons {
        panic!();
    }
}
impl VoxelVolume for TileCollectingVolume {
    fn get(&self, xyz: [f64; 3], _downsampling: i32) -> u8 {
        let tile: (usize, usize, usize) = ((xyz[0] as usize) >> 6, (xyz[1] as usize) >> 6, (xyz[2] as usize) >> 6);

        self.add_tile(tile);
        0
    }
    fn get_interpolated(&self, xyz: [f64; 3], downsampling: i32) -> u8 {
        let x = xyz[0] as usize;
        let y = xyz[1] as usize;
        let z = xyz[2] as usize;

        if x & 63 == 63 || y & 63 == 63 || z & 63 == 63 {
            // slow path, call default trilinear interpolation
            self.get_interpolated_slow(xyz, downsampling);
        } else {
            self.add_tile(((xyz[0] as usize) >> 6, (xyz[1] as usize) >> 6, (xyz[2] as usize) >> 6));
        }
        0
    }
}
