use crate::gui::app::{ZOOM_MAX, ZOOM_MIN};
use egui::cache::FramePublisher;
use egui::{Color32, ColorImage, PointerButton, Response, Ui, Vec2};
use fxhash::FxBuildHasher;
use std::cell::Cell;
use std::hash::BuildHasher;
use std::ops::RangeInclusive;
use std::pin::Pin;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;
use vesuvius_rs::volume::{DrawingConfig, PaintVolume, SurfaceVolume, Volume, VoxelVolume};

const ZOOM_RES_FACTOR: f32 = 1.;
const TILE_SIZE: usize = 256;

// Time-to-live for cached tiles in milliseconds
const TILE_TTL_BASE_MS: u64 = 500;
// Random jitter range (0 to this many milliseconds added to base TTL)
const TILE_TTL_JITTER_MS: u64 = 500;
// TTL multiplier for each successive unchanged recalculation (exponential backoff)
const TILE_TTL_BACKOFF_MULTIPLIER: u64 = 2;
// Maximum TTL backoff factor
const TILE_TTL_MAX_BACKOFF: u64 = 16;

// Threshold below which we stop polling for the rest of the frame.
const MIN_POLL_DURATION: Duration = Duration::from_micros(500);
// Share of the frame budget reserved for the UV/surface pane when it is visible.
pub const UV_PANE_BUDGET_FRACTION: f32 = 0.75;

/// Tracks per-frame polling budget for `VolumePane` tile loads.
///
/// One instance is constructed at the start of each egui frame and shared across
/// every pane drawn in that frame. Each tile poll is capped at half of the
/// pane's remaining time, and once the global remaining time drops below
/// `MIN_POLL_DURATION` the `poll_disabled` latch trips and all further polling
/// is skipped for this frame.
pub struct FrameBudget {
    frame_deadline: quanta::Instant,
    pane_deadline: Cell<quanta::Instant>,
    poll_disabled: Cell<bool>,
    tiles: Cell<TileFrameStats>,
}

/// What each visible tile slot showed this frame, summed over every pane
/// drawn with the same `FrameBudget`. Telemetry only (read by the pane
/// harness); the GUI ignores it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TileFrameStats {
    /// Tile slots in the visible set (including the 1-tile margin).
    pub visible: u32,
    /// Showing a finished render (incl. one whose TTL refresh just started).
    pub ready: u32,
    /// Showing the previous render while a TTL refresh is in flight.
    pub recalculating: u32,
    /// Still loading, covered by a cross-mip placeholder.
    pub loading_fallback: u32,
    /// Still loading, nothing to draw.
    pub loading_blank: u32,
    /// Loading tiles not polled because the frame budget ran out.
    pub budget_skipped: u32,
}

#[derive(Clone, Copy)]
enum TileOutcome {
    Ready,
    Recalculating,
    Loading { has_fallback: bool, budget_skipped: bool },
}

impl FrameBudget {
    pub fn new(target: Duration) -> Self {
        let frame_deadline = quanta::Instant::now() + target;
        Self {
            frame_deadline,
            pane_deadline: Cell::new(frame_deadline),
            poll_disabled: Cell::new(false),
            tiles: Cell::new(TileFrameStats::default()),
        }
    }

    /// Tile-state counters accumulated so far this frame.
    pub fn tile_stats(&self) -> TileFrameStats {
        self.tiles.get()
    }

    fn record_tile(&self, outcome: TileOutcome) {
        let mut t = self.tiles.get();
        t.visible += 1;
        match outcome {
            TileOutcome::Ready => t.ready += 1,
            TileOutcome::Recalculating => t.recalculating += 1,
            TileOutcome::Loading {
                has_fallback,
                budget_skipped,
            } => {
                if has_fallback {
                    t.loading_fallback += 1;
                } else {
                    t.loading_blank += 1;
                }
                if budget_skipped {
                    t.budget_skipped += 1;
                }
            }
        }
        self.tiles.set(t);
    }

    /// Allocate this pane's share of the remaining frame budget. Capped by the
    /// global frame deadline.
    pub fn begin_pane(&self, allotted: Duration) {
        let now = quanta::Instant::now();
        let pane = now + allotted;
        let cap = if pane < self.frame_deadline {
            pane
        } else {
            self.frame_deadline
        };
        self.pane_deadline.set(cap);
    }

    /// Returns the timeout to use for a single tile poll, or `None` if the
    /// remaining time is too small. Once `None` is returned because the global
    /// deadline is past, `poll_disabled` latches so subsequent calls short-circuit.
    pub fn next_poll_timeout(&self) -> Option<Duration> {
        if self.poll_disabled.get() {
            return None;
        }
        let now = quanta::Instant::now();
        if now >= self.frame_deadline {
            self.poll_disabled.set(true);
            return None;
        }
        let remaining_pane = self.pane_deadline.get().saturating_duration_since(now);
        if remaining_pane < MIN_POLL_DURATION {
            return None;
        }
        let remaining_global = self.frame_deadline.saturating_duration_since(now);
        let timeout = (remaining_pane / 2).min(remaining_global);
        if timeout < MIN_POLL_DURATION {
            return None;
        }
        Some(timeout)
    }

    /// Cheap check for short non-blocking polls (e.g. the 100µs recalc peek):
    /// allow only while we are still within the global frame deadline.
    pub fn polling_allowed(&self) -> bool {
        if self.poll_disabled.get() {
            return false;
        }
        if quanta::Instant::now() >= self.frame_deadline {
            self.poll_disabled.set(true);
            return false;
        }
        true
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TileCacheKey {
    pane_type: PaneType,
    tile_u: i32,
    tile_v: i32,
    w: i32,
    min_level: u32,
    paint_zoom: u8,
    drawing_config: DrawingConfig,
    segment_outlines_coord: Option<[i32; 3]>,
    extra_resolutions: u32,
    volume_id: usize,
    overlay_volume_id: usize,
}

impl TileCacheKey {
    fn new(
        pane_type: PaneType,
        tile_u: i32,
        tile_v: i32,
        w: i32,
        zoom: f32,
        paint_zoom: u8,
        drawing_config: &DrawingConfig,
        segment_outlines_coord: Option<[i32; 3]>,
        extra_resolutions: u32,
        world: &Volume,
        overlay: Option<&Volume>,
    ) -> Self {
        let volume_id = Arc::as_ptr(&world.volume) as *const () as usize;
        let overlay_volume_id = overlay
            .map(|o| Arc::as_ptr(&o.volume) as *const () as usize)
            .unwrap_or(0);

        let natural_level = 32 - ((ZOOM_RES_FACTOR / zoom) as u32).leading_zeros();
        let min_level = (natural_level as i32 - drawing_config.lod_bias).max(0) as u32;

        Self {
            pane_type,
            tile_u,
            tile_v,
            w,
            min_level,
            paint_zoom,
            drawing_config: drawing_config.clone(),
            segment_outlines_coord,
            extra_resolutions,
            volume_id,
            overlay_volume_id,
        }
    }
}

struct CancellableImageFuture {
    future: Pin<Box<dyn futures::Future<Output = Arc<ColorImage>> + Send + Sync>>,
    is_cancelled: Arc<AtomicBool>,
}
impl Drop for CancellableImageFuture {
    fn drop(&mut self) {
        self.is_cancelled.store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

#[derive(Clone)]
enum AsyncTexture {
    Loading {
        future: Arc<Mutex<CancellableImageFuture>>,
        started_at: quanta::Instant,
    },
    Ready {
        texture: egui::TextureHandle,
        cached_at: quanta::Instant,
        content_hash: u64,   // Hash of tile pixel data
        backoff_factor: u64, // TTL multiplier for unchanged tiles (1, 2, 4, 8, 16)
    },
    ReadyRecalculating {
        texture: egui::TextureHandle,
        future: Arc<Mutex<CancellableImageFuture>>,
        cached_at: quanta::Instant,
        content_hash: u64,   // Hash of current tile data
        backoff_factor: u64, // Current backoff factor to preserve
    },
}

impl AsyncTexture {
    /// Check if this tile needs recalculation based on TTL with backoff
    fn needs_recalculation(&self) -> bool {
        match self {
            AsyncTexture::Ready {
                cached_at,
                backoff_factor,
                ..
            }
            | AsyncTexture::ReadyRecalculating {
                cached_at,
                backoff_factor,
                ..
            } => {
                use rand::Rng;
                let jitter = rand::thread_rng().gen_range(0..=TILE_TTL_JITTER_MS);
                let base_ttl = TILE_TTL_BASE_MS + jitter;
                let ttl = std::time::Duration::from_millis(base_ttl * backoff_factor);
                cached_at.elapsed() > ttl
            }
            AsyncTexture::Loading { .. } => false,
        }
    }
}

fn to_color_image(image: vesuvius_rs::volume::Image) -> ColorImage {
    ColorImage {
        size: [image.width, image.height],
        pixels: image.data,
        ..Default::default()
    }
}

/// Calculate a simple hash of image pixel data
fn hash_image(image: &ColorImage) -> u64 {
    use std::hash::{Hash, Hasher};

    let mut hasher = FxBuildHasher::default().build_hasher();
    image.size.hash(&mut hasher);
    image.pixels.hash(&mut hasher);
    hasher.finish()
}

/// Poll a future with timeout, shared logic for Loading and ReadyRecalculating states
fn poll_tile_future(future: Arc<Mutex<CancellableImageFuture>>, timeout: std::time::Duration) -> Poll<Arc<ColorImage>> {
    let mut future_guard = future.lock().unwrap();
    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    let start = quanta::Instant::now();

    loop {
        let poll_result = tokio::task::block_in_place(|| future_guard.future.as_mut().poll(&mut context));

        match poll_result {
            Poll::Ready(image) => return Poll::Ready(image),
            Poll::Pending => {
                if start.elapsed() >= timeout {
                    return Poll::Pending;
                }
                std::thread::yield_now();
            }
        }
    }
}

type TileCache = FramePublisher<TileCacheKey, AsyncTexture>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PaneType {
    XY, // u=0, v=1, d=2
    XZ, // u=0, v=2, d=1
    YZ, // u=2, v=1, d=0
    UV, // u=0, v=1, d=2 (for segment mode)
    UW, // u=0, v=2, d=1 (segment cross-section: u horizontal, w vertical, v fixed)
    VW, // u=2, v=1, d=0 (segment cross-section: w horizontal, v vertical, u fixed)
}

impl PaneType {
    pub fn coordinates(&self) -> (usize, usize, usize) {
        match self {
            PaneType::XY => (0, 1, 2),
            PaneType::XZ => (0, 2, 1),
            PaneType::YZ => (2, 1, 0),
            PaneType::UV => (0, 1, 2),
            PaneType::UW => (0, 2, 1),
            PaneType::VW => (2, 1, 0),
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            PaneType::XY => "XY",
            PaneType::XZ => "XZ",
            PaneType::YZ => "YZ",
            PaneType::UV => "UV",
            PaneType::UW => "UW",
            PaneType::VW => "VW",
        }
    }
}

pub struct VolumePane {
    pane_type: PaneType,
    is_segment_pane: bool,
}

const MAX_DOWNSAMPLE: u8 = 32;

/// Snap a continuous zoom factor to a power-of-two `paint_zoom` (1, 2, 4, …, MAX_DOWNSAMPLE).
/// Anything in `[1.0, ∞)` paints at native resolution; `[0.5, 1.0)` → 2, `[0.25, 0.5)` → 4, etc.
/// Power-of-two stepping keeps adjacent-zoom tile grids aligned so cross-mip lookups are trivial.
fn paint_zoom_for(zoom: f32) -> u8 {
    if zoom >= 1.0 {
        return 1;
    }
    let downsample = ((1.0 / zoom).ceil() as u32).clamp(1, MAX_DOWNSAMPLE as u32);
    downsample.next_power_of_two().min(MAX_DOWNSAMPLE as u32) as u8
}

fn full_uv() -> egui::Rect {
    egui::Rect::from_min_size(egui::Pos2::ZERO, egui::Vec2::splat(1.0))
}

/// A single textured draw contributing to a tile slot.
/// - `tile_sub_rect`: where within the target tile (in `[0,1]^2`) this draw lands.
///   `full_uv()` means "covers the whole tile" (real tile or zoom-in upscale).
///   A quadrant like `(0..0.5, 0..0.5)` means "covers the top-left quarter" (zoom-out child).
/// - `uv`: which part of the texture to sample (in `[0,1]^2`). `full_uv()` for whole texture.
type TileDraw = (egui::TextureHandle, egui::Rect, egui::Rect);

impl VolumePane {
    pub const fn new(pane_type: PaneType, is_segment_pane: bool) -> Self {
        Self {
            pane_type,
            is_segment_pane,
        }
    }

    fn calculate_visible_tiles(
        &self,
        coord: [i32; 3],
        zoom: f32,
        frame_width: usize,
        frame_height: usize,
    ) -> Vec<(i32, i32, egui::Rect)> {
        let (u_coord, v_coord, _) = self.pane_type.coordinates();

        let paint_zoom = paint_zoom_for(zoom);

        // When paint_zoom > 1, the effective tile size in world coordinates is larger
        let effective_tile_size = TILE_SIZE as f32 * paint_zoom as f32;

        // Calculate world space viewport dimensions
        let world_width = frame_width as f32 / zoom;
        let world_height = frame_height as f32 / zoom;

        // Calculate viewport bounds in world coordinates
        let viewport_left = coord[u_coord] as f32 - world_width / 2.0;
        let viewport_right = coord[u_coord] as f32 + world_width / 2.0;
        let viewport_top = coord[v_coord] as f32 - world_height / 2.0;
        let viewport_bottom = coord[v_coord] as f32 + world_height / 2.0;

        #[cfg(debug_assertions)]
        {
            println!("Pane {:?}: u_coord={}, v_coord={}", self.pane_type, u_coord, v_coord);
            println!("  coord=[{},{},{}]", coord[0], coord[1], coord[2]);
            println!(
                "  viewport: left={:.1}, right={:.1}, top={:.1}, bottom={:.1}",
                viewport_left, viewport_right, viewport_top, viewport_bottom
            );
            println!("  effective_tile_size={:.1}", effective_tile_size);
        }

        // Calculate tile range using effective tile size
        let start_tile_x = (viewport_left / effective_tile_size).floor() as i32;
        let end_tile_x = (viewport_right / effective_tile_size).ceil() as i32;
        let start_tile_y = (viewport_top / effective_tile_size).floor() as i32;
        let end_tile_y = (viewport_bottom / effective_tile_size).ceil() as i32;

        // Generate tile list with screen positions
        let mut tiles = Vec::new();
        for tile_y in start_tile_y - 1..end_tile_y + 1 {
            for tile_x in start_tile_x - 1..end_tile_x + 1 {
                let screen_rect =
                    self.calculate_tile_screen_rect(tile_x, tile_y, coord, zoom, frame_width, frame_height);
                tiles.push((tile_x, tile_y, screen_rect));
            }
        }

        tiles
    }

    fn calculate_tile_screen_rect(
        &self,
        tile_x: i32,
        tile_y: i32,
        coord: [i32; 3],
        zoom: f32,
        frame_width: usize,
        frame_height: usize,
    ) -> egui::Rect {
        let (u_coord, v_coord, _) = self.pane_type.coordinates();

        let paint_zoom = paint_zoom_for(zoom);

        // When paint_zoom > 1, the effective tile size in world coordinates is larger
        let effective_tile_size = TILE_SIZE as f32 * paint_zoom as f32;

        // Tile bounds in world coordinates using effective tile size
        let tile_world_left = tile_x as f32 * effective_tile_size;
        let tile_world_right = (tile_x + 1) as f32 * effective_tile_size;
        let tile_world_top = tile_y as f32 * effective_tile_size;
        let tile_world_bottom = (tile_y + 1) as f32 * effective_tile_size;

        // Convert to screen coordinates relative to the pane's viewport center
        // The painter uses coordinates relative to the allocated UI area (0,0 to frame_width,frame_height)
        let center_x = frame_width as f32 / 2.0;
        let center_y = frame_height as f32 / 2.0;

        // Calculate screen position relative to current view center
        let screen_left = center_x + (tile_world_left - coord[u_coord] as f32) * zoom;
        let screen_right = center_x + (tile_world_right - coord[u_coord] as f32) * zoom;
        let screen_top = center_y + (tile_world_top - coord[v_coord] as f32) * zoom;
        let screen_bottom = center_y + (tile_world_bottom - coord[v_coord] as f32) * zoom;

        // Ensure coordinates are within reasonable bounds for the pane
        egui::Rect::from_min_max(
            egui::pos2(screen_left, screen_top),
            egui::pos2(screen_right, screen_bottom),
        )
    }

    pub fn render(
        &self,
        ui: &mut Ui,
        coord: &mut [i32; 3],
        world: &Volume,
        overlay: Option<&Volume>,
        surface_volume: Option<Arc<dyn SurfaceVolume>>,
        zoom: &mut f32,
        drawing_config: &DrawingConfig,
        extra_resolutions: u32,
        segment_outlines_coord: Option<[i32; 3]>,
        ranges: &[RangeInclusive<i32>; 3],
        cell_size: Vec2,
        budget: &FrameBudget,
        pane_share: Duration,
    ) -> bool {
        let frame_width = cell_size.x as usize;
        let frame_height = cell_size.y as usize;

        budget.begin_pane(pane_share);

        // Get or create tiles
        let tiles = self.get_or_create_tiles(
            ui,
            *coord,
            world,
            overlay,
            *zoom,
            frame_width,
            frame_height,
            drawing_config,
            extra_resolutions,
            segment_outlines_coord,
            budget,
        );

        // Allocate space for this pane using the proper egui pattern
        let (response, painter) = ui.allocate_painter(cell_size, egui::Sense::drag());

        // Paint all tiles on the allocated space - tiles should use response.rect coordinate system
        for (texture, tile_rect, uv) in tiles {
            // Adjust tile_rect to be relative to response.rect
            let adjusted_rect =
                egui::Rect::from_min_size(response.rect.min + tile_rect.min.to_vec2(), tile_rect.size());

            painter.image(texture.id(), adjusted_rect, uv, egui::Color32::WHITE);
        }

        // Add segment outlines if configured
        if let (Some(surface_vol), Some(outlines_coord)) = (surface_volume, segment_outlines_coord) {
            // paint segment outline on a new texture that is not cached or tiled
            let (u_coord, v_coord, d_coord) = self.pane_type.coordinates();
            let paint_zoom = paint_zoom_for(*zoom);

            if !self.is_segment_pane && drawing_config.show_segment_outlines {
                let scaling = *zoom * paint_zoom as f32;
                let width = (frame_width as f32 / scaling) as usize;
                let height = (frame_height as f32 / scaling) as usize;

                let mut image = vesuvius_rs::volume::Image::new_from_color(width, height, Color32::TRANSPARENT);
                surface_vol.paint_plane_intersection(
                    *coord,
                    u_coord,
                    v_coord,
                    d_coord,
                    width,
                    height,
                    1,
                    paint_zoom,
                    Some(outlines_coord),
                    drawing_config,
                    &mut image,
                );
                let image: egui::ColorImage = to_color_image(image);
                let texture = ui.ctx().load_texture(self.pane_type.label(), image, Default::default());
                // Adjust rect to be relative to response.rect
                let adjusted_rect = egui::Rect::from_min_size(response.rect.min, response.rect.size());
                // Paint the segment outline texture
                painter.image(
                    texture.id(),
                    adjusted_rect,
                    egui::Rect::from_min_size(egui::Pos2::ZERO, egui::Vec2::splat(1.0)),
                    egui::Color32::WHITE,
                );
            }
        }

        // Handle interactions and return whether textures need clearing
        let mut interaction_happened = false;

        if self.handle_scroll(&response, ui, coord, ranges, zoom) {
            interaction_happened = true;
        }

        if self.handle_drag(&response, coord, ranges, *zoom) {
            interaction_happened = true;
        }

        interaction_happened
    }

    pub fn handle_scroll(
        &self,
        response: &Response,
        ui: &Ui,
        coord: &mut [i32; 3],
        ranges: &[RangeInclusive<i32>; 3],
        zoom: &mut f32,
    ) -> bool {
        let (_, _, d_coord) = self.pane_type.coordinates();
        let mut changed = false;

        if response.hovered() {
            let delta = ui.input(|i| i.raw_scroll_delta);
            let zoom_delta = ui.input(|i| i.zoom_delta());

            if zoom_delta != 1.0 {
                *zoom = (*zoom * zoom_delta).max(ZOOM_MIN).min(ZOOM_MAX);
                changed = true;
            } else if delta.y != 0.0 {
                let delta = delta.y.signum() * 1.0;
                let m = &mut coord[d_coord];
                *m = (*m + delta as i32).clamp(*ranges[d_coord].start(), *ranges[d_coord].end());
                changed = true;
            }
        }

        changed
    }

    pub fn handle_drag(
        &self,
        response: &Response,
        coord: &mut [i32; 3],
        ranges: &[RangeInclusive<i32>; 3],
        zoom: f32,
    ) -> bool {
        let (u_coord, v_coord, _) = self.pane_type.coordinates();
        let mut changed = false;

        if response.dragged_by(PointerButton::Primary) {
            let delta = -response.drag_delta() / zoom;
            coord[u_coord] = (coord[u_coord] + delta.x as i32).clamp(*ranges[u_coord].start(), *ranges[u_coord].end());
            coord[v_coord] = (coord[v_coord] + delta.y as i32).clamp(*ranges[v_coord].start(), *ranges[v_coord].end());
            changed = true;
        }

        changed
    }

    fn get_or_create_tiles(
        &self,
        ui: &Ui,
        coord: [i32; 3],
        world: &Volume,
        overlay: Option<&Volume>,
        zoom: f32,
        frame_width: usize,
        frame_height: usize,
        drawing_config: &DrawingConfig,
        extra_resolutions: u32,
        segment_outlines_coord: Option<[i32; 3]>,
        budget: &FrameBudget,
    ) -> Vec<(egui::TextureHandle, egui::Rect, egui::Rect)> {
        let visible_tiles = self.calculate_visible_tiles(coord, zoom, frame_width, frame_height);
        let paint_zoom = paint_zoom_for(zoom);

        let keys_and_rects = visible_tiles
            .iter()
            .map(|(tile_x, tile_y, tile_rect)| {
                let key = TileCacheKey::new(
                    self.pane_type,
                    *tile_x,
                    *tile_y,
                    coord[self.pane_type.coordinates().2],
                    zoom,
                    paint_zoom,
                    drawing_config,
                    segment_outlines_coord,
                    extra_resolutions,
                    world,
                    overlay,
                );
                (key, *tile_rect)
            })
            .collect::<Vec<_>>();

        for (key, _) in keys_and_rects.iter() {
            self.ensure_tile_async(ui, key.clone(), world, overlay);
        }

        let mut ready_tiles = Vec::new();
        for (key, tile_rect) in keys_and_rects {
            for (texture, sub_rect, uv) in self.get_or_create_tile_async(ui, key, world, overlay, budget) {
                // sub_rect is in [0,1]^2 tile-local space; map it into the on-screen tile_rect.
                let screen_rect = egui::Rect::from_min_size(
                    tile_rect.min
                        + egui::vec2(sub_rect.min.x * tile_rect.width(), sub_rect.min.y * tile_rect.height()),
                    egui::vec2(sub_rect.width() * tile_rect.width(), sub_rect.height() * tile_rect.height()),
                );
                ready_tiles.push((texture, screen_rect, uv));
            }
        }
        ready_tiles
    }
    fn ensure_tile_async(&self, ui: &Ui, key: TileCacheKey, world: &Volume, overlay: Option<&Volume>) {
        // Check if tile exists in cache
        let cached_value = ui.memory_mut(|mem| {
            let cache: &mut TileCache = mem.caches.cache::<TileCache>();
            cache.get(&key).cloned()
        });
        fn set(ui: &Ui, key: TileCacheKey, value: AsyncTexture) {
            ui.memory_mut(|mem| {
                let cache: &mut TileCache = mem.caches.cache::<TileCache>();
                cache.set(key, value);
            });
        }

        match cached_value {
            None => {
                let handle = self.create_tile_async(&key, world, overlay);

                set(
                    ui,
                    key,
                    AsyncTexture::Loading {
                        future: handle,
                        started_at: quanta::Instant::now(),
                    },
                );
            }
            _ => {}
        }
    }

    fn get_or_create_tile_async(
        &self,
        ui: &Ui,
        key: TileCacheKey,
        world: &Volume,
        overlay: Option<&Volume>,
        budget: &FrameBudget,
    ) -> Vec<TileDraw> {
        // Calculate paint_zoom for cache key (same logic as in create_tile)

        // Check if tile exists in cache
        let cached_value = ui.memory_mut(|mem| {
            let cache: &mut TileCache = mem.caches.cache::<TileCache>();

            // Clone the cached value to avoid borrow conflicts
            cache.get(&key).cloned()
        });
        fn set(ui: &Ui, key: TileCacheKey, value: AsyncTexture) {
            ui.memory_mut(|mem| {
                let cache: &mut TileCache = mem.caches.cache::<TileCache>();
                cache.set(key, value);
            });
        }

        match cached_value {
            Some(AsyncTexture::Ready {
                texture,
                cached_at,
                content_hash,
                backoff_factor,
            }) => {
                // Check if tile needs recalculation
                let async_tex = AsyncTexture::Ready {
                    texture: texture.clone(),
                    cached_at,
                    content_hash,
                    backoff_factor,
                };

                if async_tex.needs_recalculation() {
                    // TTL expired - start recalculation while showing old tile
                    let new_future = self.create_tile_async(&key, world, overlay);
                    set(
                        ui,
                        key.clone(),
                        AsyncTexture::ReadyRecalculating {
                            texture: texture.clone(),
                            future: new_future,
                            cached_at,
                            content_hash,
                            backoff_factor,
                        },
                    );
                } else {
                    // Refresh cache entry to keep it alive
                    set(ui, key, async_tex);
                }
                budget.record_tile(TileOutcome::Ready);
                vec![(texture, full_uv(), full_uv())]
            }

            Some(AsyncTexture::ReadyRecalculating {
                texture,
                future,
                cached_at,
                content_hash,
                backoff_factor,
            }) => {
                // Skip the recalc peek entirely if the frame deadline is gone.
                if !budget.polling_allowed() {
                    set(
                        ui,
                        key,
                        AsyncTexture::ReadyRecalculating {
                            texture: texture.clone(),
                            future,
                            cached_at,
                            content_hash,
                            backoff_factor,
                        },
                    );
                    ui.ctx().request_repaint();
                    budget.record_tile(TileOutcome::Recalculating);
                    return vec![(texture, full_uv(), full_uv())];
                }
                // Poll the recalculation future briefly (non-blocking check)
                // Use minimal timeout since we don't want to block UI
                match poll_tile_future(future.clone(), Duration::from_micros(100)) {
                    Poll::Ready(new_image) => {
                        // Calculate hash of new image to check if content changed
                        let new_hash = hash_image(new_image.as_ref());

                        // If content unchanged, increase backoff; otherwise reset
                        let new_backoff = if new_hash == content_hash {
                            (backoff_factor * TILE_TTL_BACKOFF_MULTIPLIER).min(TILE_TTL_MAX_BACKOFF)
                        } else {
                            1 // Reset backoff when content changes
                        };

                        // Recalculation complete - swap to new texture
                        let new_texture = ui.ctx().load_texture(
                            format!(
                                "{}_{}_{}_{}_{}",
                                self.pane_type.label(),
                                key.tile_u,
                                key.tile_v,
                                self.pane_type.coordinates().2,
                                key.volume_id
                            ),
                            new_image.as_ref().clone(),
                            Default::default(),
                        );
                        set(
                            ui,
                            key,
                            AsyncTexture::Ready {
                                texture: new_texture.clone(),
                                cached_at: quanta::Instant::now(),
                                content_hash: new_hash,
                                backoff_factor: new_backoff,
                            },
                        );
                        budget.record_tile(TileOutcome::Ready);
                        vec![(new_texture, full_uv(), full_uv())]
                    }
                    Poll::Pending => {
                        // Still recalculating - keep showing old texture
                        set(
                            ui,
                            key,
                            AsyncTexture::ReadyRecalculating {
                                texture: texture.clone(),
                                future: future.clone(),
                                cached_at,
                                content_hash,
                                backoff_factor,
                            },
                        );
                        ui.ctx().request_repaint(); // Check again next frame
                        budget.record_tile(TileOutcome::Recalculating);
                        vec![(texture, full_uv(), full_uv())]
                    }
                }
            }

            Some(AsyncTexture::Loading { future, started_at }) => {
                // Pull this tile's poll timeout from the frame budget. If the
                // budget has run out for this frame, skip polling entirely and
                // try again next frame.
                let Some(timeout) = budget.next_poll_timeout() else {
                    set(ui, key.clone(), AsyncTexture::Loading { future, started_at });
                    ui.ctx().request_repaint();
                    log::info!(
                        "cross-mip: trigger=loading_no_budget pane={:?} tile=({},{}) paint_zoom={}",
                        self.pane_type, key.tile_u, key.tile_v, key.paint_zoom
                    );
                    let draws = self.try_cross_mip_fallback(ui, &key);
                    budget.record_tile(TileOutcome::Loading {
                        has_fallback: !draws.is_empty(),
                        budget_skipped: true,
                    });
                    return draws;
                };
                match poll_tile_future(future.clone(), timeout) {
                    Poll::Ready(image) => {
                        let content_hash = hash_image(image.as_ref());
                        let texture = ui.ctx().load_texture(
                            format!(
                                "{}_{}_{}_{}_{}",
                                self.pane_type.label(),
                                key.tile_u,
                                key.tile_v,
                                self.pane_type.coordinates().2,
                                key.volume_id
                            ),
                            image.as_ref().clone(),
                            Default::default(),
                        );
                        set(
                            ui,
                            key,
                            AsyncTexture::Ready {
                                texture: texture.clone(),
                                cached_at: quanta::Instant::now(),
                                content_hash,
                                backoff_factor: 1, // Initial backoff
                            },
                        );
                        budget.record_tile(TileOutcome::Ready);
                        return vec![(texture, full_uv(), full_uv())];
                    }
                    Poll::Pending => {
                        // Deadline exceeded, still loading
                        set(
                            ui,
                            key.clone(),
                            AsyncTexture::Loading {
                                future: future.clone(),
                                started_at,
                            },
                        );
                        ui.ctx().request_repaint();
                        log::info!(
                            "cross-mip: trigger=loading_pending pane={:?} tile=({},{}) paint_zoom={}",
                            self.pane_type, key.tile_u, key.tile_v, key.paint_zoom
                        );
                        let draws = self.try_cross_mip_fallback(ui, &key);
                        budget.record_tile(TileOutcome::Loading {
                            has_fallback: !draws.is_empty(),
                            budget_skipped: false,
                        });
                        draws
                    }
                }
            }

            None => {
                // Start async rendering
                let handle = self.create_tile_async(&key, world, overlay);

                set(
                    ui,
                    key.clone(),
                    AsyncTexture::Loading {
                        future: handle,
                        started_at: quanta::Instant::now(),
                    },
                );
                ui.ctx().request_repaint();
                log::info!(
                    "cross-mip: trigger=fresh pane={:?} tile=({},{}) paint_zoom={}",
                    self.pane_type, key.tile_u, key.tile_v, key.paint_zoom
                );
                let draws = self.try_cross_mip_fallback(ui, &key);
                budget.record_tile(TileOutcome::Loading {
                    has_fallback: !draws.is_empty(),
                    budget_skipped: false,
                });
                draws
            }
        }
    }

    /// Try to find cached tiles at a different `paint_zoom` covering the same world region as
    /// `key`, for use as a placeholder while the real tile is still loading.
    ///
    /// Strategy: zoom-in (use a coarser cached parent, blurry but complete) is preferred because
    /// one cached tile fills the entire missing slot. If no zoom-in fallback exists, fall back to
    /// zoom-out (composite from finer cached children — sharp where covered but may have gaps).
    fn try_cross_mip_fallback(&self, ui: &Ui, key: &TileCacheKey) -> Vec<TileDraw> {
        if let Some(draw) = self.try_cross_mip_zoom_in(ui, key) {
            return vec![draw];
        }
        self.try_cross_mip_zoom_out(ui, key)
    }

    /// Probe coarser `paint_zoom` keys for a single tile that covers the target's region.
    /// Returns one `TileDraw` with the whole tile area mapped to a UV sub-rect of the parent.
    fn try_cross_mip_zoom_in(&self, ui: &Ui, key: &TileCacheKey) -> Option<TileDraw> {
        let target_zoom = key.paint_zoom;
        let mut level: u32 = 1;
        loop {
            let probe_zoom = (target_zoom as u32) << level;
            if probe_zoom > MAX_DOWNSAMPLE as u32 {
                log::info!(
                    "cross-mip: zoom-in MISS pane={:?} tile=({},{}) paint_zoom={} min_level={}",
                    self.pane_type, key.tile_u, key.tile_v, key.paint_zoom, key.min_level
                );
                return None;
            }
            let parent_u = key.tile_u >> level;
            let parent_v = key.tile_v >> level;
            if let Some((texture, hit_min_level)) = self.probe_cache(ui, key, parent_u, parent_v, probe_zoom as u8) {
                let mask = (1i32 << level) - 1;
                let inv = 1.0 / (1u32 << level) as f32;
                let local_u = (key.tile_u & mask) as f32 * inv;
                let local_v = (key.tile_v & mask) as f32 * inv;
                let uv = egui::Rect::from_min_size(egui::pos2(local_u, local_v), egui::vec2(inv, inv));
                log::info!(
                    "cross-mip: zoom-in HIT pane={:?} tile=({},{}) paint_zoom={} -> parent=({},{}) paint_zoom={} min_level={} level={} uv={:?}",
                    self.pane_type, key.tile_u, key.tile_v, key.paint_zoom,
                    parent_u, parent_v, probe_zoom, hit_min_level, level, uv
                );
                return Some((texture, full_uv(), uv));
            }
            level += 1;
        }
    }

    /// Probe finer `paint_zoom` keys for tiles whose union covers the target's region.
    /// Returns up to `(2^L)^2` draws (one per available child), each painting into a sub-rect of
    /// the target tile. Gaps where no child is cached are simply not drawn.
    ///
    /// Walks `level = 1, 2, …` and returns the first level with at least one cached child.
    fn try_cross_mip_zoom_out(&self, ui: &Ui, key: &TileCacheKey) -> Vec<TileDraw> {
        let target_zoom = key.paint_zoom as u32;
        // Bound how deep we composite — at level 3 we'd be probing 64 child slots per tile.
        const MAX_ZOOM_OUT_LEVEL: u32 = 3;
        for level in 1u32..=MAX_ZOOM_OUT_LEVEL {
            let child_zoom = target_zoom >> level;
            if child_zoom == 0 {
                break;
            }
            let span = 1i32 << level;
            let inv = 1.0 / span as f32;
            let mut draws = Vec::new();
            for cy in 0..span {
                for cx in 0..span {
                    let child_u = (key.tile_u << level) + cx;
                    let child_v = (key.tile_v << level) + cy;
                    if let Some((texture, _)) = self.probe_cache(ui, key, child_u, child_v, child_zoom as u8) {
                        let sub_rect = egui::Rect::from_min_size(
                            egui::pos2(cx as f32 * inv, cy as f32 * inv),
                            egui::vec2(inv, inv),
                        );
                        draws.push((texture, sub_rect, full_uv()));
                    }
                }
            }
            if !draws.is_empty() {
                log::info!(
                    "cross-mip: zoom-out HIT pane={:?} tile=({},{}) paint_zoom={} -> {}/{} children at paint_zoom={} level={}",
                    self.pane_type, key.tile_u, key.tile_v, key.paint_zoom,
                    draws.len(), span * span, child_zoom, level
                );
                return draws;
            }
        }
        log::info!(
            "cross-mip: zoom-out MISS pane={:?} tile=({},{}) paint_zoom={} min_level={}",
            self.pane_type, key.tile_u, key.tile_v, key.paint_zoom, key.min_level
        );
        Vec::new()
    }

    /// Look up a tile at `(probe_u, probe_v, probe_zoom)` in the cache, trying both the natural
    /// `min_level` for that `paint_zoom` and the +1 boundary case. On hit, re-publishes the entry
    /// so it survives `FramePublisher`'s per-frame eviction. Returns the texture and which
    /// `min_level` matched.
    fn probe_cache(
        &self,
        ui: &Ui,
        key: &TileCacheKey,
        probe_u: i32,
        probe_v: i32,
        probe_zoom: u8,
    ) -> Option<(egui::TextureHandle, u32)> {
        // paint_zoom is always pow2; log2 == trailing_zeros. sfactor = 1 << min_level matches
        // paint_zoom at the natural min_level (shifted by the configured lod_bias). +1 covers
        // the off-by-one at zoom boundaries.
        let natural_min_level = (probe_zoom as u32).trailing_zeros();
        let biased_min_level = (natural_min_level as i32 - key.drawing_config.lod_bias).max(0) as u32;
        for candidate_min_level in [biased_min_level, biased_min_level + 1] {
            let probe_key = TileCacheKey {
                tile_u: probe_u,
                tile_v: probe_v,
                paint_zoom: probe_zoom,
                min_level: candidate_min_level,
                ..key.clone()
            };
            let outcome = ui.memory_mut(|mem| -> Option<egui::TextureHandle> {
                let cache: &mut TileCache = mem.caches.cache::<TileCache>();
                let entry = cache.get(&probe_key)?.clone();
                let tex = match &entry {
                    AsyncTexture::Ready { texture, .. }
                    | AsyncTexture::ReadyRecalculating { texture, .. } => texture.clone(),
                    AsyncTexture::Loading { .. } => return None,
                };
                cache.set(probe_key.clone(), entry);
                Some(tex)
            });
            if let Some(texture) = outcome {
                return Some((texture, candidate_min_level));
            }
        }
        None
    }

    fn create_tile_async(
        &self,
        key: &TileCacheKey,
        world: &Volume,
        overlay: Option<&Volume>,
    ) -> Arc<Mutex<CancellableImageFuture>> {
        let pane_type = self.pane_type;
        let is_segment_pane = self.is_segment_pane;
        let key_clone = key.clone();
        let shared = world.shared();
        let overlay_shared = overlay.map(|o| o.shared());
        let is_cancelled = Arc::new(AtomicBool::new(false));
        let is_cancelled_clone = is_cancelled.clone();

        let handle = tokio::task::spawn_blocking(move || {
            if is_cancelled_clone.load(std::sync::atomic::Ordering::SeqCst) {
                return Arc::new(egui::ColorImage::example());
            }

            let volume_pane = VolumePane::new(pane_type, is_segment_pane);
            let overlay = overlay_shared.map(|c| c());
            let image = volume_pane.create_tile_sync(&key_clone, shared(), overlay);
            Arc::new(image)
        });

        let key = key.clone();

        // Map the JoinError to a default error image and box the future
        let future: Pin<Box<dyn futures::Future<Output = Arc<ColorImage>> + Send + Sync>> = Box::pin(async move {
            match handle.await {
                Ok(image) => image,
                Err(_join_error) => {
                    println!("Error loading tile ({}, {}): task failed", key.tile_u, key.tile_v);
                    // Return a simple error image
                    Arc::new(egui::ColorImage::example())
                }
            }
        });

        Arc::new(Mutex::new(CancellableImageFuture {
            future: future,
            is_cancelled,
        }))
    }

    fn create_tile_sync(&self, key: &TileCacheKey, world: Volume, overlay: Option<Volume>) -> egui::ColorImage {
        let (u_coord, v_coord, d_coord) = self.pane_type.coordinates();

        // Use integer paint zoom levels like the original code
        let paint_zoom = key.paint_zoom;

        // Always use fixed tile size - let paint_zoom handle the scaling
        let tile_width = TILE_SIZE;
        let tile_height = TILE_SIZE;
        let mut image = vesuvius_rs::volume::Image::new(tile_width, tile_height);

        // Calculate world coordinates for this tile
        // When paint_zoom > 1, each tile covers a larger world area
        let effective_tile_size = TILE_SIZE as f32 * paint_zoom as f32;

        // tile_x corresponds to u_coord, tile_y corresponds to v_coord
        let tile_world_u = key.tile_u as f32 * effective_tile_size;
        let tile_world_v = key.tile_v as f32 * effective_tile_size;

        // Set tile center in world coordinates for this pane's coordinate system
        let mut tile_coord = [0, 0, 0];
        tile_coord[u_coord] = (tile_world_u + effective_tile_size / 2.0) as i32;
        tile_coord[v_coord] = (tile_world_v + effective_tile_size / 2.0) as i32;
        tile_coord[d_coord] = key.w;

        let min_level = key.min_level;
        let max_level: u32 = min_level + key.extra_resolutions;

        for level in (min_level..=max_level).rev() {
            let sfactor = 1 << level as u8;
            world.reset_for_painting();
            world.paint(
                tile_coord,
                u_coord,
                v_coord,
                d_coord,
                tile_width,
                tile_height,
                sfactor,
                paint_zoom,
                &key.drawing_config,
                &mut image,
            );

            // Orthographic panes apply the overlay as a second paint pass on top
            // of the base. The segment/UV pane handles overlay inside-out via
            // OverlayVolume wrapped around the obj base, so skip here.
            if !self.is_segment_pane {
                if let Some(overlay) = overlay.as_ref() {
                    overlay.reset_for_painting();
                    overlay.paint(
                        tile_coord,
                        u_coord,
                        v_coord,
                        d_coord,
                        tile_width,
                        tile_height,
                        sfactor,
                        paint_zoom,
                        &key.drawing_config,
                        &mut image,
                    );
                }
            }
        }

        to_color_image(image)
    }
}
