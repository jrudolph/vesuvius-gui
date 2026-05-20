//! `UnifiedVolume` — single `VoxelVolume` + `PaintVolume` impl over a
//! `ChunkCache`. The only place every backfiller-backed volume needs.
//!
//! ## LOD fallback strategy
//!
//! `paint()` and `get()` both walk the LOD pyramid: if the requested
//! ("target") LOD chunk isn't resident yet, we sample from the first
//! coarser-LOD parent chunk that is. The target tile's screen region is
//! painted **once** at whichever LOD has data — we never overdraw multiple
//! LODs into the same region. Pre-dispatch in `paint()` happens coarse-first
//! so the wider, viewport-covering parent chunks land in the LIFO queue
//! before their finer-LOD children pile on top: a low-res preview shows
//! up promptly while detail streams in.
//!
//! `get()` is the per-voxel sampler used by surface (ObjVolume) and PPM
//! renderers that don't go through `UnifiedVolume::paint`. It can't rely on
//! a prior pre-dispatch pass, so on a target-LOD miss it dispatches every
//! coarser LOD along the same column and picks the first resident one. The
//! per-volume hot slot caches that chosen `(lod, chunk_state)` keyed by the
//! target-LOD chunk so subsequent pixels in the same target chunk skip the
//! pyramid walk entirely.
//!
//! ## Coordinate conventions
//!
//! * `paint()` takes `xyz: [i32; 3]` in **world voxel coords** (LOD 0). The
//!   per-LOD chunk lookup divides by `64 * (1 << lod)` to land on the right
//!   chunk and the rendering loop knows how to step samples per pixel.
//! * `get()` follows the `VoxelVolume::get` convention used everywhere else
//!   in the codebase (see `VolumeGrid64x4Mapped`, `ZarrContext`, and
//!   `PPMVolume`/`ObjVolume` callers): `xyz` is in **voxel coords at the
//!   requested downsampling** — the caller has already divided world coords
//!   by `downsampling`. This is the convention surface painting depends on,
//!   so do NOT redivide here.

use super::cache::ChunkCache;
use super::state::{ChunkKey, ChunkState};
use crate::volume::{DrawingConfig, Image, PaintVolume, VolumeCons, VoxelPaintVolume, VoxelVolume};
use ecolor::Color32;
use std::cell::RefCell;
use std::sync::Arc;

/// Overlay alpha for `debug_chunk_overlay`. Low enough that voxel detail is
/// still readable underneath.
const OVERLAY_ALPHA: f32 = 0.35;

pub struct UnifiedVolume {
    cache: ChunkCache,
    // Per-volume hot slot for the last target chunk touched by `get`. The
    // cached `chosen` may be a coarser-LOD parent when the target chunk
    // wasn't resident on first hit. `paint()` and external callers clear it
    // between frames via `reset_for_painting`. `Volume` clones produce fresh
    // `UnifiedVolume` instances with their own slot via the `shared()`
    // constructor, so the `RefCell` is never shared across threads.
    local: RefCell<LocalSlot>,
}

#[derive(Default)]
struct LocalSlot {
    target_key: Option<ChunkKey>,
    chosen: Option<(u8, Arc<ChunkState>)>,
}

impl UnifiedVolume {
    pub fn new(cache: ChunkCache) -> Self {
        Self {
            cache,
            local: RefCell::new(LocalSlot::default()),
        }
    }

    pub fn cache(&self) -> &ChunkCache {
        &self.cache
    }

    fn drop_hot_slot(&self) {
        let mut b = self.local.borrow_mut();
        b.target_key = None;
        b.chosen = None;
    }
}

fn lod_for(sfactor: u8) -> u8 {
    // sfactor is expected to be a power of two: 1, 2, 4, 8, …
    (sfactor as u32).max(1).trailing_zeros() as u8
}

impl VoxelVolume for UnifiedVolume {
    fn reset_for_painting(&self) {
        self.drop_hot_slot();
    }

    fn get(&self, xyz: [f64; 3], downsampling: i32) -> u8 {
        // Per the `VoxelVolume::get` convention shared with VolumeGrid64x4Mapped,
        // ZarrContext, and PPMVolume callers (notably `ObjVolume::paint`, which
        // passes `[x / sfactor, y / sfactor, z / sfactor]`): `xyz` is in
        // voxel-coords at the requested downsampling. We do NOT re-divide by
        // scale here. To consult coarser LODs we shift right; to consult finer
        // LODs (only when target_lod exceeds the volume's `max_lod`) we shift
        // left.
        let target_lod = lod_for(downsampling.max(1) as u8);
        let max_lod = self.cache.max_lod();

        let target_sx = (xyz[0] as i64).max(0) as u64;
        let target_sy = (xyz[1] as i64).max(0) as u64;
        let target_sz = (xyz[2] as i64).max(0) as u64;
        let key_t = ChunkKey::new(
            target_lod,
            (target_sx / 64) as u32,
            (target_sy / 64) as u32,
            (target_sz / 64) as u32,
        );

        // Hot slot keyed by the target chunk: if we've already resolved this
        // target chunk (either to itself or to a coarser parent, or as
        // Empty) this frame, skip the pyramid walk and just sample.
        {
            let b = self.local.borrow();
            if b.target_key == Some(key_t) {
                if let Some((lod_use, s)) = &b.chosen {
                    if let Some(mmap) = s.as_resident() {
                        return sample_at(target_sx, target_sy, target_sz, target_lod, *lod_use, mmap);
                    }
                    if s.is_empty() {
                        return 0;
                    }
                }
            }
        }

        // Walk target → coarsest (or just `max_lod` when the requested target
        // is coarser than anything the volume has), dispatching each. Surface
        // and PPM renderers reach `get()` without going through
        // `UnifiedVolume::paint`, so we can't assume a prior pre-dispatch
        // primed the coarser LODs — kick the fetches here. Stop at the
        // first *terminal* state (Resident or Empty): `Empty` at a fine LOD
        // overrides whatever a coarser parent might have, because the
        // fine-grained data structure is what tells us "no data here".
        let walk_lo = target_lod.min(max_lod);
        let walk_hi = max_lod;
        let mut chosen: Option<(u8, Arc<ChunkState>)> = None;
        for lod_try in walk_lo..=walk_hi {
            let (lx, ly, lz) = coord_at_lod(target_sx, target_sy, target_sz, target_lod, lod_try);
            let key = ChunkKey::new(lod_try, (lx / 64) as u32, (ly / 64) as u32, (lz / 64) as u32);
            let s = self.cache.state_or_fetch(key);
            if s.is_terminal() {
                chosen = Some((lod_try, s));
                break;
            }
        }

        let Some((lod_use, state)) = chosen else {
            // Nothing terminal yet. Don't poison the hot slot — a later
            // pixel in this frame might land after the dispatch completes.
            return 0;
        };
        let value = match state.as_resident() {
            Some(mmap) => sample_at(target_sx, target_sy, target_sz, target_lod, lod_use, mmap),
            None => 0, // Empty
        };

        let mut b = self.local.borrow_mut();
        b.target_key = Some(key_t);
        b.chosen = Some((lod_use, state));
        value
    }
}

/// Map a target-LOD voxel coord to its corresponding voxel coord at `lod_use`.
/// Shifts right when going coarser, left when going finer.
fn coord_at_lod(sx: u64, sy: u64, sz: u64, from_lod: u8, to_lod: u8) -> (u64, u64, u64) {
    if to_lod >= from_lod {
        let shift = to_lod - from_lod;
        (sx >> shift, sy >> shift, sz >> shift)
    } else {
        let shift = from_lod - to_lod;
        (sx << shift, sy << shift, sz << shift)
    }
}

/// Map the cache state of one target-LOD chunk (plus which LOD actually
/// rendered and whether an HTTP GET is in flight) to an overlay tint.
/// `None` means "ready at target LOD; no overlay needed".
///
/// Color legend:
/// - Green   — Empty (definitively absent, cached as a sentinel)
/// - Blue    — served from coarser LOD, waiting in the work queue
/// - Cyan    — served from coarser LOD, actively downloading right now
/// - Yellow  — Pending, no coarser fallback yet, waiting in queue
/// - Orange  — Pending, no coarser fallback yet, actively downloading
/// - Red     — CooldownMiss (recent fetch failed)
/// - Magenta — Missing / never dispatched
fn overlay_color_for(
    target_lod: u8,
    chosen_lod: Option<u8>,
    target_state: Option<&ChunkState>,
    is_downloading: bool,
) -> Option<Color32> {
    // Empty target wins over any LOD fallback — the chunk is definitively
    // absent. Use a vivid green so it doesn't blend with mid-gray voxel data
    // the way the previous neutral tint did.
    if matches!(target_state, Some(ChunkState::Empty)) {
        return Some(Color32::from_rgb(60, 200, 110)); // green
    }
    match (chosen_lod, target_state) {
        // Rendered at target LOD with real data — happy path, no tint.
        (Some(l), _) if l == target_lod => None,
        // Rendered via a coarser parent (target not ready). Cyan if the
        // target's bytes are actually moving over the wire right now, blue
        // if it's just sitting in the queue waiting for a worker.
        (Some(_), _) => {
            if is_downloading {
                Some(Color32::from_rgb(40, 200, 220)) // cyan
            } else {
                Some(Color32::from_rgb(60, 130, 230)) // blue
            }
        }
        // Nothing resident yet — inspect target state for finer signal.
        (None, Some(ChunkState::Pending)) => {
            if is_downloading {
                Some(Color32::from_rgb(230, 140, 40)) // orange
            } else {
                Some(Color32::from_rgb(230, 200, 40)) // yellow
            }
        }
        (None, Some(ChunkState::CooldownMiss { .. })) => Some(Color32::from_rgb(220, 60, 60)), // red
        (None, Some(ChunkState::Missing)) | (None, None) => Some(Color32::from_rgb(220, 60, 220)), // magenta
        // Defensive: Resident / Empty here mean the LOD-walk produced no
        // chosen — shouldn't happen, but if it does, no overlay.
        (None, Some(ChunkState::Resident { .. } | ChunkState::Empty)) => None,
    }
}

fn sample_at(sx: u64, sy: u64, sz: u64, from_lod: u8, to_lod: u8, mmap: &[u8]) -> u8 {
    let (lx, ly, lz) = coord_at_lod(sx, sy, sz, from_lod, to_lod);
    let off = ((lz & 63) as usize) * 64 * 64 + ((ly & 63) as usize) * 64 + ((lx & 63) as usize);
    mmap[off]
}

/// Geometry describing the chunk grid at one LOD inside one viewport.
struct ViewportTiles {
    chunk_world: i32,
    tile_u_lo: i32,
    tile_u_hi: i32,
    tile_v_lo: i32,
    tile_v_hi: i32,
    tile_pc: i32,
}

fn viewport_tiles(min_uc: i32, max_uc: i32, min_vc: i32, max_vc: i32, pc: i32, lod: u8) -> Option<ViewportTiles> {
    let scale = 1i32 << lod;
    let chunk_world = 64 * scale;
    let tile_pc = pc.div_euclid(chunk_world);
    if tile_pc < 0 {
        return None;
    }
    Some(ViewportTiles {
        chunk_world,
        tile_u_lo: min_uc.div_euclid(chunk_world).max(0),
        tile_u_hi: max_uc.div_euclid(chunk_world),
        tile_v_lo: min_vc.div_euclid(chunk_world).max(0),
        tile_v_hi: max_vc.div_euclid(chunk_world),
        tile_pc,
    })
}

impl PaintVolume for UnifiedVolume {
    fn paint(
        &self,
        xyz: [i32; 3],
        u_coord: usize,
        v_coord: usize,
        plane_coord: usize,
        canvas_width: usize,
        canvas_height: usize,
        sfactor: u8,
        paint_zoom: u8,
        config: &DrawingConfig,
        buffer: &mut Image,
    ) {
        // The hot slot is sized for one chunk and doesn't help the per-tile
        // walk; drop it so we don't accidentally serve stale data after a
        // pan.
        self.drop_hot_slot();

        let target_lod = lod_for(sfactor);
        let max_lod = self.cache.max_lod();

        let pzoom = paint_zoom as i32;
        let width_world = pzoom * canvas_width as i32;
        let height_world = pzoom * canvas_height as i32;
        let min_uc = xyz[u_coord] - width_world / 2;
        let max_uc = xyz[u_coord] + width_world / 2;
        let min_vc = xyz[v_coord] - height_world / 2;
        let max_vc = xyz[v_coord] + height_world / 2;
        let pc = xyz[plane_coord];
        if pc < 0 {
            return;
        }

        // -------- Pass 1: dispatch coarse → fine --------
        // Walking coarse → fine means the first submissions of a cold
        // viewport are the low-LOD preview chunks. The cache + downloader
        // queues are pure LIFO, so the most recently submitted (finest)
        // work pops first — but coarse-first submission still gets the
        // low-LOD chunks into flight before the worker pool can drain
        // them, so a quick preview shows up promptly while detail
        // streams in behind it.
        for lod in (target_lod..=max_lod).rev() {
            let Some(tiles) = viewport_tiles(min_uc, max_uc, min_vc, max_vc, pc, lod) else {
                continue;
            };
            for tu in tiles.tile_u_lo..=tiles.tile_u_hi {
                for tv in tiles.tile_v_lo..=tiles.tile_v_hi {
                    let mut chunk = [0i32; 3];
                    chunk[u_coord] = tu;
                    chunk[v_coord] = tv;
                    chunk[plane_coord] = tiles.tile_pc;
                    let key = ChunkKey::new(lod, chunk[0] as u32, chunk[1] as u32, chunk[2] as u32);
                    let _ = self.cache.state_or_fetch(key);
                }
            }
        }

        // -------- Pass 2: render per target tile, picking best resident LOD --------
        let Some(t) = viewport_tiles(min_uc, max_uc, min_vc, max_vc, pc, target_lod) else {
            return;
        };
        let ceil_div = |x: i32, d: i32| (x + d - 1).div_euclid(d);

        for tu in t.tile_u_lo..=t.tile_u_hi {
            for tv in t.tile_v_lo..=t.tile_v_hi {
                // Screen rect this target tile owns.
                let chunk_u_lo_t = tu * t.chunk_world;
                let chunk_u_hi_t = chunk_u_lo_t + t.chunk_world;
                let chunk_v_lo_t = tv * t.chunk_world;
                let chunk_v_hi_t = chunk_v_lo_t + t.chunk_world;
                let u_px_lo = ceil_div(chunk_u_lo_t - min_uc, pzoom).max(0);
                let u_px_hi = ceil_div(chunk_u_hi_t - min_uc, pzoom).min(canvas_width as i32);
                let v_px_lo = ceil_div(chunk_v_lo_t - min_vc, pzoom).max(0);
                let v_px_hi = ceil_div(chunk_v_hi_t - min_vc, pzoom).min(canvas_height as i32);
                if u_px_lo >= u_px_hi || v_px_lo >= v_px_hi {
                    continue;
                }

                // Walk target → coarsest, stopping at the first *terminal*
                // state (Resident or Empty). An `Empty` at a finer LOD wins
                // over coarser data: the fine-grained structure is what tells
                // us "no data here", so we shouldn't paint coarser-LOD values
                // through it.
                let mut chosen: Option<(u8, Arc<ChunkState>)> = None;
                for lod_try in target_lod..=max_lod {
                    let shift = lod_try - target_lod;
                    let mut chunk = [0u32; 3];
                    chunk[u_coord] = (tu as u32) >> shift;
                    chunk[v_coord] = (tv as u32) >> shift;
                    chunk[plane_coord] = (t.tile_pc as u32) >> shift;
                    let key = ChunkKey::new(lod_try, chunk[0], chunk[1], chunk[2]);
                    let s = if lod_try == target_lod {
                        self.cache.state_or_fetch(key)
                    } else {
                        match self.cache.peek(key) {
                            Some(s) => s,
                            None => continue,
                        }
                    };
                    if s.is_terminal() {
                        chosen = Some((lod_try, s));
                        break;
                    }
                }

                let painted = if let Some((lod_use, state)) = chosen.as_ref() {
                    if let Some(mmap) = state.as_resident() {
                        let scale_use = 1i32 << *lod_use;
                        let chunk_world_use = 64 * scale_use;
                        let shift = *lod_use - target_lod;
                        let parent_tu = tu >> shift;
                        let parent_tv = tv >> shift;
                        let parent_tpc = t.tile_pc >> shift;
                        let chunk_u_lo = parent_tu * chunk_world_use;
                        let chunk_v_lo = parent_tv * chunk_world_use;
                        let chunk_pc_lo = parent_tpc * chunk_world_use;
                        let plane_sample = ((pc - chunk_pc_lo) / scale_use) as usize;

                        for v_px in v_px_lo..v_px_hi {
                            let world_v = min_vc + v_px * pzoom;
                            let sample_v = ((world_v - chunk_v_lo) / scale_use) as usize;
                            for u_px in u_px_lo..u_px_hi {
                                let world_u = min_uc + u_px * pzoom;
                                let sample_u = ((world_u - chunk_u_lo) / scale_use) as usize;
                                let mut s = [0usize; 3];
                                s[u_coord] = sample_u;
                                s[v_coord] = sample_v;
                                s[plane_coord] = plane_sample;
                                let off = s[2] * 64 * 64 + s[1] * 64 + s[0];
                                let value = config.filter(mmap[off]);
                                buffer.set_gray(u_px as usize, v_px as usize, value);
                            }
                        }
                        true
                    } else {
                        // Empty at this LOD — fill the rect with the filtered
                        // zero value so the user sees a clean "no data" cell
                        // instead of whatever the buffer happened to contain.
                        let zero = config.filter(0);
                        for v_px in v_px_lo..v_px_hi {
                            for u_px in u_px_lo..u_px_hi {
                                buffer.set_gray(u_px as usize, v_px as usize, zero);
                            }
                        }
                        true
                    }
                } else {
                    false
                };

                if !painted && !config.debug_chunk_overlay {
                    // Nothing terminal yet, no overlay requested — leave the
                    // rect alone (caller cleared the buffer).
                    continue;
                }

                if config.debug_chunk_overlay {
                    let mut chunk = [0u32; 3];
                    chunk[u_coord] = tu as u32;
                    chunk[v_coord] = tv as u32;
                    chunk[plane_coord] = t.tile_pc as u32;
                    let target_key = ChunkKey::new(target_lod, chunk[0], chunk[1], chunk[2]);
                    let target_state = self.cache.peek(target_key);
                    let chosen_lod = chosen.as_ref().map(|(l, _)| *l);
                    let is_downloading = self.cache.is_downloading(target_key);
                    let color =
                        overlay_color_for(target_lod, chosen_lod, target_state.as_deref(), is_downloading);
                    if let Some(c) = color {
                        for v_px in v_px_lo..v_px_hi {
                            for u_px in u_px_lo..u_px_hi {
                                buffer.blend(u_px as usize, v_px as usize, c, OVERLAY_ALPHA);
                            }
                        }
                    }
                }
            }
        }
    }

    fn shared(&self) -> VolumeCons {
        let cache = self.cache.clone();
        Box::new(move || VoxelPaintVolume::into_volume(UnifiedVolume::new(cache)))
    }
}
