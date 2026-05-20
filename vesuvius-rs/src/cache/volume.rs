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
//! so the bounded task queue gives priority to the wider, viewport-covering
//! parent chunks: a low-res preview shows up promptly while detail streams in.
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
use super::priority::{LodView, Viewport};
use super::state::{ChunkKey, ChunkState};
use crate::volume::{DrawingConfig, Image, PaintVolume, VolumeCons, VoxelPaintVolume, VoxelVolume};
use std::cell::RefCell;
use std::sync::Arc;

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
        // target chunk (either to itself or to a coarser parent) this frame,
        // skip the pyramid walk and just sample.
        {
            let b = self.local.borrow();
            if b.target_key == Some(key_t) {
                if let Some((lod_use, s)) = &b.chosen {
                    if let Some(mmap) = s.as_resident() {
                        return sample_at(target_sx, target_sy, target_sz, target_lod, *lod_use, mmap);
                    }
                }
            }
        }

        // Walk target → coarsest (or just `max_lod` when the requested target
        // is coarser than anything the volume has), dispatching each. Surface
        // and PPM renderers reach `get()` without going through
        // `UnifiedVolume::paint`, so we can't assume a prior pre-dispatch
        // primed the coarser LODs — kick the fetches here.
        let walk_lo = target_lod.min(max_lod);
        let walk_hi = max_lod;
        let mut chosen: Option<(u8, Arc<ChunkState>)> = None;
        for lod_try in walk_lo..=walk_hi {
            let (lx, ly, lz) = coord_at_lod(target_sx, target_sy, target_sz, target_lod, lod_try);
            let key = ChunkKey::new(lod_try, (lx / 64) as u32, (ly / 64) as u32, (lz / 64) as u32);
            let s = self.cache.state_or_fetch(key);
            if s.as_resident().is_some() {
                chosen = Some((lod_try, s));
                break;
            }
        }

        let Some((lod_use, state)) = chosen else {
            // Nothing resident yet. Don't poison the hot slot — a later
            // pixel in this frame might land after the dispatch completes.
            return 0;
        };
        let mmap = state.as_resident().expect("just checked resident");
        let value = sample_at(target_sx, target_sy, target_sz, target_lod, lod_use, mmap);

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

fn sample_at(sx: u64, sy: u64, sz: u64, from_lod: u8, to_lod: u8, mmap: &[u8]) -> u8 {
    let (lx, ly, lz) = coord_at_lod(sx, sy, sz, from_lod, to_lod);
    let off = ((lz & 63) as usize) * 64 * 64 + ((ly & 63) as usize) * 64 + ((lx & 63) as usize);
    mmap[off]
}

/// Build the paint-loop's view of the chunk grid across all LODs it intends
/// to dispatch this frame. Used to prioritize + prune work in the cache and
/// downloader: chunks inside the rect win; chunks more than a few chunks
/// outside get pruned; LODs not present here get pruned too.
#[allow(clippy::too_many_arguments)]
fn build_viewport(
    xyz: [i32; 3],
    u_coord: usize,
    v_coord: usize,
    plane_coord: usize,
    min_uc: i32,
    max_uc: i32,
    min_vc: i32,
    max_vc: i32,
    pc: i32,
    target_lod: u8,
    max_lod: u8,
) -> Viewport {
    let mut per_lod: Vec<Option<LodView>> = vec![None; (max_lod as usize) + 1];
    for lod in target_lod..=max_lod {
        let Some(t) = viewport_tiles(min_uc, max_uc, min_vc, max_vc, pc, lod) else {
            continue;
        };
        let scale = 1i32 << lod;
        let chunk_world = 64 * scale;
        let center_u = xyz[u_coord].div_euclid(chunk_world);
        let center_v = xyz[v_coord].div_euclid(chunk_world);
        let center_p = t.tile_pc;
        let mut center = [0i32; 3];
        center[u_coord] = center_u;
        center[v_coord] = center_v;
        center[plane_coord] = center_p;
        let mut rect_lo = [0i32; 3];
        let mut rect_hi = [0i32; 3];
        rect_lo[u_coord] = t.tile_u_lo;
        rect_hi[u_coord] = t.tile_u_hi;
        rect_lo[v_coord] = t.tile_v_lo;
        rect_hi[v_coord] = t.tile_v_hi;
        rect_lo[plane_coord] = t.tile_pc;
        rect_hi[plane_coord] = t.tile_pc;
        per_lod[lod as usize] = Some(LodView { center, rect_lo, rect_hi });
    }
    Viewport { max_lod, per_lod }
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

        // Local viewport — used purely to derive a per-chunk priority for
        // this paint pass. The cache + downloader queues sort by the
        // priority value passed at submit time; submission order across
        // panes and LODs gives the right global ordering without any
        // viewport-state tracking inside the queues.
        let viewport = build_viewport(
            xyz,
            u_coord,
            v_coord,
            plane_coord,
            min_uc,
            max_uc,
            min_vc,
            max_vc,
            pc,
            target_lod,
            max_lod,
        );

        // -------- Pass 1: dispatch coarse → fine --------
        // Walking coarse → fine means the first submissions of a cold
        // viewport are the low-LOD preview chunks. Combined with the
        // BTreeMap-by-priority queue, the workers prefer those even if
        // another pane has already enqueued finer-LOD work.
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
                    let priority = viewport.priority_for(key);
                    let _ = self.cache.state_or_fetch_with_priority(key, priority);
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

                // Walk target → coarsest, take the first resident parent.
                let mut chosen: Option<(u8, Arc<ChunkState>)> = None;
                for lod_try in target_lod..=max_lod {
                    let shift = lod_try - target_lod;
                    let mut chunk = [0u32; 3];
                    chunk[u_coord] = (tu as u32) >> shift;
                    chunk[v_coord] = (tv as u32) >> shift;
                    chunk[plane_coord] = (t.tile_pc as u32) >> shift;
                    let key = ChunkKey::new(lod_try, chunk[0], chunk[1], chunk[2]);
                    let s = if lod_try == target_lod {
                        let priority = viewport.priority_for(key);
                        self.cache.state_or_fetch_with_priority(key, priority)
                    } else {
                        match self.cache.peek(key) {
                            Some(s) => s,
                            None => continue,
                        }
                    };
                    if s.as_resident().is_some() {
                        chosen = Some((lod_try, s));
                        break;
                    }
                }
                let Some((lod_use, state)) = chosen else { continue };
                let mmap = state.as_resident().expect("just checked resident");

                let scale_use = 1i32 << lod_use;
                let chunk_world_use = 64 * scale_use;
                let shift = lod_use - target_lod;
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
            }
        }
    }

    fn shared(&self) -> VolumeCons {
        let cache = self.cache.clone();
        Box::new(move || VoxelPaintVolume::into_volume(UnifiedVolume::new(cache)))
    }
}
