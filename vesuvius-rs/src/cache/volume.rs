//! `UnifiedVolume` — single `VoxelVolume` + `PaintVolume` impl over a
//! `ChunkCache`. The only place every backfiller-backed volume needs.

use super::cache::ChunkCache;
use super::state::{ChunkKey, ChunkState};
use crate::volume::{DrawingConfig, Image, PaintVolume, VolumeCons, VoxelPaintVolume, VoxelVolume};
use std::cell::RefCell;
use std::sync::Arc;

pub struct UnifiedVolume {
    cache: ChunkCache,
    // Per-volume hot slot for the last chunk touched by `get`. The painting
    // path drops this between frames via `reset_for_painting`. `Volume`
    // clones produce fresh `UnifiedVolume` instances with their own slot via
    // the `shared()` constructor, so the `RefCell` is never shared across
    // threads.
    local: RefCell<LocalSlot>,
}

#[derive(Default)]
struct LocalSlot {
    key: Option<ChunkKey>,
    state: Option<Arc<ChunkState>>,
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

    /// Per-pixel chunk lookup with the hot-slot shortcut.
    fn chunk_state(&self, key: ChunkKey) -> Arc<ChunkState> {
        {
            let b = self.local.borrow();
            if b.key == Some(key) {
                if let Some(s) = &b.state {
                    return s.clone();
                }
            }
        }
        let state = self.cache.state_or_fetch(key);
        let mut b = self.local.borrow_mut();
        b.key = Some(key);
        b.state = Some(state.clone());
        state
    }

    fn drop_hot_slot(&self) {
        let mut b = self.local.borrow_mut();
        b.key = None;
        b.state = None;
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
        let lod = lod_for(downsampling.max(1) as u8);
        let scale = 1u32 << lod;
        // World voxel → sample index at this LOD.
        let sx = (xyz[0] as i64 / scale as i64).max(0) as u32;
        let sy = (xyz[1] as i64 / scale as i64).max(0) as u32;
        let sz = (xyz[2] as i64 / scale as i64).max(0) as u32;

        let key = ChunkKey::new(lod, sx / 64, sy / 64, sz / 64);
        let state = self.chunk_state(key);
        if let Some(mmap) = state.as_resident() {
            let off = ((sz & 63) as usize) * 64 * 64 + ((sy & 63) as usize) * 64 + (sx & 63) as usize;
            mmap[off]
        } else {
            0
        }
    }
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
        // We do per-chunk lookup; the hot slot doesn't help here and just
        // wastes cache space.
        self.drop_hot_slot();

        let lod = lod_for(sfactor);
        let scale = 1i32 << lod;
        // 2D viewport in world coordinates.
        let width_world = paint_zoom as i32 * canvas_width as i32;
        let height_world = paint_zoom as i32 * canvas_height as i32;
        let min_uc = xyz[u_coord] - width_world / 2;
        let max_uc = xyz[u_coord] + width_world / 2;
        let min_vc = xyz[v_coord] - height_world / 2;
        let max_vc = xyz[v_coord] + height_world / 2;
        let pc = xyz[plane_coord];
        if pc < 0 {
            return;
        }

        let chunk_world = 64 * scale; // one chunk covers this many world voxels on each axis
        let pzoom = paint_zoom as i32;
        // Chunk indices spanned by the viewport. div_euclid keeps negative
        // viewport positions outside the chunk grid so the loop simply skips.
        let tile_min_uc = min_uc.div_euclid(chunk_world).max(0);
        let tile_max_uc = max_uc.div_euclid(chunk_world);
        let tile_min_vc = min_vc.div_euclid(chunk_world).max(0);
        let tile_max_vc = max_vc.div_euclid(chunk_world);
        let tile_pc = pc.div_euclid(chunk_world);
        if tile_pc < 0 {
            return;
        }
        // Sample (intra-chunk) coordinate on the plane axis at this LOD.
        let plane_sample = (pc.rem_euclid(chunk_world) / scale) as usize;

        for tu in tile_min_uc..=tile_max_uc {
            for tv in tile_min_vc..=tile_max_vc {
                let mut chunk = [0i32; 3];
                chunk[u_coord] = tu;
                chunk[v_coord] = tv;
                chunk[plane_coord] = tile_pc;
                let key = ChunkKey::new(lod, chunk[0] as u32, chunk[1] as u32, chunk[2] as u32);
                let state = self.cache.state_or_fetch(key);
                let mmap = match state.as_resident() {
                    Some(m) => m,
                    None => continue,
                };

                // Iterate by destination pixel: for each pixel mapped into
                // this chunk, look up the sample. This is the right
                // ordering — stepping by sample units would skip world
                // pixels whenever `scale > paint_zoom`, producing visible
                // gaps at sfactor > 1.
                let chunk_u_lo = tu * chunk_world;
                let chunk_u_hi = chunk_u_lo + chunk_world;
                let chunk_v_lo = tv * chunk_world;
                let chunk_v_hi = chunk_v_lo + chunk_world;

                // Both bounds are ceil((edge - origin) / pzoom): the smallest
                // u_px with min_uc + u_px*pzoom >= edge. u_px_hi must use the
                // same formula as u_px_lo or boundary pixels get skipped
                // when (chunk_hi - min_uc) isn't divisible by pzoom — the
                // "grid lines at zoomed-out LODs" symptom.
                let ceil_div = |x: i32, d: i32| (x + d - 1).div_euclid(d);
                let u_px_lo = ceil_div(chunk_u_lo - min_uc, pzoom).max(0);
                let u_px_hi = ceil_div(chunk_u_hi - min_uc, pzoom).min(canvas_width as i32);
                let v_px_lo = ceil_div(chunk_v_lo - min_vc, pzoom).max(0);
                let v_px_hi = ceil_div(chunk_v_hi - min_vc, pzoom).min(canvas_height as i32);

                for v_px in v_px_lo..v_px_hi {
                    let world_v = min_vc + v_px * pzoom;
                    let sample_v = ((world_v - chunk_v_lo) / scale) as usize;
                    for u_px in u_px_lo..u_px_hi {
                        let world_u = min_uc + u_px * pzoom;
                        let sample_u = ((world_u - chunk_u_lo) / scale) as usize;
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
