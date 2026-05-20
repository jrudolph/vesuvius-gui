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

        let chunk_world = 64 * scale;
        let tile_min_uc = (min_uc.div_euclid(chunk_world)).max(0);
        let tile_max_uc = max_uc.div_euclid(chunk_world);
        let tile_min_vc = (min_vc.div_euclid(chunk_world)).max(0);
        let tile_max_vc = max_vc.div_euclid(chunk_world);
        let tile_pc = pc.div_euclid(chunk_world);
        if tile_pc < 0 {
            return;
        }
        // Intra-chunk plane offset in sample units.
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

                // Range of intra-chunk sample indices on this tile.
                let chunk_min_u = (tu * chunk_world).max(min_uc) - tu * chunk_world;
                let chunk_max_u = ((tu + 1) * chunk_world).min(max_uc) - tu * chunk_world;
                let chunk_min_v = (tv * chunk_world).max(min_vc) - tv * chunk_world;
                let chunk_max_v = ((tv + 1) * chunk_world).min(max_vc) - tv * chunk_world;
                // Convert world offsets to sample indices.
                let su_lo = (chunk_min_u / scale) as usize;
                let su_hi = ((chunk_max_u + scale - 1) / scale) as usize;
                let sv_lo = (chunk_min_v / scale) as usize;
                let sv_hi = ((chunk_max_v + scale - 1) / scale) as usize;

                let step = paint_zoom as usize;
                for sv in (sv_lo..sv_hi.min(64)).step_by(step) {
                    for su in (su_lo..su_hi.min(64)).step_by(step) {
                        let mut s = [0usize; 3];
                        s[u_coord] = su;
                        s[v_coord] = sv;
                        s[plane_coord] = plane_sample;
                        let off = s[2] * 64 * 64 + s[1] * 64 + s[0];
                        let value = config.filter(mmap[off]);

                        // World coord → pixel coord.
                        let world_u = tu * chunk_world + su as i32 * scale;
                        let world_v = tv * chunk_world + sv as i32 * scale;
                        let u_px = (world_u - min_uc) / paint_zoom as i32;
                        let v_px = (world_v - min_vc) / paint_zoom as i32;
                        if u_px >= 0 && u_px < canvas_width as i32 && v_px >= 0 && v_px < canvas_height as i32 {
                            buffer.set_gray(u_px as usize, v_px as usize, value);
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
