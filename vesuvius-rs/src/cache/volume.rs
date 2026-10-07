//! `UnifiedVolume` — single `VoxelVolume` + `PaintVolume` impl over a
//! `ChunkCache`. The only place every backfiller-backed volume needs.
//!
//! ## LOD fallback strategy
//!
//! `paint()` and `get()` both walk the LOD pyramid: if the requested
//! ("target") LOD chunk isn't resident yet, we sample from the first
//! coarser-LOD parent chunk that is. The target tile's screen region is
//! painted **once** at whichever LOD has data — we never overdraw multiple
//! LODs into the same region. Every missing level of the climb is
//! requested; the work queues fetch the coarser ones first (see
//! `lease::chunk_priority`), so a low-res preview shows up promptly while
//! detail streams in.
//!
//! `get()` is the per-voxel sampler used by surface (ObjVolume) and PPM
//! renderers that don't go through `UnifiedVolume::paint`. It climbs the
//! pyramid through the per-LOD shard hot slots (`resolve_chunk`): each
//! level is one atomic sidecar-byte probe, a never-dispatched chunk gets
//! its dispatch kicked off exactly once, and the first `Resident` level
//! wins (`Empty` stops the climb — fine-grained absence overrides coarser
//! data). `interpolate_u8` additionally keeps a chunk-grain slot caching
//! its resolved `(lod, chunk_state)` binding for repeat samples in the
//! same target chunk.
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
use super::disk::{DispatchedBits, ShardCoord};
use super::sidecar::{Sidecar, STATE_EMPTY, STATE_LOCKED, STATE_RESIDENT};
use super::state::{ChunkKey, ChunkState};
use super::CHUNK_VOXELS;
#[cfg(test)]
use crate::volume::composition::MaxCompositionState;
use crate::volume::composition::{CompositionState, CompositorRef};
use crate::volume::{DrawingConfig, Image, PaintVolume, VolumeCons, VoxelPaintVolume, VoxelVolume};
use ecolor::Color32;
use memmap::Mmap;
use std::cell::RefCell;
use std::sync::Arc;

/// Overlay alpha for `debug_chunk_overlay`. Low enough that voxel detail is
/// still readable underneath.
const OVERLAY_ALPHA: f32 = 0.35;

/// Number of shard hot slots, one per LOD level. 16 covers any conceivable
/// pyramid depth — at 64³ chunks the deepest is ≈ 8 (2⁸ · 64 = 16384 base
/// voxels per coarsest chunk). Indexing by `lod as usize` skips a slot
/// match key compare in the per-voxel fast path.
const SHARD_SLOTS_PER_LOD: usize = 16;

pub struct UnifiedVolume {
    cache: ChunkCache,
    /// Durable per-chunk state map, shared with the cache's DiskStore.
    /// The per-voxel fast path probes one atomic byte here per LOD-climb
    /// level: Resident / Empty / Locked come straight from this map; only
    /// the in-memory "dispatched" claim lives elsewhere (per-shard bits
    /// on the `ShardSlot`).
    sidecar: Arc<Sidecar>,
    /// Shard side (in chunks) as log2 + mask — cached from
    /// `ChunkCache::shard_chunks_per_axis` at construction (production
    /// value 128 → shift 7). Stored pre-decomposed so the per-voxel
    /// shard-coord derivation is shifts/ands instead of div/mod by a
    /// runtime value.
    shard_shift: u32,
    shard_mask: u32,
    /// World-voxel (LOD 0) extent of the volume, cached from the
    /// backfiller. Samples at or beyond the extent return 0 up front —
    /// without this, out-of-bounds chunks (whose dispatched bits stay
    /// clear forever because dispatch bails before marking them) would
    /// send every OOB voxel down the DashMap + clock-read slow path at
    /// every LOD climb level.
    extent: [u32; 3],
    // Per-volume hot slot for the last target chunk touched by `get`. The
    // cached `chosen` may be a coarser-LOD parent when the target chunk
    // wasn't resident on first hit. `paint()` and external callers clear it
    // between frames via `reset_for_painting`. `Volume` clones produce fresh
    // `UnifiedVolume` instances with their own slot via the `shared()`
    // constructor, so the `RefCell` is never shared across threads.
    local: RefCell<LocalSlot>,
}

struct LocalSlot {
    /// One shard-grain fast path *per LOD*. Indexed by `lod as usize`.
    /// The interpolation lattice path in `interpolate_u8` recurses into
    /// `get` at the *chosen* LOD (which may be coarser than the original
    /// target LOD), so a single shared slot would thrash between e.g.
    /// LOD 0 and a coarser-parent LOD as the +1 corner crosses a chunk
    /// boundary. Per-LOD slots keep both alive simultaneously without
    /// cross-eviction. The sidecar probe gates reads so unwritten chunks
    /// fall back to the slow path instead of returning the kernel's zero
    /// page as if it were data.
    shards: [Option<ShardSlot>; SHARD_SLOTS_PER_LOD],
    /// Single auxiliary shard slot for cross-shard +1-neighbor reads in
    /// `composite_along_normal_inner`'s boundary path. Kept separate from
    /// `shards` so reading a neighbor across a shard plane never evicts
    /// the home slot mid-ray. A surface that runs parallel to a shard
    /// plane grazes the *same* neighbor shard for an entire band of rays,
    /// so one slot amortizes nearly all the opens.
    neighbor_shard: Option<ShardSlot>,
    /// Chunk-grain slot for the slow path's LOD-fallback result. Lets
    /// repeat samples in the same target chunk skip the pyramid walk
    /// while a resolved coarser parent is the best we have.
    target_key: Option<ChunkKey>,
    chosen: Option<(u8, Arc<ChunkState>)>,
    /// Already-dispatched, still-missing chunks this paint has re-reported
    /// to the cache (see `resolve_chunk`). Cleared per paint, so each such
    /// chunk goes through `state_or_fetch` once per paint.
    revisited: fxhash::FxHashSet<ChunkKey>,
    /// Last chunk passed to `paint_scope::record`; skips the thread-local
    /// set insert for the run of samples that hit the same missing chunk.
    last_missing: Option<ChunkKey>,
}

impl Default for LocalSlot {
    fn default() -> Self {
        Self {
            shards: [const { None }; SHARD_SLOTS_PER_LOD],
            neighbor_shard: None,
            target_key: None,
            chosen: None,
            revisited: Default::default(),
            last_missing: None,
        }
    }
}

struct ShardSlot {
    shard: ShardCoord,
    base: *const u8,
    _mmap: Arc<Mmap>,
    /// Per-chunk "dispatch attempted" bits for this shard. Consulted only
    /// when the sidecar says MISSING, so the slow path kicks each chunk's
    /// dispatch exactly once instead of re-entering the DashMap per voxel.
    dispatched: Arc<DispatchedBits>,
}

impl UnifiedVolume {
    pub fn new(cache: ChunkCache) -> Self {
        let sca = cache.shard_chunks_per_axis();
        assert!(sca.is_power_of_two(), "shard side must be a power of two, got {}", sca);
        let extent = cache.voxel_extent();
        let sidecar = cache.sidecar();
        Self {
            cache,
            sidecar,
            shard_shift: sca.trailing_zeros(),
            shard_mask: sca - 1,
            extent,
            local: RefCell::new(LocalSlot::default()),
        }
    }

    /// True iff a sample at `(sx, sy, sz)` (voxel coords at `lod`) falls
    /// inside the volume extent.
    #[inline]
    fn sample_in_bounds(&self, sx: u64, sy: u64, sz: u64, lod: u8) -> bool {
        (sx << lod) < self.extent[0] as u64 && (sy << lod) < self.extent[1] as u64 && (sz << lod) < self.extent[2] as u64
    }

    pub fn cache(&self) -> &ChunkCache {
        &self.cache
    }

    fn drop_hot_slot(&self) {
        let mut b = self.local.borrow_mut();
        for s in b.shards.iter_mut() {
            *s = None;
        }
        b.neighbor_shard = None;
        b.target_key = None;
        b.chosen = None;
        b.revisited.clear();
        b.last_missing = None;
    }

    /// Report a target-LOD chunk this paint couldn't use (see `paint_scope`).
    /// `fallback`: the LOD drawn instead, `None` if nothing was.
    fn note_missing(&self, key: ChunkKey, fallback: Option<u8>) {
        let mut b = self.local.borrow_mut();
        if b.last_missing != Some(key) {
            b.last_missing = Some(key);
            drop(b);
            super::paint_scope::record(&self.cache, key, fallback);
        }
    }

    /// Decompose a target-LOD chunk coord into `(shard_coord,
    /// in_shard_chunk_idx)`. The byte offset of voxel `(vx, vy, vz)` inside
    /// the shard mmap is then
    /// `in_shard_chunk_idx * CHUNK_VOXELS + ((vz & 63) * 64 * 64 + (vy & 63) * 64 + (vx & 63))`.
    #[inline]
    fn shard_decompose(&self, cx: u32, cy: u32, cz: u32) -> (ShardCoord, u64) {
        let shift = self.shard_shift;
        let mask = self.shard_mask;
        let shard = (cx >> shift, cy >> shift, cz >> shift);
        let wx = (cx & mask) as u64;
        let wy = (cy & mask) as u64;
        let wz = (cz & mask) as u64;
        let in_shard_idx = ((wz << shift) | wy) << shift | wx;
        (shard, in_shard_idx)
    }

    /// Probe the sidecar's state byte for the chunk at `(lod, cx, cy, cz)`.
    /// One atomic Acquire load — pairs with `write_atomic`'s Release store
    /// so an observed `STATE_RESIDENT` guarantees the mmap bytes are
    /// visible. `None` when the chunk is outside the LOD's grid.
    #[inline]
    fn sidecar_state(&self, lod: u8, cx: u32, cy: u32, cz: u32) -> Option<u8> {
        let dims = self.sidecar.header.lods.get(lod as usize)?;
        let idx = dims.linear_index(cx, cy, cz)?;
        Some(self.sidecar.get_state(lod, idx))
    }

    /// Run `f` against the hot shard slot for `(lod, shard)`, populating
    /// the slot (and opening the shard file) on a miss. The common case —
    /// slot already addresses this shard, true for every voxel after the
    /// first in a shard — is a single RefCell borrow; only a slot miss
    /// pays the populate + re-probe. Returns `None` when the shard can't
    /// be opened (I/O error or out-of-grid).
    #[inline]
    fn with_shard_slot<R>(&self, lod: u8, shard: ShardCoord, f: impl FnOnce(&ShardSlot) -> R) -> Option<R> {
        let lod_ix = lod as usize;
        if lod_ix >= SHARD_SLOTS_PER_LOD {
            return None;
        }
        {
            let b = self.local.borrow();
            if let Some(slot) = b.shards[lod_ix].as_ref() {
                if slot.shard == shard {
                    return Some(f(slot));
                }
            }
        }
        self.populate_shard_slot(lod, shard);
        let b = self.local.borrow();
        match b.shards[lod_ix].as_ref() {
            Some(slot) if slot.shard == shard => Some(f(slot)),
            _ => None,
        }
    }

    /// Borrow the resident chunk's 64³ slice from the shard hot slot, if
    /// the slot addresses the same `(lod, shard)` that contains
    /// `(target_cx, target_cy, target_cz)` AND the chunk's sidecar byte
    /// reads Resident. Returns the slice base pointer; `None` for any
    /// other state (caller must slow-path — `resolve_chunk` distinguishes
    /// Empty from Missing/Dispatched there).
    #[inline]
    fn shard_slot_chunk_slice(&self, target_lod: u8, target_cx: u32, target_cy: u32, target_cz: u32) -> Option<*const u8> {
        let lod_ix = target_lod as usize;
        if lod_ix >= SHARD_SLOTS_PER_LOD {
            return None;
        }
        let (shard, in_shard_idx) = self.shard_decompose(target_cx, target_cy, target_cz);
        let b = self.local.borrow();
        let slot = b.shards[lod_ix].as_ref()?;
        if slot.shard != shard {
            return None;
        }
        match self.sidecar_state(target_lod, target_cx, target_cy, target_cz) {
            // SAFETY: in_shard_idx * CHUNK_VOXELS + CHUNK_VOXELS ≤
            // shard mmap length.
            Some(STATE_RESIDENT) => Some(unsafe { slot.base.add((in_shard_idx as usize) * CHUNK_VOXELS) }),
            _ => None,
        }
    }

    /// Populate the shard hot slot for `(target_lod, shard)`, opening the
    /// shard file (sparse mmap) if it isn't already. After this call,
    /// per-voxel reads in this shard run entirely off the cached mmap
    /// base + direct sidecar probes — never re-entering the DashMap or
    /// the per-LOD `opened` mutex.
    ///
    /// No-op if the slot already addresses this shard.
    fn populate_shard_slot(&self, target_lod: u8, shard: ShardCoord) {
        let lod_ix = target_lod as usize;
        if lod_ix >= SHARD_SLOTS_PER_LOD {
            return;
        }
        {
            let b = self.local.borrow();
            if let Some(slot) = &b.shards[lod_ix] {
                if slot.shard == shard {
                    return;
                }
            }
        }
        if let Some(snap) = self.cache.ensure_shard_open(target_lod, shard) {
            let base = snap.mmap.as_ptr();
            let mut b = self.local.borrow_mut();
            b.shards[lod_ix] = Some(ShardSlot {
                shard,
                base,
                _mmap: snap.mmap,
                dispatched: snap.dispatched,
            });
        }
    }

    /// Resolve the chunk base pointer for a +1 neighbor chunk that lives in
    /// a *different* shard than the home slot, using the dedicated
    /// `neighbor_shard` slot so the home slot is never evicted. Returns the
    /// chunk base pointer plus an `Arc<Mmap>` keepalive so the caller can
    /// build a slice that outlives the `local` borrow. `None` when the
    /// neighbor shard is out of grid / can't be opened — caller then treats
    /// the neighbor corners as 0 (the true volume edge, same as before).
    #[inline]
    fn neighbor_chunk_base(&self, lod: u8, ncx: u32, ncy: u32, ncz: u32) -> Option<(*const u8, Arc<Mmap>)> {
        let (shard, in_shard_idx) = self.shard_decompose(ncx, ncy, ncz);
        let chunk_off = (in_shard_idx as usize) * CHUNK_VOXELS;
        {
            let b = self.local.borrow();
            if let Some(slot) = b.neighbor_shard.as_ref() {
                if slot.shard == shard {
                    // SAFETY: in_shard_idx * CHUNK_VOXELS + CHUNK_VOXELS ≤ shard mmap len.
                    return Some((unsafe { slot.base.add(chunk_off) }, slot._mmap.clone()));
                }
            }
        }
        let snap = self.cache.ensure_shard_open(lod, shard)?;
        let base = snap.mmap.as_ptr();
        let mmap = snap.mmap.clone();
        {
            let mut b = self.local.borrow_mut();
            b.neighbor_shard = Some(ShardSlot {
                shard,
                base,
                _mmap: snap.mmap,
                dispatched: snap.dispatched,
            });
        }
        Some((unsafe { base.add(chunk_off) }, mmap))
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

    fn touch_aabb(&self, min: [f64; 3], max: [f64; 3], downsampling: i32) {
        let target_lod = lod_for(downsampling.max(1) as u8);
        self.cache.touch_aabb(min, max, target_lod);
    }

    fn get(&self, xyz: [f64; 3], downsampling: i32) -> u8 {
        // Per the `VoxelVolume::get` convention shared with VolumeGrid64x4Mapped,
        // ZarrContext, and PPMVolume callers (notably `ObjVolume::paint`, which
        // passes `[x / sfactor, y / sfactor, z / sfactor]`): `xyz` is in
        // voxel-coords at the requested downsampling. We do NOT re-divide by
        // scale here.
        let target_lod = lod_for(downsampling.max(1) as u8);
        let max_lod = self.cache.max_lod();

        let target_sx = (xyz[0] as i64).max(0) as u64;
        let target_sy = (xyz[1] as i64).max(0) as u64;
        let target_sz = (xyz[2] as i64).max(0) as u64;
        if !self.sample_in_bounds(target_sx, target_sy, target_sz, target_lod) {
            return 0;
        }
        let target_cx = (target_sx / 64) as u32;
        let target_cy = (target_sy / 64) as u32;
        let target_cz = (target_sz / 64) as u32;

        match self.resolve_chunk(target_lod, max_lod, target_cx, target_cy, target_cz) {
            Some(bound) if bound.shift == 0 => {
                let off = ((target_sz & 63) as usize) * 64 * 64
                    + ((target_sy & 63) as usize) * 64
                    + (target_sx & 63) as usize;
                // SAFETY: off < CHUNK_VOXELS by construction (each
                // component masked to 0..=63).
                unsafe { *bound.chunk_ptr.add(off) }
            }
            Some(bound) => {
                // Sampled via a coarser parent — re-derive the in-chunk
                // offset in the parent's coord space.
                let lod_use = target_lod + bound.shift;
                let mmap = unsafe { bound.mmap_slice() };
                sample_at(target_sx, target_sy, target_sz, target_lod, lod_use, mmap)
            }
            None => 0,
        }
    }

    fn get_interpolated(&self, xyz: [f64; 3], downsampling: i32) -> u8 {
        self.interpolate_u8(xyz, downsampling)
    }

    fn get_color_interpolated(&self, xyz: [f64; 3], downsampling: i32) -> Color32 {
        Color32::from_gray(self.interpolate_u8(xyz, downsampling))
    }

    /// Fast-path override of the trait default. Amortizes the per-sample
    /// chunk lookup that `get_interpolated` redoes from scratch each call,
    /// then walks the ray as Q32.32 fixed-point with Q0.8 trilerp inside
    /// each chunk. Boundary samples (4.7% at random directions) use an
    /// inline 2-chunk path; 2- and 3-axis chunk corners (~0.07% / 0.0004%)
    /// fall back to `interpolate_u8`.
    ///
    /// The unswitch on `CompositorRef` is what buys per-sample
    /// monomorphization: each arm calls a separate instantiation of the
    /// generic `composite_along_normal_inner` so `state.update(v)` folds
    /// into the inner loop with zero virtual dispatch.
    fn composite_along_normal(
        &self,
        base: [f64; 3],
        dir: [f64; 3],
        w_lo: f64,
        w_hi: f64,
        downsampling: i32,
        compositor: &mut CompositorRef<'_>,
    ) {
        match compositor {
            CompositorRef::Max(s) => {
                self.composite_along_normal_inner(base, dir, w_lo, w_hi, downsampling, |v| s.update(v))
            }
            CompositorRef::Alpha(s) => {
                self.composite_along_normal_inner(base, dir, w_lo, w_hi, downsampling, |v| s.update(v))
            }
            CompositorRef::HeightMap(s) => {
                self.composite_along_normal_inner(base, dir, w_lo, w_hi, downsampling, |v| s.update(v))
            }
            CompositorRef::None(s) => {
                self.composite_along_normal_inner(base, dir, w_lo, w_hi, downsampling, |v| s.update(v))
            }
        }
    }

    /// Fast-path override: fill `out` with samples along the ray using the
    /// same amortized shard walk as `composite_along_normal`, instead of the
    /// trait default's per-sample `get_interpolated`.
    fn gather_along_normal(&self, base: [f64; 3], dir: [f64; 3], downsampling: i32, out: &mut [u8]) {
        let n = out.len();
        if n == 0 {
            return;
        }
        let mut i = 0usize;
        self.composite_along_normal_inner(base, dir, 0.0, n as f64, downsampling, |v| {
            out[i] = v;
            i += 1;
            i < n
        });
    }
}

impl UnifiedVolume {
    /// Shard-based Q32.32 + Q0.8 ray walker. Generic over the per-sample
    /// sink so each `CompositorRef` arm gets its own monomorphization —
    /// `sink(v)` inlines to the concrete `CompositionState::update` body,
    /// no virtual dispatch in the hot loop.
    ///
    /// The inner loop is intentionally chunk-state-unaware: it reads
    /// straight off the target-LOD shard mmap with no atomic probes, no
    /// LOD climb, no slow-mode fallback. Pre-dispatch (see
    /// `touch_aabb`, called per-triangle by `ObjVolume::paint`) lands the
    /// chunks the ray crosses before this is called; un-arrived bytes
    /// read as zero from the sparse mmap, and the optional one-level
    /// upscale-from-parent path in `touch_aabb` fills target chunks with
    /// a preview while the real bytes stream in.
    fn composite_along_normal_inner<F: FnMut(u8) -> bool>(
        &self,
        base: [f64; 3],
        dir: [f64; 3],
        w_lo: f64,
        w_hi: f64,
        downsampling: i32,
        mut sink: F,
    ) {
        let target_lod = lod_for(downsampling.max(1) as u8);
        let n_total = (w_hi - w_lo) as i32;
        if n_total <= 0 {
            return;
        }

        let dx = dir[0];
        let dy = dir[1];
        let dz = dir[2];

        let mut px = base[0] + w_lo * dx;
        let mut py = base[1] + w_lo * dy;
        let mut pz = base[2] + w_lo * dz;

        let mut remaining = n_total as usize;
        let sh = self.shard_shift;
        let mask = self.shard_mask;

        // Upper bounds in target-LOD voxel units. Rays running past the
        // volume extent (common for surfaces near the scroll edge) feed 0
        // here instead of probing forever-missing chunks.
        let scale = (1u64 << target_lod) as f64;
        let ext_x = self.extent[0] as f64 / scale;
        let ext_y = self.extent[1] as f64 / scale;
        let ext_z = self.extent[2] as f64 / scale;

        while remaining > 0 {
            if px < 0.0 || py < 0.0 || pz < 0.0 || px >= ext_x || py >= ext_y || pz >= ext_z {
                if !sink(0) {
                    return;
                }
                px += dx;
                py += dy;
                pz += dz;
                remaining -= 1;
                continue;
            }
            let txu = px as u64;
            let tyu = py as u64;
            let tzu = pz as u64;
            let target_cx = (txu / 64) as u32;
            let target_cy = (tyu / 64) as u32;
            let target_cz = (tzu / 64) as u32;
            let (shard, in_shard_idx) = self.shard_decompose(target_cx, target_cy, target_cz);
            self.populate_shard_slot(target_lod, shard);

            // Borrow the slot just long enough to clone pointers + the
            // mmap Arc. After this block we never re-enter `self.local`
            // until the next outer iteration, so the raw pointers remain
            // valid.
            let (slot_base, _mmap_keepalive, slot_shard) = {
                let b = self.local.borrow();
                match b.shards[target_lod as usize].as_ref() {
                    Some(slot) => (slot.base, slot._mmap.clone(), slot.shard),
                    None => {
                        // Shard couldn't be opened (I/O error); feed 0
                        // and advance.
                        if !sink(0) {
                            return;
                        }
                        px += dx;
                        py += dy;
                        pz += dz;
                        remaining -= 1;
                        continue;
                    }
                }
            };
            if slot_shard != shard {
                if !sink(0) {
                    return;
                }
                px += dx;
                py += dy;
                pz += dz;
                remaining -= 1;
                continue;
            }
            let chunk_base = unsafe { slot_base.add((in_shard_idx as usize) * CHUNK_VOXELS) };

            let bx = (txu & 63) == 63;
            let by = (tyu & 63) == 63;
            let bz = (tzu & 63) == 63;
            let n_boundary = bx as u8 + by as u8 + bz as u8;

            if n_boundary >= 2 {
                // 2/3-axis +1 corner crossings need samples from up to 7
                // neighbor chunks across possibly multiple shards. Rare
                // (~0.07% / 0.0004%); defer to the legacy trilerp path.
                let v = self.interpolate_u8([px, py, pz], downsampling);
                if !sink(v) {
                    return;
                }
                px += dx;
                py += dy;
                pz += dz;
                remaining -= 1;
                continue;
            }
            if n_boundary == 1 {
                let (nx_c, ny_c, nz_c) = if bx {
                    (target_cx + 1, target_cy, target_cz)
                } else if by {
                    (target_cx, target_cy + 1, target_cz)
                } else {
                    (target_cx, target_cy, target_cz + 1)
                };
                // Same-shard neighbor: read direct from the home shard
                // mmap. Cross-shard +1 corner: pull the neighbor chunk via
                // the dedicated `neighbor_shard` slot so we don't evict the
                // home slot. (Treating cross-shard neighbors as 0 produced a
                // visible dark seam wherever a surface runs parallel to a
                // shard plane — the whole band grazes the boundary instead of
                // the ~0.04% a random ray direction would.) Either way an
                // unwritten chunk reads kernel-zero from the sparse mmap.
                let same_shard = nx_c >> sh == shard.0 && ny_c >> sh == shard.1 && nz_c >> sh == shard.2;
                let home_slice = unsafe { std::slice::from_raw_parts(chunk_base, CHUNK_VOXELS) };
                let v = if same_shard {
                    let nwx = (nx_c & mask) as u64;
                    let nwy = (ny_c & mask) as u64;
                    let nwz = (nz_c & mask) as u64;
                    let neighbor_idx = ((nwz << sh) | nwy) << sh | nwx;
                    let neighbor_base =
                        unsafe { slot_base.add((neighbor_idx as usize) * CHUNK_VOXELS) };
                    let neighbor_slice = unsafe { std::slice::from_raw_parts(neighbor_base, CHUNK_VOXELS) };
                    sample_boundary_1axis(home_slice, Some(neighbor_slice), bx, by, bz, txu, tyu, tzu, px, py, pz)
                } else {
                    match self.neighbor_chunk_base(target_lod, nx_c, ny_c, nz_c) {
                        Some((neighbor_base, _keepalive)) => {
                            let neighbor_slice = unsafe { std::slice::from_raw_parts(neighbor_base, CHUNK_VOXELS) };
                            sample_boundary_1axis(home_slice, Some(neighbor_slice), bx, by, bz, txu, tyu, tzu, px, py, pz)
                        }
                        None => sample_boundary_1axis(home_slice, None, bx, by, bz, txu, tyu, tzu, px, py, pz),
                    }
                };
                if !sink(v) {
                    return;
                }
                px += dx;
                py += dy;
                pz += dz;
                remaining -= 1;
                continue;
            }

            // Pure in-chunk run-length trilerp from the shard mmap.
            let lo_x = (target_cx as f64) * 64.0;
            let lo_y = (target_cy as f64) * 64.0;
            let lo_z = (target_cz as f64) * 64.0;
            let hi_x = lo_x + 63.0;
            let hi_y = lo_y + 63.0;
            let hi_z = lo_z + 63.0;
            let kx = run_length_1d(px, dx, lo_x, hi_x);
            let ky = run_length_1d(py, dy, lo_y, hi_y);
            let kz = run_length_1d(pz, dz, lo_z, hi_z);
            let run = kx.min(ky).min(kz).min(remaining);
            debug_assert!(run >= 1);

            let mmap = unsafe { std::slice::from_raw_parts(chunk_base, CHUNK_VOXELS) };
            const Q32_F: f64 = 4294967296.0; // 2^32
            let mut posx = unsafe { (px * Q32_F).to_int_unchecked::<i64>() };
            let mut posy = unsafe { (py * Q32_F).to_int_unchecked::<i64>() };
            let mut posz = unsafe { (pz * Q32_F).to_int_unchecked::<i64>() };
            let dx_q = unsafe { (dx * Q32_F).to_int_unchecked::<i64>() };
            let dy_q = unsafe { (dy * Q32_F).to_int_unchecked::<i64>() };
            let dz_q = unsafe { (dz * Q32_F).to_int_unchecked::<i64>() };

            let mut stopped = false;
            let mut consumed = 0usize;
            unsafe {
                for _ in 0..run {
                    let cx = (posx >> 32) as u64;
                    let cy = (posy >> 32) as u64;
                    let cz = (posz >> 32) as u64;
                    let tx = (cx & 63) as usize;
                    let ty = (cy & 63) as usize;
                    let tz = (cz & 63) as usize;
                    let idx = tz * 64 * 64 + ty * 64 + tx;
                    let p000 = *mmap.get_unchecked(idx);
                    let p100 = *mmap.get_unchecked(idx + 1);
                    let p010 = *mmap.get_unchecked(idx + 64);
                    let p110 = *mmap.get_unchecked(idx + 65);
                    let p001 = *mmap.get_unchecked(idx + 64 * 64);
                    let p101 = *mmap.get_unchecked(idx + 64 * 64 + 1);
                    let p011 = *mmap.get_unchecked(idx + 64 * 64 + 64);
                    let p111 = *mmap.get_unchecked(idx + 64 * 64 + 65);
                    let fx_q8 = ((posx >> 24) & 0xFF) as u32;
                    let fy_q8 = ((posy >> 24) & 0xFF) as u32;
                    let fz_q8 = ((posz >> 24) & 0xFF) as u32;
                    let v = trilerp_q8(
                        fx_q8,
                        fy_q8,
                        fz_q8,
                        [p000, p100, p010, p110, p001, p101, p011, p111],
                    );
                    if !sink(v) {
                        stopped = true;
                        break;
                    }
                    consumed += 1;
                    posx = posx.wrapping_add(dx_q);
                    posy = posy.wrapping_add(dy_q);
                    posz = posz.wrapping_add(dz_q);
                }
            }
            const INV_Q32: f64 = 1.0 / 4294967296.0;
            px = posx as f64 * INV_Q32;
            py = posy as f64 * INV_Q32;
            pz = posz as f64 * INV_Q32;
            remaining -= consumed;
            if stopped {
                return;
            }
        }
    }
}

impl UnifiedVolume {
    /// Trilinear interpolation specialized over the cache's 64³ chunks.
    ///
    /// Mirrors the fast/slow split in
    /// `volume64x4::VolumeGrid64x4Mapped::get_interpolated` and
    /// `vesuvius_zarr::ZarrContext::get_interpolated`, with one extra
    /// wrinkle for the cache's LOD pyramid: when the target chunk isn't
    /// resident and a coarser parent is used instead, the **interpolation
    /// lattice shifts to the chosen LOD's coordinate space**. Otherwise
    /// `target_sx` and `target_sx + 1` both map onto the same coarse voxel
    /// (since `floor((target_sx + 1) / 2^shift) == floor(target_sx /
    /// 2^shift)` for most `target_sx`), the 8 corners collapse to one
    /// value, and the output bands instead of smoothly interpolating.
    fn interpolate_u8(&self, xyz: [f64; 3], downsampling: i32) -> u8 {
        let target_lod = lod_for(downsampling.max(1) as u8);
        let max_lod = self.cache.max_lod();

        let target_sx = (xyz[0] as i64).max(0) as u64;
        let target_sy = (xyz[1] as i64).max(0) as u64;
        let target_sz = (xyz[2] as i64).max(0) as u64;
        if !self.sample_in_bounds(target_sx, target_sy, target_sz, target_lod) {
            return 0;
        }
        let target_cx = (target_sx / 64) as u32;
        let target_cy = (target_sy / 64) as u32;
        let target_cz = (target_sz / 64) as u32;

        // Trust-resident path (offline renderer): the ensure stage already
        // made every chunk this tile touches resident, so skip the
        // per-voxel chunk-state probe + DashMap lookup entirely and read
        // straight off the target-LOD shard mmap, like the composite path.
        if self.cache.assume_resident() {
            return self.interpolate_u8_trusting(xyz, target_lod, target_cx, target_cy, target_cz);
        }

        let key_t = ChunkKey::new(target_lod, target_cx, target_cy, target_cz);

        // Shard-slot fast path. If the target shard is mmapped *and* this
        // chunk is marked Resident, do the trilerp straight against the
        // shard's bytes. Only the in-chunk fast case runs here — the
        // +1-crosses-chunk boundary path uses `get`, which itself
        // fast-paths through the shard slot.
        if let Some(chunk_ptr) = self.shard_slot_chunk_slice(target_lod, target_cx, target_cy, target_cz) {
            // `lod_use == target_lod` for the shard slot path. trunc()
            // matches modf's truncate-toward-zero split in one
            // instruction instead of a libm call.
            let cx0_f = xyz[0].trunc();
            let cy0_f = xyz[1].trunc();
            let cz0_f = xyz[2].trunc();
            let (fx_frac, fy_frac, fz_frac) = (xyz[0] - cx0_f, xyz[1] - cy0_f, xyz[2] - cz0_f);
            let cx0 = (cx0_f as i64).max(0) as u64;
            let cy0 = (cy0_f as i64).max(0) as u64;
            let cz0 = (cz0_f as i64).max(0) as u64;
            let fast = (cx0 & 63) != 63 && (cy0 & 63) != 63 && (cz0 & 63) != 63;
            if fast {
                let tx = (cx0 & 63) as usize;
                let ty = (cy0 & 63) as usize;
                let tz = (cz0 & 63) as usize;
                let idx = tz * 64 * 64 + ty * 64 + tx;
                // SAFETY: chunk_ptr addresses a 64³ region inside the shard
                // mmap (see shard_slot_chunk_slice). idx + 65 + 64*65 ≤
                // CHUNK_VOXELS for any (tx,ty,tz) with each < 63.
                let ps: [u8; 8] = unsafe {
                    [
                        *chunk_ptr.add(idx),
                        *chunk_ptr.add(idx + 1),
                        *chunk_ptr.add(idx + 64),
                        *chunk_ptr.add(idx + 65),
                        *chunk_ptr.add(idx + 64 * 64),
                        *chunk_ptr.add(idx + 64 * 64 + 1),
                        *chunk_ptr.add(idx + 64 * 64 + 64),
                        *chunk_ptr.add(idx + 64 * 64 + 65),
                    ]
                };
                return trilerp_q8(frac_q8(fx_frac), frac_q8(fy_frac), frac_q8(fz_frac), ps);
            }
            // Fall through to the legacy path which handles the +1-corner
            // crossing into a neighbor chunk via recursive `get` calls.
        }

        // Resolve the LOD to interpolate at: reuse the chunk slot if it
        // still points at our target chunk, otherwise walk the pyramid.
        let cached: Option<(u8, Arc<ChunkState>)> = {
            let b = self.local.borrow();
            if b.target_key == Some(key_t) {
                b.chosen.clone()
            } else {
                None
            }
        };
        let from_slot = cached.is_some();
        let chosen = match cached {
            Some(c) => c,
            None => {
                let walk_lo = target_lod.min(max_lod);
                // Offline renderer disables the climb: stay at the target LOD
                // (ensure-stage pre-fetched it) instead of fetching coarse
                // chunks that are never rendered.
                let walk_hi = if self.cache.lod_climb_enabled() { max_lod } else { walk_lo };
                let mut found: Option<(u8, Arc<ChunkState>)> = None;
                for lod_try in walk_lo..=walk_hi {
                    let (lx, ly, lz) = coord_at_lod(target_sx, target_sy, target_sz, target_lod, lod_try);
                    let key = ChunkKey::new(lod_try, (lx / 64) as u32, (ly / 64) as u32, (lz / 64) as u32);
                    let s = self.cache.state_or_fetch(key);
                    if s.is_terminal() {
                        found = Some((lod_try, s));
                        break;
                    }
                }
                // Populate the shard slot only when we landed on a
                // Resident chunk at target_lod — that's when we know the
                // shard is mmapped. Calling peek_shard on a full miss
                // would acquire the per-LOD mutex per voxel during
                // streaming, serializing the worker pool.
                if let Some((lt, s)) = &found {
                    if *lt == target_lod && matches!(s.as_ref(), ChunkState::Resident { .. }) {
                        let (target_shard, _) = self.shard_decompose(key_t.x, key_t.y, key_t.z);
                        self.populate_shard_slot(target_lod, target_shard);
                    }
                }
                // Recorded once per (paint, target chunk): later samples in
                // this chunk come back through the hot slot above.
                if found.as_ref().map_or(true, |(lt, _)| *lt != target_lod) {
                    self.note_missing(key_t, found.as_ref().map(|(lt, _)| *lt));
                }
                match found {
                    Some(c) => c,
                    None => return 0,
                }
            }
        };
        let (lod_use, state) = chosen;

        // Compute the interpolation lattice in `lod_use` coordinates: dividing
        // the raw f64 inputs by `2^shift` gives fractional coarse-voxel
        // positions whose 8 surrounding corners are genuine neighbors in the
        // coarse mip, not duplicates of the same coarse voxel.
        let shift = lod_use - target_lod;
        let scale = (1u64 << shift) as f64;
        let lx = xyz[0] / scale;
        let ly = xyz[1] / scale;
        let lz = xyz[2] / scale;
        let cx0_f = lx.trunc();
        let cy0_f = ly.trunc();
        let cz0_f = lz.trunc();
        let (cdx, cdy, cdz) = (lx - cx0_f, ly - cy0_f, lz - cz0_f);
        let cx0 = (cx0_f as i64).max(0) as u64;
        let cy0 = (cy0_f as i64).max(0) as u64;
        let cz0 = (cz0_f as i64).max(0) as u64;

        let result = match state.as_resident() {
            Some(mmap) => {
                // Fast path: the 2×2×2 block at the chosen LOD lives inside
                // the chunk we just resolved. `target_sx / (64 << shift) ==
                // cx0 / 64` always (and likewise for y/z), so the boundary
                // check is purely about `cx0 + 1` crossing into the next
                // coarse chunk.
                let fast = (cx0 & 63) != 63 && (cy0 & 63) != 63 && (cz0 & 63) != 63;
                let ps: [u8; 8] = if fast {
                    let tx = (cx0 & 63) as usize;
                    let ty = (cy0 & 63) as usize;
                    let tz = (cz0 & 63) as usize;
                    let idx = tz * 64 * 64 + ty * 64 + tx;
                    [
                        mmap[idx],
                        mmap[idx + 1],
                        mmap[idx + 64],
                        mmap[idx + 65],
                        mmap[idx + 64 * 64],
                        mmap[idx + 64 * 64 + 1],
                        mmap[idx + 64 * 64 + 64],
                        mmap[idx + 64 * 64 + 65],
                    ]
                } else {
                    // +1 corner crosses into a neighboring coarse chunk.
                    // Sample each corner via `get` *at the chosen LOD*
                    // (downsampling = 1 << lod_use), so the per-corner
                    // pyramid walks start at `lod_use` instead of redoing
                    // target-LOD lookups that would just collapse onto
                    // duplicate coarse voxels again.
                    let ds = 1i32 << lod_use;
                    let cx0f = cx0 as f64;
                    let cy0f = cy0 as f64;
                    let cz0f = cz0 as f64;
                    [
                        self.get([cx0f, cy0f, cz0f], ds),
                        self.get([cx0f + 1.0, cy0f, cz0f], ds),
                        self.get([cx0f, cy0f + 1.0, cz0f], ds),
                        self.get([cx0f + 1.0, cy0f + 1.0, cz0f], ds),
                        self.get([cx0f, cy0f, cz0f + 1.0], ds),
                        self.get([cx0f + 1.0, cy0f, cz0f + 1.0], ds),
                        self.get([cx0f, cy0f + 1.0, cz0f + 1.0], ds),
                        self.get([cx0f + 1.0, cy0f + 1.0, cz0f + 1.0], ds),
                    ]
                };
                trilerp_q8(frac_q8(cdx), frac_q8(cdy), frac_q8(cdz), ps)
            }
            None => 0, // Empty
        };

        // Anchor the hot slot to our (target chunk → chosen LOD) binding,
        // but only when it changed: the streaming slow path re-enters
        // here once per sample, and an unconditional rewrite costs a
        // borrow_mut + Arc refcount round-trip per voxel. Nothing else
        // writes this slot (`get`/`resolve_chunk` only touch the
        // per-LOD shard slots), so a cached hit is still current.
        if !from_slot {
            let mut b = self.local.borrow_mut();
            b.target_key = Some(key_t);
            b.chosen = Some((lod_use, state));
        }
        result
    }

    /// Trust-resident variant of `interpolate_u8` (see `ChunkCache`'s
    /// `assume_resident` field). Reads straight off the target-LOD shard
    /// mmap with no chunk-state probe, no DashMap lookup, and no LOD climb,
    /// mirroring `composite_along_normal_inner`'s inner read. The offline
    /// renderer's ensure stage guarantees the bytes are present; anything
    /// un-arrived reads as zero from the sparse mmap.
    fn interpolate_u8_trusting(
        &self,
        xyz: [f64; 3],
        target_lod: u8,
        target_cx: u32,
        target_cy: u32,
        target_cz: u32,
    ) -> u8 {
        let cx0_f = xyz[0].trunc();
        let cy0_f = xyz[1].trunc();
        let cz0_f = xyz[2].trunc();
        let (fx, fy, fz) = (xyz[0] - cx0_f, xyz[1] - cy0_f, xyz[2] - cz0_f);
        let cx0 = (cx0_f as i64).max(0) as u64;
        let cy0 = (cy0_f as i64).max(0) as u64;
        let cz0 = (cz0_f as i64).max(0) as u64;

        // In-chunk fast case: the +1 trilerp corner stays inside the home
        // chunk, so all 8 taps come from one shard-slot read.
        let fast = (cx0 & 63) != 63 && (cy0 & 63) != 63 && (cz0 & 63) != 63;
        if fast {
            let (shard, in_shard_idx) = self.shard_decompose(target_cx, target_cy, target_cz);
            self.populate_shard_slot(target_lod, shard);
            let chunk_base = {
                let b = self.local.borrow();
                match b.shards.get(target_lod as usize).and_then(|s| s.as_ref()) {
                    // SAFETY: in_shard_idx * CHUNK_VOXELS + CHUNK_VOXELS ≤ shard mmap length.
                    Some(slot) if slot.shard == shard => unsafe {
                        slot.base.add((in_shard_idx as usize) * CHUNK_VOXELS)
                    },
                    _ => return 0,
                }
            };
            let tx = (cx0 & 63) as usize;
            let ty = (cy0 & 63) as usize;
            let tz = (cz0 & 63) as usize;
            let idx = tz * 64 * 64 + ty * 64 + tx;
            // SAFETY: idx + 64*64 + 65 ≤ CHUNK_VOXELS since each of tx/ty/tz < 63.
            let ps: [u8; 8] = unsafe {
                [
                    *chunk_base.add(idx),
                    *chunk_base.add(idx + 1),
                    *chunk_base.add(idx + 64),
                    *chunk_base.add(idx + 65),
                    *chunk_base.add(idx + 64 * 64),
                    *chunk_base.add(idx + 64 * 64 + 1),
                    *chunk_base.add(idx + 64 * 64 + 64),
                    *chunk_base.add(idx + 64 * 64 + 65),
                ]
            };
            return trilerp_q8(frac_q8(fx), frac_q8(fy), frac_q8(fz), ps);
        }

        // +1 corner crosses into a neighbor chunk — sample each of the 8
        // corners with its own trusting shard read.
        let ps = [
            self.read_voxel_trusting(cx0, cy0, cz0, target_lod),
            self.read_voxel_trusting(cx0 + 1, cy0, cz0, target_lod),
            self.read_voxel_trusting(cx0, cy0 + 1, cz0, target_lod),
            self.read_voxel_trusting(cx0 + 1, cy0 + 1, cz0, target_lod),
            self.read_voxel_trusting(cx0, cy0, cz0 + 1, target_lod),
            self.read_voxel_trusting(cx0 + 1, cy0, cz0 + 1, target_lod),
            self.read_voxel_trusting(cx0, cy0 + 1, cz0 + 1, target_lod),
            self.read_voxel_trusting(cx0 + 1, cy0 + 1, cz0 + 1, target_lod),
        ];
        trilerp_q8(frac_q8(fx), frac_q8(fy), frac_q8(fz), ps)
    }

    /// Single-voxel trusting read off the target-LOD shard mmap. Companion
    /// to `interpolate_u8_trusting` for the boundary corner case.
    #[inline]
    fn read_voxel_trusting(&self, sx: u64, sy: u64, sz: u64, target_lod: u8) -> u8 {
        if !self.sample_in_bounds(sx, sy, sz, target_lod) {
            return 0;
        }
        let cx = (sx / 64) as u32;
        let cy = (sy / 64) as u32;
        let cz = (sz / 64) as u32;
        let (shard, in_shard_idx) = self.shard_decompose(cx, cy, cz);
        self.populate_shard_slot(target_lod, shard);
        let b = self.local.borrow();
        match b.shards.get(target_lod as usize).and_then(|s| s.as_ref()) {
            Some(slot) if slot.shard == shard => {
                let off = ((sz & 63) as usize) * 64 * 64 + ((sy & 63) as usize) * 64 + (sx & 63) as usize;
                // SAFETY: in_shard_idx * CHUNK_VOXELS + off < shard mmap length.
                unsafe { *slot.base.add((in_shard_idx as usize) * CHUNK_VOXELS + off) }
            }
            _ => 0,
        }
    }

    /// Test support: convenience wrapper around `composite_along_normal`
    /// returning the per-sample max along the ray — exercises the same
    /// unswitch + monomorphized inner loop `ObjVolume::paint` reaches via
    /// the trait.
    #[cfg(test)]
    pub fn max_along_normal(&self, base: [f64; 3], dir: [f64; 3], w_lo: f64, w_hi: f64, downsampling: i32) -> u8 {
        let mut state = MaxCompositionState::new();
        let mut compositor = CompositorRef::Max(&mut state);
        // Drive the trait override so we exercise the same unswitch +
        // monomorphized inner-loop path callers reach via Volume.
        <Self as VoxelVolume>::composite_along_normal(
            self,
            base,
            dir,
            w_lo,
            w_hi,
            downsampling,
            &mut compositor,
        );
        state.result(0)
    }
}

/// How many integer-step advances `p, p+d, p+2d, …` stay strictly inside
/// `[lo, hi)`, counting the initial sample (which the caller has confirmed
/// is inside the range — i.e. `lo <= p < hi`).
///
/// For the in-chunk fast path: `lo = cx*64`, `hi = cx*64 + 63` (NOT 64 — the
/// `+1` corner of the trilerp must stay inside the chunk).
#[inline]
fn run_length_1d(p: f64, d: f64, lo: f64, hi: f64) -> usize {
    if d > 0.0 {
        // Stop one step before crossing `hi`. The largest k with
        // `p + k*d < hi` is `floor((hi - p - eps) / d)`. We use the
        // half-open `< hi` semantics directly with `floor`, taking care
        // that an exact landing on `hi` does NOT count.
        let q = (hi - p) / d;
        // `q` ≥ 0 since p < hi and d > 0 (caller guarantees boundary safety).
        let k = if q.fract() == 0.0 { q - 1.0 } else { q.floor() };
        if k < 0.0 {
            1
        } else {
            (k as usize).saturating_add(1)
        }
    } else if d < 0.0 {
        // Symmetric: stop one step before crossing `lo`. Largest k with
        // `p + k*d >= lo` is `floor((lo - p) / d)` (d<0 flips inequality).
        let q = (lo - p) / d;
        let k = q.floor();
        if k < 0.0 {
            1
        } else {
            (k as usize).saturating_add(1)
        }
    } else {
        usize::MAX
    }
}

/// A resolved chunk handle returned by `resolve_chunk`. Like the pointer
/// returned by `shard_slot_chunk_slice`, `chunk_ptr` points into the LOD's
/// hot shard slot mmap — it does NOT own a keepalive `Arc`. The slot (and
/// the mmap behind it) is only mutated through `&self` methods on this
/// same single-threaded volume, so the pointer is valid until the next
/// call that can repopulate a shard slot (`populate_shard_slot`,
/// `drop_hot_slot`, recursive `get`/`interpolate_u8`). Callers must read
/// the bytes before making any such call. Skipping the per-voxel
/// `Arc<Mmap>` clone matters: `get()` runs once per sampled voxel in the
/// PPM/Obj render paths.
struct BoundChunk {
    /// `lod_use - target_lod`. Zero in the common case (target chunk
    /// resident at the requested LOD); positive when a coarser parent was
    /// used as a fallback.
    shift: u8,
    chunk_ptr: *const u8,
}

impl BoundChunk {
    /// Materialize the chunk's 64³ bytes as a slice. See the struct docs
    /// for the validity window.
    unsafe fn mmap_slice(&self) -> &[u8] {
        std::slice::from_raw_parts(self.chunk_ptr, CHUNK_VOXELS)
    }
}

impl UnifiedVolume {
    /// Shard-based LOD climb. Every per-voxel decision (resident / empty /
    /// locked / missing) is one lock-free atomic byte read off the sidecar;
    /// the per-shard dispatched bits — reached through each LOD's hot
    /// shard slot — only disambiguate Missing into "fetch in flight" vs
    /// "never tried". The DashMap is consulted exactly once per chunk, on
    /// the first voxel that finds it never-dispatched, to kick off the
    /// fetch; subsequent voxels see the dispatched bit and skip the cache
    /// layer entirely.
    fn resolve_chunk(&self, target_lod: u8, max_lod: u8, cx: u32, cy: u32, cz: u32) -> Option<BoundChunk> {
        let walk_lo = target_lod.min(max_lod);
        // Offline renderer disables the climb (see `interpolate_u8`).
        let walk_hi = if self.cache.lod_climb_enabled() { max_lod } else { walk_lo };
        for lod_try in walk_lo..=walk_hi {
            let shift = lod_try - target_lod;
            let cx_try = cx >> shift;
            let cy_try = cy >> shift;
            let cz_try = cz >> shift;
            let lod_ix = lod_try as usize;
            if lod_ix >= SHARD_SLOTS_PER_LOD {
                continue;
            }
            // Out-of-grid at this LOD (shouldn't happen for in-extent
            // samples) — climb on.
            let Some(state) = self.sidecar_state(lod_try, cx_try, cy_try, cz_try) else {
                continue;
            };
            let (shard, in_shard_idx) = self.shard_decompose(cx_try, cy_try, cz_try);
            match state {
                STATE_RESIDENT => {
                    let chunk_ptr = self.with_shard_slot(lod_try, shard, |slot| {
                        // SAFETY: in_shard_idx * CHUNK_VOXELS + CHUNK_VOXELS
                        // ≤ shard mmap length.
                        unsafe { slot.base.add((in_shard_idx as usize) * CHUNK_VOXELS) }
                    });
                    match chunk_ptr {
                        Some(p) => {
                            if shift > 0 {
                                self.note_missing(ChunkKey::new(target_lod, cx, cy, cz), Some(lod_try));
                            }
                            return Some(BoundChunk { shift, chunk_ptr: p });
                        }
                        // Shard couldn't be opened (I/O error) — climb on.
                        None => continue,
                    }
                }
                STATE_EMPTY => return None,
                // A write or punch is mid-flight on this slot; not
                // readable yet, and something is already driving it.
                STATE_LOCKED => continue,
                _ /* STATE_MISSING */ => {
                    // First voxel to find this chunk never-dispatched kicks
                    // off the fetch. `dispatch_chunk` sets the dispatched
                    // bit, so siblings of this voxel never re-enter the
                    // DashMap. If the shard can't be opened, skip the
                    // dispatch too (`state_or_fetch` would fail the same
                    // way) and climb on.
                    //
                    // An already-dispatched chunk still goes through
                    // `state_or_fetch` once per paint: that is what keeps a
                    // still-wanted Pending fetch fresh in the queues (touch)
                    // and re-dispatches one whose fetch was cancelled or
                    // failed once its cooldown has passed. The bit alone
                    // can't tell those apart from "in flight" — it also
                    // outlives the fetch to mark the slot's preview as done.
                    let dispatched = self.with_shard_slot(lod_try, shard, |slot| slot.dispatched.get(in_shard_idx));
                    let key = ChunkKey::new(lod_try, cx_try, cy_try, cz_try);
                    match dispatched {
                        Some(false) => {
                            let _ = self.cache.state_or_fetch(key);
                        }
                        Some(true) => {
                            if self.local.borrow_mut().revisited.insert(key) {
                                let _ = self.cache.state_or_fetch(key);
                            }
                        }
                        None => {}
                    }
                    continue;
                }
            }
        }
        self.note_missing(ChunkKey::new(target_lod, cx, cy, cz), None);
        None
    }
}

/// Sample at a position whose `floor()` lands on the +63 row of exactly
/// **one** chunk axis — the +1 trilerp corner along that axis crosses into
/// a neighbor chunk. 4 of the 8 corners come from `home`, the other 4 from
/// `neigh` (or are 0 if `neigh` is None, i.e. an Empty neighbor).
///
/// Exactly one of `bx`, `by`, `bz` must be true; the caller asserts this.
/// The 4 non-crossed corners are safe to read from `home` because the
/// other two axes' `+1` stay below 64 (their boundary bits are false).
#[inline]
fn sample_boundary_1axis(
    home: &[u8],
    neigh: Option<&[u8]>,
    bx: bool,
    by: bool,
    bz: bool,
    txu: u64,
    tyu: u64,
    tzu: u64,
    px: f64,
    py: f64,
    pz: f64,
) -> u8 {
    debug_assert!((bx as u8 + by as u8 + bz as u8) == 1);
    let tx = (txu & 63) as usize;
    let ty = (tyu & 63) as usize;
    let tz = (tzu & 63) as usize;

    // SAFETY: `tx + (1-bx)`, `ty + (1-by)`, `tz + (1-bz)` are all ≤ 63
    // for the home reads; the neighbor reads use 0 on the crossed axis.
    // All indices stay inside their respective 64³ mmap.
    let (h0, h1, h2, h3, n0, n1, n2, n3) = unsafe {
        if bx {
            // Crossed: +x. Home rows at tx=63; neighbor rows at tx=0.
            let h = tz * 64 * 64 + ty * 64 + 63;
            let n = tz * 64 * 64 + ty * 64;
            (
                *home.get_unchecked(h),                  // p000 home[63, ty, tz]
                *home.get_unchecked(h + 64),             // p010 home[63, ty+1, tz]
                *home.get_unchecked(h + 64 * 64),        // p001 home[63, ty, tz+1]
                *home.get_unchecked(h + 64 * 64 + 64),   // p011 home[63, ty+1, tz+1]
                neigh.map(|s| *s.get_unchecked(n)).unwrap_or(0),                  // p100
                neigh.map(|s| *s.get_unchecked(n + 64)).unwrap_or(0),             // p110
                neigh.map(|s| *s.get_unchecked(n + 64 * 64)).unwrap_or(0),        // p101
                neigh.map(|s| *s.get_unchecked(n + 64 * 64 + 64)).unwrap_or(0),   // p111
            )
        } else if by {
            // Crossed: +y. Home rows at ty=63; neighbor rows at ty=0.
            let h = tz * 64 * 64 + 63 * 64 + tx;
            let n = tz * 64 * 64 + tx;
            (
                *home.get_unchecked(h),                  // p000 home[tx, 63, tz]
                *home.get_unchecked(h + 1),              // p100 home[tx+1, 63, tz]
                *home.get_unchecked(h + 64 * 64),        // p001 home[tx, 63, tz+1]
                *home.get_unchecked(h + 64 * 64 + 1),    // p101 home[tx+1, 63, tz+1]
                neigh.map(|s| *s.get_unchecked(n)).unwrap_or(0),              // p010
                neigh.map(|s| *s.get_unchecked(n + 1)).unwrap_or(0),          // p110
                neigh.map(|s| *s.get_unchecked(n + 64 * 64)).unwrap_or(0),    // p011
                neigh.map(|s| *s.get_unchecked(n + 64 * 64 + 1)).unwrap_or(0),// p111
            )
        } else {
            // Crossed: +z. Home rows at tz=63; neighbor rows at tz=0.
            let h = 63 * 64 * 64 + ty * 64 + tx;
            let n = ty * 64 + tx;
            (
                *home.get_unchecked(h),                  // p000 home[tx, ty, 63]
                *home.get_unchecked(h + 1),              // p100 home[tx+1, ty, 63]
                *home.get_unchecked(h + 64),             // p010 home[tx, ty+1, 63]
                *home.get_unchecked(h + 65),             // p110 home[tx+1, ty+1, 63]
                neigh.map(|s| *s.get_unchecked(n)).unwrap_or(0),         // p001
                neigh.map(|s| *s.get_unchecked(n + 1)).unwrap_or(0),     // p101
                neigh.map(|s| *s.get_unchecked(n + 64)).unwrap_or(0),    // p011
                neigh.map(|s| *s.get_unchecked(n + 65)).unwrap_or(0),    // p111
            )
        }
    };

    // Re-order into the canonical (p000, p100, p010, p110, p001, p101, p011, p111).
    // Above we built ordered groups per axis; map back here.
    let p: [u8; 8] = if bx {
        [h0, n0, h1, n1, h2, n2, h3, n3]
    } else if by {
        [h0, h1, n0, n1, h2, h3, n2, n3]
    } else {
        [h0, h1, h2, h3, n0, n1, n2, n3]
    };

    let fx = px - txu as f64;
    let fy = py - tyu as f64;
    let fz = pz - tzu as f64;
    trilerp_q8(frac_q8(fx), frac_q8(fy), frac_q8(fz), p)
}

/// Trilinear blend of 8 u8 corner samples ordered (p000, p100, p010, p110,
/// p001, p101, p011, p111), where bits encode (z, y, x) offsets of 0/1.
///
/// Weights are Q0.8 (`0..=256` represents `0..=1.0`). Output is u8 with
/// truncate-toward-zero semantics — matches the previous f64 `c as u8` for
/// non-negative blends, which is everything we get from a u8 corner field.
///
/// All intermediates fit in u32: the deepest nest is
/// `c0 * (256 - fz) + c1 * fz ≤ 65280 * 256 * 256 = 0xFF000000`, which is
/// just under `2^32`.
#[inline]
fn trilerp_q8(fx: u32, fy: u32, fz: u32, p: [u8; 8]) -> u8 {
    let nfx = 256 - fx;
    let nfy = 256 - fy;
    let nfz = 256 - fz;
    let c00 = (p[0] as u32) * nfx + (p[1] as u32) * fx;
    let c10 = (p[2] as u32) * nfx + (p[3] as u32) * fx;
    let c01 = (p[4] as u32) * nfx + (p[5] as u32) * fx;
    let c11 = (p[6] as u32) * nfx + (p[7] as u32) * fx;
    let c0 = c00 * nfy + c10 * fy;
    let c1 = c01 * nfy + c11 * fy;
    let c = c0 * nfz + c1 * fz;
    (c >> 24) as u8
}

/// Convert a fractional `[0, 1)` weight into Q0.8 (`0..=255`). Negative
/// inputs saturate to 0; inputs ≥ 1.0 saturate to 256 (still valid for
/// `trilerp_q8`).
#[inline]
fn frac_q8(f: f64) -> u32 {
    (f * 256.0) as u32
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
        (None, Some(ChunkState::Pending { .. })) => {
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

        // Per-paint filter LUT: `DrawingConfig::filter` re-evaluates
        // `filters_active()`, the quant bit-mask match, and an f32 divide
        // on every call — per pixel that dominates the inner loop. It's a
        // pure function of `value` for a fixed config, so tabulate it once.
        let filter_lut: [u8; 256] = std::array::from_fn(|i| config.filter(i as u8));

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

        // -------- Pass 0: which target tiles still need data? --------
        // Terminal tiles (Resident / Empty) need neither a dispatch nor a
        // coarse preview; gating pass 1 on this stops the steady-state
        // viewport from re-fetching the whole LOD pyramid every frame.
        let Some(t) = viewport_tiles(min_uc, max_uc, min_vc, max_vc, pc, target_lod) else {
            return;
        };
        let mut pending_tiles: Vec<(i32, i32)> = Vec::new();
        for tu in t.tile_u_lo..=t.tile_u_hi {
            for tv in t.tile_v_lo..=t.tile_v_hi {
                let mut chunk = [0i32; 3];
                chunk[u_coord] = tu;
                chunk[v_coord] = tv;
                chunk[plane_coord] = t.tile_pc;
                let key = ChunkKey::new(target_lod, chunk[0] as u32, chunk[1] as u32, chunk[2] as u32);
                let terminal = self.cache.peek(key).map(|s| s.is_terminal()).unwrap_or(false);
                if !terminal {
                    pending_tiles.push((tu, tv));
                }
            }
        }

        // -------- Pass 1: dispatch coarse → fine for pending tiles --------
        // Walking coarse → fine means the first submissions of a cold
        // viewport are the low-LOD preview chunks. The cache + downloader
        // queues are pure LIFO, so the most recently submitted (finest)
        // work pops first — but coarse-first submission still gets the
        // low-LOD chunks into flight before the worker pool can drain
        // them, so a quick preview shows up promptly while detail
        // streams in behind it.
        //
        // Only coarse ancestors of still-pending target tiles are
        // touched: target tiles themselves are dispatched by pass 2, and
        // ancestors of already-terminal tiles would be previews nobody
        // renders. In-flight coarse fetches whose target tiles have all
        // landed stop being re-touched here and age out of the queues.
        if !pending_tiles.is_empty() {
            let mut seen: std::collections::HashSet<(i32, i32)> = std::collections::HashSet::new();
            for lod in (target_lod + 1..=max_lod).rev() {
                let Some(tiles) = viewport_tiles(min_uc, max_uc, min_vc, max_vc, pc, lod) else {
                    continue;
                };
                let shift = lod - target_lod;
                seen.clear();
                for &(tu, tv) in &pending_tiles {
                    let ctu = tu >> shift;
                    let ctv = tv >> shift;
                    if !seen.insert((ctu, ctv)) {
                        continue;
                    }
                    let mut chunk = [0i32; 3];
                    chunk[u_coord] = ctu;
                    chunk[v_coord] = ctv;
                    chunk[plane_coord] = tiles.tile_pc;
                    let key = ChunkKey::new(lod, chunk[0] as u32, chunk[1] as u32, chunk[2] as u32);
                    let _ = self.cache.state_or_fetch(key);
                }
            }
        }

        // -------- Pass 2: render per target tile, picking best resident LOD --------
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

                if chosen.as_ref().map_or(true, |(l, _)| *l != target_lod) {
                    let mut chunk = [0u32; 3];
                    chunk[u_coord] = tu as u32;
                    chunk[v_coord] = tv as u32;
                    chunk[plane_coord] = t.tile_pc as u32;
                    self.note_missing(
                        ChunkKey::new(target_lod, chunk[0], chunk[1], chunk[2]),
                        chosen.as_ref().map(|(l, _)| *l),
                    );
                }

                let painted = if let Some((lod_use, state)) = chosen.as_ref() {
                    if let Some(mmap) = state.as_resident() {
                        let lod_use = *lod_use;
                        let chunk_world_use = 64i32 << lod_use;
                        let shift = lod_use - target_lod;
                        let parent_tu = tu >> shift;
                        let parent_tv = tv >> shift;
                        let parent_tpc = t.tile_pc >> shift;
                        let chunk_u_lo = parent_tu * chunk_world_use;
                        let chunk_v_lo = parent_tv * chunk_world_use;
                        let chunk_pc_lo = parent_tpc * chunk_world_use;
                        // `world - chunk_lo` is non-negative inside the tile's
                        // pixel rect, so `>> lod_use` matches `/ scale_use`
                        // without the per-pixel idiv.
                        let plane_sample = ((pc - chunk_pc_lo) >> lod_use) as usize;

                        // Per-axis strides replace the dynamic
                        // `s[u_coord]/s[v_coord]/s[plane_coord]` permutation
                        // inside the pixel loop: the chunk is z-major
                        // (`off = z*64*64 + y*64 + x`), so axis index i
                        // contributes a factor of 64^i.
                        const STRIDES: [usize; 3] = [1, 64, 64 * 64];
                        let stride_u = STRIDES[u_coord];
                        let stride_v = STRIDES[v_coord];
                        let row_base = plane_sample * STRIDES[plane_coord];

                        for v_px in v_px_lo..v_px_hi {
                            let world_v = min_vc + v_px * pzoom;
                            let sample_v = ((world_v - chunk_v_lo) >> lod_use) as usize;
                            let row = row_base + sample_v * stride_v;
                            for u_px in u_px_lo..u_px_hi {
                                let world_u = min_uc + u_px * pzoom;
                                let sample_u = ((world_u - chunk_u_lo) >> lod_use) as usize;
                                let value = filter_lut[mmap[row + sample_u * stride_u] as usize];
                                buffer.set_gray(u_px as usize, v_px as usize, value);
                            }
                        }
                        true
                    } else {
                        // Empty at this LOD — fill the rect with the filtered
                        // zero value so the user sees a clean "no data" cell
                        // instead of whatever the buffer happened to contain.
                        let zero = filter_lut[0];
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
