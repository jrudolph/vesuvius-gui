//! Decoded-chunk store: process memory, no persistence.
//!
//! Layout: per LOD, a 3D grid of shards, each an anonymous [`Region`]
//! holding `SHARD_CHUNKS_PER_AXIS³` 64³ chunk slots (64 GiB of reserved
//! address space, committed only where chunks are written). A chunk's bytes
//! live at `in_shard_idx · CHUNK_VOXELS` inside its shard, so the volume's
//! hot path addresses chunks as `shard base + offset` exactly like the old
//! file-backed shards.
//!
//! Chunk state (Missing / Locked / Resident / Empty) and the access epoch
//! live in an anonymous [`Sidecar`]: writers fill the slot under the
//! per-slot LOCKED claim and publish RESIDENT with Release; readers that
//! Acquire-load RESIDENT see the full chunk. Eviction claims the slot,
//! releases its pages (`MADV_DONTNEED`), and demotes it to MISSING.
//!
//! The persistent copy of every chunk lives compressed in the segment log
//! (`seglog.rs`); this store only holds what the process is using.

use super::region::Region;
use super::sidecar::{Header, LodDims, Sidecar, STATE_EMPTY, STATE_LOCKED, STATE_MISSING, STATE_RESIDENT};
use super::state::ChunkKey;
use super::CHUNK_VOXELS;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

pub enum LoadOutcome {
    Resident { mmap: Arc<Region>, offset: usize },
    Empty,
    Missing,
}

/// Result of `SlotStore::write_atomic`: whether this call filled the slot
/// (caller records the fill for LRU accounting) or another thread had
/// already published a definitive state.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WriteOutcome {
    Wrote,
    AlreadyResident,
    AlreadyEmpty,
}

/// A shard's memory plus its per-chunk dispatched bits, for the volume's
/// per-voxel hot path.
pub struct ShardSnapshot {
    pub mmap: Arc<Region>,
    pub dispatched: Arc<DispatchedBits>,
}

pub type ShardCoord = (u32, u32, u32);

/// Per-shard "fetch dispatched (and preview synthesized) this session" bits,
/// one per chunk in `in_shard_idx` order.
pub struct DispatchedBits {
    words: Box<[AtomicU64]>,
}

impl DispatchedBits {
    fn new(num_chunks: u64) -> Self {
        let num_words = num_chunks.div_ceil(64) as usize;
        Self {
            words: (0..num_words).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    /// Relaxed: a dispatch-dedup hint; data visibility is ordered by the
    /// state bytes.
    #[inline]
    pub fn get(&self, in_shard_idx: u64) -> bool {
        let word = &self.words[(in_shard_idx / 64) as usize];
        (word.load(Ordering::Relaxed) >> (in_shard_idx % 64)) & 1 != 0
    }

    /// Set the bit; true iff this call transitioned it 0→1.
    pub fn set_if_clear(&self, in_shard_idx: u64) -> bool {
        let mask = 1u64 << (in_shard_idx % 64);
        let prev = self.words[(in_shard_idx / 64) as usize].fetch_or(mask, Ordering::Relaxed);
        prev & mask == 0
    }

    /// Clear the bit (eviction), so the next reader re-dispatches.
    pub fn clear(&self, in_shard_idx: u64) {
        let mask = 1u64 << (in_shard_idx % 64);
        self.words[(in_shard_idx / 64) as usize].fetch_and(!mask, Ordering::Relaxed);
    }
}

#[derive(Clone)]
struct Shard {
    mem: Arc<Region>,
    dispatched: Arc<DispatchedBits>,
}

struct LodSlot {
    dims: LodDims,
    shards: Mutex<HashMap<ShardCoord, Shard>>,
}

/// Default shard side in chunks: 64³ · 256 KiB = 64 GiB of address space
/// per shard (4096³ voxels).
pub(crate) const SHARD_CHUNKS_PER_AXIS: u32 = 64;

struct ResolvedKey {
    sidecar_idx: u64,
    shard: ShardCoord,
    in_shard_idx: u64,
}

pub struct SlotStore {
    shard_chunks_per_axis: u32,
    sidecar: Arc<Sidecar>,
    lods: Vec<LodSlot>,
}

impl SlotStore {

    pub(crate) fn new_with_shard_chunks_per_axis(
        volume_id: String,
        extent: [u32; 3],
        max_lod: u8,
        shard_chunks_per_axis: u32,
    ) -> Self {
        assert!(shard_chunks_per_axis.is_power_of_two(), "shard side must be a power of two");
        let sidecar = Arc::new(Sidecar::empty(Header::new(volume_id, extent, max_lod)));
        let lods = sidecar
            .header
            .lods
            .iter()
            .map(|d| LodSlot {
                dims: *d,
                shards: Mutex::new(HashMap::new()),
            })
            .collect();
        Self {
            shard_chunks_per_axis,
            sidecar,
            lods,
        }
    }

    pub fn load(&self, key: ChunkKey) -> LoadOutcome {
        let Some(r) = self.resolve(key) else {
            return LoadOutcome::Missing;
        };
        match self.sidecar.get_state(key.lod, r.sidecar_idx) {
            STATE_RESIDENT => {
                let shard = self.shard(key.lod, r.shard);
                LoadOutcome::Resident {
                    mmap: shard.mem,
                    offset: r.in_shard_idx as usize * CHUNK_VOXELS,
                }
            }
            STATE_EMPTY => LoadOutcome::Empty,
            // LOCKED: a peer write/evict is in flight on this slot.
            STATE_MISSING | STATE_LOCKED => LoadOutcome::Missing,
            other => panic!("[{}] unknown slot state {}", key, other),
        }
    }

    #[cfg(test)]
    pub fn try_load(&self, key: ChunkKey) -> Option<(Arc<Region>, usize)> {
        match self.load(key) {
            LoadOutcome::Resident { mmap, offset } => Some((mmap, offset)),
            _ => None,
        }
    }

    /// Fill `key`'s slot: CAS MISSING→LOCKED, copy, publish RESIDENT.
    pub fn write_atomic(&self, key: ChunkKey, bytes: &[u8]) -> WriteOutcome {
        assert_eq!(bytes.len(), CHUNK_VOXELS, "[{}] chunk of {} bytes", key, bytes.len());
        let r = self.resolve(key).unwrap_or_else(|| panic!("[{}] out of bounds", key));
        let shard = self.shard(key.lod, r.shard);
        loop {
            match self
                .sidecar
                .compare_exchange_state(key.lod, r.sidecar_idx, STATE_MISSING, STATE_LOCKED)
            {
                Ok(_) => break,
                Err(STATE_LOCKED) => std::thread::yield_now(),
                Err(STATE_RESIDENT) => return WriteOutcome::AlreadyResident,
                Err(STATE_EMPTY) => return WriteOutcome::AlreadyEmpty,
                Err(other) => panic!("[{}] unknown slot state {}", key, other),
            }
        }
        // SAFETY: we hold the slot's LOCKED claim.
        unsafe { shard.mem.write(r.in_shard_idx as usize * CHUNK_VOXELS, bytes) };
        self.sidecar.set_state(key.lod, r.sidecar_idx, STATE_RESIDENT);
        WriteOutcome::Wrote
    }

    /// Best-effort preview (upscale from a coarser LOD): written only if the
    /// slot is MISSING and not contended, and left MISSING, so the real
    /// fetch overwrites it.
    pub fn write_unconfirmed(&self, key: ChunkKey, bytes: &[u8]) -> bool {
        assert_eq!(bytes.len(), CHUNK_VOXELS);
        let r = self.resolve(key).unwrap_or_else(|| panic!("[{}] out of bounds", key));
        let shard = self.shard(key.lod, r.shard);
        if self
            .sidecar
            .compare_exchange_state(key.lod, r.sidecar_idx, STATE_MISSING, STATE_LOCKED)
            .is_err()
        {
            return false;
        }
        // SAFETY: we hold the slot's LOCKED claim.
        unsafe { shard.mem.write(r.in_shard_idx as usize * CHUNK_VOXELS, bytes) };
        self.sidecar.set_state(key.lod, r.sidecar_idx, STATE_MISSING);
        true
    }

    pub fn mark_empty(&self, key: ChunkKey) {
        let r = self.resolve(key).unwrap_or_else(|| panic!("[{}] out of bounds", key));
        self.sidecar.set_state(key.lod, r.sidecar_idx, STATE_EMPTY);
    }

    /// Set `key`'s dispatched bit; true iff it was clear.
    pub fn mark_dispatched(&self, key: ChunkKey) -> bool {
        let r = self.resolve(key).unwrap_or_else(|| panic!("[{}] out of bounds", key));
        self.shard(key.lod, r.shard).dispatched.set_if_clear(r.in_shard_idx)
    }

    pub fn shard_chunks_per_axis(&self) -> u32 {
        self.shard_chunks_per_axis
    }

    /// The shard if it has been created, without creating it.
    pub fn peek_shard(&self, lod: u8, shard: ShardCoord) -> Option<ShardSnapshot> {
        let slot = self.lods.get(lod as usize)?;
        slot.shards.lock().unwrap().get(&shard).map(|s| ShardSnapshot {
            mmap: s.mem.clone(),
            dispatched: s.dispatched.clone(),
        })
    }

    /// The shard, created on demand; `None` outside the LOD's chunk grid.
    pub fn ensure_shard_open(&self, lod: u8, shard: ShardCoord) -> Option<ShardSnapshot> {
        let slot = self.lods.get(lod as usize)?;
        let sca = self.shard_chunks_per_axis as u64;
        if (shard.0 as u64) * sca >= slot.dims.nx as u64
            || (shard.1 as u64) * sca >= slot.dims.ny as u64
            || (shard.2 as u64) * sca >= slot.dims.nz as u64
        {
            return None;
        }
        let s = self.shard(lod, shard);
        Some(ShardSnapshot {
            mmap: s.mem,
            dispatched: s.dispatched,
        })
    }

    /// The memory backing `key`'s slot, whatever its state.
    pub fn slot_memory(&self, key: ChunkKey) -> (Arc<Region>, usize) {
        let r = self.resolve(key).unwrap_or_else(|| panic!("[{}] out of bounds", key));
        (self.shard(key.lod, r.shard).mem, r.in_shard_idx as usize * CHUNK_VOXELS)
    }

    /// `(shard, in_shard_idx)` of `key`; `None` out of bounds.
    pub fn locate(&self, key: ChunkKey) -> Option<(ShardCoord, u64)> {
        self.resolve(key).map(|r| (r.shard, r.in_shard_idx))
    }

    pub fn sidecar(&self) -> Arc<Sidecar> {
        self.sidecar.clone()
    }

    pub fn get_access_epoch(&self, key: ChunkKey) -> Option<u8> {
        let r = self.resolve(key)?;
        Some(self.sidecar.get_access_epoch(key.lod, r.sidecar_idx))
    }

    pub fn set_access_epoch(&self, key: ChunkKey, epoch: u8) {
        if let Some(r) = self.resolve(key) {
            self.sidecar.set_access_epoch(key.lod, r.sidecar_idx, epoch);
        }
    }

    pub fn cas_access_epoch(&self, key: ChunkKey, current: u8, new: u8) -> Option<Result<u8, u8>> {
        let r = self.resolve(key)?;
        Some(
            self.sidecar
                .compare_exchange_access_epoch(key.lod, r.sidecar_idx, current, new),
        )
    }

    /// Release the memory of `n_chunks` consecutive slots of one shard.
    /// Callers hold the LOCKED claim of every slot in the run.
    pub fn release_run(&self, lod: u8, shard: ShardCoord, start_in_shard_idx: u64, n_chunks: u64) {
        let Some(s) = self.lods[lod as usize].shards.lock().unwrap().get(&shard).cloned() else {
            return;
        };
        // SAFETY: the caller holds the slots' LOCKED claims.
        unsafe {
            s.mem.release(
                start_in_shard_idx as usize * CHUNK_VOXELS,
                n_chunks as usize * CHUNK_VOXELS,
            )
        };
    }

    fn resolve(&self, key: ChunkKey) -> Option<ResolvedKey> {
        let slot = self.lods.get(key.lod as usize)?;
        let sidecar_idx = slot.dims.linear_index(key.x, key.y, key.z)?;
        let sca = self.shard_chunks_per_axis;
        let s = sca as u64;
        let in_shard_idx = ((key.z % sca) as u64 * s + (key.y % sca) as u64) * s + (key.x % sca) as u64;
        Some(ResolvedKey {
            sidecar_idx,
            shard: (key.x / sca, key.y / sca, key.z / sca),
            in_shard_idx,
        })
    }

    fn shard(&self, lod: u8, coord: ShardCoord) -> Shard {
        let mut shards = self.lods[lod as usize].shards.lock().unwrap();
        shards
            .entry(coord)
            .or_insert_with(|| {
                let n = self.shard_chunks_per_axis as u64;
                Shard {
                    mem: Arc::new(Region::reserve((n * n * n) as usize * CHUNK_VOXELS)),
                    dispatched: Arc::new(DispatchedBits::new(n * n * n)),
                }
            })
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunk(seed: u8) -> Vec<u8> {
        (0..CHUNK_VOXELS).map(|i| ((i as u8) ^ seed).wrapping_mul(31)).collect()
    }

    #[test]
    fn write_then_load_roundtrip() {
        let store = SlotStore::new_with_shard_chunks_per_axis("v".into(), [128, 128, 128], 0, SHARD_CHUNKS_PER_AXIS);
        let key = ChunkKey::new(0, 1, 0, 1);
        assert!(matches!(store.load(key), LoadOutcome::Missing));
        let bytes = make_chunk(0x42);
        assert_eq!(store.write_atomic(key, &bytes), WriteOutcome::Wrote);
        assert_eq!(store.write_atomic(key, &bytes), WriteOutcome::AlreadyResident);
        let (mem, off) = store.try_load(key).expect("resident after write");
        assert_eq!(&mem[off..off + CHUNK_VOXELS], &bytes[..]);
    }

    #[test]
    fn multi_lod_and_cross_shard_offsets_disjoint() {
        let store = SlotStore::new_with_shard_chunks_per_axis("v".into(), [1024, 256, 256], 2, 4);
        let mut written = Vec::new();
        for lod in 0..=2u8 {
            for cx in 0..(16u32 >> lod) {
                let key = ChunkKey::new(lod, cx, 0, 0);
                let bytes = make_chunk(lod * 16 + cx as u8);
                store.write_atomic(key, &bytes);
                written.push((key, bytes));
            }
        }
        for (key, bytes) in written {
            let (mem, off) = store.try_load(key).unwrap();
            assert_eq!(&mem[off..off + CHUNK_VOXELS], &bytes[..], "{}", key);
        }
    }

    #[test]
    fn release_zeroes_and_demotes() {
        let store = SlotStore::new_with_shard_chunks_per_axis("v".into(), [128, 64, 64], 0, SHARD_CHUNKS_PER_AXIS);
        let key = ChunkKey::new(0, 1, 0, 0);
        store.write_atomic(key, &make_chunk(1));
        let (shard, idx) = store.locate(key).unwrap();
        let sc = store.sidecar();
        sc.compare_exchange_state(0, 1, STATE_RESIDENT, STATE_LOCKED).unwrap();
        store.release_run(0, shard, idx, 1);
        sc.set_state(0, 1, STATE_MISSING);
        assert!(matches!(store.load(key), LoadOutcome::Missing));
        let snap = store.peek_shard(0, shard).unwrap();
        let off = idx as usize * CHUNK_VOXELS;
        assert!(snap.mmap[off..off + CHUNK_VOXELS].iter().all(|&b| b == 0));
    }

    #[test]
    fn concurrent_writers_distinct_chunks() {
        let store = Arc::new(SlotStore::new_with_shard_chunks_per_axis("v".into(), [2048, 64, 1024], 0, SHARD_CHUNKS_PER_AXIS));
        let handles: Vec<_> = (0..4u8)
            .map(|t| {
                let store = store.clone();
                std::thread::spawn(move || {
                    for i in 0..16u32 {
                        let mut bytes = make_chunk(t);
                        bytes[0] = i as u8;
                        store.write_atomic(ChunkKey::new(0, t as u32, 0, i), &bytes);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        for t in 0..4u8 {
            for i in 0..16u32 {
                let (mem, off) = store.try_load(ChunkKey::new(0, t as u32, 0, i)).unwrap();
                assert_eq!(mem[off], i as u8);
                assert_eq!(mem[off + 1], (1u8 ^ t).wrapping_mul(31));
            }
        }
    }
}
