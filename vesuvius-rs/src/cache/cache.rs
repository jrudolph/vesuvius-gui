//! In-memory chunk cache + async dispatch executor.

use super::backfiller::{BackfillError, ChunkBackfiller};
use super::disk::DiskStore;
use super::state::{ChunkKey, ChunkState};
use dashmap::DashMap;
use std::path::PathBuf;
use std::sync::mpsc::{self, Sender};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

const COOLDOWN: Duration = Duration::from_secs(10);
const DEFAULT_WORKERS: usize = 4;

/// In-memory + on-disk cache, parameterized over a backfiller.
///
/// Cheap to clone (everything inside is Arc/handle-shaped).
pub struct ChunkCache {
    inner: Arc<Inner>,
}

struct Inner {
    map: DashMap<ChunkKey, Arc<ChunkState>>,
    disk: DiskStore,
    backfiller: Arc<dyn ChunkBackfiller>,
    job_tx: Sender<ChunkKey>,
}

impl ChunkCache {
    /// Build a cache rooted under `cache_root`, with the on-disk path
    /// `{cache_root}/unified/{volume_id}/`. Spawns `DEFAULT_WORKERS` worker
    /// threads to service backfill jobs.
    pub fn new(cache_root: impl Into<PathBuf>, backfiller: Arc<dyn ChunkBackfiller>) -> Self {
        let root = cache_root.into().join("unified").join(backfiller.volume_id());
        let _ = std::fs::create_dir_all(&root);
        Self::new_at(root, backfiller, DEFAULT_WORKERS)
    }

    /// Variant for tests / advanced callers: explicit on-disk root and worker
    /// count.
    pub fn new_at(root: PathBuf, backfiller: Arc<dyn ChunkBackfiller>, workers: usize) -> Self {
        let (job_tx, job_rx) = mpsc::channel::<ChunkKey>();
        let job_rx = Arc::new(std::sync::Mutex::new(job_rx));

        let inner = Arc::new(Inner {
            map: DashMap::new(),
            disk: DiskStore::new(root),
            backfiller,
            job_tx,
        });

        for i in 0..workers.max(1) {
            let inner = inner.clone();
            let rx = job_rx.clone();
            std::thread::Builder::new()
                .name(format!("vesuvius-cache-{}", i))
                .spawn(move || worker_loop(inner, rx))
                .expect("spawn cache worker");
        }

        Self { inner }
    }

    /// Hot path: get one byte from the volume at `(x, y, z)` in voxel
    /// coordinates at the given LOD. Returns 0 for chunks that aren't
    /// resident yet (and kicks off async fetch on first miss).
    pub fn voxel(&self, x: u32, y: u32, z: u32, lod: u8) -> u8 {
        let key = ChunkKey::new(lod, x / 64, y / 64, z / 64);
        let state = self.state_or_fetch(key);
        if let Some(mmap) = state.as_resident() {
            let off = ((z & 63) as usize) * 64 * 64 + ((y & 63) as usize) * 64 + (x & 63) as usize;
            mmap[off]
        } else {
            0
        }
    }

    /// Returns the current state for `key`. If `Missing` (i.e. never seen),
    /// the cache:
    ///   1. tries the disk first; on hit, transitions to `Resident`,
    ///   2. otherwise enqueues a fetch job and returns `Pending`.
    pub fn state_or_fetch(&self, key: ChunkKey) -> Arc<ChunkState> {
        // Quick read.
        if let Some(entry) = self.inner.map.get(&key) {
            let state = entry.clone();
            drop(entry);
            return self.maybe_retry(key, state);
        }

        // Miss: take the write side, double-check, then init.
        let entry = self
            .inner
            .map
            .entry(key)
            .or_insert_with(|| self.inner.load_or_dispatch(key));
        entry.clone()
    }

    /// Recheck cooldown timers. Mostly a no-op for `Resident` / `Pending`.
    fn maybe_retry(&self, key: ChunkKey, state: Arc<ChunkState>) -> Arc<ChunkState> {
        if let ChunkState::CooldownMiss { until } = state.as_ref() {
            if SystemTime::now() >= *until {
                let new_state = self.inner.load_or_dispatch(key);
                self.inner.map.insert(key, new_state.clone());
                return new_state;
            }
        }
        state
    }

    pub fn voxel_extent(&self) -> [u32; 3] {
        self.inner.backfiller.voxel_extent()
    }

    pub fn max_lod(&self) -> u8 {
        self.inner.backfiller.max_lod()
    }

    /// For tests: synchronously wait for `key` to leave the `Pending` state.
    pub fn wait_for(&self, key: ChunkKey, timeout: Duration) -> Arc<ChunkState> {
        let start = std::time::Instant::now();
        loop {
            let state = self.state_or_fetch(key);
            if !matches!(state.as_ref(), ChunkState::Pending) {
                return state;
            }
            if start.elapsed() >= timeout {
                return state;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

impl Clone for ChunkCache {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl Inner {
    /// Either load `key` from disk (and mark Resident) or dispatch a fetch
    /// job (and mark Pending). Either way returns the state to insert.
    fn load_or_dispatch(&self, key: ChunkKey) -> Arc<ChunkState> {
        if let Some(mmap) = self.disk.try_load(key) {
            return Arc::new(ChunkState::Resident(mmap));
        }
        // Out-of-bounds chunks short-circuit so we don't create a worker job
        // for every off-volume paint pixel.
        if self.is_out_of_bounds(key) {
            return Arc::new(ChunkState::CooldownMiss {
                until: SystemTime::now() + Duration::from_secs(60 * 60 * 24 * 365),
            });
        }
        let _ = self.job_tx.send(key);
        Arc::new(ChunkState::Pending)
    }

    fn is_out_of_bounds(&self, key: ChunkKey) -> bool {
        let extent = self.backfiller.voxel_extent();
        let scale = 1u64 << key.lod;
        let chunk_voxels = 64u64 * scale;
        let start = [
            key.x as u64 * chunk_voxels,
            key.y as u64 * chunk_voxels,
            key.z as u64 * chunk_voxels,
        ];
        start[0] >= extent[0] as u64
            || start[1] >= extent[1] as u64
            || start[2] >= extent[2] as u64
            || key.lod > self.backfiller.max_lod()
    }
}

fn worker_loop(inner: Arc<Inner>, rx: Arc<std::sync::Mutex<mpsc::Receiver<ChunkKey>>>) {
    loop {
        let key = {
            let guard = rx.lock().unwrap();
            match guard.recv() {
                Ok(k) => k,
                Err(_) => return, // senders dropped → cache gone
            }
        };

        let new_state = match inner.backfiller.fetch(key) {
            Ok(bytes) => match inner.disk.write_atomic(key, &bytes) {
                Ok(()) => match inner.disk.try_load(key) {
                    Some(mmap) => Arc::new(ChunkState::Resident(mmap)),
                    None => {
                        log::warn!("unified-cache: chunk {:?} written but mmap reload failed", key);
                        Arc::new(ChunkState::CooldownMiss {
                            until: SystemTime::now() + COOLDOWN,
                        })
                    }
                },
                Err(e) => {
                    log::warn!("unified-cache: disk write for {:?} failed: {}", key, e);
                    Arc::new(ChunkState::CooldownMiss {
                        until: SystemTime::now() + COOLDOWN,
                    })
                }
            },
            Err(BackfillError::OutOfBounds) => Arc::new(ChunkState::CooldownMiss {
                until: SystemTime::now() + Duration::from_secs(60 * 60 * 24 * 365),
            }),
            Err(BackfillError::Permanent(reason)) => {
                log::warn!("unified-cache: chunk {:?} permanently unavailable: {}", key, reason);
                Arc::new(ChunkState::CooldownMiss {
                    until: SystemTime::now() + Duration::from_secs(60 * 60 * 24 * 365),
                })
            }
            Err(BackfillError::Transient(reason)) => {
                log::debug!("unified-cache: chunk {:?} transient fetch error: {}", key, reason);
                Arc::new(ChunkState::CooldownMiss {
                    until: SystemTime::now() + COOLDOWN,
                })
            }
        };

        inner.map.insert(key, new_state);
    }
}
