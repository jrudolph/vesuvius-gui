//! In-memory chunk cache + plan-based async dispatch executor.
//!
//! Two-level scheduling:
//!   - **Chunks** are the unit the paint loop asks for (64³).
//!   - **Sources** are the unit backfillers actually fetch (one native zarr
//!     chunk, one HTTP GET, one decoded blob, …). A single source can be
//!     consumed by many chunks; the cache deduplicates so the fetch runs
//!     exactly once per source key.
//!
//! Flow for one chunk miss:
//!   1. `state_or_fetch` → `dispatch_chunk` → backfiller emits a `BackfillPlan`.
//!   2. The chunk is parked in `chunks` with a counter of unresolved sources.
//!   3. For each source: if first seen, queue a `FetchSource` task; otherwise
//!      attach the chunk as a waiter on the existing source.
//!   4. When a source resolves, every waiter chunk's counter drops by one;
//!      when a chunk's counter hits zero, an `Extract` task is queued.
//!   5. Extract runs the backfiller's closure → writes to disk → mmaps →
//!      transitions chunk state to `Resident`.
//!
//! ### Backpressure
//!
//! The task channel is intentionally tiny (`TASK_QUEUE_CAPACITY`). When the
//! pool can't accept new work we drop the task and revert the chunk to a
//! short cooldown — the paint loop is calling us at frame rate and will
//! re-request only the chunks that are *still in view*, so dropping stale
//! requests is the right policy. The alternative — letting the queue grow —
//! means the user pans away, and we keep on fetching for the *previous*
//! viewport while the current frame's chunks queue up behind hundreds of
//! stale ones.

use super::backfiller::{BackfillError, BackfillPlan, ChunkBackfiller, SourceOutcome, SourceSpec};
use super::disk::DiskStore;
use super::state::{ChunkKey, ChunkState};
use dashmap::DashMap;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

const COOLDOWN: Duration = Duration::from_secs(10);
const SHORT_COOLDOWN: Duration = Duration::from_millis(150);
const PERMANENT_COOLDOWN: Duration = Duration::from_secs(60 * 60 * 24 * 365);
const DEFAULT_WORKERS: usize = 4;
/// Hard cap on outstanding tasks. Combined with 4 workers, this means at most
/// ~8 in-flight tasks at any moment — keeps the cache responsive to the
/// freshest viewport request even under bursty paint loops.
const TASK_QUEUE_CAPACITY: usize = 4;

pub struct ChunkCache {
    inner: Arc<Inner>,
}

struct Inner {
    map: DashMap<ChunkKey, Arc<ChunkState>>,
    disk: DiskStore,
    backfiller: Arc<dyn ChunkBackfiller>,
    /// `Mutex<HashMap>` rather than `DashMap` so we can hold the lock across
    /// `try_send` — that closes the race where two threads register the
    /// same source concurrently and we'd otherwise lose one fetch.
    sources: Mutex<HashMap<String, Arc<Mutex<SourceState>>>>,
    chunks: DashMap<ChunkKey, Arc<Mutex<ChunkProgress>>>,
    task_tx: SyncSender<Task>,
}

enum SourceState {
    Pending { waiters: Vec<ChunkKey> },
    Done(SourceOutcome),
}

struct ChunkProgress {
    /// Order matches the original `BackfillPlan.sources` ordering — the
    /// extract closure receives outcomes in this order.
    order: Vec<String>,
    remaining: usize,
    results: HashMap<String, SourceOutcome>,
    extract: Option<Box<dyn FnOnce(&[SourceOutcome]) -> Result<Vec<u8>, BackfillError> + Send + 'static>>,
}

enum Task {
    FetchSource {
        key: String,
        fetch: Box<dyn FnOnce() -> SourceOutcome + Send + 'static>,
    },
    Extract(ChunkKey),
}

enum RegisterResult {
    /// First observer — `FetchSource` task was queued.
    Queued,
    /// An earlier observer's fetch is in-flight; we're now a waiter.
    AttachedPending,
    /// Source already resolved; outcome is returned for the caller to apply.
    AlreadyDone(SourceOutcome),
    /// Task queue is full; nothing was registered.
    QueueFull,
}

enum SatisfyResult {
    /// Either: progress updated and we're still waiting on other sources,
    /// or: all sources done and the Extract task was queued successfully,
    /// or: chunk progress was already evicted.
    Ok,
    /// All sources done but the Extract task couldn't be enqueued. Caller
    /// should put the chunk into a short cooldown so paint retries.
    QueueFullOnExtract,
}

impl ChunkCache {
    pub fn new(cache_root: impl Into<PathBuf>, backfiller: Arc<dyn ChunkBackfiller>) -> Self {
        let root = cache_root.into().join("unified").join(backfiller.volume_id());
        let _ = std::fs::create_dir_all(&root);
        Self::new_at(root, backfiller, DEFAULT_WORKERS)
    }

    pub fn new_at(root: PathBuf, backfiller: Arc<dyn ChunkBackfiller>, workers: usize) -> Self {
        let (task_tx, task_rx) = mpsc::sync_channel::<Task>(TASK_QUEUE_CAPACITY);
        let task_rx = Arc::new(Mutex::new(task_rx));

        let inner = Arc::new(Inner {
            map: DashMap::new(),
            disk: DiskStore::new(root),
            backfiller,
            sources: Mutex::new(HashMap::new()),
            chunks: DashMap::new(),
            task_tx,
        });

        for i in 0..workers.max(1) {
            let inner = inner.clone();
            let rx = task_rx.clone();
            std::thread::Builder::new()
                .name(format!("vesuvius-cache-{}", i))
                .spawn(move || worker_loop(inner, rx))
                .expect("spawn cache worker");
        }

        Self { inner }
    }

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

    pub fn state_or_fetch(&self, key: ChunkKey) -> Arc<ChunkState> {
        if let Some(entry) = self.inner.map.get(&key) {
            let state = entry.clone();
            drop(entry);
            return self.maybe_retry(key, state);
        }
        let entry = self
            .inner
            .map
            .entry(key)
            .or_insert_with(|| self.inner.dispatch_chunk(key));
        entry.clone()
    }

    fn maybe_retry(&self, key: ChunkKey, state: Arc<ChunkState>) -> Arc<ChunkState> {
        if let ChunkState::CooldownMiss { until } = state.as_ref() {
            if SystemTime::now() >= *until {
                let new_state = self.inner.dispatch_chunk(key);
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
        Self { inner: self.inner.clone() }
    }
}

fn long_cooldown() -> Arc<ChunkState> {
    Arc::new(ChunkState::CooldownMiss { until: SystemTime::now() + PERMANENT_COOLDOWN })
}
fn cooldown() -> Arc<ChunkState> {
    Arc::new(ChunkState::CooldownMiss { until: SystemTime::now() + COOLDOWN })
}
fn short_cooldown() -> Arc<ChunkState> {
    Arc::new(ChunkState::CooldownMiss { until: SystemTime::now() + SHORT_COOLDOWN })
}

impl Inner {
    /// Drive one chunk from Missing to Pending. Plans + registers sources;
    /// queues either FetchSource or (if all sources already Done) Extract.
    ///
    /// Returns the state to insert in `self.map`. Caller is `state_or_fetch`'s
    /// `or_insert_with` closure — we must NEVER call `self.map.insert(key, …)`
    /// for this same key from here (it would deadlock the shard lock that the
    /// entry guard is holding).
    fn dispatch_chunk(self: &Arc<Self>, key: ChunkKey) -> Arc<ChunkState> {
        if let Some(mmap) = self.disk.try_load(key) {
            log::trace!("cache: chunk {:?} hit disk", key);
            return Arc::new(ChunkState::Resident(mmap));
        }
        if self.is_out_of_bounds(key) {
            log::trace!("cache: chunk {:?} out of bounds", key);
            return long_cooldown();
        }

        log::debug!("cache: dispatching chunk {:?}", key);
        let plan = match self.backfiller.plan(key) {
            Ok(p) => p,
            Err(BackfillError::OutOfBounds) => return long_cooldown(),
            Err(BackfillError::Permanent(reason)) => {
                log::warn!("unified-cache: plan({:?}) permanent: {}", key, reason);
                return long_cooldown();
            }
            Err(BackfillError::Transient(reason)) => {
                log::debug!("unified-cache: plan({:?}) transient: {}", key, reason);
                return cooldown();
            }
        };

        let BackfillPlan { sources, extract } = plan;
        let order: Vec<String> = sources.iter().map(|s| s.key.clone()).collect();
        let progress_arc = Arc::new(Mutex::new(ChunkProgress {
            order: order.clone(),
            remaining: order.len(),
            results: HashMap::new(),
            extract: Some(extract),
        }));
        self.chunks.insert(key, progress_arc.clone());

        if order.is_empty() {
            // 0-source plan: queue Extract immediately.
            match self.task_tx.try_send(Task::Extract(key)) {
                Ok(()) => return Arc::new(ChunkState::Pending),
                Err(TrySendError::Full(_)) => {
                    log::trace!("cache: extract for {:?} dropped at dispatch (queue full)", key);
                    self.chunks.remove(&key);
                    return short_cooldown();
                }
                Err(TrySendError::Disconnected(_)) => {
                    self.chunks.remove(&key);
                    return short_cooldown();
                }
            }
        }

        // Register sources. Track immediate Dones so we can satisfy them
        // after the loop (avoids deep recursion through self.satisfy during
        // dispatch, and lets us bail on queue-full cleanly).
        let mut immediates: Vec<(String, SourceOutcome)> = Vec::new();
        for spec in sources {
            let source_key = spec.key.clone();
            match self.register_source(key, spec) {
                RegisterResult::Queued | RegisterResult::AttachedPending => {}
                RegisterResult::AlreadyDone(outcome) => immediates.push((source_key, outcome)),
                RegisterResult::QueueFull => {
                    log::trace!("cache: chunk {:?} aborted at dispatch (source queue full)", key);
                    self.chunks.remove(&key);
                    return short_cooldown();
                }
            }
        }

        // Apply immediates. If they push the chunk to Extract-ready, queue
        // Extract here. (Concurrent worker satisfies on the other sources
        // race-safely against this via `results.contains_key` + the
        // single-fire `remaining == 0` check inside the same lock.)
        for (sk, outcome) in immediates {
            match self.satisfy(key, &sk, outcome) {
                SatisfyResult::Ok => {}
                SatisfyResult::QueueFullOnExtract => {
                    // We can't `self.map.insert(key, short_cooldown())` here:
                    // we're inside our own `or_insert_with`. Returning the
                    // cooldown state has the same effect.
                    log::trace!("cache: chunk {:?} extract dropped at dispatch", key);
                    return short_cooldown();
                }
            }
        }
        Arc::new(ChunkState::Pending)
    }

    fn register_source(self: &Arc<Self>, chunk_key: ChunkKey, spec: SourceSpec) -> RegisterResult {
        let SourceSpec { key: source_key, fetch } = spec;
        let mut sources = self.sources.lock().unwrap();
        if let Some(arc) = sources.get(&source_key) {
            let arc = arc.clone();
            drop(sources);
            // Don't run another fetch; the first observer's is authoritative.
            drop(fetch);
            let mut s = arc.lock().unwrap();
            match &mut *s {
                SourceState::Pending { waiters } => {
                    waiters.push(chunk_key);
                    log::trace!(
                        "cache: source {} dedup-pending (chunk {:?} attached)",
                        source_key,
                        chunk_key
                    );
                    RegisterResult::AttachedPending
                }
                SourceState::Done(outcome) => {
                    log::trace!(
                        "cache: source {} dedup-done (chunk {:?} → immediate satisfy)",
                        source_key,
                        chunk_key
                    );
                    RegisterResult::AlreadyDone(outcome.clone())
                }
            }
        } else {
            // Fresh source: try to enqueue *while still holding the sources
            // lock*. That way the worker can't pop the task and look up an
            // entry that doesn't yet exist.
            match self.task_tx.try_send(Task::FetchSource {
                key: source_key.clone(),
                fetch,
            }) {
                Ok(()) => {
                    let arc = Arc::new(Mutex::new(SourceState::Pending {
                        waiters: vec![chunk_key],
                    }));
                    sources.insert(source_key.clone(), arc);
                    drop(sources);
                    log::trace!("cache: source {} queued (chunk {:?})", source_key, chunk_key);
                    RegisterResult::Queued
                }
                Err(_) => {
                    drop(sources);
                    log::trace!("cache: source {} dropped (queue full)", source_key);
                    RegisterResult::QueueFull
                }
            }
        }
    }

    fn fetch_source(
        self: &Arc<Self>,
        source_key: String,
        fetch: Box<dyn FnOnce() -> SourceOutcome + Send + 'static>,
    ) {
        let t0 = std::time::Instant::now();
        let outcome = fetch();

        let arc = {
            let sources = self.sources.lock().unwrap();
            match sources.get(&source_key) {
                Some(a) => a.clone(),
                None => {
                    log::trace!("cache: source {} evicted before completion", source_key);
                    return;
                }
            }
        };

        let waiters = {
            let mut s = arc.lock().unwrap();
            let prev = std::mem::replace(&mut *s, SourceState::Done(outcome.clone()));
            match prev {
                SourceState::Pending { waiters } => waiters,
                SourceState::Done(_) => return,
            }
        };

        let outcome_label = match &outcome {
            Ok(Some(_)) => "ok",
            Ok(None) => "absent",
            Err(e) => match e {
                BackfillError::Transient(_) => "transient",
                BackfillError::Permanent(_) => "permanent",
                BackfillError::OutOfBounds => "oob",
            },
        };
        log::debug!(
            "cache: source {} resolved [{}] in {:?}, notifying {} waiter(s)",
            source_key,
            outcome_label,
            t0.elapsed(),
            waiters.len()
        );

        // Evict errored entries so future plans retry instead of permanently
        // inheriting the failure.
        if outcome.is_err() {
            self.sources.lock().unwrap().remove(&source_key);
        }

        for w in waiters {
            if matches!(
                self.satisfy(w, &source_key, outcome.clone()),
                SatisfyResult::QueueFullOnExtract
            ) {
                // Worker-side path: safe to mutate self.map directly.
                self.map.insert(w, short_cooldown());
            }
        }
    }

    /// Apply one source's outcome to the chunk's progress. Returns
    /// `QueueFullOnExtract` only when all sources are now resolved but
    /// the resulting Extract task couldn't be enqueued; caller must then
    /// move the chunk to a short cooldown.
    fn satisfy(self: &Arc<Self>, chunk_key: ChunkKey, source_key: &str, outcome: SourceOutcome) -> SatisfyResult {
        let arc = match self.chunks.get(&chunk_key).map(|e| e.clone()) {
            Some(a) => a,
            None => return SatisfyResult::Ok,
        };
        let queue_extract = {
            let mut p = arc.lock().unwrap();
            if p.results.contains_key(source_key) {
                false
            } else {
                p.results.insert(source_key.to_string(), outcome);
                p.remaining = p.remaining.saturating_sub(1);
                p.remaining == 0
            }
        };
        if queue_extract {
            match self.task_tx.try_send(Task::Extract(chunk_key)) {
                Ok(()) => SatisfyResult::Ok,
                Err(_) => {
                    log::debug!("cache: extract for {:?} dropped (queue full)", chunk_key);
                    self.chunks.remove(&chunk_key);
                    SatisfyResult::QueueFullOnExtract
                }
            }
        } else {
            SatisfyResult::Ok
        }
    }

    fn extract_chunk(self: &Arc<Self>, key: ChunkKey) {
        let t0 = std::time::Instant::now();
        log::trace!("cache: extract start for chunk {:?}", key);
        let (_, arc) = match self.chunks.remove(&key) {
            Some(v) => v,
            None => return,
        };
        let (order, results, extract) = {
            let mut p = arc.lock().unwrap();
            let extract = match p.extract.take() {
                Some(e) => e,
                None => return,
            };
            let order = std::mem::take(&mut p.order);
            let results = std::mem::take(&mut p.results);
            (order, results, extract)
        };
        let mut inputs: Vec<SourceOutcome> = Vec::with_capacity(order.len());
        let mut results = results;
        for k in &order {
            inputs.push(results.remove(k).unwrap_or_else(|| Ok(None)));
        }

        let new_state = match extract(&inputs) {
            Ok(bytes) => match self.disk.write_atomic(key, &bytes) {
                Ok(()) => match self.disk.try_load(key) {
                    Some(mmap) => {
                        log::debug!("cache: chunk {:?} resident after {:?}", key, t0.elapsed());
                        Arc::new(ChunkState::Resident(mmap))
                    }
                    None => {
                        log::warn!("unified-cache: chunk {:?} written but mmap reload failed", key);
                        cooldown()
                    }
                },
                Err(e) => {
                    log::warn!("unified-cache: disk write for {:?} failed: {}", key, e);
                    cooldown()
                }
            },
            Err(BackfillError::OutOfBounds) => long_cooldown(),
            Err(BackfillError::Permanent(reason)) => {
                log::warn!("unified-cache: chunk {:?} permanently unavailable: {}", key, reason);
                long_cooldown()
            }
            Err(BackfillError::Transient(reason)) => {
                log::debug!("unified-cache: chunk {:?} transient extract error: {}", key, reason);
                cooldown()
            }
        };
        self.map.insert(key, new_state);
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

fn worker_loop(inner: Arc<Inner>, rx: Arc<Mutex<mpsc::Receiver<Task>>>) {
    loop {
        let task = {
            let guard = rx.lock().unwrap();
            match guard.recv() {
                Ok(t) => t,
                Err(_) => return,
            }
        };
        match task {
            Task::FetchSource { key, fetch } => inner.fetch_source(key, fetch),
            Task::Extract(key) => inner.extract_chunk(key),
        }
    }
}
