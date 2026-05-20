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
//!   1. `state_or_fetch_with_priority` → `dispatch_chunk` → backfiller emits
//!      a `BackfillPlan`.
//!   2. The chunk is parked in `chunks` with a counter of unresolved sources.
//!   3. For each source: if first seen, queue a `FetchSource` task (Compute)
//!      or hand the URL to the downloader (Download); otherwise attach the
//!      chunk as a waiter on the existing source.
//!   4. When a source resolves, every waiter chunk's counter drops by one;
//!      when a chunk's counter hits zero, an `Extract` task is queued.
//!   5. Extract runs the backfiller's closure → writes to disk → mmaps →
//!      transitions chunk state to `Resident`.
//!
//! ### Priority + age pruning
//!
//! Tasks live in a `BTreeMap` keyed by `(priority, seq)`. The paint loop
//! computes a per-chunk priority from its local viewport (coarse LOD first,
//! then closest-to-center) and passes it via
//! `state_or_fetch_with_priority`. Submission order within a frame is
//! priority order, so the BTreeMap naturally pops the most urgent work
//! across all panes first.
//!
//! The queue is **unbounded**: dedup happens upstream (cache's source map
//! ensures one source-key → one FetchSource enqueue; `satisfy` enqueues at
//! most one Extract per chunk). Workers prune at two points:
//!
//!   * **Age:** entries older than `MAX_AGE` are dropped + cancelled at pop.
//!   * **Already-met:** at pop, skip Extract for chunks that became
//!     Resident through another path, and FetchSource for sources that
//!     are already Done. Defensive against cooldown-retry races.

use super::backfiller::{BackfillError, BackfillPlan, ChunkBackfiller, SourceOutcome, SourcePayload, SourceSpec};
use super::disk::{DiskStore, LoadOutcome};
use super::downloader::{DownloadError, DownloadResult, Downloader, OnDone, SubmitResult};
use super::priority::{Priority, MAX_AGE};
use super::spill::SpillStore;
use super::state::{ChunkKey, ChunkState};
use dashmap::DashMap;
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime};

const COOLDOWN: Duration = Duration::from_secs(10);
const SHORT_COOLDOWN: Duration = Duration::from_millis(150);
const PERMANENT_COOLDOWN: Duration = Duration::from_secs(60 * 60 * 24 * 365);
/// Small worker pool — extract + decode is CPU-bound but lock-light. Keeping
/// the count low reduces the chance that a worker stalls behind a
/// DashMap shard another worker is holding.
const DEFAULT_WORKERS: usize = 2;

pub struct ChunkCache {
    inner: Arc<Inner>,
}

struct Inner {
    map: DashMap<ChunkKey, Arc<ChunkState>>,
    /// Chunks currently being dispatched. Acts as an atomic claim so two
    /// threads racing on the same key don't both run `dispatch_chunk`. We
    /// can't use `map.entry().or_insert_with` for this because the
    /// `or_insert_with` closure runs inside a shard-write lock, and
    /// `dispatch_chunk` synchronously triggers `complete_source` paths
    /// that may try to insert into the same shard — re-entrant DashMap
    /// access deadlocks.
    dispatching: DashMap<ChunkKey, ()>,
    disk: DiskStore,
    /// On-disk spill for downloaded source bytes. Sits between the
    /// downloader and Extract so the compressed payload doesn't live on
    /// the heap.
    spill: SpillStore,
    backfiller: Arc<dyn ChunkBackfiller>,
    /// `Mutex<HashMap>` rather than `DashMap` so claim-the-slot for a fresh
    /// source key is atomic. The lock is never held across `try_submit`
    /// (downloader/task queue) because submit can synchronously invoke
    /// callbacks that re-enter `complete_source` on the same thread.
    sources: Mutex<HashMap<String, Arc<Mutex<SourceState>>>>,
    chunks: DashMap<ChunkKey, Arc<Mutex<ChunkProgress>>>,
    /// Reverse index for `SourceSpec::Chunk` dependencies: when chunk `K`
    /// transitions to `Resident`, `publish_resident(K, …)` drains
    /// `pending_chunk_sources[K]` and completes each listed source key with
    /// `K`'s `Arc<ChunkState>` as the payload. Source-key entries are
    /// deduplicated by `register_source`'s Phase 1, so at most one entry per
    /// child chunk lands here — multiple parents waiting on the same child
    /// all attach as waiters on the same source state.
    pending_chunk_sources: DashMap<ChunkKey, Vec<String>>,
    task_queue: TaskQueue,
    downloader: Arc<Downloader>,
}

enum SourceState {
    Pending {
        waiters: Vec<ChunkKey>,
    },
    /// Source completed. `remaining_consumers` counts chunks that have
    /// claimed this source but haven't yet finished `extract_chunk`. When
    /// it reaches zero the source entry is evicted from
    /// `Inner::sources`, dropping its `SourcePayload` (typically an
    /// `Arc<Mmap>`) so the kernel reclaims the spilled file's pages and
    /// disk space.
    Done {
        outcome: SourceOutcome,
        remaining_consumers: usize,
    },
}

struct ChunkProgress {
    /// Order matches the original `BackfillPlan.sources` ordering — the
    /// extract closure receives outcomes in this order.
    order: Vec<String>,
    remaining: usize,
    results: HashMap<String, SourceOutcome>,
    extract: Option<Box<dyn FnOnce(&[SourceOutcome]) -> Result<Vec<u8>, BackfillError> + Send + 'static>>,
    /// Priority captured at dispatch time. Used to enqueue the `Extract`
    /// task once all sources resolve, so late-arriving completions inherit
    /// the original chunk priority instead of falling back to `worst`.
    priority: Priority,
}

enum Task {
    FetchSource {
        key: String,
        fetch: Box<dyn FnOnce() -> SourceOutcome + Send + 'static>,
    },
    Extract,
}

/// Priority queue for cache-side `Task`s. BTreeMap keyed by `(priority,
/// seq)` so workers always pop the most-urgent submitted entry. Unbounded;
/// dedup happens at the cache layer (source-key + chunk-key uniqueness).
/// The only staleness check is age — see the module docs.
struct TaskQueue {
    inner: Mutex<TaskQueueInner>,
    not_empty: Condvar,
    max_age: Duration,
}

struct TaskQueueInner {
    entries: BTreeMap<(u64, u64), TaskEntry>,
    next_seq: u64,
    closed: bool,
}

struct TaskEntry {
    chunk: ChunkKey,
    added_at: Instant,
    task: Task,
}

enum RegisterResult {
    /// First observer — fetch was queued (cache pool or downloader).
    Queued,
    /// An earlier observer's fetch is in-flight; we're now a waiter.
    AttachedPending,
    /// Source already resolved; outcome is returned for the caller to apply.
    AlreadyDone(SourceOutcome),
    /// Task / download queue is full; nothing was registered.
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
        Self::new_with_downloader(root, backfiller, workers, Arc::new(Downloader::new()))
    }

    pub fn new_with_downloader(
        root: PathBuf,
        backfiller: Arc<dyn ChunkBackfiller>,
        workers: usize,
        downloader: Arc<Downloader>,
    ) -> Self {
        let task_queue = TaskQueue::new(MAX_AGE);
        let spill_root = root.join("spill");
        let chunks_root = root;
        let _ = std::fs::create_dir_all(&spill_root);

        let inner = Arc::new(Inner {
            map: DashMap::new(),
            dispatching: DashMap::new(),
            disk: DiskStore::new(chunks_root),
            spill: SpillStore::new(spill_root),
            backfiller,
            sources: Mutex::new(HashMap::new()),
            chunks: DashMap::new(),
            pending_chunk_sources: DashMap::new(),
            task_queue,
            downloader,
        });

        for i in 0..workers.max(1) {
            let inner = inner.clone();
            std::thread::Builder::new()
                .name(format!("vesuvius-cache-{}", i))
                .spawn(move || worker_loop(inner))
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

    /// Best-effort dispatch with worst-case priority. Used by tests, by
    /// `get()`, and anywhere a caller doesn't have viewport context. Prefer
    /// `state_or_fetch_with_priority` from the paint loop.
    pub fn state_or_fetch(&self, key: ChunkKey) -> Arc<ChunkState> {
        self.state_or_fetch_with_priority(key, Priority::worst())
    }

    pub fn state_or_fetch_with_priority(&self, key: ChunkKey, priority: Priority) -> Arc<ChunkState> {
        self.inner.state_or_fetch_with_priority(key, priority)
    }

    /// Cheap state lookup without dispatching a fetch. Returns `None` if no
    /// entry exists for `key` yet. Useful for LOD-fallback paths that only
    /// want to render whatever is already resident.
    pub fn peek(&self, key: ChunkKey) -> Option<Arc<ChunkState>> {
        self.inner.map.get(&key).map(|e| e.clone())
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

/// RAII guard that releases a `dispatching` claim no matter how
/// `state_or_fetch_with_priority` returns — including panics.
struct DispatchGuard {
    inner: Arc<Inner>,
    key: ChunkKey,
}

impl Drop for DispatchGuard {
    fn drop(&mut self) {
        self.inner.dispatching.remove(&self.key);
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
    /// Same semantics as `ChunkCache::state_or_fetch_with_priority` but
    /// callable from inside cache internals (notably `register_source`'s
    /// Chunk-variant handler, which dispatches a child chunk synchronously
    /// without ever blocking a worker on its completion).
    fn state_or_fetch_with_priority(self: &Arc<Self>, key: ChunkKey, priority: Priority) -> Arc<ChunkState> {
        if let Some(entry) = self.map.get(&key) {
            let state = entry.clone();
            drop(entry);
            if let ChunkState::CooldownMiss { until } = state.as_ref() {
                if SystemTime::now() < *until {
                    return state;
                }
            } else {
                return state;
            }
        }

        let claimed = self.dispatching.insert(key, ()).is_none();
        if !claimed {
            return self
                .map
                .get(&key)
                .map(|e| e.clone())
                .unwrap_or_else(|| Arc::new(ChunkState::Pending));
        }

        let _guard = DispatchGuard { inner: self.clone(), key };
        let state = self.dispatch_chunk(key, priority);
        self.map.insert(key, state.clone());
        self.publish_terminal(key, &state);
        state
    }

    /// Called right after a chunk is written into `self.map`. If the new
    /// state is terminal (`Resident` or `Empty`), drain
    /// `pending_chunk_sources[key]` and complete each waiting source:
    ///   - `Resident` → payload is the chunk's `Arc<ChunkState>`.
    ///   - `Empty`    → `Ok(None)` (parent sees the child as absent).
    /// Source completion is idempotent (`complete_source` no-ops on a Done
    /// source), so double-fires from racing publishers are safe.
    fn publish_terminal(self: &Arc<Self>, key: ChunkKey, state: &Arc<ChunkState>) {
        let outcome: SourceOutcome = match state.as_ref() {
            ChunkState::Resident(_) => Ok(Some(state.clone() as SourcePayload)),
            ChunkState::Empty => Ok(None),
            _ => return,
        };
        let waiters: Vec<String> = match self.pending_chunk_sources.remove(&key) {
            Some((_, v)) => v,
            None => return,
        };
        for source_key in waiters {
            self.complete_source(source_key, outcome.clone());
        }
    }

    /// Drive one chunk from Missing to Pending. Plans + registers sources;
    /// queues either FetchSource / a download or (if all sources already
    /// Done) Extract.
    ///
    /// Returns the state to insert in `self.map`. Caller is the
    /// `or_insert_with` closure on the entry guard — we must NEVER call
    /// `self.map.insert(key, …)` for this same key from here.
    fn dispatch_chunk(self: &Arc<Self>, key: ChunkKey, priority: Priority) -> Arc<ChunkState> {
        match self.disk.load(key) {
            LoadOutcome::Resident(mmap) => {
                log::trace!("[{}] disk hit", key);
                return Arc::new(ChunkState::Resident(mmap));
            }
            LoadOutcome::Empty => {
                log::trace!("[{}] disk hit (empty)", key);
                return Arc::new(ChunkState::Empty);
            }
            LoadOutcome::Missing => {}
        }
        if self.is_out_of_bounds(key) {
            log::trace!("[{}] out of bounds", key);
            return long_cooldown();
        }

        let plan = match self.backfiller.plan(key) {
            Ok(p) => p,
            Err(BackfillError::OutOfBounds) => return long_cooldown(),
            Err(BackfillError::Permanent(reason)) => {
                log::warn!("[{}] permanent (plan): {}", key, reason);
                return long_cooldown();
            }
            Err(BackfillError::Transient(reason)) => {
                log::debug!("[{}] transient (plan): {}", key, reason);
                return cooldown();
            }
        };

        let BackfillPlan { sources, extract } = plan;
        let order: Vec<String> = sources.iter().map(|s| s.key().to_string()).collect();
        log::debug!("[{}] miss → fetching {} source(s)", key, order.len());
        let progress_arc = Arc::new(Mutex::new(ChunkProgress {
            order: order.clone(),
            remaining: order.len(),
            results: HashMap::new(),
            extract: Some(extract),
            priority,
        }));
        self.chunks.insert(key, progress_arc.clone());

        if order.is_empty() {
            // 0-source plan: queue Extract immediately.
            return match self.task_queue.try_submit(key, priority, Task::Extract) {
                Ok(()) => Arc::new(ChunkState::Pending),
                Err(dropped) => {
                    log::debug!("[{}] dropped: extract queue full (dispatch)", key);
                    self.chunks.remove(&key);
                    drop(dropped);
                    short_cooldown()
                }
            };
        }

        // Register sources. Track immediate Dones so we can satisfy them
        // after the loop (avoids deep recursion through self.satisfy during
        // dispatch, and lets us bail on queue-full cleanly).
        let mut immediates: Vec<(String, SourceOutcome)> = Vec::new();
        for spec in sources {
            let source_key = spec.key().to_string();
            match self.register_source(key, spec, priority) {
                RegisterResult::Queued | RegisterResult::AttachedPending => {}
                RegisterResult::AlreadyDone(outcome) => immediates.push((source_key, outcome)),
                RegisterResult::QueueFull => {
                    log::debug!("[{}] dropped: source queue full (dispatch)", key);
                    self.chunks.remove(&key);
                    return short_cooldown();
                }
            }
        }

        // Apply immediates. If they push the chunk to Extract-ready, queue
        // Extract here.
        for (sk, outcome) in immediates {
            match self.satisfy(key, &sk, outcome, priority) {
                SatisfyResult::Ok => {}
                SatisfyResult::QueueFullOnExtract => {
                    log::debug!("[{}] dropped: extract queue full (immediate)", key);
                    return short_cooldown();
                }
            }
        }
        Arc::new(ChunkState::Pending)
    }

    fn register_source(
        self: &Arc<Self>,
        chunk_key: ChunkKey,
        spec: SourceSpec,
        priority: Priority,
    ) -> RegisterResult {
        let source_key = spec.key().to_string();

        // Phase 1: under self.sources lock, either attach as a waiter on
        // an existing source or install a fresh `Pending` placeholder for
        // this source key.
        //
        // We MUST NOT call `downloader.try_submit` or `task_queue.try_submit`
        // while holding this lock. Both can fire callbacks synchronously
        // (downloader on queue-full / eviction). Those callbacks re-enter
        // `complete_source`, which locks `self.sources` again on the same
        // thread — a self-deadlock that wedged the cache after a few frames.
        enum Slot {
            Existing(Arc<Mutex<SourceState>>),
            Fresh,
        }
        let slot = {
            let mut sources = self.sources.lock().unwrap();
            if let Some(arc) = sources.get(&source_key) {
                Slot::Existing(arc.clone())
            } else {
                let arc = Arc::new(Mutex::new(SourceState::Pending {
                    waiters: vec![chunk_key],
                }));
                sources.insert(source_key.clone(), arc);
                Slot::Fresh
            }
        };

        if let Slot::Existing(arc) = slot {
            drop(spec); // first observer's fetch/url is authoritative
            let mut s = arc.lock().unwrap();
            return match &mut *s {
                SourceState::Pending { waiters } => {
                    waiters.push(chunk_key);
                    log::trace!("[{}] attach (chunk {})", source_key, chunk_key);
                    RegisterResult::AttachedPending
                }
                SourceState::Done {
                    outcome,
                    remaining_consumers,
                } => {
                    // New consumer for an already-completed source — bump
                    // the refcount so `extract_chunk`'s eviction logic
                    // waits for this chunk too.
                    *remaining_consumers += 1;
                    log::trace!(
                        "[{}] reuse (chunk {}, refcount {})",
                        source_key,
                        chunk_key,
                        remaining_consumers
                    );
                    RegisterResult::AlreadyDone(outcome.clone())
                }
            };
        }

        // Phase 2: we're the first observer and we own the placeholder slot.
        // Dispatch the fetch with no locks held.
        match spec {
            SourceSpec::Compute { key: _, fetch } => {
                match self.task_queue.try_submit(
                    chunk_key,
                    priority,
                    Task::FetchSource {
                        key: source_key.clone(),
                        fetch,
                    },
                ) {
                    Ok(()) => {
                        log::trace!("[{}] queued (chunk {})", source_key, chunk_key);
                        RegisterResult::Queued
                    }
                    Err(_dropped) => {
                        // Roll the placeholder forward to Done(Err) so any
                        // concurrent waiters (the only chunk that could have
                        // attached after we released the lock) get notified.
                        log::trace!("[{}] dropped: cache queue full", source_key);
                        self.complete_source(
                            source_key,
                            Err(BackfillError::Transient("cache queue full".into())),
                        );
                        RegisterResult::QueueFull
                    }
                }
            }
            SourceSpec::Chunk { key: _, chunk_key: child_key } => {
                // Register our interest BEFORE dispatching the child. If
                // dispatch happens to disk-hit and synchronously transition
                // the child to Resident, the publish_resident call inside
                // state_or_fetch_with_priority drains our entry and
                // completes the source right then. complete_source is
                // idempotent, so the post-dispatch check below double-firing
                // is harmless — the second call no-ops on the already-Done
                // source.
                self.pending_chunk_sources
                    .entry(child_key)
                    .or_insert_with(Vec::new)
                    .push(source_key.clone());

                let child_state = self.state_or_fetch_with_priority(child_key, priority);
                match child_state.as_ref() {
                    ChunkState::Resident(_) => {
                        // publish_terminal either fired during the dispatch
                        // call above or is about to via the disk path; in
                        // both cases the source is (or will be) Done.
                        log::trace!(
                            "[{}] chunk dep on {} resident (chunk {})",
                            source_key,
                            child_key,
                            chunk_key
                        );
                        RegisterResult::Queued
                    }
                    ChunkState::Empty | ChunkState::CooldownMiss { .. } => {
                        // Child is definitively absent or won't load — resolve
                        // our source as absent. (Empty also gets handled by
                        // publish_terminal, but we may have raced ahead of
                        // that; complete_source is idempotent.)
                        log::trace!(
                            "[{}] chunk dep on {} unresolvable (chunk {})",
                            source_key,
                            child_key,
                            chunk_key
                        );
                        self.complete_source(source_key.clone(), Ok(None));
                        RegisterResult::Queued
                    }
                    _ => {
                        // Pending / Missing — publish_terminal will satisfy
                        // us when the child reaches a terminal state.
                        log::trace!(
                            "[{}] chunk dep on {} pending (chunk {})",
                            source_key,
                            child_key,
                            chunk_key
                        );
                        RegisterResult::Queued
                    }
                }
            }
            SourceSpec::Download { key: _, url } => {
                let inner = self.clone();
                let key_for_done = source_key.clone();
                let on_done: OnDone = Box::new(move |result: DownloadResult| {
                    // Spill the bytes to disk as soon as they arrive so the
                    // payload is mmap-backed by the time Extract picks it
                    // up. This keeps the heap bounded even when many
                    // downloads complete faster than the (limited) cache
                    // workers can extract them.
                    let outcome: SourceOutcome = match result {
                        Ok(Some(bytes)) => match inner.spill.write_and_mmap(&key_for_done, &bytes) {
                            Ok(mmap) => Ok(Some(Arc::new(mmap) as SourcePayload)),
                            Err(e) => {
                                log::warn!("[{}] spill failed ({}); falling back to in-memory", key_for_done, e);
                                Ok(Some(Arc::new(bytes) as SourcePayload))
                            }
                        },
                        Ok(None) => Ok(None),
                        Err(DownloadError::Transient(s)) => {
                            Err(BackfillError::Transient(format!("download: {}", s)))
                        }
                    };
                    inner.complete_source(key_for_done, outcome);
                });

                // Submit without holding self.sources. The downloader may
                // synchronously fire `on_done` (queue full / eviction), which
                // calls complete_source — that path now safely re-locks
                // self.sources.
                match self.downloader.try_submit(&url, chunk_key, priority, on_done) {
                    SubmitResult::Submitted => {
                        log::trace!("[{}] submitted (chunk {})", source_key, chunk_key);
                        RegisterResult::Queued
                    }
                    SubmitResult::QueueFull => {
                        // The downloader already invoked our on_done with
                        // Err synchronously — complete_source has run and
                        // cleared the placeholder. Nothing more to do.
                        log::trace!("[{}] dropped: downloader queue full", source_key);
                        RegisterResult::QueueFull
                    }
                }
            }
        }
    }

    fn fetch_source(
        self: &Arc<Self>,
        source_key: String,
        fetch: Box<dyn FnOnce() -> SourceOutcome + Send + 'static>,
    ) {
        let outcome = fetch();
        self.complete_source(source_key, outcome);
    }

    /// Mark a source as `Done(outcome)` and notify every chunk currently
    /// waiting on it. Called from both the synchronous `FetchSource` worker
    /// path and the download callback path.
    fn complete_source(self: &Arc<Self>, source_key: String, outcome: SourceOutcome) {
        let arc = {
            let sources = self.sources.lock().unwrap();
            match sources.get(&source_key) {
                Some(a) => a.clone(),
                None => {
                    log::trace!("[{}] evicted before completion", source_key);
                    return;
                }
            }
        };

        let waiters = {
            let mut s = arc.lock().unwrap();
            let n = match &*s {
                SourceState::Pending { waiters } => waiters.len(),
                _ => 0,
            };
            let prev = std::mem::replace(
                &mut *s,
                SourceState::Done {
                    outcome: outcome.clone(),
                    remaining_consumers: n,
                },
            );
            match prev {
                SourceState::Pending { waiters } => waiters,
                SourceState::Done { .. } => return,
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
        log::trace!(
            "[{}] resolved {} → {} waiter(s)",
            source_key,
            outcome_label,
            waiters.len()
        );

        // Evict errored entries so future plans retry instead of permanently
        // inheriting the failure.
        if outcome.is_err() {
            self.sources.lock().unwrap().remove(&source_key);
        }

        for w in waiters {
            // Each waiter chunk's priority is the one captured when its
            // own dispatch_chunk ran. Look it up via the progress entry.
            let priority = self
                .chunks
                .get(&w)
                .map(|p| p.lock().unwrap().priority)
                .unwrap_or_else(Priority::worst);
            if matches!(
                self.satisfy(w, &source_key, outcome.clone(), priority),
                SatisfyResult::QueueFullOnExtract
            ) {
                self.map.insert(w, short_cooldown());
            }
        }
    }

    /// Apply one source's outcome to the chunk's progress. Returns
    /// `QueueFullOnExtract` only when all sources are now resolved but
    /// the resulting Extract task couldn't be enqueued; caller must then
    /// move the chunk to a short cooldown.
    fn satisfy(
        self: &Arc<Self>,
        chunk_key: ChunkKey,
        source_key: &str,
        outcome: SourceOutcome,
        priority: Priority,
    ) -> SatisfyResult {
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
            match self.task_queue.try_submit(chunk_key, priority, Task::Extract) {
                Ok(()) => SatisfyResult::Ok,
                Err(_dropped) => {
                    log::debug!("[{}] dropped: extract queue full", chunk_key);
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
        log::trace!("[{}] extract start", key);
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

        // All sources resolved to "definitively absent" (404/403 from the
        // downloader, or an explicit `Ok(None)` from a Compute fetch). The
        // chunk has no data anywhere — persist an `.empty` sentinel and
        // transition to `Empty` so future sessions hit the disk path
        // immediately without re-fetching, and so the LOD-fallback walk in
        // paint/get can stop here rather than serving stale coarser data.
        let all_absent = !inputs.is_empty() && inputs.iter().all(|o| matches!(o, Ok(None)));
        if all_absent {
            if let Err(e) = self.disk.mark_empty(key) {
                log::warn!("[{}] mark_empty failed ({}); empty still cached in-memory only", key, e);
            }
            log::debug!("[{}] empty ({:?})", key, t0.elapsed());
            let new_state = Arc::new(ChunkState::Empty);
            drop(inputs);
            self.release_sources(&order);
            self.map.insert(key, new_state.clone());
            self.publish_terminal(key, &new_state);
            return;
        }

        let new_state = match extract(&inputs) {
            Ok(bytes) => match self.disk.write_atomic(key, &bytes) {
                Ok(()) => match self.disk.try_load(key) {
                    Some(mmap) => {
                        log::debug!("[{}] ready ({:?})", key, t0.elapsed());
                        Arc::new(ChunkState::Resident(mmap))
                    }
                    None => {
                        log::warn!("[{}] write ok but mmap reload failed", key);
                        cooldown()
                    }
                },
                Err(e) => {
                    log::warn!("[{}] disk write failed: {}", key, e);
                    cooldown()
                }
            },
            Err(BackfillError::OutOfBounds) => long_cooldown(),
            Err(BackfillError::Permanent(reason)) => {
                log::warn!("[{}] permanent: {}", key, reason);
                long_cooldown()
            }
            Err(BackfillError::Transient(reason)) => {
                // Aged-out / cancelled fetches aren't a chunk failure — they
                // just mean the viewport moved on before the source landed.
                // Surface them as cancellations so they don't look like errors.
                if reason.contains("aged out") || reason.contains("queue closed") {
                    log::trace!("[{}] cancelled: {}", key, reason);
                } else {
                    log::debug!("[{}] transient: {}", key, reason);
                }
                cooldown()
            }
        };
        // Drop our inputs so the per-source payloads (mmaps in the
        // download path) only stay alive while the refcount-eviction loop
        // can see them — Arc clones inside the source entries remain.
        drop(inputs);
        self.release_sources(&order);
        self.map.insert(key, new_state.clone());
        self.publish_terminal(key, &new_state);
    }

    /// Decrement consumer refcounts for each source this chunk used. When a
    /// count hits zero we evict the source entry from `self.sources`; that
    /// drops its `Arc<Mmap>`, the kernel reclaims the spill file's pages,
    /// and (because the spill file was already unlinked) the disk space is
    /// freed too.
    fn release_sources(&self, order: &[String]) {
        if order.is_empty() {
            return;
        }
        let mut sources = self.sources.lock().unwrap();
        for source_key in order {
            let should_evict = if let Some(arc) = sources.get(source_key) {
                let mut s = arc.lock().unwrap();
                if let SourceState::Done { remaining_consumers, .. } = &mut *s {
                    *remaining_consumers = remaining_consumers.saturating_sub(1);
                    *remaining_consumers == 0
                } else {
                    false
                }
            } else {
                false
            };
            if should_evict {
                log::trace!("[{}] evict (last consumer)", source_key);
                sources.remove(source_key);
            }
        }
    }

    /// Handle a `TaskEntry` that the priority queue dropped (age / distance
    /// / eviction). Acts as if the task ran and failed transiently:
    /// `FetchSource` resolves with a Transient error so waiters back off;
    /// `Extract` just cleans up progress and reverts the chunk to cooldown.
    fn cancel_dropped_task(self: &Arc<Self>, entry: TaskEntry, reason: &str) {
        match entry.task {
            Task::FetchSource { key, fetch: _ } => {
                log::trace!("[{}] cancel: {}", key, reason);
                self.complete_source(key, Err(BackfillError::Transient(reason.into())));
            }
            Task::Extract => {
                let chunk_key = entry.chunk;
                log::debug!("[{}] dropped: {}", chunk_key, reason);
                self.chunks.remove(&chunk_key);
                self.map.insert(chunk_key, short_cooldown());
            }
        }
    }

    fn is_chunk_done(&self, key: ChunkKey) -> bool {
        self.map
            .get(&key)
            .map(|s| s.as_ref().is_terminal())
            .unwrap_or(false)
    }

    fn is_source_done(&self, source_key: &str) -> bool {
        let sources = self.sources.lock().unwrap();
        match sources.get(source_key) {
            Some(arc) => matches!(*arc.lock().unwrap(), SourceState::Done { .. }),
            None => false,
        }
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

impl TaskQueue {
    fn new(max_age: Duration) -> Self {
        Self {
            inner: Mutex::new(TaskQueueInner {
                entries: BTreeMap::new(),
                next_seq: 0,
                closed: false,
            }),
            not_empty: Condvar::new(),
            max_age,
        }
    }

    /// Submit a task. Only fails if the queue is closed (shutdown). The
    /// queue is otherwise unbounded; cache-layer dedup ensures we don't
    /// queue duplicate Fetch/Extract tasks for the same key.
    fn try_submit(&self, chunk: ChunkKey, priority: Priority, task: Task) -> Result<(), Task> {
        let mut q = self.inner.lock().unwrap();
        if q.closed {
            return Err(task);
        }
        q.next_seq += 1;
        let key = (priority.value(), q.next_seq);
        q.entries.insert(
            key,
            TaskEntry {
                chunk,
                added_at: Instant::now(),
                task,
            },
        );
        self.not_empty.notify_one();
        Ok(())
    }

    /// Block until either a non-stale entry is available (returned) or the
    /// queue is closed. Entries dropped along the way (older than
    /// `max_age`) are surfaced as `dropped` so the caller can run their
    /// cancellation paths.
    fn pop(&self) -> PopResult {
        let mut q = self.inner.lock().unwrap();
        let mut dropped: Vec<TaskEntry> = Vec::new();
        loop {
            if q.closed && q.entries.is_empty() {
                return PopResult::Closed { dropped };
            }
            let Some((_, entry)) = q.entries.pop_first() else {
                q = self.not_empty.wait(q).unwrap();
                continue;
            };
            if entry.added_at.elapsed() > self.max_age {
                dropped.push(entry);
                continue;
            }
            return PopResult::Ready { entry, dropped };
        }
    }
}

enum PopResult {
    Ready {
        entry: TaskEntry,
        dropped: Vec<TaskEntry>,
    },
    Closed {
        dropped: Vec<TaskEntry>,
    },
}

fn worker_loop(inner: Arc<Inner>) {
    loop {
        match inner.task_queue.pop() {
            PopResult::Ready { entry, dropped } => {
                for d in dropped {
                    inner.cancel_dropped_task(d, "stale on pop");
                }
                // Skip-met: cooldown-retry races or duplicate enqueues can
                // leave stale work in the queue. Drop it instead of doing
                // redundant disk + decode work.
                let chunk = entry.chunk;
                match &entry.task {
                    Task::Extract => {
                        if inner.is_chunk_done(chunk) {
                            log::trace!("[{}] skip extract (already terminal)", chunk);
                            inner.chunks.remove(&chunk);
                            continue;
                        }
                    }
                    Task::FetchSource { key, .. } => {
                        if inner.is_source_done(key) {
                            log::trace!("[{}] skip fetch (already done)", key);
                            continue;
                        }
                    }
                }
                match entry.task {
                    Task::FetchSource { key, fetch } => inner.fetch_source(key, fetch),
                    Task::Extract => inner.extract_chunk(chunk),
                }
            }
            PopResult::Closed { dropped } => {
                for d in dropped {
                    inner.cancel_dropped_task(d, "queue closed");
                }
                return;
            }
        }
    }
}
