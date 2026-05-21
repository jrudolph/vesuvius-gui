//! Centralized HTTP downloader for the unified cache.
//!
//! Owns the only `reqwest::blocking::Client` for cache-managed downloads,
//! plus a thread pool that's sized for HTTP concurrency rather than CPU
//! concurrency.
//!
//! ## LIFO queue + age pruning
//!
//! Jobs feed a `BTreeMap` keyed by `!seq` (pure LIFO). The paint loop
//! re-submits / re-touches what it wants every frame, so the queue head
//! stays aligned with the current viewport without any priority sorting.
//! The queue is unbounded — cache-layer dedup (one entry per source key
//! in `self.sources`) means we never submit the same URL twice, and the
//! only staleness check is age: jobs older than `MAX_AGE` are cancelled
//! at pop with a Transient error so the cache rolls the chunk back to a
//! short cooldown.

use super::state::ChunkKey;
use super::MAX_AGE;
use dashmap::DashMap;
use reqwest::blocking::Client;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

const DEFAULT_HTTP_WORKERS: usize = 16;
const DEFAULT_HTTP_TIMEOUT_SECS: u64 = 60;

#[derive(Debug, Clone)]
pub enum DownloadError {
    /// Transport failure, 5xx, queue rejection, or stale-on-pop. Caller may
    /// retry.
    Transient(String),
}

pub type DownloadResult = Result<Option<Vec<u8>>, DownloadError>;

pub type OnDone = Box<dyn FnOnce(DownloadResult) + Send + 'static>;

#[derive(Debug)]
pub enum SubmitResult {
    /// Job was queued.
    Submitted,
    /// Downloader was shut down; nothing queued. `on_done` was invoked with
    /// a Transient error so the cache's cancellation path runs uniformly.
    QueueFull,
}

pub struct Downloader {
    inner: Arc<DownloaderInner>,
}

struct DownloaderInner {
    queue: Mutex<Queue>,
    not_empty: Condvar,
    max_age: Duration,
    /// Chunks with at least one HTTP GET currently in flight on a worker.
    /// Value is the count of concurrent in-flight downloads for that chunk
    /// (a chunk's backfill plan may issue multiple source URLs). Entries are
    /// removed when the count drops to zero, so `contains_key` is a sufficient
    /// "is actively downloading" check.
    active: DashMap<ChunkKey, usize>,
}

struct Queue {
    /// Pure-LIFO ordering: key is `!seq` so the most recent submit (or
    /// `touch`) pops first. See the cache's TaskQueue doc for the
    /// rationale. Older queued downloads slide to the tail and are
    /// processed in LIFO order when workers catch up, or culled by
    /// MAX_AGE.
    entries: BTreeMap<u64, Entry>,
    /// Reverse index by chunk for O(1) `touch`. Maintained in lockstep
    /// with `entries` on submit, pop, and touch.
    chunk_index: HashMap<ChunkKey, Vec<u64>>,
    next_seq: u64,
    closed: bool,
}

fn rev_seq(seq: u64) -> u64 {
    !seq
}

struct Entry {
    url: String,
    /// Optional byte range `(offset, len)`. When set, the worker sends a
    /// `Range: bytes=offset-(offset+len-1)` header and accepts 206.
    range: Option<(u64, u64)>,
    /// The cache chunk this download is on behalf of. Used for logging and
    /// for the per-chunk "active GET in flight" counter exposed via
    /// `is_active_chunk` — the downloader still does not order or schedule
    /// based on chunk identity.
    chunk: ChunkKey,
    added_at: Instant,
    on_done: OnDone,
}

impl Downloader {
    pub fn new() -> Self {
        Self::with_workers(DEFAULT_HTTP_WORKERS)
    }

    pub fn with_workers(workers: usize) -> Self {
        Self::with_settings(workers, MAX_AGE)
    }

    pub fn with_settings(workers: usize, max_age: Duration) -> Self {
        let inner = Arc::new(DownloaderInner {
            queue: Mutex::new(Queue {
                entries: BTreeMap::new(),
                chunk_index: HashMap::new(),
                next_seq: 0,
                closed: false,
            }),
            not_empty: Condvar::new(),
            max_age,
            active: DashMap::new(),
        });

        let client = Client::builder()
            .pool_max_idle_per_host(workers)
            .pool_idle_timeout(Some(Duration::from_secs(60)))
            .http2_adaptive_window(true)
            .tcp_keepalive(Some(Duration::from_secs(30)))
            .timeout(Some(Duration::from_secs(DEFAULT_HTTP_TIMEOUT_SECS)))
            .build()
            .expect("failed to build reqwest client for cache Downloader");

        for i in 0..workers.max(1) {
            let inner = inner.clone();
            let client = client.clone();
            std::thread::Builder::new()
                .name(format!("vesuvius-downloader-{}", i))
                .spawn(move || worker_loop(inner, client))
                .expect("spawn downloader worker");
        }

        Self { inner }
    }

    /// Non-blocking submission.
    ///
    /// `chunk` is the cache chunk this download is on behalf of (used for
    /// logging + the in-flight counter — the downloader doesn't otherwise
    /// schedule based on it). `range`, when `Some((offset, len))`, becomes
    /// a `Range: bytes=offset-(offset+len-1)` header on the request; 206
    /// Partial Content is accepted as success.
    ///
    /// On `QueueFull`, `on_done` is invoked synchronously with
    /// `Err(Transient(...))` so the caller's cancellation path runs
    /// uniformly with stale-pop cancellation. Callers MUST NOT hold any
    /// lock that `on_done` re-enters.
    pub fn try_submit(
        &self,
        url: &str,
        range: Option<(u64, u64)>,
        chunk: ChunkKey,
        on_done: OnDone,
    ) -> SubmitResult {
        let rejected_on_done = {
            let mut q = self.inner.queue.lock().unwrap();
            if q.closed {
                Some(on_done)
            } else {
                q.next_seq += 1;
                let key = rev_seq(q.next_seq);
                q.entries.insert(
                    key,
                    Entry {
                        url: url.to_string(),
                        range,
                        chunk,
                        added_at: Instant::now(),
                        on_done,
                    },
                );
                q.chunk_index.entry(chunk).or_default().push(key);
                self.inner.not_empty.notify_one();
                None
            }
        };
        match rejected_on_done {
            None => {
                log::trace!("[{}] submitted", url);
                SubmitResult::Submitted
            }
            Some(on_done) => {
                log::trace!("[{}] dropped: downloader closed", url);
                on_done(Err(DownloadError::Transient("downloader closed".into())));
                SubmitResult::QueueFull
            }
        }
    }

    /// True iff a worker is currently executing an HTTP GET for at least one
    /// source URL submitted on behalf of `chunk`. Queued-but-not-yet-popped
    /// entries don't count — this is for the debug overlay to distinguish
    /// "waiting in queue" from "bytes coming over the wire right now".
    pub fn is_active_chunk(&self, chunk: ChunkKey) -> bool {
        self.inner.active.contains_key(&chunk)
    }

    /// Refresh every queued download for `chunk`: bump seq (moving it
    /// to the head of the LIFO queue) and reset `added_at` so MAX_AGE
    /// re-counts from now. No-op when the chunk has no queued downloads.
    /// Called from the cache's `state_or_fetch` on every paint poll so
    /// the queue head tracks the current viewport.
    pub fn touch(&self, chunk: ChunkKey) {
        let mut q = self.inner.queue.lock().unwrap();
        let old_keys = match q.chunk_index.remove(&chunk) {
            Some(v) if !v.is_empty() => v,
            _ => return,
        };
        let mut new_keys = Vec::with_capacity(old_keys.len());
        let now = Instant::now();
        for old_key in old_keys {
            let Some(mut entry) = q.entries.remove(&old_key) else {
                continue;
            };
            entry.added_at = now;
            q.next_seq += 1;
            let new_key = rev_seq(q.next_seq);
            q.entries.insert(new_key, entry);
            new_keys.push(new_key);
        }
        if !new_keys.is_empty() {
            q.chunk_index.insert(chunk, new_keys);
            self.inner.not_empty.notify_one();
        }
    }
}

impl DownloaderInner {
    fn mark_active(&self, chunk: ChunkKey) {
        *self.active.entry(chunk).or_insert(0) += 1;
    }

    fn unmark_active(&self, chunk: ChunkKey) {
        if let dashmap::mapref::entry::Entry::Occupied(mut e) = self.active.entry(chunk) {
            let v = e.get_mut();
            *v = v.saturating_sub(1);
            if *v == 0 {
                e.remove();
            }
        }
    }
}

/// RAII guard that decrements the active-download counter for `chunk` when
/// dropped. Held across the HTTP GET so a panic still releases the slot.
struct ActiveGuard<'a> {
    inner: &'a DownloaderInner,
    chunk: ChunkKey,
}

impl Drop for ActiveGuard<'_> {
    fn drop(&mut self) {
        self.inner.unmark_active(self.chunk);
    }
}

impl Default for Downloader {
    fn default() -> Self {
        Self::new()
    }
}

fn worker_loop(inner: Arc<DownloaderInner>, client: Client) {
    loop {
        let entry: Entry = {
            let mut q = inner.queue.lock().unwrap();
            let max_age = inner.max_age;
            loop {
                if q.closed && q.entries.is_empty() {
                    return;
                }
                let Some((key, entry)) = q.entries.pop_first() else {
                    q = inner.not_empty.wait(q).unwrap();
                    continue;
                };
                if let Some(keys) = q.chunk_index.get_mut(&entry.chunk) {
                    keys.retain(|k| *k != key);
                    if keys.is_empty() {
                        q.chunk_index.remove(&entry.chunk);
                    }
                }
                if entry.added_at.elapsed() > max_age {
                    // Stale by age — cancel + loop.
                    drop(q);
                    log::trace!("[{}] aged out", entry.url);
                    (entry.on_done)(Err(DownloadError::Transient("aged out".into())));
                    q = inner.queue.lock().unwrap();
                    continue;
                }
                break entry;
            }
        };

        let t0 = Instant::now();
        let mut req = client.get(&entry.url);
        if let Some((off, len)) = entry.range {
            // bytes=off-end is inclusive on both ends; end = off + len - 1.
            let end = off.saturating_add(len.saturating_sub(1));
            let header = format!("bytes={}-{}", off, end);
            log::trace!("[{}] GET {}", entry.url, header);
            req = req.header(reqwest::header::RANGE, header);
        } else {
            log::trace!("[{}] GET", entry.url);
        }
        inner.mark_active(entry.chunk);
        let _active = ActiveGuard {
            inner: &inner,
            chunk: entry.chunk,
        };
        let outcome: DownloadResult = match req.send() {
            Ok(resp) => {
                let status = resp.status();
                let code = status.as_u16();
                // 200 OK is the un-ranged success; 206 Partial Content is the
                // ranged success. Some servers return 200 with the full body
                // when they ignore Range — caller decides whether that's
                // acceptable. Here we surface either as Ok(Some(bytes)).
                if code == 200 || code == 206 {
                    match resp.bytes() {
                        Ok(bytes) => {
                            log::trace!("[{}] {} ({} bytes, {:?})", entry.url, code, bytes.len(), t0.elapsed());
                            Ok(Some(bytes.to_vec()))
                        }
                        Err(e) => Err(DownloadError::Transient(format!("read body: {}", e))),
                    }
                } else if code == 404 || code == 403 {
                    // 404 (not found) and 403 (forbidden) are both definitive
                    // absences for our purposes: many static-object stores
                    // serve 403 instead of 404 for unlisted keys. Surface as
                    // `Ok(None)` so the cache can negatively cache the chunk
                    // rather than retry on a cooldown loop.
                    log::trace!("[{}] {} ({:?})", entry.url, code, t0.elapsed());
                    Ok(None)
                } else {
                    // 416 Range Not Satisfiable: shouldn't happen post-index
                    // lookup. Treat as transient so the cooldown surfaces it.
                    log::debug!("[{}] {} ({:?})", entry.url, code, t0.elapsed());
                    Err(DownloadError::Transient(format!("status {}", code)))
                }
            }
            Err(e) => {
                log::debug!("[{}] transport error: {}", entry.url, e);
                Err(DownloadError::Transient(format!("transport: {}", e)))
            }
        };

        (entry.on_done)(outcome);
    }
}
