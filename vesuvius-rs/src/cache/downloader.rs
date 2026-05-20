//! Centralized HTTP downloader for the unified cache.
//!
//! Owns the only `reqwest::blocking::Client` for cache-managed downloads,
//! plus a thread pool that's sized for HTTP concurrency rather than CPU
//! concurrency.
//!
//! ## Priority queue + age pruning
//!
//! Jobs feed a `BTreeMap` keyed by `(priority, seq)`. The paint loop submits
//! coarse-LOD-first / center-first, so the BTreeMap naturally pops the
//! most urgent work across panes. The queue is unbounded — cache-layer
//! dedup (one entry per source key in `self.sources`) means we never
//! submit the same URL twice, and the only staleness check is age: jobs
//! older than `MAX_AGE` are cancelled at pop with a Transient error so
//! the cache rolls the chunk back to a short cooldown.

use super::priority::{Priority, MAX_AGE};
use super::state::ChunkKey;
use reqwest::blocking::Client;
use std::collections::BTreeMap;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

const DEFAULT_HTTP_WORKERS: usize = 32;
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
}

struct Queue {
    /// Key: (priority value, monotonic seq). Lower keys = popped first.
    /// `next_seq` is the FIFO tiebreaker for entries with identical priority.
    entries: BTreeMap<(u64, u64), Entry>,
    next_seq: u64,
    closed: bool,
}

struct Entry {
    url: String,
    /// The cache chunk this download is on behalf of. Currently logged for
    /// observability only — the downloader itself doesn't make decisions
    /// based on it.
    #[allow(dead_code)]
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
                next_seq: 0,
                closed: false,
            }),
            not_empty: Condvar::new(),
            max_age,
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

    /// Non-blocking submission with explicit priority + chunk identity.
    ///
    /// `chunk` is the cache chunk this download is on behalf of (used for
    /// logging only — the downloader doesn't otherwise care).
    ///
    /// On `QueueFull`, `on_done` is invoked synchronously with
    /// `Err(Transient(...))` so the caller's cancellation path runs
    /// uniformly with stale-pop cancellation. Callers MUST NOT hold any
    /// lock that `on_done` re-enters.
    pub fn try_submit(&self, url: &str, chunk: ChunkKey, priority: Priority, on_done: OnDone) -> SubmitResult {
        let rejected_on_done = {
            let mut q = self.inner.queue.lock().unwrap();
            if q.closed {
                Some(on_done)
            } else {
                q.next_seq += 1;
                let key = (priority.value(), q.next_seq);
                q.entries.insert(
                    key,
                    Entry {
                        url: url.to_string(),
                        chunk,
                        added_at: Instant::now(),
                        on_done,
                    },
                );
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
                let Some((_, entry)) = q.entries.pop_first() else {
                    q = inner.not_empty.wait(q).unwrap();
                    continue;
                };
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
        log::trace!("[{}] GET", entry.url);
        let outcome: DownloadResult = match client.get(&entry.url).send() {
            Ok(resp) => {
                let status = resp.status();
                if status.as_u16() == 200 {
                    match resp.bytes() {
                        Ok(bytes) => {
                            log::trace!("[{}] 200 ({} bytes, {:?})", entry.url, bytes.len(), t0.elapsed());
                            Ok(Some(bytes.to_vec()))
                        }
                        Err(e) => Err(DownloadError::Transient(format!("read body: {}", e))),
                    }
                } else if status.as_u16() == 404 || status.as_u16() == 403 {
                    // 404 (not found) and 403 (forbidden) are both definitive
                    // absences for our purposes: many static-object stores
                    // serve 403 instead of 404 for unlisted keys. Surface as
                    // `Ok(None)` so the cache can negatively cache the chunk
                    // rather than retry on a cooldown loop.
                    log::trace!("[{}] {} ({:?})", entry.url, status.as_u16(), t0.elapsed());
                    Ok(None)
                } else {
                    log::debug!("[{}] {} ({:?})", entry.url, status.as_u16(), t0.elapsed());
                    Err(DownloadError::Transient(format!("status {}", status.as_u16())))
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
