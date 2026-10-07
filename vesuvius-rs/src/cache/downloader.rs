//! Centralized HTTP downloader for the unified cache.
//!
//! Owns the only `reqwest::blocking::Client` for cache-managed downloads,
//! plus a thread pool that's sized for HTTP concurrency rather than CPU
//! concurrency.
//!
//! ## LIFO queue + age pruning
//!
//! Jobs feed the shared `LifoQueue` (see `lifo.rs`). The paint loop
//! re-submits / re-touches what it wants every frame, so the queue head
//! stays aligned with the current viewport without any priority sorting.
//! The queue is unbounded — cache-layer dedup (one entry per source key
//! in `self.sources`) means we never submit the same URL twice, and the
//! only staleness check is age: jobs older than `MAX_AGE` are cancelled
//! at pop with a Transient error so the cache rolls the chunk back to a
//! short cooldown.

use super::lifo::LifoQueue;
use super::netlog;
use super::s3_auth::{self, S3Signer};
use super::state::ChunkKey;
use super::MAX_AGE;
use dashmap::DashMap;
use reqwest::blocking::Client;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const DEFAULT_HTTP_WORKERS: usize = 16;
const DEFAULT_HTTP_TIMEOUT_SECS: u64 = 60;

/// Worker (= max concurrent HTTP GET) count, overridable via
/// `VESUVIUS_HTTP_WORKERS`. Direct-to-S3 small-object fetching is latency-bound
/// and benefits from far more than the default 16 (mountpoint tops out where a
/// few hundred parallel GETs would keep climbing).
fn configured_http_workers() -> usize {
    std::env::var("VESUVIUS_HTTP_WORKERS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_HTTP_WORKERS)
}

/// Whether to attempt AWS SigV4 signing of S3 downloads. Auto-enabled when the
/// environment looks like it carries AWS credentials (IRSA web-identity, an
/// assumed role, or static keys); force on/off with `VESUVIUS_S3_AUTH=1`/`0`.
/// Kept off by default elsewhere so a laptop/GUI run never probes IMDS.
fn s3_signing_enabled() -> bool {
    match std::env::var("VESUVIUS_S3_AUTH") {
        Ok(v) => v == "1" || v.eq_ignore_ascii_case("true"),
        Err(_) => {
            std::env::var_os("AWS_WEB_IDENTITY_TOKEN_FILE").is_some()
                || std::env::var_os("AWS_ROLE_ARN").is_some()
                || std::env::var_os("AWS_ACCESS_KEY_ID").is_some()
        }
    }
}

#[derive(Debug, Clone)]
pub enum DownloadError {
    /// Transport failure, 5xx, queue rejection, or stale-on-pop. Caller may
    /// retry.
    Transient(String),
}

/// Successful bodies are delivered as `bytes::Bytes` — the zero-copy
/// buffer `reqwest` already produced. Converting to `Vec<u8>` here would
/// copy every multi-MB shard read a second time before the spill write.
pub type DownloadResult = Result<Option<bytes::Bytes>, DownloadError>;

pub type OnDone = Box<dyn FnOnce(DownloadResult) + Send + 'static>;

pub struct Downloader {
    inner: Arc<DownloaderInner>,
}

struct DownloaderInner {
    queue: LifoQueue<Job>,
    /// Chunks with at least one HTTP GET currently in flight on a worker.
    /// Value is the count of concurrent in-flight downloads for that chunk
    /// (a chunk's backfill plan may issue multiple source URLs). Entries are
    /// removed when the count drops to zero, so `contains_key` is a sufficient
    /// "is actively downloading" check.
    active: DashMap<ChunkKey, usize>,
    /// Total HTTP GETs currently on the wire across all workers. Telemetry
    /// only (the netlog records the concurrency each request contended with).
    in_flight: AtomicUsize,
    /// Cumulative counters since construction. Telemetry only — snapshotted
    /// via `Downloader::stats` (e.g. per frame by the pane harness).
    counters: Counters,
    /// Optional SigV4 signer for S3-hosted URLs. `None` when signing is disabled
    /// or no credentials resolved; non-S3 URLs are never signed regardless.
    signer: Option<Arc<S3Signer>>,
}

#[derive(Default)]
struct Counters {
    submitted: AtomicU64,
    completed: AtomicU64,
    not_found: AtomicU64,
    failed: AtomicU64,
    aged_out: AtomicU64,
    bytes: AtomicU64,
}

/// Point-in-time snapshot of the downloader: live gauges (`in_flight`,
/// `queued`) plus cumulative counters since construction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DownloaderStats {
    /// HTTP GETs currently on the wire.
    pub in_flight: usize,
    /// Jobs waiting in the LIFO queue (not yet popped by a worker).
    pub queued: usize,
    pub submitted: u64,
    /// 200/206 responses with a body.
    pub completed: u64,
    /// 403/404 — definitive absences.
    pub not_found: u64,
    /// Transport errors and other statuses.
    pub failed: u64,
    /// Culled at pop for exceeding `MAX_AGE` without being fetched.
    pub aged_out: u64,
    /// Body bytes received across all completed GETs.
    pub bytes: u64,
}

struct Job {
    url: String,
    /// Optional byte range `(offset, len)`. When set, the worker sends a
    /// `Range: bytes=offset-(offset+len-1)` header and accepts 206.
    range: Option<(u64, u64)>,
    on_done: OnDone,
}

impl Downloader {
    pub fn new() -> Self {
        Self::with_workers(configured_http_workers())
    }

    pub fn with_workers(workers: usize) -> Self {
        Self::with_settings(workers, MAX_AGE, Arc::new(AtomicBool::new(true)))
    }

    /// Construct with a caller-owned cull flag so the cache can share one
    /// `Arc<AtomicBool>` across the downloader queue and its task queue and
    /// toggle both with a single `ChunkCache::set_culling`.
    pub fn with_shared_cull(cull_enabled: Arc<AtomicBool>) -> Self {
        Self::with_settings(configured_http_workers(), MAX_AGE, cull_enabled)
    }

    pub fn with_settings(workers: usize, max_age: Duration, cull_enabled: Arc<AtomicBool>) -> Self {
        // Resolve S3 credentials once up front (blocks briefly on the first
        // STS/IRSA exchange) so workers can sign without async credential I/O.
        let signer = if s3_signing_enabled() { S3Signer::try_new() } else { None };

        let inner = Arc::new(DownloaderInner {
            queue: LifoQueue::new(max_age, cull_enabled),
            active: DashMap::new(),
            in_flight: AtomicUsize::new(0),
            counters: Counters::default(),
            signer,
        });

        // HTTP/1.1 only, deliberately: with ALPN h2, reqwest multiplexes
        // ALL workers onto a single TCP connection per host (the pool
        // settings below only apply to http/1.1). One connection means one
        // congestion window — measured ~3MB/s on a 10MB/s link against
        // CloudFront — and every in-flight chunk inflates every other
        // chunk's latency (2MB bodies taking 8-12s under load). Sixteen
        // http/1.1 connections reach ~7-8MB/s on the same link.
        //
        // VESUVIUS_H2_WINDOW=<bytes> opts back into h2 with a fixed stream
        // window (connection window 4x that) for experiments; a ~4MB window
        // recovers throughput but keeps the shared-connection latency
        // coupling, so it's not the default.
        let mut builder = Client::builder()
            .pool_max_idle_per_host(workers)
            .pool_idle_timeout(Some(Duration::from_secs(60)))
            .tcp_keepalive(Some(Duration::from_secs(30)))
            .timeout(Some(Duration::from_secs(DEFAULT_HTTP_TIMEOUT_SECS)));
        if let Some(window) = std::env::var("VESUVIUS_H2_WINDOW").ok().and_then(|v| v.parse::<u32>().ok()) {
            builder = builder
                .http2_initial_stream_window_size(window)
                .http2_initial_connection_window_size(window.saturating_mul(4));
        } else {
            builder = builder.http1_only();
        }
        let client = builder
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

    /// Non-blocking submission. The queue is unbounded — dedup happens at
    /// the cache's source map — so submission always succeeds; the only
    /// way a job dies unprocessed is the MAX_AGE cull at pop, which
    /// invokes `on_done` with a Transient error.
    ///
    /// `chunk` is the cache chunk this download is on behalf of (used for
    /// logging + the in-flight counter — the downloader doesn't otherwise
    /// schedule based on it). `range`, when `Some((offset, len))`, becomes
    /// a `Range: bytes=offset-(offset+len-1)` header on the request; 206
    /// Partial Content is accepted as success.
    pub fn submit(&self, url: &str, range: Option<(u64, u64)>, chunk: ChunkKey, on_done: OnDone) {
        self.inner.counters.submitted.fetch_add(1, Ordering::Relaxed);
        self.inner.queue.submit(
            chunk,
            Job {
                url: url.to_string(),
                range,
                on_done,
            },
        );
        log::trace!("[{}] submitted", url);
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
        self.inner.queue.touch(chunk);
    }

    /// Snapshot of live gauges and cumulative counters. Takes the queue
    /// lock once (for `queued`); cheap enough to call per frame.
    pub fn stats(&self) -> DownloaderStats {
        let c = &self.inner.counters;
        DownloaderStats {
            in_flight: self.inner.in_flight.load(Ordering::Relaxed),
            queued: self.inner.queue.len(),
            submitted: c.submitted.load(Ordering::Relaxed),
            completed: c.completed.load(Ordering::Relaxed),
            not_found: c.not_found.load(Ordering::Relaxed),
            failed: c.failed.load(Ordering::Relaxed),
            aged_out: c.aged_out.load(Ordering::Relaxed),
            bytes: c.bytes.load(Ordering::Relaxed),
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

/// Host part of `url`, for per-host aggregation in the netlog.
fn url_host(url: &str) -> &str {
    url.split('/').nth(2).unwrap_or("")
}

fn worker_loop(inner: Arc<DownloaderInner>, client: Client) {
    loop {
        let (entry, dropped) = inner.queue.pop();
        for d in dropped {
            // Stale by age — cancel so the cache rolls the chunk back to
            // a cooldown.
            log::trace!("[{}] aged out", d.item.url);
            inner.counters.aged_out.fetch_add(1, Ordering::Relaxed);
            if netlog::enabled() {
                netlog::emit(serde_json::json!({
                    "t": netlog::now_ms(),
                    "event": "aged_out",
                    "host": url_host(&d.item.url),
                    "url": d.item.url,
                    "chunk": format!("{:?}", d.chunk),
                    "range_off": d.item.range.map(|(off, _)| off),
                    "queued_ms": d.submitted_at.elapsed().as_millis() as u64,
                    "touches": d.touch_count,
                }));
            }
            (d.item.on_done)(Err(DownloadError::Transient("aged out".into())));
        }
        let chunk = entry.chunk;
        let job = entry.item;
        // Queue wait split two ways: since the last touch (how long the
        // *current* viewport waited) and since first submission.
        let wait_ms = entry.added_at.elapsed().as_millis() as u64;
        let wait_total_ms = entry.submitted_at.elapsed().as_millis() as u64;
        let q_depth = if netlog::enabled() { inner.queue.len() } else { 0 };

        let t0 = Instant::now();
        let mut req = client.get(&job.url);
        if let Some((off, len)) = job.range {
            // bytes=off-end is inclusive on both ends; end = off + len - 1.
            let end = off.saturating_add(len.saturating_sub(1));
            let header = format!("bytes={}-{}", off, end);
            log::trace!("[{}] GET {}", job.url, header);
            req = req.header(reqwest::header::RANGE, header);
        } else {
            log::trace!("[{}] GET", job.url);
        }
        // SigV4-sign S3-hosted requests; everything else goes out unsigned.
        if let Some(signer) = inner.signer.as_ref() {
            if s3_auth::is_s3_host(&job.url) {
                match signer.sign_get(&job.url) {
                    Ok(headers) => {
                        for (name, value) in headers {
                            req = req.header(name, value);
                        }
                    }
                    Err(e) => log::warn!("[{}] SigV4 signing failed: {}", job.url, e),
                }
            }
        }
        inner.mark_active(chunk);
        let in_flight = inner.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        let _active = ActiveGuard { inner: &inner, chunk };
        let mut ttfb_ms: u64 = 0;
        let mut body_ms: u64 = 0;
        let mut got_bytes: u64 = 0;
        let mut http_status: u16 = 0;
        let mut cdn_cache: Option<String> = None;
        let outcome: DownloadResult = match req.send() {
            Ok(resp) => {
                // `send` returns once response headers are in: TTFB covers
                // queue-free request latency (connect/TLS if not pooled,
                // plus server processing and one RTT).
                ttfb_ms = t0.elapsed().as_millis() as u64;
                let status = resp.status();
                let code = status.as_u16();
                http_status = code;
                if netlog::enabled() {
                    cdn_cache = resp
                        .headers()
                        .get("x-cache")
                        .and_then(|v| v.to_str().ok())
                        .map(|s| s.to_string());
                }
                // 200 OK is the un-ranged success; 206 Partial Content is the
                // ranged success. Some servers return 200 with the full body
                // when they ignore Range — caller decides whether that's
                // acceptable. Here we surface either as Ok(Some(bytes)).
                if code == 200 || code == 206 {
                    let t_body = Instant::now();
                    match resp.bytes() {
                        Ok(bytes) => {
                            body_ms = t_body.elapsed().as_millis() as u64;
                            got_bytes = bytes.len() as u64;
                            log::trace!("[{}] {} ({} bytes, {:?})", job.url, code, bytes.len(), t0.elapsed());
                            Ok(Some(bytes))
                        }
                        Err(e) => Err(DownloadError::Transient(format!("read body: {}", e))),
                    }
                } else if code == 404 || code == 403 {
                    // 404 (not found) and 403 (forbidden) are both definitive
                    // absences for our purposes: many static-object stores
                    // serve 403 instead of 404 for unlisted keys. Surface as
                    // `Ok(None)` so the cache can negatively cache the chunk
                    // rather than retry on a cooldown loop.
                    log::trace!("[{}] {} ({:?})", job.url, code, t0.elapsed());
                    Ok(None)
                } else {
                    // 416 Range Not Satisfiable: shouldn't happen post-index
                    // lookup. Treat as transient so the cooldown surfaces it.
                    log::debug!("[{}] {} ({:?})", job.url, code, t0.elapsed());
                    Err(DownloadError::Transient(format!("status {}", code)))
                }
            }
            Err(e) => {
                log::debug!("[{}] transport error: {}", job.url, e);
                Err(DownloadError::Transient(format!("transport: {}", e)))
            }
        };
        inner.in_flight.fetch_sub(1, Ordering::Relaxed);
        let counter = match &outcome {
            Ok(Some(_)) => &inner.counters.completed,
            Ok(None) => &inner.counters.not_found,
            Err(_) => &inner.counters.failed,
        };
        counter.fetch_add(1, Ordering::Relaxed);
        inner.counters.bytes.fetch_add(got_bytes, Ordering::Relaxed);

        if netlog::enabled() {
            netlog::emit(serde_json::json!({
                "t": netlog::now_ms(),
                "event": "download",
                "host": url_host(&job.url),
                "url": job.url,
                "chunk": format!("{:?}", chunk),
                "range_off": job.range.map(|(off, _)| off),
                "range_len": job.range.map(|(_, len)| len),
                "status": http_status,
                "ok": outcome.is_ok(),
                "x_cache": cdn_cache,
                "wait_ms": wait_ms,
                "wait_total_ms": wait_total_ms,
                "touches": entry.touch_count,
                "ttfb_ms": ttfb_ms,
                "body_ms": body_ms,
                "bytes": got_bytes,
                "in_flight": in_flight,
                "q_depth": q_depth,
            }));
        }

        (job.on_done)(outcome);
    }
}
