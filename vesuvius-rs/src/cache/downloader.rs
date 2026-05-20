//! Centralized HTTP downloader for the unified cache.
//!
//! Owns the only `reqwest::blocking::Client` for cache-managed downloads,
//! plus a thread pool that's sized for HTTP concurrency rather than CPU
//! concurrency.
//!
//! Dedup lives one layer up: the cache's source map ensures only the first
//! observer of a source key reaches `try_submit`. Other chunks that want the
//! same source attach as waiters on the source's `Pending` state and get
//! woken when its outcome lands. So this module is intentionally dumb — it
//! does HTTP, fans the bytes back through `on_done`, and that's it.
//!
//! Submission is non-blocking (`try_submit`). The rendezvous channel between
//! the submit side and the worker pool is the cap on outstanding HTTP work:
//! when no worker is parked, `try_submit` returns `QueueFull` rather than
//! buffering — so a panned-away viewport can't pile up stale fetches.

use reqwest::blocking::Client;
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const DEFAULT_HTTP_WORKERS: usize = 32;
const DEFAULT_HTTP_TIMEOUT_SECS: u64 = 60;

#[derive(Debug, Clone)]
pub enum DownloadError {
    /// Transport failure, 5xx, or the pool was unreachable. Caller may
    /// retry.
    Transient(String),
}

pub type DownloadResult = Result<Option<Vec<u8>>, DownloadError>;

pub type OnDone = Box<dyn FnOnce(DownloadResult) + Send + 'static>;

#[derive(Debug)]
pub enum SubmitResult {
    /// Job was handed to an idle worker.
    Submitted,
    /// No worker is parked; nothing was registered. Caller decides what to
    /// do (typically: cool the chunk down and retry next frame).
    QueueFull,
}

pub struct Downloader {
    job_tx: SyncSender<Job>,
}

struct Job {
    url: String,
    on_done: OnDone,
}

impl Downloader {
    pub fn new() -> Self {
        Self::with_workers(DEFAULT_HTTP_WORKERS)
    }

    pub fn with_workers(workers: usize) -> Self {
        let (job_tx, job_rx) = mpsc::sync_channel::<Job>(0);
        let job_rx = Arc::new(Mutex::new(job_rx));

        let client = Client::builder()
            .pool_max_idle_per_host(workers)
            .pool_idle_timeout(Some(Duration::from_secs(60)))
            .http2_adaptive_window(true)
            .tcp_keepalive(Some(Duration::from_secs(30)))
            .timeout(Some(Duration::from_secs(DEFAULT_HTTP_TIMEOUT_SECS)))
            .build()
            .expect("failed to build reqwest client for cache Downloader");

        for i in 0..workers.max(1) {
            let rx = job_rx.clone();
            let client = client.clone();
            std::thread::Builder::new()
                .name(format!("vesuvius-downloader-{}", i))
                .spawn(move || worker_loop(rx, client))
                .expect("spawn downloader worker");
        }

        Self { job_tx }
    }

    /// Non-blocking submission. On `QueueFull`, `on_done` is dropped (never
    /// invoked) — the caller is responsible for rolling back whatever state
    /// it set up in anticipation of the download.
    pub fn try_submit(&self, url: &str, on_done: OnDone) -> SubmitResult {
        match self.job_tx.try_send(Job {
            url: url.to_string(),
            on_done,
        }) {
            Ok(()) => {
                log::trace!("downloader: submitted {}", url);
                SubmitResult::Submitted
            }
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                log::trace!("downloader: queue full, dropping {}", url);
                SubmitResult::QueueFull
            }
        }
    }
}

fn worker_loop(rx: Arc<Mutex<Receiver<Job>>>, client: Client) {
    loop {
        let job = {
            let guard = rx.lock().unwrap();
            match guard.recv() {
                Ok(j) => j,
                Err(_) => return,
            }
        };

        let t0 = std::time::Instant::now();
        log::trace!("downloader: GET {}", job.url);
        let outcome: DownloadResult = match client.get(&job.url).send() {
            Ok(resp) => {
                let status = resp.status();
                if status.as_u16() == 200 {
                    match resp.bytes() {
                        Ok(bytes) => {
                            log::trace!(
                                "downloader: {} → 200 ({} bytes, {:?})",
                                job.url,
                                bytes.len(),
                                t0.elapsed()
                            );
                            Ok(Some(bytes.to_vec()))
                        }
                        Err(e) => Err(DownloadError::Transient(format!("read body: {}", e))),
                    }
                } else if status.as_u16() == 404 {
                    log::trace!("downloader: {} → 404 ({:?})", job.url, t0.elapsed());
                    Ok(None)
                } else {
                    log::debug!(
                        "downloader: {} → {} ({:?})",
                        job.url,
                        status.as_u16(),
                        t0.elapsed()
                    );
                    Err(DownloadError::Transient(format!("status {}", status.as_u16())))
                }
            }
            Err(e) => {
                log::debug!("downloader: {} → transport error: {}", job.url, e);
                Err(DownloadError::Transient(format!("transport: {}", e)))
            }
        };

        (job.on_done)(outcome);
    }
}

impl Default for Downloader {
    fn default() -> Self {
        Self::new()
    }
}
