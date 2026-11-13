//! LRU bookkeeping and memory budget for the decoded-chunk tier.
//!
//! One `EpochState` per unified cache root, shared by every volume opened in
//! this process, in memory only. It tracks:
//!
//! - `current` — the active u8 epoch. Wraps after 256 advances; purge keeps
//!   surviving chunks within 255 epochs of `current`, so wrap is safe.
//! - `bytes_since_advance` — bytes filled into the slot store since the last
//!   advance. Advancing happens once `bytes_per_epoch` is crossed.
//! - `bytes_per_epoch` — derived from the budget so the 256-slot range covers
//!   ≈1.3× the budget.
//! - `total_chunks` — resident decoded chunks (all volumes).
//! - `epoch_chunks[i]` — chunks whose `access_epoch == i`, for goal-based
//!   purge planning.
//!
//! Hot path (per chunk access) does at most one relaxed `current()` load,
//! one comparison, and on transition two histogram updates under the mutex.
//!
//! The budget is a plain number (`VESUVIUS_MEMORY_CACHE_GB`, default 2 GiB of
//! decoded chunks). A fill that pushes the resident set over the high-water
//! mark wakes the purge thread, which evicts the oldest chunks down to the
//! low-water mark. There is no reaction to system memory pressure; swap is the
//! backstop.

use super::purge::{PurgePlan, PurgeTarget, VolumeBreakdown};
use super::CHUNK_VOXELS;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak};
use std::time::Duration;

/// Fallback purge check interval, in case a wake-up was missed.
const PURGE_INTERVAL: Duration = Duration::from_secs(5);

/// At or above this fill (% of budget) a purge runs.
const HIGH_WATER_PCT: u64 = 95;
/// Each purge evicts down to this fill (% of budget).
const LOW_WATER_PCT: u64 = 80;

pub const EPOCH_SLOTS: usize = 256;

/// Default decoded-chunk budget.
pub const DEFAULT_CAP_BYTES: u64 = 2 << 30;

/// Env var for the decoded-chunk budget (memory), in GiB.
pub const CAP_ENV_VAR: &str = "VESUVIUS_MEMORY_CACHE_GB";

/// Env vars of the old disk-backed cache. They no longer do anything; the
/// disk cap is now `seglog::LOG_CAP_ENV_VAR`.
const OBSOLETE_ENV_VARS: [&str; 2] = ["VESUVIUS_CACHE_CAP_GB", "VESUVIUS_RAW_CACHE_GB"];

/// The budget from `VESUVIUS_MEMORY_CACHE_GB`, or `DEFAULT_CAP_BYTES`.
pub fn cap_bytes_from_env() -> u64 {
    static CACHED: OnceLock<u64> = OnceLock::new();
    for var in OBSOLETE_ENV_VARS {
        if std::env::var_os(var).is_some() {
            log::warn!(
                "{} is obsolete and ignored: the memory budget is {}, the disk cache cap {}",
                var,
                CAP_ENV_VAR,
                super::seglog::LOG_CAP_ENV_VAR
            );
        }
    }
    *CACHED.get_or_init(|| match std::env::var(CAP_ENV_VAR) {
        Ok(raw) => {
            let gb: u64 = raw
                .trim()
                .parse()
                .unwrap_or_else(|e| panic!("{} must be a positive integer number of GiB: {}", CAP_ENV_VAR, e));
            assert!(gb > 0, "{} must be positive", CAP_ENV_VAR);
            gb << 30
        }
        Err(_) => DEFAULT_CAP_BYTES,
    })
}

const HEADROOM_NUM: u128 = 13;
const HEADROOM_DEN: u128 = 10;

pub struct EpochState {
    /// Kept outside `inner` so the paint hot path can read it without the
    /// mutex. Written only while holding `inner`.
    current: AtomicU8,
    inner: Mutex<Inner>,
    targets: Mutex<Vec<Weak<dyn PurgeTarget>>>,
    /// Wakes the purge thread: set when a fill crosses the high-water mark.
    purge_signal: Arc<(Mutex<bool>, Condvar)>,
}

struct Inner {
    bytes_since_advance: u64,
    bytes_per_epoch: u64,
    cap_bytes: u64,
    total_chunks: u64,
    epoch_chunks: [u32; EPOCH_SLOTS],
}

impl EpochState {
    pub fn new(cap_bytes: u64) -> Self {
        Self {
            current: AtomicU8::new(1),
            inner: Mutex::new(Inner {
                bytes_since_advance: 0,
                bytes_per_epoch: derive_bytes_per_epoch(cap_bytes),
                cap_bytes,
                total_chunks: 0,
                epoch_chunks: [0; EPOCH_SLOTS],
            }),
            targets: Mutex::new(Vec::new()),
            purge_signal: Arc::new((Mutex::new(false), Condvar::new())),
        }
    }

    /// Register a volume the purge can evict from. Weakly held.
    pub fn register_target(&self, target: Weak<dyn PurgeTarget>) {
        self.targets.lock().unwrap().push(target);
    }

    fn high_water_chunks(&self) -> u64 {
        (self.cap_bytes() / CHUNK_VOXELS as u64).saturating_mul(HIGH_WATER_PCT) / 100
    }

    /// Evict down to the low-water mark if over the high-water mark. Returns
    /// the number of chunks evicted.
    pub fn run_purge_pass(&self, source: &str) -> u64 {
        let chunks_capacity = self.cap_bytes() / CHUNK_VOXELS as u64;
        let high_water = self.high_water_chunks();
        let low_water = chunks_capacity.saturating_mul(LOW_WATER_PCT) / 100;
        let total = self.total_chunks();
        if total <= high_water {
            return 0;
        }
        let target = total - low_water;
        let evicted = self.purge_all_to_target(target);
        log::info!(
            target: super::purge::LOG_TARGET,
            "epoch {}: purge target={} evicted={} (was {} chunks resident, high_water={}, low_water={})",
            source,
            target,
            evicted,
            total,
            high_water,
            low_water,
        );
        if super::netlog::enabled() {
            super::netlog::emit(serde_json::json!({
                "t": super::netlog::now_ms(),
                "event": "purge",
                "trigger": source,
                "target": target,
                "evicted": evicted,
                "resident_before": total,
                "high_water": high_water,
                "low_water": low_water,
            }));
        }
        evicted
    }

    /// Plan against the current histogram and run it on every live volume.
    pub fn purge_all_to_target(&self, target_chunks: u64) -> u64 {
        let Some(plan) = PurgePlan::build(self, target_chunks) else {
            return 0;
        };
        let current = self.current();
        let live: Vec<Arc<dyn PurgeTarget>> = {
            let mut targets = self.targets.lock().unwrap();
            targets.retain(|w| w.strong_count() > 0);
            targets.iter().filter_map(|w| w.upgrade()).collect()
        };
        if log::log_enabled!(target: super::purge::LOG_TARGET, log::Level::Info) {
            let summaries: Vec<VolumeBreakdown> = live.iter().map(|t| t.summarize(plan, current)).collect();
            log_plan(plan, current, &summaries);
        }
        live.iter().map(|t| t.run_purge(plan)).sum()
    }

    pub fn current(&self) -> u8 {
        self.current.load(Ordering::Relaxed)
    }

    pub fn total_chunks(&self) -> u64 {
        self.inner.lock().unwrap().total_chunks
    }

    pub fn cap_bytes(&self) -> u64 {
        self.inner.lock().unwrap().cap_bytes
    }

    pub fn bytes_per_epoch(&self) -> u64 {
        self.inner.lock().unwrap().bytes_per_epoch
    }

    pub fn epoch_chunks(&self) -> [u32; EPOCH_SLOTS] {
        self.inner.lock().unwrap().epoch_chunks
    }

    /// Account for a newly filled chunk; returns the epoch to tag it with.
    /// Wakes the purge thread when the resident set crosses the high-water
    /// mark.
    pub fn record_fill(&self, bytes_written: u64) -> u8 {
        let (cur, total) = {
            let mut s = self.inner.lock().unwrap();
            s.bytes_since_advance = s.bytes_since_advance.saturating_add(bytes_written);
            self.maybe_advance(&mut s);
            s.total_chunks += 1;
            let cur = self.current.load(Ordering::Relaxed);
            s.epoch_chunks[cur as usize] = s.epoch_chunks[cur as usize].saturating_add(1);
            (cur, s.total_chunks)
        };
        if total > self.high_water_chunks() {
            let (m, cv) = &*self.purge_signal;
            *m.lock().unwrap() = true;
            cv.notify_one();
        }
        cur
    }

    /// Reflect an access-epoch CAS the caller already won (`from → to`).
    pub fn record_access_transition(&self, from: u8, to: u8) {
        if from == to {
            return;
        }
        let mut s = self.inner.lock().unwrap();
        s.epoch_chunks[from as usize] = s.epoch_chunks[from as usize].saturating_sub(1);
        s.epoch_chunks[to as usize] = s.epoch_chunks[to as usize].saturating_add(1);
    }

    pub fn record_evict(&self, victim_epoch: u8) {
        let mut s = self.inner.lock().unwrap();
        s.epoch_chunks[victim_epoch as usize] = s.epoch_chunks[victim_epoch as usize].saturating_sub(1);
        s.total_chunks = s.total_chunks.saturating_sub(1);
    }

    /// Caller holds `inner`, which serializes writes to `current`.
    fn maybe_advance(&self, s: &mut Inner) {
        let mut advances = 0usize;
        while s.bytes_since_advance >= s.bytes_per_epoch && advances < EPOCH_SLOTS {
            s.bytes_since_advance -= s.bytes_per_epoch;
            advances += 1;
            let next = self.current.load(Ordering::Relaxed).wrapping_add(1);
            if s.epoch_chunks[next as usize] != 0 {
                log::warn!(
                    target: crate::cache::purge::LOG_TARGET,
                    "epoch advance into occupied slot {} ({} chunks); purge needed",
                    next,
                    s.epoch_chunks[next as usize]
                );
            }
            self.current.store(next, Ordering::Relaxed);
        }
        if advances == EPOCH_SLOTS {
            s.bytes_since_advance = 0;
        }
    }
}

fn derive_bytes_per_epoch(cap_bytes: u64) -> u64 {
    let n = (cap_bytes as u128 * HEADROOM_NUM) / (HEADROOM_DEN * EPOCH_SLOTS as u128);
    n.max(1) as u64
}

/// The process-wide `EpochState` for a unified root, created with `cap_bytes`
/// on first use (later calls ignore `cap_bytes`), with its purge thread.
pub(crate) fn shared_for_unified_root(unified_root: &Path, cap_bytes: u64) -> Arc<EpochState> {
    static REGISTRY: OnceLock<Mutex<HashMap<PathBuf, Arc<EpochState>>>> = OnceLock::new();
    let mut g = REGISTRY.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap();
    if let Some(s) = g.get(unified_root) {
        return s.clone();
    }
    let state = Arc::new(EpochState::new(cap_bytes));
    log::info!(
        target: super::purge::LOG_TARGET,
        "decoded chunk budget for {}: {} MiB",
        unified_root.display(),
        cap_bytes >> 20
    );
    g.insert(unified_root.to_path_buf(), state.clone());
    let weak = Arc::downgrade(&state);
    let signal = state.purge_signal.clone();
    std::thread::Builder::new()
        .name("vesuvius-cache-purge".into())
        .spawn(move || purge_loop(weak, signal))
        .expect("spawn purge thread");
    state
}

fn purge_loop(weak: Weak<EpochState>, signal: Arc<(Mutex<bool>, Condvar)>) {
    loop {
        {
            let g = signal.0.lock().unwrap();
            let (mut g, _) = signal.1.wait_timeout_while(g, PURGE_INTERVAL, |wake| !*wake).unwrap();
            *g = false;
        }
        let Some(state) = weak.upgrade() else { return };
        state.run_purge_pass("budget");
    }
}

fn log_plan(plan: PurgePlan, current: u8, summaries: &[VolumeBreakdown]) {
    let mut sorted: Vec<&VolumeBreakdown> = summaries.iter().filter(|b| b.resident > 0).collect();
    sorted.sort_by(|a, b| b.victims.cmp(&a.victims).then_with(|| b.resident.cmp(&a.resident)));
    let mut lines = format!(
        "purge plan: current_epoch={} threshold={} target={} expected_freed={}",
        current, plan.age_threshold, plan.target_chunks, plan.freed_chunks
    );
    for b in &sorted {
        let pct = b.victims.saturating_mul(100) / b.resident;
        let oldest = b.oldest_age.map(|a| a.to_string()).unwrap_or_else(|| "-".to_string());
        lines.push_str(&format!(
            "\n  vol={} resident={} oldest_age={} victims={} ({}%)",
            b.volume_id, b.resident, oldest, b.victims, pct
        ));
    }
    log::info!(target: super::purge::LOG_TARGET, "{}", lines);
}

#[cfg(test)]
impl EpochState {
    /// Test-only: advance `current` by `n` epochs without planting a chunk.
    pub fn force_advance(&self, n: u8) {
        let _s = self.inner.lock().unwrap();
        let next = self.current.load(Ordering::Relaxed).wrapping_add(n);
        self.current.store(next, Ordering::Relaxed);
    }

    /// Test-only single-shot access accounting (not race-safe).
    pub fn record_access(&self, old_epoch: u8) -> u8 {
        let cur = self.current.load(Ordering::Relaxed);
        if old_epoch != cur {
            self.record_access_transition(old_epoch, cur);
        }
        cur
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_per_epoch_is_capx13_div_256() {
        let cap = 200u64 << 30;
        assert_eq!(derive_bytes_per_epoch(cap), (cap as u128 * 13 / 10 / 256) as u64);
    }

    #[test]
    fn record_fill_advances_after_threshold() {
        let s = EpochState::new(1024);
        let initial = s.current();
        for _ in 0..10 {
            s.record_fill(100);
        }
        assert_ne!(s.current(), initial, "expected epoch advance after fills");
        assert_eq!(s.total_chunks(), 10);
    }

    #[test]
    fn record_access_moves_histogram_bucket() {
        let s = EpochState::new(1 << 20);
        let cur0 = s.record_fill(0);
        let cur1 = s.record_fill(s.bytes_per_epoch() + 1);
        assert_ne!(cur0, cur1, "expected one epoch advance");
        assert_eq!(s.record_access(cur0), cur1);
        let h = s.epoch_chunks();
        assert_eq!(h[cur0 as usize], 0);
        assert_eq!(h[cur1 as usize], 2);
    }

    /// Racing touches on one slot move the histogram exactly once.
    #[test]
    fn cas_arbitrated_touch_no_histogram_drift() {
        use super::super::sidecar::{Header, Sidecar, STATE_RESIDENT};

        let sidecar = Arc::new(Sidecar::empty(Header::new("v".into(), [256, 256, 256], 0)));
        sidecar.set_state(0, 0, STATE_RESIDENT);
        let epoch = Arc::new(EpochState::new(1 << 30));
        let source = epoch.record_fill(0);
        sidecar.set_access_epoch(0, 0, source);
        epoch.force_advance(42);
        let target = epoch.current();
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let (sidecar, epoch) = (sidecar.clone(), epoch.clone());
                std::thread::spawn(move || {
                    for _ in 0..200 {
                        let current = epoch.current();
                        let old = sidecar.get_access_epoch(0, 0);
                        if old != current && sidecar.compare_exchange_access_epoch(0, 0, old, current).is_ok() {
                            epoch.record_access_transition(old, current);
                        }
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let h = epoch.epoch_chunks();
        assert_eq!(h[target as usize], 1);
        assert_eq!(h.iter().map(|&v| v as u64).sum::<u64>(), 1);
    }
}
