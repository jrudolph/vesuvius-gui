//! Per-paint dependency capture: which chunks a paint wanted at its target
//! LOD but couldn't use yet (it fell back to a coarser LOD, a preview, or
//! nothing) — and, after the paint, when it is worth repeating.
//!
//! Wrap a paint in [`capture`]; every unified-cache lookup on this thread
//! that misses its target chunk records it. An empty report means the paint
//! drew final data everywhere — Empty chunks and out-of-volume samples count
//! as complete. Under a [`Lease`], every chunk the paint requests (target or
//! coarser) records the lease as wanting it, which keeps its fetch queued
//! at the lease's priority (see `lease.rs`).
//!
//! A kept report then drives refreshes instead of a re-render timer:
//! [`PaintReport::poll`] says whether one of its missing chunks (or a finer
//! fallback than the one used) has landed since the paint started, and
//! optionally re-requests missing chunks whose fetch failed and whose
//! cooldown is over — nothing else re-paints them meanwhile. Landings bump a global epoch, so
//! polling is a single atomic load until something actually arrives, and
//! call the [`set_landing_hook`] (rate-limited) so an idle GUI wakes up.
//!
//! Thread-local on purpose (for now): a tile render runs on one thread with
//! its own volume instance, so no paint signature has to change and wrappers
//! (TifXyz, Obj, overlays) need no forwarding. A paint that fanned out to
//! other threads would under-report. To be replaced by an explicit paint
//! context argument — see plans/demand-driven-loading.md.

use super::cache::ChunkCache;
use super::lease::Lease;
use super::state::ChunkKey;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::{Duration, Instant};

/// A chunk a paint needed at its target LOD, tagged with the cache it
/// belongs to (`ChunkCache::id`) since one paint can read several volumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MissingChunk {
    pub cache: usize,
    pub key: ChunkKey,
}

/// How many levels above a missing chunk `poll` looks for a better
/// fallback when the paint drew nothing for it.
const MAX_WATCHED_LEVELS: u8 = 6;

#[derive(Default)]
pub struct PaintReport {
    /// Missing target chunk → the coarsest LOD the paint fell back to for
    /// it (`None`: drew nothing). Levels strictly between the two would
    /// improve the paint too.
    pub missing: fxhash::FxHashMap<MissingChunk, Option<u8>>,
    caches: Vec<ChunkCache>,
    /// The lease the paint ran under, and the paint's generation in it.
    lease: Option<Arc<Lease>>,
    generation: u64,
    /// (cache id, chunk) pairs that already recorded `lease` during the
    /// paint. Emptied when the paint ends.
    attached: fxhash::FxHashSet<(usize, ChunkKey)>,
    /// The paint was cancelled before it ran. Incomplete, but `poll` never
    /// asks to repeat it (nobody shows it).
    failed: bool,
    /// Landing epoch already checked (starts at the paint's start).
    seen_epoch: AtomicU64,
    /// Last renewal, ms since `clock_origin`.
    renewed_at_ms: AtomicU64,
}

impl PaintReport {
    /// Report for a paint that didn't finish.
    pub fn failed() -> Self {
        Self { failed: true, ..Default::default() }
    }

    /// True when the paint used final data everywhere.
    pub fn is_complete(&self) -> bool {
        !self.failed && self.missing.is_empty()
    }

    pub fn is_failed(&self) -> bool {
        self.failed
    }

    /// Missing target chunks (a failed paint counts as one).
    pub fn missing_count(&self) -> usize {
        self.missing.len() + self.failed as usize
    }

    fn cache(&self, id: usize) -> Option<&ChunkCache> {
        self.caches.iter().find(|c| c.id() == id)
    }

    /// Whether repainting now would draw something better than this
    /// report's paint: a missing chunk, or a finer fallback for one, has
    /// landed since. Cheap when nothing landed anywhere (one atomic load).
    ///
    /// Every `renew_every`, also re-requests the missing chunks under the
    /// paint's lease (`ChunkCache::renew`): ones whose fetch failed are
    /// dispatched again once their cooldown is over. Call it only while the
    /// painted region is on screen.
    pub fn poll(&self, renew_every: Duration) -> bool {
        if self.failed || self.missing.is_empty() {
            return false;
        }
        let now_ms = clock_ms();
        let last = self.renewed_at_ms.load(Ordering::Relaxed);
        let renew = self.lease.is_some()
            && now_ms.saturating_sub(last) >= renew_every.as_millis() as u64
            && self
                .renewed_at_ms
                .compare_exchange(last, now_ms, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok();
        let epoch = LANDED_EPOCH.load(Ordering::Acquire);
        let landed = self.seen_epoch.swap(epoch, Ordering::Relaxed) != epoch;
        if !landed && !renew {
            return false;
        }
        let terminal = |cache: &ChunkCache, key| cache.peek(key).is_some_and(|s| s.is_terminal());
        for (m, fallback) in &self.missing {
            let cache = self.cache(m.cache).expect("missing chunk's cache was recorded with it");
            let k = m.key;
            if let (true, Some(lease)) = (renew, &self.lease) {
                if cache.renew(k, lease).is_terminal() {
                    return true;
                }
            } else if terminal(cache, k) {
                return true;
            }
            if landed {
                let upto = fallback.unwrap_or(k.lod.saturating_add(MAX_WATCHED_LEVELS + 1));
                for lod in k.lod.saturating_add(1)..upto {
                    let s = lod - k.lod;
                    if terminal(cache, ChunkKey::new(lod, k.x >> s, k.y >> s, k.z >> s)) {
                        return true;
                    }
                }
            }
        }
        false
    }
}

thread_local! {
    static SCOPE: RefCell<Option<PaintReport>> = const { RefCell::new(None) };
}

/// Restores the enclosing scope (merging this one's misses into it) even if
/// the paint panics.
struct ScopeGuard {
    prev: Option<Option<PaintReport>>,
}

impl ScopeGuard {
    fn finish(&mut self) -> PaintReport {
        let prev = self.prev.take().expect("finished once");
        let mut report = SCOPE.with(|s| std::mem::replace(&mut *s.borrow_mut(), prev)).unwrap_or_default();
        report.attached = Default::default();
        if let Some(lease) = &report.lease {
            lease.end_paint(report.generation);
        }
        // Nested scopes also count towards the outer one.
        SCOPE.with(|s| {
            if let Some(outer) = s.borrow_mut().as_mut() {
                for (m, f) in &report.missing {
                    merge(outer, *m, *f);
                }
                for c in &report.caches {
                    if outer.cache(c.id()).is_none() {
                        outer.caches.push(c.clone());
                    }
                }
            }
        });
        report
    }
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        if self.prev.is_some() {
            self.finish();
        }
    }
}

/// Run `f` (a paint) under `lease` and report the target chunks it
/// couldn't use.
pub fn capture<R>(lease: Option<Arc<Lease>>, f: impl FnOnce() -> R) -> (R, PaintReport) {
    let generation = lease.as_ref().map_or(0, |l| l.begin_paint());
    let fresh = PaintReport {
        lease,
        generation,
        // Anything landing from here on may have been missed by the paint.
        seen_epoch: AtomicU64::new(LANDED_EPOCH.load(Ordering::Acquire)),
        renewed_at_ms: AtomicU64::new(clock_ms()),
        ..Default::default()
    };
    let prev = SCOPE.with(|s| s.replace(Some(fresh)));
    let mut guard = ScopeGuard { prev: Some(prev) };
    let result = f();
    let report = guard.finish();
    (result, report)
}

fn merge(report: &mut PaintReport, m: MissingChunk, fallback: Option<u8>) {
    let e = report.missing.entry(m).or_insert(fallback);
    // Keep the worst fallback: anything finer than it can still help.
    *e = match (*e, fallback) {
        (Some(a), Some(b)) => Some(a.max(b)),
        _ => None,
    };
}

/// The current scope's lease, if `key` (in cache `cache`) hasn't recorded
/// it yet during this paint.
pub(super) fn lease_to_attach(cache: usize, key: ChunkKey) -> Option<Arc<Lease>> {
    SCOPE.with(|s| {
        let mut s = s.borrow_mut();
        let report = s.as_mut()?;
        let lease = report.lease.clone()?;
        report.attached.insert((cache, key)).then_some(lease)
    })
}

/// Record a missed target chunk in the current scope, if any. `fallback`
/// is the LOD drawn instead (`None`: nothing; the target LOD itself when
/// coarser levels can't improve this paint).
pub(super) fn record(cache: &ChunkCache, key: ChunkKey, fallback: Option<u8>) {
    SCOPE.with(|s| {
        if let Some(report) = s.borrow_mut().as_mut() {
            let id = cache.id();
            if report.cache(id).is_none() {
                report.caches.push(cache.clone());
            }
            merge(report, MissingChunk { cache: id, key }, fallback);
        }
    });
}

static LANDED_EPOCH: AtomicU64 = AtomicU64::new(0);
static LAST_HOOK_MS: AtomicU64 = AtomicU64::new(0);
static LANDING_HOOK: RwLock<Option<Arc<dyn Fn(Duration) + Send + Sync>>> = RwLock::new(None);

/// Landings call the hook at most once per this window, passing it; the
/// hook must repaint no sooner than that so later landings in the window
/// are included.
pub const LANDING_COALESCE: Duration = Duration::from_millis(20);

/// Install the hook called when chunks land (e.g. egui's
/// `request_repaint_after`), rate-limited to one call per
/// [`LANDING_COALESCE`] across all caches.
pub fn set_landing_hook(hook: Option<Arc<dyn Fn(Duration) + Send + Sync>>) {
    *LANDING_HOOK.write().unwrap() = hook;
}

fn clock_ms() -> u64 {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    // Offset by the window so the first landing always passes the limiter.
    ORIGIN.get_or_init(Instant::now).elapsed().as_millis() as u64 + LANDING_COALESCE.as_millis() as u64
}

/// A chunk became Resident or Empty (after its state is in the map).
pub(super) fn landed() {
    LANDED_EPOCH.fetch_add(1, Ordering::Release);
    let now = clock_ms();
    let last = LAST_HOOK_MS.load(Ordering::Relaxed);
    if now.saturating_sub(last) >= LANDING_COALESCE.as_millis() as u64
        && LAST_HOOK_MS
            .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
    {
        if let Some(hook) = LANDING_HOOK.read().unwrap().clone() {
            hook(LANDING_COALESCE);
        }
    }
}
