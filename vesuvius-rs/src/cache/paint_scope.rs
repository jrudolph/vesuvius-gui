//! Per-paint dependency capture: which chunks a paint wanted at its target
//! LOD but couldn't use yet (it fell back to a coarser LOD, a preview, or
//! nothing).
//!
//! Wrap a paint in [`capture`]; every unified-cache lookup on this thread
//! that misses its target chunk records it. An empty report means the paint
//! drew final data everywhere — Empty chunks and out-of-volume samples count
//! as complete.
//!
//! Thread-local on purpose (for now): a tile render runs on one thread with
//! its own volume instance, so no paint signature has to change and wrappers
//! (TifXyz, Obj, overlays) need no forwarding. A paint that fanned out to
//! other threads would under-report. To be replaced by an explicit paint
//! context argument — see plans/demand-driven-loading.md.

use super::state::ChunkKey;
use std::cell::RefCell;

/// A chunk a paint needed at its target LOD, tagged with the cache it
/// belongs to (`ChunkCache::id`) since one paint can read several volumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MissingChunk {
    pub cache: usize,
    pub key: ChunkKey,
}

#[derive(Debug, Default, Clone)]
pub struct PaintReport {
    pub missing: fxhash::FxHashSet<MissingChunk>,
}

impl PaintReport {
    /// True when the paint used final data everywhere.
    pub fn is_complete(&self) -> bool {
        self.missing.is_empty()
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
        let report = SCOPE.with(|s| std::mem::replace(&mut *s.borrow_mut(), prev)).unwrap_or_default();
        // Nested scopes also count towards the outer one.
        SCOPE.with(|s| {
            if let Some(outer) = s.borrow_mut().as_mut() {
                outer.missing.extend(report.missing.iter().copied());
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

/// Run `f` (a paint) and report the target chunks it couldn't use.
pub fn capture<R>(f: impl FnOnce() -> R) -> (R, PaintReport) {
    let prev = SCOPE.with(|s| s.replace(Some(PaintReport::default())));
    let mut guard = ScopeGuard { prev: Some(prev) };
    let result = f();
    let report = guard.finish();
    (result, report)
}

/// Record a missed target chunk in the current scope, if any.
pub(super) fn record(cache: usize, key: ChunkKey) {
    SCOPE.with(|s| {
        if let Some(report) = s.borrow_mut().as_mut() {
            report.missing.insert(MissingChunk { cache, key });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_only_inside_scope_and_merges_nested() {
        record(1, ChunkKey::new(0, 0, 0, 0)); // no scope: dropped
        let ((), outer) = capture(|| {
            record(1, ChunkKey::new(0, 1, 0, 0));
            let ((), inner) = capture(|| record(2, ChunkKey::new(1, 0, 0, 0)));
            assert_eq!(inner.missing.len(), 1);
            record(1, ChunkKey::new(0, 1, 0, 0)); // duplicate
        });
        assert_eq!(outer.missing.len(), 2);
        let ((), empty) = capture(|| ());
        assert!(empty.is_complete());
    }
}
