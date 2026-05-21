//! Priority + viewport hints for the cache's bounded work queues.
//!
//! The paint loop publishes a `Viewport` per frame: for each LOD it visits,
//! the chunk-space rectangle currently on screen plus the chunk-space center.
//! The cache and downloader use that to:
//!
//!   1. **Order** queued work — coarse LOD first, then closest-to-center.
//!   2. **Prune** queued work that's gone stale — anything more than a few
//!      chunks outside the current viewport, or older than `MAX_AGE`.
//!
//! Ordering is encoded as a single `u64` so the work-queue's BTreeMap can sort
//! everything in one key. Lower numeric value = higher actual priority.

use super::state::ChunkKey;
use std::time::Duration;

/// Queued work older than this is treated as stale and dropped at pop time.
pub const MAX_AGE: Duration = Duration::from_secs(10);

/// Packed priority value. Smaller is more urgent.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Priority(u64);

impl Priority {
    /// Encode `(lod_rank, distance)` into a sort key.
    ///
    /// - `lod_rank` is `max_lod - lod`, so the coarsest LOD has the smallest
    ///   rank and wins. We give it the top 16 bits.
    /// - `distance` is chunks-from-center at the entry's own LOD (Chebyshev).
    ///   Closer to the screen center → smaller value → wins ties on LOD.
    pub fn new(lod_rank: u16, distance_chunks: u32) -> Self {
        Self(((lod_rank as u64) << 32) | (distance_chunks as u64))
    }

    /// Worst-possible priority — used for un-prioritized fallback inserts
    /// (e.g. `get()` triggers a dispatch with no viewport info).
    pub fn worst() -> Self {
        Self(u64::MAX)
    }

    pub fn value(self) -> u64 {
        self.0
    }
}

/// Per-LOD slice of the current viewport, in chunk coordinates at that LOD.
#[derive(Clone, Debug)]
pub struct LodView {
    /// Center of the painted area in chunks at this LOD. The plane axis
    /// (perpendicular to the painted slice) has a single chunk value; the two
    /// in-plane axes carry the geometric center.
    pub center: [i32; 3],
    /// Inclusive lo/hi bounds of the chunk-space rectangle currently visible
    /// at this LOD. Out-of-plane axes use lo==hi==the slice's chunk index.
    pub rect_lo: [i32; 3],
    pub rect_hi: [i32; 3],
}

/// Viewport snapshot for all LODs the current paint pass visits.
///
/// Indexing: `per_lod[lod]`. Missing LODs (the paint pass didn't visit them,
/// or they're out of range) carry `None`, which treats everything at that LOD
/// as "very far away" so it gets pruned and de-prioritized.
#[derive(Clone, Debug, Default)]
pub struct Viewport {
    /// Highest LOD level the volume has — used to compute `lod_rank` so the
    /// coarsest level sorts first.
    pub max_lod: u8,
    pub per_lod: Vec<Option<LodView>>,
}

impl Viewport {
    pub fn empty() -> Self {
        Self::default()
    }

    /// True when the viewport carries any per-LOD information. The default
    /// `empty()` viewport is inactive — callers (tests, direct-API users)
    /// who never publish a viewport want distance pruning effectively off.
    pub fn is_active(&self) -> bool {
        !self.per_lod.is_empty()
    }

    /// Chebyshev distance, in chunks, from `key` to the viewport rect at its
    /// own LOD. Returns 0 if `key` lies inside the rect. If the viewport is
    /// inactive, returns 0 everywhere (no pruning). If the viewport is
    /// active but this LOD wasn't visited this pass, returns `u32::MAX` so
    /// callers can prune it.
    pub fn distance_outside(&self, key: ChunkKey) -> u32 {
        if !self.is_active() {
            return 0;
        }
        let Some(view) = self.lod_view(key.lod) else {
            return u32::MAX;
        };
        let p = [key.x as i32, key.y as i32, key.z as i32];
        let mut d = 0i32;
        for ax in 0..3 {
            let dx = (view.rect_lo[ax] - p[ax]).max(p[ax] - view.rect_hi[ax]).max(0);
            d = d.max(dx);
        }
        d as u32
    }

    /// Chebyshev distance, in chunks, from `key` to the viewport center at
    /// its own LOD. Used to break LOD-ties when sorting work. Same inactive-
    /// viewport semantics as `distance_outside`.
    pub fn distance_to_center(&self, key: ChunkKey) -> u32 {
        if !self.is_active() {
            return 0;
        }
        let Some(view) = self.lod_view(key.lod) else {
            return u32::MAX;
        };
        let p = [key.x as i32, key.y as i32, key.z as i32];
        let mut d = 0i32;
        for ax in 0..3 {
            let dx = (p[ax] - view.center[ax]).abs();
            d = d.max(dx);
        }
        d as u32
    }

    pub fn priority_for(&self, key: ChunkKey) -> Priority {
        let lod_rank = (self.max_lod as i32 - key.lod as i32).max(0) as u16;
        let dist = self.distance_to_center(key);
        // Saturate dist into the low 32 bits; ridiculously far chunks all
        // collapse onto the worst tier.
        Priority::new(lod_rank, dist)
    }

    fn lod_view(&self, lod: u8) -> Option<&LodView> {
        self.per_lod.get(lod as usize).and_then(|v| v.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_viewport() -> Viewport {
        // max_lod = 3. LOD 0 viewport = 4 chunks centered at (10, 20, 30).
        let mut per_lod = vec![None; 4];
        per_lod[0] = Some(LodView {
            center: [10, 20, 30],
            rect_lo: [8, 18, 30],
            rect_hi: [12, 22, 30],
        });
        per_lod[1] = Some(LodView {
            center: [5, 10, 15],
            rect_lo: [4, 9, 15],
            rect_hi: [6, 11, 15],
        });
        Viewport { max_lod: 3, per_lod }
    }

    #[test]
    fn distance_outside_inside_rect_is_zero() {
        let v = make_viewport();
        assert_eq!(v.distance_outside(ChunkKey::new(0, 10, 20, 30)), 0);
        assert_eq!(v.distance_outside(ChunkKey::new(0, 8, 18, 30)), 0);
    }

    #[test]
    fn distance_outside_grows_with_distance() {
        let v = make_viewport();
        assert_eq!(v.distance_outside(ChunkKey::new(0, 15, 20, 30)), 3); // 3 chunks past hi_x
        assert_eq!(v.distance_outside(ChunkKey::new(0, 8, 25, 30)), 3); // 3 chunks past hi_y
    }

    #[test]
    fn priority_orders_coarse_before_fine() {
        let v = make_viewport();
        // Same chunk position but different LODs. Coarse should win.
        let coarse = v.priority_for(ChunkKey::new(2, 100, 100, 100));
        let fine = v.priority_for(ChunkKey::new(0, 10, 20, 30));
        assert!(coarse < fine, "coarse {:?} should be < fine {:?}", coarse, fine);
    }

    #[test]
    fn priority_orders_center_before_edge_same_lod() {
        let v = make_viewport();
        let center = v.priority_for(ChunkKey::new(0, 10, 20, 30));
        let edge = v.priority_for(ChunkKey::new(0, 12, 22, 30));
        assert!(center < edge, "center {:?} should be < edge {:?}", center, edge);
    }

    #[test]
    fn missing_lod_gets_worst_distance_when_viewport_is_active() {
        let v = make_viewport();
        // LOD 2 has no view → distance_outside = u32::MAX.
        assert_eq!(v.distance_outside(ChunkKey::new(2, 0, 0, 0)), u32::MAX);
        assert_eq!(v.distance_to_center(ChunkKey::new(2, 0, 0, 0)), u32::MAX);
    }

    #[test]
    fn inactive_viewport_disables_pruning() {
        // Default empty viewport must NOT prune — tests + direct-API callers
        // that never publish one would otherwise see every chunk rejected.
        let v = Viewport::empty();
        assert!(!v.is_active());
        assert_eq!(v.distance_outside(ChunkKey::new(0, 999, 999, 999)), 0);
        assert_eq!(v.distance_to_center(ChunkKey::new(5, 1, 2, 3)), 0);
    }
}
