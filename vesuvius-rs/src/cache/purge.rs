//! Cache purge planning.
//!
//! Goal: free at least `target_chunks` chunks across the cache by evicting
//! the oldest chunks, where "oldest" is defined by the access-epoch column
//! in the sidecar (see `epoch.rs`).
//!
//! Two-pass design — chosen so we don't have to maintain per-purge
//! histograms separately from the global `epoch_chunks` snapshot:
//!
//! 1. **Plan.** Walk the global `EpochState::epoch_chunks()` histogram in
//!    modular age order (oldest first, starting at `(current + 1) % 256`).
//!    Accumulate until we cross the target; the threshold age T is the
//!    cutoff.
//! 2. **Sweep.** Walk every per-volume Sidecar; for each chunk with
//!    `access_epoch` in the "older than T" set, transition the state
//!    bitmap to Missing (Release) and release the slot's memory.
//!
//! The state-bitmap transition happens BEFORE the release on purpose —
//! a reader that already passed the Resident check before the bitmap was
//! updated may read zeros, which our consumers tolerate as a transient
//! glitch (re-validates on the next frame). The reverse order would let a
//! reader return Resident bytes that have already been deallocated, which
//! is a permanent corruption window. Always: bitmap-first, then release.
//!
//! This module holds the planner and the `PurgeTarget` contract; the
//! trigger lives in `epoch.rs` (budget high-water mark) and the sweep itself
//! in `cache.rs` (`Inner::run_purge`).

use super::epoch::{EpochState, EPOCH_SLOTS};

/// Shared `log` target for every purge / epoch / cache-cap message. Set
/// to the path of this module so a single `RUST_LOG=vesuvius_rs::cache::purge=info`
/// filter covers all cache lifecycle output (planning, sweep, watchdog,
/// stats, offline sweep, env parse). The producers live in `cache.rs`
/// and `epoch.rs`; routing them through this constant keeps the filter
/// stable when code moves between files.
pub const LOG_TARGET: &str = "vesuvius_rs::cache::purge";

/// A planned purge: chunks whose age (relative to `current`) is `>= T`
/// will be evicted. `target_chunks` is the goal; `freed_chunks` is what
/// the histogram says we'd actually free at this threshold (may exceed
/// `target_chunks` because we can only cut at slot boundaries).
#[derive(Debug, Clone, Copy)]
pub struct PurgePlan {
    /// Modular age (in epochs) above which a chunk is a victim. A chunk
    /// at epoch E with current C has age `(C.wrapping_sub(E)) as usize`.
    pub age_threshold: u16,
    pub target_chunks: u64,
    pub freed_chunks: u64,
}

impl PurgePlan {
    /// Build a plan against the current global histogram. Returns None if
    /// the cache holds fewer chunks than `target_chunks` (purge cannot
    /// satisfy the request — caller decides whether to over-evict or
    /// adjust the target).
    pub fn build(epoch: &EpochState, target_chunks: u64) -> Option<PurgePlan> {
        let hist = epoch.epoch_chunks();
        let current = epoch.current();
        plan_from_histogram(&hist, current, target_chunks)
    }

    /// True if `victim_epoch` falls into the eviction set under this plan.
    pub fn is_victim(&self, victim_epoch: u8, current: u8) -> bool {
        let age = current.wrapping_sub(victim_epoch) as u16;
        age >= self.age_threshold
    }
}

/// Pure-function planner — split out so it's unit-testable without
/// needing a real `EpochState`. Iterates from the oldest slot
/// (`(current + 1) % 256`) toward the youngest (`current`), accumulating
/// until `target_chunks` is reached. Returns the modular age threshold T
/// such that all chunks with age `>= T` are victims.
pub fn plan_from_histogram(
    hist: &[u32; EPOCH_SLOTS],
    current: u8,
    target_chunks: u64,
) -> Option<PurgePlan> {
    let total: u64 = hist.iter().map(|c| *c as u64).sum();
    if total < target_chunks {
        return None;
    }

    let mut freed: u64 = 0;
    // Walk oldest -> youngest. Age 255 = the slot just past `current`
    // (oldest), age 1 = the slot one back from `current`. We don't
    // include the current slot itself in the eviction set.
    for age in (1..=255u16).rev() {
        let slot = current.wrapping_sub(age as u8) as usize;
        freed += hist[slot] as u64;
        if freed >= target_chunks {
            return Some(PurgePlan {
                age_threshold: age,
                target_chunks,
                freed_chunks: freed,
            });
        }
    }
    // Couldn't reach target without evicting the current epoch — refuse.
    None
}

// The sweep lives on `ChunkCache::purge_to_target` (in `cache.rs`)
// because it also needs to invalidate `Inner::map` for evicted keys —
// otherwise readers holding `ChunkState::Resident` Arcs would keep
// reading zeros from the punched mmap forever. Pure planning stays here
// so it can be tested without a full cache.

/// Anything the watchdog can ask to evict chunks under a shared plan.
/// `ChunkCache::Inner` implements this; the watchdog holds `Weak`s and
/// upgrades them on each tick so dropped caches are GC'd naturally.
pub trait PurgeTarget: Send + Sync {
    /// Identifier matching `Sidecar::header.volume_id`. Used by the
    /// offline sweep to skip volumes already covered by a live target.
    fn volume_id(&self) -> String;
    /// Read-only count of how this volume would contribute to `plan`.
    /// Walks the same sidecar that `run_purge` would, but accumulates
    /// counts instead of evicting. Called once per watchdog tick to log
    /// the per-volume plan breakdown so the human reader can tell which
    /// volume is dominating the eviction set.
    fn summarize(&self, plan: PurgePlan, current: u8) -> VolumeBreakdown;
    fn run_purge(&self, plan: PurgePlan) -> u64;
}

/// One volume's contribution to a planned purge. Aggregated and logged
/// by `EpochState::purge_all_to_target` so the operator can see *where*
/// the planner's `expected_freed` count lives before any chunks are
/// touched.
#[derive(Debug, Clone)]
pub struct VolumeBreakdown {
    pub volume_id: String,
    /// Total Resident chunks across every LOD.
    pub resident: u64,
    /// Modular age of the oldest Resident chunk. `None` when the volume
    /// has nothing Resident.
    pub oldest_age: Option<u16>,
    /// Resident chunks matching `plan.is_victim(access_epoch, current)`.
    /// Does not account for shard-level sparing — that's a sweep-time
    /// concern; this number is the *planner's* view of available
    /// victims in this volume.
    pub victims: u64,
}

impl VolumeBreakdown {
    /// Build a breakdown by walking `sidecar` once. Shared between the
    /// live `PurgeTarget::summarize` impl and the offline path.
    pub fn from_sidecar(sidecar: &super::sidecar::Sidecar, plan: PurgePlan, current: u8) -> Self {
        let mut resident: u64 = 0;
        let mut victims: u64 = 0;
        let mut oldest_age: Option<u16> = None;
        for (lod_idx, dims) in sidecar.header.lods.iter().enumerate() {
            let lod = lod_idx as u8;
            for idx in 0..dims.count() {
                if sidecar.get_state(lod, idx) != super::sidecar::STATE_RESIDENT {
                    continue;
                }
                resident += 1;
                let ae = sidecar.get_access_epoch(lod, idx);
                let age = current.wrapping_sub(ae) as u16;
                if oldest_age.map_or(true, |o| age > o) {
                    oldest_age = Some(age);
                }
                if plan.is_victim(ae, current) {
                    victims += 1;
                }
            }
        }
        Self {
            volume_id: sidecar.header.volume_id.clone(),
            resident,
            oldest_age,
            victims,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_picks_smallest_threshold_meeting_target() {
        // current = 100. Slots ..99,100 are recent; 101..=99 (wrap) is
        // oldest -> youngest in the walk.
        let mut hist = [0u32; EPOCH_SLOTS];
        hist[101] = 5; // age 255 (oldest)
        hist[102] = 10; // age 254
        hist[103] = 20; // age 253
        hist[100] = 1000; // current — never a victim

        let plan = plan_from_histogram(&hist, 100, 12).unwrap();
        // 5 + 10 = 15 >= 12 → threshold is age 254
        assert_eq!(plan.age_threshold, 254);
        assert_eq!(plan.freed_chunks, 15);
    }

    #[test]
    fn plan_refuses_if_target_exceeds_cache() {
        let hist = [1u32; EPOCH_SLOTS];
        // Total = 256, but the current slot (1 chunk) is excluded, so the
        // available pool is 255. Asking for 300 must refuse.
        assert!(plan_from_histogram(&hist, 0, 300).is_none());
    }

    #[test]
    fn is_victim_modular() {
        let plan = PurgePlan {
            age_threshold: 100,
            target_chunks: 0,
            freed_chunks: 0,
        };
        // current = 50, victim_epoch = 200 → age = 50.wrapping_sub(200) = 106
        assert!(plan.is_victim(200, 50));
        // current = 50, victim_epoch = 0 → age = 50, below threshold
        assert!(!plan.is_victim(0, 50));
    }
}
