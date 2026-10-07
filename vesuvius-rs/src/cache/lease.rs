//! Interest leases: who still wants a chunk, and how urgently.
//!
//! A [`Lease`] belongs to one consumer of chunk data — in the GUI, one tile.
//! The consumer renews it every frame it is on screen, with its current
//! priority (lower is more urgent). Chunks requested while painting under a
//! lease (see `paint_scope::capture`) remember it; the work queues ask, at
//! pop, whether any lease behind an entry is still live:
//!
//! - live → run in priority order (re-filed if its priority got worse),
//! - dead (not renewed for [`LIVE_FRAMES`] frames, or dropped) → cancelled;
//!   the chunk becomes requestable again,
//! - never leased (offline renderer, requests outside any paint scope) →
//!   durable, at [`UNLEASED`] priority.
//!
//! Liveness counts frames, not wall time, so an idle GUI (no frames) keeps
//! its requests.
//!
//! A lease only stands for what its owner *currently* needs: each paint
//! under it is a render generation, and a chunk stays wanted by the lease
//! only if the latest finished paint (or the one in progress) requested it
//! again. When a tile's target data lands, the coarser levels it had been
//! falling back on drop out of its interest with its next paint.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Weak};

pub type Priority = u32;

/// Priority of work no lease asked for.
pub const UNLEASED: Priority = Priority::MAX;

/// A lease not renewed for more than this many frames is dead.
pub const LIVE_FRAMES: u64 = 3;

static FRAME: AtomicU64 = AtomicU64::new(0);
static PRIORITY_GENERATION: AtomicU64 = AtomicU64::new(0);

/// Advance the lease clock to `frame` (monotonic; e.g. egui's pass number).
/// Several panes may report the same frame.
pub fn advance_frame(frame: u64) {
    FRAME.fetch_max(frame, Ordering::Relaxed);
}

fn frame() -> u64 {
    FRAME.load(Ordering::Relaxed)
}

/// Bumped whenever some lease's priority changes; the queues re-evaluate
/// their entries when they see it move.
pub(super) fn priority_generation() -> u64 {
    PRIORITY_GENERATION.load(Ordering::Relaxed)
}

pub struct Lease {
    renewed_frame: AtomicU64,
    priority: AtomicU32,
    /// Generation of the latest paint started under this lease.
    started: AtomicU64,
    /// Generation of the latest paint finished under this lease. Chunks
    /// attached before it are no longer wanted by this lease.
    finished: AtomicU64,
}

impl Lease {
    pub fn new(priority: Priority) -> Arc<Self> {
        Arc::new(Self {
            renewed_frame: AtomicU64::new(frame()),
            priority: AtomicU32::new(priority),
            started: AtomicU64::new(0),
            finished: AtomicU64::new(0),
        })
    }

    /// A paint starts under this lease; chunks it requests are attached at
    /// the returned generation.
    pub(super) fn begin_paint(&self) -> u64 {
        self.started.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// The paint of `generation` finished: from now on only chunks it (or a
    /// later paint) requested are wanted.
    pub(super) fn end_paint(&self, generation: u64) {
        self.finished.fetch_max(generation, Ordering::Relaxed);
    }

    /// Keep the lease live through the current frame at `priority`.
    /// Returns true if it had died: its queued work was cancelled, so the
    /// owner should request again.
    pub fn renew(&self, priority: Priority) -> bool {
        let now = frame();
        let was_dead = now.saturating_sub(self.renewed_frame.swap(now, Ordering::Relaxed)) > LIVE_FRAMES;
        if self.priority.swap(priority, Ordering::Relaxed) != priority {
            PRIORITY_GENERATION.fetch_add(1, Ordering::Relaxed);
        }
        was_dead
    }

    fn live_priority(&self) -> Option<Priority> {
        (frame().saturating_sub(self.renewed_frame.load(Ordering::Relaxed)) <= LIVE_FRAMES)
            .then(|| self.priority.load(Ordering::Relaxed))
    }
}

/// Whether queued work is still wanted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Liveness {
    /// Nobody leased it: keep it, at `UNLEASED` priority.
    Unleased,
    Live(Priority),
    /// Every lease behind it is gone or expired.
    Dead,
}

/// Evaluated by the work queues (outside their locks) to ask whether an
/// entry is still wanted.
pub type LivenessFn = Arc<dyn Fn() -> Liveness + Send + Sync>;

/// Liveness of work no lease will ever ask about.
pub fn unleased() -> LivenessFn {
    Arc::new(|| Liveness::Unleased)
}

impl Liveness {
    /// Combined interest of two sets of leases: the best live one wins; any
    /// leased-but-dead set makes an otherwise unleased combination dead.
    pub fn or(self, other: Liveness) -> Liveness {
        use Liveness::*;
        match (self, other) {
            (Live(a), Live(b)) => Live(a.min(b)),
            (Live(a), _) | (_, Live(a)) => Live(a),
            (Dead, _) | (_, Dead) => Dead,
            (Unleased, Unleased) => Unleased,
        }
    }

    pub fn priority(self) -> Option<Priority> {
        match self {
            Liveness::Unleased => Some(UNLEASED),
            Liveness::Live(p) => Some(p),
            Liveness::Dead => None,
        }
    }
}

/// The leases that asked for one chunk, each with the generation of the
/// lease's paint that asked last.
#[derive(Default)]
pub(super) struct Interest {
    leases: Vec<(Weak<Lease>, u64)>,
}

impl Interest {
    /// Record that `lease`'s latest started paint wants the chunk.
    pub fn attach(&mut self, lease: &Arc<Lease>) {
        let weak = Arc::downgrade(lease);
        let generation = lease.started.load(Ordering::Relaxed);
        match self.leases.iter_mut().find(|(l, _)| l.ptr_eq(&weak)) {
            Some((_, g)) => *g = (*g).max(generation),
            None => {
                self.leases.retain(|(l, _)| l.strong_count() > 0);
                self.leases.push((weak, generation));
            }
        }
    }

    /// `Live` with the best priority among live leases that still want the
    /// chunk, else `Dead`.
    pub fn liveness(&self) -> Liveness {
        self.leases
            .iter()
            .filter_map(|(l, generation)| {
                let lease = l.upgrade()?;
                (*generation >= lease.finished.load(Ordering::Relaxed)).then_some(())?;
                lease.live_priority()
            })
            .min()
            .map_or(Liveness::Dead, Liveness::Live)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // FRAME is process-global; this is the only test that moves it.
    #[test]
    fn leases_die_after_live_frames_and_revive_on_renew() {
        let base = frame() + 100;
        advance_frame(base);
        let near = Lease::new(1);
        let far = Lease::new(5);
        let mut interest = Interest::default();
        interest.attach(&far);
        interest.attach(&near);
        interest.attach(&near); // deduped
        assert_eq!(interest.leases.len(), 2);
        assert_eq!(interest.liveness(), Liveness::Live(1));

        advance_frame(base + LIVE_FRAMES);
        assert!(!far.renew(5));
        assert_eq!(interest.liveness(), Liveness::Live(1));
        advance_frame(base + LIVE_FRAMES + 1);
        // `near` expired; `far` (renewed one frame later) still live.
        assert_eq!(interest.liveness(), Liveness::Live(5));
        drop(far);
        assert_eq!(interest.liveness(), Liveness::Dead);
        assert!(near.renew(1), "renewing an expired lease reports it died");
        assert_eq!(interest.liveness(), Liveness::Live(1));

        // Only what the latest finished paint (or the running one) asked for
        // stays wanted.
        let coarse = {
            let mut i = Interest::default();
            i.attach(&near);
            i
        };
        let mut target = Interest::default();
        let g = near.begin_paint();
        target.attach(&near);
        assert_eq!(coarse.liveness(), Liveness::Live(1), "still wanted while the paint runs");
        near.end_paint(g);
        assert_eq!(coarse.liveness(), Liveness::Dead, "not requested by the finished paint");
        assert_eq!(target.liveness(), Liveness::Live(1));

        assert_eq!(Liveness::Unleased.or(Liveness::Dead), Liveness::Dead);
        assert_eq!(Liveness::Dead.or(Liveness::Live(3)), Liveness::Live(3));
        assert_eq!(Liveness::Unleased.or(Liveness::Unleased), Liveness::Unleased);
    }
}
