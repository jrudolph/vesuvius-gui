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
}

impl Lease {
    pub fn new(priority: Priority) -> Arc<Self> {
        Arc::new(Self {
            renewed_frame: AtomicU64::new(frame()),
            priority: AtomicU32::new(priority),
        })
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

/// The leases that asked for one chunk.
#[derive(Default)]
pub(super) struct Interest {
    leases: Vec<Weak<Lease>>,
}

impl Interest {
    pub fn attach(&mut self, lease: &Arc<Lease>) {
        let weak = Arc::downgrade(lease);
        if !self.leases.iter().any(|l| l.ptr_eq(&weak)) {
            self.leases.retain(|l| l.strong_count() > 0);
            self.leases.push(weak);
        }
    }

    /// `Live` with the best priority among live leases, else `Dead`.
    pub fn liveness(&self) -> Liveness {
        self.leases
            .iter()
            .filter_map(|l| l.upgrade()?.live_priority())
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

        assert_eq!(Liveness::Unleased.or(Liveness::Dead), Liveness::Dead);
        assert_eq!(Liveness::Dead.or(Liveness::Live(3)), Liveness::Live(3));
        assert_eq!(Liveness::Unleased.or(Liveness::Unleased), Liveness::Unleased);
    }
}
