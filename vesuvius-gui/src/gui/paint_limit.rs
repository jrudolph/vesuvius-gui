//! Bound on concurrently painting tiles.
//!
//! Tile renders run on tokio's blocking pool (`spawn_blocking`), which grows
//! to 512 threads: without a bound, every visible tile paints at once. That
//! oversubscribes the cores, and every tile's chunks have to be resident at
//! the same time. With the bound, the decoded working set is about
//! `limit × chunks per tile`.
//!
//! Waiters are served in arrival order, which follows the order tiles are
//! scheduled (most urgent first). `VESUVIUS_PAINT_THREADS` sets the limit
//! (default: available parallelism; `0`: unlimited, still counted).

use std::sync::{Condvar, Mutex, OnceLock};

pub const PAINT_THREADS_ENV_VAR: &str = "VESUVIUS_PAINT_THREADS";

#[derive(Default, Debug, Clone, Copy)]
pub struct PaintLimitStats {
    pub limit: usize,
    pub active: usize,
    pub waiting: usize,
    pub peak_active: usize,
    pub peak_waiting: usize,
}

struct State {
    next_ticket: u64,
    /// Lowest ticket not yet admitted.
    serving: u64,
    stats: PaintLimitStats,
}

struct PaintLimit {
    state: Mutex<State>,
    cv: Condvar,
}

fn limit() -> &'static PaintLimit {
    static L: OnceLock<PaintLimit> = OnceLock::new();
    L.get_or_init(|| {
        let limit = match std::env::var(PAINT_THREADS_ENV_VAR) {
            Ok(v) => v
                .trim()
                .parse()
                .unwrap_or_else(|e| panic!("{} must be a number: {}", PAINT_THREADS_ENV_VAR, e)),
            Err(_) => std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8),
        };
        PaintLimit {
            state: Mutex::new(State {
                next_ticket: 0,
                serving: 0,
                stats: PaintLimitStats {
                    limit,
                    ..Default::default()
                },
            }),
            cv: Condvar::new(),
        }
    })
}

/// Held while a tile paints.
pub struct PaintPermit(());

impl Drop for PaintPermit {
    fn drop(&mut self) {
        let l = limit();
        l.state.lock().unwrap().stats.active -= 1;
        l.cv.notify_all();
    }
}

/// Block until this tile may paint.
pub fn acquire() -> PaintPermit {
    let l = limit();
    let mut s = l.state.lock().unwrap();
    let ticket = s.next_ticket;
    s.next_ticket += 1;
    s.stats.waiting += 1;
    s.stats.peak_waiting = s.stats.peak_waiting.max(s.stats.waiting);
    while !(ticket == s.serving && (s.stats.limit == 0 || s.stats.active < s.stats.limit)) {
        s = l.cv.wait(s).unwrap();
    }
    s.serving += 1;
    s.stats.waiting -= 1;
    s.stats.active += 1;
    s.stats.peak_active = s.stats.peak_active.max(s.stats.active);
    drop(s);
    // The next ticket may be admissible right away.
    l.cv.notify_all();
    PaintPermit(())
}

pub fn stats() -> PaintLimitStats {
    limit().state.lock().unwrap().stats
}
