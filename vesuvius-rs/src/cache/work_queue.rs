//! Priority work queue shared by the cache's task queue and the
//! downloader's job queue.
//!
//! Entries are ordered by priority (lower first; see `lease`), newest first
//! within a priority. Each non-durable entry carries a liveness function —
//! "which leases still want this, and how urgently?" — evaluated outside
//! the queue lock:
//!
//! - **at pop**, for the head entry: dead → cancelled (handed back to the
//!   caller); priority got worse → re-filed and the next head is tried;
//!   otherwise it runs;
//! - **for every entry**, at the next pop after any lease's priority
//!   changed (`lease::priority_generation`): re-filed under its current
//!   priority, dead ones cancelled. A static view never triggers this.
//!
//! Durable entries (an Extract whose sources are already downloaded) carry
//! no liveness: they always run, at their submit priority. Work no lease
//! asked for is `Unleased` and stays queued at `lease::UNLEASED` priority.
//!
//! Unbounded; dedup happens at the cache layer (source-key + chunk-key
//! uniqueness), so submission always succeeds.

use super::lease::{self, LivenessFn, Priority};
use super::state::ChunkKey;
use std::collections::BTreeMap;
use std::sync::{Condvar, Mutex};
use std::time::Instant;

/// Priority, then newest first.
type QueueKey = (Priority, u64);

pub(super) struct WorkQueue<T> {
    inner: Mutex<Inner<T>>,
    not_empty: Condvar,
}

struct Inner<T> {
    entries: BTreeMap<QueueKey, QueueEntry<T>>,
    next_seq: u64,
    /// `lease::priority_generation` the entries were last filed under.
    seen_generation: u64,
}

pub(super) struct QueueEntry<T> {
    /// The cache chunk this work was first submitted for. Telemetry.
    pub chunk: ChunkKey,
    pub submitted_at: Instant,
    /// How often the entry was re-filed under a worse priority. Telemetry.
    pub refiles: u32,
    key: QueueKey,
    /// `None`: durable.
    liveness: Option<LivenessFn>,
    pub item: T,
}

impl<T> WorkQueue<T> {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                entries: BTreeMap::new(),
                next_seq: 0,
                seen_generation: lease::priority_generation(),
            }),
            not_empty: Condvar::new(),
        }
    }

    /// Submit work for `chunk`, kept while `liveness` says it is wanted.
    pub fn submit(&self, chunk: ChunkKey, liveness: LivenessFn, item: T) {
        // A request that is dead on arrival still runs its course through
        // pop, which cancels it like any other dead entry.
        let priority = liveness().priority().unwrap_or(lease::UNLEASED);
        self.insert(chunk, priority, Some(liveness), item);
    }

    /// Submit work that always runs, at `priority` (see module docs).
    pub fn submit_durable(&self, chunk: ChunkKey, priority: Priority, item: T) {
        self.insert(chunk, priority, None, item);
    }

    fn insert(&self, chunk: ChunkKey, priority: Priority, liveness: Option<LivenessFn>, item: T) {
        let mut q = self.inner.lock().unwrap();
        q.next_seq += 1;
        let key = (priority, !q.next_seq);
        q.entries.insert(
            key,
            QueueEntry {
                chunk,
                submitted_at: Instant::now(),
                refiles: 0,
                key,
                liveness,
                item,
            },
        );
        self.not_empty.notify_one();
    }

    /// Number of queued entries right now. Telemetry only.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().entries.len()
    }

    /// Block until a wanted entry is available, or until there are dead
    /// entries to cancel. Dead entries found along the way are returned as
    /// the second tuple element so the caller can run their cancellation
    /// paths outside the queue lock.
    ///
    /// The entry is `None` when only dead entries were found: they are
    /// handed back immediately instead of being held across the wait for
    /// new work, which would leave their chunks Pending with nothing in
    /// flight until the next submit.
    pub fn pop(&self) -> (Option<QueueEntry<T>>, Vec<QueueEntry<T>>) {
        let mut dead: Vec<QueueEntry<T>> = Vec::new();
        loop {
            self.refile_all_if_priorities_changed(&mut dead);
            let entry = {
                let mut q = self.inner.lock().unwrap();
                loop {
                    if let Some((_, entry)) = q.entries.pop_first() {
                        break entry;
                    }
                    if !dead.is_empty() {
                        return (None, dead);
                    }
                    q = self.not_empty.wait(q).unwrap();
                }
            };
            let Some(liveness) = &entry.liveness else {
                return (Some(entry), dead);
            };
            match liveness().priority() {
                None => dead.push(entry),
                Some(p) if p > entry.key.0 => {
                    let mut entry = entry;
                    entry.refiles += 1;
                    entry.key.0 = p;
                    let mut q = self.inner.lock().unwrap();
                    q.entries.insert(entry.key, entry);
                }
                Some(_) => return (Some(entry), dead),
            }
        }
    }

    /// If some lease's priority changed since the entries were filed,
    /// re-file every entry under its current priority (collecting the dead
    /// ones). Liveness is evaluated outside the lock; entries popped
    /// meanwhile are skipped. One caller per generation does the work.
    fn refile_all_if_priorities_changed(&self, dead: &mut Vec<QueueEntry<T>>) {
        let generation = lease::priority_generation();
        let snapshot: Vec<(QueueKey, LivenessFn)> = {
            let mut q = self.inner.lock().unwrap();
            if q.seen_generation == generation {
                return;
            }
            q.seen_generation = generation;
            q.entries
                .iter()
                .filter_map(|(k, e)| Some((*k, e.liveness.clone()?)))
                .collect()
        };
        let evaluated: Vec<(QueueKey, Option<Priority>)> =
            snapshot.into_iter().map(|(k, f)| (k, f().priority())).collect();
        let mut q = self.inner.lock().unwrap();
        for (key, priority) in evaluated {
            if priority == Some(key.0) {
                continue;
            }
            let Some(mut entry) = q.entries.remove(&key) else {
                continue;
            };
            match priority {
                None => dead.push(entry),
                Some(p) => {
                    entry.key.0 = p;
                    q.entries.insert(entry.key, entry);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::lease::Liveness;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;
    use std::sync::mpsc;
    use std::time::Duration;

    fn key(i: u32) -> ChunkKey {
        ChunkKey::new(0, i, 0, 0)
    }

    /// Liveness controlled by the test: `u32::MAX - 1` = dead.
    fn knob(p: u32) -> (Arc<AtomicU32>, LivenessFn) {
        let k = Arc::new(AtomicU32::new(p));
        let k2 = k.clone();
        let f: LivenessFn = Arc::new(move || match k2.load(Ordering::Relaxed) {
            u32::MAX => Liveness::Unleased,
            DEAD => Liveness::Dead,
            p => Liveness::Live(p),
        });
        (k, f)
    }
    const DEAD: u32 = u32::MAX - 1;

    fn items<T: Copy>(entries: &[QueueEntry<T>]) -> Vec<T> {
        entries.iter().map(|e| e.item).collect()
    }

    #[test]
    fn pops_by_priority_then_newest_first() {
        let q = WorkQueue::new();
        q.submit(key(1), knob(2).1, 1);
        q.submit(key(2), knob(1).1, 2);
        q.submit(key(3), knob(2).1, 3);
        q.submit(key(4), knob(u32::MAX).1, 4); // unleased: last
        assert_eq!(q.pop().0.map(|e| e.item), Some(2));
        assert_eq!(q.pop().0.map(|e| e.item), Some(3));
        assert_eq!(q.pop().0.map(|e| e.item), Some(1));
        assert_eq!(q.pop().0.map(|e| e.item), Some(4));
    }

    #[test]
    fn worse_head_is_refiled_and_dead_ones_are_returned() {
        let q = WorkQueue::new();
        let (a, fa) = knob(0);
        let (b, fb) = knob(1);
        q.submit(key(1), fa, 1);
        q.submit(key(2), fb, 2);
        q.submit_durable(key(3), 5, 3);
        a.store(3, Ordering::Relaxed); // got worse than 2 (1)
        b.store(DEAD, Ordering::Relaxed);
        let (entry, dead) = q.pop();
        assert_eq!(entry.map(|e| (e.item, e.refiles)), Some((1, 1)));
        assert_eq!(items(&dead), vec![2]);
        let (entry, dead) = q.pop();
        assert_eq!(entry.map(|e| e.item), Some(3), "durable runs regardless");
        assert!(dead.is_empty());
    }

    #[test]
    fn dead_entries_are_returned_when_queue_drains() {
        let q = Arc::new(WorkQueue::new());
        let (a, fa) = knob(0);
        q.submit(key(1), fa.clone(), 1);
        q.submit(key(2), fa, 2);
        a.store(DEAD, Ordering::Relaxed);

        // Must not block: nothing live is left, but the dead batch has to
        // reach the caller so it can cancel it.
        let (tx, rx) = mpsc::channel();
        let q2 = q.clone();
        std::thread::spawn(move || {
            let (entry, dead) = q2.pop();
            tx.send((entry.map(|e| e.item), dead.len())).unwrap();
        });
        let (entry, dead) = rx.recv_timeout(Duration::from_secs(5)).expect("pop blocked holding dead entries");
        assert_eq!(entry, None);
        assert_eq!(dead, 2);
        assert_eq!(q.len(), 0);
    }
}
