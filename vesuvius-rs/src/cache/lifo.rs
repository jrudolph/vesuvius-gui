//! Shared pure-LIFO work queue, used by both the cache's task queue and
//! the downloader's job queue.
//!
//! Entries live in a `BTreeMap` keyed by `!seq`, so the most recently
//! submitted (or re-touched) entry pops first. Earlier iterations split
//! work across LOD-rank and viewport-distance tiers, but in practice that
//! let coarse-LOD work outrank current-viewport work and stall painting.
//! The simpler model: paint always asks for what it wants right now, and
//! "right now" wins. Older requests slide toward the tail and either get
//! processed in LIFO order when workers catch up, or culled by `max_age`
//! at pop.
//!
//! `chunk_index` mirrors `entries` keyed by `ChunkKey` for the O(1)
//! lookup `touch` needs to refresh seq + `added_at` on in-flight entries
//! — the paint loop re-touches what it wants every frame so the queue
//! head tracks the current viewport.
//!
//! Unbounded; dedup happens at the cache layer (source-key + chunk-key
//! uniqueness), so submission always succeeds. The only staleness check
//! is age, and `submit_durable` entries are exempt from it — used for
//! Extract tasks whose downloads already completed, where discarding
//! would waste the paid-for bytes.

use super::state::ChunkKey;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

pub(super) struct LifoQueue<T> {
    inner: Mutex<Inner<T>>,
    not_empty: Condvar,
    max_age: Duration,
    /// When false, the `max_age` cull at pop is disabled — every entry runs
    /// when LIFO order reaches it, no matter how stale. Shared (same `Arc`)
    /// with the cache's other queue and toggled via `ChunkCache::set_culling`.
    /// The interactive GUI leaves culling on so work for a viewport the user
    /// has scrolled past dies at the tail; the offline renderer turns it off,
    /// because it dispatches exactly the chunks it wants and a slow link can
    /// easily leave a wanted fetch queued past `max_age` — culling it there
    /// strands the chunk in a cooldown and the ensure stage never sees it land.
    cull_enabled: Arc<AtomicBool>,
}

struct Inner<T> {
    entries: BTreeMap<u64, QueueEntry<T>>,
    /// Reverse lookup: every queue key currently registered for a given
    /// chunk. A chunk can have several entries (e.g. its sources'
    /// FetchSource tasks plus an Extract task once sources resolve).
    /// Maintained in lockstep with `entries`.
    chunk_index: HashMap<ChunkKey, Vec<u64>>,
    next_seq: u64,
}

pub(super) struct QueueEntry<T> {
    /// The cache chunk this work is on behalf of — the `touch` handle.
    pub chunk: ChunkKey,
    pub added_at: Instant,
    /// First submission time — unlike `added_at`, never reset by `touch`.
    /// Telemetry only: total time from "first wanted" to pop.
    pub submitted_at: Instant,
    /// How many times `touch` refreshed this entry while it waited. A high
    /// count marks a chunk the paint loop kept asking for — telemetry for
    /// spotting priority inversion / head-of-line blocking.
    pub touch_count: u32,
    /// Exempt from the `max_age` cull at pop. For work whose inputs are
    /// already paid for (an Extract whose sources finished downloading),
    /// running late is strictly cheaper than discarding: the alternative
    /// is re-downloading the same bytes on the next visit. Durable entries
    /// keep normal LIFO order — they just wait at the tail instead of
    /// dying there.
    pub durable: bool,
    pub item: T,
}

/// Encode `seq` so that **larger** seq sorts BEFORE smaller (BTreeMap pops
/// smallest key first → LIFO). `seq` is monotonically increasing so `!seq`
/// is monotonically decreasing.
fn rev_seq(seq: u64) -> u64 {
    !seq
}

impl<T> LifoQueue<T> {
    pub fn new(max_age: Duration, cull_enabled: Arc<AtomicBool>) -> Self {
        Self {
            inner: Mutex::new(Inner {
                entries: BTreeMap::new(),
                chunk_index: HashMap::new(),
                next_seq: 0,
            }),
            not_empty: Condvar::new(),
            max_age,
            cull_enabled,
        }
    }

    /// Submit one work item on behalf of `chunk`. Always succeeds (the
    /// queue is unbounded); the only way queued work dies unprocessed is
    /// the `max_age` cull at pop.
    pub fn submit(&self, chunk: ChunkKey, item: T) {
        self.submit_inner(chunk, item, false);
    }

    /// Like `submit`, but the entry is exempt from the `max_age` cull —
    /// it runs whenever the LIFO order reaches it, no matter how stale.
    /// For work whose inputs are already paid for (see `QueueEntry::durable`).
    pub fn submit_durable(&self, chunk: ChunkKey, item: T) {
        self.submit_inner(chunk, item, true);
    }

    fn submit_inner(&self, chunk: ChunkKey, item: T, durable: bool) {
        let mut q = self.inner.lock().unwrap();
        q.next_seq += 1;
        let key = rev_seq(q.next_seq);
        let now = Instant::now();
        q.entries.insert(
            key,
            QueueEntry {
                chunk,
                added_at: now,
                submitted_at: now,
                touch_count: 0,
                durable,
                item,
            },
        );
        q.chunk_index.entry(chunk).or_default().push(key);
        self.not_empty.notify_one();
    }

    /// Refresh every queued entry for `chunk`: bump its seq (moving it
    /// to the head of the LIFO order) and reset `added_at` so `max_age`
    /// re-counts from now. No-op when the chunk has no queued entries.
    pub fn touch(&self, chunk: ChunkKey) {
        let mut q = self.inner.lock().unwrap();
        let old_keys = match q.chunk_index.remove(&chunk) {
            Some(v) if !v.is_empty() => v,
            _ => return,
        };
        let mut new_keys = Vec::with_capacity(old_keys.len());
        let now = Instant::now();
        for old_key in old_keys {
            let Some(mut entry) = q.entries.remove(&old_key) else {
                continue;
            };
            entry.added_at = now;
            entry.touch_count += 1;
            q.next_seq += 1;
            let new_key = rev_seq(q.next_seq);
            q.entries.insert(new_key, entry);
            new_keys.push(new_key);
        }
        if !new_keys.is_empty() {
            q.chunk_index.insert(chunk, new_keys);
            self.not_empty.notify_one();
        }
    }

    /// Number of queued entries right now. Telemetry only.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().entries.len()
    }

    /// Block until a non-stale entry is available, or until culling has
    /// produced entries to cancel. Entries culled along the way (older than
    /// `max_age`) are returned as the second tuple element so the caller can
    /// run their cancellation paths outside the queue lock.
    ///
    /// The entry is `None` when culling drained the queue: the dropped batch
    /// is handed back immediately instead of being held across the wait for
    /// new work, which would leave their chunks Pending with nothing in
    /// flight until the next submit/touch.
    pub fn pop(&self) -> (Option<QueueEntry<T>>, Vec<QueueEntry<T>>) {
        let mut q = self.inner.lock().unwrap();
        let mut dropped: Vec<QueueEntry<T>> = Vec::new();
        loop {
            let Some((key, entry)) = q.entries.pop_first() else {
                if !dropped.is_empty() {
                    return (None, dropped);
                }
                q = self.not_empty.wait(q).unwrap();
                continue;
            };
            if let Some(keys) = q.chunk_index.get_mut(&entry.chunk) {
                keys.retain(|k| *k != key);
                if keys.is_empty() {
                    q.chunk_index.remove(&entry.chunk);
                }
            }
            if !entry.durable
                && self.cull_enabled.load(Ordering::Relaxed)
                && entry.added_at.elapsed() > self.max_age
            {
                dropped.push(entry);
                continue;
            }
            return (Some(entry), dropped);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    fn queue(max_age: Duration) -> Arc<LifoQueue<u32>> {
        Arc::new(LifoQueue::new(max_age, Arc::new(AtomicBool::new(true))))
    }

    fn key(i: u32) -> ChunkKey {
        ChunkKey::new(0, i, 0, 0)
    }

    #[test]
    fn pops_newest_first() {
        let q = queue(Duration::from_secs(60));
        q.submit(key(1), 1);
        q.submit(key(2), 2);
        q.touch(key(1));
        assert_eq!(q.pop().0.map(|e| e.item), Some(1));
        assert_eq!(q.pop().0.map(|e| e.item), Some(2));
    }

    #[test]
    fn culled_entries_are_returned_when_queue_drains() {
        let q = queue(Duration::from_millis(1));
        q.submit(key(1), 1);
        q.submit(key(2), 2);
        std::thread::sleep(Duration::from_millis(10));

        // Must not block: nothing fresh is left, but the stale batch has to
        // reach the caller so it can cancel it.
        let (tx, rx) = mpsc::channel();
        let q2 = q.clone();
        std::thread::spawn(move || {
            let (entry, dropped) = q2.pop();
            tx.send((entry.map(|e| e.item), dropped.len())).unwrap();
        });
        let (entry, dropped) = rx.recv_timeout(Duration::from_secs(5)).expect("pop blocked holding dropped entries");
        assert_eq!(entry, None);
        assert_eq!(dropped, 2);
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn culled_entries_ride_along_with_a_fresh_one() {
        let q = queue(Duration::from_millis(1));
        q.submit(key(1), 1);
        std::thread::sleep(Duration::from_millis(10));
        q.submit_durable(key(2), 2);
        q.submit(key(3), 3);
        std::thread::sleep(Duration::from_millis(10));
        // 3 is stale, 2 is durable (exempt), 1 is stale and never reached.
        let (entry, dropped) = q.pop();
        assert_eq!(entry.map(|e| e.item), Some(2));
        assert_eq!(dropped.iter().map(|e| e.item).collect::<Vec<_>>(), vec![3]);
        let (entry, dropped) = q.pop();
        assert_eq!(entry.map(|e| e.item), None);
        assert_eq!(dropped.iter().map(|e| e.item).collect::<Vec<_>>(), vec![1]);
    }
}
