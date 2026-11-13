//! Persistent compressed chunk store: an append-only log of segment files.
//!
//! Every record is one 64³ cache chunk in compressed form, keyed by
//! `(volume hash, lod, x, y, z)`:
//!
//! - [`Format::Empty`]: no payload, the chunk is definitively absent.
//! - [`Format::Vc`]: a vc-delta3d payload as downloaded (ng).
//! - [`Format::Delta`]: a vc-delta3d payload `delta`-encoded by us (cheaper
//!   to decode).
//!
//! The payload describes itself, so readers decode both with
//! `vesuvius_delta3d::decode_into`; the format only says whether it is
//! worth re-encoding.
//!
//! ### Files
//!
//! `<dir>/seg-<created ms>-<pid>-<n>.open` is written by exactly one process,
//! which holds an exclusive `flock` on it. When it reaches `seg_bytes`, gets
//! older than `seg_max_age`, or the log shuts down, it is fsynced and renamed to
//! `.seg`. Sealed segments never change again. An `.open` file whose lock can be
//! taken belongs to a dead process: the next rescan checks it record by record,
//! truncates it after the last intact record, and seals it.
//!
//! Each process keeps its own in-memory index (key → newest copy), built from
//! the sealed segments at open and refreshed by periodic rescans. Other
//! processes' records become visible once their segment is sealed.
//!
//! ### Eviction: FIFO with lazy reinsertion
//!
//! Segments are ordered by age (sealed ones by mtime, open ones newest). Walking
//! from the newest, once `(1 − old_fraction) · cap` bytes have been passed, every
//! remaining segment is *old*. Reading a chunk whose copy lives in an old
//! segment (or touching it in the decoded tier, see [`SegLog::touch`]) copies
//! the record unchanged into the current segment. When the total exceeds `cap`,
//! whoever notices deletes the oldest sealed segments. A chunk is lost only if
//! nobody used it while its segment passed through the old zone.
//!
//! The only shared state is the directory itself. Staleness of a process's view
//! (old-zone flags, segment list) only costs a missed promotion. Deleting a
//! segment another process still reads from is fine: its open `File` keeps the
//! data alive until its next rescan drops it.

use super::state::ChunkKey;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock, Weak};
use std::time::{Duration, Instant, SystemTime};

/// Env var for the log's size cap, in GiB.
pub const LOG_CAP_ENV_VAR: &str = "VESUVIUS_LOG_CACHE_GB";
const DEFAULT_CAP_BYTES: u64 = 24 << 30;

const MAGIC: [u8; 4] = *b"VCL1";
const HEADER_LEN: usize = 40;
const ALIGN: u64 = 8;
const EVICT_LOCK: &str = "evict.lock";

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Format {
    Empty = 0,
    Vc = 1,
    Delta = 2,
}

impl Format {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Format::Empty),
            1 => Some(Format::Vc),
            2 => Some(Format::Delta),
            _ => None,
        }
    }
}

/// Identifies one chunk across volumes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct LogKey {
    pub volume: u64,
    pub chunk: ChunkKey,
}

/// Stable 64-bit id for a volume id string (first 8 bytes of its SHA-256).
pub fn volume_hash(volume_id: &str) -> u64 {
    let d = Sha256::digest(volume_id.as_bytes());
    u64::from_le_bytes(d[..8].try_into().unwrap())
}

/// A record as read back from the log.
#[derive(Debug, PartialEq, Eq)]
pub enum Record {
    Empty,
    Vc(Vec<u8>),
    Delta(Vec<u8>),
}

#[derive(Clone, Debug)]
pub struct Params {
    pub cap_bytes: u64,
    /// Seal the current segment once it reaches this size.
    pub seg_bytes: u64,
    /// Seal the current segment once it is this old (so a mostly idle
    /// process doesn't keep young-looking data in an old segment).
    pub seg_max_age: Duration,
    /// Fraction of `cap` (oldest bytes) whose records get promoted on use.
    pub old_fraction: f64,
    /// Background rescan interval. `None`: no background thread (tests call
    /// [`SegLog::rescan`] explicitly).
    pub rescan_interval: Option<Duration>,
}

impl Params {
    pub fn from_env() -> Self {
        let cap_bytes = match std::env::var(LOG_CAP_ENV_VAR) {
            Ok(v) => v
                .trim()
                .parse::<u64>()
                .unwrap_or_else(|e| panic!("{} must be an integer number of GiB: {}", LOG_CAP_ENV_VAR, e))
                << 30,
            Err(_) => DEFAULT_CAP_BYTES,
        };
        Self {
            cap_bytes,
            seg_bytes: 256 << 20,
            seg_max_age: Duration::from_secs(600),
            old_fraction: 0.25,
            rescan_interval: Some(Duration::from_secs(10)),
        }
    }
}

/// Where a record lives.
#[derive(Copy, Clone, Debug)]
struct Loc {
    seg: u32,
    off: u64,
    len: u32,
    format: Format,
}

struct Segment {
    path: Mutex<PathBuf>,
    file: File,
    /// Sort key for age: mtime (ms) once sealed, `u64::MAX` while it is our
    /// own open segment.
    order: AtomicU64,
    old: AtomicBool,
}

struct Writer {
    seg_id: u32,
    seg: Arc<Segment>,
    off: u64,
    created: Instant,
}

#[derive(Default)]
struct State {
    segs: HashMap<u32, Arc<Segment>>,
    /// File name (stem, without `.open`/`.seg`) → segment id.
    by_stem: HashMap<String, u32>,
    index: HashMap<LogKey, Loc>,
    next_id: u32,
}

/// Running statistics.
#[derive(Default, Debug, Clone, Copy)]
pub struct LogStats {
    pub records: usize,
    pub segments: usize,
    pub bytes_on_disk: u64,
    pub appended: u64,
    pub promoted: u64,
    pub evicted_segments: u64,
}

struct Inner {
    dir: PathBuf,
    params: Params,
    state: RwLock<State>,
    writer: Mutex<Option<Writer>>,
    appended: AtomicU64,
    promoted: AtomicU64,
    evicted_segments: AtomicU64,
    bytes_on_disk: AtomicU64,
    rescan_lock: Mutex<()>,
    shutdown: (Mutex<bool>, Condvar),
}

#[derive(Clone)]
pub struct SegLog {
    inner: Arc<Inner>,
}

static SEG_COUNTER: AtomicU32 = AtomicU32::new(0);

impl SegLog {
    /// Open (or create) the log in `dir`: recovers segments of dead
    /// processes, indexes every sealed segment and, with
    /// `params.rescan_interval`, starts the background rescan thread.
    pub fn open(dir: impl Into<PathBuf>, params: Params) -> std::io::Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        let interval = params.rescan_interval;
        let log = Self {
            inner: Arc::new(Inner {
                dir,
                params,
                state: RwLock::new(State::default()),
                writer: Mutex::new(None),
                appended: AtomicU64::new(0),
                promoted: AtomicU64::new(0),
                evicted_segments: AtomicU64::new(0),
                bytes_on_disk: AtomicU64::new(0),
                rescan_lock: Mutex::new(()),
                shutdown: (Mutex::new(false), Condvar::new()),
            }),
        };
        let t0 = Instant::now();
        log.rescan();
        let st = log.stats();
        log::info!(
            "[seglog] opened {}: {} records in {} segments, {} MiB, in {:?}",
            log.inner.dir.display(),
            st.records,
            st.segments,
            st.bytes_on_disk >> 20,
            t0.elapsed()
        );
        if let Some(interval) = interval {
            let weak = Arc::downgrade(&log.inner);
            std::thread::Builder::new()
                .name("vesuvius-seglog".into())
                .spawn(move || rescan_loop(weak, interval))
                .expect("spawn seglog thread");
        }
        Ok(log)
    }

    pub fn dir(&self) -> &Path {
        &self.inner.dir
    }

    /// Format of the newest copy of `key`, if any. Cheap: one index lookup.
    pub fn contains(&self, key: &LogKey) -> Option<Format> {
        self.inner.state.read().unwrap().index.get(key).map(|l| l.format)
    }

    /// Read the newest copy of `key`, promoting it when it sits in an old
    /// segment.
    pub fn read(&self, key: &LogKey) -> Option<Record> {
        let (loc, seg) = self.inner.locate(key)?;
        let payload = read_record(&seg, &loc, key);
        if seg.old.load(Ordering::Relaxed) {
            self.inner.promote(key, &loc, &payload);
        }
        Some(match loc.format {
            Format::Empty => Record::Empty,
            Format::Vc => Record::Vc(payload),
            Format::Delta => Record::Delta(payload),
        })
    }

    /// The chunk is in use (e.g. painted from the decoded tier): if its log
    /// copy is in an old segment, copy it forward.
    pub fn touch(&self, key: &LogKey) {
        let Some((loc, seg)) = self.inner.locate(key) else { return };
        if seg.old.load(Ordering::Relaxed) {
            let payload = read_record(&seg, &loc, key);
            self.inner.promote(key, &loc, &payload);
        }
    }

    /// Append a record. `payload` must be empty for [`Format::Empty`].
    pub fn append(&self, key: &LogKey, format: Format, payload: &[u8]) -> std::io::Result<()> {
        assert!(format != Format::Empty || payload.is_empty(), "Empty record with payload");
        self.inner.append(key, format, payload)?;
        self.inner.appended.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Pick up other processes' sealed segments, drop deleted ones, recover
    /// dead processes' open segments, evict over cap, and recompute the old
    /// zone. Runs periodically in the background; public for tests.
    pub fn rescan(&self) {
        self.inner.rescan();
    }

    /// Seal the current segment (if any) so other processes can see it.
    pub fn seal(&self) {
        let mut w = self.inner.writer.lock().unwrap();
        if let Some(w) = w.take() {
            self.inner.seal(w);
        }
    }

    /// Seal and stop the background thread. The log stays usable.
    pub fn shutdown(&self) {
        self.seal();
        let (m, cv) = &self.inner.shutdown;
        *m.lock().unwrap() = true;
        cv.notify_all();
    }

    pub fn stats(&self) -> LogStats {
        let st = self.inner.state.read().unwrap();
        LogStats {
            records: st.index.len(),
            segments: st.segs.len(),
            bytes_on_disk: self.inner.bytes_on_disk.load(Ordering::Relaxed),
            appended: self.inner.appended.load(Ordering::Relaxed),
            promoted: self.inner.promoted.load(Ordering::Relaxed),
            evicted_segments: self.inner.evicted_segments.load(Ordering::Relaxed),
        }
    }
}

fn rescan_loop(weak: Weak<Inner>, interval: Duration) {
    loop {
        let Some(inner) = weak.upgrade() else { return };
        {
            let (m, cv) = &inner.shutdown;
            let guard = m.lock().unwrap();
            let (guard, _) = cv.wait_timeout(guard, interval).unwrap();
            if *guard {
                return;
            }
        }
        inner.maybe_seal_by_age();
        inner.rescan();
    }
}

impl Inner {
    fn locate(&self, key: &LogKey) -> Option<(Loc, Arc<Segment>)> {
        let st = self.state.read().unwrap();
        let loc = *st.index.get(key)?;
        let seg = st.segs.get(&loc.seg).expect("index points at a known segment").clone();
        Some((loc, seg))
    }

    fn promote(&self, key: &LogKey, from: &Loc, payload: &[u8]) {
        // Another reader may have promoted it already.
        if self.state.read().unwrap().index.get(key).is_some_and(|l| l.seg != from.seg) {
            return;
        }
        match self.append(key, from.format, payload) {
            Ok(()) => {
                self.promoted.fetch_add(1, Ordering::Relaxed);
            }
            Err(e) => log::warn!("[seglog] promoting {:?} failed: {}", key, e),
        }
    }

    fn append(&self, key: &LogKey, format: Format, payload: &[u8]) -> std::io::Result<()> {
        let rec = encode_record(key, format, payload);
        let mut wg = self.writer.lock().unwrap();
        let roll = match wg.as_ref() {
            None => false,
            Some(w) => w.off > 0 && w.off + rec.len() as u64 > self.params.seg_bytes,
        };
        if roll {
            self.seal(wg.take().unwrap());
        }
        if wg.is_none() {
            *wg = Some(self.new_segment()?);
        }
        let w = wg.as_mut().unwrap();
        if let Err(e) = w.seg.file.write_all_at(&rec, w.off) {
            // Cut the torn tail so the segment stays sealable.
            w.seg.file.set_len(w.off)?;
            return Err(e);
        }
        let loc = Loc {
            seg: w.seg_id,
            off: w.off,
            len: payload.len() as u32,
            format,
        };
        w.off += rec.len() as u64;
        self.bytes_on_disk.fetch_add(rec.len() as u64, Ordering::Relaxed);
        self.state.write().unwrap().index.insert(*key, loc);
        Ok(())
    }

    fn new_segment(&self) -> std::io::Result<Writer> {
        let ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis();
        let stem = format!(
            "seg-{:013}-{}-{:05}",
            ms,
            std::process::id(),
            SEG_COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        let path = self.dir.join(format!("{}.open", stem));
        let file = OpenOptions::new().read(true).write(true).create_new(true).open(&path)?;
        assert!(try_flock(&file), "fresh segment {} is locked by someone else", path.display());
        let seg = Arc::new(Segment {
            path: Mutex::new(path),
            file,
            order: AtomicU64::new(u64::MAX),
            old: AtomicBool::new(false),
        });
        let mut st = self.state.write().unwrap();
        let id = st.next_id;
        st.next_id += 1;
        st.segs.insert(id, seg.clone());
        st.by_stem.insert(stem, id);
        Ok(Writer {
            seg_id: id,
            seg,
            off: 0,
            created: Instant::now(),
        })
    }

    /// fsync and rename `.open` → `.seg`. The segment stays in the table with
    /// its (still locked) descriptor; the lock no longer matters, recovery
    /// only looks at `.open` files.
    fn seal(&self, w: Writer) {
        if w.off == 0 {
            let path = w.seg.path.lock().unwrap().clone();
            let mut st = self.state.write().unwrap();
            st.segs.remove(&w.seg_id);
            st.by_stem.retain(|_, id| *id != w.seg_id);
            drop(st);
            std::fs::remove_file(&path).unwrap_or_else(|e| panic!("removing empty {}: {}", path.display(), e));
            return;
        }
        w.seg.file.sync_data().expect("fsync segment");
        let open_path = w.seg.path.lock().unwrap().clone();
        let sealed = open_path.with_extension("seg");
        std::fs::rename(&open_path, &sealed).unwrap_or_else(|e| panic!("sealing {}: {}", open_path.display(), e));
        *w.seg.path.lock().unwrap() = sealed.clone();
        let mtime = file_mtime_ms(&w.seg.file);
        w.seg.order.store(mtime, Ordering::Relaxed);
        log::debug!("[seglog] sealed {} ({} MiB)", sealed.display(), w.off >> 20);
    }

    fn maybe_seal_by_age(&self) {
        let mut wg = self.writer.lock().unwrap();
        if wg.as_ref().is_some_and(|w| w.created.elapsed() >= self.params.seg_max_age) {
            self.seal(wg.take().unwrap());
        }
    }

    fn rescan(&self) {
        let _serial = self.rescan_lock.lock().unwrap();
        let own_open: Option<u32> = self.writer.lock().unwrap().as_ref().map(|w| w.seg_id);

        // 1. Recover dead processes' open segments.
        for (stem, ext, _) in list_dir(&self.dir) {
            if ext == "open" && !self.state.read().unwrap().by_stem.contains_key(&stem) {
                self.recover(&stem);
            }
        }

        // 2. Index new sealed segments, forget vanished ones.
        let listing = list_dir(&self.dir);
        let present: std::collections::HashSet<&str> = listing.iter().map(|(s, _, _)| s.as_str()).collect();
        let vanished: Vec<u32> = {
            let st = self.state.read().unwrap();
            st.by_stem
                .iter()
                .filter(|(stem, id)| !present.contains(stem.as_str()) && Some(**id) != own_open)
                .map(|(_, id)| *id)
                .collect()
        };
        if !vanished.is_empty() {
            let mut st = self.state.write().unwrap();
            for id in &vanished {
                st.segs.remove(id);
            }
            st.by_stem.retain(|_, id| !vanished.contains(id));
            st.index.retain(|_, l| !vanished.contains(&l.seg));
        }
        for (stem, ext, _) in &listing {
            if ext == "seg" && !self.state.read().unwrap().by_stem.contains_key(stem) {
                self.index_segment(stem);
            }
        }

        // 3. Evict over cap, oldest sealed first.
        let listing = list_dir(&self.dir);
        let total: u64 = listing.iter().map(|(_, _, len)| len).sum();
        if total > self.params.cap_bytes {
            self.evict(&listing, total);
        }
        let total: u64 = list_dir(&self.dir).iter().map(|(_, _, len)| len).sum();
        self.bytes_on_disk.store(total, Ordering::Relaxed);

        // 4. Old zone: newest first, everything past (1 − f)·cap is old.
        // Segments we don't hold (other processes' open ones) count as
        // newest and take up young space.
        let st = self.state.read().unwrap();
        let mut known_bytes = 0u64;
        let mut segs: Vec<(u64, u64, &Arc<Segment>)> = st
            .segs
            .values()
            .map(|s| {
                let len = s.file.metadata().map(|m| m.len()).unwrap_or(0);
                known_bytes += len;
                (s.order.load(Ordering::Relaxed), len, s)
            })
            .collect();
        segs.sort_by(|a, b| b.0.cmp(&a.0));
        let young_cap = ((1.0 - self.params.old_fraction) * self.params.cap_bytes as f64) as u64;
        let mut acc = total.saturating_sub(known_bytes);
        for (_, len, seg) in segs {
            seg.old.store(acc >= young_cap, Ordering::Relaxed);
            acc += len;
        }
    }

    fn evict(&self, listing: &[(String, String, u64)], mut total: u64) {
        let lock_path = self.dir.join(EVICT_LOCK);
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .unwrap_or_else(|e| panic!("opening {}: {}", lock_path.display(), e));
        if !try_flock(&lock) {
            // Someone else is evicting right now.
            return;
        }
        let mut sealed: Vec<(u64, String, u64)> = listing
            .iter()
            .filter(|(_, ext, _)| ext == "seg")
            .filter_map(|(stem, _, len)| {
                let path = self.dir.join(format!("{}.seg", stem));
                let mtime = std::fs::metadata(&path).ok()?.modified().ok()?;
                Some((system_time_ms(mtime), stem.clone(), *len))
            })
            .collect();
        sealed.sort();
        for (_, stem, len) in sealed {
            if total <= self.params.cap_bytes {
                break;
            }
            let path = self.dir.join(format!("{}.seg", stem));
            match std::fs::remove_file(&path) {
                Ok(()) => {
                    total -= len;
                    self.evicted_segments.fetch_add(1, Ordering::Relaxed);
                    log::debug!("[seglog] evicted {} ({} MiB)", path.display(), len >> 20);
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => panic!("evicting {}: {}", path.display(), e),
            }
            let mut st = self.state.write().unwrap();
            if let Some(id) = st.by_stem.remove(&stem) {
                st.segs.remove(&id);
                st.index.retain(|_, l| l.seg != id);
            }
        }
    }

    /// An `.open` segment we don't own: if its writer is gone (we can take
    /// its lock), cut it after the last intact record and seal it.
    fn recover(&self, stem: &str) {
        let path = self.dir.join(format!("{}.open", stem));
        let file = match OpenOptions::new().read(true).write(true).open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return,
            Err(e) => panic!("opening {}: {}", path.display(), e),
        };
        if !try_flock(&file) {
            return; // Its writer is alive.
        }
        // Between our open and lock, a concurrent recoverer may have
        // finished and renamed it; then it is already sealed.
        if !path.exists() {
            return;
        }
        let len = file.metadata().expect("stat open segment").len();
        let mut off = 0u64;
        let mut n = 0usize;
        while let Some((_, _, _, rec_len)) = read_header(&file, off, len) {
            let mut buf = vec![0u8; rec_len as usize];
            file.read_exact_at(&mut buf, off).expect("read record");
            if !record_crc_ok(&buf) {
                break;
            }
            off += rec_len;
            n += 1;
        }
        if off < len {
            log::warn!(
                "[seglog] recovering {}: keeping {} records, cutting {} torn bytes",
                path.display(),
                n,
                len - off
            );
            file.set_len(off).expect("truncate recovered segment");
        }
        file.sync_data().expect("fsync recovered segment");
        if off == 0 {
            let _ = std::fs::remove_file(&path);
            return;
        }
        match std::fs::rename(&path, path.with_extension("seg")) {
            Ok(()) => log::info!("[seglog] recovered {} ({} records)", path.display(), n),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => panic!("sealing recovered {}: {}", path.display(), e),
        }
    }

    fn index_segment(&self, stem: &str) {
        let path = self.dir.join(format!("{}.seg", stem));
        let file = match File::open(&path) {
            Ok(f) => f,
            // Evicted by someone else since the listing.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return,
            Err(e) => panic!("opening {}: {}", path.display(), e),
        };
        let len = file.metadata().expect("stat segment").len();
        let order = file_mtime_ms(&file);
        let mut entries = Vec::new();
        let mut off = 0u64;
        while off < len {
            let (key, format, payload_len, rec_len) = read_header(&file, off, len)
                .unwrap_or_else(|| panic!("corrupt record header in sealed segment {} at {}", path.display(), off));
            entries.push((key, off, payload_len, format));
            off += rec_len;
        }
        let seg = Arc::new(Segment {
            path: Mutex::new(path),
            file,
            order: AtomicU64::new(order),
            old: AtomicBool::new(false),
        });
        let mut st = self.state.write().unwrap();
        let id = st.next_id;
        st.next_id += 1;
        st.segs.insert(id, seg);
        st.by_stem.insert(stem.to_string(), id);
        for (key, off, len, format) in entries {
            let newer = match st.index.get(&key) {
                Some(l) => st.segs[&l.seg].order.load(Ordering::Relaxed) <= order,
                None => true,
            };
            if newer {
                st.index.insert(key, Loc { seg: id, off, len, format });
            }
        }
    }
}

fn encode_record(key: &LogKey, format: Format, payload: &[u8]) -> Vec<u8> {
    let total = (HEADER_LEN + payload.len()) as u64;
    let padded = total.div_ceil(ALIGN) * ALIGN;
    let mut rec = Vec::with_capacity(padded as usize);
    rec.extend_from_slice(&MAGIC);
    rec.push(format as u8);
    rec.push(key.chunk.lod);
    rec.extend_from_slice(&[0, 0]);
    rec.extend_from_slice(&key.volume.to_le_bytes());
    rec.extend_from_slice(&key.chunk.x.to_le_bytes());
    rec.extend_from_slice(&key.chunk.y.to_le_bytes());
    rec.extend_from_slice(&key.chunk.z.to_le_bytes());
    rec.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    rec.extend_from_slice(&crc32fast::hash(payload).to_le_bytes());
    let hcrc = crc32fast::hash(&rec[..36]);
    rec.extend_from_slice(&hcrc.to_le_bytes());
    debug_assert_eq!(rec.len(), HEADER_LEN);
    rec.extend_from_slice(payload);
    rec.resize(padded as usize, 0);
    rec
}

/// Parse and check the header at `off`. Returns the key, format, payload
/// length and padded record length, or `None` if there is no intact header
/// (end of data or a torn tail).
fn read_header(file: &File, off: u64, file_len: u64) -> Option<(LogKey, Format, u32, u64)> {
    if off + HEADER_LEN as u64 > file_len {
        return None;
    }
    let mut h = [0u8; HEADER_LEN];
    file.read_exact_at(&mut h, off).ok()?;
    if h[..4] != MAGIC || crc32fast::hash(&h[..36]) != u32::from_le_bytes(h[36..40].try_into().unwrap()) {
        return None;
    }
    let format = Format::from_u8(h[4])?;
    let key = LogKey {
        volume: u64::from_le_bytes(h[8..16].try_into().unwrap()),
        chunk: ChunkKey {
            lod: h[5],
            x: u32::from_le_bytes(h[16..20].try_into().unwrap()),
            y: u32::from_le_bytes(h[20..24].try_into().unwrap()),
            z: u32::from_le_bytes(h[24..28].try_into().unwrap()),
        },
    };
    let payload_len = u32::from_le_bytes(h[28..32].try_into().unwrap()) as u64;
    let rec_len = (HEADER_LEN as u64 + payload_len).div_ceil(ALIGN) * ALIGN;
    if off + rec_len > file_len {
        return None;
    }
    Some((key, format, payload_len as u32, rec_len))
}

fn record_crc_ok(rec: &[u8]) -> bool {
    let payload_len = u32::from_le_bytes(rec[28..32].try_into().unwrap()) as usize;
    let crc = u32::from_le_bytes(rec[32..36].try_into().unwrap());
    crc32fast::hash(&rec[HEADER_LEN..HEADER_LEN + payload_len]) == crc
}

/// Read and verify a record's payload. A mismatch means the segment was
/// corrupted after it was written: fail loudly.
fn read_record(seg: &Segment, loc: &Loc, key: &LogKey) -> Vec<u8> {
    let mut buf = vec![0u8; HEADER_LEN + loc.len as usize];
    seg.file
        .read_exact_at(&mut buf, loc.off)
        .unwrap_or_else(|e| panic!("reading {:?} from {}: {}", key, seg.path.lock().unwrap().display(), e));
    let hdr_key = LogKey {
        volume: u64::from_le_bytes(buf[8..16].try_into().unwrap()),
        chunk: ChunkKey {
            lod: buf[5],
            x: u32::from_le_bytes(buf[16..20].try_into().unwrap()),
            y: u32::from_le_bytes(buf[20..24].try_into().unwrap()),
            z: u32::from_le_bytes(buf[24..28].try_into().unwrap()),
        },
    };
    assert!(
        buf[..4] == MAGIC && hdr_key == *key && record_crc_ok(&buf),
        "corrupt record for {:?} at {} in {}",
        key,
        loc.off,
        seg.path.lock().unwrap().display()
    );
    buf.drain(..HEADER_LEN);
    buf
}

/// `(stem, extension, length)` of every segment file in `dir`.
fn list_dir(dir: &Path) -> Vec<(String, String, u64)> {
    let rd = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("listing {}: {}", dir.display(), e));
    let mut out = Vec::new();
    for e in rd {
        let e = e.unwrap_or_else(|e| panic!("listing {}: {}", dir.display(), e));
        let name = e.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.starts_with("seg-") {
            continue;
        }
        let Some((stem, ext)) = name.rsplit_once('.') else { continue };
        if ext != "seg" && ext != "open" {
            continue;
        }
        // Gone between readdir and stat: evicted or renamed by a peer.
        let Ok(md) = e.metadata() else { continue };
        out.push((stem.to_string(), ext.to_string(), md.len()));
    }
    out
}

fn try_flock(file: &File) -> bool {
    // SAFETY: plain syscall on a valid descriptor.
    let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if rc == 0 {
        return true;
    }
    let err = std::io::Error::last_os_error();
    assert_eq!(err.raw_os_error(), Some(libc::EWOULDBLOCK), "flock failed: {}", err);
    false
}

fn system_time_ms(t: SystemTime) -> u64 {
    t.duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis() as u64
}

fn file_mtime_ms(file: &File) -> u64 {
    system_time_ms(file.metadata().expect("stat segment").modified().expect("segment mtime"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(cap: u64, seg: u64) -> Params {
        Params {
            cap_bytes: cap,
            seg_bytes: seg,
            seg_max_age: Duration::from_secs(3600),
            old_fraction: 0.25,
            rescan_interval: None,
        }
    }

    fn key(i: u32) -> LogKey {
        LogKey {
            volume: 7,
            chunk: ChunkKey::new(0, i, 0, 0),
        }
    }

    fn payload(i: u32, len: usize) -> Vec<u8> {
        (0..len).map(|j| (i as usize * 31 + j) as u8).collect()
    }

    #[test]
    fn append_read_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let log = SegLog::open(dir.path(), params(1 << 30, 1 << 20)).unwrap();
        log.append(&key(1), Format::Vc, &payload(1, 1000)).unwrap();
        log.append(&key(2), Format::Empty, &[]).unwrap();
        assert_eq!(log.read(&key(1)), Some(Record::Vc(payload(1, 1000))));
        assert_eq!(log.read(&key(2)), Some(Record::Empty));
        assert_eq!(log.read(&key(3)), None);
        // Newest copy wins.
        log.append(&key(1), Format::Vc, &payload(9, 13)).unwrap();
        assert_eq!(log.read(&key(1)), Some(Record::Vc(payload(9, 13))));
    }

    #[test]
    fn sealed_segments_are_visible_to_other_instances_and_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let a = SegLog::open(dir.path(), params(1 << 30, 1 << 20)).unwrap();
        let b = SegLog::open(dir.path(), params(1 << 30, 1 << 20)).unwrap();
        a.append(&key(1), Format::Vc, &payload(1, 500)).unwrap();
        b.rescan();
        // Open segment of a live writer: neither visible nor recovered.
        assert_eq!(b.read(&key(1)), None);
        assert!(list_dir(dir.path()).iter().any(|(_, ext, _)| ext == "open"));
        a.seal();
        b.rescan();
        assert_eq!(b.read(&key(1)), Some(Record::Vc(payload(1, 500))));
        drop((a, b));
        let c = SegLog::open(dir.path(), params(1 << 30, 1 << 20)).unwrap();
        assert_eq!(c.read(&key(1)), Some(Record::Vc(payload(1, 500))));
    }

    #[test]
    fn dead_writers_open_segment_is_recovered_and_cut_at_torn_tail() {
        let dir = tempfile::tempdir().unwrap();
        let path = {
            let a = SegLog::open(dir.path(), params(1 << 30, 1 << 20)).unwrap();
            a.append(&key(1), Format::Vc, &payload(1, 300)).unwrap();
            a.append(&key(2), Format::Vc, &payload(2, 300)).unwrap();
            let w = a.inner.writer.lock().unwrap().take().unwrap();
            // "Crash": drop the writer (and its lock) without sealing, after
            // tearing the second record.
            let p = w.seg.path.lock().unwrap().clone();
            let len = w.seg.file.metadata().unwrap().len();
            w.seg.file.set_len(len - 100).unwrap();
            a.inner.state.write().unwrap().segs.clear();
            drop(w);
            p
        };
        assert!(path.exists());
        let b = SegLog::open(dir.path(), params(1 << 30, 1 << 20)).unwrap();
        assert!(!path.exists());
        assert_eq!(b.read(&key(1)), Some(Record::Vc(payload(1, 300))));
        assert_eq!(b.read(&key(2)), None);
    }

    #[test]
    fn eviction_deletes_oldest_and_promotion_saves_used_chunks() {
        let dir = tempfile::tempdir().unwrap();
        // ~10 KiB records, segments of 4 records, cap ~40 records: the old zone
        // (25%) spans more than two segments.
        let rec = 10_000;
        let log = SegLog::open(dir.path(), params(400_000, 41_000)).unwrap();
        for i in 0..8 {
            log.append(&key(i), Format::Vc, &payload(i, rec)).unwrap();
            std::thread::sleep(Duration::from_millis(5));
        }
        log.seal();
        std::thread::sleep(Duration::from_millis(20));
        let mut hot_promotions = 0;
        for i in 8..80 {
            log.append(&key(i), Format::Vc, &payload(i, rec)).unwrap();
            if i % 4 == 3 {
                log.seal();
                std::thread::sleep(Duration::from_millis(20));
                log.rescan();
                // Keep chunk 0 in use.
                let before = log.stats().promoted;
                log.touch(&key(0));
                hot_promotions += log.stats().promoted - before;
            }
        }
        log.rescan();
        let st = log.stats();
        assert!(st.bytes_on_disk <= 400_000, "{:?}", st);
        assert!(st.evicted_segments > 0, "{:?}", st);
        assert_eq!(log.read(&key(0)), Some(Record::Vc(payload(0, rec))), "used chunk survives");
        assert_eq!(log.read(&key(1)), None, "unused old chunk evicted");
        assert_eq!(log.read(&key(79)), Some(Record::Vc(payload(79, rec))));
        // Chunk 0 was copied forward only when its copy aged into the old
        // zone, not on every touch.
        assert!((1..=4).contains(&hot_promotions), "{} promotions", hot_promotions);
    }
}
