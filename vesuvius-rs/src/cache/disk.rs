//! On-disk persistence for cache chunks.
//!
//! Layout: one sparse data file per LOD,
//! `{root}/chunks-L{lod:02}.dat`, dense per-LOD index:
//! `offset = ((z * nyL + y) * nxL + x) * CHUNK_VOXELS`. Each file is
//! `set_len(nx*ny*nz*CHUNK_VOXELS)` at first open so the kernel reports a
//! sparse file (ext4/xfs/btrfs/APFS only); only chunks actually written
//! occupy physical blocks.
//!
//! Chunk state (Missing / Resident / Empty) lives in a `chunks.idx` sidecar
//! (`super::sidecar::Sidecar`) — one byte per slot per LOD. Writers update
//! the bitmap with `Release` after pwriting bytes; readers observing
//! `Resident` with `Acquire` are guaranteed to see the full 256 KiB through
//! the shared mmap (same inode → same page cache).
//!
//! A background sync thread snapshots the bitmap, fsyncs LOD data files
//! that had transitions, and atomically renames the sidecar into place
//! every `SYNC_INTERVAL` or after `SYNC_COUNT_THRESHOLD` transitions —
//! whichever comes first. Sidecar is always a strict subset of durable
//! bytes, so a crash loses at most the last sync interval of work
//! (chunks are re-downloaded).

use super::sidecar::{self, LodDims, Sidecar, STATE_EMPTY, STATE_MISSING, STATE_RESIDENT};
use super::state::ChunkKey;
use super::CHUNK_VOXELS;
use memmap::{Mmap, MmapOptions};
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

pub enum LoadOutcome {
    Resident { mmap: Arc<Mmap>, offset: usize },
    Empty,
    Missing,
}

#[derive(Clone)]
struct LodFile {
    file: Arc<File>,
    mmap: Arc<Mmap>,
}

struct LodSlot {
    dims: LodDims,
    /// Lazily opened on first touch.
    opened: Mutex<Option<LodFile>>,
}

struct SyncInner {
    shutdown: bool,
    wake_now: bool,
    last_sync: Instant,
}

struct SyncState {
    inner: Mutex<SyncInner>,
    cv: Condvar,
}

struct DiskStoreInner {
    root: PathBuf,
    sidecar: Arc<Sidecar>,
    lods: Vec<LodSlot>,
    sync_state: SyncState,
}

pub struct DiskStore {
    inner: Arc<DiskStoreInner>,
    sync_thread: Mutex<Option<JoinHandle<()>>>,
}

const SYNC_INTERVAL: Duration = Duration::from_secs(10);
const SYNC_COUNT_THRESHOLD: u64 = 256;

impl DiskStore {
    /// Open (or create) the cache directory for a volume.
    ///
    /// `extent` and `max_lod` come from the backfiller and define the sparse
    /// file layout. If an existing sidecar describes a different layout, all
    /// data files plus the sidecar are renamed aside (`*.stale.<unix_ts>`)
    /// and a fresh cache is created — silent corruption is worse than the
    /// re-download cost.
    pub fn new(root: impl Into<PathBuf>, volume_id: String, extent: [u32; 3], max_lod: u8) -> Self {
        let root = root.into();
        std::fs::create_dir_all(&root).expect("create cache root");

        let expected = sidecar::Header::new(volume_id, extent, max_lod);
        let sidecar_path = sidecar::sidecar_path(&root);

        let sidecar = match Sidecar::load(&sidecar_path) {
            Ok(Some(s)) if s.header.matches(&expected) => Arc::new(s),
            Ok(Some(other)) => {
                log::warn!(
                    "[cache] sidecar mismatch (have vol={} extent={:?} max_lod={} chunk_side={}; want vol={} extent={:?} max_lod={} chunk_side={}); rebuilding",
                    other.header.volume_id,
                    other.header.extent,
                    other.header.max_lod,
                    other.header.chunk_side,
                    expected.volume_id,
                    expected.extent,
                    expected.max_lod,
                    expected.chunk_side,
                );
                stale_rename_everything(&root);
                Arc::new(Sidecar::empty(expected))
            }
            Ok(None) => Arc::new(Sidecar::empty(expected)),
            Err(e) => {
                log::warn!("[cache] sidecar load failed ({}); treating as empty + renaming aside", e);
                stale_rename_everything(&root);
                Arc::new(Sidecar::empty(expected))
            }
        };

        let lods: Vec<LodSlot> = sidecar
            .header
            .lods
            .iter()
            .map(|d| LodSlot {
                dims: *d,
                opened: Mutex::new(None),
            })
            .collect();

        let inner = Arc::new(DiskStoreInner {
            root,
            sidecar,
            lods,
            sync_state: SyncState {
                inner: Mutex::new(SyncInner {
                    shutdown: false,
                    wake_now: false,
                    last_sync: Instant::now(),
                }),
                cv: Condvar::new(),
            },
        });

        let sync_inner = inner.clone();
        let sync_thread = std::thread::Builder::new()
            .name("vesuvius-cache-sync".into())
            .spawn(move || sync_loop(sync_inner))
            .expect("spawn sync thread");

        Self {
            inner,
            sync_thread: Mutex::new(Some(sync_thread)),
        }
    }

    pub fn load(&self, key: ChunkKey) -> LoadOutcome {
        let (_, idx) = match self.inner.resolve(key) {
            Some(p) => p,
            None => return LoadOutcome::Missing,
        };
        match self.inner.sidecar.get_state(key.lod, idx) {
            STATE_MISSING => LoadOutcome::Missing,
            STATE_EMPTY => LoadOutcome::Empty,
            STATE_RESIDENT => match self.inner.ensure_open(key.lod) {
                Ok(lf) => {
                    let off = (idx as usize) * CHUNK_VOXELS;
                    debug_assert!(off + CHUNK_VOXELS <= lf.mmap.len());
                    LoadOutcome::Resident {
                        mmap: lf.mmap,
                        offset: off,
                    }
                }
                Err(e) => {
                    log::warn!("[{}] ensure_open failed during load: {}", key, e);
                    LoadOutcome::Missing
                }
            },
            other => {
                log::warn!("[{}] unknown sidecar state {}", key, other);
                LoadOutcome::Missing
            }
        }
    }

    /// Convenience used after a `write_atomic` to obtain the freshly-written
    /// slice for in-memory residency. Returns `None` if the chunk isn't
    /// marked Resident in the sidecar.
    pub fn try_load(&self, key: ChunkKey) -> Option<(Arc<Mmap>, usize)> {
        match self.load(key) {
            LoadOutcome::Resident { mmap, offset } => Some((mmap, offset)),
            _ => None,
        }
    }

    /// Write a 64³ chunk into its slot in the LOD data file, then publish
    /// `Resident` in the sidecar bitmap with `Release`. Concurrent readers
    /// using `Acquire` are guaranteed to see all 256 KiB of bytes before
    /// observing the Resident transition.
    pub fn write_atomic(&self, key: ChunkKey, bytes: &[u8]) -> std::io::Result<()> {
        assert_eq!(
            bytes.len(),
            CHUNK_VOXELS,
            "unified-cache: backfiller returned {} bytes, expected {}",
            bytes.len(),
            CHUNK_VOXELS
        );
        let (_, idx) = self.inner.resolve(key).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("chunk {} out of bounds for this LOD", key),
            )
        })?;
        let lf = self.inner.ensure_open(key.lod)?;
        let off = (idx as u64) * CHUNK_VOXELS as u64;
        pwrite_all(&lf.file, off, bytes)?;
        self.inner.sidecar.set_state(key.lod, idx, STATE_RESIDENT);
        self.inner.maybe_wake_sync();
        Ok(())
    }

    /// Synchronously snapshot the sidecar, fsync any LOD data files that
    /// had transitions since the last sync, and atomically write the
    /// sidecar to disk. Used on graceful shutdown and by tests that need
    /// durability before the periodic sync would fire.
    pub fn flush(&self) {
        do_sync(&self.inner);
    }

    /// Mark `key` as definitively absent in the sidecar. No bytes are written
    /// to the data file.
    pub fn mark_empty(&self, key: ChunkKey) -> std::io::Result<()> {
        let (_, idx) = self.inner.resolve(key).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("chunk {} out of bounds for this LOD", key),
            )
        })?;
        self.inner.sidecar.set_state(key.lod, idx, STATE_EMPTY);
        self.inner.maybe_wake_sync();
        Ok(())
    }
}

impl DiskStoreInner {
    fn resolve(&self, key: ChunkKey) -> Option<(LodDims, u64)> {
        let slot = self.lods.get(key.lod as usize)?;
        slot.dims.linear_index(key.x, key.y, key.z).map(|i| (slot.dims, i))
    }

    fn ensure_open(&self, lod: u8) -> std::io::Result<LodFile> {
        let slot = &self.lods[lod as usize];
        if let Some(lf) = slot.opened.lock().unwrap().as_ref() {
            return Ok(lf.clone());
        }
        let mut guard = slot.opened.lock().unwrap();
        if let Some(lf) = guard.as_ref() {
            return Ok(lf.clone());
        }
        let total = slot.dims.total_bytes();
        let path = self.root.join(lod_filename(lod));
        let file = OpenOptions::new().read(true).write(true).create(true).open(&path)?;
        let cur_len = file.metadata()?.len();
        if cur_len == 0 {
            file.set_len(total)?;
        } else if cur_len != total {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "lod {} data file at {} has length {} (expected {}); refusing to use",
                    lod,
                    path.display(),
                    cur_len,
                    total,
                ),
            ));
        }
        let mmap = unsafe { MmapOptions::new().len(total as usize).map(&file)? };
        let lf = LodFile {
            file: Arc::new(file),
            mmap: Arc::new(mmap),
        };
        *guard = Some(lf.clone());
        Ok(lf)
    }

    fn maybe_wake_sync(&self) {
        if self.sidecar.total_pending() >= SYNC_COUNT_THRESHOLD {
            let mut g = self.sync_state.inner.lock().unwrap();
            g.wake_now = true;
            self.sync_state.cv.notify_all();
        }
    }
}

impl Drop for DiskStore {
    fn drop(&mut self) {
        {
            let mut g = self.inner.sync_state.inner.lock().unwrap();
            g.shutdown = true;
            self.inner.sync_state.cv.notify_all();
        }
        if let Some(h) = self.sync_thread.lock().unwrap().take() {
            let _ = h.join();
        }
    }
}

fn sync_loop(inner: Arc<DiskStoreInner>) {
    loop {
        let should_shutdown = {
            let g = inner.sync_state.inner.lock().unwrap();
            let timeout = SYNC_INTERVAL
                .checked_sub(g.last_sync.elapsed())
                .unwrap_or(Duration::from_secs(0));
            let (mut g, _) = inner
                .sync_state
                .cv
                .wait_timeout_while(g, timeout, |inner| !inner.shutdown && !inner.wake_now)
                .unwrap();
            let shutdown = g.shutdown;
            g.wake_now = false;
            g.last_sync = Instant::now();
            shutdown
        };
        do_sync(&inner);
        if should_shutdown {
            return;
        }
    }
}

fn do_sync(inner: &DiskStoreInner) {
    let snap = inner.sidecar.snapshot();
    if snap.pending.iter().all(|&p| p == 0) {
        // Nothing changed since the last sync; nothing to flush.
        return;
    }
    // fsync only LODs that had transitions captured in this snapshot.
    for (lod_idx, &count) in snap.pending.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let lf_opt = inner.lods[lod_idx].opened.lock().unwrap().as_ref().cloned();
        let Some(lf) = lf_opt else {
            log::warn!("[cache] sync: LOD {} has {} pending but no open file", lod_idx, count);
            continue;
        };
        if let Err(e) = lf.file.sync_data() {
            log::warn!("[cache] sync: fsync LOD {} failed: {}", lod_idx, e);
        }
    }
    if let Err(e) = snap.write_to(&inner.sidecar.header, &sidecar::sidecar_path(&inner.root)) {
        log::warn!("[cache] sync: sidecar write failed: {}", e);
    }
}

fn lod_filename(lod: u8) -> String {
    format!("chunks-L{:02}.dat", lod)
}

fn stale_rename_everything(root: &Path) {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        let is_data = name_str.starts_with("chunks-L") && name_str.ends_with(".dat");
        let is_sidecar = name_str == "chunks.idx";
        if !is_data && !is_sidecar {
            continue;
        }
        let from = entry.path();
        let to = root.join(format!("{}.stale.{}", name_str, ts));
        if let Err(e) = std::fs::rename(&from, &to) {
            log::warn!(
                "[cache] failed to rename {} → {}: {}",
                from.display(),
                to.display(),
                e
            );
        }
    }
}

fn pwrite_all(file: &File, off: u64, mut buf: &[u8]) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        let mut off = off;
        while !buf.is_empty() {
            let n = file.write_at(buf, off)?;
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "write_at returned 0",
                ));
            }
            buf = &buf[n..];
            off += n as u64;
        }
        Ok(())
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut off = off;
        while !buf.is_empty() {
            let n = file.seek_write(buf, off)?;
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "seek_write returned 0",
                ));
            }
            buf = &buf[n..];
            off += n as u64;
        }
        Ok(())
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (file, off, buf);
        compile_error!("sparse cache store requires unix or windows");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tempdir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "vesuvius-cache-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn make_chunk(seed: u8) -> Vec<u8> {
        let mut v = vec![0u8; CHUNK_VOXELS];
        for (i, b) in v.iter_mut().enumerate() {
            *b = ((i as u8) ^ seed).wrapping_mul(31);
        }
        v
    }

    #[test]
    fn write_then_load_roundtrip() {
        let tmp = tempdir();
        let store = DiskStore::new(&tmp, "v".into(), [128, 128, 128], 0);
        let key = ChunkKey::new(0, 1, 0, 1);
        assert!(matches!(store.load(key), LoadOutcome::Missing));

        let bytes = make_chunk(0x42);
        store.write_atomic(key, &bytes).unwrap();
        let (mmap, off) = store.try_load(key).expect("resident after write");
        assert_eq!(&mmap[off..off + 16], &bytes[..16]);
        assert_eq!(mmap[off + CHUNK_VOXELS - 1], bytes[CHUNK_VOXELS - 1]);
    }

    #[test]
    fn empty_sentinel_persisted_across_reopen() {
        let tmp = tempdir();
        let key = ChunkKey::new(0, 0, 0, 0);
        {
            let store = DiskStore::new(&tmp, "v".into(), [64, 64, 64], 0);
            store.mark_empty(key).unwrap();
            // Drop runs final sync via background thread.
        }
        let store = DiskStore::new(&tmp, "v".into(), [64, 64, 64], 0);
        assert!(matches!(store.load(key), LoadOutcome::Empty));
    }

    #[test]
    fn multi_lod_offsets_disjoint() {
        let tmp = tempdir();
        let store = DiskStore::new(&tmp, "v".into(), [256, 256, 256], 2);
        // Write distinguishable chunks at every LOD, at (0,0,0).
        let mut written = Vec::new();
        for lod in 0..=2u8 {
            let bytes = make_chunk(lod ^ 0xa5);
            store.write_atomic(ChunkKey::new(lod, 0, 0, 0), &bytes).unwrap();
            written.push(bytes);
        }
        for lod in 0..=2u8 {
            let (mmap, off) = store.try_load(ChunkKey::new(lod, 0, 0, 0)).unwrap();
            assert_eq!(&mmap[off..off + 32], &written[lod as usize][..32]);
        }
        assert!(tmp.join("chunks-L00.dat").exists());
        assert!(tmp.join("chunks-L01.dat").exists());
        assert!(tmp.join("chunks-L02.dat").exists());
    }

    #[test]
    fn unused_lod_creates_no_file() {
        let tmp = tempdir();
        let store = DiskStore::new(&tmp, "v".into(), [256, 256, 256], 2);
        store.write_atomic(ChunkKey::new(0, 0, 0, 0), &make_chunk(1)).unwrap();
        assert!(tmp.join("chunks-L00.dat").exists());
        assert!(!tmp.join("chunks-L01.dat").exists());
        assert!(!tmp.join("chunks-L02.dat").exists());
    }

    #[test]
    fn extent_mismatch_rebuilds_files() {
        let tmp = tempdir();
        {
            let store = DiskStore::new(&tmp, "v".into(), [128, 128, 128], 0);
            store.write_atomic(ChunkKey::new(0, 0, 0, 0), &make_chunk(7)).unwrap();
        }
        let store = DiskStore::new(&tmp, "v".into(), [256, 128, 128], 0);
        assert!(matches!(store.load(ChunkKey::new(0, 0, 0, 0)), LoadOutcome::Missing));
        let any_stale = std::fs::read_dir(&tmp)
            .unwrap()
            .flatten()
            .any(|e| e.file_name().to_string_lossy().contains(".stale."));
        assert!(any_stale, "expected a .stale.* file in {}", tmp.display());
    }

    #[test]
    fn rejects_truncated_data_file() {
        let tmp = tempdir();
        {
            let store = DiskStore::new(&tmp, "v".into(), [128, 128, 128], 0);
            store.write_atomic(ChunkKey::new(0, 0, 0, 0), &make_chunk(1)).unwrap();
        }
        let data = tmp.join("chunks-L00.dat");
        std::fs::OpenOptions::new()
            .write(true)
            .open(&data)
            .unwrap()
            .set_len(128)
            .unwrap();
        let store = DiskStore::new(&tmp, "v".into(), [128, 128, 128], 0);
        assert!(matches!(store.load(ChunkKey::new(0, 0, 0, 0)), LoadOutcome::Missing));
    }

    #[test]
    fn concurrent_writers_distinct_chunks() {
        // At LOD 0 a [2048, 64, 1024] extent gives 32 × 1 × 16 chunks —
        // enough room for the (t, 0, i) keys below.
        let tmp = tempdir();
        let store = Arc::new(DiskStore::new(&tmp, "v".into(), [2048, 64, 1024], 0));
        let mut handles = Vec::new();
        for t in 0..4u8 {
            let store = store.clone();
            handles.push(std::thread::spawn(move || {
                for i in 0..16u32 {
                    let key = ChunkKey::new(0, t as u32, 0, i);
                    let mut bytes = make_chunk(t);
                    bytes[0] = i as u8;
                    store.write_atomic(key, &bytes).unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        for t in 0..4u8 {
            for i in 0..16u32 {
                let key = ChunkKey::new(0, t as u32, 0, i);
                let (mmap, off) = store.try_load(key).unwrap();
                assert_eq!(mmap[off], i as u8, "{}", key);
                assert_eq!(mmap[off + 1], (1u8 ^ t).wrapping_mul(31), "{}", key);
            }
        }
    }
}
