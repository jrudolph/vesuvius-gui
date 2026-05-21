//! On-disk persistence for cache chunks.
//!
//! Layout: a 3D grid of sparse "shard" data files per LOD,
//! `{root}/chunks-L{lod:02}-X{sx:03}-Y{sy:03}-Z{sz:03}.dat`. A shard owns
//! every cache chunk falling inside a `SHARD_CHUNKS_PER_AXIS³` cube
//! (defaults to 128³ chunks = 8192³ voxels). At 256 KiB per chunk that
//! gives a constant logical file size of `128³ · 256 KiB = 2³⁹ bytes`
//! (512 GiB) — well below the per-file sparse-allocation ceilings imposed
//! by common file systems on very large single files. Each shard is
//! `set_len(SHARD_BYTES)` on first open; only chunks actually written
//! occupy physical blocks (ext4/xfs/btrfs/APFS).
//!
//! Chunk state (Missing / Resident / Empty) still lives in a single
//! `chunks.idx` sidecar (`super::sidecar::Sidecar`) — one byte per slot per
//! LOD, addressed by the global linear index `((z·ny + y)·nx + x)`
//! regardless of which shard the chunk lives in. Writers update the bitmap
//! with `Release` after pwriting bytes; readers observing `Resident` with
//! `Acquire` are guaranteed to see the full 256 KiB through the mmap of
//! the matching shard (same inode → same page cache).
//!
//! A background sync thread snapshots the bitmap, fsyncs every shard file
//! currently open for any LOD that saw transitions, and atomically renames
//! the sidecar into place every `SYNC_INTERVAL` or after
//! `SYNC_COUNT_THRESHOLD` transitions — whichever comes first. The sidecar
//! is always a strict subset of durable bytes, so a crash loses at most
//! the last sync interval of work (chunks are re-downloaded).

use super::sidecar::{self, LodDims, Sidecar, STATE_EMPTY, STATE_MISSING, STATE_RESIDENT};
use super::state::ChunkKey;
use super::CHUNK_VOXELS;
use memmap::{Mmap, MmapOptions};
use std::collections::HashMap;
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

type ShardCoord = (u32, u32, u32);

struct LodSlot {
    dims: LodDims,
    /// Lazily populated per shard. The HashMap lookup runs once per chunk
    /// state transition (inside `load` / `write_atomic`), not per voxel
    /// read — voxel access goes through the cached `Arc<Mmap>` returned
    /// from `load` and never re-enters this map.
    opened: Mutex<HashMap<ShardCoord, LodFile>>,
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
    shard_chunks_per_axis: u32,
    sidecar: Arc<Sidecar>,
    lods: Vec<LodSlot>,
    sync_state: SyncState,
}

pub struct DiskStore {
    inner: Arc<DiskStoreInner>,
    sync_thread: Mutex<Option<JoinHandle<()>>>,
}

/// Default shard side length in chunks. `128³ · 256 KiB = 2³⁹` bytes per
/// shard — comfortably under per-file sparse ceilings while keeping shard
/// counts tiny for typical volumes (worst case 80 TiB across ~160 shards
/// for our largest current volume).
const SHARD_CHUNKS_PER_AXIS: u32 = 128;

const SYNC_INTERVAL: Duration = Duration::from_secs(10);
const SYNC_COUNT_THRESHOLD: u64 = 256;

struct ResolvedKey {
    sidecar_idx: u64,
    shard: ShardCoord,
    in_shard_idx: u64,
}

impl DiskStore {
    /// Open (or create) the cache directory for a volume.
    ///
    /// `extent` and `max_lod` come from the backfiller and define the sparse
    /// shard layout. If an existing sidecar describes a different layout,
    /// all data files plus the sidecar are renamed aside
    /// (`*.stale.<unix_ts>`) and a fresh cache is created — silent
    /// corruption is worse than the re-download cost.
    pub fn new(root: impl Into<PathBuf>, volume_id: String, extent: [u32; 3], max_lod: u8) -> Self {
        Self::new_with_shard_chunks_per_axis(root, volume_id, extent, max_lod, SHARD_CHUNKS_PER_AXIS)
    }

    /// Test-only constructor letting callers pick a smaller shard side so
    /// multi-shard layouts can be exercised without inflating extents.
    fn new_with_shard_chunks_per_axis(
        root: impl Into<PathBuf>,
        volume_id: String,
        extent: [u32; 3],
        max_lod: u8,
        shard_chunks_per_axis: u32,
    ) -> Self {
        assert!(shard_chunks_per_axis > 0, "shard side must be > 0");
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
                opened: Mutex::new(HashMap::new()),
            })
            .collect();

        let inner = Arc::new(DiskStoreInner {
            root,
            shard_chunks_per_axis,
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
        let r = match self.inner.resolve(key) {
            Some(r) => r,
            None => return LoadOutcome::Missing,
        };
        match self.inner.sidecar.get_state(key.lod, r.sidecar_idx) {
            STATE_MISSING => LoadOutcome::Missing,
            STATE_EMPTY => LoadOutcome::Empty,
            STATE_RESIDENT => match self.inner.ensure_open(key.lod, r.shard) {
                Ok(lf) => {
                    let off = (r.in_shard_idx as usize) * CHUNK_VOXELS;
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

    /// Write a 64³ chunk into its slot in the matching shard file, then
    /// publish `Resident` in the sidecar bitmap with `Release`. Concurrent
    /// readers using `Acquire` are guaranteed to see all 256 KiB of bytes
    /// before observing the Resident transition.
    pub fn write_atomic(&self, key: ChunkKey, bytes: &[u8]) -> std::io::Result<()> {
        assert_eq!(
            bytes.len(),
            CHUNK_VOXELS,
            "unified-cache: backfiller returned {} bytes, expected {}",
            bytes.len(),
            CHUNK_VOXELS
        );
        let r = self.inner.resolve(key).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("chunk {} out of bounds for this LOD", key),
            )
        })?;
        let lf = self.inner.ensure_open(key.lod, r.shard)?;
        let off = r.in_shard_idx * CHUNK_VOXELS as u64;
        pwrite_all(&lf.file, off, bytes)?;
        self.inner.sidecar.set_state(key.lod, r.sidecar_idx, STATE_RESIDENT);
        self.inner.maybe_wake_sync();
        Ok(())
    }

    /// Synchronously snapshot the sidecar, fsync any shard files that
    /// had transitions since the last sync, and atomically write the
    /// sidecar to disk. Used on graceful shutdown and by tests that need
    /// durability before the periodic sync would fire.
    pub fn flush(&self) {
        do_sync(&self.inner);
    }

    /// Mark `key` as definitively absent in the sidecar. No bytes are written
    /// to a data file.
    pub fn mark_empty(&self, key: ChunkKey) -> std::io::Result<()> {
        let r = self.inner.resolve(key).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("chunk {} out of bounds for this LOD", key),
            )
        })?;
        self.inner.sidecar.set_state(key.lod, r.sidecar_idx, STATE_EMPTY);
        self.inner.maybe_wake_sync();
        Ok(())
    }
}

impl DiskStoreInner {
    fn resolve(&self, key: ChunkKey) -> Option<ResolvedKey> {
        let slot = self.lods.get(key.lod as usize)?;
        let sidecar_idx = slot.dims.linear_index(key.x, key.y, key.z)?;
        let sca = self.shard_chunks_per_axis;
        let shard = (key.x / sca, key.y / sca, key.z / sca);
        let wx = (key.x % sca) as u64;
        let wy = (key.y % sca) as u64;
        let wz = (key.z % sca) as u64;
        let s = sca as u64;
        let in_shard_idx = (wz * s + wy) * s + wx;
        Some(ResolvedKey {
            sidecar_idx,
            shard,
            in_shard_idx,
        })
    }

    fn ensure_open(&self, lod: u8, shard: ShardCoord) -> std::io::Result<LodFile> {
        let slot = &self.lods[lod as usize];
        if let Some(lf) = slot.opened.lock().unwrap().get(&shard) {
            return Ok(lf.clone());
        }
        let mut guard = slot.opened.lock().unwrap();
        if let Some(lf) = guard.get(&shard) {
            return Ok(lf.clone());
        }
        let total = self.shard_bytes();
        let path = self.root.join(shard_filename(lod, shard));
        let file = OpenOptions::new().read(true).write(true).create(true).open(&path)?;
        let cur_len = file.metadata()?.len();
        if cur_len == 0 {
            file.set_len(total)?;
        } else if cur_len != total {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "lod {} shard {:?} at {} has length {} (expected {}); refusing to use",
                    lod,
                    shard,
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
        guard.insert(shard, lf.clone());
        Ok(lf)
    }

    fn shard_bytes(&self) -> u64 {
        let n = self.shard_chunks_per_axis as u64;
        n * n * n * CHUNK_VOXELS as u64
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
    // Pending counts LOD-wide transitions; we don't track which shard each
    // transition landed in, so conservatively fsync every shard we have
    // open for that LOD. A write-path shard is always open here (its
    // pwrite happened-before the Release that bumped pending). fsync on
    // read-only / clean shards is harmless — essentially a no-op.
    for (lod_idx, &count) in snap.pending.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let opened: Vec<(ShardCoord, LodFile)> = inner.lods[lod_idx]
            .opened
            .lock()
            .unwrap()
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        if opened.is_empty() {
            log::warn!(
                "[cache] sync: LOD {} has {} pending but no open shards",
                lod_idx,
                count
            );
            continue;
        }
        for (shard, lf) in opened {
            if let Err(e) = lf.file.sync_data() {
                log::warn!(
                    "[cache] sync: fsync LOD {} shard {:?} failed: {}",
                    lod_idx,
                    shard,
                    e
                );
            }
        }
    }
    if let Err(e) = snap.write_to(&inner.sidecar.header, &sidecar::sidecar_path(&inner.root)) {
        log::warn!("[cache] sync: sidecar write failed: {}", e);
    }
}

fn shard_filename(lod: u8, shard: ShardCoord) -> String {
    format!(
        "chunks-L{:02}-X{:03}-Y{:03}-Z{:03}.dat",
        lod, shard.0, shard.1, shard.2
    )
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
        assert!(tmp.join("chunks-L00-X000-Y000-Z000.dat").exists());
        assert!(tmp.join("chunks-L01-X000-Y000-Z000.dat").exists());
        assert!(tmp.join("chunks-L02-X000-Y000-Z000.dat").exists());
    }

    #[test]
    fn unused_lod_creates_no_file() {
        let tmp = tempdir();
        let store = DiskStore::new(&tmp, "v".into(), [256, 256, 256], 2);
        store.write_atomic(ChunkKey::new(0, 0, 0, 0), &make_chunk(1)).unwrap();
        assert!(tmp.join("chunks-L00-X000-Y000-Z000.dat").exists());
        assert!(!tmp.join("chunks-L01-X000-Y000-Z000.dat").exists());
        assert!(!tmp.join("chunks-L02-X000-Y000-Z000.dat").exists());
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
        let data = tmp.join("chunks-L00-X000-Y000-Z000.dat");
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

    #[test]
    fn writes_across_shard_boundary() {
        // Shard side of 4 chunks → 4³ × 256 KiB = 16 MiB per shard file.
        // Extent [1024, 64, 64] gives 16 × 1 × 1 chunks at LOD 0, which
        // spans 4 shards along X.
        let tmp = tempdir();
        let store = DiskStore::new_with_shard_chunks_per_axis(&tmp, "v".into(), [1024, 64, 64], 0, 4);
        for cx in 0..16u32 {
            let mut bytes = make_chunk(cx as u8);
            bytes[0] = cx as u8;
            bytes[1] = (cx as u8).wrapping_add(0x80);
            store.write_atomic(ChunkKey::new(0, cx, 0, 0), &bytes).unwrap();
        }
        for cx in 0..16u32 {
            let (mmap, off) = store.try_load(ChunkKey::new(0, cx, 0, 0)).unwrap();
            assert_eq!(mmap[off], cx as u8, "chunk {}", cx);
            assert_eq!(mmap[off + 1], (cx as u8).wrapping_add(0x80), "chunk {}", cx);
        }
        for sx in 0..4u32 {
            let p = tmp.join(format!("chunks-L00-X{:03}-Y000-Z000.dat", sx));
            assert!(p.exists(), "missing shard {}: {}", sx, p.display());
        }
        assert!(!tmp.join("chunks-L00-X004-Y000-Z000.dat").exists());
    }

}
