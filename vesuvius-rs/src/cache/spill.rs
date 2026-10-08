//! Downloaded-source byte stores: the keyed `RawStore` (retention) and
//! the anonymous `SpillStore` (its fallback).
//!
//! Between "the HTTP body landed" and "all sibling sources for the same
//! cache chunk are ready and Extract runs," compressed source bytes would
//! otherwise sit on the heap. With many parallel downloads (32 HTTP
//! workers, hundreds of in-flight chunks per frame) that easily costs
//! hundreds of MB of resident memory. Both stores write the bytes to a
//! file and hand back an `Mmap`, keeping the heap bounded.
//!
//! `RawStore` additionally *retains* the file, keyed by (url, byte
//! range): decoded 64³ chunks amplify ~25x over the compressed wire
//! bytes, so the decoded cache cycles fast — retaining the compressed
//! source turns "decoded chunks evicted, region revisited" into a local
//! re-decode instead of a repeat download.
//!
//! `SpillStore` is the legacy/fallback path: write a temp file, mmap,
//! then **immediately unlink**. Linux keeps the inode alive while the
//! mmap references it — an anonymous, never-listed kernel blob reclaimed
//! the moment the last `Mmap` drops.
//!
//! Sibling cache chunks consuming the same source share one mmap via the
//! `Arc<Mmap>` SourcePayload. The cache evicts the source entry (last
//! `Arc<Mmap>` reference) when its consumer refcount drops to zero — see
//! `cache::Inner::extract_chunk`.

use memmap::{Mmap, MmapOptions};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

#[derive(Debug, Clone)]
pub struct SpillStore {
    root: PathBuf,
}

impl SpillStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Write `bytes` to a temp file under `root`, mmap, then unlink. The
    /// returned `Mmap` is the only reference to the inode after this
    /// returns — drop it and the kernel reclaims the file. Spill files
    /// are single-use and never looked up again by name.
    pub fn write_and_mmap(&self, bytes: &[u8]) -> std::io::Result<Mmap> {
        std::fs::create_dir_all(&self.root)?;

        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let path = self.root.join(format!("spill-{}-{}.bin", pid, n));

        {
            let mut f = File::create(&path)?;
            f.write_all(bytes)?;
        }
        let file = File::open(&path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        // Anonymise: drop the directory entry; the open fd + mmap keep the
        // file alive. Failure here is non-fatal — worst case the file is
        // left behind for the next run to clean up.
        if let Err(e) = std::fs::remove_file(&path) {
            log::trace!("spill unlink failed: {} (leaving {} on disk)", e, path.display());
        }
        Ok(mmap)
    }
}

/// Env var overriding the raw-source cache budget, in GB. `0` disables
/// retention entirely (downloads spill anonymously, exactly the old
/// behavior).
pub const RAW_CACHE_CAP_ENV_VAR: &str = "VESUVIUS_RAW_CACHE_GB";

/// Default raw-source budget: 24 GB. At the c3d target compression of
/// ~25x this "remembers" ~600 GB of decoded volume — about 2x what the
/// decoded chunk cache can hold at its default cap — for 8% of the disk.
const DEFAULT_RAW_CACHE_CAP: u64 = 24 * 1024 * 1024 * 1024;

/// Evict down to this fraction of the cap (percent) so eviction passes
/// amortize instead of firing on every put.
const RAW_EVICT_TO_PCT: u64 = 90;

/// Minimum free space on the store's filesystem. When a put sees less
/// than this, it runs an eviction pass that also reclaims the deficit —
/// retention must never be the thing that fills the disk. Matches the
/// unified purger's floor.
const RAW_MIN_FREE_BYTES: u64 = 5 * 1024 * 1024 * 1024;

/// Keyed retention store for downloaded source bytes (compressed c3d
/// sub-chunks, zarr chunks, …), keyed by `(url, byte range)`.
///
/// Rationale: decoded chunks amplify ~25x over the wire bytes, so the
/// decoded cache cycles fast and evictions force re-downloads. Keeping
/// the *compressed* bytes on disk turns "evicted, revisit" into a local
/// re-decode instead of a CloudFront round trip. Files are named by the
/// SHA-256 of the key, mtime is bumped on hit, and eviction is
/// oldest-mtime-first once the budget is exceeded.
///
/// Multiple `ChunkCache` instances (one per volume) may point at the
/// same directory; the running total is per-instance and approximate,
/// but eviction passes re-list the directory, so the cap itself is
/// enforced against filesystem truth.
pub struct RawStore {
    root: PathBuf,
    cap_bytes: u64,
    /// Approximate bytes in the store. Seeded by a directory walk at
    /// construction, bumped on put, recomputed by each eviction pass.
    total: AtomicU64,
    /// Serializes eviction passes (puts from many downloader threads).
    evict_lock: Mutex<()>,
    /// Anonymous-spill fallback for `cap_bytes == 0` and for write
    /// failures (disk full): same root, single-use unlinked files.
    spill: SpillStore,
}

impl RawStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root: PathBuf = root.into();
        let cap_bytes = match std::env::var(RAW_CACHE_CAP_ENV_VAR) {
            Ok(v) => match v.trim().parse::<u64>() {
                Ok(gb) => gb * 1024 * 1024 * 1024,
                Err(_) => {
                    log::warn!("{} must be an integer; using default", RAW_CACHE_CAP_ENV_VAR);
                    DEFAULT_RAW_CACHE_CAP
                }
            },
            Err(_) => DEFAULT_RAW_CACHE_CAP,
        };
        Self::with_cap(root, cap_bytes)
    }

    /// Explicit-cap constructor (tests; `new` derives the cap from env).
    pub fn with_cap(root: impl Into<PathBuf>, cap_bytes: u64) -> Self {
        let root: PathBuf = root.into();
        let total = scan_total(&root);
        Self {
            spill: SpillStore::new(root.clone()),
            root,
            cap_bytes,
            total: AtomicU64::new(total),
            evict_lock: Mutex::new(()),
        }
    }

    fn path_for(&self, key: &str) -> PathBuf {
        let hash = format!("{:x}", Sha256::digest(key.as_bytes()));
        self.root.join(format!("{}.raw", hash))
    }

    /// Look up previously downloaded bytes for `key`. Bumps the file's
    /// mtime so eviction treats it as recently used.
    pub fn get(&self, key: &str) -> Option<Mmap> {
        if self.cap_bytes == 0 {
            return None;
        }
        let path = self.path_for(key);
        let file = File::options().read(true).write(true).open(&path).ok()?;
        let len = file.metadata().ok()?.len();
        if len == 0 {
            return None;
        }
        let _ = file.set_modified(std::time::SystemTime::now());
        // SAFETY: standard read-only mmap of a regular file we just opened.
        unsafe { MmapOptions::new().map(&file).ok() }
    }

    /// Drop the bytes stored under `key` (e.g. found to be corrupt), so the
    /// next request downloads them again. Mappings already handed out stay
    /// valid.
    pub fn remove(&self, key: &str) {
        let path = self.path_for(key);
        match std::fs::metadata(&path).and_then(|m| std::fs::remove_file(&path).map(|_| m.len())) {
            Ok(len) => {
                self.total.fetch_sub(len.min(self.total.load(Ordering::Relaxed)), Ordering::Relaxed);
            }
            // Already gone (evicted, or the payload came from the
            // anonymous spill fallback).
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => log::error!("raw store: removing {} failed: {}", path.display(), e),
        }
    }

    /// Persist `bytes` under `key` and return a read mmap of them. On any
    /// write failure (or with retention disabled) falls back to the
    /// anonymous single-use spill so the download itself still succeeds.
    pub fn put(&self, key: &str, bytes: &[u8]) -> std::io::Result<Mmap> {
        if self.cap_bytes == 0 {
            return self.spill.write_and_mmap(bytes);
        }
        match self.put_keyed(key, bytes) {
            Ok(mmap) => Ok(mmap),
            Err(e) => {
                log::debug!("raw store put failed ({}); falling back to anonymous spill", e);
                self.spill.write_and_mmap(bytes)
            }
        }
    }

    fn put_keyed(&self, key: &str, bytes: &[u8]) -> std::io::Result<Mmap> {
        std::fs::create_dir_all(&self.root)?;
        // Disk-free floor: if the filesystem is nearly full, evict enough
        // raw entries to restore the floor before writing — and refuse the
        // keyed write entirely if eviction can't (a full disk is somebody
        // else's data; the caller falls back to the anonymous spill).
        if let Some(free) = super::epoch::statvfs_free(&self.root) {
            if free + (bytes.len() as u64) < RAW_MIN_FREE_BYTES {
                self.evict_pass(Some(RAW_MIN_FREE_BYTES - free));
                let still_low = super::epoch::statvfs_free(&self.root)
                    .is_some_and(|f| f + (bytes.len() as u64) < RAW_MIN_FREE_BYTES);
                if still_low {
                    return Err(std::io::Error::other("raw store: disk below free floor"));
                }
            }
        }
        // The same key can be written concurrently (two fetches of one
        // URL). Each writer gets its own temp file and maps the inode it
        // wrote, so another writer truncating or replacing the file under
        // `path` can never shorten our mapping; the renames just race for
        // which complete copy stays.
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let path = self.path_for(key);
        let tmp = path.with_extension(format!(
            "{}-{}.tmp",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = File::options().read(true).write(true).create_new(true).open(&tmp)?;
        file.write_all(bytes)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        std::fs::rename(&tmp, &path)?;

        let total = self.total.fetch_add(bytes.len() as u64, Ordering::Relaxed) + bytes.len() as u64;
        if total > self.cap_bytes {
            self.evict_pass(None);
        }
        Ok(mmap)
    }

    /// Delete oldest-mtime files until the store is at `RAW_EVICT_TO_PCT`
    /// of cap — or, with `extra_free`, until that many additional bytes
    /// have been reclaimed (disk-floor recovery). Lists the directory
    /// (filesystem truth) rather than trusting the running counter.
    /// Unlinking a file another thread has mmapped is fine — the inode
    /// survives until the mmap drops.
    fn evict_pass(&self, extra_free: Option<u64>) {
        let Ok(_guard) = self.evict_lock.try_lock() else {
            return; // another thread is already evicting
        };
        let mut entries: Vec<(PathBuf, std::time::SystemTime, u64)> = match std::fs::read_dir(&self.root) {
            Ok(rd) => rd
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|x| x == "raw"))
                .filter_map(|e| {
                    let md = e.metadata().ok()?;
                    Some((e.path(), md.modified().ok()?, md.len()))
                })
                .collect(),
            Err(_) => return,
        };
        let mut total: u64 = entries.iter().map(|(_, _, len)| len).sum();
        let mut target = self.cap_bytes * RAW_EVICT_TO_PCT / 100;
        if let Some(extra) = extra_free {
            target = target.min(total.saturating_sub(extra));
        }
        if total > target {
            entries.sort_by_key(|(_, mtime, _)| *mtime);
            let mut evicted = 0usize;
            for (path, _, len) in &entries {
                if total <= target {
                    break;
                }
                if std::fs::remove_file(path).is_ok() {
                    total -= len;
                    evicted += 1;
                }
            }
            log::debug!(
                "raw store evicted {} files, {} MiB resident",
                evicted,
                total / (1024 * 1024)
            );
        }
        self.total.store(total, Ordering::Relaxed);
    }
}

fn scan_total(root: &PathBuf) -> u64 {
    let Ok(rd) = std::fs::read_dir(root) else {
        return 0;
    };
    rd.filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "raw"))
        .filter_map(|e| e.metadata().ok())
        .map(|md| md.len())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    fn tmp_root() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "vesuvius-spill-test-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn write_and_mmap_roundtrip() {
        let root = tmp_root();
        let store = SpillStore::new(&root);
        let bytes: Vec<u8> = (0..4096u32).map(|i| (i & 0xff) as u8).collect();
        let mmap = store.write_and_mmap(&bytes).unwrap();
        assert_eq!(mmap.len(), bytes.len());
        assert_eq!(&mmap[..16], &bytes[..16]);
    }

    #[test]
    fn file_is_unlinked_after_write() {
        let root = tmp_root();
        let store = SpillStore::new(&root);
        let _mmap = store.write_and_mmap(&[42u8; 128]).unwrap();
        // The mmap holds the inode but the directory entry must be gone.
        let listing: Vec<_> = std::fs::read_dir(&root)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_file())
            .collect();
        assert!(
            listing.is_empty(),
            "spill files must be unlinked; found: {:?}",
            listing.iter().map(|e| e.path()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn concurrent_puts_of_one_key_each_map_a_complete_copy() {
        let root = tmp_root();
        let store = std::sync::Arc::new(RawStore::with_cap(&root, 1 << 30));
        let bytes: std::sync::Arc<Vec<u8>> = std::sync::Arc::new((0..262144u32).map(|i| (i * 13 & 0xff) as u8).collect());
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let (store, bytes) = (store.clone(), bytes.clone());
                std::thread::spawn(move || {
                    for _ in 0..20 {
                        let mmap = store.put("https://x/same-chunk", &bytes).unwrap();
                        assert_eq!(mmap.len(), bytes.len(), "mapping shortened by a concurrent writer");
                        assert!(mmap[..] == bytes[..]);
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(store.get("https://x/same-chunk").unwrap().len(), bytes.len());
    }

    #[test]
    fn removed_raw_entries_miss_but_handed_out_mappings_stay_valid() {
        let root = tmp_root();
        let store = RawStore::with_cap(&root, 1 << 30);
        let mmap = store.put("https://x/bad", &[7u8; 32768]).unwrap();
        store.remove("https://x/bad");
        assert!(store.get("https://x/bad").is_none());
        assert_eq!(mmap.len(), 32768);
        store.remove("https://x/bad"); // already gone: fine
    }

    #[test]
    fn raw_store_roundtrip_and_miss() {
        let root = tmp_root();
        let store = RawStore::with_cap(&root, 1 << 30);
        assert!(store.get("https://x/shard#0+100").is_none());
        let bytes: Vec<u8> = (0..2048u32).map(|i| (i * 7 & 0xff) as u8).collect();
        let mmap = store.put("https://x/shard#0+100", &bytes).unwrap();
        assert_eq!(&mmap[..], &bytes[..]);
        let hit = store.get("https://x/shard#0+100").expect("hit after put");
        assert_eq!(&hit[..], &bytes[..]);
        assert!(store.get("https://x/shard#100+100").is_none());
    }

    #[test]
    fn raw_store_disabled_spills_anonymously() {
        let root = tmp_root();
        let store = RawStore::with_cap(&root, 0);
        let mmap = store.put("k", &[7u8; 64]).unwrap();
        assert_eq!(&mmap[..], &[7u8; 64]);
        assert!(store.get("k").is_none());
        // disabled mode must not leave named files behind
        let names: Vec<_> = std::fs::read_dir(&root)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|x| x == "raw"))
            .collect();
        assert!(names.is_empty());
    }

    #[test]
    fn raw_store_evicts_oldest_when_over_cap() {
        let root = tmp_root();
        // cap of 4 KiB, entries of 1 KiB: the 5th put must evict the oldest.
        let store = RawStore::with_cap(&root, 4096);
        for i in 0..5u32 {
            store.put(&format!("key-{}", i), &[i as u8; 1024]).unwrap();
            // distinct mtimes so eviction order is deterministic
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(store.get("key-0").is_none(), "oldest entry should be evicted");
        assert!(store.get("key-4").is_some(), "newest entry must survive");
    }

    #[test]
    fn mmap_outlives_unlink() {
        // After unlink the mmap must still read the original bytes — that's
        // the whole point of the anonymous-file trick.
        let root = tmp_root();
        let store = SpillStore::new(&root);
        let bytes: Vec<u8> = (0..1024u32).map(|i| (i * 31 & 0xff) as u8).collect();
        let mmap = store.write_and_mmap(&bytes).unwrap();
        assert_eq!(&mmap[..], &bytes[..]);
    }
}
