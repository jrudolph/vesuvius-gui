//! Per-source download spill store.
//!
//! Between "the HTTP body landed" and "all sibling sources for the same
//! cache chunk are ready and Extract runs," compressed source bytes would
//! otherwise sit on the heap. With many parallel downloads (32 HTTP
//! workers, hundreds of in-flight chunks per frame) that easily costs
//! hundreds of MB of resident memory.
//!
//! The spill store writes those bytes to a temp file, mmaps it, then
//! **immediately unlinks the file**. Linux keeps the inode alive while
//! the open fd or mmap references it — the spill becomes an anonymous,
//! never-directory-listed kernel blob. The kernel reclaims the disk space
//! and any resident pages as soon as the last `Mmap` drops, or as soon as
//! the process exits (no orphan files left in the cache dir).
//!
//! Sibling cache chunks consuming the same source share one mmap via the
//! `Arc<Mmap>` SourcePayload, so we get free sharing across the 64
//! cache-chunks-per-native-chunk fan-out. The cache evicts the source
//! entry (last `Arc<Mmap>` reference) when its consumer refcount drops to
//! zero — see `cache::Inner::extract_chunk`.

use memmap::{Mmap, MmapOptions};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

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
    /// returns — drop it and the kernel reclaims the file. The
    /// `source_key` parameter is accepted for symmetry with previous
    /// designs but currently used only in log messages: spill files are
    /// single-use and never looked up by key.
    pub fn write_and_mmap(&self, source_key: &str, bytes: &[u8]) -> std::io::Result<Mmap> {
        std::fs::create_dir_all(&self.root)?;

        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let tid = std::thread::current().id();
        let path = self.root.join(format!("spill-{}-{:?}-{}.bin", pid, tid, n));

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
            log::trace!(
                "spill: failed to unlink {} for source {} ({}); leaving on disk",
                path.display(),
                source_key,
                e
            );
        }
        Ok(mmap)
    }
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
        let mmap = store.write_and_mmap("vol/L00/0/0/0", &bytes).unwrap();
        assert_eq!(mmap.len(), bytes.len());
        assert_eq!(&mmap[..16], &bytes[..16]);
    }

    #[test]
    fn file_is_unlinked_after_write() {
        let root = tmp_root();
        let store = SpillStore::new(&root);
        let _mmap = store.write_and_mmap("vol/L00/0/0/0", &[42u8; 128]).unwrap();
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
    fn mmap_outlives_unlink() {
        // After unlink the mmap must still read the original bytes — that's
        // the whole point of the anonymous-file trick.
        let root = tmp_root();
        let store = SpillStore::new(&root);
        let bytes: Vec<u8> = (0..1024u32).map(|i| (i * 31 & 0xff) as u8).collect();
        let mmap = store.write_and_mmap("anon-test", &bytes).unwrap();
        assert_eq!(&mmap[..], &bytes[..]);
    }
}
