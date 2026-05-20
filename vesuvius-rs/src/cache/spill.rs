//! Per-source download spill store.
//!
//! Between "the HTTP body landed" and "all sibling sources for the same
//! cache chunk are ready and Extract runs," compressed source bytes would
//! otherwise sit on the heap. With many parallel downloads (32 HTTP
//! workers, hundreds of in-flight chunks per frame) that easily costs
//! hundreds of MB of resident memory.
//!
//! The spill store writes those bytes to disk atomically (temp + rename)
//! and hands back an `Mmap`. Extract reads the bytes through the mmap, so
//! the kernel — not the process heap — decides what's resident. Sibling
//! cache chunks consuming the same source share one mmap via the
//! `Arc<Mmap>` SourcePayload, so we get free sharing across the 64
//! cache-chunks-per-native-chunk fan-out.
//!
//! Layout: `{root}/{aa}/{rest}.bin`, where `{aa}` is the first two hex
//! chars of the SHA-256 of the source key (filesystem-safe regardless of
//! the backfiller's key format) and `{rest}` is the remaining hex.
//!
//! No automatic cleanup — spill files accumulate alongside chunk files in
//! the cache dir. Users can wipe the directory on demand; a future LRU
//! pass over the on-disk cache will treat spill files the same way.

use memmap::{Mmap, MmapOptions};
use sha2::{Digest, Sha256};
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

    fn path_for(&self, source_key: &str) -> PathBuf {
        let hex = format!("{:x}", Sha256::digest(source_key.as_bytes()));
        self.root.join(&hex[..2]).join(format!("{}.bin", &hex[2..]))
    }

    /// Try to mmap an already-spilled source. Returns `None` if the file
    /// doesn't exist or can't be opened. Currently used by tests only —
    /// production reuses the `Mmap` returned from `write_and_mmap`, since
    /// the source state holds it for the lifetime of the consuming chunks.
    #[allow(dead_code)]
    pub fn try_mmap(&self, source_key: &str) -> Option<Mmap> {
        let path = self.path_for(source_key);
        let file = File::open(&path).ok()?;
        unsafe { MmapOptions::new().map(&file).ok() }
    }

    /// Write `bytes` to the spill location for `source_key` and return an
    /// mmap of the final file. Stages to a temp name in the same
    /// directory, then renames atomically — concurrent writers race
    /// harmlessly (the final state is some valid copy of the bytes).
    pub fn write_and_mmap(&self, source_key: &str, bytes: &[u8]) -> std::io::Result<Mmap> {
        let final_path = self.path_for(source_key);
        let parent = final_path
            .parent()
            .expect("spill path always has a parent");
        std::fs::create_dir_all(parent)?;

        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let tid = std::thread::current().id();
        let tmp = parent.join(format!(
            "{}.bin.tmp.{}.{:?}",
            final_path.file_stem().and_then(|s| s.to_str()).unwrap_or("spill"),
            n,
            tid
        ));

        {
            let mut f = File::create(&tmp).map_err(|e| {
                log::warn!("spill: create tmp failed at {}: {}", tmp.display(), e);
                e
            })?;
            f.write_all(bytes)?;
        }
        if let Err(e) = std::fs::rename(&tmp, &final_path) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        let file = File::open(&final_path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
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
    fn try_mmap_returns_none_for_missing() {
        let root = tmp_root();
        let store = SpillStore::new(&root);
        assert!(store.try_mmap("never-written").is_none());
    }

    #[test]
    fn try_mmap_finds_previously_written() {
        let root = tmp_root();
        let store = SpillStore::new(&root);
        let _ = store.write_and_mmap("vol/L00/1/2/3", &[7u8; 256]).unwrap();
        let again = store.try_mmap("vol/L00/1/2/3").expect("mmap after write");
        assert_eq!(again.len(), 256);
        assert!(again.iter().all(|b| *b == 7));
    }
}
