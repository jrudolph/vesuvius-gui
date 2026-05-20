//! On-disk persistence for cache chunks.
//!
//! Layout: `{root}/L{lod:02}/{z}/{y}/{x}.raw`, fixed `CHUNK_VOXELS` bytes per
//! file. Writes are atomic via temp-file + rename, following the pattern in
//! `vesuvius-zarr::v3::DecodedCache`.

use super::state::ChunkKey;
use super::CHUNK_VOXELS;
use memmap::{Mmap, MmapOptions};
use std::fs::File;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug)]
pub struct DiskStore {
    root: PathBuf,
}

/// Result of consulting on-disk state for a chunk.
pub enum LoadOutcome {
    /// A full `.raw` chunk was found and mmap'd.
    Resident(Mmap),
    /// A `.empty` sentinel was found — chunk is definitively absent.
    Empty,
    /// Nothing on disk yet.
    Missing,
}

impl DiskStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn path_for(&self, key: ChunkKey) -> PathBuf {
        self.root
            .join(format!("L{:02}", key.lod))
            .join(key.z.to_string())
            .join(key.y.to_string())
            .join(format!("{}.raw", key.x))
    }

    fn empty_path_for(&self, key: ChunkKey) -> PathBuf {
        self.root
            .join(format!("L{:02}", key.lod))
            .join(key.z.to_string())
            .join(key.y.to_string())
            .join(format!("{}.empty", key.x))
    }

    /// Try to load and mmap a chunk file. Returns `None` for any failure
    /// (missing, truncated, IO error) — caller falls through to fetch.
    pub fn try_load(&self, key: ChunkKey) -> Option<Mmap> {
        let path = self.path_for(key);
        let file = File::open(&path).ok()?;
        let mmap = unsafe { MmapOptions::new().map(&file).ok()? };
        if mmap.len() != CHUNK_VOXELS {
            log::warn!(
                "[{}] wrong size {} (expected {}) at {}",
                key,
                mmap.len(),
                CHUNK_VOXELS,
                path.display()
            );
            return None;
        }
        Some(mmap)
    }

    /// Resolve a chunk against on-disk state: prefer a `.raw` chunk; fall
    /// back to a `.empty` sentinel; otherwise Missing. Checked atomically
    /// enough for our purposes — concurrent writes resolve via rename.
    pub fn load(&self, key: ChunkKey) -> LoadOutcome {
        if let Some(mmap) = self.try_load(key) {
            return LoadOutcome::Resident(mmap);
        }
        if self.empty_path_for(key).exists() {
            return LoadOutcome::Empty;
        }
        LoadOutcome::Missing
    }

    /// Persist a definitive-absence sentinel for `key`. Non-fatal on
    /// failure: returning Err just means the next session will retry the
    /// fetch (and likely re-mark empty), so log + drop.
    pub fn mark_empty(&self, key: ChunkKey) -> std::io::Result<()> {
        let path = self.empty_path_for(key);
        let parent = path.parent().expect("disk path has parent");
        std::fs::create_dir_all(parent)?;
        // Zero-byte file is enough — existence is the signal.
        std::fs::File::create(&path)?;
        Ok(())
    }

    /// Atomic write: stage to a unique temp name in the same directory, then
    /// rename. Concurrent writes of identical bytes are safe (last rename
    /// wins).
    pub fn write_atomic(&self, key: ChunkKey, bytes: &[u8]) -> std::io::Result<()> {
        assert_eq!(
            bytes.len(),
            CHUNK_VOXELS,
            "unified-cache: backfiller returned {} bytes, expected {}",
            bytes.len(),
            CHUNK_VOXELS
        );
        let final_path = self.path_for(key);
        let parent = final_path.parent().expect("disk path has parent");
        std::fs::create_dir_all(parent)?;

        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let tid = std::thread::current().id();
        let tmp = parent.join(format!("{}.raw.tmp.{}.{:?}", key.x, n, tid));

        if let Err(e) = std::fs::write(&tmp, bytes) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        if let Err(e) = std::fs::rename(&tmp, &final_path) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_then_load_roundtrip() {
        let tmp = tempdir();
        let store = DiskStore::new(&tmp);
        let key = ChunkKey::new(0, 3, 5, 7);

        assert!(store.try_load(key).is_none());

        let mut bytes = vec![0u8; CHUNK_VOXELS];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = (i & 0xff) as u8;
        }
        store.write_atomic(key, &bytes).unwrap();

        let mmap = store.try_load(key).expect("chunk should load after write");
        assert_eq!(mmap.len(), CHUNK_VOXELS);
        assert_eq!(&mmap[..16], &bytes[..16]);
        assert_eq!(mmap[CHUNK_VOXELS - 1], bytes[CHUNK_VOXELS - 1]);
    }

    #[test]
    fn rejects_truncated_file() {
        let tmp = tempdir();
        let store = DiskStore::new(&tmp);
        let key = ChunkKey::new(0, 0, 0, 0);

        let path = store.path_for(key);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"short").unwrap();

        assert!(store.try_load(key).is_none());
    }

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
}
