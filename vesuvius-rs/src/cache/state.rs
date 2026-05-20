use memmap::Mmap;
use std::sync::Arc;
use std::time::SystemTime;

/// Address of one cache chunk within a volume.
///
/// `lod` is the mip level — LOD 0 is native resolution, LOD N covers a
/// 2^N-sided voxel block per cell. `(x, y, z)` are chunk coordinates at that
/// LOD (so the world voxel range covered is `[x*64*2^lod, (x+1)*64*2^lod)` and
/// likewise for y, z).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ChunkKey {
    pub lod: u8,
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl ChunkKey {
    pub fn new(lod: u8, x: u32, y: u32, z: u32) -> Self {
        Self { lod, x, y, z }
    }
}

/// In-memory state for one chunk.
#[derive(Debug)]
pub enum ChunkState {
    /// Not on disk, not being fetched. Inserted transiently before dispatch.
    Missing,
    /// Fetch in flight on a worker thread.
    Pending,
    /// Loaded, mmap'd from disk.
    Resident(Mmap),
    /// Most recent fetch failed; don't retry until `until`.
    CooldownMiss { until: SystemTime },
}

impl ChunkState {
    pub fn as_resident(self: &Arc<Self>) -> Option<&Mmap> {
        match self.as_ref() {
            ChunkState::Resident(m) => Some(m),
            _ => None,
        }
    }
}
