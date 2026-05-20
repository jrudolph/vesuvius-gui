use memmap::Mmap;
use std::fmt;
use std::sync::Arc;
use std::time::SystemTime;

use super::CHUNK_VOXELS;

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

impl fmt::Display for ChunkKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "L{}:{},{},{}", self.lod, self.x, self.y, self.z)
    }
}

/// In-memory state for one chunk.
#[derive(Debug)]
pub enum ChunkState {
    /// Not on disk, not being fetched. Inserted transiently before dispatch.
    Missing,
    /// Fetch in flight on a worker thread.
    Pending,
    /// Loaded — bytes live at `offset..offset+CHUNK_VOXELS` inside `mmap`.
    /// The mmap is shared with every other resident chunk in the same LOD
    /// data file.
    Resident { mmap: Arc<Mmap>, offset: usize },
    /// Definitively absent: every backing source reported "not present"
    /// (e.g., 404 / 403). Sampling returns 0 without consulting any LOD
    /// fallback. Persisted in the chunk-state sidecar so subsequent sessions
    /// hit this without re-fetching.
    Empty,
    /// Most recent fetch failed; don't retry until `until`.
    CooldownMiss { until: SystemTime },
}

impl ChunkState {
    pub fn as_resident(self: &Arc<Self>) -> Option<&[u8]> {
        match self.as_ref() {
            ChunkState::Resident { mmap, offset } => Some(&mmap[*offset..*offset + CHUNK_VOXELS]),
            _ => None,
        }
    }

    /// True for states the cache will not revisit — sampling is final.
    /// Used by the paint/get LOD-fallback walks: stop at the first terminal
    /// state, because `Empty` at the fine LOD overrides any data at a
    /// coarser LOD.
    pub fn is_terminal(&self) -> bool {
        matches!(self, ChunkState::Resident { .. } | ChunkState::Empty)
    }

    pub fn is_empty(&self) -> bool {
        matches!(self, ChunkState::Empty)
    }
}
