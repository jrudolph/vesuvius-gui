use super::state::ChunkKey;

#[derive(Debug)]
pub enum BackfillError {
    /// Chunk is outside the volume bounds — return Missing permanently.
    OutOfBounds,
    /// Transient failure (network, decode glitch, …). Triggers a cooldown.
    Transient(String),
    /// Permanent failure — the backfiller is sure this chunk will never load.
    Permanent(String),
}

impl std::fmt::Display for BackfillError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackfillError::OutOfBounds => write!(f, "out of bounds"),
            BackfillError::Transient(s) => write!(f, "transient: {}", s),
            BackfillError::Permanent(s) => write!(f, "permanent: {}", s),
        }
    }
}

impl std::error::Error for BackfillError {}

/// Source of chunk data for the unified cache.
///
/// Implementations are owned by a `ChunkCache` and called from worker threads,
/// so they must be `Send + Sync`. Returned `Vec<u8>` must be exactly
/// `CHUNK_VOXELS` bytes — backends with a larger native unit are expected to
/// slice down internally.
pub trait ChunkBackfiller: Send + Sync {
    /// Highest LOD level the backfiller can serve (inclusive). LOD 0 must
    /// always be supported.
    fn max_lod(&self) -> u8;

    /// Volume extent in voxels at LOD 0, used to detect out-of-bounds reads
    /// without round-tripping through the backfiller.
    fn voxel_extent(&self) -> [u32; 3];

    /// Synchronously fetch one chunk. May block on I/O / decode. Called from
    /// a worker thread pool owned by the cache.
    fn fetch(&self, key: ChunkKey) -> Result<Vec<u8>, BackfillError>;

    /// Stable identifier used for the on-disk cache directory. Different
    /// volumes must produce different ids; the same volume must produce the
    /// same id across runs.
    fn volume_id(&self) -> String;
}
