//! Backfiller trait + plan types.
//!
//! A backfiller doesn't produce chunk bytes directly. It produces a *plan*:
//!   - a list of opaque, deduplicatable **sources** (e.g. native zarr chunks),
//!   - a single **extract** closure that assembles the 64³ output once all
//!     sources have been fetched.
//!
//! The cache executor pulls source fetches into a shared, deduping pool so
//! that one native chunk requested by many cache chunks is fetched once. Each
//! source carries its own dedup key (typically a URL or canonical path).

use super::state::ChunkKey;
use std::any::Any;
use std::sync::Arc;

#[derive(Debug, Clone)]
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

/// Type-erased payload carried by a source. The backfiller downcasts in its
/// extract closure. Arc so that 64 sibling cache chunks can share one copy of
/// a native zarr chunk without cloning the bytes.
pub type SourcePayload = Arc<dyn Any + Send + Sync>;

/// Outcome of one source fetch.
///   - `Ok(Some(payload))`: the source loaded successfully.
///   - `Ok(None)`: the source is definitively absent (e.g. 404 on a sparse
///     zarr) — extract should fill with zeros.
///   - `Err(e)`: transient/permanent failure — extract typically propagates.
pub type SourceOutcome = Result<Option<SourcePayload>, BackfillError>;

/// One fetchable input artifact.
pub struct SourceSpec {
    /// Globally unique identifier (URL, canonical path, …). Two SourceSpecs
    /// with the same key are deduplicated: the first fetch runs, later ones
    /// attach as waiters.
    pub key: String,
    /// Synchronous fetch. Called at most once per `key` across the lifetime
    /// of the cache (until evicted). Runs on the cache's worker pool, so it
    /// may block on I/O.
    pub fetch: Box<dyn FnOnce() -> SourceOutcome + Send + 'static>,
}

/// Plan for filling one 64³ cache chunk.
pub struct BackfillPlan {
    pub sources: Vec<SourceSpec>,
    /// Run once every source in `sources` has resolved. Receives outcomes
    /// in the same order as `sources`. Returns exactly `CHUNK_VOXELS` bytes.
    pub extract: Box<dyn FnOnce(&[SourceOutcome]) -> Result<Vec<u8>, BackfillError> + Send + 'static>,
}

pub trait ChunkBackfiller: Send + Sync {
    /// Highest LOD level the backfiller can serve (inclusive). LOD 0 must
    /// always be supported.
    fn max_lod(&self) -> u8;

    /// Volume extent in voxels at LOD 0, used to detect out-of-bounds reads
    /// without round-tripping through the backfiller.
    fn voxel_extent(&self) -> [u32; 3];

    /// Plan how to fill the chunk. Fast and cheap — only computes coordinate
    /// math and constructs closures. Actual I/O happens inside source fetches.
    fn plan(&self, key: ChunkKey) -> Result<BackfillPlan, BackfillError>;

    /// Stable identifier used for the on-disk cache directory. Different
    /// volumes must produce different ids; the same volume must produce the
    /// same id across runs.
    fn volume_id(&self) -> String;
}
