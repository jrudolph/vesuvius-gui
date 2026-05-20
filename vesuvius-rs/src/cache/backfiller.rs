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

/// One fetchable input artifact. Two specs with the same key are
/// deduplicated by the cache: the first observer's fetch (or download)
/// runs, later observers attach as waiters and share the outcome.
///
/// `Download` is the async path: the cache hands the URL to its central
/// downloader, gets bytes back without ever blocking a cache worker, and
/// surfaces the raw bytes as the source payload. The expensive decode
/// (blosc/zstd/c3d) belongs to the backfiller's `extract` closure, which
/// runs on the cache worker pool — i.e. on CPU-sized concurrency, not
/// I/O-sized.
pub enum SourceSpec {
    Compute {
        key: String,
        /// Synchronous fetch. Runs on the cache's worker pool. Use this for
        /// in-process or local-disk sources where there's no benefit to
        /// async dispatch.
        fetch: Box<dyn FnOnce() -> SourceOutcome + Send + 'static>,
    },
    Download {
        key: String,
        /// HTTP URL. The cache submits this to its centralized downloader,
        /// which delivers the raw bytes as the source payload
        /// (`Arc<Vec<u8>>`). No decode happens here — the backfiller's
        /// `extract` closure receives the raw bytes and is responsible for
        /// any decompression.
        url: String,
    },
    /// Depend on another cache chunk. The cache dispatches the child chunk
    /// (no worker thread blocks waiting on it) and, when the child becomes
    /// Resident, satisfies this source with the child's `Arc<ChunkState>`
    /// as payload — extract closures downcast and call `as_resident()` to
    /// access the child's mmap'd bytes. If the child enters a permanent /
    /// cooldown state instead, the source resolves with `Ok(None)`.
    ///
    /// Used by `SynthesizedLodBackfiller` to express "this synthesized LOD
    /// chunk is built from these 8 children at the next-finer LOD."
    Chunk {
        key: String,
        chunk_key: super::state::ChunkKey,
    },
}

impl SourceSpec {
    pub fn key(&self) -> &str {
        match self {
            SourceSpec::Compute { key, .. } => key,
            SourceSpec::Download { key, .. } => key,
            SourceSpec::Chunk { key, .. } => key,
        }
    }
}

/// Outcome of materializing one cache chunk inside an `extract` call.
#[derive(Debug)]
pub enum ExtractedChunk {
    /// Bytes for the chunk — exactly `CHUNK_VOXELS`. Cache writes them via
    /// `write_atomic` and (for the chunk that triggered the extract) mmaps
    /// for in-memory residency.
    Bytes(Vec<u8>),
    /// Chunk is definitively absent — cache writes a `.empty` sentinel and
    /// (for the chunk that triggered the extract) transitions to
    /// `ChunkState::Empty`. Use this when the backfiller can determine,
    /// given the resolved sources, that no data exists for this chunk at
    /// this LOD.
    Empty,
}

/// Plan for filling one 64³ cache chunk.
pub struct BackfillPlan {
    pub sources: Vec<SourceSpec>,
    /// Run once every source in `sources` has resolved. Receives outcomes
    /// in the same order as `sources`. Returns one or more `(ChunkKey,
    /// ExtractedChunk)` entries.
    ///
    /// One entry MUST match the `ChunkKey` that the cache passed to `plan`
    /// — that entry is published into the in-memory cache map as
    /// `Resident` / `Empty`. Any additional entries are written to disk as
    /// a byproduct, so that future dispatches for those keys skip the
    /// download + decode entirely. They do NOT touch the in-memory map.
    ///
    /// The flat shape lets backfillers express "I have full data for these
    /// 8 cache chunks because I downloaded one 128³ native chunk that
    /// covers them" without a separate primary/siblings split.
    pub extract:
        Box<dyn FnOnce(&[SourceOutcome]) -> Result<Vec<(ChunkKey, ExtractedChunk)>, BackfillError> + Send + 'static>,
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
