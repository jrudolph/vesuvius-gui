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
        /// Optional byte range `(offset, len)` to request. When `Some`, the
        /// downloader issues a `Range: bytes=offset-(offset+len-1)` request
        /// and treats `206 Partial Content` as success. Used by sharded
        /// formats (e.g. zarr v3 c3d) where one URL backs many sub-chunks
        /// addressed by offset+length pairs derived from a shard index.
        range: Option<(u64, u64)>,
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
    /// Every cache chunk this plan will fill. MUST contain the primary —
    /// the same `ChunkKey` the cache passed to `plan` — plus any siblings
    /// the same extract will materialize as a byproduct (typically
    /// because one downloaded native chunk covers them all). Order is
    /// not significant.
    ///
    /// The cache claims every sibling as `ChunkState::Pending` the moment
    /// dispatch returns, so a viewport that fans out into 8 sibling cache
    /// chunks sharing one native source doesn't trigger 8 redundant plans +
    /// extracts: only the first observer plans, and the others see Pending
    /// immediately. When extract finishes, every covered key transitions
    /// straight to `Resident` / `Empty` in-memory — no need for a follow-up
    /// dispatch + disk reload.
    ///
    /// Plans whose extract only fills the primary (e.g. synthesized-LOD)
    /// should set `covered = vec![primary]`.
    pub covered: Vec<ChunkKey>,
    pub sources: Vec<SourceSpec>,
    /// Run once every source in `sources` has resolved. Receives outcomes
    /// in the same order as `sources`. Returns one or more `(ChunkKey,
    /// ExtractedChunk)` entries.
    ///
    /// Every entry's key SHOULD appear in `covered`; any entry whose key
    /// is not in `covered` is still persisted to disk but won't be promoted
    /// to in-memory `Resident`. Conversely, any `covered` key not present in
    /// the extract output is treated as transient (its Pending claim is
    /// cleared so a later dispatch can retry).
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
