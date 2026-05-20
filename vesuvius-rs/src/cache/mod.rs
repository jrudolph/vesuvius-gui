//! Unified chunk cache.
//!
//! A single in-memory + on-disk LRU-ish cache of fixed-size 64³ raw u8 chunks,
//! shared by every volume backend. Backends become `ChunkBackfiller`s that
//! produce chunks on demand; the cache owns all the state (mmap, fetch
//! dispatch, hot-slot, …).
//!
//! Design synthesizes two existing patterns in this tree:
//! - `volume::volume64x4::TileCache` for the in-memory state machine
//!   (`Missing | Pending | Resident | CooldownMiss`) and the thread-local
//!   hot-chunk shortcut.
//! - `vesuvius-zarr::v3::DecodedCache` for atomic temp+rename writes and
//!   mmap reload of decoded chunks.

mod backfiller;
mod cache;
mod disk;
mod downloader;
mod sidecar;
mod spill;
mod state;
mod volume;

pub mod backfillers;

#[cfg(test)]
mod tests;

pub use backfiller::{BackfillError, BackfillPlan, ChunkBackfiller, SourceOutcome, SourcePayload, SourceSpec};
pub use cache::ChunkCache;
pub use downloader::{DownloadError, Downloader};
pub use state::{ChunkKey, ChunkState};
pub use volume::UnifiedVolume;

/// Side length (in voxels) of one cache chunk. Fixed for now; matches the
/// natural granularity of the existing paint loop in `volume64x4.rs`.
pub const CHUNK_SIDE: usize = 64;
pub const CHUNK_VOXELS: usize = CHUNK_SIDE * CHUNK_SIDE * CHUNK_SIDE;

/// Queued work older than this is treated as stale and dropped at pop time.
/// Shared between the cache's TaskQueue and the Downloader queue.
pub(crate) const MAX_AGE: std::time::Duration = std::time::Duration::from_secs(10);
