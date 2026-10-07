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
pub mod lease;
mod netlog;
pub mod paint_scope;
mod s3_auth;
pub mod epoch;
pub mod purge;
mod sidecar;
mod spill;
mod state;
mod volume;
mod work_queue;

pub mod backfillers;

#[cfg(test)]
mod tests;

pub use backfiller::{BackfillError, BackfillPlan, ChunkBackfiller, SourceOutcome, SourcePayload, SourceSpec};
pub use cache::{configured_workers, ChunkCache, UnifiedCache};
pub use disk::{DispatchedBits, ShardCoord, ShardSnapshot};
pub use downloader::{DownloadError, Downloader, DownloaderStats};
pub use state::{ChunkKey, ChunkState};
pub use volume::UnifiedVolume;

/// Side length (in voxels) of one cache chunk. Fixed for now; matches the
/// natural granularity of the existing paint loop in `volume64x4.rs`.
pub const CHUNK_SIDE: usize = 64;
pub const CHUNK_VOXELS: usize = CHUNK_SIDE * CHUNK_SIDE * CHUNK_SIDE;

