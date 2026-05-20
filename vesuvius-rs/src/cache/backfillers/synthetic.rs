//! Backfiller that builds chunks from a user closure. Useful for tests and
//! for driving the cache against an in-memory procedural volume before
//! plugging in real backends.

use crate::cache::backfiller::{BackfillError, ChunkBackfiller};
use crate::cache::state::ChunkKey;
use crate::cache::{CHUNK_SIDE, CHUNK_VOXELS};
use std::sync::Arc;

pub struct SyntheticBackfiller {
    volume_id: String,
    extent: [u32; 3],
    max_lod: u8,
    f: Arc<dyn Fn(u32, u32, u32, u8) -> u8 + Send + Sync>,
}

impl SyntheticBackfiller {
    /// `f(x, y, z, lod) -> u8` is invoked per sample inside each requested
    /// chunk. Coordinates are sample coordinates at the requested LOD.
    pub fn new<F>(volume_id: impl Into<String>, extent: [u32; 3], max_lod: u8, f: F) -> Self
    where
        F: Fn(u32, u32, u32, u8) -> u8 + Send + Sync + 'static,
    {
        Self {
            volume_id: volume_id.into(),
            extent,
            max_lod,
            f: Arc::new(f),
        }
    }
}

impl ChunkBackfiller for SyntheticBackfiller {
    fn max_lod(&self) -> u8 {
        self.max_lod
    }

    fn voxel_extent(&self) -> [u32; 3] {
        self.extent
    }

    fn fetch(&self, key: ChunkKey) -> Result<Vec<u8>, BackfillError> {
        let mut out = vec![0u8; CHUNK_VOXELS];
        for z in 0..CHUNK_SIDE {
            for y in 0..CHUNK_SIDE {
                for x in 0..CHUNK_SIDE {
                    let sx = key.x * CHUNK_SIDE as u32 + x as u32;
                    let sy = key.y * CHUNK_SIDE as u32 + y as u32;
                    let sz = key.z * CHUNK_SIDE as u32 + z as u32;
                    let off = z * CHUNK_SIDE * CHUNK_SIDE + y * CHUNK_SIDE + x;
                    out[off] = (self.f)(sx, sy, sz, key.lod);
                }
            }
        }
        Ok(out)
    }

    fn volume_id(&self) -> String {
        self.volume_id.clone()
    }
}
