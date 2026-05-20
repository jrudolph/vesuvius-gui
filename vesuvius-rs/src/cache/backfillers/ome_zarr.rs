//! `OmeZarrBackfiller` — fetch unified-cache chunks from an OME-Zarr
//! multiscale volume.
//!
//! Each multiscale level becomes one LOD of the cache. To fill one 64³ unified
//! chunk we:
//!   1. compute the world voxel range it covers at this LOD,
//!   2. ask the underlying `ZarrArray` for each native zarr chunk that
//!      intersects that range (typically 1, 2, 4, or 8 native chunks for
//!      common 128³/256³ chunk shapes against our 64³),
//!   3. copy intersecting voxels from each native chunk into the output.
//!
//! We deliberately don't go through `ZarrContext::get` voxel-by-voxel — that
//! would do 262 144 lookups per fetch and serialize all workers on the
//! context's `RefCell` fast-path. Holding `ZarrArray` directly lets multiple
//! workers fetch the same LOD in parallel.
//!
//! Within one fetch, an intra-call one-slot cache for the most recently
//! loaded native chunk handles the common case where adjacent rows in the
//! output land in the same source chunk.

use crate::cache::backfiller::{BackfillError, ChunkBackfiller};
use crate::cache::state::ChunkKey;
use crate::cache::{CHUNK_SIDE, CHUNK_VOXELS};
use std::sync::Arc;
use vesuvius_zarr::{ChunkContext, OmeZarrContext, ZarrArray};

pub struct OmeZarrBackfiller {
    volume_id: String,
    extent_xyz: [u32; 3],
    /// One zarr array per multiscale level. `ZarrArray` is `Send + Sync` and
    /// stateless across calls, so workers can fetch from the same array in
    /// parallel — no mutex.
    arrays: Vec<ZarrArray<3, u8>>,
}

impl OmeZarrBackfiller {
    pub fn from_ome(volume_id: impl Into<String>, ome: OmeZarrContext) -> Self {
        // OME shape is ZYX; expose XYZ to the cache trait surface.
        let shape0 = ome
            .zarr_contexts
            .first()
            .map(|c| c.shape().to_vec())
            .unwrap_or_else(|| vec![0, 0, 0]);
        let extent_xyz = [
            *shape0.get(2).unwrap_or(&0) as u32,
            *shape0.get(1).unwrap_or(&0) as u32,
            *shape0.first().unwrap_or(&0) as u32,
        ];
        let arrays = ome.zarr_contexts.iter().map(|ctx| ctx.array().clone()).collect();
        Self {
            volume_id: volume_id.into(),
            extent_xyz,
            arrays,
        }
    }
}

impl ChunkBackfiller for OmeZarrBackfiller {
    fn max_lod(&self) -> u8 {
        self.arrays.len().saturating_sub(1) as u8
    }

    fn voxel_extent(&self) -> [u32; 3] {
        self.extent_xyz
    }

    fn fetch(&self, key: ChunkKey) -> Result<Vec<u8>, BackfillError> {
        let started = std::time::Instant::now();
        let lod = key.lod as usize;
        let array = self.arrays.get(lod).ok_or(BackfillError::OutOfBounds)?;
        let def = array.def();
        // Native shape and chunk shape are stored ZYX.
        let shape = &def.shape;
        let nchunk = &def.chunks;
        if shape.len() != 3 || nchunk.len() != 3 {
            return Err(BackfillError::Permanent(format!(
                "expected 3D zarr at lod {}, got shape={:?} chunks={:?}",
                lod, shape, nchunk
            )));
        }

        // Sample range covered by this unified chunk in the LOD-N sample
        // space (which is what the zarr array is indexed in).
        let base_x = key.x as usize * CHUNK_SIDE;
        let base_y = key.y as usize * CHUNK_SIDE;
        let base_z = key.z as usize * CHUNK_SIDE;
        let end_x = (base_x + CHUNK_SIDE).min(shape[2]);
        let end_y = (base_y + CHUNK_SIDE).min(shape[1]);
        let end_z = (base_z + CHUNK_SIDE).min(shape[0]);
        if base_x >= shape[2] || base_y >= shape[1] || base_z >= shape[0] {
            // Pre-checked at dispatch time, but cheap to be defensive.
            return Err(BackfillError::OutOfBounds);
        }

        // Range of native chunks intersecting that sample range.
        let cx_lo = base_x / nchunk[2];
        let cx_hi = (end_x - 1) / nchunk[2];
        let cy_lo = base_y / nchunk[1];
        let cy_hi = (end_y - 1) / nchunk[1];
        let cz_lo = base_z / nchunk[0];
        let cz_hi = (end_z - 1) / nchunk[0];

        let definitive = array.cache_missing();
        log::debug!(
            "ome-zarr backfill start: lod={} key=({},{},{}) native_chunks={:?} definitive_missing={}",
            key.lod,
            key.x,
            key.y,
            key.z,
            [cz_hi - cz_lo + 1, cy_hi - cy_lo + 1, cx_hi - cx_lo + 1],
            definitive
        );
        let mut out = vec![0u8; CHUNK_VOXELS];
        let mut native_loaded = 0usize;
        let mut native_missing = 0usize;
        // Intra-call one-slot cache for the last native chunk we loaded.
        // Adjacent (cz, cy, cx) values in our nested loop tend to repeat
        // along the inner axes; one slot is enough.
        let mut last: Option<([usize; 3], Arc<ChunkContext>)> = None;

        for cz in cz_lo..=cz_hi {
            // Z-range of this native chunk in sample space.
            let nz_lo = (cz * nchunk[0]).max(base_z);
            let nz_hi = ((cz + 1) * nchunk[0]).min(end_z);
            for cy in cy_lo..=cy_hi {
                let ny_lo = (cy * nchunk[1]).max(base_y);
                let ny_hi = ((cy + 1) * nchunk[1]).min(end_y);
                for cx in cx_lo..=cx_hi {
                    let nx_lo = (cx * nchunk[2]).max(base_x);
                    let nx_hi = ((cx + 1) * nchunk[2]).min(end_x);

                    let coord = [cz, cy, cx];
                    let chunk = if let Some((k, ref c)) = last {
                        if k == coord {
                            Some(c.clone())
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    let chunk = match chunk {
                        Some(c) => c,
                        None => match array.load_chunk(coord) {
                            Some(c) => {
                                native_loaded += 1;
                                let arc = Arc::new(c);
                                last = Some((coord, arc.clone()));
                                arc
                            }
                            None => {
                                // None has two meanings depending on access:
                                // - `cache_missing() = true`: the source has
                                //   confirmed this native chunk doesn't
                                //   exist. Leave the region as zero (fill
                                //   value) and continue.
                                // - `cache_missing() = false`: async fetch
                                //   may still be in flight. We bail with a
                                //   Transient error so the cache retries on
                                //   cooldown instead of caching zeros
                                //   permanently.
                                native_missing += 1;
                                last = None;
                                if !definitive {
                                    log::debug!(
                                        "ome-zarr backfiller: lod={} key=({},{},{}) native chunk {:?} not yet available, retrying",
                                        key.lod, key.x, key.y, key.z, coord
                                    );
                                    return Err(BackfillError::Transient(format!(
                                        "native chunk {:?} not yet available",
                                        coord
                                    )));
                                }
                                continue;
                            }
                        },
                    };

                    let chunk_base_z = cz * nchunk[0];
                    let chunk_base_y = cy * nchunk[1];
                    let chunk_base_x = cx * nchunk[2];
                    let stride_y = nchunk[2];
                    let stride_z = nchunk[1] * nchunk[2];

                    // Copy the intersection.
                    for sz in nz_lo..nz_hi {
                        let lz = sz - base_z; // 0..CHUNK_SIDE
                        let in_z = (sz - chunk_base_z) * stride_z;
                        let out_z = lz * CHUNK_SIDE * CHUNK_SIDE;
                        for sy in ny_lo..ny_hi {
                            let ly = sy - base_y;
                            let in_y = in_z + (sy - chunk_base_y) * stride_y;
                            let out_y = out_z + ly * CHUNK_SIDE;
                            // Tight inner copy over x.
                            for sx in nx_lo..nx_hi {
                                let lx = sx - base_x;
                                let in_idx = in_y + (sx - chunk_base_x);
                                out[out_y + lx] = chunk.get(in_idx);
                            }
                        }
                    }
                }
            }
        }
        log::info!(
            "ome-zarr backfill done: lod={} key=({},{},{}) loaded={} missing={} elapsed={:?}",
            key.lod,
            key.x,
            key.y,
            key.z,
            native_loaded,
            native_missing,
            started.elapsed()
        );
        Ok(out)
    }

    fn volume_id(&self) -> String {
        self.volume_id.clone()
    }
}
