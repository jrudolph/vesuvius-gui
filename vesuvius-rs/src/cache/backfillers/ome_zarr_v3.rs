//! V3-sharded-c3d specialization for `OmeZarrBackfiller`.
//!
//! V3 sharded c3d arrays publish a single shard file per group of (typically
//! 2×2×2 = 8) 256³ c3d sub-chunks. The shard's first bytes are a small
//! index of `(offset, len)` pairs telling you where each sub-chunk lives
//! inside the shard. The default OME-Zarr path doesn't model this — it
//! falls through to `array.load_chunk(coord)` (a `Compute` source) which
//! does the index fetch + Range request synchronously on a cache worker
//! and bypasses the central downloader pool.
//!
//! This module replaces that path for remote v3 arrays:
//!
//!   1. Iterate the 256³ sub-chunks overlapping the 64³ cache chunk.
//!   2. For each, look up its shard, fetch the shard index *blockingly*
//!      (single-flight inside the V3 access handle so concurrent
//!      `plan()` calls coalesce), read the sub-chunk's `(offset, len)`.
//!   3. Emit one `SourceSpec::Download { url, range }` per non-empty
//!      sub-chunk. Sentinel index entries and 404-on-shard-index get
//!      no source — the extract closure treats their region as zero.
//!   4. Extract: c3d-decode each received sub-chunk and slice it into
//!      the 64³ cache chunk (primary + siblings, same shape as the v2
//!      extract in `ome_zarr.rs`).
//!
//! Source payload is the same as `Download` produces elsewhere:
//! `Arc<Mmap>` (spilled to disk by the cache's on_done) with a
//! fallback to `Arc<Vec<u8>>` if the spill write failed.

use crate::cache::backfiller::{
    BackfillError, BackfillPlan, ExtractedChunk, SourceOutcome, SourcePayload, SourceSpec,
};
use crate::cache::state::ChunkKey;
use crate::cache::{CHUNK_SIDE, CHUNK_VOXELS};
use memmap::Mmap;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use vesuvius_zarr::v3::V3RemoteShardedAccess;

/// Plan one 64³ cache chunk against a remote v3 sharded c3d array.
///
/// `volume_id` and `lod` only flow through into source keys so two cache
/// chunks needing the same sub-chunk dedup at the cache's source map.
pub(super) fn plan_v3_chunk(
    handle: &Arc<dyn V3RemoteShardedAccess>,
    volume_id: &str,
    key: ChunkKey,
    lod: usize,
) -> Result<BackfillPlan, BackfillError> {
    let shape: [usize; 3] = match handle.shape().try_into() {
        Ok(s) => s,
        Err(_) => {
            return Err(BackfillError::Permanent(format!(
                "v3 array shape is not 3D: {:?}",
                handle.shape()
            )));
        }
    };
    let sub: [usize; 3] = handle.sub_chunk_shape();
    let per_shard: [usize; 3] = handle.sub_chunks_per_shard();

    // Voxel extent of this cache chunk inside the array.
    let base_x = key.x as usize * CHUNK_SIDE;
    let base_y = key.y as usize * CHUNK_SIDE;
    let base_z = key.z as usize * CHUNK_SIDE;
    if base_x >= shape[2] || base_y >= shape[1] || base_z >= shape[0] {
        return Err(BackfillError::OutOfBounds);
    }
    let end_x = (base_x + CHUNK_SIDE).min(shape[2]);
    let end_y = (base_y + CHUNK_SIDE).min(shape[1]);
    let end_z = (base_z + CHUNK_SIDE).min(shape[0]);

    // Sub-chunk coordinate range (z, y, x) overlapping this cache chunk.
    let cz_lo = base_z / sub[0];
    let cz_hi = (end_z - 1) / sub[0];
    let cy_lo = base_y / sub[1];
    let cy_hi = (end_y - 1) / sub[1];
    let cx_lo = base_x / sub[2];
    let cx_hi = (end_x - 1) / sub[2];

    // Build the source list. Two parallel lists are needed:
    //   - `all_coords`: every sub-chunk we considered. Used for the bbox
    //     enumeration (so siblings know "the source set covers me") and
    //     for the "all overlaps in source set" coverage check.
    //   - `source_coords`: only sub-chunks that produced a Download.
    //     Zips 1:1 with `outcomes` in extract.
    //
    // Sub-chunks skipped because their shard is absent or the index entry
    // is a sentinel are in `all_coords` but not in `source_coords`. They
    // resolve to zero-filled regions in the output.
    let mut sources: Vec<SourceSpec> = Vec::new();
    let mut source_coords: Vec<[usize; 3]> = Vec::new();
    let mut all_coords: Vec<[usize; 3]> = Vec::new();
    // Cache shard indices we've already fetched within this `plan` call to
    // avoid repeating the (already-cached) hashmap lookup. Hot when one
    // cache chunk overlaps multiple sub-chunks of the same shard, which is
    // typical (cache chunk = 64³, shard = 512³).
    let mut shard_index_cache: HashMap<[usize; 3], Option<Arc<vesuvius_zarr::sharding::ShardIndex>>> =
        HashMap::new();

    for cz in cz_lo..=cz_hi {
        for cy in cy_lo..=cy_hi {
            for cx in cx_lo..=cx_hi {
                let coord = [cz, cy, cx];
                all_coords.push(coord);

                let shard = [cz / per_shard[0], cy / per_shard[1], cx / per_shard[2]];
                let sub_in_shard = [cz % per_shard[0], cy % per_shard[1], cx % per_shard[2]];
                let flat = (sub_in_shard[0] * per_shard[1] + sub_in_shard[1]) * per_shard[2] + sub_in_shard[2];

                // Resolve shard index (single-flight inside the handle).
                let idx_opt = match shard_index_cache.get(&shard) {
                    Some(v) => v.clone(),
                    None => {
                        let v = handle.shard_index(shard);
                        shard_index_cache.insert(shard, v.clone());
                        v
                    }
                };
                let Some(idx) = idx_opt else {
                    // Whole shard absent — no source for this sub-chunk.
                    continue;
                };
                let Some((off, len)) = idx.lookup(flat) else {
                    // Sentinel entry — sub-chunk absent.
                    continue;
                };

                let url = handle.shard_url(shard);
                let source_key = format!(
                    "{}/L{:02}/c/{}_{}_{}/sub_{:05}",
                    volume_id, lod, shard[0], shard[1], shard[2], flat
                );
                sources.push(SourceSpec::Download {
                    key: source_key,
                    url,
                    range: Some((off, len)),
                });
                source_coords.push(coord);
            }
        }
    }

    log::trace!(
        "[{}] v3 plan → {} download(s) over {} considered sub-chunk(s)",
        key,
        sources.len(),
        all_coords.len()
    );

    let key_dbg = key;
    let extract = Box::new(
        move |outcomes: &[SourceOutcome]| -> Result<Vec<(ChunkKey, ExtractedChunk)>, BackfillError> {
            let started = std::time::Instant::now();

            // Bail fast on errors. Pick worst severity (same convention as
            // OmeZarrBackfiller v2 path).
            let mut transient: Option<String> = None;
            let mut permanent: Option<String> = None;
            for o in outcomes {
                if let Err(e) = o {
                    match e {
                        BackfillError::OutOfBounds => permanent = Some("oob source".into()),
                        BackfillError::Permanent(s) => permanent = Some(s.clone()),
                        BackfillError::Transient(s) => {
                            if transient.is_none() {
                                transient = Some(s.clone());
                            }
                        }
                    }
                }
            }
            if let Some(s) = permanent {
                return Err(BackfillError::Permanent(s));
            }
            if let Some(s) = transient {
                return Err(BackfillError::Transient(s));
            }

            debug_assert_eq!(
                source_coords.len(),
                outcomes.len(),
                "v3 extract: source_coords must zip 1:1 with outcomes"
            );

            // Decode each present sub-chunk into a 256³ heap buffer keyed
            // by sub-chunk coordinate. Absent (Ok(None)) and skipped
            // sub-chunks stay out of the map → extract zero-fills.
            let mut decoded: HashMap<[usize; 3], Arc<Vec<u8>>> = HashMap::new();
            for (i, coord) in source_coords.iter().enumerate() {
                let payload_opt: &Option<SourcePayload> = match &outcomes[i] {
                    Ok(p) => p,
                    Err(_) => unreachable!("error already short-circuited"),
                };
                let Some(payload) = payload_opt else { continue };
                let decoded_bytes = if let Ok(mmap_arc) = payload.clone().downcast::<Mmap>() {
                    vesuvius_c3d::with_decoder(|d| d.decode(&mmap_arc[..]))
                        .map_err(|e| BackfillError::Permanent(format!("c3d decode at {:?}: {}", coord, e)))?
                } else if let Ok(bytes_arc) = payload.clone().downcast::<Vec<u8>>() {
                    vesuvius_c3d::with_decoder(|d| d.decode(&bytes_arc))
                        .map_err(|e| BackfillError::Permanent(format!("c3d decode at {:?}: {}", coord, e)))?
                } else {
                    return Err(BackfillError::Permanent("source payload type".into()));
                };
                decoded.insert(*coord, Arc::new(decoded_bytes));
            }

            // Bounding box (in cache-chunk coords) of cache chunks that the
            // source set could cover. Same algorithm as the v2 extract.
            let mut bbox_lo = [usize::MAX; 3];
            let mut bbox_hi = [0usize; 3];
            for coord in all_coords.iter() {
                let voxel_lo = [coord[0] * sub[0], coord[1] * sub[1], coord[2] * sub[2]];
                let voxel_hi = [
                    ((coord[0] + 1) * sub[0]).min(shape[0]),
                    ((coord[1] + 1) * sub[1]).min(shape[1]),
                    ((coord[2] + 1) * sub[2]).min(shape[2]),
                ];
                let cache_lo = [
                    voxel_lo[0] / CHUNK_SIDE,
                    voxel_lo[1] / CHUNK_SIDE,
                    voxel_lo[2] / CHUNK_SIDE,
                ];
                let cache_hi = [
                    voxel_hi[0].div_ceil(CHUNK_SIDE),
                    voxel_hi[1].div_ceil(CHUNK_SIDE),
                    voxel_hi[2].div_ceil(CHUNK_SIDE),
                ];
                for i in 0..3 {
                    bbox_lo[i] = bbox_lo[i].min(cache_lo[i]);
                    bbox_hi[i] = bbox_hi[i].max(cache_hi[i]);
                }
            }

            let coord_set: HashSet<[usize; 3]> = all_coords.iter().copied().collect();

            let mut output: Vec<(ChunkKey, ExtractedChunk)> = Vec::new();
            let mut primary_found = false;
            let mut empty_count = 0usize;
            let mut bytes_count = 0usize;

            let stride_y = sub[2];
            let stride_z = sub[1] * sub[2];

            for kz in bbox_lo[0]..bbox_hi[0] {
                for ky in bbox_lo[1]..bbox_hi[1] {
                    for kx in bbox_lo[2]..bbox_hi[2] {
                        let cv_base = [kz * CHUNK_SIDE, ky * CHUNK_SIDE, kx * CHUNK_SIDE];
                        if cv_base[0] >= shape[0] || cv_base[1] >= shape[1] || cv_base[2] >= shape[2] {
                            continue;
                        }
                        let cv_end = [
                            (cv_base[0] + CHUNK_SIDE).min(shape[0]),
                            (cv_base[1] + CHUNK_SIDE).min(shape[1]),
                            (cv_base[2] + CHUNK_SIDE).min(shape[2]),
                        ];

                        // Sub-chunks overlapping this cache chunk.
                        let ncz_lo = cv_base[0] / sub[0];
                        let ncz_hi = (cv_end[0] - 1) / sub[0];
                        let ncy_lo = cv_base[1] / sub[1];
                        let ncy_hi = (cv_end[1] - 1) / sub[1];
                        let ncx_lo = cv_base[2] / sub[2];
                        let ncx_hi = (cv_end[2] - 1) / sub[2];

                        // All overlapping sub-chunks must be in our coord
                        // universe (either present-with-data or skipped-
                        // because-absent). Anything outside the coord set
                        // means we don't have full coverage from this plan.
                        let mut covered = true;
                        let mut all_absent = true;
                        'check: for ncz in ncz_lo..=ncz_hi {
                            for ncy in ncy_lo..=ncy_hi {
                                for ncx in ncx_lo..=ncx_hi {
                                    let coord = [ncz, ncy, ncx];
                                    if !coord_set.contains(&coord) {
                                        covered = false;
                                        break 'check;
                                    }
                                    if decoded.contains_key(&coord) {
                                        all_absent = false;
                                    }
                                }
                            }
                        }
                        if !covered {
                            continue;
                        }

                        let chunk_key = ChunkKey::new(key.lod, kx as u32, ky as u32, kz as u32);
                        if chunk_key == key {
                            primary_found = true;
                        }

                        if all_absent {
                            output.push((chunk_key, ExtractedChunk::Empty));
                            empty_count += 1;
                            continue;
                        }

                        let mut out = vec![0u8; CHUNK_VOXELS];
                        for ncz in ncz_lo..=ncz_hi {
                            for ncy in ncy_lo..=ncy_hi {
                                for ncx in ncx_lo..=ncx_hi {
                                    let coord = [ncz, ncy, ncx];
                                    let Some(buf) = decoded.get(&coord) else {
                                        continue;
                                    };
                                    let chunk_base_z = ncz * sub[0];
                                    let chunk_base_y = ncy * sub[1];
                                    let chunk_base_x = ncx * sub[2];
                                    let nz_lo = chunk_base_z.max(cv_base[0]);
                                    let nz_hi = (chunk_base_z + sub[0]).min(cv_end[0]);
                                    let ny_lo = chunk_base_y.max(cv_base[1]);
                                    let ny_hi = (chunk_base_y + sub[1]).min(cv_end[1]);
                                    let nx_lo = chunk_base_x.max(cv_base[2]);
                                    let nx_hi = (chunk_base_x + sub[2]).min(cv_end[2]);
                                    for sz in nz_lo..nz_hi {
                                        let lz = sz - cv_base[0];
                                        let in_z = (sz - chunk_base_z) * stride_z;
                                        let out_z = lz * CHUNK_SIDE * CHUNK_SIDE;
                                        for sy in ny_lo..ny_hi {
                                            let ly = sy - cv_base[1];
                                            let in_y = in_z + (sy - chunk_base_y) * stride_y;
                                            let out_y = out_z + ly * CHUNK_SIDE;
                                            for sx in nx_lo..nx_hi {
                                                let lx = sx - cv_base[2];
                                                let in_idx = in_y + (sx - chunk_base_x);
                                                out[out_y + lx] = buf[in_idx];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        output.push((chunk_key, ExtractedChunk::Bytes(out)));
                        bytes_count += 1;
                    }
                }
            }

            if !primary_found {
                log::warn!("[{}] v3 sibling enumeration missed the primary key", key_dbg);
            }

            log::trace!(
                "[{}] v3 extract: {} bytes + {} empty ({:?})",
                key_dbg,
                bytes_count,
                empty_count,
                started.elapsed()
            );
            Ok(output)
        },
    );

    Ok(BackfillPlan { sources, extract })
}
