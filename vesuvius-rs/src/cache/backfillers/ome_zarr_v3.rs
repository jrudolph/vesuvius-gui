//! V3-sharded specialization for `OmeZarrBackfiller` (vc-delta3d sub-chunks).
//!
//! V3 sharded arrays publish a single shard file per group of sub-chunks
//! (e.g. 2×2×2 = 8 256³ sub-chunks). Each shard carries a small
//! index of `(offset, len)` pairs (at its head or tail) telling you where
//! each sub-chunk lives inside the shard. The default OME-Zarr path doesn't model this — it
//! falls through to `array.load_chunk(coord)` (a `Compute` source) which
//! does the index fetch + Range request synchronously on a cache worker
//! and bypasses the central downloader pool.
//!
//! This module replaces that path for remote v3 arrays:
//!
//!   1. Iterate the sub-chunks overlapping the 64³ cache chunk.
//!   2. For each, look up its shard, fetch the shard index *blockingly*
//!      (single-flight inside the V3 access handle so concurrent
//!      `plan()` calls coalesce), read the sub-chunk's `(offset, len)`.
//!   3. Emit one `SourceSpec::Download { url, range }` per non-empty
//!      sub-chunk. Sentinel index entries and 404-on-shard-index get
//!      no source — the extract closure treats their region as zero.
//!   4. Extract: decode the sub-chunk (memoized on the shared
//!      `LazySource` payload, so concurrent sibling extracts decode once)
//!      and slice out the requesting 64³ cache chunk. Coverage is lazy —
//!      only requested chunks are materialized; see the `covered`
//!      comment in `plan_v3_chunk`.
//!
//! Source payload is the same as `Download` produces elsewhere:
//! `Arc<LazySource>` over spilled (mmap-backed) bytes.

use crate::cache::backfiller::{
    BackfillError, BackfillPlan, ExtractedChunk, LazySource, SourceOutcome, SourcePayload, SourceSpec,
};
use crate::cache::state::ChunkKey;
use crate::cache::CHUNK_SIDE;
use std::collections::HashMap;
use std::sync::Arc;
use vesuvius_zarr::v3::V3RemoteShardedAccess;

/// Plan one 64³ cache chunk against a remote v3 sharded array.
///
/// `volume_id` and `lod` only flow through into source keys so two cache
/// chunks needing the same sub-chunk dedup at the cache's source map.
pub(super) fn plan_v3_chunk(
    handle: &Arc<dyn V3RemoteShardedAccess>,
    volume_id: &str,
    key: ChunkKey,
    lod: usize,
    eager: bool,
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

    // Build the source list. `source_coords` zips 1:1 with `outcomes` in
    // extract; it lists only sub-chunks that produced a Download.
    // Sub-chunks skipped because their shard is absent or the index entry
    // is a sentinel get no source — the whole dense sub-chunk box is
    // still considered covered (plan-time `covered_cache_chunks` below),
    // and their regions resolve to zero-fill / Empty in the output.
    let mut sources: Vec<SourceSpec> = Vec::new();
    let mut source_coords: Vec<[usize; 3]> = Vec::new();
    let mut n_considered = 0usize;
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
                n_considered += 1;

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

    // Lazy materialization: when there's something to download, claim only
    // the requesting chunk, so slab/slice browsing doesn't fill the decoded
    // tier with depth the viewport never visits. Siblings that ARE requested
    // register on the same source (deduped) and share the decode via
    // `LazySource`; the rest of a larger sub-chunk goes to the segment log
    // only (`log_box`), so a later request decodes it locally.
    //
    // The all-absent case keeps full-box coverage: zero-filling the whole
    // box costs no decode and saves ~63 future dispatches over empty
    // space. `eager` (offline renderer) also claims the full box: it
    // eventually reads the whole sub-chunk.
    let full_box =
        || super::ome_zarr::covered_cache_chunks([cz_lo, cy_lo, cx_lo], [cz_hi, cy_hi, cx_hi], &sub, &shape, key.lod);
    let covered = if sources.is_empty() || eager { full_box() } else { vec![key] };
    // Sub-chunks larger than a cache chunk (c3d 256³): the extract also
    // fills every other cache chunk of the box, so one decode puts all of
    // them into the segment log. The cache only makes the requested ones
    // resident; the rest are logged, ready for a cheap decode later.
    let log_box = if !sources.is_empty() && sub != [CHUNK_SIDE; 3] { full_box() } else { Vec::new() };

    log::trace!(
        "[{}] v3 plan → {} download(s) over {} considered sub-chunk(s), covers {} cache chunk(s)",
        key,
        sources.len(),
        n_considered,
        covered.len()
    );

    let key_dbg = key;
    let covered_for_extract = covered.clone();
    let decoder = handle.clone();
    let extract = Box::new(
        move |outcomes: &[SourceOutcome]| -> Result<Vec<(ChunkKey, ExtractedChunk)>, BackfillError> {
            super::ome_zarr::triage_outcomes(outcomes)?;

            debug_assert_eq!(
                source_coords.len(),
                outcomes.len(),
                "v3 extract: source_coords must zip 1:1 with outcomes"
            );

            // Decode each present sub-chunk into a heap buffer keyed
            // by sub-chunk coordinate. Absent (Ok(None)) and skipped
            // sub-chunks stay out of the map → extract zero-fills.
            //
            // The decode is memoized on the shared `LazySource` payload,
            // and this extract is the wave's batch leader (the cache
            // queues one extract per source completion): decode once,
            // then fill every requester chunk this plan's sub-chunk box
            // covers. Followers skip-met against the promoted states.
            let mut decoded: HashMap<[usize; 3], Arc<Vec<u8>>> = HashMap::new();
            let mut raws: HashMap<[usize; 3], Arc<LazySource>> = HashMap::new();
            let mut targets: Vec<ChunkKey> = covered_for_extract.clone();
            for (i, coord) in source_coords.iter().enumerate() {
                let payload_opt: &Option<SourcePayload> = match &outcomes[i] {
                    Ok(p) => p,
                    Err(_) => unreachable!("error already short-circuited"),
                };
                let Some(payload) = payload_opt else { continue };
                let lazy = payload
                    .clone()
                    .downcast::<LazySource>()
                    .map_err(|_| BackfillError::Permanent("source payload type".into()))?;
                for r in lazy.requesters() {
                    // Fillable iff the requester's full sub-chunk overlap
                    // set lies inside this plan's considered box — a
                    // requester spanning a sub-chunk we didn't plan for
                    // (multi-source chunk) must run its own extract.
                    if r.lod != key_dbg.lod || targets.contains(&r) {
                        continue;
                    }
                    let rb = [r.z as usize * CHUNK_SIDE, r.y as usize * CHUNK_SIDE, r.x as usize * CHUNK_SIDE];
                    if rb[0] >= shape[0] || rb[1] >= shape[1] || rb[2] >= shape[2] {
                        continue;
                    }
                    let re = [
                        (rb[0] + CHUNK_SIDE).min(shape[0]),
                        (rb[1] + CHUNK_SIDE).min(shape[1]),
                        (rb[2] + CHUNK_SIDE).min(shape[2]),
                    ];
                    let inside = rb[0] / sub[0] >= cz_lo
                        && (re[0] - 1) / sub[0] <= cz_hi
                        && rb[1] / sub[1] >= cy_lo
                        && (re[1] - 1) / sub[1] <= cy_hi
                        && rb[2] / sub[2] >= cx_lo
                        && (re[2] - 1) / sub[2] <= cx_hi;
                    if inside {
                        targets.push(r);
                    }
                }
                let decoded_bytes = lazy
                    .decoded_with(|bytes| decoder.decode_sub_chunk(bytes))
                    .map_err(|e| BackfillError::Permanent(format!("v3 sub-chunk decode at {:?}: {}", coord, e)))?;
                decoded.insert(*coord, decoded_bytes);
                raws.insert(*coord, lazy);
            }
            for k in &log_box {
                if !targets.contains(k) {
                    targets.push(*k);
                }
            }

            let fills = super::ome_zarr::fill_covered_chunks(key_dbg, &targets, &shape, &sub, &decoded, "v3");
            Ok(fills
                .into_iter()
                .map(|(k, data)| (k, passthrough(k, data, &raws, &shape, &sub)))
                .collect())
        },
    );

    Ok(BackfillPlan {
        covered,
        sources,
        extract,
    })
}

/// A cache chunk that is exactly one interior vc-delta3d 64³ sub-chunk keeps
/// the downloaded bytes, so the segment log stores them instead of encoding
/// the chunk again.
fn passthrough(
    k: ChunkKey,
    data: ExtractedChunk,
    raws: &HashMap<[usize; 3], Arc<LazySource>>,
    shape: &[usize; 3],
    sub: &[usize; 3],
) -> ExtractedChunk {
    let ExtractedChunk::Bytes(bytes) = data else { return data };
    let c = [k.z as usize, k.y as usize, k.x as usize];
    let interior = (0..3).all(|i| (c[i] + 1) * CHUNK_SIDE <= shape[i]);
    if *sub != [CHUNK_SIDE; 3] || !interior {
        return ExtractedChunk::Bytes(bytes);
    }
    let Some(raw) = raws.get(&c).and_then(|l| l.raw_bytes()) else {
        return ExtractedChunk::Bytes(bytes);
    };
    match vesuvius_delta3d::decoded_size(raw) {
        Ok(n) if n == bytes.len() => ExtractedChunk::Encoded {
            bytes,
            encoded: raw.to_vec(),
        },
        // c3d sub-chunks: re-encode.
        _ => ExtractedChunk::Bytes(bytes),
    }
}
