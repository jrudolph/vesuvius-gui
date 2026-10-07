//! `OmeZarrBackfiller` — fetch unified-cache chunks from an OME-Zarr
//! multiscale volume.
//!
//! The plan for one 64³ cache chunk lists the native zarr chunks (typically
//! 128³ or 256³) that overlap it. Each source is a `Download` request —
//! HTTP work runs in the cache's centralized downloader pool, never blocking
//! a cache worker. The downloader hands back the raw chunk bytes and the
//! cache calls our `extract` closure on a worker thread to do the decode
//! (blosc / zstd / raw) and slice into the 64³ output. Decode is CPU-bound;
//! keeping it in extract means it runs on the cache worker pool (CPU-sized)
//! rather than the download pool (I/O-sized).
//!
//! Source payload type: `Arc<LazySource>` over the raw HTTP body (mmap-
//! backed via the raw store). Sibling cache chunks that consume the same
//! source share one allocation via the `Arc`.
//!
//! Arrays without a chunk URL (e.g. local-disk zarrs) fall back to the
//! `Compute` source variant: the fetch closure runs synchronously on a
//! cache worker and reads the chunk via `array.load_chunk`. Local-disk I/O
//! is fast enough that an async path here would be needless complexity.

use crate::cache::backfiller::{
    BackfillError, BackfillPlan, ChunkBackfiller, ExtractedChunk, LazySource, SourceOutcome, SourcePayload,
    SourceSpec,
};
use crate::cache::state::ChunkKey;
use crate::cache::{CHUNK_SIDE, CHUNK_VOXELS};
use std::collections::HashMap;
use std::sync::Arc;
use vesuvius_zarr::{ChunkContext, OmeZarrContext, ZarrArray};

pub struct OmeZarrBackfiller {
    volume_id: String,
    extent_xyz: [u32; 3],
    arrays: Vec<ZarrArray<3, u8>>,
    /// When true, the v3 plan covers (decodes + persists) every 64³ cache
    /// chunk inside a decoded 256³ sub-chunk on the first request, instead
    /// of just the requesting chunk. The lazy default suits interactive
    /// slab-browsing (the viewport never visits ~3/4 of a sub-chunk's
    /// depth); the offline renderer eventually reads the whole sub-chunk,
    /// so persisting all siblings amortizes the ~500ms decode over 64
    /// chunks instead of re-decoding from the raw store per sibling.
    eager_materialization: bool,
}

impl OmeZarrBackfiller {
    pub fn from_ome(volume_id: impl Into<String>, ome: OmeZarrContext) -> Self {
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
            eager_materialization: false,
        }
    }

    /// Enable eager full-sub-chunk materialization (see the
    /// `eager_materialization` field). Builder-style for use at construction.
    pub fn with_eager_materialization(mut self, enabled: bool) -> Self {
        self.eager_materialization = enabled;
        self
    }
}

impl ChunkBackfiller for OmeZarrBackfiller {
    fn max_lod(&self) -> u8 {
        self.arrays.len().saturating_sub(1) as u8
    }

    fn voxel_extent(&self) -> [u32; 3] {
        self.extent_xyz
    }

    fn plan(&self, key: ChunkKey) -> Result<BackfillPlan, BackfillError> {
        let lod = key.lod as usize;
        let array = self.arrays.get(lod).ok_or(BackfillError::OutOfBounds)?;

        // V3 sharded c3d remote arrays take a specialized planning path
        // that issues HTTP Range requests for individual sub-chunks
        // through the central downloader pool. Everything else (v2, local
        // v3, anything that doesn't expose a v3 remote handle) keeps the
        // existing native-chunk path below.
        if let Some(v3) = array.v3_remote_sharded() {
            return super::ome_zarr_v3::plan_v3_chunk(&v3, &self.volume_id, key, lod, self.eager_materialization);
        }

        let def = array.def();
        let shape = def.shape.clone();
        let nchunk = def.chunks.clone();
        if shape.len() != 3 || nchunk.len() != 3 {
            return Err(BackfillError::Permanent(format!(
                "expected 3D zarr at lod {}, got shape={:?} chunks={:?}",
                lod, shape, nchunk
            )));
        }

        let base_x = key.x as usize * CHUNK_SIDE;
        let base_y = key.y as usize * CHUNK_SIDE;
        let base_z = key.z as usize * CHUNK_SIDE;
        let end_x = (base_x + CHUNK_SIDE).min(shape[2]);
        let end_y = (base_y + CHUNK_SIDE).min(shape[1]);
        let end_z = (base_z + CHUNK_SIDE).min(shape[0]);
        if base_x >= shape[2] || base_y >= shape[1] || base_z >= shape[0] {
            return Err(BackfillError::OutOfBounds);
        }

        let cx_lo = base_x / nchunk[2];
        let cx_hi = (end_x - 1) / nchunk[2];
        let cy_lo = base_y / nchunk[1];
        let cy_hi = (end_y - 1) / nchunk[1];
        let cz_lo = base_z / nchunk[0];
        let cz_hi = (end_z - 1) / nchunk[0];

        let definitive_missing = array.cache_missing();
        let n_sources = (cz_hi - cz_lo + 1) * (cy_hi - cy_lo + 1) * (cx_hi - cx_lo + 1);
        log::trace!(
            "[{}] plan → {} native source(s) (cz={}..={}, cy={}..={}, cx={}..={})",
            key,
            n_sources,
            cz_lo,
            cz_hi,
            cy_lo,
            cy_hi,
            cx_lo,
            cx_hi
        );

        let mut sources: Vec<SourceSpec> = Vec::new();
        let mut coords: Vec<[usize; 3]> = Vec::new();
        let covered = covered_cache_chunks([cz_lo, cy_lo, cx_lo], [cz_hi, cy_hi, cx_hi], &nchunk, &shape, key.lod);
        for cz in cz_lo..=cz_hi {
            for cy in cy_lo..=cy_hi {
                for cx in cx_lo..=cx_hi {
                    let coord = [cz, cy, cx];
                    let source_key = format!(
                        "{}/L{:02}/{}/{}/{}",
                        self.volume_id, lod, coord[0], coord[1], coord[2]
                    );
                    let spec = match array.chunk_url(coord) {
                        Some(url) => SourceSpec::Download {
                            key: source_key,
                            url,
                            range: None,
                        },
                        None => {
                            // Local-disk array: no URL, fall back to a
                            // synchronous compute source.
                            let array_clone = array.clone();
                            let source_key_log = source_key.clone();
                            let fetch: Box<dyn FnOnce() -> SourceOutcome + Send + 'static> = Box::new(move || {
                                let t0 = std::time::Instant::now();
                                log::trace!("[{}] local fetch start", source_key_log);
                                match array_clone.load_chunk(coord) {
                                    Some(ctx) => {
                                        log::trace!("[{}] local fetch done ({:?})", source_key_log, t0.elapsed());
                                        Ok(Some(Arc::new(ctx) as SourcePayload))
                                    }
                                    None => {
                                        if definitive_missing {
                                            Ok(None)
                                        } else {
                                            Err(BackfillError::Transient(format!(
                                                "async native chunk {:?} not ready",
                                                coord
                                            )))
                                        }
                                    }
                                }
                            });
                            SourceSpec::Compute { key: source_key, fetch }
                        }
                    };
                    sources.push(spec);
                    coords.push(coord);
                }
            }
        }

        let key_dbg = key;
        let array_for_decode = array.clone();
        let covered_for_extract = covered.clone();
        let extract = Box::new(
            move |outcomes: &[SourceOutcome]| -> Result<Vec<(ChunkKey, ExtractedChunk)>, BackfillError> {
                triage_outcomes(outcomes)?;

                // Decode each loaded source upfront. With 128³ native chunks
                // each one feeds 8 sibling 64³ cache chunks; doing the decode
                // once and then slicing into all siblings is the whole point
                // of this extract path.
                let mut decoded: HashMap<[usize; 3], Arc<ChunkContext>> = HashMap::new();
                for (i, coord) in coords.iter().enumerate() {
                    let payload_opt: &Option<SourcePayload> = match &outcomes[i] {
                        Ok(p) => p,
                        Err(_) => unreachable!("error already short-circuited"),
                    };
                    let Some(payload) = payload_opt else { continue };
                    let ctx = if let Ok(ctx_arc) = payload.clone().downcast::<ChunkContext>() {
                        // Local-fallback Compute payload, already decoded.
                        ctx_arc
                    } else if let Ok(lazy) = payload.clone().downcast::<LazySource>() {
                        // Download payload — raw bytes are mmap-backed by
                        // the cache's raw store (or heap on spill failure).
                        // v2 decodes into a ChunkContext rather than a flat
                        // buffer, so the LazySource decode memo isn't used
                        // here — just the raw bytes.
                        let bytes = lazy
                            .raw_bytes()
                            .ok_or_else(|| BackfillError::Permanent("source payload type".into()))?;
                        Arc::new(array_for_decode.decode_chunk_bytes(bytes))
                    } else {
                        return Err(BackfillError::Permanent("source payload type".into()));
                    };
                    decoded.insert(*coord, ctx);
                }

                Ok(fill_covered_chunks(
                    key_dbg,
                    &covered_for_extract,
                    &shape,
                    &nchunk,
                    &decoded,
                    "v2",
                ))
            },
        );

        Ok(BackfillPlan {
            covered,
            sources,
            extract,
        })
    }

    fn volume_id(&self) -> String {
        self.volume_id.clone()
    }
}

/// Compute the set of cache chunks whose every overlapping native chunk
/// lies inside the dense native-chunk box `[lo, hi]` (inclusive, (z, y, x)
/// order). That's exactly the set of cache chunks an extract over the
/// box's sources could fully materialize — its `covered` list.
///
/// Shared between the v2 OME-Zarr planner and the v3 sub-chunk planner;
/// the math is identical once the native-chunk shape is plugged in
/// (`nchunk` for v2 = native chunk shape, for v3 = c3d sub-chunk shape).
/// Both planners enumerate full axis-aligned ranges, so the box form
/// replaces the old `HashSet` membership walk with six comparisons per
/// cache chunk.
pub(super) fn covered_cache_chunks(
    lo: [usize; 3],
    hi: [usize; 3],
    nchunk: &[usize],
    shape: &[usize],
    lod: u8,
) -> Vec<ChunkKey> {
    let voxel_lo = [lo[0] * nchunk[0], lo[1] * nchunk[1], lo[2] * nchunk[2]];
    let voxel_hi = [
        ((hi[0] + 1) * nchunk[0]).min(shape[0]),
        ((hi[1] + 1) * nchunk[1]).min(shape[1]),
        ((hi[2] + 1) * nchunk[2]).min(shape[2]),
    ];
    let cache_lo = [voxel_lo[0] / CHUNK_SIDE, voxel_lo[1] / CHUNK_SIDE, voxel_lo[2] / CHUNK_SIDE];
    let cache_hi = [
        voxel_hi[0].div_ceil(CHUNK_SIDE),
        voxel_hi[1].div_ceil(CHUNK_SIDE),
        voxel_hi[2].div_ceil(CHUNK_SIDE),
    ];
    let mut out = Vec::new();
    for kz in cache_lo[0]..cache_hi[0] {
        for ky in cache_lo[1]..cache_hi[1] {
            for kx in cache_lo[2]..cache_hi[2] {
                let cv_base = [kz * CHUNK_SIDE, ky * CHUNK_SIDE, kx * CHUNK_SIDE];
                if cv_base[0] >= shape[0] || cv_base[1] >= shape[1] || cv_base[2] >= shape[2] {
                    continue;
                }
                let cv_end = [
                    (cv_base[0] + CHUNK_SIDE).min(shape[0]),
                    (cv_base[1] + CHUNK_SIDE).min(shape[1]),
                    (cv_base[2] + CHUNK_SIDE).min(shape[2]),
                ];
                // Covered iff every overlapping native chunk sits inside
                // the box.
                let inside = cv_base[0] / nchunk[0] >= lo[0]
                    && (cv_end[0] - 1) / nchunk[0] <= hi[0]
                    && cv_base[1] / nchunk[1] >= lo[1]
                    && (cv_end[1] - 1) / nchunk[1] <= hi[1]
                    && cv_base[2] / nchunk[2] >= lo[2]
                    && (cv_end[2] - 1) / nchunk[2] <= hi[2];
                if inside {
                    out.push(ChunkKey::new(lod, kx as u32, ky as u32, kz as u32));
                }
            }
        }
    }
    out
}

/// Sampling interface over one decoded native chunk, shared by the v2
/// (`ChunkContext`) and v3 (raw c3d buffer) extract paths so they can
/// drive the same slicing loop.
pub(super) trait NativeSample {
    fn at(&self, idx: usize) -> u8;
}

impl NativeSample for ChunkContext {
    #[inline]
    fn at(&self, idx: usize) -> u8 {
        self.get(idx)
    }
}

impl NativeSample for Vec<u8> {
    #[inline]
    fn at(&self, idx: usize) -> u8 {
        self[idx]
    }
}

/// Short-circuit an extract on the worst error among `outcomes`:
/// any permanent (or out-of-bounds) failure dominates, then the first
/// transient one, then a cancellation. `Ok(())` means every source either
/// loaded or was definitively absent.
pub(super) fn triage_outcomes(outcomes: &[SourceOutcome]) -> Result<(), BackfillError> {
    let mut transient: Option<String> = None;
    let mut cancelled = false;
    for o in outcomes {
        if let Err(e) = o {
            match e {
                BackfillError::OutOfBounds => return Err(BackfillError::Permanent("oob source".into())),
                BackfillError::Permanent(s) => return Err(BackfillError::Permanent(s.clone())),
                BackfillError::Transient(s) => {
                    if transient.is_none() {
                        transient = Some(s.clone());
                    }
                }
                BackfillError::Cancelled => cancelled = true,
            }
        }
    }
    match transient {
        Some(s) => Err(BackfillError::Transient(s)),
        None if cancelled => Err(BackfillError::Cancelled),
        None => Ok(()),
    }
}

/// Fill every plan-time `covered` cache chunk from `decoded` (the map of
/// successfully loaded + decoded native chunks). Chunks none of whose
/// overlapping natives decoded come out `Empty`; otherwise each decoded
/// overlap is sliced into the 64³ buffer and absent overlaps leave their
/// region zero-filled. This is the shared back half of the v2 and v3
/// extract closures — the plan-time coverage computation guarantees
/// every key in `covered` (the primary and its siblings) has its full
/// overlap set accounted for.
pub(super) fn fill_covered_chunks<D: NativeSample>(
    primary: ChunkKey,
    covered: &[ChunkKey],
    shape: &[usize],
    nchunk: &[usize],
    decoded: &HashMap<[usize; 3], Arc<D>>,
    label: &str,
) -> Vec<(ChunkKey, ExtractedChunk)> {
    let started = std::time::Instant::now();
    let stride_y = nchunk[2];
    let stride_z = nchunk[1] * nchunk[2];
    let mut output: Vec<(ChunkKey, ExtractedChunk)> = Vec::with_capacity(covered.len());
    let mut primary_found = false;
    let mut empty_count = 0usize;
    let mut bytes_count = 0usize;

    for &chunk_key in covered {
        let cv_base = [
            chunk_key.z as usize * CHUNK_SIDE,
            chunk_key.y as usize * CHUNK_SIDE,
            chunk_key.x as usize * CHUNK_SIDE,
        ];
        let cv_end = [
            (cv_base[0] + CHUNK_SIDE).min(shape[0]),
            (cv_base[1] + CHUNK_SIDE).min(shape[1]),
            (cv_base[2] + CHUNK_SIDE).min(shape[2]),
        ];

        // Native chunks overlapping this cache chunk.
        let ncz_lo = cv_base[0] / nchunk[0];
        let ncz_hi = (cv_end[0] - 1) / nchunk[0];
        let ncy_lo = cv_base[1] / nchunk[1];
        let ncy_hi = (cv_end[1] - 1) / nchunk[1];
        let ncx_lo = cv_base[2] / nchunk[2];
        let ncx_hi = (cv_end[2] - 1) / nchunk[2];

        if chunk_key == primary {
            primary_found = true;
        }

        let mut all_absent = true;
        'probe: for ncz in ncz_lo..=ncz_hi {
            for ncy in ncy_lo..=ncy_hi {
                for ncx in ncx_lo..=ncx_hi {
                    if decoded.contains_key(&[ncz, ncy, ncx]) {
                        all_absent = false;
                        break 'probe;
                    }
                }
            }
        }
        if all_absent {
            output.push((chunk_key, ExtractedChunk::Empty));
            empty_count += 1;
            continue;
        }

        // Slice each overlapping decoded native chunk into this cache
        // chunk's buffer. Absent overlaps leave their region zero-filled.
        let mut out = vec![0u8; CHUNK_VOXELS];
        for ncz in ncz_lo..=ncz_hi {
            for ncy in ncy_lo..=ncy_hi {
                for ncx in ncx_lo..=ncx_hi {
                    let Some(d) = decoded.get(&[ncz, ncy, ncx]) else {
                        continue;
                    };
                    let chunk_base_z = ncz * nchunk[0];
                    let chunk_base_y = ncy * nchunk[1];
                    let chunk_base_x = ncx * nchunk[2];
                    let nz_lo = chunk_base_z.max(cv_base[0]);
                    let nz_hi = (chunk_base_z + nchunk[0]).min(cv_end[0]);
                    let ny_lo = chunk_base_y.max(cv_base[1]);
                    let ny_hi = (chunk_base_y + nchunk[1]).min(cv_end[1]);
                    let nx_lo = chunk_base_x.max(cv_base[2]);
                    let nx_hi = (chunk_base_x + nchunk[2]).min(cv_end[2]);
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
                                out[out_y + lx] = d.at(in_y + (sx - chunk_base_x));
                            }
                        }
                    }
                }
            }
        }
        output.push((chunk_key, ExtractedChunk::Bytes(out)));
        bytes_count += 1;
    }

    if !primary_found {
        // The primary's overlap set is inside the plan's source box by
        // construction, so this is structurally impossible — log loudly
        // and let the cache fall back to cooldown.
        log::warn!("[{}] {} extract output missed the primary key", primary, label);
    }

    log::trace!(
        "[{}] {} extract: {} bytes + {} empty ({:?})",
        primary,
        label,
        bytes_count,
        empty_count,
        started.elapsed()
    );
    output
}
