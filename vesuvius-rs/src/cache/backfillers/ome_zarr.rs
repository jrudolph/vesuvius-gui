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
//! Source payload type: `Arc<Vec<u8>>` — the raw HTTP body. Sibling cache
//! chunks that consume the same source share one allocation via the `Arc`.
//!
//! Arrays without a chunk URL (e.g. local-disk zarrs) fall back to the
//! `Compute` source variant: the fetch closure runs synchronously on a
//! cache worker and reads the chunk via `array.load_chunk`. Local-disk I/O
//! is fast enough that an async path here would be needless complexity.

use crate::cache::backfiller::{
    BackfillError, BackfillPlan, ChunkBackfiller, ExtractedChunk, SourceOutcome, SourcePayload, SourceSpec,
};
use crate::cache::state::ChunkKey;
use crate::cache::{CHUNK_SIDE, CHUNK_VOXELS};
use memmap::Mmap;
use std::collections::HashMap;
use std::sync::Arc;
use vesuvius_zarr::{ChunkContext, OmeZarrContext, ZarrArray};

pub struct OmeZarrBackfiller {
    volume_id: String,
    extent_xyz: [u32; 3],
    arrays: Vec<ZarrArray<3, u8>>,
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

    fn plan(&self, key: ChunkKey) -> Result<BackfillPlan, BackfillError> {
        let lod = key.lod as usize;
        let array = self.arrays.get(lod).ok_or(BackfillError::OutOfBounds)?;

        // V3 sharded c3d remote arrays take a specialized planning path
        // that issues HTTP Range requests for individual sub-chunks
        // through the central downloader pool. Everything else (v2, local
        // v3, anything that doesn't expose a v3 remote handle) keeps the
        // existing native-chunk path below.
        if let Some(v3) = array.v3_remote_sharded() {
            return super::ome_zarr_v3::plan_v3_chunk(&v3, &self.volume_id, key, lod);
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
        let extract = Box::new(
            move |outcomes: &[SourceOutcome]| -> Result<Vec<(ChunkKey, ExtractedChunk)>, BackfillError> {
                let started = std::time::Instant::now();
                // Bail fast on errors. Pick worst severity.
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

                let stride_y = nchunk[2];
                let stride_z = nchunk[1] * nchunk[2];

                // coord → outcome index for O(1) "is this native in our
                // source set?" checks.
                let coord_idx: HashMap<[usize; 3], usize> =
                    coords.iter().copied().enumerate().map(|(i, c)| (c, i)).collect();

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
                    } else if let Ok(mmap_arc) = payload.clone().downcast::<Mmap>() {
                        // Download payload — bytes were spilled to disk by
                        // the cache's on_done handler. Decode from the mmap
                        // so we don't materialize the compressed payload
                        // back into the heap.
                        Arc::new(array_for_decode.decode_chunk_bytes(&mmap_arc[..]))
                    } else if let Ok(bytes_arc) = payload.clone().downcast::<Vec<u8>>() {
                        // Fallback when spill write failed: bytes still in
                        // memory.
                        Arc::new(array_for_decode.decode_chunk_bytes(&bytes_arc))
                    } else {
                        return Err(BackfillError::Permanent("source payload type".into()));
                    };
                    decoded.insert(*coord, ctx);
                }

                // Bounding box (in cache-chunk coords at the current LOD)
                // of cache chunks that the loaded source set could possibly
                // cover. Iterating this box and filtering by "all overlaps
                // in source set" yields every cache chunk we have full data
                // for — the primary and its siblings.
                let mut bbox_lo = [usize::MAX; 3];
                let mut bbox_hi = [0usize; 3];
                for coord in coords.iter() {
                    let voxel_lo = [coord[0] * nchunk[0], coord[1] * nchunk[1], coord[2] * nchunk[2]];
                    let voxel_hi = [
                        ((coord[0] + 1) * nchunk[0]).min(shape[0]),
                        ((coord[1] + 1) * nchunk[1]).min(shape[1]),
                        ((coord[2] + 1) * nchunk[2]).min(shape[2]),
                    ];
                    let cache_lo = [voxel_lo[0] / CHUNK_SIDE, voxel_lo[1] / CHUNK_SIDE, voxel_lo[2] / CHUNK_SIDE];
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

                let mut output: Vec<(ChunkKey, ExtractedChunk)> = Vec::new();
                let mut primary_found = false;
                let mut empty_count = 0usize;
                let mut bytes_count = 0usize;

                for kz in bbox_lo[0]..bbox_hi[0] {
                    for ky in bbox_lo[1]..bbox_hi[1] {
                        for kx in bbox_lo[2]..bbox_hi[2] {
                            let cv_base = [kz * CHUNK_SIDE, ky * CHUNK_SIDE, kx * CHUNK_SIDE];
                            // Fully out of volume bounds → skip. OOB chunks
                            // are never dispatched (dispatch_chunk rejects
                            // them) so we shouldn't fabricate Empties for
                            // them here.
                            if cv_base[0] >= shape[0] || cv_base[1] >= shape[1] || cv_base[2] >= shape[2] {
                                continue;
                            }
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

                            // All overlapping native chunks must be in our
                            // source set. If not, we can't fill this cache
                            // chunk from this extract.
                            let mut covered = true;
                            let mut all_absent = true;
                            'check: for ncz in ncz_lo..=ncz_hi {
                                for ncy in ncy_lo..=ncy_hi {
                                    for ncx in ncx_lo..=ncx_hi {
                                        let coord = [ncz, ncy, ncx];
                                        let Some(&idx) = coord_idx.get(&coord) else {
                                            covered = false;
                                            break 'check;
                                        };
                                        if matches!(&outcomes[idx], Ok(Some(_))) {
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

                            // Slice each overlapping loaded native chunk into
                            // this cache chunk's buffer. Absent (Ok(None))
                            // overlaps leave their region zero-filled.
                            let mut out = vec![0u8; CHUNK_VOXELS];
                            for ncz in ncz_lo..=ncz_hi {
                                for ncy in ncy_lo..=ncy_hi {
                                    for ncx in ncx_lo..=ncx_hi {
                                        let coord = [ncz, ncy, ncx];
                                        let Some(ctx) = decoded.get(&coord) else {
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
                                                    let in_idx = in_y + (sx - chunk_base_x);
                                                    out[out_y + lx] = ctx.get(in_idx);
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
                    // The primary's overlap set IS our source set by
                    // construction, so this is structurally impossible —
                    // log loudly and let the cache fall back to cooldown.
                    log::warn!("[{}] sibling enumeration missed the primary key", key_dbg);
                }

                log::trace!(
                    "[{}] extract: {} bytes + {} empty ({:?})",
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

    fn volume_id(&self) -> String {
        self.volume_id.clone()
    }
}
