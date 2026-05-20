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
    BackfillError, BackfillPlan, ChunkBackfiller, SourceOutcome, SourcePayload, SourceSpec,
};
use crate::cache::state::ChunkKey;
use crate::cache::{CHUNK_SIDE, CHUNK_VOXELS};
use memmap::Mmap;
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
                        Some(url) => SourceSpec::Download { key: source_key, url },
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
        let base = [base_z, base_y, base_x];
        let end = [end_z, end_y, end_x];
        let array_for_decode = array.clone();
        let extract = Box::new(move |outcomes: &[SourceOutcome]| -> Result<Vec<u8>, BackfillError> {
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

            let mut out = vec![0u8; CHUNK_VOXELS];
            let mut loaded = 0usize;
            let mut missing = 0usize;

            let stride_y = nchunk[2];
            let stride_z = nchunk[1] * nchunk[2];

            // Cache one decoded chunk per source, since extract owns the
            // bytes payload via Arc and decoding from raw bytes is the
            // hot CPU work. We do it lazily inside the loop so absent
            // sources don't pay for it.
            for (idx, coord) in coords.iter().enumerate() {
                let cz = coord[0];
                let cy = coord[1];
                let cx = coord[2];
                let nz_lo = (cz * nchunk[0]).max(base[0]);
                let nz_hi = ((cz + 1) * nchunk[0]).min(end[0]);
                let ny_lo = (cy * nchunk[1]).max(base[1]);
                let ny_hi = ((cy + 1) * nchunk[1]).min(end[1]);
                let nx_lo = (cx * nchunk[2]).max(base[2]);
                let nx_hi = ((cx + 1) * nchunk[2]).min(end[2]);

                let payload_opt: &Option<SourcePayload> = match &outcomes[idx] {
                    Ok(p) => p,
                    Err(_) => unreachable!("error already short-circuited"),
                };
                let ctx: Arc<ChunkContext> = match payload_opt {
                    Some(p) => {
                        if let Ok(ctx_arc) = p.clone().downcast::<ChunkContext>() {
                            // Local-fallback Compute payload, already
                            // decoded.
                            ctx_arc
                        } else if let Ok(mmap_arc) = p.clone().downcast::<Mmap>() {
                            // Download payload — bytes were spilled to disk
                            // by the cache's on_done handler. Decode from
                            // the mmap so we don't materialize the
                            // compressed payload back into the heap.
                            Arc::new(array_for_decode.decode_chunk_bytes(&mmap_arc[..]))
                        } else if let Ok(bytes_arc) = p.clone().downcast::<Vec<u8>>() {
                            // Fallback when spill write failed: bytes still
                            // in memory.
                            Arc::new(array_for_decode.decode_chunk_bytes(&bytes_arc))
                        } else {
                            return Err(BackfillError::Permanent("source payload type".into()));
                        }
                    }
                    None => {
                        missing += 1;
                        continue;
                    }
                };
                loaded += 1;

                let chunk_base_z = cz * nchunk[0];
                let chunk_base_y = cy * nchunk[1];
                let chunk_base_x = cx * nchunk[2];

                for sz in nz_lo..nz_hi {
                    let lz = sz - base[0];
                    let in_z = (sz - chunk_base_z) * stride_z;
                    let out_z = lz * CHUNK_SIDE * CHUNK_SIDE;
                    for sy in ny_lo..ny_hi {
                        let ly = sy - base[1];
                        let in_y = in_z + (sy - chunk_base_y) * stride_y;
                        let out_y = out_z + ly * CHUNK_SIDE;
                        for sx in nx_lo..nx_hi {
                            let lx = sx - base[2];
                            let in_idx = in_y + (sx - chunk_base_x);
                            out[out_y + lx] = ctx.get(in_idx);
                        }
                    }
                }
            }
            log::trace!(
                "[{}] extract: loaded={} missing={} ({:?})",
                key_dbg,
                loaded,
                missing,
                started.elapsed()
            );
            Ok(out)
        });

        Ok(BackfillPlan { sources, extract })
    }

    fn volume_id(&self) -> String {
        self.volume_id.clone()
    }
}
