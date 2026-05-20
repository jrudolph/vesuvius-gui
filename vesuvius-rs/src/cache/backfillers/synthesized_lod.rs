//! Wraps another `ChunkBackfiller` and synthesizes additional coarse LODs
//! beyond what the inner backfiller supports.
//!
//! For LOD ≤ `inner.max_lod()` the plan is delegated unchanged. For LOD > that,
//! the plan declares 8 `SourceSpec::Chunk` dependencies — the 2×2×2 children at
//! the next-finer LOD — and the extract closure averages each 2×2×2 voxel block
//! from those children into one output voxel. Recursion happens at the cache
//! layer: a synth chunk at LOD `L` depends on chunks at LOD `L-1`, which may
//! themselves be synthesized.
//!
//! Why cache-level (rather than per-pixel averaging in the volume layer)?
//! Synthesized chunks land in the on-disk cache like any other, so the cost
//! amortizes across paint passes and across `get()` calls from any consumer
//! (slice paint, surface paint, PPM, …). The hot slot in `UnifiedVolume::get`
//! also keeps per-pixel sampling at one mmap hit even at extreme zoom-out.

use crate::cache::backfiller::{BackfillError, BackfillPlan, ChunkBackfiller, SourceOutcome, SourceSpec};
use crate::cache::state::{ChunkKey, ChunkState};
use crate::cache::{CHUNK_SIDE, CHUNK_VOXELS};
use std::sync::Arc;

pub struct SynthesizedLodBackfiller {
    inner: Arc<dyn ChunkBackfiller>,
    extra_levels: u8,
}

impl SynthesizedLodBackfiller {
    /// Wrap `inner` and expose an additional `extra_levels` LODs above
    /// `inner.max_lod()`. Each extra level is built by 2× averaging from the
    /// next-finer level (recursing through previously-synthesized levels).
    pub fn new(inner: Arc<dyn ChunkBackfiller>, extra_levels: u8) -> Self {
        Self { inner, extra_levels }
    }
}

impl ChunkBackfiller for SynthesizedLodBackfiller {
    fn max_lod(&self) -> u8 {
        self.inner.max_lod().saturating_add(self.extra_levels)
    }

    fn voxel_extent(&self) -> [u32; 3] {
        self.inner.voxel_extent()
    }

    fn volume_id(&self) -> String {
        // Synthesized chunks share an on-disk namespace with the inner
        // backfiller — LOD distinguishes them — so they can coexist in the
        // same per-volume cache directory.
        self.inner.volume_id()
    }

    fn plan(&self, key: ChunkKey) -> Result<BackfillPlan, BackfillError> {
        let native_max = self.inner.max_lod();
        if key.lod <= native_max {
            return self.inner.plan(key);
        }

        let child_lod = key.lod - 1;
        let mut sources: Vec<SourceSpec> = Vec::with_capacity(8);
        // Source order matters: `extract` indexes outcomes by
        // (dz, dy, dx) packed as `dz*4 + dy*2 + dx`.
        for dz in 0..2u32 {
            for dy in 0..2u32 {
                for dx in 0..2u32 {
                    let child_key = ChunkKey::new(child_lod, key.x * 2 + dx, key.y * 2 + dy, key.z * 2 + dz);
                    let source_key = format!("synth:{}", child_key);
                    sources.push(SourceSpec::Chunk { key: source_key, chunk_key: child_key });
                }
            }
        }

        let extract = Box::new(move |outcomes: &[SourceOutcome]| -> Result<Vec<u8>, BackfillError> {
            if outcomes.len() != 8 {
                return Err(BackfillError::Permanent(format!(
                    "synth lod {}: expected 8 child outcomes, got {}",
                    key.lod,
                    outcomes.len()
                )));
            }

            // Resolve each child to a `&[u8]` view into its 64³ chunk mmap,
            // or `None` if the child is missing (extract treats that as zeros
            // for that octant).
            let children: [Option<&[u8]>; 8] = {
                let mut arr: [Option<&[u8]>; 8] = [None; 8];
                for (i, outcome) in outcomes.iter().enumerate() {
                    match outcome {
                        Ok(Some(payload)) => {
                            // Sources from the chunk-dep path carry the
                            // child's `ChunkState` (wrapped in the
                            // `Arc<dyn Any + Send + Sync>` SourcePayload);
                            // native-source payloads (Compute / Download)
                            // never reach a synth plan.
                            let state = payload.downcast_ref::<ChunkState>().ok_or_else(|| {
                                BackfillError::Permanent(format!(
                                    "synth lod {}: unexpected payload type for child {}",
                                    key.lod, i
                                ))
                            })?;
                            if let ChunkState::Resident(mmap) = state {
                                arr[i] = Some(&mmap[..]);
                            } else {
                                arr[i] = None;
                            }
                        }
                        Ok(None) => arr[i] = None,
                        Err(e) => return Err(e.clone()),
                    }
                }
                arr
            };

            let mut out = vec![0u8; CHUNK_VOXELS];
            // Each output voxel (ox, oy, oz) maps to exactly one child
            // (dx, dy, dz) = (ox / 32, oy / 32, oz / 32) and to a 2×2×2 block
            // within that child starting at (2*(ox%32), 2*(oy%32), 2*(oz%32)).
            for oz in 0..CHUNK_SIDE {
                let dz = oz / 32;
                let bz = (oz % 32) * 2;
                for oy in 0..CHUNK_SIDE {
                    let dy = oy / 32;
                    let by = (oy % 32) * 2;
                    for ox in 0..CHUNK_SIDE {
                        let dx = ox / 32;
                        let bx = (ox % 32) * 2;
                        let child_idx = dz * 4 + dy * 2 + dx;
                        let Some(child) = children[child_idx] else {
                            continue;
                        };
                        // Average 2×2×2 = 8 source voxels. Promote to u16 to
                        // avoid overflow when summing 8 u8s.
                        let mut sum: u16 = 0;
                        for ddz in 0..2 {
                            for ddy in 0..2 {
                                for ddx in 0..2 {
                                    let src_off =
                                        (bz + ddz) * CHUNK_SIDE * CHUNK_SIDE + (by + ddy) * CHUNK_SIDE + (bx + ddx);
                                    sum += child[src_off] as u16;
                                }
                            }
                        }
                        let out_off = oz * CHUNK_SIDE * CHUNK_SIDE + oy * CHUNK_SIDE + ox;
                        out[out_off] = (sum / 8) as u8;
                    }
                }
            }
            Ok(out)
        });

        Ok(BackfillPlan { sources, extract })
    }
}
