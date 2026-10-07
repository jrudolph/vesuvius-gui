use super::backfiller::{BackfillError, BackfillPlan, ExtractedChunk};
use super::backfillers::synthesized_lod::SynthesizedLodBackfiller;
use super::backfillers::synthetic::SyntheticBackfiller;
use super::*;
use crate::volume::{DrawingConfig, Image, PaintVolume, VoxelVolume};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

fn tmp_root(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "vesuvius-cache-{}-{}-{}",
        label,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn miss_then_fetch_then_resident() {
    let root = tmp_root("miss-fetch");
    let backfiller = Arc::new(SyntheticBackfiller::new("test", [128, 128, 128], 0, |x, y, z, _| {
        (x ^ y ^ z) as u8
    }));
    let cache = UnifiedCache::for_cache_dir(&root).open_volume(backfiller);

    let key = ChunkKey::new(0, 0, 0, 0);
    // First touch returns Pending (or already Resident if the worker is
    // fast — both are acceptable).
    let state = cache.state_or_fetch(key);
    assert!(matches!(state.as_ref(), ChunkState::Pending { .. } | ChunkState::Resident { .. }));

    let state = cache.wait_for(key, Duration::from_secs(2));
    assert!(state.as_resident().is_some(), "chunk should be resident: {:?}", state);

    // Direct voxel read at (1, 2, 3).
    let v = cache.voxel(1, 2, 3, 0);
    assert_eq!(v, (1u32 ^ 2 ^ 3) as u8);
}

/// Backfiller whose plan declares N `Compute` sources that all resolve to
/// `Ok(None)`. Used to exercise the all-absent → `Empty` path.
struct AllAbsentBackfiller {
    volume_id: String,
    extent: [u32; 3],
    /// Counts how many `Compute` fetches were actually invoked. After a
    /// fresh fetch + persisted reload, this should not increment on the
    /// reload — the disk sentinel short-circuits.
    fetch_count: Arc<std::sync::atomic::AtomicUsize>,
}

impl crate::cache::backfiller::ChunkBackfiller for AllAbsentBackfiller {
    fn max_lod(&self) -> u8 {
        0
    }
    fn voxel_extent(&self) -> [u32; 3] {
        self.extent
    }
    fn volume_id(&self) -> String {
        self.volume_id.clone()
    }
    fn plan(
        &self,
        key: ChunkKey,
    ) -> Result<crate::cache::backfiller::BackfillPlan, crate::cache::backfiller::BackfillError> {
        use crate::cache::backfiller::{BackfillPlan, SourceOutcome, SourceSpec};
        let counter = self.fetch_count.clone();
        let source_key = format!("absent/{}/{}/{}/{}", key.lod, key.z, key.y, key.x);
        let fetch: Box<dyn FnOnce() -> SourceOutcome + Send + 'static> = Box::new(move || {
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(None)
        });
        let sources = vec![SourceSpec::Compute { key: source_key, fetch }];
        // With every source absent the backfiller knows the chunk is
        // definitively empty — surface that explicitly so the cache marks
        // it Empty on disk + in the map. (Pre-sibling-fill, the cache had
        // an all-absent fast path that skipped extract; we now expect
        // extract to encode this itself.)
        let extract = Box::new(move |_inputs: &[SourceOutcome]| Ok(vec![(key, ExtractedChunk::Empty)]));
        Ok(BackfillPlan {
            covered: vec![key],
            sources,
            extract,
        })
    }
}

#[test]
fn all_absent_sources_transition_to_empty_and_persist() {
    let root = tmp_root("all-absent");
    let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let backfiller = Arc::new(AllAbsentBackfiller {
        volume_id: "absent-test".to_string(),
        extent: [128, 128, 128],
        fetch_count: counter.clone(),
    });
    let cache = UnifiedCache::for_cache_dir(&root).open_volume(backfiller.clone());

    let key = ChunkKey::new(0, 0, 0, 0);
    let state = cache.wait_for(key, Duration::from_secs(2));
    assert!(matches!(state.as_ref(), ChunkState::Empty), "expected Empty, got {:?}", state);
    assert!(state.is_terminal());
    assert!(state.as_resident().is_none());
    assert_eq!(
        counter.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "fetch should have run exactly once"
    );

    // Reopen with a fresh cache (same disk root). The Empty entry in the
    // sidecar should short-circuit dispatch — no second fetch.
    cache.flush();
    drop(cache);
    let backfiller2 = Arc::new(AllAbsentBackfiller {
        volume_id: "absent-test".to_string(),
        extent: [128, 128, 128],
        fetch_count: counter.clone(),
    });
    let cache2 = UnifiedCache::for_cache_dir(&root).open_volume(backfiller2);
    let state2 = cache2.state_or_fetch(key);
    assert!(matches!(state2.as_ref(), ChunkState::Empty), "expected Empty on reload, got {:?}", state2);
    assert_eq!(
        counter.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "no refetch expected after disk sentinel hit"
    );

    // Voxel sampler on an Empty chunk returns 0 (zero data) without any
    // refetch attempt.
    let v = cache2.voxel(5, 5, 5, 0);
    assert_eq!(v, 0);
}

/// Backfiller that simulates one native chunk feeding 8 sibling 64³ cache
/// chunks. Its plan lists a single Compute source; the extract closure
/// emits an `ExtractedChunk` entry for every cache chunk in the 2×2×2
/// volume regardless of which one triggered the dispatch.
///
/// Used to exercise the sibling-fill machinery in `extract_chunk`: a
/// dispatch for any of the 8 chunks should fetch + extract exactly once,
/// and dispatching the other 7 afterwards should hit the disk path without
/// triggering another fetch.
struct SiblingFillBackfiller {
    volume_id: String,
    fetch_count: Arc<std::sync::atomic::AtomicUsize>,
    extract_count: Arc<std::sync::atomic::AtomicUsize>,
}

impl crate::cache::backfiller::ChunkBackfiller for SiblingFillBackfiller {
    fn max_lod(&self) -> u8 {
        0
    }
    fn voxel_extent(&self) -> [u32; 3] {
        [128, 128, 128]
    }
    fn volume_id(&self) -> String {
        self.volume_id.clone()
    }
    fn plan(
        &self,
        _key: ChunkKey,
    ) -> Result<crate::cache::backfiller::BackfillPlan, crate::cache::backfiller::BackfillError> {
        use crate::cache::backfiller::{BackfillPlan, SourceOutcome, SourcePayload, SourceSpec};
        let fetch_counter = self.fetch_count.clone();
        let extract_counter = self.extract_count.clone();
        // One source standing in for the single 128³ native chunk that
        // covers all 8 sibling cache chunks. The source key is shared
        // across plans, so the cache dedupes within a frame; the disk
        // path dedupes across frames.
        let source_key = "native/0/0/0".to_string();
        let fetch: Box<dyn FnOnce() -> SourceOutcome + Send + 'static> = Box::new(move || {
            fetch_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            // Payload contents don't matter — the extract ignores them.
            Ok(Some(Arc::new(vec![0u8; 1]) as SourcePayload))
        });
        let sources = vec![SourceSpec::Compute { key: source_key, fetch }];
        let extract = Box::new(move |_inputs: &[SourceOutcome]| {
            extract_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let mut out = Vec::with_capacity(8);
            for kz in 0..2u32 {
                for ky in 0..2u32 {
                    for kx in 0..2u32 {
                        // Marker uniquely identifies the cache chunk so the
                        // disk-path assertion can verify byte-for-byte
                        // correctness.
                        let marker = (kz * 4 + ky * 2 + kx) as u8;
                        let sib_key = ChunkKey::new(0, kx, ky, kz);
                        out.push((sib_key, ExtractedChunk::Bytes(vec![marker; CHUNK_VOXELS])));
                    }
                }
            }
            Ok(out)
        });
        let mut covered = Vec::with_capacity(8);
        for kz in 0..2u32 {
            for ky in 0..2u32 {
                for kx in 0..2u32 {
                    covered.push(ChunkKey::new(0, kx, ky, kz));
                }
            }
        }
        Ok(BackfillPlan {
            covered,
            sources,
            extract,
        })
    }
}

#[test]
fn extract_writes_sibling_chunks_so_subsequent_dispatches_skip_fetch() {
    let root = tmp_root("sibling-fill");
    let fetch_counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let extract_counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let backfiller = Arc::new(SiblingFillBackfiller {
        volume_id: "sibling-fill".to_string(),
        fetch_count: fetch_counter.clone(),
        extract_count: extract_counter.clone(),
    });
    let cache = UnifiedCache::for_cache_dir(&root).open_volume(backfiller);

    // Dispatch ONE primary chunk and wait for it to land.
    let primary = ChunkKey::new(0, 0, 0, 0);
    let state = cache.wait_for(primary, Duration::from_secs(2));
    assert!(state.as_resident().is_some(), "primary should be resident: {:?}", state);
    assert_eq!(state.as_resident().unwrap()[0], 0, "primary marker");
    assert_eq!(
        fetch_counter.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "fetch should have run exactly once for the shared source"
    );
    assert_eq!(extract_counter.load(std::sync::atomic::Ordering::SeqCst), 1);

    // All 7 sibling cache chunks were written to disk inside the same
    // extract call. Dispatching any of them now must short-circuit on the
    // disk path — no second fetch, no second extract.
    for kz in 0..2u32 {
        for ky in 0..2u32 {
            for kx in 0..2u32 {
                let sib_key = ChunkKey::new(0, kx, ky, kz);
                let sib_state = cache.state_or_fetch(sib_key);
                assert!(
                    sib_state.as_resident().is_some(),
                    "sibling {:?} should be resident from disk: {:?}",
                    sib_key,
                    sib_state
                );
                let marker = (kz * 4 + ky * 2 + kx) as u8;
                assert_eq!(
                    sib_state.as_resident().unwrap()[0],
                    marker,
                    "sibling {:?} should carry its own marker",
                    sib_key
                );
            }
        }
    }
    assert_eq!(
        fetch_counter.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "siblings must come from disk — no additional fetch"
    );
    assert_eq!(
        extract_counter.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "siblings must come from disk — no additional extract"
    );
}

#[test]
fn out_of_bounds_short_circuits() {
    let root = tmp_root("oob");
    let backfiller = Arc::new(SyntheticBackfiller::new("test", [64, 64, 64], 0, |_, _, _, _| 7));
    let cache = UnifiedCache::for_cache_dir(&root).open_volume(backfiller);

    // (chunk_x=2) at LOD 0 covers voxels 128..192 — past the extent.
    let key = ChunkKey::new(0, 2, 0, 0);
    let state = cache.state_or_fetch(key);
    assert!(matches!(state.as_ref(), ChunkState::CooldownMiss { .. }));
}

#[test]
fn paint_renders_synthetic_pattern() {
    let root = tmp_root("paint");
    // Pattern: gray = x & 0xff. We paint an XY slab through the middle.
    let backfiller = Arc::new(SyntheticBackfiller::new("test", [128, 128, 128], 0, |x, _, _, _| {
        (x & 0xff) as u8
    }));
    let cache = UnifiedCache::for_cache_dir(&root).open_volume(backfiller);
    let volume = UnifiedVolume::new(cache.clone());

    // Warm: explicitly wait for the chunk we'll paint from.
    for cx in 0..2 {
        for cy in 0..2 {
            cache.wait_for(ChunkKey::new(0, cx, cy, 0), Duration::from_secs(2));
        }
    }

    let mut img = Image::new(64, 64);
    let cfg = DrawingConfig::default();
    // Paint at (64, 64, 32): XY plane (u=0, v=1, plane=2), zoom=1, sfactor=1.
    volume.paint([64, 64, 32], 0, 1, 2, 64, 64, 1, 1, &cfg, &mut img);

    // Pixel (0, 0) corresponds to world (32, 32, 32) → gray = 32.
    let p00 = img.data[0];
    assert_eq!(p00.r(), 32, "expected gray=32, got {:?}", p00);
    // Pixel (63, 0) corresponds to world (95, 32, 32) → gray = 95.
    let p63 = img.data[63];
    assert_eq!(p63.r(), 95, "expected gray=95, got {:?}", p63);
}

#[test]
fn paint_no_gaps_with_pzoom_misaligned_at_chunk_edge() {
    // Reproduces the chunk-boundary grid lines: paint_zoom=2 with min_uc set
    // so (chunk_world - min_uc) is odd. The boundary pixel was previously
    // dropped because u_px_hi used floor.
    let root = tmp_root("paint-gridlines");
    let backfiller = Arc::new(SyntheticBackfiller::new("test", [512, 512, 512], 2, |_, _, _, _| 0xab));
    let cache = UnifiedCache::for_cache_dir(&root).open_volume(backfiller);
    let volume = UnifiedVolume::new(cache.clone());

    for cx in 0..3 {
        for cy in 0..3 {
            cache.wait_for(ChunkKey::new(1, cx, cy, 0), Duration::from_secs(2));
        }
    }

    let mut img = Image::new(64, 64);
    let cfg = DrawingConfig::default();
    // paint_zoom=2, sfactor=2 → lod=1, scale=2, chunk_world=128.
    // min_uc = xyz[0] - canvas/2 * pzoom = 129 - 64 = 65 (odd).
    volume.paint([129, 129, 16], 0, 1, 2, 64, 64, 2, 2, &cfg, &mut img);

    for (i, px) in img.data.iter().enumerate() {
        assert_eq!(px.r(), 0xab, "pixel {} = {:?}, expected 0xab", i, px);
    }
}

#[test]
fn paint_no_gaps_at_higher_lod() {
    // At LOD 1, each cache sample covers 2 world voxels. If we step by sample
    // we'd leave every odd pixel black. Verify every pixel is set.
    let root = tmp_root("paint-lod1");
    let backfiller = Arc::new(SyntheticBackfiller::new(
        "test",
        [256, 256, 256],
        2,
        // Pattern: constant non-zero so any "skipped" pixel stands out.
        |_, _, _, _| 0x42,
    ));
    let cache = UnifiedCache::for_cache_dir(&root).open_volume(backfiller);
    let volume = UnifiedVolume::new(cache.clone());

    for cx in 0..2 {
        for cy in 0..2 {
            cache.wait_for(ChunkKey::new(1, cx, cy, 0), Duration::from_secs(2));
        }
    }

    let mut img = Image::new(64, 64);
    let cfg = DrawingConfig::default();
    // sfactor=2 → lod=1. paint_zoom=1 so world step matches pixel step.
    volume.paint([64, 64, 16], 0, 1, 2, 64, 64, 2, 1, &cfg, &mut img);

    for (i, px) in img.data.iter().enumerate() {
        assert_eq!(px.r(), 0x42, "pixel {} = {:?}, expected 0x42", i, px);
    }
}

#[test]
fn paint_falls_back_to_coarser_lod_when_target_missing() {
    // Refuse all LOD-0 chunks (Permanent → CooldownMiss in the cache).
    // Coarser LODs return a marker byte equal to `0x10 + lod`. Pre-warm the
    // LOD-1 chunk that covers the viewport; then paint at sfactor=1
    // (target_lod=0). Every pixel must be 0x11 — proof we sampled from the
    // coarser parent because the target chunk isn't resident.
    struct LodGated {
        extent: [u32; 3],
        max_lod: u8,
    }
    impl ChunkBackfiller for LodGated {
        fn max_lod(&self) -> u8 {
            self.max_lod
        }
        fn voxel_extent(&self) -> [u32; 3] {
            self.extent
        }
        fn volume_id(&self) -> String {
            "lod-gated".into()
        }
        fn plan(&self, key: ChunkKey) -> Result<BackfillPlan, BackfillError> {
            if key.lod == 0 {
                return Err(BackfillError::Permanent("no L0".into()));
            }
            let marker = 0x10u8 + key.lod;
            let extract =
                Box::new(move |_inputs: &[_]| Ok(vec![(key, ExtractedChunk::Bytes(vec![marker; CHUNK_VOXELS]))]));
            Ok(BackfillPlan {
                covered: vec![key],
                sources: Vec::new(),
                extract,
            })
        }
    }

    let root = tmp_root("paint-lod-fallback");
    let backfiller = Arc::new(LodGated { extent: [256, 256, 256], max_lod: 2 });
    let cache = UnifiedCache::for_cache_dir(&root).open_volume(backfiller);
    let volume = UnifiedVolume::new(cache.clone());

    // Pre-warm the LOD-1 chunk covering the viewport.
    let s1 = cache.wait_for(ChunkKey::new(1, 0, 0, 0), Duration::from_secs(2));
    assert!(s1.as_resident().is_some(), "L1 should be resident: {:?}", s1);

    let mut img = Image::new(64, 64);
    let cfg = DrawingConfig::default();
    // sfactor=1 → target_lod=0. paint_zoom=1.
    volume.paint([64, 64, 16], 0, 1, 2, 64, 64, 1, 1, &cfg, &mut img);

    for (i, px) in img.data.iter().enumerate() {
        assert_eq!(px.r(), 0x11, "pixel {} = {:?}, expected 0x11 (L1 fallback)", i, px);
    }
}

#[test]
fn get_falls_back_to_coarser_lod_when_target_missing() {
    // Same setup as the paint test: LOD 0 chunks refused, LOD 1 returns 0x11.
    // VoxelVolume::get must return the LOD-1 byte when the target chunk
    // isn't resident — and must do so without any caller pre-warming the
    // coarser LOD, since surface/PPM renderers reach `get()` without going
    // through `UnifiedVolume::paint`.
    struct LodGated;
    impl ChunkBackfiller for LodGated {
        fn max_lod(&self) -> u8 {
            2
        }
        fn voxel_extent(&self) -> [u32; 3] {
            [256, 256, 256]
        }
        fn volume_id(&self) -> String {
            "lod-gated-get".into()
        }
        fn plan(&self, key: ChunkKey) -> Result<BackfillPlan, BackfillError> {
            if key.lod == 0 {
                return Err(BackfillError::Permanent("no L0".into()));
            }
            let marker = 0x10u8 + key.lod;
            let extract =
                Box::new(move |_inputs: &[_]| Ok(vec![(key, ExtractedChunk::Bytes(vec![marker; CHUNK_VOXELS]))]));
            Ok(BackfillPlan {
                covered: vec![key],
                sources: Vec::new(),
                extract,
            })
        }
    }

    let root = tmp_root("get-lod-fallback");
    let cache = UnifiedCache::for_cache_dir(&root).open_volume(Arc::new(LodGated));
    let volume = UnifiedVolume::new(cache.clone());

    // No manual pre-warm: `get()` itself must kick the coarser-LOD fetch.
    // Poll until the dispatched L1 chunk lands (the synthetic backfiller is
    // near-instant but still asynchronous).
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    loop {
        if volume.get([42.0, 17.0, 9.0], 1) == 0x11 {
            break;
        }
        assert!(std::time::Instant::now() < deadline, "L1 fallback never resolved");
        std::thread::sleep(Duration::from_millis(5));
    }

    // Same target chunk hits the hot slot; a different target chunk that
    // happens to share the L1 parent re-walks the pyramid and lands on L1
    // again (L1 (0,0,0,0) covers x∈[0,128), y,z∈[0,128)).
    assert_eq!(volume.get([43.0, 18.0, 10.0], 1), 0x11);
    assert_eq!(volume.get([100.0, 50.0, 12.0], 1), 0x11);
}

#[test]
fn get_uses_downsampled_xyz_convention_at_sfactor_gt_1() {
    // VoxelVolume::get takes `xyz` in voxel coords at the requested
    // downsampling (the convention used by VolumeGrid64x4Mapped, ZarrContext,
    // and the ObjVolume / PPMVolume callers, which pre-divide world coords
    // by sfactor before calling get). The cache MUST NOT re-divide by scale,
    // or surface painting at zoom < 1 (sfactor ≥ 2) ends up looking at the
    // wrong 3D position — visible as "no coarser-LOD fallback at zoom < 1".
    //
    // Encoding: a chunk at LOD L, x-index X carries marker (L << 4) | X.
    // LOD 1 is refused so the sfactor=2 path must reach LOD 2 via fallback.
    struct PositionMarked;
    impl ChunkBackfiller for PositionMarked {
        fn max_lod(&self) -> u8 {
            3
        }
        fn voxel_extent(&self) -> [u32; 3] {
            [1024, 1024, 1024]
        }
        fn volume_id(&self) -> String {
            "position-marked".into()
        }
        fn plan(&self, key: ChunkKey) -> Result<BackfillPlan, BackfillError> {
            if key.lod == 1 {
                return Err(BackfillError::Permanent("no L1".into()));
            }
            let marker = (key.lod << 4) | (key.x as u8 & 0x0f);
            let extract =
                Box::new(move |_inputs: &[_]| Ok(vec![(key, ExtractedChunk::Bytes(vec![marker; CHUNK_VOXELS]))]));
            Ok(BackfillPlan {
                covered: vec![key],
                sources: Vec::new(),
                extract,
            })
        }
    }

    let root = tmp_root("get-coord-convention");
    let cache = UnifiedCache::for_cache_dir(&root).open_volume(Arc::new(PositionMarked));
    let volume = UnifiedVolume::new(cache.clone());

    // sfactor=2 → target_lod=1. xyz=[200, 5, 5] is in LOD-1 coords:
    //   correct: shift to LOD 2 → (100, 2, 2) → LOD-2 chunk x=1 → 0x21.
    //   broken (double-divide): scale away → LOD-2 chunk x=0 → 0x20.
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    loop {
        if volume.get([200.0, 5.0, 5.0], 2) == 0x21 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "L2 fallback for sfactor=2 never resolved at the right chunk"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    // Same hot-slot target chunk: a nearby coord reuses the chosen L2 chunk.
    assert_eq!(volume.get([201.0, 6.0, 6.0], 2), 0x21);

    // Different target chunk → re-walks; xyz=[50, 5, 5] at LOD 1 →
    // LOD-2 coord (25, 2, 2) → chunk x=0 → marker 0x20.
    cache.wait_for(ChunkKey::new(2, 0, 0, 0), Duration::from_secs(2));
    assert_eq!(volume.get([50.0, 5.0, 5.0], 2), 0x20);
}

#[test]
fn synth_lod_one_level_above_native_averages_children() {
    // Native backfiller exposes only LOD 0; per-chunk constant value =
    // dz*4 + dy*2 + dx (0..=7). Wrap with SynthesizedLodBackfiller for one
    // extra level → cache.max_lod() == 1.
    //
    // A synthesized LOD-1 chunk at (0,0,0) covers the world region spanned
    // by LOD-0 chunks (dx,dy,dz) for d{x,y,z} ∈ {0,1}. Each output octant
    // sits over exactly one of those children, and that child is uniformly
    // filled, so each octant of the synth chunk should be uniformly filled
    // with that child's constant value.
    struct PerChunkConst;
    impl ChunkBackfiller for PerChunkConst {
        fn max_lod(&self) -> u8 {
            0
        }
        fn voxel_extent(&self) -> [u32; 3] {
            [256, 256, 256]
        }
        fn volume_id(&self) -> String {
            "per-chunk-const".into()
        }
        fn plan(&self, key: ChunkKey) -> Result<BackfillPlan, BackfillError> {
            let marker = (key.z as u8) * 4 + (key.y as u8) * 2 + (key.x as u8);
            let extract =
                Box::new(move |_inputs: &[_]| Ok(vec![(key, ExtractedChunk::Bytes(vec![marker; CHUNK_VOXELS]))]));
            Ok(BackfillPlan {
                covered: vec![key],
                sources: Vec::new(),
                extract,
            })
        }
    }

    let root = tmp_root("synth-l1");
    let inner: Arc<dyn ChunkBackfiller> = Arc::new(PerChunkConst);
    let synth = Arc::new(SynthesizedLodBackfiller::with_extra_levels(inner, 1));
    let cache = UnifiedCache::for_cache_dir(&root).open_volume(synth);
    assert_eq!(cache.max_lod(), 1);

    let state = cache.wait_for(ChunkKey::new(1, 0, 0, 0), Duration::from_secs(5));
    let mmap = state
        .as_resident()
        .unwrap_or_else(|| panic!("synth L1 chunk should be resident: {:?}", state));

    // Sample one voxel from each octant.
    let probe = |ox: usize, oy: usize, oz: usize| {
        let off = oz * CHUNK_SIDE * CHUNK_SIDE + oy * CHUNK_SIDE + ox;
        mmap[off]
    };
    // (0,0,0) → child (0,0,0) → marker 0.
    assert_eq!(probe(10, 10, 10), 0);
    // (40, 10, 10) → child (1,0,0) → marker 1.
    assert_eq!(probe(40, 10, 10), 1);
    // (10, 40, 10) → child (0,1,0) → marker 2.
    assert_eq!(probe(10, 40, 10), 2);
    // (40, 40, 40) → child (1,1,1) → marker 7.
    assert_eq!(probe(40, 40, 40), 7);
}

#[test]
fn synth_lod_two_levels_above_native_recurses() {
    // Native max_lod=0, extra_levels=2 → cache.max_lod()=2. The LOD-2 chunk
    // depends on 8 synthesized LOD-1 chunks, which themselves depend on 8×8
    // = 64 native LOD-0 chunks. Verifies that chunk-as-source dependencies
    // recurse correctly through the cache.
    //
    // Use a uniform native value so the expected output is also uniform:
    // averaging-of-averages preserves a constant value.
    struct Const(u8);
    impl ChunkBackfiller for Const {
        fn max_lod(&self) -> u8 {
            0
        }
        fn voxel_extent(&self) -> [u32; 3] {
            [512, 512, 512]
        }
        fn volume_id(&self) -> String {
            "synth-recurse".into()
        }
        fn plan(&self, key: ChunkKey) -> Result<BackfillPlan, BackfillError> {
            let v = self.0;
            let extract =
                Box::new(move |_inputs: &[_]| Ok(vec![(key, ExtractedChunk::Bytes(vec![v; CHUNK_VOXELS]))]));
            Ok(BackfillPlan {
                covered: vec![key],
                sources: Vec::new(),
                extract,
            })
        }
    }

    let root = tmp_root("synth-l2");
    let inner: Arc<dyn ChunkBackfiller> = Arc::new(Const(123));
    let synth = Arc::new(SynthesizedLodBackfiller::with_extra_levels(inner, 2));
    let cache = UnifiedCache::for_cache_dir(&root).open_volume(synth);
    assert_eq!(cache.max_lod(), 2);

    let state = cache.wait_for(ChunkKey::new(2, 0, 0, 0), Duration::from_secs(10));
    let mmap = state
        .as_resident()
        .unwrap_or_else(|| panic!("synth L2 chunk should be resident: {:?}", state));
    for &b in mmap.iter() {
        assert_eq!(b, 123, "uniform source averaged through 2 synth levels");
    }
}

#[test]
fn unified_volume_renders_at_target_lod_above_native_max() {
    // Drive UnifiedVolume::get at a target_lod beyond the native max so the
    // fallback walk lands on a synthesized chunk. This is the regression
    // path: at zoom small enough that target_lod > native_max, surface
    // rendering used to paint black.
    struct Const(u8);
    impl ChunkBackfiller for Const {
        fn max_lod(&self) -> u8 {
            0
        }
        fn voxel_extent(&self) -> [u32; 3] {
            [256, 256, 256]
        }
        fn volume_id(&self) -> String {
            "synth-get".into()
        }
        fn plan(&self, key: ChunkKey) -> Result<BackfillPlan, BackfillError> {
            let v = self.0;
            let extract =
                Box::new(move |_inputs: &[_]| Ok(vec![(key, ExtractedChunk::Bytes(vec![v; CHUNK_VOXELS]))]));
            Ok(BackfillPlan {
                covered: vec![key],
                sources: Vec::new(),
                extract,
            })
        }
    }

    let root = tmp_root("synth-volume-get");
    let inner: Arc<dyn ChunkBackfiller> = Arc::new(Const(200));
    let synth = Arc::new(SynthesizedLodBackfiller::with_extra_levels(inner, 1));
    let cache = UnifiedCache::for_cache_dir(&root).open_volume(synth);
    let volume = UnifiedVolume::new(cache.clone());

    // sfactor=2 → target_lod=1, which is exactly cache.max_lod(). xyz is in
    // LOD-1 coords; the LOD-1 chunk needed is the synthesized (0,0,0).
    cache.wait_for(ChunkKey::new(1, 0, 0, 0), Duration::from_secs(5));
    assert_eq!(volume.get([10.0, 10.0, 10.0], 2), 200);
}

#[test]
fn synth_gate_disables_when_source_has_too_many_native_chunks() {
    // Budget = 32, inner has a single native LOD over a huge extent
    // (1024³ voxels at LOD 0 → 16³ = 4096 native chunks). That's way over
    // budget, so synthesis must be disabled — `cache.max_lod()` should
    // report the inner's max_lod unchanged and chunks above it should
    // cooldown-miss as if no wrapper were present.
    struct WideSingleLod;
    impl ChunkBackfiller for WideSingleLod {
        fn max_lod(&self) -> u8 {
            0
        }
        fn voxel_extent(&self) -> [u32; 3] {
            [1024, 1024, 1024]
        }
        fn volume_id(&self) -> String {
            "wide-single-lod".into()
        }
        fn plan(&self, key: ChunkKey) -> Result<BackfillPlan, BackfillError> {
            let extract =
                Box::new(move |_inputs: &[_]| Ok(vec![(key, ExtractedChunk::Bytes(vec![55u8; CHUNK_VOXELS]))]));
            Ok(BackfillPlan {
                covered: vec![key],
                sources: Vec::new(),
                extract,
            })
        }
    }

    let root = tmp_root("synth-gate-off");
    let inner: Arc<dyn ChunkBackfiller> = Arc::new(WideSingleLod);
    let synth = Arc::new(SynthesizedLodBackfiller::new(inner, 32));
    let cache = UnifiedCache::for_cache_dir(&root).open_volume(synth);
    assert_eq!(cache.max_lod(), 0, "budget exceeded → no synth levels added");

    // LOD 1 is genuinely out of bounds when synth is disabled — the cache's
    // is_out_of_bounds check refuses any key.lod > backfiller.max_lod().
    // Without the gate, we'd see a Resident chunk synthesized from 4096
    // native LOD-0 chunks; with the gate firing, it cooldown-misses
    // immediately.
    let state = cache.wait_for(ChunkKey::new(1, 0, 0, 0), Duration::from_secs(5));
    assert!(
        matches!(state.as_ref(), ChunkState::CooldownMiss { .. }),
        "expected CooldownMiss for above-max_lod key when synth is gated off, got {:?}",
        state
    );

    // The inner's native LOD 0 still works.
    let l0 = cache.wait_for(ChunkKey::new(0, 0, 0, 0), Duration::from_secs(5));
    assert!(l0.as_resident().is_some());
}

#[test]
fn synth_gate_enables_when_source_is_pyramidal_enough() {
    // Inner reports a coarsest level where the whole volume is 2x2x2 = 8
    // native chunks — well under budget=32. With synthesis enabled, the
    // wrapper exposes one extra level on top.
    struct SmallPyramid;
    impl ChunkBackfiller for SmallPyramid {
        fn max_lod(&self) -> u8 {
            3
        }
        fn voxel_extent(&self) -> [u32; 3] {
            // 2 chunks per axis at LOD 3 (each chunk covers 512 world voxels).
            [1024, 1024, 1024]
        }
        fn volume_id(&self) -> String {
            "small-pyramid".into()
        }
        fn plan(&self, key: ChunkKey) -> Result<BackfillPlan, BackfillError> {
            let extract =
                Box::new(move |_inputs: &[_]| Ok(vec![(key, ExtractedChunk::Bytes(vec![77u8; CHUNK_VOXELS]))]));
            Ok(BackfillPlan {
                covered: vec![key],
                sources: Vec::new(),
                extract,
            })
        }
    }

    let root = tmp_root("synth-gate-on");
    let inner: Arc<dyn ChunkBackfiller> = Arc::new(SmallPyramid);
    let synth = Arc::new(SynthesizedLodBackfiller::new(inner, 32));
    let cache = UnifiedCache::for_cache_dir(&root).open_volume(synth);
    assert_eq!(cache.max_lod(), 4, "8 native chunks ≤ 32 → 1 synth level added");

    let state = cache.wait_for(ChunkKey::new(4, 0, 0, 0), Duration::from_secs(5));
    let mmap = state
        .as_resident()
        .unwrap_or_else(|| panic!("synth L4 should be resident: {:?}", state));
    assert!(mmap.iter().all(|&b| b == 77), "uniform source averages to 77");
}

#[test]
fn get_interpolated_blends_corners_within_a_chunk() {
    // x-linear gradient: gray = x. Trilinear-interp at a fractional x should
    // produce the same gradient (the y / z corners are equal so the blend
    // collapses to plain linear interpolation in x).
    let root = tmp_root("interp-fast");
    let backfiller = Arc::new(SyntheticBackfiller::new(
        "interp-fast",
        [128, 128, 128],
        0,
        |x, _, _, _| (x & 0xff) as u8,
    ));
    let cache = UnifiedCache::for_cache_dir(&root).open_volume(backfiller);
    let volume = UnifiedVolume::new(cache.clone());
    cache.wait_for(ChunkKey::new(0, 0, 0, 0), Duration::from_secs(2));

    // Fast path: all 8 corners in chunk (0,0,0) — (10, 10, 10) → (11, 11, 11).
    // Expected = 10 * (1 - 0.25) + 11 * 0.25 = 10.25 → cast u8 = 10.
    let v = volume.get_interpolated([10.25, 10.0, 10.0], 1);
    assert_eq!(v, 10);

    // Half-blend on x — 10 * 0.5 + 11 * 0.5 = 10.5 → cast u8 = 10.
    let v = volume.get_interpolated([10.5, 10.0, 10.0], 1);
    assert_eq!(v, 10);

    // No fractional component — must equal the raw voxel.
    let v = volume.get_interpolated([20.0, 30.0, 40.0], 1);
    assert_eq!(v, 20);

    // get_color_interpolated wraps the same value as a gray Color32.
    let c = volume.get_color_interpolated([10.5, 10.0, 10.0], 1);
    assert_eq!((c.r(), c.g(), c.b()), (10, 10, 10));
}

#[test]
fn get_interpolated_crosses_chunk_boundary() {
    // Same gradient, but interpolate across the +x face of chunk (0,0,0) →
    // (1,0,0). With pattern gray = x, expected interp(63.5) = 63.5 → 63.
    let root = tmp_root("interp-slow");
    let backfiller = Arc::new(SyntheticBackfiller::new(
        "interp-slow",
        [256, 256, 256],
        0,
        |x, _, _, _| (x & 0xff) as u8,
    ));
    let cache = UnifiedCache::for_cache_dir(&root).open_volume(backfiller);
    let volume = UnifiedVolume::new(cache.clone());
    cache.wait_for(ChunkKey::new(0, 0, 0, 0), Duration::from_secs(2));
    cache.wait_for(ChunkKey::new(0, 1, 0, 0), Duration::from_secs(2));

    // x0 = 63, x1 = 64 — x0 & 63 == 63, so this hits the slow path.
    let v = volume.get_interpolated([63.5, 10.0, 10.0], 1);
    assert_eq!(v, 63);

    // x0 = 63, dx = 0.0 — degenerate "boundary but no blend" case still slow.
    let v = volume.get_interpolated([63.0, 10.0, 10.0], 1);
    assert_eq!(v, 63);
}

#[test]
fn get_interpolated_at_coarser_lod_blends_distinct_coarse_voxels() {
    // Regression: when the chosen LOD is coarser than the target LOD,
    // sampling 8 corners at *target-LOD* positions `(target_sx, target_sx+1, …)`
    // collapses adjacent corners onto the same coarse voxel (since
    // `(target_sx + 1) >> shift == target_sx >> shift` for most positions).
    // The interpolation then averages duplicates and outputs flat bands —
    // values jumping in steps of `2^shift` target voxels.
    //
    // The fix shifts the interpolation lattice into the chosen LOD's
    // coordinate space: corners are sampled at coarse voxel positions
    // `(cx0, cx0+1)` with the fractional weight `frac(xyz[0] / 2^shift)`.
    //
    // LOD-0 is refused, LOD-1 carries the gradient `gray = (x * 16) & 0xff`.
    // At downsampling=1 (target_lod=0), interpolating over the +x gradient
    // must produce smooth output, not bands of width 2.
    struct LodGated;
    impl ChunkBackfiller for LodGated {
        fn max_lod(&self) -> u8 {
            1
        }
        fn voxel_extent(&self) -> [u32; 3] {
            [128, 128, 128]
        }
        fn volume_id(&self) -> String {
            "interp-coarse-blend".into()
        }
        fn plan(&self, key: ChunkKey) -> Result<BackfillPlan, BackfillError> {
            if key.lod == 0 {
                return Err(BackfillError::Permanent("no L0".into()));
            }
            let extract = Box::new(move |_inputs: &[_]| {
                let mut out = vec![0u8; CHUNK_VOXELS];
                for z in 0..CHUNK_SIDE {
                    for y in 0..CHUNK_SIDE {
                        for x in 0..CHUNK_SIDE {
                            let sx = key.x * CHUNK_SIDE as u32 + x as u32;
                            out[z * CHUNK_SIDE * CHUNK_SIDE + y * CHUNK_SIDE + x] = ((sx * 16) & 0xff) as u8;
                        }
                    }
                }
                Ok(vec![(key, ExtractedChunk::Bytes(out))])
            });
            Ok(BackfillPlan {
                covered: vec![key],
                sources: Vec::new(),
                extract,
            })
        }
    }

    let root = tmp_root("interp-coarse-blend");
    let cache = UnifiedCache::for_cache_dir(&root).open_volume(Arc::new(LodGated));
    let volume = UnifiedVolume::new(cache.clone());
    cache.wait_for(ChunkKey::new(1, 0, 0, 0), Duration::from_secs(2));

    // xyz=(10.5, 10, 10) at target_lod=0 falls back to lod_use=1 (shift=1).
    // Coarse lattice: cx0_f = 5.25 → cx0=5, cdx=0.25.
    //   p000 = 5 * 16 = 80, p100 = 6 * 16 = 96.
    //   interp = 80 * 0.75 + 96 * 0.25 = 84.
    // Pre-fix behavior would sample target-LOD corners (10, 11), both
    // mapping to coarse x=5 → both 80, output 80.
    let v = volume.get_interpolated([10.5, 10.0, 10.0], 1);
    assert_eq!(v, 84, "expected smooth interp 84, got {} (coarse-LOD banding?)", v);

    // xyz=(11.5, 10, 10): cx0_f = 5.75 → cx0=5, cdx=0.75.
    //   interp = 80 * 0.25 + 96 * 0.75 = 92.
    // Pre-fix behavior at this position happens to sample corners (11, 12)
    // that DO straddle a coarse boundary (5 vs 6), so it would output 88
    // (= mean of 80 and 96) — still wrong, just less so.
    let v = volume.get_interpolated([11.5, 10.0, 10.0], 1);
    assert_eq!(v, 92, "expected smooth interp 92, got {} (coarse-LOD banding?)", v);

    // Integer coarse-voxel boundary: cx0_f = 6.0, cdx=0 → output is the
    // bare coarse voxel value 96.
    let v = volume.get_interpolated([12.0, 10.0, 10.0], 1);
    assert_eq!(v, 96);

    // get_color_interpolated wraps the same value as gray.
    let c = volume.get_color_interpolated([10.5, 10.0, 10.0], 1);
    assert_eq!((c.r(), c.g(), c.b()), (84, 84, 84));
}

#[test]
fn get_interpolated_walks_lod_pyramid_when_target_missing() {
    // LOD-0 forbidden, LOD-1 returns 0x80 everywhere. Interpolation across
    // a uniform parent must just return 0x80 (and exercise the coarser-LOD
    // fast path inside `interpolate_u8`).
    struct LodGated;
    impl ChunkBackfiller for LodGated {
        fn max_lod(&self) -> u8 {
            1
        }
        fn voxel_extent(&self) -> [u32; 3] {
            [128, 128, 128]
        }
        fn volume_id(&self) -> String {
            "interp-lod".into()
        }
        fn plan(&self, key: ChunkKey) -> Result<BackfillPlan, BackfillError> {
            if key.lod == 0 {
                return Err(BackfillError::Permanent("no L0".into()));
            }
            let extract =
                Box::new(move |_inputs: &[_]| Ok(vec![(key, ExtractedChunk::Bytes(vec![0x80u8; CHUNK_VOXELS]))]));
            Ok(BackfillPlan {
                covered: vec![key],
                sources: Vec::new(),
                extract,
            })
        }
    }

    let root = tmp_root("interp-lod-fallback");
    let cache = UnifiedCache::for_cache_dir(&root).open_volume(Arc::new(LodGated));
    let volume = UnifiedVolume::new(cache.clone());

    // Spin until LOD-1 fallback resolves (no explicit pre-warm — the get
    // path itself kicks the coarser-LOD fetch).
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    loop {
        if volume.get_interpolated([10.5, 10.5, 10.5], 1) == 0x80 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "L1 interp fallback never resolved"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn max_along_normal_matches_naive_baseline() {
    use crate::cache::UnifiedVolume;
    use crate::volume::VoxelVolume;

    let root = tmp_root("composite-fast");
    let backfiller = Arc::new(SyntheticBackfiller::new(
        "composite-fast",
        [256, 256, 256],
        0,
        |x, y, z, _| {
            // Hashy but deterministic per-voxel byte. Spans 0..=255 with
            // some structure so trilinear samples aren't degenerate.
            let mut h: u64 = 0xcbf29ce484222325;
            h ^= x as u64;
            h = h.wrapping_mul(0x100000001b3);
            h ^= y as u64;
            h = h.wrapping_mul(0x100000001b3);
            h ^= z as u64;
            h = h.wrapping_mul(0x100000001b3);
            (h >> 24) as u8
        },
    ));
    let cache = UnifiedCache::for_cache_dir(&root).open_volume(backfiller);

    // Pre-warm all the chunks the rays below touch.
    for cz in 0..4 {
        for cy in 0..4 {
            for cx in 0..4 {
                cache.wait_for(ChunkKey::new(0, cx, cy, cz), Duration::from_secs(5));
            }
        }
    }

    let vol = UnifiedVolume::new(cache);

    fn naive_max(vol: &UnifiedVolume, base: [f64; 3], dir: [f64; 3], w_lo: f64, w_hi: f64) -> u8 {
        let n = (w_hi - w_lo) as i32;
        let mut acc = 0u8;
        for k in 0..n {
            let w = w_lo + k as f64;
            let xyz = [base[0] + w * dir[0], base[1] + w * dir[1], base[2] + w * dir[2]];
            let v = vol.get_interpolated(xyz, 1);
            if v > acc {
                acc = v;
            }
        }
        acc
    }

    // A grid of starting points that includes chunk boundaries (every 64),
    // and a mix of normal directions that exercise:
    //   - the all-inside-one-chunk fast path
    //   - chunk crossings mid-run
    //   - boundary-on-entry (floor(p) lands on row 63)
    //   - axis-aligned rays where d = 0 on two axes
    let cases: &[([f64; 3], [f64; 3])] = &[
        // Pure +X.
        ([100.2, 50.3, 30.4], [1.0, 0.0, 0.0]),
        // Mostly +X but crosses Y chunk boundary mid-ray.
        ([100.2, 63.4, 30.4], [0.7, 0.7, 0.1]),
        // Diagonal that crosses several chunk boundaries.
        ([60.0, 60.0, 60.0], [0.577, 0.577, 0.577]),
        // Boundary-on-entry: floor(px) lands on row 63 of the chunk.
        ([63.5, 50.0, 50.0], [0.5, 0.5, 0.0]),
        // Negative-direction crossings.
        ([100.0, 100.0, 100.0], [-0.8, -0.4, -0.2]),
        // Mixed: small steps so the run is long.
        ([80.0, 80.0, 80.0], [0.1, 0.05, -0.07]),
    ];

    for (i, (base, dir)) in cases.iter().enumerate() {
        // Normalize so each direction is a real unit vector.
        let l = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        let dir = [dir[0] / l, dir[1] / l, dir[2] / l];
        for &w_lo in &[-12.0_f64, -6.0, 0.0, 5.0] {
            let w_hi = w_lo + 25.0;
            let expected = naive_max(&vol, *base, dir, w_lo, w_hi);
            let got = vol.max_along_normal(*base, dir, w_lo, w_hi, 1);
            assert_eq!(
                got, expected,
                "case {}: base={:?} dir={:?} w in [{}, {})",
                i, base, dir, w_lo, w_hi
            );
        }
    }
}

/// Regression test for the dark seam at shard boundaries. The composite
/// fast path used to treat a +1 trilinear neighbor that fell in the *next*
/// shard as 0, dragging boundary samples toward black — invisible at random
/// ray directions (~0.04%) but a continuous dark line wherever a surface
/// runs parallel to a shard plane. With a 2-chunk shard side (128 voxels),
/// the boundaries land at z/x/y = 128, 256, 384, so the rays below straddle
/// them and the fast path must match the cross-shard-correct naive baseline.
#[test]
fn composite_fast_path_has_no_shard_boundary_seam() {
    use crate::cache::UnifiedVolume;
    use crate::volume::VoxelVolume;

    let root = tmp_root("shard-seam");
    let backfiller = Arc::new(SyntheticBackfiller::new(
        "shard-seam",
        [512, 512, 512],
        0,
        |x, y, z, _| {
            let mut h: u64 = 0xcbf29ce484222325;
            h ^= x as u64;
            h = h.wrapping_mul(0x100000001b3);
            h ^= y as u64;
            h = h.wrapping_mul(0x100000001b3);
            h ^= z as u64;
            h = h.wrapping_mul(0x100000001b3);
            (h >> 24) as u8
        },
    ));
    // 2-chunk shard side → shard boundaries at every 128 voxels.
    let cache = UnifiedCache::for_cache_dir(&root).open_volume_with_shard_chunks_per_axis(backfiller, 2);
    assert_eq!(cache.shard_chunks_per_axis(), 2);

    // Pre-warm the 8³ chunk grid the rays cross.
    for cz in 0..8 {
        for cy in 0..8 {
            for cx in 0..8 {
                cache.wait_for(ChunkKey::new(0, cx, cy, cz), Duration::from_secs(5));
            }
        }
    }

    let vol = UnifiedVolume::new(cache);

    fn naive_max(vol: &UnifiedVolume, base: [f64; 3], dir: [f64; 3], w_lo: f64, w_hi: f64) -> u8 {
        let n = (w_hi - w_lo) as i32;
        let mut acc = 0u8;
        for k in 0..n {
            let w = w_lo + k as f64;
            let xyz = [base[0] + w * dir[0], base[1] + w * dir[1], base[2] + w * dir[2]];
            let v = vol.get_interpolated(xyz, 1);
            if v > acc {
                acc = v;
            }
        }
        acc
    }

    let cases: &[([f64; 3], [f64; 3])] = &[
        // Crossings perpendicular to each shard plane (z=128, x=256, y=384).
        ([70.3, 90.7, 120.4], [0.0, 0.0, 1.0]),
        ([250.4, 70.2, 90.1], [1.0, 0.0, 0.0]),
        ([70.1, 378.5, 90.9], [0.0, 1.0, 0.0]),
        // Negative-direction crossing of the x=256 plane.
        ([262.0, 100.0, 100.0], [-1.0, 0.0, 0.0]),
        // Near-tangent skim of the z=256 plane — the actual seam geometry,
        // many consecutive samples sit in the last voxel-layer of the shard.
        ([90.0, 90.0, 255.4], [0.999, 0.0, 0.001]),
        ([90.0, 255.4, 90.0], [0.0, 0.999, 0.001]),
        // Diagonal through a shard corner.
        ([120.0, 120.0, 120.0], [0.577, 0.577, 0.577]),
    ];

    for (i, (base, dir)) in cases.iter().enumerate() {
        let l = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        let dir = [dir[0] / l, dir[1] / l, dir[2] / l];
        for &w_lo in &[-12.0_f64, -6.0, 0.0, 5.0] {
            let w_hi = w_lo + 40.0;
            let expected = naive_max(&vol, *base, dir, w_lo, w_hi);
            let got = vol.max_along_normal(*base, dir, w_lo, w_hi, 1);
            assert_eq!(
                got, expected,
                "case {}: base={:?} dir={:?} w in [{}, {})",
                i, base, dir, w_lo, w_hi
            );
        }
    }
}

#[test]
fn second_open_picks_up_disk_cache() {
    let root = tmp_root("persist");
    let key = ChunkKey::new(0, 0, 0, 0);

    {
        let backfiller = Arc::new(SyntheticBackfiller::new("vol", [64, 64, 64], 0, |_, _, _, _| 42));
        let cache = UnifiedCache::for_cache_dir(&root).open_volume(backfiller);
        cache.wait_for(key, Duration::from_secs(2));
        assert_eq!(cache.voxel(0, 0, 0, 0), 42);
        cache.flush();
    }

    // New cache, same volume_id + root → should hit the disk without a fetch.
    let backfiller = Arc::new(SyntheticBackfiller::new("vol", [64, 64, 64], 0, |_, _, _, _| 99));
    let cache = UnifiedCache::for_cache_dir(&root).open_volume(backfiller);
    let state = cache.state_or_fetch(key);
    // It should already be resident from disk (no worker dispatch).
    assert!(state.as_resident().is_some());
    assert_eq!(
        cache.voxel(0, 0, 0, 0),
        42,
        "disk-cached value should override new backfiller"
    );
}

#[test]
fn purge_evicts_oldest_and_preserves_survivors() {
    // Four chunks along the X axis at LOD 0; same shard so we exercise
    // the per-shard dispatched-bit clear alongside the sidecar transition.
    let root = tmp_root("purge-basic");
    let backfiller = Arc::new(SyntheticBackfiller::new(
        "purge-vol",
        [256, 64, 64],
        0,
        |x, y, z, _| ((x as u32 ^ y as u32 ^ z as u32) & 0xff) as u8,
    ));
    let cache = UnifiedCache::for_cache_dir(&root).open_volume(backfiller);

    // Fill all four chunks (chunk_x = 0..4, chunk_y/z = 0).
    let keys: Vec<ChunkKey> = (0..4).map(|cx| ChunkKey::new(0, cx, 0, 0)).collect();
    for k in &keys {
        let state = cache.wait_for(*k, Duration::from_secs(2));
        assert!(state.as_resident().is_some(), "fill failed for {}", k);
    }
    let epoch = cache.epoch_state();
    assert_eq!(epoch.total_chunks(), 4);
    let initial_epoch = epoch.current();
    assert!(initial_epoch >= 1, "fresh EpochState starts at current=1");
    assert_eq!(epoch.epoch_chunks()[initial_epoch as usize], 4);

    // Age the cache by 10 epochs and re-touch the last two chunks,
    // moving their access_epoch from `initial_epoch` to `initial_epoch + 10`.
    // The first two stay at `initial_epoch` → they're the oldest.
    epoch.force_advance(10);
    for k in &keys[2..] {
        let state = cache.state_or_fetch(*k);
        assert!(state.as_resident().is_some());
    }
    let new_epoch = epoch.current();
    assert_eq!(new_epoch, initial_epoch.wrapping_add(10));
    let hist = epoch.epoch_chunks();
    assert_eq!(hist[initial_epoch as usize], 2, "two chunks stayed at the old epoch");
    assert_eq!(hist[new_epoch as usize], 2, "two chunks moved to the current epoch");

    // Purge 2 chunks. Planner should pick a threshold that evicts
    // exactly the two old ones; the two recently-touched ones survive.
    let evicted = cache.purge_to_target(2);
    assert_eq!(evicted, 2, "expected exactly two evictions");
    assert_eq!(epoch.total_chunks(), 2);
    let hist = epoch.epoch_chunks();
    assert_eq!(hist[initial_epoch as usize], 0);
    assert_eq!(hist[new_epoch as usize], 2);

    // Victims (first two): no longer in the in-memory map, sidecar slot
    // demoted to MISSING, dispatched bit cleared for re-dispatch.
    let sidecar = cache.sidecar();
    let lod0_dims = sidecar.header.lods[0];
    for k in &keys[..2] {
        assert!(cache.peek(*k).is_none(), "victim {} should be removed from map", k);
        let idx = lod0_dims.linear_index(k.x, k.y, k.z).unwrap();
        assert_eq!(
            sidecar.get_state(0, idx),
            super::sidecar::STATE_MISSING,
            "victim {} should be MISSING in the sidecar",
            k
        );
        let (shard, in_shard_idx) = cache.locate(*k).unwrap();
        let snap = cache.peek_shard(0, shard).unwrap();
        assert!(
            !snap.dispatched.get(in_shard_idx),
            "victim {}'s dispatched bit should be cleared",
            k
        );
    }
    // Survivors: still readable, bytes match what the backfiller would
    // produce. The backfiller's deterministic checker means we can
    // verify without re-fetching.
    for k in &keys[2..] {
        let state = cache.peek(*k).expect("survivor should still be in map");
        assert!(state.as_resident().is_some(), "survivor {} should be Resident", k);
        // Sample a voxel from its 64³ slot.
        let x = k.x * 64;
        let y = k.y * 64;
        let z = k.z * 64;
        let expected = ((x ^ y ^ z) & 0xff) as u8;
        assert_eq!(cache.voxel(x, y, z, 0), expected, "survivor {} voxel mismatch", k);
    }
}

/// Shutdown flushes the sidecar synchronously so a fresh `Sidecar::load`
/// against the same root sees the writes — without waiting for the
/// per-volume sync watchdog (~10 s cadence).
///
/// Asserts: (a) shutdown persists what's in memory, (b) shutdown is
/// idempotent, (c) `shutdown_all` walks the registry.
///
/// `#[ignore]`'d by default because the synchronous shutdown path runs
/// `sync_data` on every open shard file; under the `cargo test`
/// parallel runner the I/O can starve other timing-sensitive tests
/// (`paint_falls_back_to_coarser_lod_when_target_missing` et al.).
/// Run with:
///   cargo test -p vesuvius-rs --lib cache::tests::shutdown -- --ignored
#[test]
#[ignore]
fn shutdown_flushes_sidecar_and_is_idempotent() {
    use super::sidecar::{sidecar_path, Sidecar, STATE_RESIDENT};

    let root = tmp_root("shutdown-flush");
    let backfiller = Arc::new(SyntheticBackfiller::new(
        "shutdown-vol",
        [128, 128, 128],
        0,
        |x, y, z, _| (x ^ y ^ z) as u8,
    ));
    let unified = UnifiedCache::for_cache_dir(&root);
    let cache = unified.open_volume(backfiller);

    // Materialize two chunks. wait_for guarantees they're Resident in
    // the in-memory map + the sidecar's atomic byte, but the sync
    // watchdog hasn't fired yet (10s cadence), so the on-disk sidecar
    // file is still the empty one written at construction.
    let k0 = ChunkKey::new(0, 0, 0, 0);
    let k1 = ChunkKey::new(0, 1, 0, 0);
    for k in [k0, k1] {
        let s = cache.wait_for(k, Duration::from_secs(2));
        assert!(s.as_resident().is_some(), "expected Resident, got {:?}", s);
    }

    // Drive shutdown explicitly (this is what eframe::App::on_exit
    // would do in the app).
    unified.shutdown();

    // Reload the sidecar straight from disk — it must reflect both
    // writes now that shutdown has run.
    let vol_root = unified.unified_root().join("shutdown-vol");
    let sc = Sidecar::load(&sidecar_path(&vol_root))
        .expect("sidecar load")
        .expect("sidecar present");
    let dims = sc.header.lods[0];
    let idx0 = dims.linear_index(0, 0, 0).unwrap();
    let idx1 = dims.linear_index(1, 0, 0).unwrap();
    assert_eq!(sc.get_state(0, idx0), STATE_RESIDENT, "k0 should be Resident on disk");
    assert_eq!(sc.get_state(0, idx1), STATE_RESIDENT, "k1 should be Resident on disk");

    // Idempotency: second shutdown is a cheap no-op (nothing pending).
    unified.shutdown();

    // shutdown_all() walks the process-wide registry; calling it must
    // not panic and must keep the persisted sidecar consistent.
    UnifiedCache::shutdown_all();
    let sc2 = Sidecar::load(&sidecar_path(&vol_root))
        .expect("sidecar load")
        .expect("sidecar present");
    assert_eq!(sc2.get_state(0, idx0), STATE_RESIDENT);
    assert_eq!(sc2.get_state(0, idx1), STATE_RESIDENT);
}

#[test]
fn paint_report_records_merges_and_polls_landings() {
    use super::paint_scope::{capture, MissingChunk, PaintReport};
    let root = tmp_root("paint-scope");
    let backfiller = Arc::new(SyntheticBackfiller::new("test", [256, 256, 256], 0, |x, y, z, _| {
        (x ^ y ^ z) as u8
    }));
    let cache = UnifiedCache::for_cache_dir(&root).open_volume(backfiller);
    let a = ChunkKey::new(0, 1, 0, 0);

    super::paint_scope::record(&cache, ChunkKey::new(0, 0, 0, 0), None); // no scope: dropped
    let ((), report) = capture(|| {
        super::paint_scope::record(&cache, a, Some(2));
        let ((), inner) = capture(|| super::paint_scope::record(&cache, ChunkKey::new(0, 2, 0, 0), None));
        assert_eq!(inner.missing.len(), 1);
        super::paint_scope::record(&cache, a, Some(3)); // duplicate, worse fallback
    });
    assert_eq!(report.missing.len(), 2);
    assert_eq!(report.missing[&MissingChunk { cache: cache.id(), key: a }], Some(3));
    let ((), empty) = capture(|| ());
    assert!(empty.is_complete());
    assert!(!PaintReport::failed().is_complete());
    assert!(!PaintReport::failed().poll(Duration::from_secs(60)));

    // Nothing landed since the paint: no repaint wanted.
    assert!(!report.poll(Duration::from_secs(60)));
    // A missing chunk lands: the next poll says repaint.
    cache.state_or_fetch(a);
    assert!(cache.wait_for(a, Duration::from_secs(2)).is_terminal());
    assert!(report.poll(Duration::from_secs(60)));
}
