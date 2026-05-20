use super::backfiller::{BackfillError, BackfillPlan};
use super::backfillers::synthesized_lod::SynthesizedLodBackfiller;
use super::backfillers::synthetic::SyntheticBackfiller;
use super::priority::{LodView, Priority};
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
    let cache = ChunkCache::new(&root, backfiller);

    let key = ChunkKey::new(0, 0, 0, 0);
    // First touch returns Pending (or already Resident if the worker is
    // fast — both are acceptable).
    let state = cache.state_or_fetch(key);
    assert!(matches!(state.as_ref(), ChunkState::Pending | ChunkState::Resident(_)));

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
        let extract = Box::new(|_inputs: &[SourceOutcome]| -> Result<Vec<u8>, BackfillError> {
            panic!("extract must not run when every source is absent")
        });
        Ok(BackfillPlan { sources, extract })
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
    let cache = ChunkCache::new(&root, backfiller.clone());

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

    // Reopen with a fresh cache (same disk root). The `.empty` sentinel
    // should short-circuit dispatch — no second fetch.
    drop(cache);
    let backfiller2 = Arc::new(AllAbsentBackfiller {
        volume_id: "absent-test".to_string(),
        extent: [128, 128, 128],
        fetch_count: counter.clone(),
    });
    let cache2 = ChunkCache::new(&root, backfiller2);
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

#[test]
fn out_of_bounds_short_circuits() {
    let root = tmp_root("oob");
    let backfiller = Arc::new(SyntheticBackfiller::new("test", [64, 64, 64], 0, |_, _, _, _| 7));
    let cache = ChunkCache::new(&root, backfiller);

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
    let cache = ChunkCache::new(&root, backfiller);
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
    let cache = ChunkCache::new(&root, backfiller);
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
    let cache = ChunkCache::new(&root, backfiller);
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
            let extract = Box::new(move |_inputs: &[_]| Ok(vec![marker; CHUNK_VOXELS]));
            Ok(BackfillPlan { sources: Vec::new(), extract })
        }
    }

    let root = tmp_root("paint-lod-fallback");
    let backfiller = Arc::new(LodGated { extent: [256, 256, 256], max_lod: 2 });
    let cache = ChunkCache::new(&root, backfiller);
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
            let extract = Box::new(move |_inputs: &[_]| Ok(vec![marker; CHUNK_VOXELS]));
            Ok(BackfillPlan { sources: Vec::new(), extract })
        }
    }

    let root = tmp_root("get-lod-fallback");
    let cache = ChunkCache::new(&root, Arc::new(LodGated));
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
            let extract = Box::new(move |_inputs: &[_]| Ok(vec![marker; CHUNK_VOXELS]));
            Ok(BackfillPlan { sources: Vec::new(), extract })
        }
    }

    let root = tmp_root("get-coord-convention");
    let cache = ChunkCache::new(&root, Arc::new(PositionMarked));
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
            let extract = Box::new(move |_inputs: &[_]| Ok(vec![marker; CHUNK_VOXELS]));
            Ok(BackfillPlan { sources: Vec::new(), extract })
        }
    }

    let root = tmp_root("synth-l1");
    let inner: Arc<dyn ChunkBackfiller> = Arc::new(PerChunkConst);
    let synth = Arc::new(SynthesizedLodBackfiller::with_extra_levels(inner, 1));
    let cache = ChunkCache::new(&root, synth);
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
        fn plan(&self, _key: ChunkKey) -> Result<BackfillPlan, BackfillError> {
            let v = self.0;
            let extract = Box::new(move |_inputs: &[_]| Ok(vec![v; CHUNK_VOXELS]));
            Ok(BackfillPlan { sources: Vec::new(), extract })
        }
    }

    let root = tmp_root("synth-l2");
    let inner: Arc<dyn ChunkBackfiller> = Arc::new(Const(123));
    let synth = Arc::new(SynthesizedLodBackfiller::with_extra_levels(inner, 2));
    let cache = ChunkCache::new(&root, synth);
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
        fn plan(&self, _key: ChunkKey) -> Result<BackfillPlan, BackfillError> {
            let v = self.0;
            let extract = Box::new(move |_inputs: &[_]| Ok(vec![v; CHUNK_VOXELS]));
            Ok(BackfillPlan { sources: Vec::new(), extract })
        }
    }

    let root = tmp_root("synth-volume-get");
    let inner: Arc<dyn ChunkBackfiller> = Arc::new(Const(200));
    let synth = Arc::new(SynthesizedLodBackfiller::with_extra_levels(inner, 1));
    let cache = ChunkCache::new(&root, synth);
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
        fn plan(&self, _key: ChunkKey) -> Result<BackfillPlan, BackfillError> {
            let extract = Box::new(move |_inputs: &[_]| Ok(vec![55u8; CHUNK_VOXELS]));
            Ok(BackfillPlan { sources: Vec::new(), extract })
        }
    }

    let root = tmp_root("synth-gate-off");
    let inner: Arc<dyn ChunkBackfiller> = Arc::new(WideSingleLod);
    let synth = Arc::new(SynthesizedLodBackfiller::new(inner, 32));
    let cache = ChunkCache::new(&root, synth);
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
        fn plan(&self, _key: ChunkKey) -> Result<BackfillPlan, BackfillError> {
            let extract = Box::new(move |_inputs: &[_]| Ok(vec![77u8; CHUNK_VOXELS]));
            Ok(BackfillPlan { sources: Vec::new(), extract })
        }
    }

    let root = tmp_root("synth-gate-on");
    let inner: Arc<dyn ChunkBackfiller> = Arc::new(SmallPyramid);
    let synth = Arc::new(SynthesizedLodBackfiller::new(inner, 32));
    let cache = ChunkCache::new(&root, synth);
    assert_eq!(cache.max_lod(), 4, "8 native chunks ≤ 32 → 1 synth level added");

    let state = cache.wait_for(ChunkKey::new(4, 0, 0, 0), Duration::from_secs(5));
    let mmap = state
        .as_resident()
        .unwrap_or_else(|| panic!("synth L4 should be resident: {:?}", state));
    assert!(mmap.iter().all(|&b| b == 77), "uniform source averages to 77");
}

#[test]
fn viewport_priority_orders_coarse_before_fine() {
    // Same chunk position, coarse LOD should have a smaller numeric
    // priority than fine LOD. Paint composes a local Viewport and uses it
    // to compute priorities passed into state_or_fetch_with_priority.
    let mut per_lod = vec![None; 3];
    for lod in 0..=2 {
        per_lod[lod] = Some(LodView {
            center: [0, 0, 0],
            rect_lo: [0, 0, 0],
            rect_hi: [1, 1, 1],
        });
    }
    let vp = Viewport { max_lod: 2, per_lod };
    let coarse = vp.priority_for(ChunkKey::new(2, 0, 0, 0));
    let mid = vp.priority_for(ChunkKey::new(1, 0, 0, 0));
    let fine = vp.priority_for(ChunkKey::new(0, 0, 0, 0));
    assert!(coarse < mid && mid < fine, "coarse < mid < fine: {:?} {:?} {:?}", coarse, mid, fine);

    let center = vp.priority_for(ChunkKey::new(0, 0, 0, 0));
    let edge = vp.priority_for(ChunkKey::new(0, 1, 1, 1));
    assert!(center < edge, "center < edge: {:?} {:?}", center, edge);
}

#[test]
fn priority_worst_is_worse_than_any_viewport_priority() {
    // Direct-API callers that have no viewport context use Priority::worst().
    // It must be larger than any priority a published viewport produces, so
    // those calls don't preempt viewport-aware work.
    let any = Priority::new(0, 0);
    assert!(any < Priority::worst());
}

#[test]
fn second_open_picks_up_disk_cache() {
    let root = tmp_root("persist");
    let key = ChunkKey::new(0, 0, 0, 0);

    {
        let backfiller = Arc::new(SyntheticBackfiller::new("vol", [64, 64, 64], 0, |_, _, _, _| 42));
        let cache = ChunkCache::new(&root, backfiller);
        cache.wait_for(key, Duration::from_secs(2));
        assert_eq!(cache.voxel(0, 0, 0, 0), 42);
    }

    // New cache, same volume_id + root → should hit the disk without a fetch.
    let backfiller = Arc::new(SyntheticBackfiller::new("vol", [64, 64, 64], 0, |_, _, _, _| 99));
    let cache = ChunkCache::new(&root, backfiller);
    let state = cache.state_or_fetch(key);
    // It should already be resident from disk (no worker dispatch).
    assert!(state.as_resident().is_some());
    assert_eq!(
        cache.voxel(0, 0, 0, 0),
        42,
        "disk-cached value should override new backfiller"
    );
}
