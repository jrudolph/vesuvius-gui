use super::backfiller::{BackfillError, BackfillPlan};
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
    // isn't resident.
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

    cache.wait_for(ChunkKey::new(1, 0, 0, 0), Duration::from_secs(2));

    // First call dispatches LOD 0 (→ permanent cooldown) and finds the LOD-1
    // peek resident; should already return 0x11.
    assert_eq!(volume.get([42.0, 17.0, 9.0], 1), 0x11);
    // Subsequent calls stay on the fallback path.
    assert_eq!(volume.get([100.0, 50.0, 12.0], 1), 0x11);
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
