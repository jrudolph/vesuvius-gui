use super::backfillers::synthetic::SyntheticBackfiller;
use super::*;
use crate::volume::{DrawingConfig, Image, PaintVolume};
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
