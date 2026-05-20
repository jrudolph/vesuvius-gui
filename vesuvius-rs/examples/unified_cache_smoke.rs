//! Standalone convergence smoke test for the unified cache + OmeZarr
//! backfiller. Points at a real remote OME-Zarr volume, requests one cache
//! chunk near the volume centre at LOD 0, and polls until it's Resident or
//! the timeout expires. Prints state transitions and a few sample voxels.
//!
//! Usage:
//!   RUST_LOG=info cargo run -p vesuvius-rs --example unified_cache_smoke

use std::sync::Arc;
use std::time::{Duration, Instant};

use vesuvius_rs::cache::backfillers::ome_zarr::OmeZarrBackfiller;
use vesuvius_rs::cache::{ChunkBackfiller, ChunkCache, ChunkKey, ChunkState};
use vesuvius_zarr::{default_cache_dir_for_url, OmeZarrContext};

const URL: &str = "https://volumes.aws.ash2txt.org/dls/remasked/PHercParis4-20230205180739_masked.zarr";

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("opening {}", URL);
    let t0 = Instant::now();
    let ome = OmeZarrContext::from_url_blocking_to_default_cache_dir(URL);
    println!("opened in {:?} ({} multiscale levels)", t0.elapsed(), ome.zarr_contexts.len());
    for (i, ctx) in ome.zarr_contexts.iter().enumerate() {
        println!("  L{}: shape={:?} chunks={:?}", i, ctx.shape(), ctx.array().def().chunks);
    }

    let backfiller = Arc::new(OmeZarrBackfiller::from_ome("smoke-test", ome));
    let extent = backfiller.voxel_extent();
    let max_lod = backfiller.max_lod();
    println!("extent_xyz={:?} max_lod={}", extent, max_lod);

    let cache_root = std::path::PathBuf::from(default_cache_dir_for_url(URL));
    let cache = ChunkCache::new(cache_root, backfiller);

    // Burst-request a 4×4×4 grid of cache chunks at LOD 2 centred in the
    // volume. With the bounded task queue (size 4), most of these dispatches
    // will hit "queue full" on the first pass and bounce into short
    // cooldowns; the loop should drain them across multiple iterations.
    let cx0 = extent[0] / 2 / (64 * 4);
    let cy0 = extent[1] / 2 / (64 * 4);
    let cz0 = extent[2] / 2 / (64 * 4);
    let lod = 2u8.min(max_lod);
    let mut keys = Vec::new();
    for dz in 0..4 {
        for dy in 0..4 {
            for dx in 0..4 {
                keys.push(ChunkKey::new(lod, cx0 + dx, cy0 + dy, cz0 + dz));
            }
        }
    }
    println!("requesting {} cache chunks at lod={}", keys.len(), lod);

    let timeout = Duration::from_secs(180);
    let deadline = Instant::now() + timeout;
    let mut last_summary = String::new();
    loop {
        let mut resident = 0;
        let mut empty = 0;
        let mut pending = 0;
        let mut cooldown = 0;
        for k in &keys {
            let s = cache.state_or_fetch(*k);
            match s.as_ref() {
                ChunkState::Resident { .. } => resident += 1,
                ChunkState::Empty => empty += 1,
                ChunkState::Pending => pending += 1,
                ChunkState::CooldownMiss { .. } => cooldown += 1,
                ChunkState::Missing => {}
            }
        }
        let summary = format!(
            "resident={}/{} pending={} cooldown={} empty={}",
            resident,
            keys.len(),
            pending,
            cooldown,
            empty
        );
        if summary != last_summary {
            println!("  [{:>6.2}s] {}", t0.elapsed().as_secs_f32(), summary);
            last_summary = summary.clone();
        }
        if resident + empty == keys.len() {
            println!("all chunks settled after {:?} (resident={}, empty={})", t0.elapsed(), resident, empty);
            return;
        }
        if Instant::now() >= deadline {
            println!("TIMEOUT after {:?}, final: {}", t0.elapsed(), last_summary);
            std::process::exit(1);
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}
