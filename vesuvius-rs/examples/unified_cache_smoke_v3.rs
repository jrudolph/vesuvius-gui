//! Convergence smoke test for the unified cache against a remote v3
//! sharded-c3d volume. Spins up a tiny Range-aware HTTP server pointing
//! at a local OME-Zarr v3 directory, then opens it through the cache and
//! requests a small grid of chunks at LOD 2.
//!
//! Usage:
//!   RUST_LOG=info cargo run -p vesuvius-rs --example unified_cache_smoke_v3 --release
//!
//! Override the source directory with:
//!   VESUVIUS_V3_ROOT=/path/to/some.zarr cargo run ...

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use vesuvius_rs::cache::backfillers::ome_zarr::OmeZarrBackfiller;
use vesuvius_rs::cache::{ChunkBackfiller, ChunkCache, ChunkKey, ChunkState};
use vesuvius_zarr::OmeZarrContext;

const DEFAULT_ROOT: &str =
    "/home/johannes/tmp/pap/PHercParis4/c3d-volume/20260411134726-2.400um-0.2m-78keV-masked.zarr";

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let root_str = std::env::var("VESUVIUS_V3_ROOT").unwrap_or_else(|_| DEFAULT_ROOT.to_string());
    let root = PathBuf::from(&root_str);
    if !root.join("zarr.json").exists() {
        eprintln!("error: {} does not look like a zarr v3 root (no zarr.json)", root.display());
        std::process::exit(2);
    }

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind 127.0.0.1");
    let port = listener.local_addr().unwrap().port();
    let url = format!("http://127.0.0.1:{port}");
    eprintln!("serving {} on {url}", root.display());

    let root_arc = Arc::new(root);
    thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let root = root_arc.clone();
            thread::spawn(move || {
                if let Err(e) = serve_one(stream, &root) {
                    eprintln!("server: {e}");
                }
            });
        }
    });

    // Throwaway cache dir so each run exercises the cold path end-to-end.
    let cache_dir = tempdir();
    eprintln!("cache dir: {}", cache_dir.display());

    let t0 = Instant::now();
    let ome = OmeZarrContext::from_url_blocking(&url, cache_dir.to_str().unwrap());
    println!(
        "opened in {:?} ({} multiscale levels)",
        t0.elapsed(),
        ome.zarr_contexts.len()
    );
    for (i, ctx) in ome.zarr_contexts.iter().enumerate() {
        println!(
            "  L{}: shape={:?} chunks={:?} v3_remote={}",
            i,
            ctx.shape(),
            ctx.array().def().chunks,
            ctx.array().v3_remote_sharded().is_some()
        );
    }

    let backfiller = Arc::new(OmeZarrBackfiller::from_ome("smoke-test-v3", ome));
    let extent = backfiller.voxel_extent();
    let max_lod = backfiller.max_lod();
    println!("extent_xyz={:?} max_lod={}", extent, max_lod);

    let cache = ChunkCache::new(cache_dir.clone(), backfiller);

    // 4×4×4 cache chunks at LOD 0 around a known-good voxel in the
    // companion `read_c3d_remote.rs` example: (z=20500, y=12320, x=12330).
    // 64-voxel cache chunks → divide by 64 and subtract 2 to centre 4×4×4.
    let lod = 0u8.min(max_lod);
    let cx0 = (12330u32 / 64).saturating_sub(2);
    let cy0 = (12320u32 / 64).saturating_sub(2);
    let cz0 = (20500u32 / 64).saturating_sub(2);
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
        let mut pending = 0;
        let mut cooldown = 0;
        let mut empty = 0;
        for k in &keys {
            let s = cache.state_or_fetch(*k);
            match s.as_ref() {
                ChunkState::Resident(_) => resident += 1,
                ChunkState::Pending => pending += 1,
                ChunkState::CooldownMiss { .. } => cooldown += 1,
                ChunkState::Empty => empty += 1,
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
            println!(
                "all chunks settled after {:?} (resident={}, empty={})",
                t0.elapsed(),
                resident,
                empty
            );
            return;
        }
        if Instant::now() >= deadline {
            println!("TIMEOUT after {:?}, final: {}", t0.elapsed(), last_summary);
            std::process::exit(1);
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn tempdir() -> PathBuf {
    let mut p = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("vesuvius_unified_cache_v3_{pid}_{nanos}"));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn serve_one(mut stream: TcpStream, root: &std::path::Path) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("/");
    let mut range: Option<(u64, u64)> = None;
    loop {
        let mut hdr = String::new();
        let n = reader.read_line(&mut hdr)?;
        if n == 0 || hdr == "\r\n" || hdr == "\n" {
            break;
        }
        let lower = hdr.to_lowercase();
        if let Some(v) = lower.strip_prefix("range:") {
            let v = v.trim();
            if let Some(rest) = v.strip_prefix("bytes=") {
                let mut it = rest.splitn(2, '-');
                let a: u64 = it.next().unwrap_or("0").trim().parse().unwrap_or(0);
                let b: u64 = it.next().and_then(|s| s.trim().parse().ok()).unwrap_or(u64::MAX);
                range = Some((a, b));
            }
        }
    }
    if method != "GET" {
        let _ = stream
            .write_all(b"HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        return Ok(());
    }
    let local = root.join(path.trim_start_matches('/'));
    let bytes = match std::fs::read(&local) {
        Ok(b) => b,
        Err(_) => {
            let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            return Ok(());
        }
    };
    if let Some((a, b)) = range {
        let total = bytes.len() as u64;
        let b = b.min(total - 1);
        if a > b {
            let _ = stream
                .write_all(b"HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            return Ok(());
        }
        let slice = &bytes[a as usize..=b as usize];
        let mut header = String::new();
        header.push_str("HTTP/1.1 206 Partial Content\r\n");
        header.push_str(&format!("Content-Length: {}\r\n", slice.len()));
        header.push_str(&format!("Content-Range: bytes {}-{}/{}\r\n", a, b, total));
        header.push_str("Connection: close\r\n");
        header.push_str("\r\n");
        stream.write_all(header.as_bytes())?;
        stream.write_all(slice)?;
    } else {
        let mut header = String::new();
        header.push_str("HTTP/1.1 200 OK\r\n");
        header.push_str(&format!("Content-Length: {}\r\n", bytes.len()));
        header.push_str("Connection: close\r\n");
        header.push_str("\r\n");
        stream.write_all(header.as_bytes())?;
        stream.write_all(&bytes)?;
    }
    let _ = stream.shutdown(std::net::Shutdown::Both);
    Ok(())
}
