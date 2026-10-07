//! Drive the cache `Downloader` against a list of URLs, through the exact
//! production stack (LIFO queue, reqwest pool, sync worker threads).
//!
//! Usage:
//!   VESUVIUS_NET_LOG=/tmp/netlog.jsonl \
//!     cargo run --release -p vesuvius-rs --example download-probe -- \
//!     [--workers N] urls.txt
//!
//! Prints wall time and aggregate throughput; the per-request breakdown
//! lands in the netlog (analyze with `scripts/analyze-netlog.py`).

use std::sync::mpsc;
use std::time::Instant;
use vesuvius_rs::cache::{ChunkKey, Downloader};

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut workers = 16usize;
    if args.len() >= 2 && args[0] == "--workers" {
        workers = args[1].parse().expect("--workers N");
        args.drain(0..2);
    }
    let path = args.first().expect("usage: download-probe [--workers N] urls.txt");
    let urls: Vec<String> = std::fs::read_to_string(path)
        .expect("read url file")
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();

    let downloader = Downloader::with_workers(workers);
    let (tx, rx) = mpsc::channel::<(usize, u64)>();

    let t0 = Instant::now();
    for (i, url) in urls.iter().enumerate() {
        let tx = tx.clone();
        // Fake chunk key — the downloader only uses it for bookkeeping.
        let key = ChunkKey::new(0, i as u32, 0, 0);
        downloader.submit(url, None, key, vesuvius_rs::cache::lease::unleased(), Box::new(move |res| {
            let bytes = match &res {
                Ok(Some(b)) => b.len() as u64,
                _ => 0,
            };
            let _ = tx.send((i, bytes));
        }));
    }
    drop(tx);

    let mut total: u64 = 0;
    let mut ok = 0usize;
    for (_, bytes) in rx.iter() {
        if bytes > 0 {
            ok += 1;
            total += bytes;
        }
    }
    let wall = t0.elapsed().as_secs_f64();
    println!(
        "{} urls ({} ok), {:.1}MB in {:.2}s -> {:.2}MB/s (workers={})",
        urls.len(),
        ok,
        total as f64 / 1e6,
        wall,
        total as f64 / wall / 1e6,
        workers
    );
}
