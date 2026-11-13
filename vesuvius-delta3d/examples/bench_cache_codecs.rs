//! Cache-tier codec benchmark: takes a directory of vc-delta3d payloads (e.g. the unified raw store, which holds
//! downloaded ng 64³ chunks), decodes each, re-encodes with several presets and reports size and single-thread
//! throughput (MB/s of decoded voxels).
//!
//! cargo run --release -p vesuvius-delta3d --example bench_cache_codecs -- <dir> [max_files]

use std::time::{Duration, Instant};
use vesuvius_delta3d::{decode, decode_into, encode_preset, Preset};

const SIDE: usize = 64;
const VOX: usize = SIDE * SIDE * SIDE;

fn mbps(bytes: usize, d: Duration) -> f64 {
    bytes as f64 / d.as_secs_f64() / 1e6
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: bench_cache_codecs <dir> [max_files]");
    let max: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(usize::MAX);

    let mut payloads: Vec<Vec<u8>> = Vec::new();
    let mut paths: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).collect();
    paths.sort();
    for p in paths {
        if payloads.len() >= max {
            break;
        }
        let b = std::fs::read(&p).unwrap();
        match vesuvius_delta3d::decoded_size(&b) {
            Ok(VOX) => payloads.push(b),
            Ok(n) => eprintln!("skip {} (decodes to {} bytes)", p.display(), n),
            Err(_) => eprintln!("skip {} (not vc-delta3d)", p.display()),
        }
    }
    let n = payloads.len();
    assert!(n > 0, "no 64³ payloads found");
    let decoded_total = n * VOX;
    let ng_bytes: usize = payloads.iter().map(|p| p.len()).sum();

    // ng decode (source format as downloaded)
    let mut buf = vec![0u8; VOX];
    let t = Instant::now();
    for p in &payloads {
        decode_into(p, &mut buf).unwrap();
    }
    let dt = t.elapsed();
    println!("chunks: {}  ({:.1} MB decoded)", n, decoded_total as f64 / 1e6);
    println!(
        "{:<10} {:>8.2}x {:>9.0} B/chunk   enc {:>8}   dec {:>7.1} MB/s  ({:>6.2} ms/chunk)",
        "ng(src)",
        decoded_total as f64 / ng_bytes as f64,
        ng_bytes as f64 / n as f64,
        "-",
        mbps(decoded_total, dt),
        dt.as_secs_f64() * 1e3 / n as f64
    );

    let raw: Vec<Vec<u8>> = payloads.iter().map(|p| decode(p).unwrap()).collect();
    for preset in [Preset::Delta, Preset::Faster, Preset::Fast, Preset::Balanced] {
        let t = Instant::now();
        let enc: Vec<Vec<u8>> = raw.iter().map(|r| encode_preset(r, [SIDE; 3], preset)).collect();
        let et = t.elapsed();
        let bytes: usize = enc.iter().map(|e| e.len()).sum();
        let t = Instant::now();
        for (e, r) in enc.iter().zip(&raw) {
            decode_into(e, &mut buf).unwrap();
            assert!(buf == *r, "{:?} roundtrip mismatch", preset);
        }
        let dt = t.elapsed();
        println!(
            "{:<10} {:>8.2}x {:>9.0} B/chunk   enc {:>6.1} MB/s   dec {:>7.1} MB/s  ({:>6.2} ms/chunk)",
            format!("{:?}", preset),
            decoded_total as f64 / bytes as f64,
            bytes as f64 / n as f64,
            mbps(decoded_total, et),
            mbps(decoded_total, dt),
            dt.as_secs_f64() * 1e3 / n as f64
        );
    }
    // memcpy reference
    let t = Instant::now();
    for r in &raw {
        buf.copy_from_slice(r);
        std::hint::black_box(&buf);
    }
    println!("memcpy     dec {:.0} MB/s", mbps(decoded_total, t.elapsed()));
}
