//! Smoke test for the v3+c3d remote (HTTP Range) pipeline.
//!
//! Spins up a tiny in-process HTTP/1.1 file server that honours `Range:
//! bytes=…` (Python's `http.server` doesn't), points it at the local c3d
//! volume directory, opens it via the new remote V3 path, and reads a voxel.
//! Cross-checks against the S3-hosted v2 source the same way
//! `read_c3d_local.rs` does.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use vesuvius_zarr::OmeZarrContext;

const LOCAL_ROOT: &str = "/home/johannes/tmp/pap/PHercParis4/c3d-volume/20260411134726-2.400um-0.2m-78keV-masked.zarr";
const V2_URL: &str =
    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(LOCAL_ROOT);
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let port = listener.local_addr()?.port();
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

    // Use a throwaway cache dir so we exercise the on-disk caching code path.
    let cache_dir = tempdir()?;
    eprintln!("cache dir: {}", cache_dir.display());

    let ome = OmeZarrContext::from_url(&url, cache_dir.to_str().unwrap());

    let (z, y, x) = (20500usize, 12320usize, 12330usize);
    let c3d_v = ome.get([z, y, x], 0);
    println!("c3d-via-http value at ({z},{y},{x}) = {c3d_v}");

    // Cross-check against the v2 source.
    let v2_chunk_url = format!("{V2_URL}/0/{}/{}/{}", z / 128, y / 128, x / 128);
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()?;
    let mut resp = client.get(&v2_chunk_url).send()?.error_for_status()?;
    let mut buf = Vec::with_capacity(128 * 128 * 128);
    resp.read_to_end(&mut buf)?;
    let in_z = z % 128;
    let in_y = y % 128;
    let in_x = x % 128;
    let v2_v = buf[(in_z * 128 + in_y) * 128 + in_x];
    println!("v2            value at ({z},{y},{x}) = {v2_v}");
    let diff = (c3d_v as i32 - v2_v as i32).abs();
    println!("|diff| = {diff}");
    if diff > 20 {
        return Err(format!("remote c3d/v2 mismatch too large at ({z},{y},{x}): {diff}").into());
    }
    Ok(())
}

fn tempdir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut p = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("vesuvius_zarr_remote_test_{pid}_{nanos}"));
    std::fs::create_dir_all(&p)?;
    Ok(p)
}

fn serve_one(mut stream: TcpStream, root: &std::path::Path) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("/");
    let mut range: Option<(u64, u64)> = None; // (start, end_inclusive)
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
        let _ = stream.write_all(b"HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
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
