//! Smoke test: open the locally-downloaded c3d-compressed OME-zarr volume,
//! read a voxel inside the materialised shard, and cross-check the value
//! against the S3-hosted v2 (uncompressed) source.
//!
//! Pass criterion is |c3d_value - v2_value| ≤ 20 (the same per-voxel max
//! abs error we observed in `vesuvius-c3d/examples/compare_against_v2.rs`
//! at target_ratio≈25).

use std::io::Read;
use vesuvius_zarr::OmeZarrContext;

const LOCAL_ROOT: &str = "/home/johannes/tmp/pap/PHercParis4/c3d-volume/20260411134726-2.400um-0.2m-78keV-masked.zarr";
const V2_URL: &str =
    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A coordinate well inside shard c/5/3/3, sub-chunk (0,0,0).
    let (z, y, x) = (20500usize, 12320usize, 12330usize);

    eprintln!("opening c3d volume at {LOCAL_ROOT}");
    let ome = OmeZarrContext::from_path(LOCAL_ROOT);
    let c3d_v = ome.get([z, y, x], 0);
    println!("c3d   value at ({z},{y},{x}) = {c3d_v}");

    // Fetch the matching v2 chunk and read the same voxel from it.
    let v2_z = z / 128;
    let v2_y = y / 128;
    let v2_x = x / 128;
    let in_z = z % 128;
    let in_y = y % 128;
    let in_x = x % 128;
    let url = format!("{V2_URL}/0/{}/{}/{}", v2_z, v2_y, v2_x);
    eprintln!("fetching v2 chunk: {url}");
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()?;
    let mut resp = client.get(&url).send()?.error_for_status()?;
    let mut buf = Vec::with_capacity(128 * 128 * 128);
    resp.read_to_end(&mut buf)?;
    if buf.len() != 128 * 128 * 128 {
        return Err(format!("v2 chunk size {} != 128^3", buf.len()).into());
    }
    let idx = (in_z * 128 + in_y) * 128 + in_x;
    let v2_v = buf[idx];
    println!("v2    value at ({z},{y},{x}) = {v2_v}");

    let diff = (c3d_v as i32 - v2_v as i32).abs();
    println!("|diff| = {diff}");
    if diff > 20 {
        return Err(format!("c3d/v2 mismatch too large at ({z},{y},{x}): {diff}").into());
    }
    Ok(())
}
