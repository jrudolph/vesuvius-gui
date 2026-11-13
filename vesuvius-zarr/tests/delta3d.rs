//! End-to-end reads of vc-delta3d volumes written on disk the way zarr-python + vc_delta3d lay them out:
//! v3 `sharding_indexed { [bytes, vc-delta3d] }` with the index at either end (with/without crc32c), and
//! v2 with `compressor: {id: "vc-delta3d"}`.

use std::path::{Path, PathBuf};
use std::sync::Once;
use vesuvius_zarr::sharding::crc32c;
use vesuvius_zarr::ZarrArray;

const SHAPE: [usize; 3] = [200, 160, 96]; // partial shards and partial inner chunks on every axis
const INNER: usize = 64;
const SHARD: usize = 128;

fn voxel(z: usize, y: usize, x: usize) -> u8 {
    let f = ((x * 3 + y * 5 + z * 7) as f32 * 0.05).sin() * 60.0 + 128.0;
    (f as u8).wrapping_add(((x ^ y ^ z) & 3) as u8)
}

/// One full inner chunk (padded with fill value 0 outside the array), C order.
fn chunk_data(c: [usize; 3]) -> Vec<u8> {
    let mut v = Vec::with_capacity(INNER * INNER * INNER);
    for z in c[0] * INNER..(c[0] + 1) * INNER {
        for y in c[1] * INNER..(c[1] + 1) * INNER {
            for x in c[2] * INNER..(c[2] + 1) * INNER {
                let inside = z < SHAPE[0] && y < SHAPE[1] && x < SHAPE[2];
                v.push(if inside { voxel(z, y, x) } else { 0 });
            }
        }
    }
    v
}

fn n_chunks(side: usize) -> [usize; 3] {
    SHAPE.map(|s| s.div_ceil(side))
}

/// Inner chunk we deliberately leave out (sentinel index entry / missing v2 file).
const EMPTY_CHUNK: [usize; 3] = [1, 0, 1];

fn test_root(name: &str) -> PathBuf {
    static INIT: Once = Once::new();
    let base = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("vesuvius-zarr-delta3d");
    INIT.call_once(|| {
        let _ = std::fs::remove_dir_all(&base);
        // Keep the decoded-sub-chunk cache out of the user's real cache dir.
        std::env::set_var("XDG_CACHE_HOME", base.join("xdg-cache"));
    });
    let dir = base.join(name);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_v3(root: &Path, index_at_end: bool, crc: bool) {
    let index_codecs = if crc {
        r#"[{"name":"bytes","configuration":{"endian":"little"}},{"name":"crc32c"}]"#
    } else {
        r#"[{"name":"bytes","configuration":{"endian":"little"}}]"#
    };
    let json = format!(
        r#"{{"zarr_format":3,"node_type":"array","shape":[{},{},{}],"data_type":"uint8",
            "chunk_grid":{{"name":"regular","configuration":{{"chunk_shape":[{s},{s},{s}]}}}},
            "chunk_key_encoding":{{"name":"default","configuration":{{"separator":"/"}}}},
            "fill_value":0,
            "codecs":[{{"name":"sharding_indexed","configuration":{{
                "chunk_shape":[{i},{i},{i}],
                "codecs":[{{"name":"bytes","configuration":{{"endian":"little"}}}},
                          {{"name":"vc-delta3d","configuration":{{"predictor":"fast","quant":1,"future_key":1}}}}],
                "index_codecs":{index_codecs},
                "index_location":"{loc}"}}}}],
            "attributes":{{}}}}"#,
        SHAPE[0],
        SHAPE[1],
        SHAPE[2],
        s = SHARD,
        i = INNER,
        loc = if index_at_end { "end" } else { "start" },
    );
    std::fs::write(root.join("zarr.json"), json).unwrap();

    let per = SHARD / INNER;
    let n = per * per * per;
    let index_size = n * 16 + if crc { 4 } else { 0 };
    let shards = n_chunks(SHARD);
    for sz in 0..shards[0] {
        for sy in 0..shards[1] {
            for sx in 0..shards[2] {
                let mut payload = Vec::new();
                let mut entries = vec![(u64::MAX, u64::MAX); n];
                let base = if index_at_end { 0 } else { index_size as u64 };
                for (flat, entry) in entries.iter_mut().enumerate() {
                    let c = [sz * per + flat / (per * per), sy * per + (flat / per) % per, sx * per + flat % per];
                    if c == EMPTY_CHUNK || c[0] * INNER >= SHAPE[0] || c[1] * INNER >= SHAPE[1] || c[2] * INNER >= SHAPE[2]
                    {
                        continue;
                    }
                    let enc = vesuvius_delta3d::encode(&chunk_data(c), [INNER; 3]);
                    *entry = (base + payload.len() as u64, enc.len() as u64);
                    payload.extend_from_slice(&enc);
                }
                let mut index = Vec::with_capacity(index_size);
                for (o, l) in &entries {
                    index.extend_from_slice(&o.to_le_bytes());
                    index.extend_from_slice(&l.to_le_bytes());
                }
                if crc {
                    let c = crc32c(&index);
                    index.extend_from_slice(&c.to_le_bytes());
                }
                let shard_bytes = if index_at_end {
                    [payload, index].concat()
                } else {
                    [index, payload].concat()
                };
                let dir = root.join("c").join(sz.to_string()).join(sy.to_string());
                std::fs::create_dir_all(&dir).unwrap();
                std::fs::write(dir.join(sx.to_string()), shard_bytes).unwrap();
            }
        }
    }
}

fn write_v2(root: &Path) {
    let json = format!(
        r#"{{"chunks":[{i},{i},{i}],"compressor":{{"id":"vc-delta3d","quant":1,"predictor":"fast"}},
            "dtype":"|u1","fill_value":0,"filters":null,"order":"C",
            "shape":[{},{},{}],"zarr_format":2}}"#,
        SHAPE[0],
        SHAPE[1],
        SHAPE[2],
        i = INNER
    );
    std::fs::write(root.join(".zarray"), json).unwrap();
    let n = n_chunks(INNER);
    for z in 0..n[0] {
        for y in 0..n[1] {
            for x in 0..n[2] {
                if [z, y, x] == EMPTY_CHUNK {
                    continue;
                }
                let enc = vesuvius_delta3d::encode(&chunk_data([z, y, x]), [INNER; 3]);
                std::fs::write(root.join(format!("{z}.{y}.{x}")), enc).unwrap();
            }
        }
    }
}

/// Read every inner chunk (twice for v3, so the second pass hits the decoded cache) and compare.
fn check(array: &ZarrArray<3, u8>, passes: usize) {
    assert_eq!(array.def().chunks, vec![INNER; 3]);
    let n = n_chunks(INNER);
    for _ in 0..passes {
        for z in 0..n[0] {
            for y in 0..n[1] {
                for x in 0..n[2] {
                    let got = array.load_chunk([z, y, x]);
                    if [z, y, x] == EMPTY_CHUNK {
                        assert!(got.is_none(), "chunk {:?} should be empty", [z, y, x]);
                        continue;
                    }
                    let got = got.unwrap_or_else(|| panic!("chunk {:?} missing", [z, y, x]));
                    let want = chunk_data([z, y, x]);
                    assert_eq!(got.len(), want.len());
                    for (i, w) in want.iter().enumerate() {
                        assert_eq!(got.get(i), *w, "chunk {:?} voxel {}", [z, y, x], i);
                    }
                }
            }
        }
    }
}

#[test]
fn v3_sharded_index_end_crc32c() {
    let root = test_root("v3-end-crc");
    write_v3(&root, true, true);
    let array = ZarrArray::<3, u8>::from_path_auto(root.to_str().unwrap());
    check(&array, 2);
}

#[test]
fn v3_sharded_index_start_plain() {
    let root = test_root("v3-start");
    write_v3(&root, false, false);
    let array = ZarrArray::<3, u8>::from_path_auto(root.to_str().unwrap());
    check(&array, 2);
}

#[test]
#[should_panic(expected = "crc32c mismatch")]
fn v3_sharded_corrupt_index_crc_panics() {
    let root = test_root("v3-bad-crc");
    write_v3(&root, true, true);
    let shard = root.join("c/0/0/0");
    let mut bytes = std::fs::read(&shard).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    std::fs::write(&shard, bytes).unwrap();
    let array = ZarrArray::<3, u8>::from_path_auto(root.to_str().unwrap());
    array.load_chunk([0, 0, 0]);
}

#[test]
fn v2_compressor() {
    let root = test_root("v2");
    write_v2(&root);
    let array = ZarrArray::<3, u8>::from_path_auto(root.to_str().unwrap());
    check(&array, 1);
}
