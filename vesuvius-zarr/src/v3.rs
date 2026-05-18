//! Zarr v3 metadata parsing + sharded c3d access.
//!
//! Scope is deliberately narrow: only the codec chain that production c3d
//! volumes use today, which is
//!     `[ sharding_indexed { codecs: [c3d], index_codecs: [bytes], index_location: start } ]`.
//! Anything else `panic!`s at array-open time rather than failing later in an
//! opaque way. When the format broadens, extend this module.

use crate::sharding::{LocalShard, RemoteShard, ShardAccess, ShardIndex};
use crate::{parse_json, ChunkContext, ZarrArray, ZarrArrayDef, ZarrDataType, ZarrFileAccess, ZarrOrder, ZarrVersion};
use dashmap::DashMap;
use fxhash::FxBuildHasher;
use reqwest::blocking::Client;
use serde::Deserialize;
use std::sync::Arc;

#[derive(Debug, Deserialize)]
pub struct V3ArrayJson {
    pub shape: Vec<usize>,
    pub data_type: String,
    pub chunk_grid: ChunkGrid,
    pub chunk_key_encoding: ChunkKeyEncoding,
    pub codecs: Vec<V3CodecJson>,
    #[allow(dead_code)]
    pub fill_value: serde_json::Value,
    #[allow(dead_code)]
    pub node_type: String,
    pub zarr_format: u8,
}

#[derive(Debug, Deserialize)]
pub struct ChunkGrid {
    pub name: String,
    pub configuration: ChunkGridConfig,
}

#[derive(Debug, Deserialize)]
pub struct ChunkGridConfig {
    pub chunk_shape: Vec<usize>,
}

#[derive(Debug, Deserialize)]
pub struct ChunkKeyEncoding {
    pub name: String,
    #[serde(default)]
    pub configuration: Option<ChunkKeyEncodingConfig>,
}

#[derive(Debug, Deserialize)]
pub struct ChunkKeyEncodingConfig {
    #[serde(default)]
    pub separator: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct V3CodecJson {
    pub name: String,
    #[serde(default)]
    pub configuration: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct ShardingConfig {
    pub chunk_shape: Vec<usize>,
    pub codecs: Vec<V3CodecJson>,
    pub index_codecs: Vec<V3CodecJson>,
    #[serde(default)]
    pub index_location: Option<String>,
}

/// The narrow codec layout we currently support.
#[derive(Debug, Clone)]
pub struct Sharded {
    pub shard_chunk_shape: [usize; 3],
    pub sub_chunk_shape: [usize; 3],
    pub sub_chunks_per_shard: [usize; 3],
    pub n_sub_chunks_per_shard: usize,
    pub separator: String,
}

impl Sharded {
    fn from_array_json(j: &V3ArrayJson) -> Self {
        assert_eq!(j.zarr_format, 3, "v3.rs: array must be zarr_format=3");
        assert_eq!(
            j.data_type, "uint8",
            "v3.rs: only data_type=uint8 supported, got {}",
            j.data_type
        );
        assert_eq!(j.chunk_grid.name, "regular", "v3.rs: only chunk_grid=regular supported");
        assert_eq!(
            j.chunk_key_encoding.name, "default",
            "v3.rs: only chunk_key_encoding=default supported"
        );
        let separator = j
            .chunk_key_encoding
            .configuration
            .as_ref()
            .and_then(|c| c.separator.clone())
            .unwrap_or_else(|| "/".to_string());

        // Outer codec chain must be exactly [sharding_indexed].
        assert_eq!(
            j.codecs.len(),
            1,
            "v3.rs: top-level codecs must be exactly [sharding_indexed], got {} entries",
            j.codecs.len()
        );
        let outer = &j.codecs[0];
        assert_eq!(
            outer.name, "sharding_indexed",
            "v3.rs: unsupported top-level codec `{}` (only sharding_indexed today)",
            outer.name
        );
        let sharding: ShardingConfig =
            serde_json::from_value(outer.configuration.clone()).expect("v3.rs: invalid sharding_indexed configuration");

        // Inner sub-chunk codec chain must be exactly [c3d].
        assert_eq!(
            sharding.codecs.len(),
            1,
            "v3.rs: sharding.codecs must be exactly [c3d], got {} entries",
            sharding.codecs.len()
        );
        assert_eq!(
            sharding.codecs[0].name, "c3d",
            "v3.rs: unsupported sub-chunk codec `{}` (only c3d today)",
            sharding.codecs[0].name
        );

        // Index codecs must be exactly [bytes].
        assert_eq!(
            sharding.index_codecs.len(),
            1,
            "v3.rs: only single-codec [bytes] index_codecs supported, got {} entries",
            sharding.index_codecs.len()
        );
        assert_eq!(
            sharding.index_codecs[0].name, "bytes",
            "v3.rs: unsupported index codec `{}`",
            sharding.index_codecs[0].name
        );

        // Index location: only "start" supported.
        let loc = sharding.index_location.as_deref().unwrap_or("end");
        assert_eq!(loc, "start", "v3.rs: only index_location=start supported, got {}", loc);

        assert_eq!(j.shape.len(), 3, "v3.rs: only 3D arrays supported");
        assert_eq!(j.chunk_grid.configuration.chunk_shape.len(), 3);
        assert_eq!(sharding.chunk_shape.len(), 3);

        let shard_chunk_shape: [usize; 3] = j.chunk_grid.configuration.chunk_shape.clone().try_into().unwrap();
        let sub_chunk_shape: [usize; 3] = sharding.chunk_shape.clone().try_into().unwrap();
        let sub_chunks_per_shard = [
            shard_chunk_shape[0] / sub_chunk_shape[0],
            shard_chunk_shape[1] / sub_chunk_shape[1],
            shard_chunk_shape[2] / sub_chunk_shape[2],
        ];
        for i in 0..3 {
            assert_eq!(
                sub_chunks_per_shard[i] * sub_chunk_shape[i],
                shard_chunk_shape[i],
                "v3.rs: shard shape {:?} not divisible by sub-chunk shape {:?} on axis {}",
                shard_chunk_shape,
                sub_chunk_shape,
                i
            );
        }
        let n_sub_chunks_per_shard = sub_chunks_per_shard[0] * sub_chunks_per_shard[1] * sub_chunks_per_shard[2];

        // The c3d codec is fixed at 256³ sub-chunks. Loudly assert until we
        // implement a more permissive integration.
        assert_eq!(
            sub_chunk_shape,
            [
                vesuvius_c3d::C3D_CHUNK_SIDE,
                vesuvius_c3d::C3D_CHUNK_SIDE,
                vesuvius_c3d::C3D_CHUNK_SIDE
            ],
            "v3.rs: c3d codec atom is fixed at 256^3, got sub_chunk_shape {:?}",
            sub_chunk_shape
        );

        Self {
            shard_chunk_shape,
            sub_chunk_shape,
            sub_chunks_per_shard,
            n_sub_chunks_per_shard,
            separator,
        }
    }

    /// Build a v2-shaped `ZarrArrayDef` so the rest of the crate can use this
    /// array via the existing `ZarrContext` pipeline. Compression is `None`
    /// because the access impl has already decoded by the time `load_chunk`
    /// returns. `chunks` is the *sub-chunk* shape (the unit `ZarrContext`
    /// thinks in) — this is what makes the per-sub-chunk cache work.
    fn synthesize_def(&self, shape: &[usize]) -> ZarrArrayDef {
        ZarrArrayDef {
            chunks: self.sub_chunk_shape.to_vec(),
            compressor: None,
            dtype: ZarrDataType::U1,
            fill_value: 0,
            filters: None,
            order: ZarrOrder::ColumnMajor,
            shape: shape.to_vec(),
            zarr_format: ZarrVersion::V2, // synthesized v2-shaped def
            dimension_separator: Some(self.separator.clone()),
        }
    }

    /// Decompose a sub-chunk index `[cz, cy, cx]` (in sub-chunks-of-the-whole-
    /// array units) into (shard coordinates, flat sub-chunk index within shard).
    fn locate(&self, chunk_no: &[usize]) -> ([usize; 3], usize) {
        let shard = [
            chunk_no[0] / self.sub_chunks_per_shard[0],
            chunk_no[1] / self.sub_chunks_per_shard[1],
            chunk_no[2] / self.sub_chunks_per_shard[2],
        ];
        let sub = [
            chunk_no[0] % self.sub_chunks_per_shard[0],
            chunk_no[1] % self.sub_chunks_per_shard[1],
            chunk_no[2] % self.sub_chunks_per_shard[2],
        ];
        let flat = (sub[0] * self.sub_chunks_per_shard[1] + sub[1]) * self.sub_chunks_per_shard[2] + sub[2];
        (shard, flat)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Local v3 access
// ────────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct LocalV3ShardedAccess {
    array_root: String,
    sharded: Sharded,
    shape: Vec<usize>,
    // Cache parsed indices to avoid re-reading the same 64 KiB on every miss.
    // Shards themselves are mmap'd, so the shard data itself doesn't need
    // a separate in-memory cache.
    index_cache: DashMap<[usize; 3], Arc<ShardIndex>, FxBuildHasher>,
    shard_cache: DashMap<[usize; 3], Option<Arc<LocalShard>>, FxBuildHasher>,
}

impl LocalV3ShardedAccess {
    fn shard_path(&self, shard: [usize; 3]) -> std::path::PathBuf {
        let sep = &self.sharded.separator;
        let joined = format!("{}{sep}{}{sep}{}", shard[0], shard[1], shard[2]);
        std::path::PathBuf::from(&self.array_root).join("c").join(joined)
    }

    fn get_shard(&self, shard: [usize; 3]) -> Option<Arc<LocalShard>> {
        if let Some(entry) = self.shard_cache.get(&shard) {
            return entry.clone();
        }
        let path = self.shard_path(shard);
        let opened = LocalShard::open(&path);
        self.shard_cache.insert(shard, opened.clone());
        opened
    }

    fn get_index(&self, shard: [usize; 3]) -> Option<Arc<ShardIndex>> {
        if let Some(entry) = self.index_cache.get(&shard) {
            return Some(entry.clone());
        }
        let shard_handle = self.get_shard(shard)?;
        let idx_bytes = shard_handle.read_index_bytes(self.sharded.n_sub_chunks_per_shard)?;
        let parsed = Arc::new(ShardIndex::parse_bytes_codec(
            &idx_bytes,
            self.sharded.n_sub_chunks_per_shard,
        ));
        self.index_cache.insert(shard, parsed.clone());
        Some(parsed)
    }
}

impl ZarrFileAccess for LocalV3ShardedAccess {
    fn load_array_def(&self) -> ZarrArrayDef {
        self.sharded.synthesize_def(&self.shape)
    }

    fn load_chunk(&self, _def: &ZarrArrayDef, chunk_no: &[usize]) -> Option<ChunkContext> {
        let (shard, flat) = self.sharded.locate(chunk_no);
        let index = self.get_index(shard)?;
        let (off, len) = index.lookup(flat)?;
        let shard_handle = self.get_shard(shard)?;
        let compressed = shard_handle.read_range(off, len)?;
        let decoded = vesuvius_c3d::with_decoder(|d| d.decode(&compressed))
            .unwrap_or_else(|e| panic!("c3d decode failed at shard {:?} sub-chunk {}: {}", shard, flat, e));
        Some(ChunkContext::Heap(decoded))
    }

    fn cache_missing(&self) -> bool {
        true
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Entry points
// ────────────────────────────────────────────────────────────────────────────

// ────────────────────────────────────────────────────────────────────────────
// Remote v3 access
// ────────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct RemoteV3ShardedAccess {
    array_url: String,
    cache_dir: String,
    client: Client,
    sharded: Sharded,
    shape: Vec<usize>,
    // In-memory caches. The on-disk cache (under `cache_dir`) survives
    // restarts; this layer just avoids re-reading the disk file on every miss.
    index_cache: DashMap<[usize; 3], Arc<ShardIndex>, FxBuildHasher>,
    // None = we tried to fetch the shard index and the shard is missing (404).
    // Some(idx) = present. Avoids re-fetching the index for empty shards.
    index_known_missing: DashMap<[usize; 3], (), FxBuildHasher>,
}

impl RemoteV3ShardedAccess {
    fn shard_url(&self, shard: [usize; 3]) -> String {
        let sep = &self.sharded.separator;
        format!("{}/c/{}{sep}{}{sep}{}", self.array_url, shard[0], shard[1], shard[2])
    }

    fn shard_cache_dir(&self, shard: [usize; 3]) -> std::path::PathBuf {
        std::path::PathBuf::from(&self.cache_dir)
            .join("c")
            .join(format!("{}_{}_{}", shard[0], shard[1], shard[2]))
    }

    fn index_disk_path(&self, shard: [usize; 3]) -> std::path::PathBuf {
        self.shard_cache_dir(shard).join("_index.bin")
    }

    fn subchunk_disk_path(&self, shard: [usize; 3], flat: usize) -> std::path::PathBuf {
        self.shard_cache_dir(shard).join(format!("sub_{:05}.c3dc", flat))
    }

    fn missing_marker_path(&self, shard: [usize; 3]) -> std::path::PathBuf {
        self.shard_cache_dir(shard).join("_missing")
    }

    fn get_index(&self, shard: [usize; 3]) -> Option<Arc<ShardIndex>> {
        if let Some(entry) = self.index_cache.get(&shard) {
            return Some(entry.clone());
        }
        if self.index_known_missing.contains_key(&shard) {
            return None;
        }

        let disk_idx = self.index_disk_path(shard);
        let missing = self.missing_marker_path(shard);
        if missing.exists() {
            self.index_known_missing.insert(shard, ());
            return None;
        }
        if let Ok(bytes) = std::fs::read(&disk_idx) {
            let parsed = Arc::new(ShardIndex::parse_bytes_codec(
                &bytes,
                self.sharded.n_sub_chunks_per_shard,
            ));
            self.index_cache.insert(shard, parsed.clone());
            return Some(parsed);
        }

        // Range-fetch the index.
        let url = self.shard_url(shard);
        let remote = RemoteShard {
            url: url.clone(),
            client: self.client.clone(),
        };
        let bytes = match remote.read_index_bytes(self.sharded.n_sub_chunks_per_shard) {
            Some(b) => b,
            None => {
                // Persist a "missing" marker so the next process doesn't retry.
                let _ = std::fs::create_dir_all(self.shard_cache_dir(shard));
                let _ = std::fs::write(&missing, b"");
                self.index_known_missing.insert(shard, ());
                return None;
            }
        };
        let parsed = Arc::new(ShardIndex::parse_bytes_codec(
            &bytes,
            self.sharded.n_sub_chunks_per_shard,
        ));

        let _ = std::fs::create_dir_all(self.shard_cache_dir(shard));
        let tmp = disk_idx.with_extension("bin.tmp");
        if std::fs::write(&tmp, &bytes).is_ok() {
            let _ = std::fs::rename(&tmp, &disk_idx);
        }

        self.index_cache.insert(shard, parsed.clone());
        Some(parsed)
    }

    fn fetch_subchunk(&self, shard: [usize; 3], flat: usize, off: u64, len: u64) -> Option<Vec<u8>> {
        let disk_path = self.subchunk_disk_path(shard, flat);
        if let Ok(bytes) = std::fs::read(&disk_path) {
            return Some(bytes);
        }

        let url = self.shard_url(shard);
        let remote = RemoteShard {
            url: url.clone(),
            client: self.client.clone(),
        };
        let bytes = remote.read_range(off, len)?;

        let _ = std::fs::create_dir_all(self.shard_cache_dir(shard));
        let tmp = disk_path.with_extension("c3dc.tmp");
        if std::fs::write(&tmp, &bytes).is_ok() {
            let _ = std::fs::rename(&tmp, &disk_path);
        }
        Some(bytes)
    }
}

impl ZarrFileAccess for RemoteV3ShardedAccess {
    fn load_array_def(&self) -> ZarrArrayDef {
        self.sharded.synthesize_def(&self.shape)
    }

    fn load_chunk(&self, _def: &ZarrArrayDef, chunk_no: &[usize]) -> Option<ChunkContext> {
        let (shard, flat) = self.sharded.locate(chunk_no);
        let index = self.get_index(shard)?;
        let (off, len) = index.lookup(flat)?;
        let compressed = self.fetch_subchunk(shard, flat, off, len)?;
        let decoded = vesuvius_c3d::with_decoder(|d| d.decode(&compressed))
            .unwrap_or_else(|e| panic!("c3d decode failed at shard {:?} sub-chunk {}: {}", shard, flat, e));
        Some(ChunkContext::Heap(decoded))
    }

    fn cache_missing(&self) -> bool {
        true
    }
}

/// Open a v3 sharded c3d array hosted at a remote URL. Caches the array
/// `zarr.json`, the per-shard index, and the per-sub-chunk compressed bytes
/// under `cache_dir` so warm reads don't go to the network.
pub fn open_v3_array_remote(url: &str, cache_dir: &str, client: Client) -> ZarrArray<3, u8> {
    let url = url.trim_end_matches('/');
    let cache_dir = cache_dir.trim_end_matches('/');

    // Cache + parse the array zarr.json.
    let local_metadata = format!("{}/zarr.json", cache_dir);
    if !std::path::Path::new(&local_metadata).exists() {
        let metadata_url = format!("{}/zarr.json", url);
        let res = client
            .get(&metadata_url)
            .send()
            .unwrap_or_else(|e| panic!("fetch {}: {}", metadata_url, e));
        if !res.status().is_success() {
            panic!("fetch {}: status {}", metadata_url, res.status());
        }
        let bytes = res.bytes().unwrap();
        std::fs::create_dir_all(cache_dir).unwrap();
        std::fs::write(&local_metadata, &bytes).unwrap();
    }
    let raw = std::fs::read_to_string(&local_metadata).unwrap();
    let json: V3ArrayJson = parse_json(&raw, &local_metadata);
    let sharded = Sharded::from_array_json(&json);
    let shape = json.shape.clone();
    let def = sharded.synthesize_def(&shape);

    let access = Arc::new(RemoteV3ShardedAccess {
        array_url: url.to_string(),
        cache_dir: cache_dir.to_string(),
        client,
        sharded,
        shape,
        index_cache: DashMap::with_hasher(FxBuildHasher::default()),
        index_known_missing: DashMap::with_hasher(FxBuildHasher::default()),
    });
    ZarrArray::from_def_and_access(def, access)
}

pub fn open_v3_array_local(path: &str) -> ZarrArray<3, u8> {
    let json_path = format!("{}/zarr.json", path);
    let raw = std::fs::read_to_string(&json_path).unwrap_or_else(|e| panic!("read {}: {}", json_path, e));
    let json: V3ArrayJson = parse_json(&raw, &json_path);
    let sharded = Sharded::from_array_json(&json);
    let shape = json.shape.clone();
    let def = sharded.synthesize_def(&shape);

    let access = Arc::new(LocalV3ShardedAccess {
        array_root: path.trim_end_matches('/').to_string(),
        sharded,
        shape,
        index_cache: DashMap::with_hasher(FxBuildHasher::default()),
        shard_cache: DashMap::with_hasher(FxBuildHasher::default()),
    });
    ZarrArray::from_def_and_access(def, access)
}

/// Read a v3 group's `zarr.json` and return its `attributes` blob. The OME
/// multiscales live at `attributes.multiscales` and have the same shape as v2
/// `.zattrs`, so the existing OmeZarrAttrs deserializer takes it directly.
pub fn read_v3_group_attributes(path: &str) -> Option<serde_json::Value> {
    let json_path = format!("{}/zarr.json", path);
    let raw = std::fs::read_to_string(&json_path).ok()?;
    parse_v3_group_attrs(&raw)
}

/// Same as [`read_v3_group_attributes`] but for a remote URL. Caches the JSON
/// under `cache_dir/zarr.json` if not already present.
pub fn read_v3_group_attributes_remote(url: &str, cache_dir: &str, client: &Client) -> Option<serde_json::Value> {
    let local = format!("{}/zarr.json", cache_dir);
    if !std::path::Path::new(&local).exists() {
        let metadata_url = format!("{}/zarr.json", url.trim_end_matches('/'));
        let res = client.get(&metadata_url).send().ok()?;
        if res.status() != reqwest::StatusCode::OK {
            return None;
        }
        let bytes = res.bytes().ok()?;
        std::fs::create_dir_all(cache_dir).ok()?;
        std::fs::write(&local, &bytes).ok()?;
    }
    let raw = std::fs::read_to_string(&local).ok()?;
    parse_v3_group_attrs(&raw)
}

fn parse_v3_group_attrs(raw: &str) -> Option<serde_json::Value> {
    let parsed: serde_json::Value = serde_json::from_str(raw).ok()?;
    let node_type = parsed.get("node_type")?.as_str()?;
    if node_type != "group" {
        return None;
    }
    parsed.get("attributes").cloned()
}
