//! Zarr v3 metadata parsing + sharded vc-delta3d access.
//!
//! Scope is deliberately narrow: `uint8` arrays with exactly one top-level
//! `sharding_indexed` codec whose inner chain is
//!     `[bytes { endian: little }, vc-delta3d]` (any sub-chunk shape)
//! and whose index is `[bytes]` or `[bytes, crc32c]` at the `start` or `end`
//! of the shard. Anything else `panic!`s at array-open time rather than
//! failing later in an opaque way. When the format broadens, extend this module.

use crate::sharding::{IndexLayout, IndexLocation, LocalShard, RemoteShard, ShardAccess, ShardIndex};
use crate::{
    default_cache_dir_for_url, parse_json, ChunkContext, RawContext, ZarrArray, ZarrArrayDef, ZarrDataType,
    ZarrFileAccess, ZarrOrder, ZarrVersion,
};
use dashmap::DashMap;
use fxhash::FxBuildHasher;
use reqwest::blocking::Client;
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

/// Public handle into a remote v3 sharded array. Exposes the shard
/// grid metadata, shard URL builder, and a single-flight, blocking
/// shard-index fetcher — i.e. everything a planner needs to issue HTTP
/// Range requests against sub-chunks without going through `load_chunk`.
///
/// Implementations cache shard indices in-memory + on-disk; the first
/// call for a given shard does the network fetch and concurrent callers
/// for that same shard block on the result rather than racing.
pub trait V3RemoteShardedAccess: Send + Sync {
    fn sub_chunk_shape(&self) -> [usize; 3];
    fn sub_chunks_per_shard(&self) -> [usize; 3];
    fn shape(&self) -> &[usize];
    fn shard_url(&self, shard: [usize; 3]) -> String;
    /// Blocking, single-flight: returns the parsed shard index, or `None`
    /// if the shard is definitively absent (404/403 on first fetch, or a
    /// previously persisted `_missing` marker on disk). Concurrent
    /// callers for the same shard coalesce to a single HTTP fetch.
    fn shard_index(&self, shard: [usize; 3]) -> Option<Arc<ShardIndex>>;
    /// Decode one compressed sub-chunk (as addressed by the shard index) to
    /// `sub_chunk_shape` u8 voxels in C order.
    fn decode_sub_chunk(&self, compressed: &[u8]) -> Result<Vec<u8>, String>;
}

/// Codec applied to each sub-chunk inside a shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubChunkCodec {
    Delta3d,
}

impl SubChunkCodec {
    /// Decode one sub-chunk that must yield exactly `expected_bytes`.
    pub fn decode(&self, compressed: &[u8], expected_bytes: usize) -> Result<Vec<u8>, String> {
        let decoded = match self {
            SubChunkCodec::Delta3d => {
                vesuvius_delta3d::decode_expecting(compressed, expected_bytes).map_err(|e| e.to_string())?
            }
        };
        if decoded.len() != expected_bytes {
            return Err(format!(
                "{:?} sub-chunk decoded to {} bytes, expected {}",
                self,
                decoded.len(),
                expected_bytes
            ));
        }
        Ok(decoded)
    }

    /// Extension of compressed sub-chunks in the remote byte cache.
    fn compressed_ext(&self) -> &'static str {
        match self {
            SubChunkCodec::Delta3d => "vcd3",
        }
    }
}

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
    pub codec: SubChunkCodec,
    pub index_layout: IndexLayout,
}

/// A `bytes` codec entry; only little-endian (or unspecified, fine for uint8) is accepted.
fn assert_bytes_little(c: &V3CodecJson, what: &str) {
    assert_eq!(c.name, "bytes", "v3.rs: {}: expected `bytes` codec, got `{}`", what, c.name);
    let endian = c.configuration.get("endian").and_then(|e| e.as_str()).unwrap_or("little");
    assert_eq!(endian, "little", "v3.rs: {}: only little-endian `bytes` supported", what);
}

fn parse_sub_chunk_codec(codecs: &[V3CodecJson]) -> SubChunkCodec {
    let names: Vec<&str> = codecs.iter().map(|c| c.name.as_str()).collect();
    match names.as_slice() {
        ["bytes", "vc-delta3d"] => {
            assert_bytes_little(&codecs[0], "sharding.codecs");
            SubChunkCodec::Delta3d
        }
        _ => panic!(
            "v3.rs: unsupported sub-chunk codec chain {:?} (supported: [bytes, vc-delta3d])",
            names
        ),
    }
}

fn parse_index_layout(sharding: &ShardingConfig) -> IndexLayout {
    let names: Vec<&str> = sharding.index_codecs.iter().map(|c| c.name.as_str()).collect();
    let crc32c = match names.as_slice() {
        ["bytes"] => false,
        ["bytes", "crc32c"] => true,
        _ => panic!(
            "v3.rs: unsupported index_codecs {:?} (supported: [bytes], [bytes, crc32c])",
            names
        ),
    };
    assert_bytes_little(&sharding.index_codecs[0], "sharding.index_codecs");
    // The v3 spec default is `end`.
    let location = match sharding.index_location.as_deref().unwrap_or("end") {
        "start" => IndexLocation::Start,
        "end" => IndexLocation::End,
        other => panic!("v3.rs: unknown index_location `{}`", other),
    };
    IndexLayout { location, crc32c }
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

        let codec = parse_sub_chunk_codec(&sharding.codecs);
        let index_layout = parse_index_layout(&sharding);

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

        Self {
            shard_chunk_shape,
            sub_chunk_shape,
            sub_chunks_per_shard,
            n_sub_chunks_per_shard,
            separator,
            codec,
            index_layout,
        }
    }

    pub fn sub_chunk_bytes(&self) -> usize {
        self.sub_chunk_shape.iter().product()
    }

    fn decode_sub_chunk(&self, compressed: &[u8], shard: [usize; 3], flat: usize) -> Vec<u8> {
        self.codec
            .decode(compressed, self.sub_chunk_bytes())
            .unwrap_or_else(|e| panic!("decode failed at shard {:?} sub-chunk {}: {}", shard, flat, e))
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
// Layer-2 (decoded) on-disk cache
// ────────────────────────────────────────────────────────────────────────────
//
// Sub-chunk decoded bytes (16 MiB each) live as plain files on disk and are
// mmapped on read. The OS page cache then acts as the implicit RAM tier — we
// don't track residency or evict anything ourselves. Files are written via
// `*.raw.tmp.<n>` + `rename` so concurrent decodes of the same sub-chunk are
// safe (decoding is deterministic, so last-rename-wins is fine).

const DECODED_SUFFIX: &str = "raw";

#[derive(Debug)]
struct DecodedCache {
    root: PathBuf,
    sub_chunk_bytes: usize,
}

impl DecodedCache {
    fn new(root: PathBuf, sub_chunk_bytes: usize) -> Self {
        Self { root, sub_chunk_bytes }
    }

    fn path_for(&self, shard: [usize; 3], flat: usize) -> PathBuf {
        self.root
            .join(format!("{}_{}_{}", shard[0], shard[1], shard[2]))
            .join(format!("sub_{:05}.{}", flat, DECODED_SUFFIX))
    }

    /// Hot path: try to mmap the decoded file. Returns `None` on any error
    /// (missing, truncated, etc.) so the caller can fall through to decode.
    /// Truncated/short files are treated as cache misses; the next decode
    /// will overwrite them via atomic rename.
    fn try_load(&self, shard: [usize; 3], flat: usize) -> Option<ChunkContext> {
        let path = self.path_for(shard, flat);
        let raw = RawContext::open(&path)?;
        if raw.len() != self.sub_chunk_bytes {
            return None;
        }
        Some(ChunkContext::Raw(raw))
    }

    /// Atomic write: stage to a unique temp name in the same directory, then
    /// rename onto the final path. The unique suffix prevents collisions
    /// between threads/processes that happen to be decoding the same sub-chunk
    /// concurrently; both writes are valid (identical bytes), so the last
    /// rename to win is harmless.
    fn write_atomic(&self, shard: [usize; 3], flat: usize, bytes: &[u8]) {
        let final_path = self.path_for(shard, flat);
        let parent = match final_path.parent() {
            Some(p) => p,
            None => return,
        };
        if let Err(e) = std::fs::create_dir_all(parent) {
            log::warn!("decoded-cache: create_dir_all({}) failed: {}", parent.display(), e);
            return;
        }
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let tid = std::thread::current().id();
        let tmp = parent.join(format!("sub_{:05}.{}.tmp.{}.{:?}", flat, DECODED_SUFFIX, n, tid));
        if let Err(e) = std::fs::write(&tmp, bytes) {
            log::warn!("decoded-cache: write({}) failed: {}", tmp.display(), e);
            let _ = std::fs::remove_file(&tmp);
            return;
        }
        if let Err(e) = std::fs::rename(&tmp, &final_path) {
            log::warn!(
                "decoded-cache: rename({} -> {}) failed: {}",
                tmp.display(),
                final_path.display(),
                e
            );
            let _ = std::fs::remove_file(&tmp);
        }
    }
}

/// Stable cache root for a local zarr array at `array_path`. Uses the same
/// hashed-by-URL scheme as remote so source volumes stay read-only and the
/// per-LOD subdirectory falls out naturally (the OME caller passes a distinct
/// `array_path` per multiscale level, e.g. `.../volume.zarr/0`).
fn local_cache_root(array_path: &str) -> PathBuf {
    let abs = match std::fs::canonicalize(array_path) {
        Ok(p) => p.to_string_lossy().into_owned(),
        Err(_) => array_path.trim_end_matches('/').to_string(),
    };
    let key = format!("file://{}", abs);
    PathBuf::from(default_cache_dir_for_url(&key))
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
    // Layer-2: persisted decoded sub-chunks. Layer 1 (the compressed bytes)
    // is the on-disk shard file itself; no separate compressed cache.
    decoded_cache: DecodedCache,
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
        let idx_bytes =
            shard_handle.read_index_bytes(&self.sharded.index_layout, self.sharded.n_sub_chunks_per_shard)?;
        let parsed = Arc::new(ShardIndex::parse(
            &idx_bytes,
            self.sharded.n_sub_chunks_per_shard,
            &self.sharded.index_layout,
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

        // Layer 2 hot path: a previously-decoded sub-chunk on disk.
        if let Some(ctx) = self.decoded_cache.try_load(shard, flat) {
            return Some(ctx);
        }

        // Layer 1: the shard file is the compressed source for local volumes.
        let index = self.get_index(shard)?;
        let (off, len) = index.lookup(flat)?;
        let shard_handle = self.get_shard(shard)?;
        let compressed = shard_handle.read_range(off, len)?;

        // Decode and persist for next time.
        let decoded = self.sharded.decode_sub_chunk(&compressed, shard, flat);
        self.decoded_cache.write_atomic(shard, flat, &decoded);

        // Prefer to return the mmap'd view (so subsequent in-process accesses
        // go through the same page-cache-backed pages). Fall back to the
        // freshly-decoded Vec if the mmap can't be opened — e.g. disk full or
        // permission denied on the write — so we still serve the request.
        if let Some(ctx) = self.decoded_cache.try_load(shard, flat) {
            Some(ctx)
        } else {
            Some(ChunkContext::Heap(decoded))
        }
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
    compressed_dir: PathBuf, // {cache_dir}/compressed
    client: Client,
    sharded: Sharded,
    shape: Vec<usize>,
    // Single-flight shard-index cache. Each shard has its own `OnceLock`;
    // the first caller into `get_index(shard)` runs `fetch_index_blocking`
    // while concurrent callers for that same shard block inside
    // `get_or_init`. The persisted Option is `Some` when present and
    // `None` when the shard is definitively absent (404/403, or a
    // pre-existing `_missing` marker on disk). The on-disk caches (under
    // `compressed_dir` and `decoded_cache.root`) survive restarts; this
    // in-memory layer just coalesces concurrent fetches and avoids
    // re-reading the disk file on every miss within one process.
    index_cells: DashMap<[usize; 3], Arc<OnceLock<Option<Arc<ShardIndex>>>>, FxBuildHasher>,
    // Layer 2: persisted decoded sub-chunks.
    decoded_cache: DecodedCache,
}

impl RemoteV3ShardedAccess {
    fn shard_url_inner(&self, shard: [usize; 3]) -> String {
        let sep = &self.sharded.separator;
        format!("{}/c/{}{sep}{}{sep}{}", self.array_url, shard[0], shard[1], shard[2])
    }

    fn shard_cache_dir(&self, shard: [usize; 3]) -> PathBuf {
        self.compressed_dir
            .join("c")
            .join(format!("{}_{}_{}", shard[0], shard[1], shard[2]))
    }

    fn index_disk_path(&self, shard: [usize; 3]) -> PathBuf {
        self.shard_cache_dir(shard).join("_index.bin")
    }

    fn subchunk_disk_path(&self, shard: [usize; 3], flat: usize) -> PathBuf {
        self.shard_cache_dir(shard)
            .join(format!("sub_{:05}.{}", flat, self.sharded.codec.compressed_ext()))
    }

    fn missing_marker_path(&self, shard: [usize; 3]) -> PathBuf {
        self.shard_cache_dir(shard).join("_missing")
    }

    fn get_index(&self, shard: [usize; 3]) -> Option<Arc<ShardIndex>> {
        // Resolve (or create) the per-shard cell. The clone+drop pattern
        // releases the dashmap bucket lock before we block inside
        // `get_or_init`, so concurrent callers for different shards never
        // contend on the dashmap and concurrent callers for the same
        // shard coalesce inside the OnceLock.
        let cell = {
            let entry = self
                .index_cells
                .entry(shard)
                .or_insert_with(|| Arc::new(OnceLock::new()));
            entry.value().clone()
        };
        cell.get_or_init(|| self.fetch_index_blocking(shard)).clone()
    }

    fn fetch_index_blocking(&self, shard: [usize; 3]) -> Option<Arc<ShardIndex>> {
        let disk_idx = self.index_disk_path(shard);
        let missing = self.missing_marker_path(shard);
        if missing.exists() {
            return None;
        }
        if let Ok(bytes) = std::fs::read(&disk_idx) {
            return Some(Arc::new(ShardIndex::parse(
                &bytes,
                self.sharded.n_sub_chunks_per_shard,
                &self.sharded.index_layout,
            )));
        }

        let url = self.shard_url_inner(shard);
        let remote = RemoteShard {
            url,
            client: self.client.clone(),
        };
        let bytes = match remote.read_index_bytes(&self.sharded.index_layout, self.sharded.n_sub_chunks_per_shard) {
            Some(b) => b,
            None => {
                // Persist a "missing" marker so the next process doesn't retry.
                let _ = std::fs::create_dir_all(self.shard_cache_dir(shard));
                let _ = std::fs::write(&missing, b"");
                return None;
            }
        };
        let parsed = Arc::new(ShardIndex::parse(
            &bytes,
            self.sharded.n_sub_chunks_per_shard,
            &self.sharded.index_layout,
        ));

        let _ = std::fs::create_dir_all(self.shard_cache_dir(shard));
        let tmp = disk_idx.with_extension("bin.tmp");
        if std::fs::write(&tmp, &bytes).is_ok() {
            let _ = std::fs::rename(&tmp, &disk_idx);
        }
        Some(parsed)
    }

    fn fetch_subchunk(&self, shard: [usize; 3], flat: usize, off: u64, len: u64) -> Option<Vec<u8>> {
        let disk_path = self.subchunk_disk_path(shard, flat);
        if let Ok(bytes) = std::fs::read(&disk_path) {
            return Some(bytes);
        }

        let url = self.shard_url_inner(shard);
        let remote = RemoteShard {
            url,
            client: self.client.clone(),
        };
        let bytes = remote.read_range(off, len)?;

        let _ = std::fs::create_dir_all(self.shard_cache_dir(shard));
        let tmp = disk_path.with_extension(format!("{}.tmp", self.sharded.codec.compressed_ext()));
        if std::fs::write(&tmp, &bytes).is_ok() {
            let _ = std::fs::rename(&tmp, &disk_path);
        }
        Some(bytes)
    }
}

impl V3RemoteShardedAccess for RemoteV3ShardedAccess {
    fn sub_chunk_shape(&self) -> [usize; 3] {
        self.sharded.sub_chunk_shape
    }
    fn sub_chunks_per_shard(&self) -> [usize; 3] {
        self.sharded.sub_chunks_per_shard
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn shard_url(&self, shard: [usize; 3]) -> String {
        self.shard_url_inner(shard)
    }
    fn shard_index(&self, shard: [usize; 3]) -> Option<Arc<ShardIndex>> {
        self.get_index(shard)
    }
    fn decode_sub_chunk(&self, compressed: &[u8]) -> Result<Vec<u8>, String> {
        self.sharded.codec.decode(compressed, self.sharded.sub_chunk_bytes())
    }
}

impl ZarrFileAccess for RemoteV3ShardedAccess {
    fn load_array_def(&self) -> ZarrArrayDef {
        self.sharded.synthesize_def(&self.shape)
    }

    fn load_chunk(&self, _def: &ZarrArrayDef, chunk_no: &[usize]) -> Option<ChunkContext> {
        let (shard, flat) = self.sharded.locate(chunk_no);

        // Layer 2 hot path.
        if let Some(ctx) = self.decoded_cache.try_load(shard, flat) {
            return Some(ctx);
        }

        // Layer 1 (with HTTP fallback inside).
        let index = self.get_index(shard)?;
        let (off, len) = index.lookup(flat)?;
        let compressed = self.fetch_subchunk(shard, flat, off, len)?;

        // Decode and persist for next time.
        let decoded = self.sharded.decode_sub_chunk(&compressed, shard, flat);
        self.decoded_cache.write_atomic(shard, flat, &decoded);

        if let Some(ctx) = self.decoded_cache.try_load(shard, flat) {
            Some(ctx)
        } else {
            Some(ChunkContext::Heap(decoded))
        }
    }

    fn cache_missing(&self) -> bool {
        true
    }
}

/// Open a v3 sharded (vc-delta3d) array hosted at a remote URL. Caches the array
/// `zarr.json`, the per-shard index, and the per-sub-chunk compressed bytes
/// under `cache_dir` so warm reads don't go to the network.
pub fn open_v3_array_remote(url: &str, cache_dir: &str, client: Client) -> ZarrArray<3, u8> {
    let url = url.trim_end_matches('/');
    let cache_dir = cache_dir.trim_end_matches('/');
    let cache_root = PathBuf::from(cache_dir);
    let compressed_dir = cache_root.join("compressed");
    let decoded_dir = cache_root.join("decoded");

    // Cache + parse the array zarr.json (lives at the cache root).
    let local_metadata = cache_root.join("zarr.json");
    if !local_metadata.exists() {
        let metadata_url = format!("{}/zarr.json", url);
        let res = client
            .get(&metadata_url)
            .send()
            .unwrap_or_else(|e| panic!("fetch {}: {}", metadata_url, e));
        if !res.status().is_success() {
            panic!("fetch {}: status {}", metadata_url, res.status());
        }
        let bytes = res.bytes().unwrap();
        std::fs::create_dir_all(&cache_root).unwrap();
        std::fs::write(&local_metadata, &bytes).unwrap();
    }
    let raw = std::fs::read_to_string(&local_metadata).unwrap();
    let json: V3ArrayJson = parse_json(&raw, &local_metadata.to_string_lossy());
    let sharded = Sharded::from_array_json(&json);
    let shape = json.shape.clone();
    let def = sharded.synthesize_def(&shape);
    let decoded_cache = DecodedCache::new(decoded_dir, sharded.sub_chunk_bytes());

    let access = Arc::new(RemoteV3ShardedAccess {
        array_url: url.to_string(),
        compressed_dir,
        client,
        sharded,
        shape,
        index_cells: DashMap::with_hasher(FxBuildHasher::default()),
        decoded_cache,
    });
    let v3_handle: Arc<dyn V3RemoteShardedAccess> = access.clone();
    let access_dyn: Arc<dyn ZarrFileAccess> = access;
    ZarrArray::from_def_access_and_v3(def, access_dyn, Some(v3_handle))
}

pub fn open_v3_array_local(path: &str) -> ZarrArray<3, u8> {
    let json_path = format!("{}/zarr.json", path);
    let raw = std::fs::read_to_string(&json_path).unwrap_or_else(|e| panic!("read {}: {}", json_path, e));
    let json: V3ArrayJson = parse_json(&raw, &json_path);
    let sharded = Sharded::from_array_json(&json);
    let shape = json.shape.clone();
    let def = sharded.synthesize_def(&shape);

    let cache_root = local_cache_root(path);
    let decoded_cache = DecodedCache::new(cache_root.join("decoded"), sharded.sub_chunk_bytes());

    let access = Arc::new(LocalV3ShardedAccess {
        array_root: path.trim_end_matches('/').to_string(),
        sharded,
        shape,
        index_cache: DashMap::with_hasher(FxBuildHasher::default()),
        shard_cache: DashMap::with_hasher(FxBuildHasher::default()),
        decoded_cache,
    });
    ZarrArray::from_def_and_access(def, access)
}

/// Read a v3 group's `zarr.json` and return its OME attributes blob: the
/// object holding `multiscales`, which has the same shape as v2 `.zattrs`, so
/// the existing OmeZarrAttrs deserializer takes it directly. That object is
/// `attributes` itself (NGFF 0.4 style) or `attributes.ome` (NGFF 0.5).
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
    let attributes = parsed.get("attributes")?;
    // NGFF 0.5 nests everything under `ome`; 0.4-style v3 groups put `multiscales` at the top.
    Some(attributes.get("ome").unwrap_or(attributes).clone())
}

#[cfg(test)]
mod tests {
    use super::parse_v3_group_attrs;

    #[test]
    fn group_attrs_ngff_04_and_05() {
        let v04 = r#"{"zarr_format":3,"node_type":"group","attributes":{"multiscales":[{"version":"0.4"}]}}"#;
        let v05 = r#"{"zarr_format":3,"node_type":"group",
            "attributes":{"ome":{"version":"0.5","multiscales":[{"name":"/"}]},"zarr_transform":{}}}"#;
        assert!(parse_v3_group_attrs(v04).unwrap().get("multiscales").is_some());
        let a = parse_v3_group_attrs(v05).unwrap();
        assert!(a.get("multiscales").is_some());
        assert_eq!(a["version"], "0.5");
        let array = r#"{"zarr_format":3,"node_type":"array","attributes":{}}"#;
        assert!(parse_v3_group_attrs(array).is_none());
    }
}
