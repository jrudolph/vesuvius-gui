//! Zarr v3 `sharding_indexed` codec — minimal partial-read implementation.
//!
//! Layout (matching the production c3d volumes today):
//!   * `index_location: start` — index sits at the head of the shard.
//!   * `index_codecs: [bytes { endian: little }]` — no compression, no checksum.
//!     Per the v3 sharding spec each index entry is `(u64 offset, u64 length)`,
//!     with the sentinel `(u64::MAX, u64::MAX)` marking an empty sub-chunk.
//!
//! We deliberately do **not** support `index_location: end`, compressed index
//! codecs, or checksum verification — every code path here `panic!`s rather
//! than silently producing wrong data on an unfamiliar layout. The v3 module
//! refuses to open arrays whose codec chain doesn't match this shape.

use std::sync::Arc;

pub const EMPTY_ENTRY: (u64, u64) = (u64::MAX, u64::MAX);

/// Parsed shard index: one `(offset, length)` per sub-chunk in raster order
/// (sub-chunks are flattened in C order over the configured sub-chunk grid).
#[derive(Debug, Clone)]
pub struct ShardIndex {
    pub entries: Vec<(u64, u64)>,
}

impl ShardIndex {
    pub fn parse_bytes_codec(bytes: &[u8], n_sub_chunks: usize) -> Self {
        assert_eq!(
            bytes.len(),
            n_sub_chunks * 16,
            "shard index has {} bytes, expected {} (n_sub_chunks={})",
            bytes.len(),
            n_sub_chunks * 16,
            n_sub_chunks
        );
        let mut entries = Vec::with_capacity(n_sub_chunks);
        for i in 0..n_sub_chunks {
            let off = u64::from_le_bytes(bytes[i * 16..i * 16 + 8].try_into().unwrap());
            let len = u64::from_le_bytes(bytes[i * 16 + 8..i * 16 + 16].try_into().unwrap());
            entries.push((off, len));
        }
        Self { entries }
    }

    pub fn lookup(&self, flat_idx: usize) -> Option<(u64, u64)> {
        let entry = self.entries.get(flat_idx).copied()?;
        if entry == EMPTY_ENTRY {
            None
        } else {
            Some(entry)
        }
    }
}

/// Per-shard random access — implementations decide whether to mmap a file or
/// issue HTTP range requests. The trait deliberately stays narrow: just
/// `read_range`. The two-tier shard-index cache lives in `v3::*Directory`,
/// which is where backend-specific deduplication happens anyway.
pub trait ShardAccess: Send + Sync {
    /// Returns the requested bytes, or `None` if the shard is missing entirely.
    fn read_range(&self, offset: u64, len: u64) -> Option<Vec<u8>>;
    /// `n_sub_chunks * 16` bytes from the head of the shard.
    fn read_index_bytes(&self, n_sub_chunks: usize) -> Option<Vec<u8>>;
}

/// `mmap`-backed shard for local files. Sized at full file length and held
/// behind an `Arc` so concurrent sub-chunk reads share one mapping per shard.
pub struct LocalShard {
    map: memmap::Mmap,
}

impl std::fmt::Debug for LocalShard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LocalShard {{ mapped {} bytes }}", self.map.len())
    }
}

impl LocalShard {
    pub fn open(path: &std::path::Path) -> Option<Arc<Self>> {
        let f = std::fs::File::open(path).ok()?;
        // SAFETY: file is owned for the duration of this call. After the
        // mapping is created the OS handles backing; we never write to it.
        let map = unsafe { memmap::Mmap::map(&f).ok()? };
        Some(Arc::new(Self { map }))
    }
}

impl ShardAccess for LocalShard {
    fn read_range(&self, offset: u64, len: u64) -> Option<Vec<u8>> {
        let start = offset as usize;
        let end = start.checked_add(len as usize)?;
        if end > self.map.len() {
            return None;
        }
        Some(self.map[start..end].to_vec())
    }

    fn read_index_bytes(&self, n_sub_chunks: usize) -> Option<Vec<u8>> {
        self.read_range(0, (n_sub_chunks * 16) as u64)
    }
}

/// HTTP-Range-backed shard. One per shard URL. Caches the parsed shard
/// presence/missingness inside `RemoteV3ShardedAccess`; this struct itself is
/// stateless beyond holding the client + url.
#[derive(Debug, Clone)]
pub struct RemoteShard {
    pub url: String,
    pub client: reqwest::blocking::Client,
}

impl RemoteShard {
    fn fetch_range(&self, offset: u64, len: u64) -> Option<Vec<u8>> {
        if len == 0 {
            return Some(Vec::new());
        }
        let end_inclusive = offset.checked_add(len)?.checked_sub(1)?;
        let range_header = format!("bytes={}-{}", offset, end_inclusive);
        let resp = self
            .client
            .get(&self.url)
            .header(reqwest::header::RANGE, &range_header)
            .send()
            .ok()?;
        let status = resp.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return None;
        }
        if status == reqwest::StatusCode::OK {
            // 200 instead of 206 means the server ignored Range and sent the
            // whole object. Refuse to silently buffer multi-GB shards.
            panic!(
                "{}: server ignored Range header (returned 200, not 206) — refusing to buffer the full shard",
                self.url
            );
        }
        if status != reqwest::StatusCode::PARTIAL_CONTENT {
            log::warn!(
                "RemoteShard {} range {}: unexpected status {}",
                self.url,
                range_header,
                status
            );
            return None;
        }
        let bytes = resp.bytes().ok()?;
        if bytes.len() != len as usize {
            log::warn!("RemoteShard {}: requested {} bytes, got {}", self.url, len, bytes.len());
            return None;
        }
        Some(bytes.to_vec())
    }
}

impl ShardAccess for RemoteShard {
    fn read_range(&self, offset: u64, len: u64) -> Option<Vec<u8>> {
        self.fetch_range(offset, len)
    }

    fn read_index_bytes(&self, n_sub_chunks: usize) -> Option<Vec<u8>> {
        self.fetch_range(0, (n_sub_chunks * 16) as u64)
    }
}
