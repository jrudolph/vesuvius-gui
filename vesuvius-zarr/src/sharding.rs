//! Zarr v3 `sharding_indexed` codec — minimal partial-read implementation.
//!
//! Supported index layouts:
//!   * `index_location`: `start` (production c3d volumes) or `end` (zarr-python's default).
//!   * `index_codecs`: `[bytes { endian: little }]` or `[bytes { endian: little }, crc32c]`
//!     (zarr-python's default). The checksum is verified on parse.
//!
//! Per the v3 sharding spec each index entry is `(u64 offset, u64 length)`, with the
//! sentinel `(u64::MAX, u64::MAX)` marking an empty sub-chunk. Anything else (compressed
//! index codecs, big-endian) is refused by the v3 module at array-open time.

use std::sync::Arc;

pub const EMPTY_ENTRY: (u64, u64) = (u64::MAX, u64::MAX);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexLocation {
    Start,
    End,
}

/// Where the shard index sits and how it is encoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexLayout {
    pub location: IndexLocation,
    /// `index_codecs` ends in `crc32c`: a 4-byte little-endian CRC-32C of the entries follows them.
    pub crc32c: bool,
}

impl IndexLayout {
    /// Encoded index size in bytes for `n_sub_chunks` entries.
    pub fn size(&self, n_sub_chunks: usize) -> usize {
        n_sub_chunks * 16 + if self.crc32c { 4 } else { 0 }
    }
}

/// Parsed shard index: one `(offset, length)` per sub-chunk in raster order
/// (sub-chunks are flattened in C order over the configured sub-chunk grid).
#[derive(Debug, Clone)]
pub struct ShardIndex {
    pub entries: Vec<(u64, u64)>,
}

impl ShardIndex {
    /// Parse an encoded index (as returned by [`ShardAccess::read_index_bytes`]). Panics on a size
    /// or checksum mismatch: either means the shard is not laid out the way the metadata says.
    pub fn parse(bytes: &[u8], n_sub_chunks: usize, layout: &IndexLayout) -> Self {
        assert_eq!(
            bytes.len(),
            layout.size(n_sub_chunks),
            "shard index has {} bytes, expected {} (n_sub_chunks={}, layout={:?})",
            bytes.len(),
            layout.size(n_sub_chunks),
            n_sub_chunks,
            layout
        );
        let entries_bytes = &bytes[..n_sub_chunks * 16];
        if layout.crc32c {
            let stored = u32::from_le_bytes(bytes[n_sub_chunks * 16..].try_into().unwrap());
            let actual = crc32c(entries_bytes);
            assert_eq!(
                stored, actual,
                "shard index crc32c mismatch: stored {:#010x}, computed {:#010x}",
                stored, actual
            );
        }
        let mut entries = Vec::with_capacity(n_sub_chunks);
        for i in 0..n_sub_chunks {
            let off = u64::from_le_bytes(entries_bytes[i * 16..i * 16 + 8].try_into().unwrap());
            let len = u64::from_le_bytes(entries_bytes[i * 16 + 8..i * 16 + 16].try_into().unwrap());
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

/// CRC-32C (Castagnoli), as used by the zarr v3 `crc32c` codec.
pub fn crc32c(data: &[u8]) -> u32 {
    const fn table() -> [u32; 256] {
        let mut t = [0u32; 256];
        let mut i = 0;
        while i < 256 {
            let mut c = i as u32;
            let mut k = 0;
            while k < 8 {
                c = if c & 1 != 0 { (c >> 1) ^ 0x82F6_3B78 } else { c >> 1 };
                k += 1;
            }
            t[i] = c;
            i += 1;
        }
        t
    }
    static TABLE: [u32; 256] = table();
    let mut crc = !0u32;
    for &b in data {
        crc = TABLE[((crc ^ b as u32) & 0xff) as usize] ^ (crc >> 8);
    }
    !crc
}

/// Per-shard random access — implementations decide whether to mmap a file or
/// issue HTTP range requests. The trait deliberately stays narrow: just
/// `read_range`. The two-tier shard-index cache lives in `v3::*Directory`,
/// which is where backend-specific deduplication happens anyway.
pub trait ShardAccess: Send + Sync {
    /// Returns the requested bytes, or `None` if the shard is missing entirely.
    fn read_range(&self, offset: u64, len: u64) -> Option<Vec<u8>>;
    /// The encoded index (`layout.size(n_sub_chunks)` bytes from the head or tail of the shard).
    fn read_index_bytes(&self, layout: &IndexLayout, n_sub_chunks: usize) -> Option<Vec<u8>>;
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

    fn read_index_bytes(&self, layout: &IndexLayout, n_sub_chunks: usize) -> Option<Vec<u8>> {
        let size = layout.size(n_sub_chunks) as u64;
        let offset = match layout.location {
            IndexLocation::Start => 0,
            IndexLocation::End => (self.map.len() as u64).checked_sub(size)?,
        };
        self.read_range(offset, size)
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
        self.fetch(&format!("bytes={}-{}", offset, end_inclusive), len)
    }

    /// GET with the given `Range` header value; expects a 206 with exactly `len` bytes.
    fn fetch(&self, range_header: &str, len: u64) -> Option<Vec<u8>> {
        let resp = self
            .client
            .get(&self.url)
            .header(reqwest::header::RANGE, range_header)
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

    fn read_index_bytes(&self, layout: &IndexLayout, n_sub_chunks: usize) -> Option<Vec<u8>> {
        let size = layout.size(n_sub_chunks) as u64;
        match layout.location {
            IndexLocation::Start => self.fetch_range(0, size),
            IndexLocation::End => self.fetch(&format!("bytes=-{}", size), size),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32c_check_value() {
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
        assert_eq!(crc32c(b""), 0);
    }

    #[test]
    fn parse_with_crc() {
        let layout = IndexLayout {
            location: IndexLocation::End,
            crc32c: true,
        };
        let mut bytes = Vec::new();
        for (o, l) in [(0u64, 10u64), (u64::MAX, u64::MAX)] {
            bytes.extend_from_slice(&o.to_le_bytes());
            bytes.extend_from_slice(&l.to_le_bytes());
        }
        let crc = crc32c(&bytes);
        bytes.extend_from_slice(&crc.to_le_bytes());
        let idx = ShardIndex::parse(&bytes, 2, &layout);
        assert_eq!(idx.lookup(0), Some((0, 10)));
        assert_eq!(idx.lookup(1), None);
    }

    #[test]
    #[should_panic(expected = "crc32c mismatch")]
    fn parse_rejects_bad_crc() {
        let layout = IndexLayout {
            location: IndexLocation::Start,
            crc32c: true,
        };
        ShardIndex::parse(&[0u8; 20], 1, &layout);
    }
}
