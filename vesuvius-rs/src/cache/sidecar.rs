//! Per-LOD chunk-state bitmap, persisted to a small `chunks.idx` sidecar.
//!
//! For each LOD level the volume has a fixed number of 64³ chunk slots
//! (`nx*ny*nz`). The sidecar stores one byte per slot — `0` Missing, `1`
//! Resident, `2` Empty — alongside a header that records the extent, LOD
//! count, and per-LOD dimensions so the offset math can be reconstructed on
//! reopen.
//!
//! In memory each LOD's bytes are wrapped in a `Vec<AtomicU8>`; transitions
//! are single-producer (the dispatch claim in `cache.rs` guarantees one
//! writer per chunk), so a plain `store(Release)` is sufficient — no CAS.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

pub const MAGIC: &[u8; 8] = b"VCSPRS01";
pub const STATE_MISSING: u8 = 0;
pub const STATE_RESIDENT: u8 = 1;
pub const STATE_EMPTY: u8 = 2;

#[derive(Clone, Copy, Debug)]
pub struct LodDims {
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
}

impl LodDims {
    pub fn for_lod(extent: [u32; 3], lod: u8) -> Self {
        let span = 64u32 << lod;
        Self {
            nx: div_ceil_u32(extent[0], span),
            ny: div_ceil_u32(extent[1], span),
            nz: div_ceil_u32(extent[2], span),
        }
    }

    pub fn count(&self) -> u64 {
        self.nx as u64 * self.ny as u64 * self.nz as u64
    }

    pub fn linear_index(&self, x: u32, y: u32, z: u32) -> Option<u64> {
        if x >= self.nx || y >= self.ny || z >= self.nz {
            return None;
        }
        Some((z as u64 * self.ny as u64 + y as u64) * self.nx as u64 + x as u64)
    }
}

#[derive(Debug)]
pub struct Header {
    pub volume_id: String,
    pub chunk_side: u8,
    pub max_lod: u8,
    pub extent: [u32; 3],
    pub lods: Vec<LodDims>,
}

impl Header {
    pub fn new(volume_id: String, extent: [u32; 3], max_lod: u8) -> Self {
        let lods = (0..=max_lod).map(|l| LodDims::for_lod(extent, l)).collect();
        Self {
            volume_id,
            chunk_side: 64,
            max_lod,
            extent,
            lods,
        }
    }

    /// True if this header describes the same volume layout as `other`.
    /// Mismatch triggers a hard rebuild of all data files.
    pub fn matches(&self, other: &Header) -> bool {
        self.volume_id == other.volume_id
            && self.chunk_side == other.chunk_side
            && self.max_lod == other.max_lod
            && self.extent == other.extent
    }

    fn write_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(MAGIC);
        let vid = self.volume_id.as_bytes();
        out.extend_from_slice(&(vid.len() as u16).to_le_bytes());
        out.extend_from_slice(vid);
        out.push(self.chunk_side);
        out.push(self.max_lod);
        for v in self.extent {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for lod in &self.lods {
            out.extend_from_slice(&lod.nx.to_le_bytes());
            out.extend_from_slice(&lod.ny.to_le_bytes());
            out.extend_from_slice(&lod.nz.to_le_bytes());
        }
    }

    fn read_from<R: Read>(r: &mut R) -> std::io::Result<Self> {
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("bad sidecar magic: {:?}", magic),
            ));
        }
        let mut len_buf = [0u8; 2];
        r.read_exact(&mut len_buf)?;
        let vid_len = u16::from_le_bytes(len_buf) as usize;
        let mut vid = vec![0u8; vid_len];
        r.read_exact(&mut vid)?;
        let volume_id = String::from_utf8(vid)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let mut byte = [0u8; 1];
        r.read_exact(&mut byte)?;
        let chunk_side = byte[0];
        r.read_exact(&mut byte)?;
        let max_lod = byte[0];
        let mut u32_buf = [0u8; 4];
        let mut extent = [0u32; 3];
        for slot in &mut extent {
            r.read_exact(&mut u32_buf)?;
            *slot = u32::from_le_bytes(u32_buf);
        }
        let mut lods = Vec::with_capacity(max_lod as usize + 1);
        for _ in 0..=max_lod {
            r.read_exact(&mut u32_buf)?;
            let nx = u32::from_le_bytes(u32_buf);
            r.read_exact(&mut u32_buf)?;
            let ny = u32::from_le_bytes(u32_buf);
            r.read_exact(&mut u32_buf)?;
            let nz = u32::from_le_bytes(u32_buf);
            lods.push(LodDims { nx, ny, nz });
        }
        Ok(Self {
            volume_id,
            chunk_side,
            max_lod,
            extent,
            lods,
        })
    }
}

/// In-memory chunk-state index. Owns one `Vec<AtomicU8>` per LOD plus a
/// per-LOD counter of transitions since last sync. Files are managed by
/// `DiskStore`; this struct is just the bookkeeping.
pub struct Sidecar {
    pub header: Header,
    bitmaps: Vec<Vec<AtomicU8>>,
    pending: Vec<AtomicU64>,
}

impl Sidecar {
    /// Fresh sidecar with every chunk Missing.
    pub fn empty(header: Header) -> Self {
        let bitmaps = header
            .lods
            .iter()
            .map(|d| {
                let n = d.count() as usize;
                let mut v = Vec::with_capacity(n);
                v.resize_with(n, || AtomicU8::new(STATE_MISSING));
                v
            })
            .collect();
        let pending = (0..header.lods.len()).map(|_| AtomicU64::new(0)).collect();
        Self {
            header,
            bitmaps,
            pending,
        }
    }

    /// Load an existing sidecar from `path`. Returns `Ok(None)` if the file
    /// doesn't exist; returns `Err` on any other failure (caller decides to
    /// treat as missing or to bail). The returned Sidecar's header must
    /// still be checked for layout-compatibility via `header.matches`.
    pub fn load(path: &Path) -> std::io::Result<Option<Self>> {
        let mut file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e),
        };
        let header = Header::read_from(&mut file)?;
        let mut bitmaps = Vec::with_capacity(header.lods.len());
        for lod in &header.lods {
            let n = lod.count() as usize;
            let mut bytes = vec![0u8; n];
            file.read_exact(&mut bytes)?;
            let atoms: Vec<AtomicU8> = bytes.into_iter().map(AtomicU8::new).collect();
            bitmaps.push(atoms);
        }
        let pending = (0..header.lods.len()).map(|_| AtomicU64::new(0)).collect();
        Ok(Some(Self {
            header,
            bitmaps,
            pending,
        }))
    }

    pub fn get_state(&self, lod: u8, idx: u64) -> u8 {
        self.bitmaps[lod as usize][idx as usize].load(Ordering::Acquire)
    }

    /// Publish a new state for `idx` and bump the per-LOD pending counter.
    /// Returns the previous state (for callers that want to detect duplicate
    /// writes; current callers ignore).
    pub fn set_state(&self, lod: u8, idx: u64, state: u8) -> u8 {
        let prev = self.bitmaps[lod as usize][idx as usize].swap(state, Ordering::Release);
        if prev != state {
            self.pending[lod as usize].fetch_add(1, Ordering::Relaxed);
        }
        prev
    }

    /// Snapshot the bitmap and the pending counters. The snapshot is taken
    /// with `Acquire` loads, then the counters are atomically reset to zero
    /// (returning the prior count). Caller must `fsync` data files for every
    /// LOD whose returned `pending` is non-zero **before** writing the
    /// snapshot to disk, so the persisted sidecar is always a strict subset
    /// of durable bytes.
    pub fn snapshot(&self) -> Snapshot {
        let mut bitmaps = Vec::with_capacity(self.bitmaps.len());
        for lod in &self.bitmaps {
            let mut bytes = Vec::with_capacity(lod.len());
            for a in lod {
                bytes.push(a.load(Ordering::Acquire));
            }
            bitmaps.push(bytes);
        }
        let pending: Vec<u64> = self.pending.iter().map(|c| c.swap(0, Ordering::AcqRel)).collect();
        Snapshot { bitmaps, pending }
    }

    /// Total transitions across all LODs since the last snapshot.
    pub fn total_pending(&self) -> u64 {
        self.pending.iter().map(|c| c.load(Ordering::Relaxed)).sum()
    }
}

pub struct Snapshot {
    bitmaps: Vec<Vec<u8>>,
    /// Per-LOD count of transitions captured in this snapshot. Caller uses
    /// it to decide which data files actually need `fsync` before the
    /// sidecar is renamed into place.
    pub pending: Vec<u64>,
}

impl Snapshot {
    pub fn write_to(&self, header: &Header, dest: &Path) -> std::io::Result<()> {
        let parent = dest.parent().expect("sidecar path has parent");
        std::fs::create_dir_all(parent)?;
        let tmp = parent.join(format!("{}.tmp", dest.file_name().unwrap().to_string_lossy()));

        let mut buf = Vec::with_capacity(4096 + self.bitmaps.iter().map(|b| b.len()).sum::<usize>());
        header.write_to(&mut buf);
        for bm in &self.bitmaps {
            buf.extend_from_slice(bm);
        }

        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&buf)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, dest)?;
        Ok(())
    }
}

fn div_ceil_u32(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}

/// Helper: derive the sidecar file path from the cache root.
pub fn sidecar_path(root: &Path) -> PathBuf {
    root.join("chunks.idx")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lod_dims_match_extent() {
        let d = LodDims::for_lod([128, 128, 128], 0);
        assert_eq!((d.nx, d.ny, d.nz), (2, 2, 2));
        let d1 = LodDims::for_lod([128, 128, 128], 1);
        assert_eq!((d1.nx, d1.ny, d1.nz), (1, 1, 1));
        let d_odd = LodDims::for_lod([65, 130, 200], 0);
        assert_eq!((d_odd.nx, d_odd.ny, d_odd.nz), (2, 3, 4));
    }

    #[test]
    fn header_roundtrip() {
        let h = Header::new("vol-x".into(), [256, 384, 512], 3);
        let mut buf = Vec::new();
        h.write_to(&mut buf);
        let mut r = std::io::Cursor::new(&buf);
        let back = Header::read_from(&mut r).unwrap();
        assert!(h.matches(&back));
        assert_eq!(back.lods.len(), 4);
    }

    #[test]
    fn snapshot_resets_pending() {
        let h = Header::new("v".into(), [64, 64, 64], 0);
        let s = Sidecar::empty(h);
        s.set_state(0, 0, STATE_RESIDENT);
        assert_eq!(s.total_pending(), 1);
        let snap = s.snapshot();
        assert_eq!(snap.pending[0], 1);
        assert_eq!(s.total_pending(), 0);
    }

    #[test]
    fn sidecar_file_roundtrip() {
        let dir = tempdir();
        let h = Header::new("vol".into(), [256, 128, 64], 1);
        let s = Sidecar::empty(h);
        s.set_state(0, 0, STATE_RESIDENT);
        s.set_state(0, 3, STATE_EMPTY);
        let path = sidecar_path(&dir);
        s.snapshot().write_to(&s.header, &path).unwrap();

        let loaded = Sidecar::load(&path).unwrap().expect("file should exist");
        assert!(loaded.header.matches(&s.header));
        assert_eq!(loaded.get_state(0, 0), STATE_RESIDENT);
        assert_eq!(loaded.get_state(0, 1), STATE_MISSING);
        assert_eq!(loaded.get_state(0, 3), STATE_EMPTY);
    }

    fn tempdir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "vesuvius-sidecar-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
