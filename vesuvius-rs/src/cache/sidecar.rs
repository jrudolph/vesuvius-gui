//! Per-LOD chunk-state table of the decoded-chunk store, in memory.
//!
//! For each LOD level the volume has a fixed number of 64³ chunk slots
//! (`nx*ny*nz`). Each slot has one state byte (`STATE_*`) and one access
//! epoch byte (LRU bookkeeping, see `epoch.rs`). Both are atomics read and
//! written lock-free: writers publish RESIDENT with Release after filling the
//! slot, readers Acquire-load it before reading the bytes.

use std::sync::atomic::{AtomicU8, Ordering};

pub const STATE_MISSING: u8 = 0;
pub const STATE_RESIDENT: u8 = 1;
pub const STATE_EMPTY: u8 = 2;
/// Transient "operation in flight on this slot" marker, i.e. a per-slot
/// mutex: every transition is a `compare_exchange` from a specific
/// predecessor to `STATE_LOCKED`, the operation runs, then a plain store of
/// the destination value releases it. Readers treat LOCKED as not readable.
pub const STATE_LOCKED: u8 = 3;

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
            nx: extent[0].div_ceil(span),
            ny: extent[1].div_ceil(span),
            nz: extent[2].div_ceil(span),
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

#[derive(Debug, Clone)]
pub struct Header {
    pub volume_id: String,
    pub max_lod: u8,
    pub extent: [u32; 3],
    pub lods: Vec<LodDims>,
}

impl Header {
    pub fn new(volume_id: String, extent: [u32; 3], max_lod: u8) -> Self {
        let lods = (0..=max_lod).map(|l| LodDims::for_lod(extent, l)).collect();
        Self {
            volume_id,
            max_lod,
            extent,
            lods,
        }
    }
}

pub struct Sidecar {
    pub header: Header,
    /// Per LOD: state bytes.
    states: Vec<Box<[AtomicU8]>>,
    /// Per LOD: access epoch bytes.
    epochs: Vec<Box<[AtomicU8]>>,
}

fn zeroed(n: u64) -> Box<[AtomicU8]> {
    (0..n).map(|_| AtomicU8::new(0)).collect()
}

impl Sidecar {
    /// Every slot Missing, every access epoch 0.
    pub fn empty(header: Header) -> Self {
        let states = header.lods.iter().map(|d| zeroed(d.count())).collect();
        let epochs = header.lods.iter().map(|d| zeroed(d.count())).collect();
        Self { header, states, epochs }
    }

    pub fn get_state(&self, lod: u8, idx: u64) -> u8 {
        self.states[lod as usize][idx as usize].load(Ordering::Acquire)
    }

    /// Publish a new state for `idx`; returns the previous one.
    pub fn set_state(&self, lod: u8, idx: u64, state: u8) -> u8 {
        self.states[lod as usize][idx as usize].swap(state, Ordering::Release)
    }

    /// CAS the state byte. AcqRel on success pairs with the readers'
    /// Acquire loads.
    pub fn compare_exchange_state(&self, lod: u8, idx: u64, current: u8, new: u8) -> Result<u8, u8> {
        self.states[lod as usize][idx as usize].compare_exchange(current, new, Ordering::AcqRel, Ordering::Acquire)
    }

    pub fn get_access_epoch(&self, lod: u8, idx: u64) -> u8 {
        self.epochs[lod as usize][idx as usize].load(Ordering::Relaxed)
    }

    pub fn set_access_epoch(&self, lod: u8, idx: u64, epoch: u8) {
        self.epochs[lod as usize][idx as usize].store(epoch, Ordering::Relaxed);
    }

    /// CAS the access epoch: only the winner of a racing touch adjusts the
    /// epoch histogram.
    pub fn compare_exchange_access_epoch(&self, lod: u8, idx: u64, current: u8, new: u8) -> Result<u8, u8> {
        self.epochs[lod as usize][idx as usize].compare_exchange(current, new, Ordering::Relaxed, Ordering::Relaxed)
    }
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
    fn state_and_epoch_columns_are_independent() {
        let s = Sidecar::empty(Header::new("v".into(), [256, 256, 256], 1));
        s.set_state(0, 5, STATE_RESIDENT);
        s.set_access_epoch(0, 5, 9);
        s.set_state(1, 0, STATE_EMPTY);
        assert_eq!(s.get_state(0, 5), STATE_RESIDENT);
        assert_eq!(s.get_access_epoch(0, 5), 9);
        assert_eq!(s.get_state(0, 4), STATE_MISSING);
        assert_eq!(s.get_access_epoch(1, 0), 0);
        assert_eq!(s.compare_exchange_state(1, 0, STATE_MISSING, STATE_LOCKED), Err(STATE_EMPTY));
    }
}
