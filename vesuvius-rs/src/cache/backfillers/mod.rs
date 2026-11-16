//! Bundled `ChunkBackfiller` implementations. Real-backend backfillers
//! (ome-zarr, …) land here as later phases.

pub mod ome_zarr;
pub mod ome_zarr_v3;
pub mod synthesized_lod;
pub mod synthetic;
