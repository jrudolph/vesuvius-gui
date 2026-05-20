//! Bundled `ChunkBackfiller` implementations. Real-backend backfillers
//! (ome-zarr, c3d, …) land here as later phases.

pub mod ome_zarr;
pub mod synthesized_lod;
pub mod synthetic;
