#![warn(clippy::all, rust_2018_idioms)]

use std::io::{Cursor, Read};

pub mod cache;
pub mod catalog;
pub mod downloader;
pub mod model;
pub mod remap_config;
pub mod volume;

#[cfg(test)]
mod zarr_test;

pub fn zstd_decompress(input: &[u8]) -> Vec<u8> {
    let mut uncompressed = Vec::new();
    ruzstd::decoding::StreamingDecoder::new(Cursor::new(input))
        .unwrap()
        .read_to_end(&mut uncompressed)
        .unwrap();

    uncompressed
}
