#![warn(clippy::all, rust_2018_idioms)]

use std::io::{Cursor, Read};

pub mod catalog;
pub mod downloader;
pub mod gui;
pub mod model;
pub mod volume;
pub mod zarr;

pub fn zstd_decompress(input: &[u8]) -> Vec<u8> {
    let mut uncompressed = Vec::new();
    ruzstd::decoding::StreamingDecoder::new(Cursor::new(input))
        .unwrap()
        .read_to_end(&mut uncompressed)
        .unwrap();

    uncompressed
}
