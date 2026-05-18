#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_int};
use std::os::raw::c_void;

pub enum c3d_decoder {}

pub type c3d_panic_fn = extern "C" fn(file: *const c_char, line: c_int, msg: *const c_char);

unsafe extern "C" {
    pub fn c3d_set_panic_hook(hook: c3d_panic_fn);

    pub fn c3d_chunk_validate(input: *const u8, in_len: usize) -> bool;

    pub fn c3d_decoder_new() -> *mut c3d_decoder;
    pub fn c3d_decoder_free(dec: *mut c3d_decoder);
    pub fn c3d_decoder_set_denoise(dec: *mut c3d_decoder, enabled: bool);
    pub fn c3d_decoder_chunk_decode(dec: *mut c3d_decoder, input: *const u8, in_len: usize, out: *mut u8);
}

// `c3d_is_chunk` is a `static inline` in c3d.h, so we re-implement the magic
// sniff in Rust rather than dragging a stub through the C compiler.
#[inline]
pub fn c3d_is_chunk_inline(buf: &[u8]) -> bool {
    buf.len() >= 4 && &buf[..4] == b"C3DC"
}

// Silence "unused" on the type alias when nothing in lib.rs needs it directly.
const _: *const c_void = std::ptr::null();
