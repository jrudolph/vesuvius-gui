//! Safe Rust bindings for libc3d (vendored under `vendor/`).
//!
//! Scope today: decoder-only, LOD 0 only. Encoder and LOD>0 are intentionally
//! omitted — the GUI just needs to read 256³ chunks.
//!
//! libc3d aborts on any structural error via `c3d_panic` → `abort()`. Every
//! safe entry point here runs `c3d_chunk_validate` first and returns `Err`
//! instead of letting that fire. The thread-local decoder pattern in
//! [`with_decoder`] mirrors the C++ glue at `utils/src/c3d_codec.cpp` — a
//! fresh `c3d_decoder_new` allocates ~80 MiB of scratch, so reusing per-thread
//! saves 50-100 ms per chunk.

mod ffi;

use std::alloc::{self, Layout};
use std::cell::RefCell;
use std::ffi::CStr;
use std::fmt;
use std::os::raw::{c_char, c_int};
use std::sync::Once;

pub const C3D_CHUNK_SIDE: usize = 256;
pub const C3D_CHUNK_BYTES: usize = C3D_CHUNK_SIDE * C3D_CHUNK_SIDE * C3D_CHUNK_SIDE;

/// libc3d asserts that the decoder's output pointer is 32-byte aligned
/// (vendor/c3d.c:129). Rust's `Vec<u8>` only guarantees `align_of::<u8>() = 1`,
/// so we hand the decoder our own aligned scratch buffer and memcpy out.
const C3D_OUT_ALIGN: usize = 32;

struct AlignedScratch {
    ptr: *mut u8,
    layout: Layout,
}

impl AlignedScratch {
    fn new() -> Self {
        let layout = Layout::from_size_align(C3D_CHUNK_BYTES, C3D_OUT_ALIGN).expect("valid layout");
        // SAFETY: layout has size > 0 and a power-of-two alignment.
        let ptr = unsafe { alloc::alloc(layout) };
        if ptr.is_null() {
            alloc::handle_alloc_error(layout);
        }
        Self { ptr, layout }
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is from alloc(); we own the full C3D_CHUNK_BYTES range.
        // The decoder writes every output byte before we read it (the lib.rs
        // doctest of Decoder confirms this; see vendor/c3d.c §T9 zero-fill).
        unsafe { std::slice::from_raw_parts(self.ptr, C3D_CHUNK_BYTES) }
    }
}

impl Drop for AlignedScratch {
    fn drop(&mut self) {
        // SAFETY: ptr came from alloc() with this exact layout and was not
        // dealloc'd elsewhere.
        unsafe { alloc::dealloc(self.ptr, self.layout) };
    }
}

#[derive(Debug)]
pub enum C3dError {
    /// Input is missing the `C3DC` magic.
    NotC3d,
    /// `c3d_chunk_validate` rejected the bitstream.
    ValidationFailed,
}

impl fmt::Display for C3dError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            C3dError::NotC3d => write!(f, "input is not a c3d chunk (missing C3DC magic)"),
            C3dError::ValidationFailed => write!(f, "c3d_chunk_validate rejected the bitstream"),
        }
    }
}

impl std::error::Error for C3dError {}

/// Magic-byte sniff (`C3DC` at offset 0).
#[inline]
pub fn is_c3d(buf: &[u8]) -> bool {
    ffi::c3d_is_chunk_inline(buf)
}

pub struct Decoder {
    ptr: *mut ffi::c3d_decoder,
    scratch: AlignedScratch,
}

// libc3d explicitly documents that one decoder is not thread-safe (see
// vendor/c3d.h:158). Don't implement Send/Sync — callers should use one
// `Decoder` per thread (see [`with_decoder`]).
impl Decoder {
    pub fn new() -> Self {
        install_panic_hook();
        let ptr = unsafe { ffi::c3d_decoder_new() };
        assert!(!ptr.is_null(), "c3d_decoder_new returned null");
        Self {
            ptr,
            scratch: AlignedScratch::new(),
        }
    }

    pub fn set_denoise(&mut self, enabled: bool) {
        unsafe { ffi::c3d_decoder_set_denoise(self.ptr, enabled) }
    }

    /// Decode a 256³ chunk. Returns a freshly-allocated 16 MiB `Vec<u8>`.
    pub fn decode(&mut self, compressed: &[u8]) -> Result<Vec<u8>, C3dError> {
        self.decode_to_scratch(compressed)?;
        Ok(self.scratch.as_slice().to_vec())
    }

    /// Decode into a caller-provided buffer. `out` must be exactly
    /// `C3D_CHUNK_BYTES` long; alignment is not required (the decoder
    /// writes into our 32-byte aligned scratch and we `copy_from_slice`).
    pub fn decode_into(&mut self, compressed: &[u8], out: &mut [u8]) -> Result<(), C3dError> {
        assert_eq!(out.len(), C3D_CHUNK_BYTES, "out must be exactly 256^3 bytes");
        self.decode_to_scratch(compressed)?;
        out.copy_from_slice(self.scratch.as_slice());
        Ok(())
    }

    fn decode_to_scratch(&mut self, compressed: &[u8]) -> Result<(), C3dError> {
        if !is_c3d(compressed) {
            return Err(C3dError::NotC3d);
        }
        // SAFETY: validation reads at most `in_len` bytes from `compressed`.
        let ok = unsafe { ffi::c3d_chunk_validate(compressed.as_ptr(), compressed.len()) };
        if !ok {
            return Err(C3dError::ValidationFailed);
        }
        // SAFETY: validated input; scratch is 32-byte aligned and the full
        // C3D_CHUNK_BYTES; the decoder writes every output byte.
        unsafe {
            ffi::c3d_decoder_chunk_decode(
                self.ptr,
                compressed.as_ptr(),
                compressed.len(),
                self.scratch.as_mut_ptr(),
            );
        }
        Ok(())
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // SAFETY: self.ptr was obtained from c3d_decoder_new and never freed elsewhere.
        unsafe { ffi::c3d_decoder_free(self.ptr) };
    }
}

thread_local! {
    static THREAD_DECODER: RefCell<Option<Decoder>> = const { RefCell::new(None) };
}

/// Run a closure with a thread-local decoder, creating it on first use.
/// Reuses ~80 MiB of scratch across calls on the same thread.
pub fn with_decoder<R>(f: impl FnOnce(&mut Decoder) -> R) -> R {
    THREAD_DECODER.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = Some(Decoder::new());
        }
        f(slot.as_mut().unwrap())
    })
}

extern "C" fn rust_panic_hook(file: *const c_char, line: c_int, msg: *const c_char) {
    // SAFETY: c3d passes static C strings; `file` and `msg` are non-null in
    // every call site in c3d.c, but we defend against null anyway.
    let file = unsafe {
        if file.is_null() {
            "<unknown>"
        } else {
            CStr::from_ptr(file).to_str().unwrap_or("<non-utf8>")
        }
    };
    let msg = unsafe {
        if msg.is_null() {
            "<no message>"
        } else {
            CStr::from_ptr(msg).to_str().unwrap_or("<non-utf8>")
        }
    };
    // `c3d_panic` aborts after the hook returns. Log loudly so the abort isn't
    // mysterious — the caller already validated, so reaching this point means
    // an unrecoverable bug in libc3d or memory corruption.
    log::error!("c3d_panic at {}:{}: {}", file, line, msg);
    eprintln!("c3d_panic at {}:{}: {}", file, line, msg);
}

fn install_panic_hook() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // SAFETY: c3d_set_panic_hook installs a function pointer with C ABI.
        unsafe { ffi::c3d_set_panic_hook(rust_panic_hook) };
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_c3d_input() {
        let mut dec = Decoder::new();
        let bogus = b"not a c3d chunk at all";
        assert!(matches!(dec.decode(bogus), Err(C3dError::NotC3d)));
    }

    #[test]
    fn rejects_short_input() {
        assert!(!is_c3d(b""));
        assert!(!is_c3d(b"C3D"));
        assert!(is_c3d(b"C3DC"));
    }

    #[test]
    fn decodes_fixture() {
        // Stage 2 produces this fixture. Skip if it isn't checked in yet.
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/subchunk_5_3_3__0_0_0.c3dc");
        if !std::path::Path::new(path).exists() {
            eprintln!("skipping decodes_fixture: fixture not present at {path}");
            return;
        }
        let bytes = std::fs::read(path).expect("read fixture");
        assert!(is_c3d(&bytes), "fixture missing C3DC magic");

        let mut dec = Decoder::new();
        let out = dec.decode(&bytes).expect("decode");
        assert_eq!(out.len(), C3D_CHUNK_BYTES);
        // Sanity: output isn't all-zero (a degenerate decode would produce that).
        let nonzero = out.iter().filter(|&&b| b != 0).count();
        assert!(
            nonzero > 1_000_000,
            "decoded chunk looks suspicious: only {nonzero} non-zero voxels"
        );
    }
}
