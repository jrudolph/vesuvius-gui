//! Safe wrapper over `vc-delta3d-sys` (which bundles and builds the C library). Decoding is what the viewer needs;
//! [`encode`] (default `fast` preset, lossless uint8) exists to build test data.
//!
//! A vc-delta3d payload describes itself (NCF1 frame, NGI1 indexed container or legacy D3D1), so decoding
//! needs nothing but the bytes. The C library is stateless and picks SIMD kernels at runtime, so these
//! functions can be called from any thread without per-thread setup.

use std::fmt;
use vc_delta3d_sys as ffi;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Delta3dError {
    /// `vc_frame_inspect` did not recognise the payload.
    NotDelta3d,
    /// The payload decodes to a size other than the caller expected.
    SizeMismatch { expected: usize, actual: usize },
    /// `vc_decompress` rejected the payload.
    DecodeFailed,
}

impl fmt::Display for Delta3dError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Delta3dError::NotDelta3d => write!(f, "input is not a vc-delta3d payload"),
            Delta3dError::SizeMismatch { expected, actual } => {
                write!(f, "vc-delta3d payload decodes to {} bytes, expected {}", actual, expected)
            }
            Delta3dError::DecodeFailed => write!(f, "vc_decompress rejected the payload"),
        }
    }
}

impl std::error::Error for Delta3dError {}

/// Number of bytes `buf` decodes to, read from the payload's own header.
pub fn decoded_size(buf: &[u8]) -> Result<usize, Delta3dError> {
    // SAFETY: plain-old-data struct; all-zero is a valid value.
    let mut info: ffi::vc_frame_info = unsafe { std::mem::zeroed() };
    // SAFETY: src/size describe a valid slice; info is a valid out-pointer.
    let ok = unsafe { ffi::vc_frame_inspect(buf.as_ptr(), buf.len(), &mut info) };
    if ok == 0 {
        return Err(Delta3dError::NotDelta3d);
    }
    Ok(info.decoded_bytes)
}

/// Decode into `out`, which must be exactly as large as the payload's decoded size.
pub fn decode_into(buf: &[u8], out: &mut [u8]) -> Result<(), Delta3dError> {
    let size = decoded_size(buf)?;
    if size != out.len() {
        return Err(Delta3dError::SizeMismatch {
            expected: out.len(),
            actual: size,
        });
    }
    // SAFETY: out is a writable slice of exactly out.len() bytes.
    let ok = unsafe { ffi::vc_decompress(buf.as_ptr(), buf.len(), out.as_mut_ptr().cast(), out.len()) };
    if ok == 0 {
        return Err(Delta3dError::DecodeFailed);
    }
    Ok(())
}

/// Decode into a fresh buffer sized from the payload's header.
pub fn decode(buf: &[u8]) -> Result<Vec<u8>, Delta3dError> {
    let mut out = vec![0u8; decoded_size(buf)?];
    decode_into(buf, &mut out)?;
    Ok(out)
}

/// Decode a payload that must hold exactly `expected` bytes (e.g. a zarr chunk of known shape).
pub fn decode_expecting(buf: &[u8], expected: usize) -> Result<Vec<u8>, Delta3dError> {
    let mut out = vec![0u8; expected];
    decode_into(buf, &mut out)?;
    Ok(out)
}

/// Encode a C-order uint8 volume of `shape` (z, y, x) with the default `fast` preset, lossless.
/// Panics if the library rejects the input.
pub fn encode(data: &[u8], shape: [usize; 3]) -> Vec<u8> {
    encode_with(data, shape, None)
}

/// Encoder presets (`vc_preset` in `codec.h`), fastest to smallest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preset {
    Delta = 1,
    Faster = 8,
    Fast = 3,
    Balanced = 9,
    Small = 10,
    Slower = 6,
}

/// Like [`encode`], with an explicit preset.
pub fn encode_preset(data: &[u8], shape: [usize; 3], preset: Preset) -> Vec<u8> {
    encode_with(data, shape, Some(preset))
}

fn encode_with(data: &[u8], shape: [usize; 3], preset: Option<Preset>) -> Vec<u8> {
    assert_eq!(data.len(), shape.iter().product::<usize>(), "data does not match shape {:?}", shape);
    // SAFETY: plain C calls; `out` is released with vc_free after copying.
    unsafe {
        let mut opts = ffi::vc_default_options();
        opts.shape = shape.map(|s| u32::try_from(s).expect("shape fits u32"));
        if let Some(p) = preset {
            opts.preset = p as c_int;
        }
        let mut out: *mut u8 = std::ptr::null_mut();
        let mut out_size = 0usize;
        let ok = ffi::vc_compress(data.as_ptr().cast(), data.len(), &opts, &mut out, &mut out_size);
        assert_eq!(ok, 1, "vc_compress rejected a {:?} volume", shape);
        let v = std::slice::from_raw_parts(out, out_size).to_vec();
        ffi::vc_free(out.cast());
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smooth-ish synthetic volume so the predictors have something to work with.
    fn volume(side: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(side * side * side);
        for z in 0..side {
            for y in 0..side {
                for x in 0..side {
                    let f = ((x * 3 + y * 5 + z * 7) as f32 * 0.05).sin() * 60.0 + 128.0;
                    v.push((f as u8).wrapping_add(((x ^ y ^ z) & 3) as u8));
                }
            }
        }
        v
    }

    #[test]
    fn roundtrip_native_frames() {
        for side in [64usize, 128] {
            let data = volume(side);
            let enc = encode(&data, [side; 3]);
            assert_eq!(&enc[..4], b"NCF1", "side {side} should use the native frame");
            assert!(enc.len() < data.len());
            assert_eq!(decoded_size(&enc).unwrap(), data.len());
            assert_eq!(decode(&enc).unwrap(), data);
            assert_eq!(
                decode_expecting(&enc, data.len() + 1),
                Err(Delta3dError::SizeMismatch {
                    expected: data.len() + 1,
                    actual: data.len()
                })
            );
        }
    }

    #[test]
    fn roundtrip_legacy_frame() {
        // Not divisible by 32: falls back to D3D1 framing.
        let side = 40usize;
        let data = volume(side);
        let enc = encode(&data, [side; 3]);
        assert_eq!(decode(&enc).unwrap(), data);
    }

    #[test]
    fn rejects_garbage() {
        assert_eq!(decoded_size(b"definitely not a frame"), Err(Delta3dError::NotDelta3d));
        assert_eq!(decoded_size(&[]), Err(Delta3dError::NotDelta3d));
    }
}
