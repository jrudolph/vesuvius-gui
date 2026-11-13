# Vendored vc-delta3d

Source: https://github.com/ScrollPrize/vc-delta3d (MIT, see `LICENSE`).

Last sync:
- Upstream tag: `v0.3.2`
- Upstream commit: `427248122a088bad7d5b1209ea55c311de3378bb` (2026-11-12)
- Files: `src/` (whole directory, including `src/ng/`), `include/vc_delta3d/{codec,indexed}.h`,
  `LICENSE` — copied verbatim, no local edits.

The zarr v3 codec contract (`[bytes{endian: little}, vc-delta3d]`, usually inside `sharding_indexed`) comes
from upstream PR #14 (merged, part of `v0.3.2`).

To re-sync: copy those paths from an updated checkout, update the commit above, and mirror any changes to the
source list or compile definitions in `cmake/modern.cmake` / `CMakeLists.txt` into `../build.rs`.
