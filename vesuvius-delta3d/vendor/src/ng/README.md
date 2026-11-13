# Retained NG/Delta kernels

Derived from Chase Hendrick's MIT-licensed NG implementation; see LICENSE.
Delta3D integration and directional prediction retain Sean Johnson's attribution
in the repository AUTHORS.md. The pre-reduction implementation is preserved on
the `feature/ng-presets` branch.

## Encoder policies

Public IDs are defined in `include/vc_delta3d/codec.h`:

| Preset | Policy |
|---|---|
| faster, fast, balanced | Occupancy-aware screening; one selected predictor for nonzero blocks; DP64 entropy thresholds. They differ in neighbourhood, weights kept and fit sampling (`ng_presets.c`). |
| small | Block modes: two models, each refitted on the blocks that select it; one entropy bank per model. |
| slower | Independent-model menu search, selected-tap fits and dense regularized sufficient statistics, with the full neighbourhood. |
| medium | From 0.3.x: direct, NG and a screened third candidate with independent residual statistics; one reselection; compare the screened singleton. |
| slow | From 0.3.x: the menu search of slower with the smaller neighbourhood and coefficient budget. |

[COMPRESSION.md](../../COMPRESSION.md) explains the presets in more detail.

All emit NCF1 and use 4³ selector blocks. The decoder retains the frame's
2/4/8/16/32 selector sizes; block size is a frame parameter, not global metadata.
Zero blocks use the existing selector alphabet. Models and residual banks are
referenced separately; a choice may share either without duplicating it.

All selector maps use the unified codec in `selector.c`, including constants.
They follow the compact rANS model/entropy header and precede voxel residuals.
There are no historical selector markers, Zstd headers or experimental model
versions. Only old Delta3D compatibility/fallback is retained. See
[FORMAT.md](../../FORMAT.md) for the frame, header and selector layout.

There is no streaming state, adaptive partition search, tree entropy coder,
LPC, tooslow, alternative fitting/refinement or user-selectable decoder trial.
Fixed-geometry instantiations share the SAME C kernels rather than copies.
See the repository's [benchmark documentation](../../benchmark/README.md)
for validation tools.

## Source map

- `ng_presets.c`: preset table and dispatch. `ng_fast.c`, `ng_split.c`,
  `ng_medium.c`, `ng_slow.c`: the search policies.
- `ng_fit.c`, `ng_screen.c`, `ng_search.c`, `ng_select.c`,
  `ng_thresholds.c`: shared fitting, occupancy, scoring and threshold primitives.
- `ng_predict.c`: directional candidates, including SIMD row implementations.
- `ng_frame_encode.c`, `ng_frame_decode.c`: separate leaf framing paths.
- `selector.c`: only the retained whole-map context/LZ choice, shared scalar rANS.
- `header.c`: one schema walk and pooled numerical header compaction.
- `ng_stream.c`: compact tables around the shared bulk rANS implementation.
- `ng_kernel_fit.h`, `ng_kernel_encode.h`: live NG fitting and encoder kernels.
- `ng_kernel_decode.h`: live entropy and reconstruction kernels.
- `ng_kernel_common.h`: shared arithmetic, geometry, prediction and scheduling.
- `ng_native_encode.c`, `ng_native_decode.c`, `ng_native_common.c`: separate
  internal entrypoints and shared scoped state. Original standalone VCNG and
  Huffman framing/decoding paths are removed, not retained for compatibility.
- `ng_kernels.h`: combined include for fixed-geometry instantiation only.
- `ng_fixed.c`: compile-time geometry specializations (32/64/128/256).
- `ng_bank_reconstruct.h` and vector headers: integrated wavefront decoder.

Rectangular input volumes are partitioned into cubic leaves; each leaf uses
the selected preset. There is no alternate rectangular encoder policy. The
reader and low-level frame machinery still support rectangular leaves and
all documented block sizes.

Public framing/dispatch lives in `../api.c`; NGI1
indexing is split between `../indexed_encode.c` and `../indexed_decode.c`, with
shared tree geometry and expansion in `../indexed_internal.h`.
Legacy Delta3D uses shared `../delta_filter.c`, `../rans.c`, `../legacy.c`.
The public C++ headers adapt ownership and errors; they contain no codec kernels.

## Invariants

- Keep exact integer residual/sign mapping, predictor rounding and rANS order.
- No fast math; floating-point contraction is disabled for fitted weights.
- Geometry and fitting configuration are thread-local and restored by callers.
- Runtime ISA dispatch retains portable/NEON fallbacks and AVX2 without AVX-512.
- Format changes are allowed; exact decoded voxels, not old new-format bytes,
  are the invariant. Frozen old frames are imported only by the external test
  harness, never by the codec. See `benchmark/README.md` at the repository root
  for validation commands.
