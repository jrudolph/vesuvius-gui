/* Retained NG prediction, fitting and reconstruction kernels.
 * Derived from Chase Hendrick's MIT-licensed NG implementation (see LICENSE).
 * No VCNG framing or retired Huffman codec is included. */
#pragma once

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "ng_geometry.h"

typedef enum {
  VCN_OK = 0,
  VCN_ERR_ARG = -1,
  VCN_ERR_CAP = -2,
  VCN_ERR_CORRUPT = -3,
  VCN_ERR_NOMEM = -4,
  VCN_ERR_VERSION = -6 /* unsupported internal parameter layout */
} vcn_status;

/* ======================================================================== */
/*                               implementation                             */
/* ======================================================================== */

/* No floating-point contraction in the implementation. The encoder's decisions are float, and a
 * compiler that fuses a * b + c into one multiply-add (one rounding instead of two) changes some of them,
 * so the streams would depend on the flags: GCC fuses by default in its GNU C modes whenever FMA is
 * available (-mfma, -march=native, every aarch64 target); before this region existed, a GCC -mfma or
 * -march=native build of the tests gave a different stream digest. GCC: optimize("fp-contract=off")
 * for every function defined here; Clang: FP_CONTRACT OFF, restored at the end of the header. Not covered:
 * Clang's -ffp-contract=fast (it ignores the pragma; -ffp-contract=fast-honor-pragmas keeps it), and
 * -ffast-math or -Ofast with any compiler. The decoders are integer only and unaffected. */

/* Process-wide flags that are set on first use (CPU features, the decoder choice): relaxed atomic loads
 * and stores, so threads that make their first call at the same time do not race on them. */
#if defined(__GNUC__) || defined(__clang__)
#define VCN_LD(p) __atomic_load_n((p), __ATOMIC_RELAXED)
#define VCN_ST(p, v) __atomic_store_n((p), (v), __ATOMIC_RELAXED)
#else
#define VCN_LD(p) (*(p))
#define VCN_ST(p, v) (*(p) = (v))
#endif

#if defined(__GNUC__) || defined(__clang__)
#define VCN_INLINE __attribute__((always_inline)) static inline
#define VCN_MAYBE_UNUSED __attribute__((unused)) /* plain C kernels that a SIMD build may not call */
#else
#define VCN_INLINE static inline
#define VCN_MAYBE_UNUSED
#endif
/* ---- small helpers ---- */
VCN_INLINE int32_t vcn_clamp32(int64_t v, int32_t lo, int32_t hi) {
  return v < lo ? lo : (v > hi ? hi : (int32_t)v);
}
static inline void vcn_wr16(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8);
}
#include "ng_entropy_state.h"
/* little-endian 32-bit loads and stores (one move on little-endian targets) */
static inline void vcn_wr32(uint8_t *p, uint32_t v) {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  memcpy(p, &v, 4);
#else
  for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i));
#endif
}
static inline uint32_t vcn_rd32(const uint8_t *p) {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  uint32_t v;
  memcpy(&v, p, 4);
  return v;
#else
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
#endif
}

/* AVX2 kernels on x86-64 with GCC or Clang, chosen at run time; the plain C kernels give the same
 * integers. */
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && !defined(VCN_NO_AVX2)
#include <immintrin.h>
#define VCN_HAVE_AVX2 1
static inline int vcn_use_avx2(void) {
  static int v = -1;
  if (ng_disable_avx2) return 0;
  int r = VCN_LD(&v);
  if (r < 0) {
    r = __builtin_cpu_supports("avx2") ? 1 : 0;
    VCN_ST(&v, r);
  }
  return r;
}
#else
#define VCN_HAVE_AVX2 0
static inline int vcn_use_avx2(void) { return 0; }
#endif

/* NEON on AArch64 (every AArch64 CPU has it, so there is no run-time check): the mode 6 kernels that have AVX2
 * forms also have NEON forms, which give the plain C integers (the test digest compares the builds).
 * -DVCN_NO_NEON or -DVCN_NO_SIMD builds the plain C kernels instead. */
#if defined(__aarch64__) && defined(__AARCH64EL__) && defined(__ARM_NEON) && (defined(__GNUC__) || defined(__clang__)) && \
    !defined(VCN_NO_NEON) && !defined(VCN_NO_SIMD)
#include <arm_neon.h>
#define VCN_HAVE_NEON 1
#else
#define VCN_HAVE_NEON 0
#endif
#define VCN_LC_VERSION 3u /* Internal parameter layout, not a standalone frame. */
#define VCN_LC_D NG_X
#define VCN_LC_Y NG_Y
#define VCN_LC_Z NG_Z
#define VCN_LC_XP 4                                                 /* x pad each side */
#define VCN_LC_YP 4                                                 /* y pad each side */
#define VCN_LC_ZP 3                                                 /* z pad before */
#define VCN_LC_RP (VCN_LC_D + 2 * VCN_LC_XP)                        /* row pitch 136 */
#define VCN_LC_SP (VCN_LC_RP * (VCN_LC_Y + 2 * VCN_LC_YP))          /* slice pitch */
#define VCN_LC_NB ((size_t)VCN_LC_SP * (VCN_LC_Z + VCN_LC_ZP) + 64)         /* (+ slack for 32-byte loads) */
#define VCN_LC_IDX(z, y, x) ((size_t)((z) + VCN_LC_ZP) * VCN_LC_SP + (size_t)((y) + VCN_LC_YP) * VCN_LC_RP + (size_t)((x) + VCN_LC_XP))
#define VCN_LC_MAXT 160
#define VCN_LC_MAXF (2 * VCN_LC_MAXT + 1) /* fit-only: taps, tap×activity, activity */
#define VCN_LC_MAXK 128 /* up to four independent entropy banks of32 classes */
#define VCN_LC_TB 12            /* table bits written by the encoder */
#define VCN_LC_RL (1u << 15)    /* rANS state lower bound; states stay below 2^31 */
#define VCN_LC_NL 32            /* rows rebuilt together */
#define VCN_LC_NS 32            /* rANS states (symbol x of a row uses state x mod 32) */
typedef struct {
  int r[3];             /* radius for dz = 0, -1, -2 (-1: none) */
  int nt, nrow;         /* taps; of them the first nrow are in-row (dx = -1..-r0) */
  int dz[VCN_LC_MAXT], dy[VCN_LC_MAXT], dx[VCN_LC_MAXT];
  ptrdiff_t off[VCN_LC_MAXT];
} vcn_lc_taps;
static void vcn_lc_build(vcn_lc_taps *t, int r0, int r1, int r2) {
  t->r[0] = r0, t->r[1] = r1, t->r[2] = r2;
  int n = 0;
  for (int k = 1; k <= r0; k++) t->dz[n] = 0, t->dy[n] = 0, t->dx[n] = -k, n++;
  t->nrow = n;
  for (int dz = -2; dz <= 0; dz++) {
    int r = t->r[-dz];
    if (r < 0) continue;
    for (int dy = -r; dy <= (dz ? r : -1); dy++)
      for (int dx = -r; dx <= r; dx++) t->dz[n] = dz, t->dy[n] = dy, t->dx[n] = dx, n++;
  }
  t->nt = n;
  for (int i = 0; i < n; i++) t->off[i] = (ptrdiff_t)t->dz[i] * VCN_LC_SP + (ptrdiff_t)t->dy[i] * VCN_LC_RP + t->dx[i];
}
/* pads, row by row (valid in raster order and in the decoder's wavefront order) */
static inline void vcn_lc_row_begin(uint8_t *b, int z, int y, uint8_t flat) {
  if (y == 0) /* top pad rows of slice z := row 0 of slice z-1 with its pads, or flat */
    for (int yy = -VCN_LC_YP; yy < 0; yy++) {
      uint8_t *r = b + VCN_LC_IDX(z, yy, -VCN_LC_XP);
      if (z > 0) memcpy(r, b + VCN_LC_IDX(z - 1, 0, -VCN_LC_XP), VCN_LC_RP);
      else memset(r, flat, VCN_LC_RP);
    }
  uint8_t *r = b + VCN_LC_IDX(z, y, 0);
  memset(r - VCN_LC_XP, r[-VCN_LC_RP], VCN_LC_XP); /* left pad := the row above's first voxel */
}
static inline void vcn_lc_row_end(uint8_t *b, int z, int y) {
  uint8_t *r = b + VCN_LC_IDX(z, y, 0);
  memset(r - VCN_LC_XP, r[0], VCN_LC_XP);
  memset(r + VCN_LC_D, r[VCN_LC_D - 1], VCN_LC_XP);
  int y1 = y, y2 = y;
  if (y == VCN_LC_Y - 1) {
    for (int yy = VCN_LC_Y; yy < VCN_LC_Y + VCN_LC_YP; yy++) memcpy(b + VCN_LC_IDX(z, yy, -VCN_LC_XP), r - VCN_LC_XP, VCN_LC_RP);
    y2 = VCN_LC_Y + VCN_LC_YP - 1;
  }
  if (y == 0) y1 = -VCN_LC_YP;
  if (z == 0) /* slices before 0 := slice 0 */
    for (int k = 1; k <= VCN_LC_ZP; k++)
      for (int yy = y1; yy <= y2; yy++) memcpy(b + VCN_LC_IDX(-k, yy, -VCN_LC_XP), b + VCN_LC_IDX(0, yy, -VCN_LC_XP), VCN_LC_RP);
}
/* context activity (all but W, WW) from |e| buffer */
static void vcn_lc_apre(const uint8_t *e, size_t r0i, uint16_t *restrict a);
static void ng_apre_scalar(const uint8_t *e, size_t r0i, uint16_t *restrict a) {
  const uint8_t *restrict n = e + r0i - VCN_LC_RP, *restrict nn = e + r0i - 2 * VCN_LC_RP, *restrict u = e + r0i - VCN_LC_SP,
                          *restrict uu = e + r0i - 2 * VCN_LC_SP;
  for (int x = 0; x < VCN_LC_D; x++)
    a[x] = (uint16_t)(2 * n[x] + 2 * u[x] + n[x - 1] + n[x + 1] + u[x - 1] + u[x + 1] + u[x - VCN_LC_RP] + u[x + VCN_LC_RP] +
                      nn[x] + uu[x]);
}
#define VCN_LC_AMAX 2560 /* activity < 14 * 129 */
typedef struct {
  vcn_lc_taps t;
  int K, tb;
  int32_t w[2][VCN_LC_MAXT], bias[2];
  uint16_t thr[VCN_LC_MAXK];
  int32_t cb[VCN_LC_MAXK]; /* class bias, 1/256 */
  uint8_t lut[VCN_LC_AMAX];
} vcn_lc_par;
static void vcn_lc_mklut(vcn_lc_par *p) {
  for (int a = 0; a < VCN_LC_AMAX; a++) {
    int c = 0;
    for (int k = 0; k < p->K - 1; k++) c += a > p->thr[k];
    p->lut[a] = (uint8_t)c;
  }
}
/* one row's classes from the activity, and A[x] += the class bias: kc[x] = lut[activity], as uint8 */
static void vcn_lc_cls_c(const uint8_t *e, size_t ri, const vcn_lc_par *p, uint16_t *restrict A, uint8_t *restrict kc) {
  _Alignas(32) uint16_t ap[VCN_LC_D];
  vcn_lc_apre(e, ri, ap);
  for (int x = 0; x < VCN_LC_D; x++) {
    const int c = kc[x] = p->lut[ap[x]];
    int32_t bias=p->cb[c];

    A[x] = (uint16_t)(A[x] + (uint16_t)bias);
  }
}
#if VCN_HAVE_AVX2
/* Shared exact activity expression for threshold and lookup classifiers. */
#define NG_AATTR __attribute__((target("avx2"),always_inline))
#define NG_AV __m256i
#define NG_ANAME vcn_lc_activity16
#define NG_ALOAD(Q) _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(Q)))
#define NG_AADD _mm256_add_epi16
#define NG_ASHIFT _mm256_slli_epi16
#include "ng_activity_vector.h"
#if (defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 11) || (defined(__clang__) && __clang_major__ >= 12)
#define NG_AATTR __attribute__((target("avx512f,avx512bw"),always_inline))
#define NG_AV __m512i
#define NG_ANAME vcn_lc_activity32
#define NG_ALOAD(Q) _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(Q)))
#define NG_AADD _mm512_add_epi16
#define NG_ASHIFT _mm512_slli_epi16
#include "ng_activity_vector.h"
#endif
/* the same for K <= 16: the class as the count of thresholds below the activity, the bias by byte shuffles */
__attribute__((target("avx2"),always_inline)) static inline void vcn_lc_cls_avx2_impl(const uint8_t *e, size_t ri, const vcn_lc_par *p, uint16_t *restrict A,
                                                            uint8_t *restrict kc,uint16_t*activity) {
  __m256i th[VCN_LC_MAXK];
  const int nk = p->K - 1;
  for (int k = 0; k < nk; k++) th[k] = _mm256_set1_epi16((short)p->thr[k]);
  uint8_t lo[16] = {0}, hi[16] = {0};
  for (int k = 0; k < p->K; k++) lo[k] = (uint8_t)p->cb[k], hi[k] = (uint8_t)((uint32_t)p->cb[k] >> 8);
  const __m256i tlo = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)lo)),
                thi = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)hi));
#define VCN_LC_ACT(x) vcn_lc_activity16(e,ri+(x))
  int tmax = 0;
  for (int k = 0; k < nk; k++) tmax = p->thr[k] > tmax ? p->thr[k] : tmax;
  if (tmax < 255) {
    /* every threshold below 255: a > t equals min(a, 255) > t, so compare saturated bytes, 32 voxels at a
       time (unsigned order by the sign-bit flip). packus leaves the bytes in the order a[0..7], b[0..7],
       a[8..15], b[8..15]; the unpacks of the bias bytes undo it, the permute puts the classes in order */
    __m256i t8[VCN_LC_MAXK];
    const __m256i kx = _mm256_set1_epi8((char)0x80);
    for (int k = 0; k < nk; k++) t8[k] = _mm256_set1_epi8((char)(p->thr[k] ^ 0x80));
    for (int x = 0; x < VCN_LC_D; x += 32) {
      const __m256i a0=VCN_LC_ACT(x),a1=VCN_LC_ACT(x+16);
      if(activity){_mm256_storeu_si256((__m256i*)(activity+x),a0);_mm256_storeu_si256((__m256i*)(activity+x+16),a1);}
      const __m256i a8 = _mm256_xor_si256(_mm256_packus_epi16(a0,a1), kx);
      __m256i c0 = _mm256_setzero_si256(), c1 = c0;
      int k = 0;
      for (; k + 1 < nk; k += 2)
        c0 = _mm256_sub_epi8(c0, _mm256_cmpgt_epi8(a8, t8[k])), c1 = _mm256_sub_epi8(c1, _mm256_cmpgt_epi8(a8, t8[k + 1]));
      if (k < nk) c0 = _mm256_sub_epi8(c0, _mm256_cmpgt_epi8(a8, t8[k]));
      const __m256i c = _mm256_add_epi8(c0, c1);
      const __m256i bl = _mm256_shuffle_epi8(tlo, c), bh = _mm256_shuffle_epi8(thi, c);
      _mm256_storeu_si256((__m256i *)(A + x), _mm256_add_epi16(_mm256_loadu_si256((const __m256i *)(A + x)), _mm256_unpacklo_epi8(bl, bh)));
      _mm256_storeu_si256((__m256i *)(A + x + 16),
                          _mm256_add_epi16(_mm256_loadu_si256((const __m256i *)(A + x + 16)), _mm256_unpackhi_epi8(bl, bh)));
      _mm256_storeu_si256((__m256i *)(kc + x), _mm256_permute4x64_epi64(c, 0xD8));
    }
    return;
  }
  for (int x = 0; x < VCN_LC_D; x += 16) {
    const __m256i a = VCN_LC_ACT(x);
    if(activity)_mm256_storeu_si256((__m256i*)(activity+x),a);
    __m256i c0 = _mm256_setzero_si256(), c1 = c0;
    int k = 0;
    for (; k + 1 < nk; k += 2)
      c0 = _mm256_sub_epi16(c0, _mm256_cmpgt_epi16(a, th[k])), c1 = _mm256_sub_epi16(c1, _mm256_cmpgt_epi16(a, th[k + 1]));
    if (k < nk) c0 = _mm256_sub_epi16(c0, _mm256_cmpgt_epi16(a, th[k]));
    const __m256i c = _mm256_add_epi16(c0, c1); /* classes, 0..15, in 16-bit lanes */
    const __m256i ix = _mm256_or_si256(c, _mm256_set1_epi16((short)0x8000)); /* the high bytes select zero */
    const __m256i cb = _mm256_or_si256(_mm256_shuffle_epi8(tlo, ix), _mm256_slli_epi16(_mm256_shuffle_epi8(thi, ix), 8));
    _mm256_storeu_si256((__m256i *)(A + x), _mm256_add_epi16(_mm256_loadu_si256((const __m256i *)(A + x)), cb));
    const __m128i c8 = _mm_packus_epi16(_mm256_castsi256_si128(c), _mm256_extracti128_si256(c, 1));
    _mm_storeu_si128((__m128i *)(kc + x), c8);
  }
#undef VCN_LC_ACT
}
__attribute__((target("avx2"))) static void vcn_lc_cls_avx2(const uint8_t *e,size_t ri,const vcn_lc_par*p,uint16_t*restrict A,uint8_t*restrict kc){
  vcn_lc_cls_avx2_impl(e,ri,p,A,kc,NULL);
}
__attribute__((target("avx2"))) static void ng_cls_export_avx2(const uint8_t*e,size_t ri,const vcn_lc_par*p,uint16_t*A,uint8_t*kc,uint16_t*activity){
  vcn_lc_cls_avx2_impl(e,ri,p,A,kc,activity);
}
#define VCN_LC_CLS(e, ri, p, A, kc) ((vcn_use_avx2() && (p)->K <= 16) ? vcn_lc_cls_avx2(e, ri, p, A, kc) : vcn_lc_cls_c(e, ri, p, A, kc))
#elif VCN_HAVE_NEON
/* the same for K <= 16 with NEON: the activity in 16-bit lanes, the class as the count of thresholds below it
   (saturated bytes, 16 voxels at a time, when every threshold is below 255; else 16-bit compares), the bias
   by table lookups of its low and high bytes */
static void vcn_lc_cls_neon(const uint8_t *e, size_t ri, const vcn_lc_par *p, uint16_t *restrict A, uint8_t *restrict kc) {
  const uint8_t *n = e + ri - VCN_LC_RP, *nn = e + ri - 2 * VCN_LC_RP, *u = e + ri - VCN_LC_SP, *uu = e + ri - 2 * VCN_LC_SP;
  const int nk = p->K - 1;
  uint8_t lo[16] = {0}, hi[16] = {0};
  for (int k = 0; k < p->K; k++) lo[k] = (uint8_t)p->cb[k], hi[k] = (uint8_t)((uint32_t)p->cb[k] >> 8);
  const uint8x16_t tlo = vld1q_u8(lo), thi = vld1q_u8(hi);
  int tmax = 0;
  for (int k = 0; k < nk; k++) tmax = p->thr[k] > tmax ? p->thr[k] : tmax;
  uint8x16_t t8[VCN_LC_MAXK];
  uint16x8_t t16[VCN_LC_MAXK];
  for (int k = 0; k < nk; k++) t8[k] = vdupq_n_u8((uint8_t)(p->thr[k] < 255 ? p->thr[k] : 255)), t16[k] = vdupq_n_u16(p->thr[k]);
  for (int x = 0; x < VCN_LC_D; x += 16) {
    const uint8x16_t N = vld1q_u8(n + x), U = vld1q_u8(u + x), Nw = vld1q_u8(n + x - 1), Ne = vld1q_u8(n + x + 1),
                     Uw = vld1q_u8(u + x - 1), Ue = vld1q_u8(u + x + 1), Un = vld1q_u8(u + x - VCN_LC_RP),
                     Us = vld1q_u8(u + x + VCN_LC_RP), NN = vld1q_u8(nn + x), UU = vld1q_u8(uu + x);
    uint16x8_t al = vaddl_u8(vget_low_u8(N), vget_low_u8(U)), ah = vaddl_high_u8(N, U);
    al = vaddq_u16(al, al), ah = vaddq_u16(ah, ah);
    al = vaddq_u16(al, vaddq_u16(vaddl_u8(vget_low_u8(Nw), vget_low_u8(Ne)), vaddl_u8(vget_low_u8(Uw), vget_low_u8(Ue))));
    ah = vaddq_u16(ah, vaddq_u16(vaddl_high_u8(Nw, Ne), vaddl_high_u8(Uw, Ue)));
    al = vaddq_u16(al, vaddq_u16(vaddl_u8(vget_low_u8(Un), vget_low_u8(Us)), vaddl_u8(vget_low_u8(NN), vget_low_u8(UU))));
    ah = vaddq_u16(ah, vaddq_u16(vaddl_high_u8(Un, Us), vaddl_high_u8(NN, UU)));
    uint8x16_t c = vdupq_n_u8(0);
    if (tmax < 255) { /* a > t equals min(a, 255) > t for t < 255 */
      const uint8x16_t a8 = vcombine_u8(vqmovn_u16(al), vqmovn_u16(ah));
      for (int k = 0; k < nk; k++) c = vsubq_u8(c, vcgtq_u8(a8, t8[k]));
    } else {
      uint16x8_t cl = vdupq_n_u16(0), ch = cl;
      for (int k = 0; k < nk; k++) cl = vsubq_u16(cl, vcgtq_u16(al, t16[k])), ch = vsubq_u16(ch, vcgtq_u16(ah, t16[k]));
      c = vcombine_u8(vmovn_u16(cl), vmovn_u16(ch));
    }
    const uint8x16_t bl = vqtbl1q_u8(tlo, c), bh = vqtbl1q_u8(thi, c);
    vst1q_u16(A + x, vaddq_u16(vld1q_u16(A + x), vreinterpretq_u16_u8(vzip1q_u8(bl, bh))));
    vst1q_u16(A + x + 8, vaddq_u16(vld1q_u16(A + x + 8), vreinterpretq_u16_u8(vzip2q_u8(bl, bh))));
    vst1q_u8(kc + x, c);
  }
}
#define VCN_LC_CLS(e, ri, p, A, kc) (((p)->K <= 16) ? vcn_lc_cls_neon(e, ri, p, A, kc) : vcn_lc_cls_c(e, ri, p, A, kc))
#else
#define VCN_LC_CLS(e, ri, p, A, kc) vcn_lc_cls_c(e, ri, p, A, kc)
#endif

#if VCN_HAVE_AVX2
__attribute__((target("avx512f,avx512bw,avx2")))
static void ng_apre_wide(const uint8_t*e,size_t ri,uint16_t*a){
  for(int x=0;x<VCN_LC_D;x+=32)_mm512_storeu_si512(a+x,vcn_lc_activity32(e,ri+x));
}
#endif
static void vcn_lc_apre(const uint8_t*e,size_t ri,uint16_t*restrict a){
#if VCN_HAVE_AVX2
  if(ng_use_avx512()&&__builtin_cpu_supports("avx512bw")){
    ng_apre_wide(e,ri,a);return;
  }
#endif
  ng_apre_scalar(e,ri,a);
}
/* previous-row taps: offsets and weights, and the same as pairs of signed 7-bit digits for vpmaddubsw,
   w = d0 + 128 d1 + 16384 d2 (mod 2^16), pairs ordered by the digits they need */
typedef struct {
  int np;
  ptrdiff_t o[VCN_LC_MAXT + 1];
  uint16_t w[VCN_LC_MAXT + 1];
  int npair, nlev[3];         /* pairs; of them nlev[0] need 1 digit, nlev[1] up to 2 */
  ptrdiff_t po[VCN_LC_MAXT / 2 + 1][2];
  int16_t pd[VCN_LC_MAXT / 2 + 1][3];
} vcn_lc_plist;
static void vcn_lc_mkplist(const vcn_lc_taps *t, const int32_t *w, vcn_lc_plist *pl) {

  int n = 0;
  int d[VCN_LC_MAXT + 1][3], lev[VCN_LC_MAXT + 1];
  for (int k = t->nrow; k < t->nt; k++)
    if (w[k]) pl->o[n] = t->off[k], pl->w[n] = (uint16_t)w[k], n++;
  pl->np = n;
  for (int k = 0; k < n; k++) {
    int x = (int16_t)pl->w[k];
    lev[k] = 1;
    for (int l = 0; l < 3; l++) {
      d[k][l] = ((x + 64) & 127) - 64;
      x = (x - d[k][l]) / 128;
      if (d[k][l]) lev[k] = l + 1;
    }
  }
  int ord[VCN_LC_MAXT + 1], m = 0;
  for (int L = 1; L <= 3; L++)
    for (int k = 0; k < n; k++)
      if (lev[k] == L) ord[m++] = k;
  if (n & 1) ord[n] = n, pl->o[n] = pl->o[0], d[n][0] = d[n][1] = d[n][2] = 0, lev[n] = 1; /* zero weight */
  pl->npair = (n + 1) / 2;
  pl->nlev[0] = pl->nlev[1] = 0;
  for (int i = 0; i < pl->npair; i++) {
    const int a = ord[2 * i], c = (2 * i + 1 < n) ? ord[2 * i + 1] : n;
    const int L = lev[a] > lev[c] ? lev[a] : lev[c];
    if (L <= 1) pl->nlev[0] = i + 1;
    if (L <= 2) pl->nlev[1] = i + 1;
    pl->po[i][0] = pl->o[a], pl->po[i][1] = pl->o[c];
    for (int l = 0; l < 3; l++) pl->pd[i][l] = (int16_t)(uint16_t)((uint8_t)d[a][l] | (unsigned)(uint8_t)d[c][l] << 8);
  }
}
/* pre[x] = bias + the previous-row taps, modulo 2^16 */
static inline void vcn_lc_pre_span(const uint8_t *row,const vcn_lc_plist *pl,uint16_t bias,uint16_t *restrict pre,int count){

  for(int x=0;x<count;x++)pre[x]=bias;
  for(int k=0;k<pl->np;k++){
    const uint8_t* restrict s=row+pl->o[k];uint16_t w=pl->w[k];
    for(int x=0;x<count;x++)pre[x]=(uint16_t)(pre[x]+(uint32_t)w*s[x]);
  }
}
static void vcn_lc_pre_c_impl(const uint8_t *row, const vcn_lc_plist *pl, uint16_t bias, uint16_t *restrict pre, unsigned needed) {
  for (int x = 0; x < VCN_LC_D; x++) pre[x] = bias;
  for(int x=0;x<VCN_LC_D;x+=32)if(needed&(1u<<(x>>5)))vcn_lc_pre_span(row+x,pl,bias,pre+x,32);
}
VCN_MAYBE_UNUSED static void vcn_lc_pre_c(const uint8_t *row,const vcn_lc_plist *pl,uint16_t bias,uint16_t *pre) {
  vcn_lc_pre_c_impl(row,pl,bias,pre,~0u);
}
#if VCN_HAVE_AVX2
#define NG_PATTR __attribute__((target("avx2")))
#define NG_PNAME vcn_lc_pre_avx2_impl
#define NG_PV __m256i
#define NG_PWIDTH 32
#define NG_PSET _mm256_set1_epi16
#define NG_PZERO _mm256_setzero_si256
#define NG_PLOAD _mm256_loadu_si256
#define NG_PADD _mm256_add_epi16
#define NG_PMADD _mm256_maddubs_epi16
#define NG_PLO _mm256_unpacklo_epi8
#define NG_PHI _mm256_unpackhi_epi8
#define NG_PSHIFT _mm256_slli_epi16
#define NG_PSTORE(P,L,H) do { \
  _mm256_storeu_si256((__m256i*)(P),_mm256_permute2x128_si256(L,H,0x20)); \
  _mm256_storeu_si256((__m256i*)((P)+16),_mm256_permute2x128_si256(L,H,0x31)); \
} while(0)
#include "ng_pre_vector.h"
#if (defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 11) || (defined(__clang__) && __clang_major__ >= 12)
#define NG_HAVE_WIDE_PRE 1
#define NG_PNAME vcn_lc_pre_avx512_impl
#include "ng_pre_wide.h"
#endif
__attribute__((target("avx2"))) static void vcn_lc_pre_avx2(const uint8_t *row,const vcn_lc_plist *pl,uint16_t bias,uint16_t *pre) {
  vcn_lc_pre_avx2_impl(row,pl,bias,pre,~0u);
}
/* Whole-weight multiplication is exact modulo 2^16, avoiding digit splitting
   when learned feature models produce large coefficients. */
__attribute__((target("avx512f,avx512bw,avx2")))
static void ng_pre_encode_multiply(const uint8_t*row,const vcn_lc_plist*pl,uint16_t bias,uint16_t*pre){
  for(int x=0;x<VCN_LC_D;x+=32){
    __m512i a=_mm512_set1_epi16((short)bias),b=_mm512_setzero_si512();
    int k=0;
    for(;k+1<pl->np;k+=2){
      __m512i v0=_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(row+x+pl->o[k])));
      __m512i v1=_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(row+x+pl->o[k+1])));
      a=_mm512_add_epi16(a,_mm512_mullo_epi16(v0,_mm512_set1_epi16((short)pl->w[k])));
      b=_mm512_add_epi16(b,_mm512_mullo_epi16(v1,_mm512_set1_epi16((short)pl->w[k+1])));
    }
    if(k<pl->np)a=_mm512_add_epi16(a,_mm512_mullo_epi16(_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(row+x+pl->o[k]))),_mm512_set1_epi16((short)pl->w[k])));
    _mm512_storeu_si512(pre+x,_mm512_add_epi16(a,b));
  }
}
static void ng_pre_encode(const uint8_t*row,const vcn_lc_plist*pl,uint16_t bias,uint16_t*pre){

  if(ng_use_avx512()&&__builtin_cpu_supports("avx512bw")){
    ng_pre_encode_multiply(row,pl,bias,pre);return;
  }
  if(vcn_use_avx2())vcn_lc_pre_avx2(row,pl,bias,pre);else vcn_lc_pre_c(row,pl,bias,pre);
}
#define VCN_LC_PRE(r, pl, b, o) ng_pre_encode(r,pl,b,o)
#elif VCN_HAVE_NEON
/* the same with 16-bit multiply-adds by the whole weight (modulo 2^16, as the C form), 32 voxels per step */
static void vcn_lc_pre_neon_impl(const uint8_t *row, const vcn_lc_plist *pl, uint16_t bias, uint16_t *restrict pre,unsigned needed) {

  for (int x = 0; x < VCN_LC_D; x += 32) {
    if(!(needed & (1u<<(x>>5)))) {memset(pre+x,0,32*sizeof(*pre));continue;}
    uint16x8_t a0 = vdupq_n_u16(bias), a1 = a0, a2 = a0, a3 = a0;
    for (int k = 0; k < pl->np; k++) {
      const uint8_t *s = row + pl->o[k] + x;
      const uint8x16_t v0 = vld1q_u8(s), v1 = vld1q_u8(s + 16);
      const uint16_t w = pl->w[k];
      a0 = vmlaq_n_u16(a0, vmovl_u8(vget_low_u8(v0)), w), a1 = vmlaq_n_u16(a1, vmovl_high_u8(v0), w);
      a2 = vmlaq_n_u16(a2, vmovl_u8(vget_low_u8(v1)), w), a3 = vmlaq_n_u16(a3, vmovl_high_u8(v1), w);
    }
    vst1q_u16(pre + x, a0), vst1q_u16(pre + x + 8, a1), vst1q_u16(pre + x + 16, a2), vst1q_u16(pre + x + 24, a3);
  }
}
static void vcn_lc_pre_neon(const uint8_t *row,const vcn_lc_plist *pl,uint16_t bias,uint16_t *pre) {
  vcn_lc_pre_neon_impl(row,pl,bias,pre,~0u);
}
#define VCN_LC_PRE(r, pl, b, o) vcn_lc_pre_neon(r, pl, b, o)
#else
#define VCN_LC_PRE(r, pl, b, o) vcn_lc_pre_c(r, pl, b, o)
#endif
/* Only omit full SIMD spans with no NG-selected voxels. Classes still use
   every residual and the causal schedule/padding are unchanged. */
static void vcn_lc_pre_selected(const uint8_t *row,const vcn_lc_plist *pl,uint16_t bias,uint16_t *pre,unsigned needed) {
#if VCN_HAVE_AVX2
  if(vcn_use_avx2()){vcn_lc_pre_avx2_impl(row,pl,bias,pre,needed);return;}
#elif VCN_HAVE_NEON
  vcn_lc_pre_neon_impl(row,pl,bias,pre,needed);return;
#endif
  vcn_lc_pre_c_impl(row,pl,bias,pre,needed);
}
/* wavefront schedule: row (z, y) runs at step y + d z; the rows of one step are independent and go in
   groups of up to 32 */
#include "ng_class_lookup.h"
#ifdef NG_HAVE_WIDE_PRE
#define NG_PNAME ng_pre_class_fused
#define NG_PFUSED 1
#include "ng_pre_wide.h"
#endif
typedef struct { int ng; uint16_t (*row)[VCN_LC_NL]; uint8_t *n; } vcn_lc_sched;
static int vcn_lc_delta(const vcn_lc_taps *t) {
  int d = t->r[1] > t->r[2] ? t->r[1] : t->r[2];
  return (d < 1 ? 1 : d) + 1;
}
static int vcn_lc_mksched(int dl, vcn_lc_sched *s) {
  s->row = (uint16_t(*)[VCN_LC_NL])malloc(sizeof *s->row * VCN_LC_Y * VCN_LC_Z);
  s->n = (uint8_t *)malloc(VCN_LC_Y * VCN_LC_Z);
  if (!s->row || !s->n) return -1;
  int g = 0;
  for (int t = 0; t <= VCN_LC_Y - 1 + dl * (VCN_LC_Z - 1); t++) {
    int k = 0;
    for (int z = 0; z < VCN_LC_Z; z++) {
      const int y = t - dl * z;
      if (y < 0) break;
      if (y >= VCN_LC_Y) continue;
      if (k == 0) s->n[g] = 0;
      s->row[g][k++] = (uint16_t)(z * VCN_LC_Y + y);
      s->n[g] = (uint8_t)k;
      if (k == VCN_LC_NL) g++, k = 0;
    }
    if (k) g++;
  }
  s->ng = g;
  return 0;
}
static int vcn_lc_read_table(const uint8_t **pp, const uint8_t *end, uint16_t *freq, uint32_t M) {
  const uint8_t *p = *pp;
  if (end - p < 2) return -1;
  unsigned n = vcn_rd16(p);
  p += 2;
  if (n == 0 || n > 256) return -1;
  uint32_t sum = 0;
  memset(freq, 0, 512);
  for (unsigned s = 0; s < n; s++) {
    uint32_t v = 0;
    unsigned sh = 0;
    for (;;) {
      if (p >= end || sh > 14) return -1;
      uint8_t by = *p++;
      v |= (uint32_t)(by & 127u) << sh;
      sh += 7;
      if (!(by & 128u)) break;
    }
    if (v > M) return -1;
    freq[s] = (uint16_t)v, sum += v;
  }
  if (sum != M) return -1;
  *pp = p;
  return 0;
}
static int vcn_lc_read_tables(const uint8_t **p,const uint8_t *end,uint32_t *T,int K,uint32_t M) {
  for(int k=0;k<K;k++) {
    uint16_t fr[256];uint32_t total=0;
    if(end-*p<512)return -1;
    for(unsigned s=0;s<256;s++,*p+=2){fr[s]=(uint16_t)vcn_rd16(*p);total+=fr[s];}
    if(total!=M)return -1;
    if(!T)continue; /* Validate/skip metadata without constructing decode tables. */
    uint32_t cum=0,*Tk=T+(size_t)k*M;
    for(unsigned s=0;s<256;s++) {
      uint32_t f=fr[s];
      for(uint32_t j=0;j<f;j++)Tk[cum+j]=s|(f-1u)<<8|j<<20;
      cum+=f;
    }
  }
  return 0;
}
static size_t vcn_lc_write_table(uint8_t* out,const uint16_t fr[256]) {
  unsigned nn=256;while(nn>1 && fr[nn-1]==0)nn--;
  vcn_wr16(out,nn);size_t pos=2;
  for(unsigned s=0;s<nn;s++) {
    uint32_t v=fr[s];do {uint8_t by=v&127u;v>>=7;out[pos++]=(uint8_t)(by|(v?128u:0u));}while(v);
  }
  return pos;
}
/* the in-row part and the rounding, as the decoder does them: pp is the predicted value, fm 1 when the
   prediction was rounded up */
#define VCN_LC_PP(P) ((uint16_t)((P) + 128u) >> 8)
#define VCN_LC_FM(P) (((P) >> 7) & 1u)
/* phase timers for the profiler (m6speed/m6_prof.c); no code by default */

static size_t vcn_lc_parameter_body_bytes(unsigned taps,unsigned classes) {
  return 2*(2*(size_t)taps+2)+2*(size_t)(classes-1)+2*(size_t)classes;
}
static vcn_status vcn_lc_readpar(const uint8_t **cursor, size_t n, vcn_lc_par **result) {
  const uint8_t *p=*cursor;
  const uint8_t *end = p + n;
  if (n < 1) return VCN_ERR_CORRUPT;
  if (p[0] != VCN_LC_VERSION) return VCN_ERR_VERSION;
  p++, n--;
  if (n < 5) return VCN_ERR_CORRUPT;
  const int r0 = p[0], r1 = (int8_t)p[1], r2 = (int8_t)p[2], K = p[3], tb = p[4];
  if (r0 < 1 || r0 > 3 || r1 < -1 || r1 > 3 || r2 < -1 || r2 > 3 || K < 2 || K > VCN_LC_MAXK || tb < 10 || tb > 12)
    return VCN_ERR_CORRUPT;
  p += 5;
  vcn_lc_par *P = (vcn_lc_par *)calloc(1, sizeof(vcn_lc_par));
  if (!P) return VCN_ERR_NOMEM;
  vcn_lc_build(&P->t, r0, r1, r2);
  P->K = K, P->tb = tb;
  const vcn_lc_taps *t = &P->t;
  const size_t need = vcn_lc_parameter_body_bytes((unsigned)t->nt,(unsigned)K);
  if ((size_t)(end - p) < need) {
    free(P);
    return VCN_ERR_CORRUPT;
  }
  for (int s2 = 0; s2 < 2; s2++) {
    for (int k = 0; k < t->nt; k++) P->w[s2][k] = (int16_t)vcn_rd16(p), p += 2;
    P->bias[s2] = (int32_t)vcn_rd16(p), p += 2;
  }
  int bad_thr = 0; /* the encoder writes thresholds below VCN_LC_AMAX; the SIMD class path needs them there */
  for (int k = 0; k < K - 1; k++) P->thr[k] = (uint16_t)vcn_rd16(p), p += 2, bad_thr |= P->thr[k] >= VCN_LC_AMAX;
  for (int k = 0; k < K; k++) P->cb[k] = (int16_t)vcn_rd16(p), p += 2;
  if (bad_thr) {
    free(P);
    return VCN_ERR_CORRUPT;
  }
  vcn_lc_mklut(P);
  *cursor=p; *result=P;
  return VCN_OK;
}
