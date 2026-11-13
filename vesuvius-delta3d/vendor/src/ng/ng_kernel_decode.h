/* Decoder-only scalar/AVX2/AVX-512/NEON entropy and reconstruction kernels. */
#pragma once
#include "ng_kernel_common.h"
/* ---- decoding ----
   Per group of up to 32 rows: (1) row by row, the previous-row part of each prediction plus its class
   bias (A) and the class; (2) the group's symbols, in row order, 32 rANS states, and from them the rows'
   residual magnitudes for later rows' classes; (3) A and the symbols turned lane-major; (4) the lanes
   rebuild the voxels x by x (in-row taps, rounding, symbol); (5) the values back to the rows. */
static uint8_t vcn_lc_sym_one(vcn_lc_dec *D,unsigned c,unsigned lane) {
    const int tb=D->tb;
    uint32_t S=D->S[lane];
    const uint32_t en=D->T[(c<<tb)+(S&((1u<<tb)-1))];
    S=(((en>>8)&0xFFFu)+1u)*(S>>tb)+(en>>20);
    if(S<VCN_LC_RL) {
      if(D->end-D->p>=2)S=(S<<16)|vcn_rd16(D->p),D->p+=2;
      else S<<=16,D->over++;
    }
    D->S[lane]=S;return (uint8_t)en;
}
static void vcn_lc_syms_c(int n, const uint8_t *restrict cl, vcn_lc_dec *D, uint8_t *restrict sym) {
  for (int i = 0; i < n; i++) {
    sym[i]=vcn_lc_sym_one(D,cl[i],i&(VCN_LC_NS-1));
  }
}
#if VCN_HAVE_AVX2
#include "ng_entropy_permutation.h"
/* eight table loads (AVX2 gathers are slow on some CPUs) */
#include "ng_gather.h"
/* the same integers, 32 symbols per step as four vectors of eight states; the lanes that need a word
   read theirs in order */
#define NG_EATTR __attribute__((target("avx2,popcnt"),always_inline))
#define NG_ENAME vcn_lc_syms_avx2_impl
#include "ng_entropy_vector.h"
#ifdef NG_HAVE_WIDE_PRE
#define NG_EATTR __attribute__((target("avx512f,avx512bw,avx512vl,popcnt"),always_inline))
#define NG_ENAME ng_syms_register_impl
#define NG_EUNROLL 1
#include "ng_entropy_vector.h"
__attribute__((target("avx512f,avx512bw,avx512vl,popcnt"))) static void ng_syms_register(int n,const uint8_t*cl,vcn_lc_dec*D,uint8_t*sym){
    ng_syms_register_impl(n,cl,D,sym,1);
}
#endif
/* Constant-specialize only the table load. Original NG keeps its existing
 * scalar-load gather; the bank decoder can use the measured hardware gather. */
__attribute__((target("avx2,popcnt"))) static void vcn_lc_syms_avx2(int n,const uint8_t*restrict cl,vcn_lc_dec*D,uint8_t*restrict sym){
  vcn_lc_syms_avx2_impl(n,cl,D,sym,0);
}
__attribute__((target("avx2,popcnt"))) static void vcn_lc_syms_gather_avx2(int n,const uint8_t*restrict cl,vcn_lc_dec*D,uint8_t*restrict sym){
  vcn_lc_syms_avx2_impl(n,cl,D,sym,1);
}
/* transposes between the group's rows and its lanes: 8 x 8 blocks of uint16, 16 x 16 blocks of bytes */
__attribute__((target("avx2"))) static void vcn_lc_tr_a_stride(int nl, const uint16_t (*restrict R)[VCN_LC_D], uint16_t *restrict T, size_t stride) {
  for (int j0 = 0; j0 < nl; j0 += 8)
    for (int x0 = 0; x0 < VCN_LC_D; x0 += 8) {
      __m128i a[8], t[8], u[8];
      for (int r = 0; r < 8; r++) a[r] = _mm_loadu_si128((const __m128i *)(R[j0 + r] + x0));
      for (int r = 0; r < 8; r += 2) t[r] = _mm_unpacklo_epi16(a[r], a[r + 1]), t[r + 1] = _mm_unpackhi_epi16(a[r], a[r + 1]);
      for (int r = 0; r < 8; r += 4)
        u[r] = _mm_unpacklo_epi32(t[r], t[r + 2]), u[r + 1] = _mm_unpackhi_epi32(t[r], t[r + 2]),
        u[r + 2] = _mm_unpacklo_epi32(t[r + 1], t[r + 3]), u[r + 3] = _mm_unpackhi_epi32(t[r + 1], t[r + 3]);
      for (int c = 0; c < 4; c++) {
        _mm_storeu_si128((__m128i *)(T + (x0 + 2 * c)*stride + j0), _mm_unpacklo_epi64(u[c], u[c + 4]));
        _mm_storeu_si128((__m128i *)(T + (x0 + 2 * c + 1)*stride + j0), _mm_unpackhi_epi64(u[c], u[c + 4]));
      }
    }
}
__attribute__((target("avx2"))) static void vcn_lc_tr_a(int nl, const uint16_t (*restrict R)[VCN_LC_D], uint16_t (*restrict T)[VCN_LC_NL]) {
  vcn_lc_tr_a_stride(nl,R,T[0],VCN_LC_NL);
}
__attribute__((target("avx2"))) static inline void vcn_lc_tr16(const uint8_t *src, size_t sp, __m128i *o) {
  __m128i a[16], b[16], c[16];
  for (int r = 0; r < 16; r++) a[r] = _mm_loadu_si128((const __m128i *)(src + r * sp));
  for (int i = 0; i < 8; i++) b[2 * i] = _mm_unpacklo_epi8(a[2 * i], a[2 * i + 1]), b[2 * i + 1] = _mm_unpackhi_epi8(a[2 * i], a[2 * i + 1]);
  for (int i = 0; i < 4; i++)
    c[4 * i] = _mm_unpacklo_epi16(b[4 * i], b[4 * i + 2]), c[4 * i + 1] = _mm_unpackhi_epi16(b[4 * i], b[4 * i + 2]),
    c[4 * i + 2] = _mm_unpacklo_epi16(b[4 * i + 1], b[4 * i + 3]), c[4 * i + 3] = _mm_unpackhi_epi16(b[4 * i + 1], b[4 * i + 3]);
  for (int i = 0; i < 2; i++)
    for (int k = 0; k < 4; k++)
      a[8 * i + 2 * k] = _mm_unpacklo_epi32(c[8 * i + k], c[8 * i + 4 + k]), a[8 * i + 2 * k + 1] = _mm_unpackhi_epi32(c[8 * i + k], c[8 * i + 4 + k]);
  for (int m = 0; m < 8; m++) o[2 * m] = _mm_unpacklo_epi64(a[m], a[8 + m]), o[2 * m + 1] = _mm_unpackhi_epi64(a[m], a[8 + m]);
}
__attribute__((target("avx2"))) static void vcn_lc_tr_y_width(int nl, const uint8_t (*restrict R)[VCN_LC_D], uint8_t (*restrict T)[VCN_LC_NL],int width) {
  __m128i o[16];
  for (int j0 = 0; j0 < nl; j0 += 16)
    for (int x0 = 0; x0 < width; x0 += 16) {
      vcn_lc_tr16(R[j0] + x0, VCN_LC_D, o);
      for (int c = 0; c < 16; c++) _mm_storeu_si128((__m128i *)(T[x0 + c] + j0), o[c]);
    }
}
__attribute__((target("avx2"))) static void vcn_lc_tr_y(int nl,const uint8_t R[][VCN_LC_D],uint8_t T[][VCN_LC_NL]){vcn_lc_tr_y_width(nl,R,T,VCN_LC_D);}
/* lanes back to the rows of b */
__attribute__((target("avx2"))) static void vcn_lc_tr_v(int nl, const uint8_t (*restrict T)[VCN_LC_NL], uint8_t *const *rows) {
  __m128i o[16];
  for (int j0 = 0; j0 < nl; j0 += 16) {
    const int m = nl - j0 < 16 ? nl - j0 : 16;
    for (int x0 = 0; x0 < VCN_LC_D; x0 += 16) {
      vcn_lc_tr16(T[x0] + j0, VCN_LC_NL, o);
      for (int c = 0; c < m; c++) _mm_storeu_si128((__m128i *)(rows[j0 + c] + x0), o[c]);
    }
  }
}
#endif
#if VCN_HAVE_NEON
/* NEON forms of the decoder's kernels (the same integers as the C forms). The symbols: 32 per step as eight
   vectors of four states; table loads one lane at a time; the lanes that need a word read theirs in order
   (lane l takes word popcount(mask & ((1 << l) - 1)), by a byte table lookup) */
static const uint8_t vcn_lc_perm4[16][16] = {
    {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3},   {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3},
    {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3},   {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 0, 1, 2, 3},
    {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3},   {0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3},
    {0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3},   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3},
    {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3},   {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7},
    {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7},   {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 8, 9, 10, 11},
    {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7},   {0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
    {0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};
static void vcn_lc_syms_neon(int n, const uint8_t *restrict cl, vcn_lc_dec *D, uint8_t *restrict sym) {
  const int tb = D->tb;
  const int32x4_t ptb = vdupq_n_s32(tb), ntb = vdupq_n_s32(-tb);
  const uint32x4_t kM = vdupq_n_u32((1u << tb) - 1u), kF = vdupq_n_u32(0xFFF), k1 = vdupq_n_u32(1), kL = vdupq_n_u32(VCN_LC_RL);
  static const uint32_t kb[4] = {1, 2, 4, 8};
  const uint32x4_t kbit = vld1q_u32(kb);
  uint32x4_t S[8];
  for (int h = 0; h < 8; h++) S[h] = vld1q_u32(D->S + 4 * h);
  const uint32_t *T = D->T;
  const uint8_t *p = D->p, *end = D->end;
  size_t over = D->over;
  for (int i = 0; i < n; i += 32) {
    const uint8x16_t ca = vld1q_u8(cl + i), cb = vld1q_u8(cl + i + 16);
    const uint16x8_t cw[4] = {vmovl_u8(vget_low_u8(ca)), vmovl_high_u8(ca), vmovl_u8(vget_low_u8(cb)), vmovl_high_u8(cb)};
    uint16x4_t y[8];
    for (int h = 0; h < 8; h++) {
      const uint32x4_t cv = (h & 1) ? vmovl_high_u16(cw[h >> 1]) : vmovl_u16(vget_low_u16(cw[h >> 1]));
      const uint32x4_t ix = vaddq_u32(vshlq_u32(cv, ptb), vandq_u32(S[h], kM));
      uint32x4_t en = vdupq_n_u32(0);
      en = vld1q_lane_u32(T + vgetq_lane_u32(ix, 0), en, 0);
      en = vld1q_lane_u32(T + vgetq_lane_u32(ix, 1), en, 1);
      en = vld1q_lane_u32(T + vgetq_lane_u32(ix, 2), en, 2);
      en = vld1q_lane_u32(T + vgetq_lane_u32(ix, 3), en, 3);
      uint32x4_t s2 = vmlaq_u32(vshrq_n_u32(en, 20), vaddq_u32(vandq_u32(vshrq_n_u32(en, 8), kF), k1), vshlq_u32(S[h], ntb));
      const uint32x4_t m = vcltq_u32(s2, kL);
      const unsigned bits = vaddvq_u32(vandq_u32(m, kbit));
      if (end - p >= 8) { /* no branch on the mask: an empty mask reads nothing */
        const uint32x4_t w = vmovl_u16(vreinterpret_u16_u8(vld1_u8(p)));
        const uint32x4_t wv = vreinterpretq_u32_u8(vqtbl1q_u8(vreinterpretq_u8_u32(w), vld1q_u8(vcn_lc_perm4[bits])));
        s2 = vbslq_u32(m, vorrq_u32(vshlq_n_u32(s2, 16), wv), s2);
        p += 2 * (unsigned)__builtin_popcount(bits);
      } else if (bits) {
        uint32_t sv[4];
        vst1q_u32(sv, s2);
        for (int k = 0; k < 4; k++)
          if (bits >> k & 1u) {
            if (end - p >= 2) sv[k] = (sv[k] << 16) | vcn_rd16(p), p += 2;
            else sv[k] <<= 16, over++;
          }
        s2 = vld1q_u32(sv);
      }
      S[h] = s2;
      y[h] = vmovn_u32(en);
    }
    for (int q = 0; q < 4; q++) vst1_u8(sym + i + 8 * q, vmovn_u16(vcombine_u16(y[2 * q], y[2 * q + 1])));
  }
  for (int h = 0; h < 8; h++) vst1q_u32(D->S + 4 * h, S[h]);
  D->p = p, D->over = over;
}
/* rebuild, 8 lanes per vector: NH vectors (2 when the group has at most 16 rows, else 4) */
/* transposes between the group's rows and its lanes, 8 x 8 blocks of uint16 and 16 x 16 blocks of bytes:
   element pairs, then groups of 2, 4 (and 8) elements */
static void vcn_lc_tr_a_neon(int nl, const uint16_t (*restrict R)[VCN_LC_D], uint16_t (*restrict T)[VCN_LC_NL]) {
  for (int j0 = 0; j0 < nl; j0 += 8)
    for (int x0 = 0; x0 < VCN_LC_D; x0 += 8) {
      uint16x8_t a[8];
      uint32x4_t b[8];
      for (int r = 0; r < 8; r++) a[r] = vld1q_u16(R[j0 + r] + x0);
      for (int i = 0; i < 4; i++) { /* b[2i + q]: rows 2i, 2i + 1 at x = q (mod 2) */
        const uint16x8x2_t t = vtrnq_u16(a[2 * i], a[2 * i + 1]);
        b[2 * i] = vreinterpretq_u32_u16(t.val[0]), b[2 * i + 1] = vreinterpretq_u32_u16(t.val[1]);
      }
      for (int i = 0; i < 2; i++) /* a[4i + q]: rows 4i..4i+3 at x = q (low half) and q + 4 (high half) */
        for (int par = 0; par < 2; par++) {
          const uint32x4x2_t t = vtrnq_u32(b[4 * i + par], b[4 * i + 2 + par]);
          a[4 * i + par] = vreinterpretq_u16_u32(t.val[0]), a[4 * i + 2 + par] = vreinterpretq_u16_u32(t.val[1]);
        }
      for (int q = 0; q < 4; q++) {
        vst1q_u16(T[x0 + q] + j0, vcombine_u16(vget_low_u16(a[q]), vget_low_u16(a[4 + q])));
        vst1q_u16(T[x0 + q + 4] + j0, vcombine_u16(vget_high_u16(a[q]), vget_high_u16(a[4 + q])));
      }
    }
}
/* o[c] = column c of the 16 x 16 block of bytes at src (row pitch sp) */
static inline void vcn_lc_tr16_neon(const uint8_t *src, size_t sp, uint8x16_t *o) {
  uint8x16_t a[16];
  uint16x8_t b[16];
  uint32x4_t c[16];
  for (int r = 0; r < 16; r++) a[r] = vld1q_u8(src + (size_t)r * sp);
  for (int i = 0; i < 8; i++) { /* b[2i + q]: rows 2i, 2i + 1 at x = q (mod 2) */
    const uint8x16x2_t t = vtrnq_u8(a[2 * i], a[2 * i + 1]);
    b[2 * i] = vreinterpretq_u16_u8(t.val[0]), b[2 * i + 1] = vreinterpretq_u16_u8(t.val[1]);
  }
  for (int i = 0; i < 4; i++) /* c[4i + q]: rows 4i..4i+3 at x = q (mod 4) */
    for (int par = 0; par < 2; par++) {
      const uint16x8x2_t t = vtrnq_u16(b[4 * i + par], b[4 * i + 2 + par]);
      c[4 * i + par] = vreinterpretq_u32_u16(t.val[0]), c[4 * i + 2 + par] = vreinterpretq_u32_u16(t.val[1]);
    }
  for (int i = 0; i < 2; i++) /* a[8i + q]: rows 8i..8i+7 at x = q (low half) and q + 8 (high half) */
    for (int q = 0; q < 4; q++) {
      const uint32x4x2_t t = vtrnq_u32(c[8 * i + q], c[8 * i + 4 + q]);
      a[8 * i + q] = vreinterpretq_u8_u32(t.val[0]), a[8 * i + 4 + q] = vreinterpretq_u8_u32(t.val[1]);
    }
  for (int q = 0; q < 8; q++)
    o[q] = vcombine_u8(vget_low_u8(a[q]), vget_low_u8(a[8 + q])), o[q + 8] = vcombine_u8(vget_high_u8(a[q]), vget_high_u8(a[8 + q]));
}
/* lanes back to the rows of b */
static void vcn_lc_tr_v_neon(int nl, const uint8_t (*restrict T)[VCN_LC_NL], uint8_t *const *rows) {
  uint8x16_t o[16];
  for (int j0 = 0; j0 < nl; j0 += 16) {
    const int m = nl - j0 < 16 ? nl - j0 : 16;
    for (int x0 = 0; x0 < VCN_LC_D; x0 += 16) {
      vcn_lc_tr16_neon(T[x0] + j0, VCN_LC_NL, o);
      for (int c = 0; c < m; c++) vst1q_u8(rows[j0 + c] + x0, o[c]);
    }
  }
}
#endif
