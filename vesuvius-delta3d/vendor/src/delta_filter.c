/* Legacy Delta3D's reversible filters. Keep pass order and sampled entropy
 * arithmetic identical to the original implementation (Sean Johnson). */
#include "delta_filter.h"
#include <float.h>
#include <math.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#endif

/* Inclusive prefix sum with a carry between SIMD vectors. Unaligned loads
 * support arbitrary rows; a scalar tail handles dimensions not divisible by 16. */
static void prefix_u8(uint8_t* p, size_t x)
{
    size_t i = 0;
    uint8_t c = 0;
#if defined(__ARM_NEON)
    const uint8x16_t zero = vdupq_n_u8(0);
    uint8x16_t carry = zero;
    for (; i + 16 <= x; i += 16) {
        uint8x16_t v = vld1q_u8(p + i);
        v = vaddq_u8(v, vextq_u8(zero, v, 15));
        v = vaddq_u8(v, vextq_u8(zero, v, 14));
        v = vaddq_u8(v, vextq_u8(zero, v, 12));
        v = vaddq_u8(v, vextq_u8(zero, v, 8));
        v = vaddq_u8(v, carry);
        vst1q_u8(p + i, v);
        carry = vdupq_laneq_u8(v, 15);
    }
    c = vgetq_lane_u8(carry, 0);
#elif defined(__SSE2__)
    __m128i carry = _mm_setzero_si128();
    for (; i + 16 <= x; i += 16) {
        __m128i v = _mm_loadu_si128((const __m128i*)(p + i));
        v = _mm_add_epi8(v, _mm_slli_si128(v, 1));
        v = _mm_add_epi8(v, _mm_slli_si128(v, 2));
        v = _mm_add_epi8(v, _mm_slli_si128(v, 4));
        v = _mm_add_epi8(v, _mm_slli_si128(v, 8));
        v = _mm_add_epi8(v, carry);
        _mm_storeu_si128((__m128i*)(p + i), v);
        carry = _mm_set1_epi8((char)(_mm_extract_epi16(v, 7) >> 8));
    }
    c = (uint8_t)_mm_cvtsi128_si32(carry);
#endif
    for (; i < x; ++i) p[i] = c = (uint8_t)(p[i] + c);
}

static void prefix_u16(uint16_t* p, size_t x)
{
    for (size_t i = 1; i < x; ++i) p[i] = (uint16_t)(p[i] + p[i - 1]);
}

/* Two type instantiations of the same loop avoid a per-element size branch.
 * Descending filter passes read unmodified predecessors; the inverse ascends. */
#define DELTA_FILTERS(T, suffix, prefix) \
void vc_delta_filter_##suffix(T* data, size_t z, size_t y, size_t x, unsigned mask) \
{ \
    const size_t slice = y * x, n = z * slice; \
    if (mask & 4) \
        for (size_t i = n; i-- > slice;) data[i] = (T)(data[i] - data[i - slice]); \
    if (mask & 2) \
        for (size_t s = 0; s < z; ++s) { \
            T* p = data + s * slice; \
            for (size_t i = slice; i-- > x;) p[i] = (T)(p[i] - p[i - x]); \
        } \
    if (mask & 1) \
        for (size_t r = 0; r < z * y; ++r) { \
            T* p = data + r * x; \
            for (size_t i = x; i-- > 1;) p[i] = (T)(p[i] - p[i - 1]); \
        } \
} \
void vc_delta_unfilter_##suffix(T* data, size_t z, size_t y, size_t x, unsigned mask) \
{ \
    const size_t slice = y * x, n = z * slice; \
    if (mask & 1) \
        for (size_t r = 0; r < z * y; ++r) prefix(data + r * x, x); \
    if (mask & 2) \
        for (size_t s = 0; s < z; ++s) { \
            T* p = data + s * slice; \
            for (size_t i = x; i < slice; ++i) p[i] = (T)(p[i] + p[i - x]); \
        } \
    if (mask & 4) \
        for (size_t i = slice; i < n; ++i) data[i] = (T)(data[i] + data[i - slice]); \
}
DELTA_FILTERS(uint8_t, u8, prefix_u8)
DELTA_FILTERS(uint16_t, u16, prefix_u16)
#undef DELTA_FILTERS

unsigned vc_delta_select_mask(const uint8_t* q, size_t z, size_t y, size_t x)
{
    uint32_t h[8][256] = {{0}};
    uint64_t n = 0;
    const size_t slice = y * x;
    /* Match the original screen: every fourth z, every second y, interior x. */
    for (size_t zz = 1; zz < z; zz += 4)
        for (size_t yy = 1; yy < y; yy += 2) {
            const uint8_t* p = q + zz * slice + yy * x;
            const uint8_t* py = p - x;
            const uint8_t* pz = p - slice;
            const uint8_t* pyz = pz - x;
            for (size_t xx = 1; xx < x; ++xx) {
                const int v = p[xx], vx = p[xx-1], vy = py[xx], vz = pz[xx];
                const int vxy = py[xx-1], vxz = pz[xx-1], vyz = pyz[xx];
                const int vxyz = pyz[xx-1];
                ++h[0][v];
                ++h[1][(uint8_t)(v-vx)];
                ++h[2][(uint8_t)(v-vy)];
                ++h[3][(uint8_t)(v-vx-vy+vxy)];
                ++h[4][(uint8_t)(v-vz)];
                ++h[5][(uint8_t)(v-vx-vz+vxz)];
                ++h[6][(uint8_t)(v-vy-vz+vyz)];
                ++h[7][(uint8_t)(v-vx-vy-vz+vxy+vxz+vyz-vxyz)];
                ++n;
            }
        }
    unsigned best = 7;
    double best_bits = DBL_MAX;
    for (unsigned m = 0; m < 8; ++m) {
        double bits = 0;
        for (int s = 0; s < 256; ++s)
            if (h[m][s]) {
                const double p = (double)h[m][s] / n;
                bits -= p * log2(p);
            }
        if (bits < best_bits) { best_bits = bits; best = m; }
    }
    return best;
}

void vc_delta_quantize_u8(uint8_t* data, size_t n, int width)
{
    uint8_t lut[256];
    const int half = width / 2;
    for (int v = 0; v < 256; ++v) {
        const int q = (v + half) / width * width;
        lut[v] = (uint8_t)(q < 255 ? q : 255);
    }
    for (size_t i = 0; i < n; ++i) data[i] = lut[data[i]];
}

void vc_delta_quantize_u16(uint16_t* data, size_t n, int width)
{
    const int half = width / 2;
    for (size_t i = 0; i < n; ++i) {
        const int q = (data[i] + half) / width * width;
        data[i] = (uint16_t)(q < 65535 ? q : 65535);
    }
}
