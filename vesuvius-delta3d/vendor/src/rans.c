#include "rans.h"
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

// eight interleaved 64-bit states, 32-bit renormalization. Encoding walks
// the symbols back to front and emits renorm words back to front, so the
// decoder streams strictly forward. Division-free encode via 64-bit
// reciprocals (Alverson); exact for all 64-bit states. ----

enum { kRansScaleBits = 12, kRansM = 4096, kRansLanes = 8, kRansTableBytes = 512, kRansStateBytes = 64 };

#define kRansL (UINT64_C(1) << 31)




typedef struct RansEncSym {
    uint64_t rcp;   // reciprocal so q = mulhi(x, rcp) >> shift = x/freq
    uint64_t bias;  // cum, or cum + M - 1 for freq 1 (rcp = 2^64 - 1)
    uint64_t xmax;  // renorm threshold ((L>>12)<<32) * freq
    uint32_t cmpl;  // M - freq
    uint32_t shift;
} RansEncSym;

/* Native-endian, unaligned store matching the existing wire format. Explicit
 * byte accesses compile to one word store without FORTIFY's dynamic-size
 * memcpy check on every backwards-growing stream write. The encoder reserves
 * the complete worst-case stream plus speculative-store slack before use. */
static inline void ransStore32(uint8_t *p, uint32_t value)
{
    const uint8_t *bytes = (const uint8_t *)&value;
    p[0] = bytes[0]; p[1] = bytes[1];
    p[2] = bytes[2]; p[3] = bytes[3];
}

static inline uint64_t ransMulHi(uint64_t a, uint64_t b)
{
#if defined(_MSC_VER)
    return __umulh(a, b);   // high 64 bits of the 64x64 product
#else
    return (uint64_t)(
        ((unsigned __int128)(a) * b) >> 64);
#endif
}

// Histogram -> frequencies summing to exactly kRansM, every present symbol
// nonzero. Rounding drift is settled against the most frequent symbol.
void vc_rans_normalize(const uint64_t hist[256],
                   uint64_t total,
                   uint32_t freq[256])
{
    if (!total) { memset(freq, 0, 256 * sizeof(*freq)); freq[0] = kRansM; return; }
    uint64_t assigned = 0;
    for (int s = 0; s < 256; ++s) {
        freq[s] = hist[s]
            ? (uint32_t)(
                  (((hist[s] * kRansM) / total) ? ((hist[s] * kRansM) / total) : 1))
            : 0;
        assigned += freq[s];
    }
    while (assigned != kRansM) {
        int s = 0;
        for (int i = 1; i < 256; ++i)
            if (freq[i] > freq[s]) s = i;
        if (assigned > kRansM) {
            const uint64_t cut = (freq[s] - 1 < assigned - kRansM) ? freq[s] - 1 : assigned - kRansM;
            freq[s] -= (uint32_t)(cut);
            assigned -= cut;
        } else {
            freq[s] += (uint32_t)(kRansM - assigned);
            assigned = kRansM;
        }
    }
}

static void ransBuildEncTable(const uint32_t freq[256], RansEncSym enc[256])
{
    uint32_t cum = 0;
    for (int s = 0; s < 256; ++s) {
        const uint32_t f = freq[s];
        RansEncSym *ep = &enc[s];
        ep->cmpl = kRansM - f;
        ep->xmax = ((kRansL >> kRansScaleBits) << 32) * f;
        if (f == 0) {
            memset(ep, 0, sizeof(*ep));
        } else if (f == 1) {
            // mulhi(x, 2^64 - 1) = x - 1; the off-by-(M-1) folds into bias
            ep->rcp = ~0ull;
            ep->shift = 0;
            ep->bias = (uint64_t)(cum) + kRansM - 1;
        } else {
            uint32_t k = 1;
            while ((1u << k) < f) ++k;  // k = ceil(log2 f)
            ep->shift = k - 1;
            // num = 2^(k+63) + (f-1): high 64 bits = 2^(k-1), low 64 bits =
            // (f-1). The quotient num/f provably fits in 64 bits (f > 2^(k-1)),
            // so the 128/64 -> 64 division is well-defined on both paths.
#if defined(_MSC_VER) && defined(__clang__)
            // ClangCL lacks MSVC's _udiv128. Exact binary division here avoids
            // a dependency on compiler-rt's 128-bit division helpers. The
            // remainder stays below 2*f (f <= kRansM), so doubling is safe.
            uint64_t rem = UINT64_C(1) << (k - 1);
            const uint64_t low = f - 1;
            ep->rcp = 0;
            for (int bit = 63; bit >= 0; --bit) {
                rem = (rem << 1) | ((low >> bit) & 1);
                if (rem >= f) {
                    rem -= f;
                    ep->rcp |= UINT64_C(1) << bit;
                }
            }
#elif defined(_MSC_VER)
            uint64_t ransRcpRem;
            ep->rcp = _udiv128(UINT64_C(1) << (k - 1),
                             (uint64_t)(f) - 1, f, &ransRcpRem);
#else
            const unsigned __int128 num =
                (((unsigned __int128)(1) << (k + 63)) + f - 1);
            ep->rcp = (uint64_t)(num / f);
#endif
            ep->bias = cum;
        }
        cum += f;
    }
}

// Compresses `input` into [headerReserve][freq table][states+stream].
static int ransEncodeFrequencies(const uint8_t*,size_t,size_t,const uint32_t[256],uint8_t**,size_t*);
int vc_rans_encode(const uint8_t *in, size_t n, size_t headerReserve,
                   uint8_t **result, size_t *resultSize)
{
    if (!result || !resultSize || (!in && n)) return 0;
    *result = NULL; *resultSize = 0;
    if (headerReserve > SIZE_MAX - 640 || n > (SIZE_MAX - headerReserve - 640) / 2) return 0;
    uint64_t h4[4][256] = {};
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        h4[0][in[i]]++;
        h4[1][in[i + 1]]++;
        h4[2][in[i + 2]]++;
        h4[3][in[i + 3]]++;
    }
    for (; i < n; ++i) h4[0][in[i]]++;
    uint64_t hist[256];
    for (int s = 0; s < 256; ++s)
        hist[s] = h4[0][s] + h4[1][s] + h4[2][s] + h4[3][s];

    uint32_t freq[256];
    vc_rans_normalize(hist, n, freq);
    return ransEncodeFrequencies(in,n,headerReserve,freq,result,resultSize);
}

int vc_rans_encode_frequencies(const uint8_t *in,size_t n,size_t headerReserve,
                   const uint32_t freq[256],uint8_t **result,size_t *resultSize)
{
    if(!result||!resultSize||!freq||(!in&&n))return 0;
    *result=NULL;*resultSize=0;
    if(headerReserve>SIZE_MAX-640||n>(SIZE_MAX-headerReserve-640)/2)return 0;
    unsigned total=0;for(unsigned s=0;s<256;s++){if(freq[s]>4096)return 0;total+=freq[s];}
    if(total!=4096)return 0;
    for(size_t i=0;i<n;i++)if(!freq[in[i]])return 0;
    return ransEncodeFrequencies(in,n,headerReserve,freq,result,resultSize);
}
static int ransEncodeFrequencies(const uint8_t *in,size_t n,size_t headerReserve,
                   const uint32_t freq[256],uint8_t **result,size_t *resultSize)
{
    RansEncSym enc[256];
    ransBuildEncTable(freq, enc);

    // Worst case is ~1.5 bytes/symbol (12-bit code ceiling); 2n is safe.
    bool constant = false;
    for (int symbol = 0; symbol < 256; ++symbol) constant |= freq[symbol] == kRansM;
    const size_t streamCap = (constant ? 0 : 2 * n) + kRansStateBytes + 64;
    const size_t capacity = headerReserve + kRansTableBytes + streamCap;
    uint8_t *base = malloc(capacity);
    if (!base) return 0;
    memset(base, 0, headerReserve);

    uint8_t* table = base + headerReserve;
    for (int s = 0; s < 256; ++s) {
        table[2 * s] = (uint8_t)(freq[s] & 0xFF);
        table[2 * s + 1] = (uint8_t)(freq[s] >> 8);
    }

    uint8_t* end = base + capacity;
    uint8_t* p = end;
    uint64_t x[kRansLanes];
    for (size_t k = 0; k < kRansLanes; ++k) x[k] = kRansL;
    // Branchless renorm: speculatively store each lane's low word below the
    // claimed stream and only advance past it when the state actually
    // overflows; the buffer's slack keeps the speculative store in bounds.
    size_t j = constant ? 0 : n;
    while (j & (kRansLanes - 1)) {
        --j;
        const RansEncSym *ep = &enc[in[j]];
        uint64_t s = x[j & (kRansLanes - 1)];
        const uint32_t w = (uint32_t)(s);
        ransStore32(p - 4, w);
        const bool renorm = s >= ep->xmax;
        p -= renorm ? 4 : 0;
        s = renorm ? s >> 32 : s;
        const uint64_t q = ransMulHi(s, ep->rcp) >> ep->shift;
        s = s + ep->bias + q * ep->cmpl;
        x[j & (kRansLanes - 1)] = s;
    }
    // Blocks of eight: renorm decisions depend only on each lane's own
    // state, so only the cheap store-offset adds serialize across lanes.
    while (j) {
        j -= kRansLanes;
        const uint8_t* sym = in + j;
        size_t o = 0;
        /* Keep the eight independent states in registers. GCC's C frontend
         * otherwise leaves this tiny loop rolled (unlike the original C++
         * kernel), serializing state loads and reciprocal multiplies. */
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 8
#endif
        for (size_t k = kRansLanes; k-- > 0;) {
            const RansEncSym *ep = &enc[sym[k]];
            uint64_t s = x[k];
            const uint32_t w = (uint32_t)(s);
            ransStore32(p - o - 4, w);
            const bool renorm = s >= ep->xmax;
            o += renorm ? 4 : 0;
            s = renorm ? s >> 32 : s;
            const uint64_t q = ransMulHi(s, ep->rcp) >> ep->shift;
            x[k] = s + ep->bias + q * ep->cmpl;
        }
        p -= o;
    }
    for (size_t k = kRansLanes; k-- > 0;) {
        p -= 8;
        memcpy(p, &x[k], 8);
    }

    const size_t streamSize = (size_t)(end - p);
    memmove(base + headerReserve + kRansTableBytes, p, streamSize);
    *resultSize = headerReserve + kRansTableBytes + streamSize;
    *result = base;
    return 1;
}

// Decodes a rANS frame produced by vc_rans_encode. Returns false on any
// inconsistency (bad table, short stream, or states/stream not ending
// exactly where encoding started them).
int vc_rans_decode(const uint8_t *table, size_t frameSize, uint8_t *out, size_t n)
{
    if (!table || (!out && n) || frameSize < kRansTableBytes + kRansStateBytes) return 0;
    uint16_t freq[256];
    for (int s = 0; s < 256; ++s)
        freq[s] = (uint16_t)(table[2*s] | (unsigned)table[2*s+1] << 8);
    return vc_rans_decode_frequencies(freq, table+kRansTableBytes,
                                     frameSize-kRansTableBytes, out, n);
}
int vc_rans_decode_frequencies(const uint16_t freq[256], const uint8_t *payload,
                              size_t size, uint8_t *out, size_t n)
{
    if (!freq || !payload || (!out && n) || size < kRansStateBytes) return 0;
    uint32_t cum[256];
    uint32_t total = 0;
    for (int s = 0; s < 256; ++s) {
        cum[s] = total;
        total += freq[s];
    }
    if (total != kRansM)
        return false;
    // Fused per-slot entry: one L1 load yields symbol, freq-1 and
    // slot - cum (each fits 12/8 bits within a uint32).
    uint32_t slotEntry[kRansM];
    for (int s = 0; s < 256; ++s)
        for (uint32_t j = 0; j < freq[s]; ++j)
            slotEntry[cum[s] + j] =
                (((uint32_t)freq[s] - 1) << 20) | (j << 8) | (uint32_t)(s);

    const uint8_t *p = payload;
    const uint8_t *end = payload + size;
    uint64_t x[kRansLanes];
    for (size_t k = 0; k < kRansLanes; ++k) {
        memcpy(&x[k], p, 8);
        p += 8;
    }

    size_t i = 0;
    // Fast path: while >= 32 stream bytes remain, all eight lanes can renorm
    // branchlessly (each consumes at most 4 bytes per symbol).
    for (; i + kRansLanes <= n && end - p >= 32; i += kRansLanes) {
        // Stage 1: symbol resolve + state advance, independent per lane.
        uint64_t t[kRansLanes];
        for (size_t k = 0; k < kRansLanes; ++k) {
            const uint32_t slot = (uint32_t)(x[k]) & (kRansM - 1);
            const uint32_t e = slotEntry[slot];
            out[i + k] = (uint8_t)(e);
            t[k] = ((uint64_t)(e >> 20) + 1) *
                       (x[k] >> kRansScaleBits) +
                   ((e >> 8) & 0xFFF);
        }
        // Stage 2: branchless renorm; only the cheap offset adds serialize.
        size_t o = 0;
        for (size_t k = 0; k < kRansLanes; ++k) {
            uint32_t w;
            memcpy(&w, p + o, 4);
            const bool renorm = t[k] < kRansL;
            x[k] = renorm ? (t[k] << 32) | w : t[k];
            o += renorm ? 4 : 0;
        }
        p += o;
    }
    for (; i < n; ++i) {
        uint64_t s = x[i & (kRansLanes - 1)];
        const uint32_t slot = (uint32_t)(s) & (kRansM - 1);
        const uint32_t e = slotEntry[slot];
        out[i] = (uint8_t)(e);
        s = ((uint64_t)(e >> 20) + 1) * (s >> kRansScaleBits) +
            ((e >> 8) & 0xFFF);
        if (s < kRansL) {
            uint32_t w = 0;
            if ((size_t)(end - p) < 4)
                return false;
            memcpy(&w, p, 4);
            p += 4;
            s = (s << 32) | w;
        }
        x[i & (kRansLanes - 1)] = s;
    }

    // Encoding started every lane at kRansL and the decoder must consume the
    // stream exactly; anything else means a corrupt payload.
    if (p != end)
        return false;
    for (size_t k = 0; k < kRansLanes; ++k)
        if (x[k] != kRansL)
            return false;
    return true;
}

