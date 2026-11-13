/* Lossless block-choice payloads. One shared small-stream rANS backend serves
 * both previous-symbol contexts and LZ sequence streams. All integers are
 * explicitly little-endian/varints; no alignment or SIMD requirements.
 * Parsing failures and allocation failures return 0, never terminate callers.
 */
#include "selector.h"
#include "selector_bits.h"
#include "buffer.h"
#include "rans.h"
#include <limits.h>

#define ANS_L (UINT32_C(1) << 23)
#define REQUIRE(x) do { if (!(x)) goto done; } while (0)
#include "reader.h"
static void fixed_frequencies(unsigned alphabet, unsigned model, uint32_t *freq) {
    uint64_t hist[256] = {0}, sum = 0;
    for (unsigned s = 0; s < alphabet; ++s) {
        unsigned w = model == 0 ? 4096 : model == 1 ? 65536 / (s + 1) :
            model == 2 ? 65536 / ((s + 1) * (s + 1)) : 65536u >> (s < 16 ? s : 16);
        hist[s] = w ? w : 1; sum += hist[s];
    }
    vc_rans_normalize(hist, sum, freq);
}
static int ans_encode(const uint8_t *p, size_t n, const uint32_t *freq, vc_buffer *out) {
    uint32_t cum[256], sum = 0, state = ANS_L;
    vc_buffer reversed = {0}; int ok = 0;
    for (unsigned s = 0; s < 256; ++s) { cum[s] = sum; sum += freq[s]; }
    REQUIRE(sum == 4096);
    for (size_t i = n; i--;) {
        unsigned s = p[i], f = freq[s]; REQUIRE(f);
        uint32_t limit = ((ANS_L >> 12) << 8) * f;
        while (state >= limit) { REQUIRE(vc_buffer_put(&reversed, state & 255, 1)); state >>= 8; }
        state = (state / f) * 4096 + state % f + cum[s];
    }
    REQUIRE(vc_buffer_put(out, state, 4));
    for (size_t i = reversed.size; i--;) REQUIRE(vc_buffer_put(out, reversed.data[i], 1));
    ok = 1;
done:
    free(reversed.data); return ok;
}
/* Stream tags: 0 raw, 1 constant, 2 sparse transmitted frequencies,
 * 3..6 predefined distributions. The last transmitted count is implied.
 * Exact candidate sizes include table, length and the four-byte ANS state. */
static int stream_encode(const uint8_t *p, size_t n, vc_buffer *out) {
    uint64_t hist[256] = {0}; uint32_t freq[256]; unsigned used = 0, alphabet = 0;
    vc_buffer best = {0}, trial = {0}, coded = {0}; int ok = 0;
    REQUIRE(n);
    for (size_t i = 0; i < n; ++i) { ++hist[p[i]]; if (alphabet <= p[i]) alphabet = p[i] + 1; }
    for (unsigned s = 0; s < 256; ++s) used += hist[s] != 0;
    if (used == 1) return vc_buffer_put(out, 1, 1) && vc_buffer_put(out, p[0], 1);
    REQUIRE(vc_buffer_put(&best, 0, 1) && vc_buffer_append(&best, p, n));
    for (unsigned model = 0; model < 5; ++model) {
        trial.size = coded.size = 0; REQUIRE(vc_buffer_put(&trial, model + 2, 1));
        if (!model) {
            vc_rans_normalize(hist, n, freq); REQUIRE(vc_varput(&trial, used));
            unsigned emitted = 0, previous = 0;
            for (unsigned s = 0; s < 256; ++s) if (freq[s]) {
                REQUIRE(vc_varput(&trial, s - previous)); previous = s + 1;
                if (++emitted < used) REQUIRE(vc_varput(&trial, freq[s]));
            }
        } else {
            REQUIRE(vc_varput(&trial, alphabet)); fixed_frequencies(alphabet, model - 1, freq);
        }
        REQUIRE(ans_encode(p, n, freq, &coded) && vc_varput(&trial, coded.size) &&
                vc_buffer_append(&trial, coded.data, coded.size));
        if (trial.size < best.size) { vc_buffer swap = best; best = trial; trial = swap; }
    }
    REQUIRE(vc_buffer_append(out, best.data, best.size)); ok = 1;
done:
    free(best.data); free(trial.data); free(coded.data); return ok;
}
static int stream_decode(vc_reader *r, size_t n, uint8_t *out) {
    unsigned mode = vc_get(r,1); uint32_t freq[256] = {0}, cum[256], sum = 0;
    uint8_t lut[4096]; int ok = 0;
    REQUIRE(r->ok && n && mode <= 6);
    if (!mode) { const uint8_t *p = vc_take(r, n); REQUIRE(r->ok); memcpy(out, p, n); return 1; }
    if (mode == 1) { unsigned v = vc_get(r,1); REQUIRE(r->ok); memset(out, v, n); return 1; }
    if (mode == 2) {
        size_t used = vc_varget(r,SIZE_MAX); unsigned previous = 0; REQUIRE(r->ok && used >= 2 && used <= 256);
        for (size_t k = 0; k < used; ++k) {
            size_t delta = vc_varget(r,SIZE_MAX); REQUIRE(r->ok && previous < 256 && delta < 256 - previous);
            unsigned s = previous + (unsigned)delta; previous = s + 1;
            size_t count = k + 1 == used ? 4096 - sum : vc_varget(r,SIZE_MAX);
            REQUIRE(r->ok && count && count <= 4096 - sum); freq[s] = (unsigned)count; sum += (unsigned)count;
        }
    } else {
        size_t alphabet = vc_varget(r,SIZE_MAX); REQUIRE(r->ok && alphabet && alphabet <= 256);
        fixed_frequencies((unsigned)alphabet, mode - 3, freq);
    }
    sum = 0;
    for (unsigned s = 0; s < 256; ++s) {
        REQUIRE(freq[s] <= 4096 - sum); cum[s] = sum; memset(lut + sum, s, freq[s]); sum += freq[s];
    }
    REQUIRE(sum == 4096);
    size_t bytes = vc_varget(r,SIZE_MAX); const uint8_t *p = vc_take(r, bytes); REQUIRE(r->ok && bytes >= 4);
    vc_reader coded = {p, bytes, 0, 1}; uint32_t state = 0;
    for (unsigned i = 0; i < 4; ++i) state |= (uint32_t)vc_get(&coded,1) << (8 * i);
    REQUIRE(state >= ANS_L);
    for (size_t i = 0; i < n; ++i) {
        unsigned low = state & 4095, s = lut[low]; out[i] = (uint8_t)s;
        state = freq[s] * (state >> 12) + low - cum[s];
        while (state < ANS_L) { unsigned v = vc_get(&coded,1); REQUIRE(coded.ok); state = (state << 8) | v; }
    }
    REQUIRE(state == ANS_L && coded.i == coded.n); ok = 1;
done:
    return ok;
}

/* Gather each preceding-ID context once, code it, then reconstruct the original
 * causal ID order at decode. Every stored context must be consumed exactly. */
static int context_encode(const uint8_t *p, size_t n, vc_buffer *out) {
    size_t count[256] = {0}, start[256], at[256]; unsigned prev = 0, alphabet = 1, used = 0;
    uint8_t *grouped = malloc(n); int ok = 0; REQUIRE(grouped);
    for (size_t i = 0; i < n; ++i) { ++count[prev]; prev = p[i]; if (alphabet <= prev) alphabet = prev + 1; }
    size_t total = 0;
    for (unsigned s = 0; s < 256; ++s) { start[s] = at[s] = total; total += count[s]; used += count[s] != 0; }
    prev = 0;
    for (size_t i = 0; i < n; ++i) { grouped[at[prev]++] = p[i]; prev = p[i]; }
    REQUIRE(vc_varput(out, alphabet) && vc_varput(out, used));
    for (unsigned s = 0; s < 256; ++s) if (count[s])
        REQUIRE(vc_varput(out, s) && vc_varput(out, count[s]) && stream_encode(grouped + start[s], count[s], out));
    ok = 1;
done:
    free(grouped); return ok;
}
static int context_decode(vc_reader *r, size_t n, uint8_t *out) {
    size_t alphabet = vc_varget(r,SIZE_MAX), used = vc_varget(r,SIZE_MAX), total = 0;
    size_t at[256] = {0}, end[256] = {0}; uint8_t *grouped = NULL; int ok = 0;
    REQUIRE(r->ok && alphabet && alphabet <= 256 && used && used <= alphabet);
    grouped = malloc(n); REQUIRE(grouped);
    for (size_t k = 0; k < used; ++k) {
        size_t s = vc_varget(r,SIZE_MAX), count = vc_varget(r,SIZE_MAX);
        REQUIRE(r->ok && s < alphabet && !end[s] && count && count <= n - total);
        at[s] = total; end[s] = total + count;
        REQUIRE(stream_decode(r, count, grouped + total)); total += count;
    }
    REQUIRE(total == n); unsigned prev = 0;
    for (size_t i = 0; i < n; ++i) {
        REQUIRE(at[prev] < end[prev]); unsigned v = grouped[at[prev]++]; REQUIRE(v < alphabet); out[i] = (uint8_t)v; prev = v;
    }
    for (unsigned s = 0; s < 256; ++s) REQUIRE(at[s] == end[s]); ok = 1;
done:
    free(grouped); return ok;
}

typedef struct { vc_buffer bytes; uint64_t value; unsigned held; } bit_writer;
typedef struct { vc_reader r; uint64_t value; unsigned held; } bit_reader;
static int bits_put(bit_writer *b, unsigned v, unsigned bits) {
    b->value |= (uint64_t)v << b->held; b->held += bits;
    while (b->held >= 8) { if (!vc_buffer_put(&b->bytes, b->value & 255, 1)) return 0; b->value >>= 8; b->held -= 8; }
    return 1;
}
static unsigned bits_get(bit_reader *b, unsigned bits) {
    while (b->held < bits && b->r.ok) { b->value |= (uint64_t)vc_get(&b->r,1) << b->held; b->held += 8; }
    if (!b->r.ok) return 0;
    unsigned v = (unsigned)b->value & ((1u << bits) - 1); b->value >>= bits; b->held -= bits; return v;
}
static int magnitude_put(unsigned v, vc_buffer *codes, bit_writer *bits) {
    unsigned x = v + 1, k = 0; for (unsigned q = x; q > 1; q >>= 1) ++k;
    return vc_buffer_put(codes, k, 1) && bits_put(bits, x - (1u << k), k);
}
static unsigned magnitude_get(unsigned code, bit_reader *bits) {
    if (code >= 31) { bits->r.ok = 0; return 0; }
    return (1u << code) - 1 + bits_get(bits, code);
}
static void recent_update(unsigned *d, unsigned distance) {
    if (d[0] == distance) return;
    if (d[1] != distance) d[2] = d[1]; d[1] = d[0]; d[0] = distance;
}
static unsigned hash4(const uint8_t *p) {
    uint32_t v = (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
    return (v * UINT32_C(2654435761)) >> 16;
}
static int lz_encode(const uint8_t *p, size_t n, vc_buffer *out) {
    uint32_t *head = NULL, *chain = NULL; vc_buffer codes[3] = {{0}}, literals = {0}; bit_writer bits = {0};
    unsigned recent[3] = {1, 4, 8}; size_t matches = 0, pending = 0; int ok = 0;
    REQUIRE(n <= UINT32_MAX && n <= SIZE_MAX / sizeof(*chain));
    head = malloc(65536 * sizeof(*head)); chain = malloc(n * sizeof(*chain)); REQUIRE(head && chain);
    for (unsigned h = 0; h < 65536; ++h) head[h] = UINT32_MAX;
    for (size_t i = 0; i < n;) {
        size_t length = 0, distance = 0;
        if (n - i >= 4) {
            unsigned h = hash4(p + i); uint32_t previous = head[h]; head[h] = (uint32_t)i; chain[i] = previous;
            for (unsigned trial = 0; trial < 256 && previous != UINT32_MAX && i > previous && i - previous <= 65536; ++trial) {
                if (!memcmp(p + i, p + previous, 4)) {
                    size_t len = 4; while (len < n - i && p[i + len] == p[previous + len]) ++len;
                    if (len > length) { length = len; distance = i - previous; }
                }
                previous = chain[previous];
            }
        }
        if (!length) { REQUIRE(vc_buffer_put(&literals, p[i], 1)); ++pending; ++i; continue; }
        REQUIRE(magnitude_put((unsigned)pending, &codes[0], &bits) && magnitude_put((unsigned)(length - 4), &codes[1], &bits));
        unsigned code = 0; while (code < 3 && recent[code] != distance) ++code;
        if (code == 3) {
            unsigned x = (unsigned)distance, k = 0; for (unsigned q = x; q > 1; q >>= 1) ++k;
            code += k; REQUIRE(bits_put(&bits, x - (1u << k), k));
        }
        REQUIRE(vc_buffer_put(&codes[2], code, 1)); recent_update(recent, (unsigned)distance);
        pending = 0; ++matches;
        for (size_t j = i + 1; j < i + length && n - j >= 4; ++j) {
            unsigned h = hash4(p + j); chain[j] = head[h]; head[h] = (uint32_t)j;
        }
        i += length;
    }
    REQUIRE(magnitude_put((unsigned)pending, &codes[0], &bits));
    if (bits.held) REQUIRE(vc_buffer_put(&bits.bytes, bits.value, 1));
    REQUIRE(vc_varput(out, matches) && vc_varput(out, literals.size));
    for (unsigned k = 0; k < 3; ++k) if (codes[k].size) REQUIRE(stream_encode(codes[k].data, codes[k].size, out));
    REQUIRE(vc_varput(out, bits.bytes.size) && vc_buffer_append(out, bits.bytes.data, bits.bytes.size));
    if (literals.size) REQUIRE(stream_encode(literals.data, literals.size, out)); ok = 1;
done:
    free(head); free(chain); free(bits.bytes.data); free(literals.data);
    for (unsigned k = 0; k < 3; ++k) free(codes[k].data); return ok;
}
static int lz_decode(vc_reader *r, size_t n, uint8_t *out) {
    size_t matches = vc_varget(r,SIZE_MAX), nlit = vc_varget(r,SIZE_MAX), at = 0, literal_at = 0;
    uint8_t *codes[3] = {0}, *literals = NULL; unsigned recent[3] = {1, 4, 8}; int ok = 0;
    REQUIRE(r->ok && matches <= n / 4 && nlit <= n);
    for (unsigned k = 0; k < 3; ++k) {
        size_t count = matches + (k == 0);
        if (count) { codes[k] = malloc(count); REQUIRE(codes[k] && stream_decode(r, count, codes[k])); }
    }
    size_t bytes = vc_varget(r,SIZE_MAX); const uint8_t *p = vc_take(r, bytes); REQUIRE(r->ok);
    bit_reader bits = {{p, bytes, 0, 1}, 0, 0};
    if (nlit) { literals = malloc(nlit); REQUIRE(literals && stream_decode(r, nlit, literals)); }
    for (size_t i = 0; i <= matches; ++i) {
        size_t count = magnitude_get(codes[0][i], &bits);
        REQUIRE(bits.r.ok && count <= nlit - literal_at && count <= n - at);
        if (count) memcpy(out + at, literals + literal_at, count); at += count; literal_at += count;
        if (i == matches) break;
        size_t length = (size_t)magnitude_get(codes[1][i], &bits) + 4;
        unsigned code = codes[2][i]; size_t distance = code < 3 ? recent[code] : (size_t)magnitude_get(code - 3, &bits) + 1;
        REQUIRE(bits.r.ok && distance <= 65536 && distance <= at && length <= n - at);
        recent_update(recent, (unsigned)distance);
        for (size_t j = 0; j < length; ++j) { out[at] = out[at - distance]; ++at; }
    }
    REQUIRE(at == n && literal_at == nlit && bits.r.i == bits.r.n && bits.value == 0); ok = 1;
done:
    free(literals); for (unsigned k = 0; k < 3; ++k) free(codes[k]); return ok;
}

/* One tag only: 0 constant, 1 context, 2 packed LZ. Output count is supplied
 * by the containing frame. Context wins equal-size ties, as in the experiment. */
int vc_selector_encode(const uint8_t *ids, size_t n, uint8_t **out, size_t *size) {
    vc_buffer context = {0}, lz = {0}; uint8_t *packed = NULL; int ok = 0;
    if (!out || !size) return 0; *out = NULL; *size = 0;
    if (!ids || !n || n >= INT32_MAX || n > (SIZE_MAX - 7) / 8) return 0;
    unsigned maximum = ids[0]; int constant = 1;
    for (size_t i = 1; i < n; ++i) { constant &= ids[i] == ids[0]; if (maximum < ids[i]) maximum = ids[i]; }
    if (constant) { REQUIRE(vc_buffer_put(&context, 0, 1) && vc_buffer_put(&context, ids[0], 1)); }
    else {
        REQUIRE(vc_buffer_put(&context, 1, 1) && context_encode(ids, n, &context));
        unsigned bits = vc_selector_bits(maximum + 1); size_t count = (n * bits + 7) / 8;
        packed = malloc(count); REQUIRE(packed); vc_selector_pack(ids, n, bits, packed);
        REQUIRE(vc_buffer_put(&lz, 2, 1) && vc_buffer_put(&lz, bits, 1) && lz_encode(packed, count, &lz));
        if (lz.size < context.size) { vc_buffer swap = context; context = lz; lz = swap; }
    }
    *out = context.data; *size = context.size; context.data = NULL; ok = 1;
done:
    free(context.data); free(lz.data); free(packed); return ok;
}
int vc_selector_decode(const uint8_t *src, size_t size, uint8_t *ids, size_t n) {
    if (!src || !size || !ids || !n || n >= INT32_MAX || n > (SIZE_MAX - 7) / 8) return 0;
    vc_reader r = {src, size, 0, 1}; unsigned mode = vc_get(&r,1); int ok = 0; uint8_t *packed = NULL;
    if (!mode) { unsigned v = vc_get(&r,1); REQUIRE(r.ok); memset(ids, v, n); }
    else if (mode == 1) { REQUIRE(context_decode(&r, n, ids)); }
    else if (mode == 2) {
        unsigned bits = vc_get(&r,1); REQUIRE(r.ok && bits && bits <= 8);
        size_t count = (n * bits + 7) / 8; packed = malloc(count); REQUIRE(packed && lz_decode(&r, count, packed));
        if ((n * bits) % 8) REQUIRE((packed[count - 1] >> ((n * bits) % 8)) == 0);
        vc_selector_unpack(packed, n, bits, ids);
    } else goto done;
    REQUIRE(r.ok && r.i == r.n); ok = 1;
done:
    free(packed); return ok;
}
