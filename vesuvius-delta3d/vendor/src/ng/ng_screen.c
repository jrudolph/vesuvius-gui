#include "ng_screen.h"
#include <stdlib.h>
#include <string.h>

size_t vc_ng_zero_scan(const uint8_t *src, unsigned d, uint8_t *empty)
{
    const unsigned e=4, nb=d/4;
    size_t count = 0;
    for (unsigned z = 0; z < d; z += e)
        for (unsigned y = 0; y < d; y += e)
            for (unsigned x = 0; x < d; x += e) {
                uint32_t bits = 0;
                for (unsigned dz = 0; dz < e; ++dz)
                    for (unsigned dy = 0; dy < e; ++dy) {
                        const uint8_t *row = src + ((size_t)(z + dz) * d + y + dy) * d + x;
                        uint32_t value;
                        memcpy(&value, row, sizeof(value));
                        bits |= value;
                    }
                size_t b = ((size_t)(z / e) * nb + y / e) * nb + x / e;
                empty[b] = bits == 0;
                count += empty[b];
            }
    return count;
}

void vc_ng_zero_mask(unsigned d, const uint8_t *empty, uint8_t *mask)
{
    const unsigned e=4, nb=d/4;
    for (unsigned z = 0; z < d; ++z)
        for (unsigned y = 0; y < d; ++y)
            for (unsigned x = 0; x < d; x += e)
                memset(mask + ((size_t)z * d + y) * d + x,
                       !empty[((size_t)(z / e) * nb + y / e) * nb + x / e], e);
}

static unsigned clamp_origin(int value, unsigned d)
{
    return value < 0 ? 0 : (unsigned)value > d - 16 ? d - 16 : (unsigned)value;
}

int vc_ng_screen_proxy(const uint8_t *src, unsigned d,
                        const uint8_t *empty, uint8_t *proxy)
{
    if (!src || !proxy || d < 32 || d % 32) return -1;
    if (d == 32) {
        memcpy(proxy, src, 32 * 32 * 32);
        return 0;
    }
    const unsigned e=4, nb=d/4;
    const size_t blocks = (size_t)nb * nb * nb;
    size_t *occupied = NULL, occupied_count = 0;
    unsigned origins[8][3];
    if (empty) {
        occupied = malloc(blocks * sizeof(*occupied));
        if (!occupied) return -1;
        for (size_t b = 0; b < blocks; ++b)
            if (!empty[b]) occupied[occupied_count++] = b;
    }
    for (unsigned k = 0; k < 8; ++k) {
        unsigned *o = origins[k];
        o[0] = (2 * (k / 4) + 1) * d / 4 - 8;
        o[1] = (2 * ((k / 2) % 2) + 1) * d / 4 - 8;
        o[2] = (2 * (k % 2) + 1) * d / 4 - 8;
        if (empty) {
            int has_signal = 0;
            for (unsigned z = o[0] / e; z < (o[0] + 16 + e - 1) / e; ++z)
                for (unsigned y = o[1] / e; y < (o[1] + 16 + e - 1) / e; ++y)
                    for (unsigned x = o[2] / e; x < (o[2] + 16 + e - 1) / e; ++x)
                        has_signal |= !empty[((size_t)z * nb + y) * nb + x];
            if (!has_signal && occupied_count) {
                size_t rank = (2 * k + 1) * occupied_count / 16;
                if (rank >= occupied_count) rank = occupied_count - 1;
                size_t b = occupied[rank];
                unsigned coords[3] = {(unsigned)(b / (nb * nb)) * e,
                    (unsigned)((b / nb) % nb) * e, (unsigned)(b % nb) * e};
                for (unsigned j = 0; j < 3; ++j) o[j] = clamp_origin((int)coords[j] - 4, d);
            }
        }
    }
    free(occupied);
    unsigned signal = 0;
    for (unsigned z = 0; z < 32; ++z)
        for (unsigned y = 0; y < 32; ++y)
            for (unsigned x = 0; x < 32; ++x) {
                const unsigned *o = origins[(z / 16) * 4 + (y / 16) * 2 + x / 16];
                uint8_t value = src[((size_t)(o[0] + z % 16) * d + o[1] + y % 16) * d + o[2] + x % 16];
                proxy[(z * 32 + y) * 32 + x] = value;
                signal |= value;
            }
    /* Coarse occupied blocks may still have empty sampled interiors. */
    if (!signal) {
        const size_t total = (size_t)d * d * d;
        size_t i = 0;
        while (i < total && !src[i]) ++i;
        if (i < total) {
            unsigned p[3] = {(unsigned)(i / (d * d)), (unsigned)((i / d) % d), (unsigned)(i % d)};
            for (unsigned j = 0; j < 3; ++j) p[j] = clamp_origin((int)p[j] - 8, d);
            for (unsigned z = 0; z < 16; ++z)
                for (unsigned y = 0; y < 16; ++y)
                    memcpy(proxy + ((size_t)z * 32 + y) * 32,
                           src + ((size_t)(p[0] + z) * d + p[1] + y) * d + p[2], 16);
        }
    }
    return 0;
}
