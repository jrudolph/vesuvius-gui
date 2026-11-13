/* D3D1 v1/v2 compatibility. The original byte layout and arithmetic are retained;
 * all size products are checked before allocation or pointer arithmetic. */
#include "legacy.h"
#include "delta_filter.h"
#include "rans.h"
#include <stdlib.h>
#include <string.h>

static const uint8_t magic[4] = {'D','3','D','1'};
static uint32_t read32(const uint8_t *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24;
}
static void write32(uint8_t *p, uint32_t v)
{
    p[0]=(uint8_t)v; p[1]=(uint8_t)(v>>8); p[2]=(uint8_t)(v>>16); p[3]=(uint8_t)(v>>24);
}
static int field_size(const uint32_t shape[3], unsigned elem, size_t *size)
{
    if (elem != 1 && elem != 2) return 0;
    size_t n = elem;
    for (unsigned a = 0; a < 3; ++a) {
        if (!shape[a] || n > SIZE_MAX / shape[a]) return 0;
        n *= shape[a];
    }
    *size = n; return 1;
}
int vc_legacy_probe(const uint8_t *src, size_t size, const uint8_t expected[4], vc_legacy_info *out)
{
    if (!src || !expected || !out || size < 20 || memcmp(src, expected, 4)) return 0;
    vc_legacy_info info = {0};
    info.version = src[4]; info.element_size = src[5];
    info.quantization = src[6] ? src[6] : 1;
    info.channels = 1;
    for (unsigned a = 0; a < 3; ++a) info.shape[a] = read32(src + 8 + 4*a);
    if (!field_size(info.shape, info.element_size, &info.bytes)) return 0;
    if (info.version == 1) {
        info.has_mask = (src[7] & 128) != 0;
        if ((src[7] & 15) != 1 || (!info.has_mask && (src[7] & 112))) return 0;
        info.mask = info.has_mask ? (src[7] >> 4) & 7 : 7;
    } else if (info.version == 2) {
        if (size < 24 || src[7] > 1) return 0;
        info.layout = src[7]; info.channels = read32(src + 20);
        if (!info.channels || info.bytes > SIZE_MAX / info.channels) return 0;
        info.bytes *= info.channels;
    } else return 0;
    *out = info; return 1;
}
static void header(uint8_t *dst, const uint32_t shape[3], unsigned elem, unsigned q,
                   unsigned version, unsigned flags)
{
    memcpy(dst, magic, 4); dst[4]=(uint8_t)version; dst[5]=(uint8_t)elem;
    dst[6]=(uint8_t)q; dst[7]=(uint8_t)flags;
    for (unsigned a = 0; a < 3; ++a) write32(dst + 8 + 4*a, shape[a]);
}
static int encode_field(const uint8_t *src, size_t size, const uint32_t shape[3],
                        unsigned elem, unsigned q, uint8_t **out, size_t *out_size)
{
    uint8_t *filtered = malloc(size);
    if (!filtered) return 0;
    memcpy(filtered, src, size);
    unsigned mask = 7;
    if (elem == 1) {
        if (q > 1) vc_delta_quantize_u8(filtered, size, q);
        if (shape[0] >= 2 && shape[1] >= 2 && shape[2] >= 2)
            mask = vc_delta_select_mask(filtered, shape[0], shape[1], shape[2]);
        vc_delta_filter_u8(filtered, shape[0], shape[1], shape[2], mask);
    } else {
        if (q > 1) vc_delta_quantize_u16((uint16_t*)filtered, size/2, q);
        vc_delta_filter_u16((uint16_t*)filtered, shape[0], shape[1], shape[2], mask);
    }
    int ok = vc_rans_encode(filtered, size, 20, out, out_size);
    free(filtered);
    if (ok) header(*out, shape, elem, q, 1, 128 | mask<<4 | 1);
    return ok;
}
int vc_legacy_encode(const uint8_t *src, size_t size, const uint32_t shape[3], unsigned elem,
                     unsigned q, uint32_t channels, unsigned layout, uint8_t **out, size_t *out_size)
{
    size_t field;
    if (!out || !out_size) return 0;
    *out = NULL; *out_size = 0;
    if (!src || !shape || q < 1 || q > 255 || layout > 1 || !field_size(shape, elem, &field)) return 0;
    if (!channels) return size == field && encode_field(src, field, shape, elem, q, out, out_size);
    uint64_t table_size = 24 + (uint64_t)channels*4;
    if (field > SIZE_MAX/channels || field*channels != size || table_size > SIZE_MAX) return 0;
    /* Keep just one channel scratch buffer. The output grows geometrically;
     * no full-volume transpose or array of individually owned frames. */
    size_t used = 24 + (size_t)channels*4, cap = used;
    uint8_t *result = malloc(cap), *scratch = layout && channels > 1 ? malloc(field) : NULL;
    if (!result || (layout && channels > 1 && !scratch)) { free(result); free(scratch); return 0; }
    header(result, shape, elem, q, 2, layout); write32(result+20, channels);
    for (uint32_t c = 0; c < channels; ++c) {
        const uint8_t *input = src + (size_t)c*field;
        if (scratch) {
            for (size_t i = 0; i < field/elem; ++i)
                memcpy(scratch+i*elem, src+(i*channels+c)*elem, elem);
            input = scratch;
        }
        uint8_t *payload; size_t bytes;
        if (!encode_field(input, field, shape, elem, q, &payload, &bytes)) goto fail;
        if (bytes > UINT32_MAX || bytes > SIZE_MAX-used) { free(payload); goto fail; }
        if (used+bytes > cap) {
            size_t next = cap <= SIZE_MAX/2 ? cap*2 : SIZE_MAX;
            if (next < used+bytes) next = used+bytes;
            uint8_t *grown = realloc(result, next);
            if (!grown) { free(payload); goto fail; }
            result = grown; cap = next;
        }
        write32(result+24+4*(size_t)c, (uint32_t)bytes);
        memcpy(result+used, payload, bytes); free(payload); used += bytes;
    }
    free(scratch); *out=result; *out_size=used; return 1;
fail:
    free(scratch); free(result); return 0;
}
static int decode_field(const uint8_t *src, size_t size, const vc_legacy_info *info, uint8_t *out)
{
    if (!vc_rans_decode(src+20, size-20, out, info->bytes)) return 0;
    if (info->element_size == 1)
        vc_delta_unfilter_u8(out, info->shape[0], info->shape[1], info->shape[2], info->mask);
    else
        vc_delta_unfilter_u16((uint16_t*)out, info->shape[0], info->shape[1], info->shape[2], info->mask);
    return 1;
}
int vc_legacy_decode(const uint8_t *src, size_t size, const uint8_t expected[4], uint8_t *out, size_t count)
{
    vc_legacy_info info;
    if (!out || !vc_legacy_probe(src, size, expected, &info) || info.bytes != count) return 0;
    if (info.version == 1) return decode_field(src, size, &info, out);
    if (info.channels > (size-24)/4) return 0;
    size_t offset = 24 + 4*(size_t)info.channels, field = count/info.channels;
    uint8_t *scratch = info.layout && info.channels > 1 ? malloc(field) : NULL;
    if (info.layout && info.channels > 1 && !scratch) return 0;
    for (uint32_t c = 0; c < info.channels; ++c) {
        size_t bytes = read32(src+24+4*(size_t)c);
        vc_legacy_info leaf;
        if (bytes > size-offset || !vc_legacy_probe(src+offset, bytes, magic, &leaf) ||
            leaf.version != 1 || leaf.element_size != info.element_size ||
            memcmp(leaf.shape, info.shape, sizeof(info.shape))) goto fail;
        uint8_t *dst = scratch ? scratch : out+(size_t)c*field;
        if (!decode_field(src+offset, bytes, &leaf, dst)) goto fail;
        if (scratch) for (size_t i = 0; i < field/info.element_size; ++i)
            memcpy(out+(i*info.channels+c)*info.element_size, scratch+i*info.element_size, info.element_size);
        offset += bytes;
    }
    free(scratch); return offset == size;
fail:
    free(scratch); return 0;
}
