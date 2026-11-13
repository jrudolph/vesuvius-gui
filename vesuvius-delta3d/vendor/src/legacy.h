#ifndef VC_DELTA_LEGACY_H
#define VC_DELTA_LEGACY_H
#include <stdint.h>
#include <stddef.h>
typedef struct {
    uint32_t shape[3], channels;
    unsigned element_size, quantization, layout, version, mask, has_mask;
    size_t bytes;
} vc_legacy_info;
/* A header probe does not read the entropy payload. Retired LPC is rejected. */
int vc_legacy_probe(const uint8_t*, size_t, const uint8_t magic[4], vc_legacy_info*);
/* Allocated result belongs to caller (free). channels=0 selects a scalar frame;
 * channels>=1 selects the original v2 channel container, even for one channel. */
int vc_legacy_encode(const uint8_t*, size_t, const uint32_t shape[3], unsigned element_size,
                     unsigned quantization, uint32_t channels, unsigned layout,
                     uint8_t**, size_t*);
int vc_legacy_decode(const uint8_t*, size_t, const uint8_t magic[4], uint8_t*, size_t);
#endif
