#ifndef VC_DELTA_RANS_H
#define VC_DELTA_RANS_H
#include <stddef.h>
#include <stdint.h>
/* Eight-lane legacy rANS frame, shared by Delta3D and preset metadata.
 * Encoded storage is allocated by the library and released with free(). */
void vc_rans_normalize(const uint64_t hist[256], uint64_t total, uint32_t freq[256]);
/* Shared encoder for normalized or fixed header distributions.
 * Frequencies must sum to 4096 and cover every input symbol. */
int vc_rans_encode_frequencies(const uint8_t *src, size_t count, size_t prefix,
                   const uint32_t freq[256], uint8_t **output, size_t *size);
int vc_rans_encode(const uint8_t *src, size_t count, size_t prefix,
                   uint8_t **output, size_t *size);
int vc_rans_decode(const uint8_t *frame, size_t size, uint8_t *out, size_t count);
/* Same payload decoder, without fabricating a legacy table prefix. */
int vc_rans_decode_frequencies(const uint16_t freq[256], const uint8_t *payload,
                              size_t size, uint8_t *out, size_t count);
#endif
