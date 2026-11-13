#ifndef VC_SELECTOR_H
#define VC_SELECTOR_H
#include <stddef.h>
#include <stdint.h>
/* Independent selector payload. Encode chooses constant, previous-ID context,
 * or bit-packed LZ by complete byte size. No global state. The caller owns
 * *out (free), and supplies the known ID count when decoding. */
int vc_selector_encode(const uint8_t *ids, size_t count, uint8_t **out, size_t *size);
int vc_selector_decode(const uint8_t *src, size_t size, uint8_t *ids, size_t count);
#endif
