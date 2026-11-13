#ifndef VC_DELTA3D_FILTER_H
#define VC_DELTA3D_FILTER_H

#include <stddef.h>
#include <stdint.h>


/* Internal primitives: callers validate shape products and buffer sizes.
 * Axis bits are x=1, y=2, z=4. Arithmetic wraps at the element width. */
void vc_delta_filter_u8(uint8_t*, size_t z, size_t y, size_t x, unsigned mask);
void vc_delta_filter_u16(uint16_t*, size_t z, size_t y, size_t x, unsigned mask);
void vc_delta_unfilter_u8(uint8_t*, size_t z, size_t y, size_t x, unsigned mask);
void vc_delta_unfilter_u16(uint16_t*, size_t z, size_t y, size_t x, unsigned mask);
unsigned vc_delta_select_mask(const uint8_t*, size_t z, size_t y, size_t x);
/* Width is validated by the public API: 1..255. */
void vc_delta_quantize_u8(uint8_t*, size_t count, int width);
void vc_delta_quantize_u16(uint16_t*, size_t count, int width);

#endif
