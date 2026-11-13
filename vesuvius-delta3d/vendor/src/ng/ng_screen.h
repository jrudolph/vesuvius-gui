#ifndef VC_NG_SCREEN_H
#define VC_NG_SCREEN_H
#include <stddef.h>
#include <stdint.h>
/* Fixed-grid occupancy. Caller provides (dimension/4)^3 bytes. */
size_t vc_ng_zero_scan(const uint8_t *source, unsigned dimension,
                       uint8_t *empty);
void vc_ng_zero_mask(unsigned dimension, const uint8_t *empty,
                     uint8_t *mask);
/* Eight distributed 16^3 interiors, packed into one 32^3 screen. Empty
 * interiors are relocated deterministically using the occupancy scan. */
int vc_ng_screen_proxy(const uint8_t *source, unsigned dimension,
                        const uint8_t *empty, uint8_t *proxy);
#endif
