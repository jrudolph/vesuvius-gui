#ifndef VC_NG_HEADER_H
#define VC_NG_HEADER_H
#include <stddef.h>
#include <stdint.h>
/* Runtime header: seven LE u32s (x,y,z,mode|edge<<8,models,choices,
 * selector bytes), models in the native parameter layout (sizes inferred),
 * 12-byte menu entries, then optional entropy parameters and dense LE u16
 * frequencies (256 per class). No compact-table reserialization in decoding.
 * This is an internal representation, NOT a historical frame decoder.
 * Wire controls derive all array shapes; one pooled numeric stream follows.
 * Allocated results belong to the caller (free). */
int vc_header_encode(const uint8_t*,size_t,uint8_t**,size_t*);
int vc_header_decode(const uint8_t*,size_t,uint8_t**,size_t*);
#endif
