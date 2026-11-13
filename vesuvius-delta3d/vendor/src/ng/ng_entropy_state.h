#pragma once
#include <stdint.h>
#include <stddef.h>
typedef struct {
  const uint32_t *T;
  const uint8_t *p, *end;
  size_t over;
  int tb;
  uint32_t S[32];
} vcn_lc_dec;
static inline uint32_t vcn_rd16(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8; }
