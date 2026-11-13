#pragma once
static const uint8_t vcn_lc_perm[256][8] = {
#define VCN_LC_P1(m) {VCN_LC_PI(m,0),VCN_LC_PI(m,1),VCN_LC_PI(m,2),VCN_LC_PI(m,3),VCN_LC_PI(m,4),VCN_LC_PI(m,5),VCN_LC_PI(m,6),VCN_LC_PI(m,7)}
/* lane i reads word popcount(m & ((1 << i) - 1)) */
#define VCN_LC_PC(m) (((m) & 1) + ((m) >> 1 & 1) + ((m) >> 2 & 1) + ((m) >> 3 & 1) + ((m) >> 4 & 1) + ((m) >> 5 & 1) + ((m) >> 6 & 1))
#define VCN_LC_PI(m, i) (uint8_t)(VCN_LC_PC((m) & ((1 << (i)) - 1)) * (((m) >> (i)) & 1))
#define VCN_LC_P4(m) VCN_LC_P1(m), VCN_LC_P1(m + 1), VCN_LC_P1(m + 2), VCN_LC_P1(m + 3)
#define VCN_LC_P16(m) VCN_LC_P4(m), VCN_LC_P4(m + 4), VCN_LC_P4(m + 8), VCN_LC_P4(m + 12)
    VCN_LC_P16(0), VCN_LC_P16(16), VCN_LC_P16(32), VCN_LC_P16(48), VCN_LC_P16(64), VCN_LC_P16(80), VCN_LC_P16(96), VCN_LC_P16(112),
    VCN_LC_P16(128), VCN_LC_P16(144), VCN_LC_P16(160), VCN_LC_P16(176), VCN_LC_P16(192), VCN_LC_P16(208), VCN_LC_P16(224), VCN_LC_P16(240)};
#undef VCN_LC_P1
#undef VCN_LC_PI
#undef VCN_LC_PC
#undef VCN_LC_P4
#undef VCN_LC_P16
