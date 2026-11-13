#pragma once
/* Shared runtime/fixed-geometry activity access. Include after the native
 * kernels: parameters and padding are exactly the original NG representation. */
typedef struct {
    vcn_lc_par* p;uint8_t *b,*e,*classes;
    vcn_lc_plist pl[2];
} ng_causal_state;
