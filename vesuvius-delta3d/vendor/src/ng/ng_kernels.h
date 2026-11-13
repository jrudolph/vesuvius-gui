/* Combined inclusion for compile-time geometry instantiations.
 * Ordinary encoder/decoder translation units include only their own kernels.
 * Chase Hendrick's NG arithmetic is retained; see LICENSE. */
#pragma once
#if defined(__clang__)
#pragma float_control(push)
#pragma STDC FP_CONTRACT OFF
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#endif
#include "ng_kernel_encode.h"
#include "ng_kernel_decode.h"
#if defined(__clang__)
#pragma float_control(pop)
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
