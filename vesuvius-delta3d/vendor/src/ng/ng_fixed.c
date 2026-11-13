/* Compile the SAME kernels with constant geometry, selected by CMake. No copied
 * predictor, entropy, scheduling or reconstruction implementation. */
#ifndef NG_FIXED_GEOMETRY
#error "Set NG_FIXED_GEOMETRY for this kernel instantiation"
#endif
#define NG_FIXED_NAME_IMPL(d,s) ng_fixed##d##s
#define NG_FIXED_NAME_EXPAND(d,s) NG_FIXED_NAME_IMPL(d,s)
#define NG_FIXED_NAME(s) NG_FIXED_NAME_EXPAND(NG_FIXED_GEOMETRY,s)
#include "ng_geometry.h"
#include "residual_bridge.h"
#include "ng_kernels.h"
#include "ng_activity_values.h"
int NG_FIXED_NAME(_activity_values)(const uint8_t*s,uint16_t*a,uint32_t*h){return ng_activity_values_impl(s,a,h);}
#include "ng_bank_symbols.h"
int NG_FIXED_NAME(_bank_symbols)(unsigned c,const uint8_t*const*p,const size_t*n,const uint16_t*const*b,const uint8_t*s,const uint8_t*x,uint8_t*const*v,uint8_t*y,uint16_t*a,int selected){
    return ng_bank_symbols_impl(c,p,n,b,s,x,v,y,a,selected);
}
#include "ng_causal_state.h"
#include "ng_bank_reconstruct.h"
int NG_FIXED_NAME(_bank_wave)(void* const* states,unsigned mode,const int32_t*w,const uint8_t*c,uint8_t*out,const uint8_t*e,size_t n,const ng_decode_context*ctx){
    return ng_bank_reconstruct_core(states,mode,w,c,out,e,n,ctx);
}

int NG_FIXED_NAME(_encode_ex)(const uint8_t* src,const void* cfg,uint8_t* out,size_t cap,size_t* n,uint8_t* sy,uint8_t* cl) {
    return vcn_lc_encode_ex(src,(const vcn_lc_cfg*)cfg,out,cap,n,sy,cl);
}
