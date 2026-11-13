/* Shared native C entrypoints for the retained presets.
 * MIT License, Copyright (c) 2026 Chase Hendrick; see LICENSE. */
#include "residual_bridge.h"
#include <limits.h>
#include <string.h>

_Thread_local ng_geometry ng_current_geometry={NG_BLOCK_DIM,NG_BLOCK_DIM,NG_BLOCK_DIM};
_Thread_local unsigned ng_choice_edge=4;
_Thread_local int ng_disable_avx512=0,ng_disable_avx2=0;
int ng_set_isa_limit(int bits){
    if(bits!=0&&bits!=256&&bits!=128)return -1;
    ng_disable_avx512=bits!=0;ng_disable_avx2=bits==128;return 0;
}
_Thread_local double ng_fit_intercept_weight=0.; /* relative to real sample mass */

int ng_geometry_valid(uint32_t x,uint32_t y,uint32_t z) {
    /* Bounds make padded pitches and signed SIMD offsets safe too. */
    return x>=32 && y>=32 && z>=32 && !(x%32) && !(y%32) && !(z%32) &&
           (uint64_t)y*z<=65536 && x<=1024;
}

#include "ng_kernel_common.h"
size_t ng_model_size(const uint8_t* c) {
    if(!c || c[0]!=VCN_LC_VERSION || c[1]<1 || c[1]>3 ||
       (int8_t)c[2]<-1 || (int8_t)c[2]>3 || (int8_t)c[3]<-1 || (int8_t)c[3]>3 ||
       c[4]<2 || c[4]>128 || c[5]<10 || c[5]>12)return 0;
    vcn_lc_taps t;vcn_lc_build(&t,c[1],(int8_t)c[2],(int8_t)c[3]);
    return 6+vcn_lc_parameter_body_bytes((unsigned)t.nt,c[4]);
}

int ng_bank_empty_model(uint8_t* out,size_t cap,size_t* size){
    vcn_lc_taps t;vcn_lc_build(&t,1,0,0);
    size_t need=6+4*(t.nt+1)+2*15+2*16;if(cap<need)return -1;
    memset(out,0,need);out[0]=VCN_LC_VERSION;out[1]=1;out[4]=16;out[5]=12;*size=need;return 0;
}


int ng_entropy_header_size(const uint8_t* data,size_t n,size_t* size){
    const uint8_t* p=data;vcn_lc_par* par=NULL;if(vcn_lc_readpar(&p,n,&par)!=VCN_OK)return -1;
    int rc=vcn_lc_read_tables(&p,data+n,NULL,par->K,1u<<par->tb);
    free(par);if(rc)return -1;*size=p-data;return 0;
}


size_t ng_write_freq_table(uint8_t* out,const uint16_t* freq) {return vcn_lc_write_table(out,freq);}
int ng_read_freq_table(const uint8_t* src,size_t n,uint16_t* freq,size_t* used) {
    return ng_read_freq_table_scaled(src,n,freq,4096,used);
}
int ng_read_freq_table_scaled(const uint8_t* src,size_t n,uint16_t* freq,unsigned total,size_t* used) {
    const uint8_t* p=src;int rc=vcn_lc_read_table(&p,src+n,freq,total);*used=p-src;return rc;
}
