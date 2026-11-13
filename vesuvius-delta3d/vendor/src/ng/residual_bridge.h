#pragma once
#include <stddef.h>
#include <stdint.h>
#include "ng_geometry.h"
#include "ng_symbol_mapping.h"

extern _Thread_local unsigned ng_choice_edge;
void* ng_fit_session_create(void);
void ng_fit_session_destroy(void*);
int ng_bank_symbols(unsigned,const uint8_t* const*,const size_t*,const uint16_t* const*,const uint8_t*,const uint8_t*,uint8_t* const*,uint8_t*,uint16_t*);
int ng_bank_selected_symbols(unsigned,const uint8_t* const*,const size_t*,const uint16_t* const*,const uint8_t*,const uint8_t*,uint8_t* const*,uint8_t*,uint16_t*);
void* ng_causal_create(const uint8_t* params,size_t n,int history);
void ng_causal_destroy(void* state);
/* config: r0,r1,r2,classes,tap budget,fit stride. Validated before use. */
int ng_prepare(const uint8_t*,const int*,const uint8_t*,const uint8_t*,void*,int32_t*,
    uint8_t*,size_t,size_t*,uint8_t*,uint8_t*,uint16_t*,uint16_t*,uint32_t*);
int ng_wave_encode(const uint8_t*,size_t,const uint8_t*,const uint8_t*,uint8_t*,size_t,size_t*);
/* Per-call decoder state; no experimental controls or nested TLS overrides. */
typedef struct {
    unsigned model_count, classes;
    const uint8_t *banks, *payload;
    size_t payload_size;
    int frequent_x;
} ng_decode_context;
int ng_choice_wave_decode(void* const*,unsigned,unsigned,const int32_t*,const uint8_t*,const uint8_t*,uint8_t*,const uint8_t*,size_t,const uint8_t*,size_t,unsigned);
int ng_entropy_header_size(const uint8_t*,size_t,size_t*);
/* Size of the single retained model representation, from its six-byte cfg. */
size_t ng_model_size(const uint8_t*);
int ng_bank_empty_model(uint8_t*,size_t,size_t*);
size_t ng_write_freq_table(uint8_t*,const uint16_t*);
int ng_read_freq_table(const uint8_t*,size_t,uint16_t*,size_t*);
int ng_read_freq_table_scaled(const uint8_t*,size_t,uint16_t*,unsigned,size_t*);
double ng_hist_cost_logs(const uint32_t*,int,const double*);
int ng_set_isa_limit(int); /* 0 automatic, 256 excludes AVX-512, 128 also AVX2 (plain C kernels); per thread */
int ng_activity_symbol_values(const uint8_t*,uint16_t*);
int ng_activity_symbol_histogram(const uint8_t*,uint16_t*,uint32_t*);
/* Shared exact directional bases. Clamping is before modulo conversion, not
 * clamping the wrapped byte. Missing selected axes retain the legacy fallback. */
typedef struct {int constant,left,clamp,lo,hi;} ng_base_parts;
static inline ng_base_parts ng_directional_parts_xyz(const uint8_t*s,size_t i,unsigned m,unsigned x,unsigned y,unsigned z,unsigned d,size_t plane){
    const unsigned masks[4]={3,5,6,7};unsigned mask=m<8?m:masks[m-8];
    unsigned available=(x?1:0)|(y?2:0)|(z?4:0);
    int clamp=m>=8&&(mask&available)==mask;mask&=available;
    int b=(mask&2)?s[i-d]:0,c=(mask&4)?s[i-plane]:0;
    int p=b+c,lo=255,hi=0;
    if(mask&2){if(b<lo)lo=b;if(b>hi)hi=b;}
    if(mask&4){if(c<lo)lo=c;if(c>hi)hi=c;}
    if((mask&3)==3)p-=s[i-d-1];
    if((mask&5)==5)p-=s[i-plane-1];
    if((mask&6)==6)p-=s[i-plane-d];
    if(mask==7)p+=s[i-plane-d-1];
    ng_base_parts r={p,(mask&1)!=0,clamp,lo,hi};return r;
}
static inline uint8_t ng_directional_base_xyz(const uint8_t*s,size_t i,unsigned m,unsigned x,unsigned y,unsigned z,unsigned d,size_t plane){
    ng_base_parts r=ng_directional_parts_xyz(s,i,m,x,y,z,d,plane);int a=r.left?s[i-1]:0,p=r.constant+a;
    if(r.clamp){if(r.left){if(a<r.lo)r.lo=a;if(a>r.hi)r.hi=a;}p=p<r.lo?r.lo:p>r.hi?r.hi:p;}
    return (uint8_t)p;
}
static inline uint8_t ng_directional_base(const uint8_t*s,size_t i,unsigned m,unsigned x,unsigned y,unsigned z,unsigned d){
    return ng_directional_base_xyz(s,i,m,x,y,z,d,(size_t)d*d);
}
