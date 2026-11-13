#pragma once
/* Shared exact recurrence for native reconstruction and entropy integration. */
#define NG_WRESTRICT restrict
typedef struct {
    uint16_t pre[32],w0[32],w1[32],w2[32],feature[32],residual[32];
    int16_t constant[32],left[32],clamp[32],lo[32],hi[32];
} ng_bank_wave_column;
typedef struct {uint16_t pre[32],constant[32],flags[32],bounds[32],residual[32];} ng_bank_compact_column;
static inline uint16_t ng_wave_prediction(const ng_bank_wave_column*c,const uint16_t prev[3][32],unsigned j){
    int16_t left=(int16_t)(c->left[j]*prev[0][j]),base=c->constant[j]+left,lo=c->lo[j],hi=c->hi[j];
    if(c->left[j]){lo=left<lo?left:lo;hi=left>hi?left:hi;}if(c->clamp[j])base=base<lo?lo:base>hi?hi:base;
    return (uint16_t)(c->pre[j]+(uint32_t)c->w0[j]*prev[0][j]+(uint32_t)c->w1[j]*prev[1][j]+(uint32_t)c->w2[j]*prev[2][j]+(uint32_t)c->feature[j]*(uint8_t)base);
}
#define NG_BANK_WAVE_KERNEL(name,attr) \
attr static void name(int width,const ng_bank_wave_column* NG_WRESTRICT a,uint16_t prev[3][32],uint8_t* NG_WRESTRICT result){ \
    for(int x=0;x<width;x++){ \
        const ng_bank_wave_column* NG_WRESTRICT c=a+x; \
        for(int j=0;j<32;j++){ \
            uint16_t q=ng_wave_prediction(c,prev,j); \
            uint16_t r=c->residual[j];if(q&128)r=(uint16_t)-r; \
            uint8_t v=(uint8_t)(((uint16_t)(q+128u)>>8)+r); \
            result[x*32+j]=v;prev[2][j]=prev[1][j];prev[1][j]=prev[0][j];prev[0][j]=v; \
        } \
    } \
}
NG_BANK_WAVE_KERNEL(ng_bank_wave_c,)
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
NG_BANK_WAVE_KERNEL(ng_bank_wave_avx2,__attribute__((target("avx2"))))
#endif
#undef NG_BANK_WAVE_KERNEL
#undef NG_WRESTRICT
