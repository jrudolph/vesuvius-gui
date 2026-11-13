/* Fixed predictor palette; exact modulo-byte arithmetic and AVX2 row kernels.
 * Specializing a single selected predictor keeps inactive arithmetic out of
 * both scalar and SIMD loops. No fast-math or approximate predictions. */
#include "ng_predict.h"
#include "residual_bridge.h"
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
__attribute__((target("avx2"),always_inline))
static inline unsigned ng_fixed_row_avx2(const uint8_t* src,uint8_t*const* out,size_t row,unsigned d,unsigned Outputs){
    const size_t plane=(size_t)d*d;unsigned x=1;
    const __m256i zero=_mm256_setzero_si256(),mask=_mm256_set1_epi16(255);
    for(;x<d;x+=16){
        // A short final group can overlap the previous store: source samples
        // are immutable and each result is independent. Avoid a scalar tail.
        if(x+16>d)x=d-16;
        const size_t i=row+x;
#define NG_LOAD16(off) _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(src+i-(off))))
        const __m256i v=NG_LOAD16(0),a=NG_LOAD16(1),b=NG_LOAD16(d),c=NG_LOAD16(plane);
        __m256i p[12];p[0]=zero;p[1]=a;p[2]=b;p[4]=c;
        p[3]=_mm256_sub_epi16(_mm256_add_epi16(a,b),NG_LOAD16(d+1));
        p[5]=_mm256_sub_epi16(_mm256_add_epi16(a,c),NG_LOAD16(plane+1));
        p[6]=_mm256_sub_epi16(_mm256_add_epi16(b,c),NG_LOAD16(plane+d));
        p[7]=_mm256_add_epi16(_mm256_sub_epi16(_mm256_add_epi16(_mm256_add_epi16(p[3],p[5]),p[6]),_mm256_add_epi16(_mm256_add_epi16(a,b),c)),NG_LOAD16(plane+d+1));
#undef NG_LOAD16
        p[8]=_mm256_min_epi16(_mm256_max_epi16(p[3],_mm256_min_epi16(a,b)),_mm256_max_epi16(a,b));
        p[9]=_mm256_min_epi16(_mm256_max_epi16(p[5],_mm256_min_epi16(a,c)),_mm256_max_epi16(a,c));
        p[10]=_mm256_min_epi16(_mm256_max_epi16(p[6],_mm256_min_epi16(b,c)),_mm256_max_epi16(b,c));
        p[11]=_mm256_min_epi16(_mm256_max_epi16(p[7],_mm256_min_epi16(a,_mm256_min_epi16(b,c))),_mm256_max_epi16(a,_mm256_max_epi16(b,c)));
        for(unsigned m=0;m<12;m++)if(Outputs&(1u<<m)){
            __m256i r=_mm256_and_si256(_mm256_sub_epi16(v,p[m]),mask);
            r=_mm256_and_si256(_mm256_xor_si256(_mm256_slli_epi16(r,1),_mm256_sub_epi16(zero,_mm256_srli_epi16(r,7))),mask);
            _mm_storeu_si128((__m128i*)(out[m]+i),_mm_packus_epi16(_mm256_castsi256_si128(r),_mm256_extracti128_si256(r,1)));
        }
    }
    return x;
}
__attribute__((target("avx2")))
static unsigned fixed_row_0(const uint8_t*s,uint8_t*const*out,size_t row,unsigned d){
    return ng_fixed_row_avx2(s,out,row,d,1);
}
__attribute__((target("avx2")))
static unsigned fixed_row_1(const uint8_t*s,uint8_t*const*out,size_t row,unsigned d){
    return ng_fixed_row_avx2(s,out,row,d,2);
}
__attribute__((target("avx2")))
static unsigned fixed_row_2(const uint8_t*s,uint8_t*const*out,size_t row,unsigned d){
    return ng_fixed_row_avx2(s,out,row,d,4);
}
__attribute__((target("avx2")))
static unsigned fixed_row_3(const uint8_t*s,uint8_t*const*out,size_t row,unsigned d){
    return ng_fixed_row_avx2(s,out,row,d,8);
}
__attribute__((target("avx2")))
static unsigned fixed_row_4(const uint8_t*s,uint8_t*const*out,size_t row,unsigned d){
    return ng_fixed_row_avx2(s,out,row,d,16);
}
__attribute__((target("avx2")))
static unsigned fixed_row_5(const uint8_t*s,uint8_t*const*out,size_t row,unsigned d){
    return ng_fixed_row_avx2(s,out,row,d,32);
}
__attribute__((target("avx2")))
static unsigned fixed_row_6(const uint8_t*s,uint8_t*const*out,size_t row,unsigned d){
    return ng_fixed_row_avx2(s,out,row,d,64);
}
__attribute__((target("avx2")))
static unsigned fixed_row_7(const uint8_t*s,uint8_t*const*out,size_t row,unsigned d){
    return ng_fixed_row_avx2(s,out,row,d,128);
}
__attribute__((target("avx2")))
static unsigned fixed_row_8(const uint8_t*s,uint8_t*const*out,size_t row,unsigned d){
    return ng_fixed_row_avx2(s,out,row,d,256);
}
__attribute__((target("avx2")))
static unsigned fixed_row_9(const uint8_t*s,uint8_t*const*out,size_t row,unsigned d){
    return ng_fixed_row_avx2(s,out,row,d,512);
}
__attribute__((target("avx2")))
static unsigned fixed_row_10(const uint8_t*s,uint8_t*const*out,size_t row,unsigned d){
    return ng_fixed_row_avx2(s,out,row,d,1024);
}
__attribute__((target("avx2")))
static unsigned fixed_row_11(const uint8_t*s,uint8_t*const*out,size_t row,unsigned d){
    return ng_fixed_row_avx2(s,out,row,d,2048);
}
__attribute__((target("avx2")))
static unsigned fixed_row_12(const uint8_t*s,uint8_t*const*out,size_t row,unsigned d){
    return ng_fixed_row_avx2(s,out,row,d,4095);
}
static inline unsigned fixed_row(const uint8_t*s,uint8_t*const*out,size_t row,unsigned d,unsigned m){
    switch(m){
    case 0:return fixed_row_0(s,out,row,d);
    case 1:return fixed_row_1(s,out,row,d);
    case 2:return fixed_row_2(s,out,row,d);
    case 3:return fixed_row_3(s,out,row,d);
    case 4:return fixed_row_4(s,out,row,d);
    case 5:return fixed_row_5(s,out,row,d);
    case 6:return fixed_row_6(s,out,row,d);
    case 7:return fixed_row_7(s,out,row,d);
    case 8:return fixed_row_8(s,out,row,d);
    case 9:return fixed_row_9(s,out,row,d);
    case 10:return fixed_row_10(s,out,row,d);
    case 11:return fixed_row_11(s,out,row,d);
    default:return fixed_row_12(s,out,row,d);
    }
}
#endif

static inline uint8_t pyramid(uint8_t r) {return (uint8_t)(((unsigned)r<<1)^(0u-((unsigned)r>>7)));}
static int min_i(int a,int b){return a<b?a:b;}
static int max_i(int a,int b){return a>b?a:b;}
static int clamp_i(int x,int l,int h){return x<l?l:x>h?h:x;}
static inline void fixed_single(const uint8_t* __restrict src,uint8_t* __restrict out,unsigned d,unsigned M){
    int simd=0;
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    simd=!ng_disable_avx2&&__builtin_cpu_supports("avx2");
    uint8_t* outputs[12]={0};outputs[M]=out;
#endif
    for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++){
        const size_t row=((size_t)z*d+y)*d;
        out[row]=pyramid((uint8_t)(src[row]-ng_directional_base(src,row,M,0,y,z,d)));
        // Dispatch once per volume, with constant predictor and interior-axis
        // availability. This lets the compiler vectorize independent values;
        // the predictor definition and boundary fallback remain shared.
        unsigned begin=1;
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
        if(simd&&z&&y)begin=fixed_row(src,outputs,row,d,M);
#endif
        for(unsigned x=begin;x<d;x++){
            size_t i=row+x;
            out[i]=pyramid((uint8_t)(src[i]-ng_directional_base(src,i,M,1,y,z,d)));
        }
    }
}
void vc_ng_fixed_candidates(const uint8_t* restrict src,uint8_t* const* buffers,unsigned d,unsigned mask){

    if(mask!=4095){
        // Fixed selections need only the requested predictor. Share the same
        // boundary/predictor definition rather than constructing other arrays.
        for(unsigned m=0;m<12;m++)if(mask&(1u<<m)){
            switch(m){
#define NG_FIXED_SINGLE(M) case M: fixed_single(src,buffers[m],d,M);break;
                NG_FIXED_SINGLE(0) NG_FIXED_SINGLE(1) NG_FIXED_SINGLE(2) NG_FIXED_SINGLE(3)
                NG_FIXED_SINGLE(4) NG_FIXED_SINGLE(5) NG_FIXED_SINGLE(6) NG_FIXED_SINGLE(7)
                NG_FIXED_SINGLE(8) NG_FIXED_SINGLE(9) NG_FIXED_SINGLE(10) NG_FIXED_SINGLE(11)
#undef NG_FIXED_SINGLE
            }
        }
        return;
    }
    // Generate the complete fixed palette with one shared neighborhood load.
    uint8_t* out[12];for(unsigned m=0;m<12;m++)out[m]=buffers[m];
    const size_t plane=(size_t)d*d;
    int simd=0;
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    simd=!ng_disable_avx2&&__builtin_cpu_supports("avx2");
#endif
    for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++){
        const size_t row=((size_t)z*d+y)*d;
        if(!z||!y){
            for(unsigned m=0;m<12;m++)for(unsigned x=0;x<d;x++)out[m][row+x]=pyramid((uint8_t)(src[row+x]-ng_directional_base(src,row+x,m,x,y,z,d)));
            continue;
        }
        for(unsigned m=0;m<12;m++)out[m][row]=pyramid((uint8_t)(src[row]-ng_directional_base(src,row,m,0,y,z,d)));
        unsigned begin=1;
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
        if(simd)begin=fixed_row_12(src,out,row,d);
#endif
        for(unsigned x=begin;x<d;x++){
            size_t i=row+x;int v=src[i],a=src[i-1],b=src[i-d],c=src[i-plane];
            int xy=a+b-src[i-d-1],xz=a+c-src[i-plane-1],yz=b+c-src[i-plane-d];
            int xyz=xy+xz+yz-a-b-c+src[i-plane-d-1];
            out[0][i]=pyramid((uint8_t)(v));out[1][i]=pyramid((uint8_t)(v-a));out[2][i]=pyramid((uint8_t)(v-b));out[3][i]=pyramid((uint8_t)(v-xy));
            out[4][i]=pyramid((uint8_t)(v-c));out[5][i]=pyramid((uint8_t)(v-xz));out[6][i]=pyramid((uint8_t)(v-yz));out[7][i]=pyramid((uint8_t)(v-xyz));
            out[8][i]=pyramid((uint8_t)(v-clamp_i(xy,min_i(a,b),max_i(a,b))));
            out[9][i]=pyramid((uint8_t)(v-clamp_i(xz,min_i(a,c),max_i(a,c))));
            out[10][i]=pyramid((uint8_t)(v-clamp_i(yz,min_i(b,c),max_i(b,c))));
            out[11][i]=pyramid((uint8_t)(v-clamp_i(xyz,min_i(a,min_i(b,c)),max_i(a,max_i(b,c)))));
        }
    }
}
