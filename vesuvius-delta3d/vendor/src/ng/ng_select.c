#include "ng_select.h"
#include "residual_bridge.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
// Four independent 4x4x4 block sums per vector. A lane's additions remain in
// raster order; there are no horizontal reductions or changed rounding rules.
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
__attribute__((target("avx2"),always_inline))
static inline __m256d ng_score_four_step(const uint8_t* candidate,const uint8_t* classes,const double* cost,__m256d sum){
    for(unsigned x=0;x<4;x++){
        __m128i idx=_mm_setr_epi32(
            (unsigned)classes[x]*256+candidate[x],
            (unsigned)classes[4+x]*256+candidate[4+x],
            (unsigned)classes[8+x]*256+candidate[8+x],
            (unsigned)classes[12+x]*256+candidate[12+x]);
        sum=_mm256_add_pd(sum,_mm256_i32gather_pd(cost,idx,8));
    }
    return sum;
}
__attribute__((target("avx2")))
static void ng_score_four_blocks(const uint8_t* candidate,const uint8_t* classes,
    const uint32_t* block,size_t count,const double* cost,double* sums){
    for(size_t i=0;i<count;i+=16){
        double* dst=sums+block[i];
        __m256d sum=_mm256_loadu_pd(dst);
        sum=ng_score_four_step(candidate+i,classes+i,cost,sum);
        _mm256_storeu_pd(dst,sum);
    }
}
#endif

int vc_ng_select(const uint8_t*const* symbols,const uint8_t*const* classes,
    const double*const* costs,unsigned count,const uint32_t *block,size_t n,
    uint8_t *choices,size_t blocks)
{
    if(!count||count>3)return 0;
    size_t frequencies[3]={0};double prior[3];
    for(size_t b=0;b<blocks;++b){if(choices[b]>=count)return 0;++frequencies[choices[b]];}
    double *scores=calloc(blocks*count,sizeof(*scores));if(!scores)return 0;
    int vector=0;
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    vector=!ng_disable_avx2&&__builtin_cpu_supports("avx2")&&n%16==0;
    if(vector)for(size_t i=0;i<n;i+=16){
        for(unsigned j=0;j<16;++j)if(block[i+j]!=block[i]+j/4){vector=0;break;}
        if(!vector)break;
    }
#endif
    for(unsigned k=0;k<count;++k){
        double *sums=scores+k*blocks;
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
        if(vector)ng_score_four_blocks(symbols[k],classes[k],block,n,costs[k],sums);
        else
#endif
        for(size_t i=0;i<n;){
            unsigned b=block[i];double sum=sums[b];
            do{sum+=costs[k][(unsigned)classes[k][i]*256+symbols[k][i]];++i;}
            while(i<n&&block[i]==b);
            sums[b]=sum;
        }
        prior[k]=log2(blocks+(double)count)-log2(frequencies[k]+1.);
    }
    for(size_t b=0;b<blocks;++b){
        unsigned best=0;double score=scores[b]+prior[0];
        for(unsigned k=1;k<count;++k){double c=scores[k*blocks+b]+prior[k];if(c<score){score=c;best=k;}}
        choices[b]=(uint8_t)best;
    }
    free(scores);return 1;
}
