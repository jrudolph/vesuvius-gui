/* Shared exact recurrence; instantiated for16 or32 independent rows. */
#ifndef NG_RMT
#define NG_RMT NG_RV
#define NG_RMASKVAL NG_RAND
#endif
NG_RATTR static inline void NG_RNAME(int width,const ng_bank_compact_column*cols,const uint16_t w[3][32],const uint16_t*feature,uint16_t prev[3][32],uint8_t*result,int homogeneous
#ifdef NG_RPACKED
,const uint8_t (*block_flags)[32],const uint8_t (*raw_bytes)[32]
#endif
#ifdef NG_RFUSED
,const uint8_t*up,const uint8_t*back,const uint8_t*upback,int neighbor_stride
#endif
#ifdef NG_RBLOCKW
/* block_w[k*blocks+b][row]: in-row weight k (k<3) or the feature weight (k=3) of selector block b */
,const uint16_t (*block_w)[32]
#endif
){
    const unsigned selector_edge=ng_choice_edge;
#ifdef NG_RPACKED
    const unsigned selector_shift=selector_edge==2?1:selector_edge==4?2:selector_edge==8?3:selector_edge==16?4:5;
#endif
    const NG_RV one=NG_RSET(1),two=NG_RSET(2),four=NG_RSET(4),bytes=NG_RSET(255);
    const int span=8;
    for(int start=0;start<width;start+=span)for(int h=0;h<NG_RHALVES;h++){
        NG_RV p0=NG_RLOAD((const NG_RV*)(prev[0]+NG_RSTEP*h));
        NG_RV p1=NG_RLOAD((const NG_RV*)(prev[1]+NG_RSTEP*h));
        NG_RV p2=NG_RLOAD((const NG_RV*)(prev[2]+NG_RSTEP*h));
#ifdef NG_RBLOCKW
        NG_RV w0,w1,w2,f;
#else
        NG_RV w0=NG_RLOAD((const NG_RV*)(w[0]+NG_RSTEP*h));
        NG_RV w1=NG_RLOAD((const NG_RV*)(w[1]+NG_RSTEP*h));
        NG_RV w2=NG_RLOAD((const NG_RV*)(w[2]+NG_RSTEP*h));
        NG_RV f=NG_RLOAD((const NG_RV*)(feature+NG_RSTEP*h));
#endif
#ifdef NG_RFUSED
        NG_RV previous_b=NG_RSET(0),previous_d=NG_RSET(0),previous_bd=NG_RSET(0);
        if(start){
            NG_RV b=_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(up+(start-1)*neighbor_stride)));
            NG_RV d=_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(back+(start-1)*neighbor_stride)));
            NG_RV bd=_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(upback+(start-1)*neighbor_stride)));
            previous_b=b;previous_d=d;previous_bd=bd;
        }
#endif
        for(int x=start;x<start+span&&x<width;){
            const ng_bank_compact_column*c=cols+x;
#define NG_COMPACT_LOAD(F) NG_RLOAD((const NG_RV*)(c->F+NG_RSTEP*h))
#ifdef NG_RBLOCKW
            {
                const int blocks=VCN_LC_D/(int)selector_edge,block=x/(int)selector_edge;
                w0=NG_RLOAD((const NG_RV*)(block_w[block]+NG_RSTEP*h));
                w1=NG_RLOAD((const NG_RV*)(block_w[blocks+block]+NG_RSTEP*h));
                w2=NG_RLOAD((const NG_RV*)(block_w[2*blocks+block]+NG_RSTEP*h));
                f=NG_RLOAD((const NG_RV*)(block_w[3*blocks+block]+NG_RSTEP*h));
            }
#endif
            NG_RV flags;
#ifdef NG_RPACKED
            flags=_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)block_flags[x>>selector_shift]));
#ifdef NG_RFUSED
            NG_RV direction=NG_RSRL(flags,4);
            if(!x)direction=NG_RAND(direction,NG_RSET(6));
            flags=NG_RAND(flags,NG_RSET(15));
#endif
            if(!x)flags=NG_RAND(flags,NG_RBLEND(NG_RSET(13),NG_RSET(9),NG_REQ(NG_RAND(flags,two),two)));
#else
            flags=NG_COMPACT_LOAD(flags);
#endif
            flags=NG_RAND(flags,bytes);
            NG_RMT ng=NG_REQ(NG_RAND(flags,one),one);
            NG_RMT left=NG_REQ(NG_RAND(flags,two),two);
            NG_RMT clamp=NG_REQ(NG_RAND(flags,four),four);
            NG_RV factor=NG_RBLEND(NG_RSET(256),f,ng);
            int simple=homogeneous||NG_RMASK(NG_REQ(flags,NG_RSET(11)))==-1;
            int xyz=NG_RMASK(NG_REQ(flags,NG_RSET(7)))==-1;
#ifdef NG_RFUSED
            int pair=NG_RMASK(NG_REQ(NG_RAND(flags,NG_RSET(3)),NG_RSET(3)))==-1;
#else
            const int pair=0;
#endif
            NG_RV wx=NG_RADD(w0,f);
            /* Four adjacent positions share the block selector, except x=0
             * where the causal left neighbor is absent. Bounds remain per pixel. */
            int end=x?((x|(selector_edge-1))+1):x+1;
            if(end>width)end=width;
            if(end>start+span)end=start+span;
            for(;x<end;x++){
            c=cols+x;
#ifdef NG_RFUSED
            NG_RV b=_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(up+x*neighbor_stride)));
            NG_RV d=_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(back+x*neighbor_stride)));
            NG_RV bd=_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(upback+x*neighbor_stride)));
#define NG_DIR_HAS(B) NG_REQ(NG_RAND(direction,NG_RSET(B)),NG_RSET(B))
            NG_RMT hasy=NG_DIR_HAS(2),hasz=NG_DIR_HAS(4);
            NG_RV direct_constant=NG_RADD(NG_RMASKVAL(hasy,b),NG_RMASKVAL(hasz,d));
            direct_constant=NG_RSUB(direct_constant,NG_RMASKVAL(NG_DIR_HAS(3),previous_b));
            direct_constant=NG_RSUB(direct_constant,NG_RMASKVAL(NG_DIR_HAS(5),previous_d));
            direct_constant=NG_RSUB(direct_constant,NG_RMASKVAL(NG_DIR_HAS(6),bd));
            direct_constant=NG_RADD(direct_constant,NG_RMASKVAL(NG_DIR_HAS(7),previous_bd));
            previous_b=b;previous_d=d;previous_bd=bd;
            NG_RV lo=NG_RMIN(NG_RBLEND(bytes,b,hasy),NG_RBLEND(bytes,d,hasz));
            NG_RV hi=NG_RMAX(NG_RMASKVAL(hasy,b),NG_RMASKVAL(hasz,d));
            NG_RV direct_bounds=_mm512_or_si512(lo,NG_RSLL(hi,8));
#undef NG_DIR_HAS
#define NG_BASE_CONST direct_constant
#define NG_BASE_BOUNDS direct_bounds
#define NG_BASE_PRE NG_RMASKVAL(ng,NG_COMPACT_LOAD(pre))
#else
#define NG_BASE_CONST NG_COMPACT_LOAD(constant)
#define NG_BASE_BOUNDS NG_COMPACT_LOAD(bounds)
#define NG_BASE_PRE NG_COMPACT_LOAD(pre)
#endif
#include "ng_recur_math.inc"
            }
#undef NG_COMPACT_LOAD
        }
        NG_RSTORE((NG_RV*)(prev[0]+NG_RSTEP*h),p0);NG_RSTORE((NG_RV*)(prev[1]+NG_RSTEP*h),p1);NG_RSTORE((NG_RV*)(prev[2]+NG_RSTEP*h),p2);
    }
}
#include "ng_recur_undef.h"
