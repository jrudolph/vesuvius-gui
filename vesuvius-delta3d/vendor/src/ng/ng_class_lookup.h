/* Shared exact activity -> predictor bias / entropy class lookup. */
#if VCN_HAVE_AVX2
__attribute__((target("avx2"))) static void ng_bank_class_lookup(const uint8_t*e,size_t ri,const uint32_t*table,uint16_t*restrict pre,uint8_t*restrict entropy_class){
    const __m256i mask=_mm256_set1_epi32(65535);
    __m256i act=_mm256_setzero_si256();
    for(int x=0;x<VCN_LC_D;x+=8){
        if(!(x&15))act=vcn_lc_activity16(e,ri+x);
        __m128i a=(x&8)?_mm256_extracti128_si256(act,1):_mm256_castsi256_si128(act);
        __m256i ix=_mm256_cvtepu16_epi32(a);
        __m256i values=_mm256_i32gather_epi32((const int*)table,ix,4);
        __m256i bias=_mm256_and_si256(values,mask);
        __m128i b=_mm_packus_epi32(_mm256_castsi256_si128(bias),_mm256_extracti128_si256(bias,1));
        _mm_storeu_si128((__m128i*)(pre+x),_mm_add_epi16(_mm_loadu_si128((const __m128i*)(pre+x)),b));
        __m256i cls=_mm256_srli_epi32(values,16);
        __m128i c=_mm_packus_epi32(_mm256_castsi256_si128(cls),_mm256_extracti128_si256(cls,1));
        _mm_storel_epi64((__m128i*)(entropy_class+x),_mm_packus_epi16(c,c));
    }
}
#ifdef NG_HAVE_WIDE_PRE
__attribute__((target("avx512f,avx512bw"),always_inline)) static inline __m512i ng_bank_class_wide32(const uint8_t*e,size_t ri,const uint32_t*table,__m512i pre,uint8_t*cls){
        __m512i a=vcn_lc_activity32(e,ri);
        __m512i lo=_mm512_i32gather_epi32(_mm512_cvtepu16_epi32(_mm512_castsi512_si256(a)),table,4);
        __m512i hi=_mm512_i32gather_epi32(_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(a,1)),table,4);
        __m512i bias=_mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtepi32_epi16(lo)),_mm512_cvtepi32_epi16(hi),1);
        _mm_storeu_si128((__m128i*)cls,_mm512_cvtepi32_epi8(_mm512_srli_epi32(lo,16)));
        _mm_storeu_si128((__m128i*)(cls+16),_mm512_cvtepi32_epi8(_mm512_srli_epi32(hi,16)));
        return _mm512_add_epi16(pre,bias);
}
__attribute__((target("avx512f,avx512bw"))) static void ng_bank_class_wide(const uint8_t*e,size_t ri,const uint32_t*table,uint16_t*pre,uint8_t*cls){
    for(int x=0;x<VCN_LC_D;x+=32){
        _mm512_storeu_si512(pre+x,ng_bank_class_wide32(e,ri+x,table,_mm512_loadu_si512(pre+x),cls+x));
    }
}
#endif
#endif
