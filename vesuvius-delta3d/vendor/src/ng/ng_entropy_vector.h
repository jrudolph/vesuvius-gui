#define NG_ERESTRICT restrict
NG_EATTR static inline void NG_ENAME(int n, const uint8_t *NG_ERESTRICT cl, vcn_lc_dec *D,
                                                                    uint8_t *NG_ERESTRICT sym, int hardware_gather
#ifdef NG_EEXTRA
 NG_EEXTRA
#endif
) {
  const int tb = D->tb;
  const __m256i kM = _mm256_set1_epi32((int)((1u << tb) - 1u)), kF = _mm256_set1_epi32(0xFFF), k1 = _mm256_set1_epi32(1),
                kL = _mm256_set1_epi32((int)VCN_LC_RL), kB = _mm256_set1_epi32(255), ix = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
  const __m128i tbv = _mm_cvtsi32_si128(tb);
  __m256i S[4];
  for (int h = 0; h < 4; h++) S[h] = _mm256_loadu_si256((const __m256i *)(D->S + 8 * h));
  const int *T = (const int *)D->T;
  const uint8_t *p = D->p, *end = D->end;
  size_t over = D->over;
#define NG_ECLASS(H) _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(cl+i+8*(H))))
#define NG_EOUTPUT(V) _mm256_storeu_si256((__m256i*)(sym+i),V)
  for (int i = 0; i < n; i += 32) {
#include "ng_entropy_step.inc"
  }
  for (int h = 0; h < 4; h++) _mm256_storeu_si256((__m256i *)(D->S + 8 * h), S[h]);
  D->p=p;
  D->over = over;
}
#undef NG_ENTROPY_STEP
#undef NG_ECLASS
#undef NG_EOUTPUT
#undef NG_EATTR
#undef NG_ENAME
#undef NG_EUNROLL
#undef NG_EEXTRA
#undef NG_ELOOKUP
#undef NG_ERESTRICT
