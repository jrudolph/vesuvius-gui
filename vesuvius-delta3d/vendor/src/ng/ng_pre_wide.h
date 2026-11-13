/* Same predictor arithmetic, optional in-register activity/bias/class finish. */
#define NG_PATTR __attribute__((target("avx512f,avx512bw")))
#define NG_PV __m512i
#define NG_PWIDTH 64
#define NG_PSET _mm512_set1_epi16
#define NG_PZERO _mm512_setzero_si512
#define NG_PLOAD _mm512_loadu_si512
#define NG_PADD _mm512_add_epi16
#define NG_PMADD _mm512_maddubs_epi16
#define NG_PLO _mm512_unpacklo_epi8
#define NG_PHI _mm512_unpackhi_epi8
#define NG_PSHIFT _mm512_slli_epi16
#ifdef NG_PFUSED
#define NG_PSTORE(P,L,H) do { \
  __m512i a=_mm512_permutex2var_epi64(L,_mm512_setr_epi64(0,1,8,9,2,3,10,11),H); \
  __m512i b=_mm512_permutex2var_epi64(L,_mm512_setr_epi64(4,5,12,13,6,7,14,15),H); \
  _mm512_storeu_si512(P,ng_bank_class_wide32(e,ri+x,table,a,classes+x)); \
  _mm512_storeu_si512((P)+32,ng_bank_class_wide32(e,ri+x+32,table,b,classes+x+32)); \
} while(0)
#else
#define NG_PSTORE(P,L,H) do { \
  _mm512_storeu_si512((P),_mm512_permutex2var_epi64(L,_mm512_setr_epi64(0,1,8,9,2,3,10,11),H)); \
  _mm512_storeu_si512((P)+32,_mm512_permutex2var_epi64(L,_mm512_setr_epi64(4,5,12,13,6,7,14,15),H)); \
} while(0)
#endif
#include "ng_pre_vector.h"
#undef NG_PFUSED
