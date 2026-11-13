#ifndef NG_ELOOKUP
#define NG_ELOOKUP(INDEX,C,S) (hardware_gather ? _mm256_i32gather_epi32(T,INDEX,4) : vcn_lc_gather(T,INDEX))
#endif
#define NG_ENTROPY_STEP(H,EN,S2) do { \
      const __m256i c = _mm256_sll_epi32(NG_ECLASS(H), tbv); \
      const __m256i index = _mm256_add_epi32(c, _mm256_and_si256(S[H], kM)); \
      EN = NG_ELOOKUP(index,c,S[H]); \
      S2 = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_and_si256(_mm256_srli_epi32(EN, 8), kF), k1), \
                                                       _mm256_srl_epi32(S[H], tbv)), \
                                    _mm256_srli_epi32(EN, 20)); \
  } while(0)
    __m256i Y[4];
#ifdef NG_EUNROLL
#if defined(__clang__)
#pragma clang loop unroll(full)
#else
#pragma GCC unroll 4
#endif
#endif
    for (int h = 0; h < 4; h++) {
      __m256i en,s2;
      NG_ENTROPY_STEP(h,en,s2);
      const __m256i m = _mm256_cmpgt_epi32(kL, s2);
      const unsigned bits = (unsigned)_mm256_movemask_ps(_mm256_castsi256_ps(m));
      if (end - p >= 16) { /* no branch on the mask: an empty mask reads nothing */
        const __m256i wv = _mm256_permutevar8x32_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)p)),
                                                       _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)vcn_lc_perm[bits])));
        s2 = _mm256_blendv_epi8(s2, _mm256_or_si256(_mm256_slli_epi32(s2, 16), wv), m);
        p += 2 * (unsigned)__builtin_popcount(bits);
      } else if (bits) {
        uint32_t sv[8];
        _mm256_storeu_si256((__m256i *)sv, s2);
        for (int k = 0; k < 8; k++)
          if (bits >> k & 1u) {
            if (end - p >= 2) sv[k] = (sv[k] << 16) | vcn_rd16(p), p += 2;
            else sv[k] <<= 16, over++;
          }
        s2 = _mm256_loadu_si256((const __m256i *)sv);
      }
      S[h] = s2;
      Y[h] = _mm256_and_si256(en, kB);
    }
    const __m256i y = _mm256_packus_epi16(_mm256_packus_epi32(Y[0], Y[1]), _mm256_packus_epi32(Y[2], Y[3]));
    NG_EOUTPUT(_mm256_permutevar8x32_epi32(y, ix));
#undef NG_ENTROPY_STEP
