#pragma once
/* Shared native NG software gather, usable by causal entropy routing too. */
__attribute__((target("avx2"))) static inline __m256i vcn_lc_gather(const int *base, __m256i idx) {
  const __m128i lo = _mm256_castsi256_si128(idx), hi = _mm256_extracti128_si256(idx, 1);
  __m128i a = _mm_cvtsi32_si128(base[_mm_cvtsi128_si32(lo)]);
  a = _mm_insert_epi32(a, base[_mm_extract_epi32(lo, 1)], 1);
  a = _mm_insert_epi32(a, base[_mm_extract_epi32(lo, 2)], 2);
  a = _mm_insert_epi32(a, base[_mm_extract_epi32(lo, 3)], 3);
  __m128i b = _mm_cvtsi32_si128(base[_mm_cvtsi128_si32(hi)]);
  b = _mm_insert_epi32(b, base[_mm_extract_epi32(hi, 1)], 1);
  b = _mm_insert_epi32(b, base[_mm_extract_epi32(hi, 2)], 2);
  b = _mm_insert_epi32(b, base[_mm_extract_epi32(hi, 3)], 3);
  return _mm256_inserti128_si256(_mm256_castsi128_si256(a), b, 1);
}
