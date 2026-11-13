/* Encoder-only least-squares fitting and exact SIMD sufficient statistics. */
#pragma once
#include "ng_kernel_common.h"
/* small dense least-squares solve (encoder only) */
static int vcn_solve(double *A, double *b, int n) { /* Cholesky; A n x n SPD, solves in place into b */
  for (int j = 0; j < n; j++) {
    double d = A[j * n + j];
    for (int k = 0; k < j; k++) d -= A[j * n + k] * A[j * n + k];
    if (d <= 1e-12) return -1;
    d = sqrt(d);
    A[j * n + j] = d;
    for (int i = j + 1; i < n; i++) {
      double s2 = A[i * n + j];
      for (int k = 0; k < j; k++) s2 -= A[i * n + k] * A[j * n + k];
      A[i * n + j] = s2 / d;
    }
  }
  for (int i = 0; i < n; i++) {
    double s2 = b[i];
    for (int k = 0; k < i; k++) s2 -= A[i * n + k] * b[k];
    b[i] = s2 / A[i * n + i];
  }
  for (int i = n - 1; i >= 0; i--) {
    double s2 = b[i];
    for (int k = i + 1; k < n; k++) s2 -= A[k * n + i] * b[k];
    b[i] = s2 / A[i * n + i];
  }
  return 0;
}

static void vcn_lc_norm(const uint32_t *cnt, uint16_t *freq, int tb) {
  const uint32_t M = 1u << tb;
  uint64_t tot = 0;
  unsigned maxs = 0;
  for (int s = 0; s < 256; s++) { tot += cnt[s]; if (cnt[s] > cnt[maxs]) maxs = s; }
  memset(freq, 0, 512);
  if (!tot) { freq[0] = (uint16_t)M; return; }
  int32_t sum = 0;
  for (int s = 0; s < 256; s++)
    if (cnt[s]) {
      uint32_t f = (uint32_t)(((uint64_t)cnt[s] * M + tot / 2) / tot);
      freq[s] = (uint16_t)(f < 1 ? 1 : f), sum += freq[s];
    }
  while (sum != (int32_t)M) {
    if (sum < (int32_t)M) { freq[maxs] += (uint16_t)(M - sum); sum = M; break; }
    unsigned best = maxs;
    for (int s = 0; s < 256; s++) if (freq[s] > freq[best]) best = s;
    int32_t d = sum - (int32_t)M, room = freq[best] - 1, t = d < room ? d : room;
    if (t < 1) return; /* cannot happen with <= 256 symbols and M >= 1024 */
    freq[best] = (uint16_t)(freq[best] - t), sum -= t;
  }
}
/* integer weights q near the least-squares w: moves of one step on one weight, or up on one and down on
   another, while they lower the added error (q - w)' C (q - w) (C the centred covariance) */
static void vcn_lc_qopt(const double *C, const double *w, int32_t *q, int n) {
  double g[VCN_LC_MAXF]; /* g = C (q - w), in steps of 1/256 */
  for (int i = 0; i < n; i++) {
    double s = 0;
    for (int k = 0; k < n; k++) s += C[i * n + k] * (q[k] - w[k] * 256.0);
    g[i] = s;
  }
  for (int it = 0; it < 64; it++) {
    double best = -1e-9 * (C[0] + 1e-30);
    int bi = -1, bj = -1, bd = 0;
    for (int i = 0; i < n; i++)
      for (int d = -1; d <= 1; d += 2) {
        const double f = C[i * n + i] + 2 * d * g[i];
        if (f < best) best = f, bi = i, bj = -1, bd = d;
        for (int j = i + 1; j < n; j++) {
          const double f2 = f + C[j * n + j] - 2 * d * g[j] - 2 * C[i * n + j];
          if (f2 < best) best = f2, bi = i, bj = j, bd = d;
        }
      }
    if (bi < 0) break;
    q[bi] += bd;
    for (int k = 0; k < n; k++) g[k] += bd * C[k * n + bi];
    if (bj >= 0) {
      q[bj] -= bd;
      for (int k = 0; k < n; k++) g[k] -= bd * C[k * n + bj];
    }
  }
}
/* G[a][k] += sum over the batch of F[a][s] F[k][s], for the 4 x 4 tiles with a-tile >= k-tile (G is
   ncp x ncp, row a; entries with a >= k are valid); F is ncp columns of VCN_LC_SB int16 at pitch VCN_LC_SBP */
#define VCN_LC_SB 1024
#define VCN_LC_SBP (VCN_LC_SB + 32) /* column pitch: one cache line more, so the columns do not share cache sets */
#include "ng_fit_cache.h"
#if VCN_HAVE_AVX2
__attribute__((target("avx2"))) static int64_t ng_fit_dot_avx2(const int16_t*a,const int16_t*b,size_t n){
  int64_t total=0;
  for(size_t start=0;start<n;start+=VCN_LC_SB){
    __m256i sum=_mm256_setzero_si256();
    for(size_t i=start;i<start+VCN_LC_SB;i+=16)
      sum=_mm256_add_epi32(sum,_mm256_madd_epi16(_mm256_loadu_si256((const __m256i*)(a+i)),_mm256_loadu_si256((const __m256i*)(b+i))));
    int32_t lanes[8];_mm256_storeu_si256((__m256i*)lanes,sum);
    for(int k=0;k<8;k++)total+=lanes[k];
  }return total;
}
#elif VCN_HAVE_NEON
static int64_t ng_fit_dot_neon(const int16_t*a,const int16_t*b,size_t n){
  int64_t total=0;
  for(size_t start=0;start<n;start+=VCN_LC_SB){
    int32x4_t sum=vdupq_n_s32(0);
    for(size_t i=start;i<start+VCN_LC_SB;i+=8){
      int16x8_t x=vld1q_s16(a+i),y=vld1q_s16(b+i);
      sum=vmlal_high_s16(vmlal_s16(sum,vget_low_s16(x),vget_low_s16(y)),x,y);
    }total+=vaddvq_s32(sum);
  }return total;
}
#endif
static int64_t ng_fit_dot(const int16_t*a,const int16_t*b,size_t n){
#if VCN_HAVE_AVX2
  if(vcn_use_avx2())return ng_fit_dot_avx2(a,b,n);
#elif VCN_HAVE_NEON
  return ng_fit_dot_neon(a,b,n);
#endif
  return ng_fit_dot_c(a,b,n);
}
static void ng_fit_store_batch(ng_fit_bank_cache*c,const int16_t*F){
  for(int k=0;k<c->ncol;k++)memcpy(c->columns+(size_t)k*c->capacity+c->stored,F+(size_t)k*VCN_LC_SBP,VCN_LC_SB*sizeof(int16_t));
  c->stored+=VCN_LC_SB;
}
VCN_MAYBE_UNUSED static void vcn_lc_gram_c(const int16_t *F, int ncp, int64_t *G) {
  for (int a = 0; a < ncp; a++)
    for (int k = 0; k <= (a | 3); k++) {
      const int16_t *restrict fa = F + (size_t)a * VCN_LC_SBP, *restrict fk = F + (size_t)k * VCN_LC_SBP;
      int32_t s = 0;
      for (int i = 0; i < VCN_LC_SB; i++) s += fa[i] * fk[i];
      G[(size_t)a * ncp + k] += s;
    }
}
#if VCN_HAVE_AVX2
__attribute__((target("avx2"))) static int ng_gram_sum256(__m256i v){
  __m128i h=_mm_add_epi32(_mm256_castsi256_si128(v),_mm256_extracti128_si256(v,1));
  h=_mm_add_epi32(h,_mm_shuffle_epi32(h,0x4e));return _mm_cvtsi128_si32(_mm_add_epi32(h,_mm_shuffle_epi32(h,0xb1)));
}
#define NG_GATTR __attribute__((target("avx2")))
#define NG_GNAME vcn_lc_gram_avx2
#define NG_GROWS 2
#define NG_GV __m256i
#define NG_GZERO _mm256_setzero_si256
#define NG_GSTEP 16
#define NG_GLOAD _mm256_loadu_si256
#define NG_GADD _mm256_add_epi32
#define NG_GMADD _mm256_madd_epi16
#define NG_GSUM ng_gram_sum256
#include "ng_gram_vector.h"
#define NG_GATTR __attribute__((target("avx512f,avx512bw")))
#define NG_GNAME ng_gram_wide
#define NG_GROWS 4
#define NG_GV __m512i
#define NG_GZERO _mm512_setzero_si512
#define NG_GSTEP 32
#define NG_GLOAD _mm512_loadu_si512
#define NG_GADD _mm512_add_epi32
#define NG_GMADD _mm512_madd_epi16
#define NG_GSUM _mm512_reduce_add_epi32
#include "ng_gram_vector.h"
static void ng_gram_dispatch(const int16_t*F,int ncp,int64_t*G){
if(ng_use_avx512()&&__builtin_cpu_supports("avx512bw"))ng_gram_wide(F,ncp,G);
  else if(vcn_use_avx2())vcn_lc_gram_avx2(F,ncp,G);else vcn_lc_gram_c(F,ncp,G);
}
#define vcn_lc_gram(F,ncp,G) ng_gram_dispatch(F,ncp,G)
#elif VCN_HAVE_NEON
/* the same 4 x 4 tiles with widening multiply-adds into int32 lanes (each lane sums 256 products of at most
   255 * 255, the tile entry at most VCN_LC_SB of them: exact, as in the C form) */
static void vcn_lc_gram_neon(const int16_t *F, int ncp, int64_t *G) {
  for (int a0 = 0; a0 < ncp; a0 += 4)
    for (int k0 = 0; k0 <= a0; k0 += 4) {
      int32x4_t acc[4][4];
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) acc[i][j] = vdupq_n_s32(0);
      const int16_t *fa = F + (size_t)a0 * VCN_LC_SBP, *fk = F + (size_t)k0 * VCN_LC_SBP;
      for (int v = 0; v < VCN_LC_SB; v += 8) {
        int16x8_t A4[4], K4[4];
        for (int i = 0; i < 4; i++) A4[i] = vld1q_s16(fa + (size_t)i * VCN_LC_SBP + v), K4[i] = vld1q_s16(fk + (size_t)i * VCN_LC_SBP + v);
        for (int i = 0; i < 4; i++)
          for (int j = 0; j < 4; j++)
            acc[i][j] = vmlal_high_s16(vmlal_s16(acc[i][j], vget_low_s16(A4[i]), vget_low_s16(K4[j])), A4[i], K4[j]);
      }
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) G[(size_t)(a0 + i) * ncp + k0 + j] += vaddvq_s32(acc[i][j]);
    }
}
#define vcn_lc_gram(F, ncp, G) vcn_lc_gram_neon(F, ncp, G)
#else
#define vcn_lc_gram(F, ncp, G) vcn_lc_gram_c(F, ncp, G)
#endif
/* least squares over the candidate taps (or those of slice 0), greedy selection of nsel, weights rounded
   to 1/256 with their sum kept; returns 1 on success, else the W predictor */
static int vcn_lc_select(double *C, double *cb, double *mf, int *id, int nt, int nsel, double tr) {
    if (nsel > 0 && nsel < nt) { /* greedy forward selection by partial covariances (Schur complements) */
      double *R = (double *)malloc(sizeof(double) * nt * nt), rb[VCN_LC_MAXF];
      memcpy(R, C, sizeof(double) * nt * nt), memcpy(rb, cb, sizeof(double) * nt);
      char used[VCN_LC_MAXF] = {0};
      int ns = 0;
      while (ns < nsel) {
        int best = -1;
        double bg = -1;
        for (int a = 0; a < nt; a++)
          if (!used[a] && R[a * nt + a] > 1e-9 * (tr / nt + 1e-12)) {
            const double g = rb[a] * rb[a] / R[a * nt + a];
            if (g > bg) bg = g, best = a;
          }
        if (best < 0) break;
        used[best] = 1, ns++;
        const double piv = R[best * nt + best];
        double col[VCN_LC_MAXF];
        for (int a = 0; a < nt; a++) col[a] = R[a * nt + best];
        const double rbb = rb[best];
        for (int a = 0; a < nt; a++) {
          const double fa = col[a] / piv;
          rb[a] -= fa * rbb;
          for (int k = 0; k < nt; k++) R[a * nt + k] -= fa * col[k];
        }
      }
      /* keep selected (in index order), rebuild the reduced system */
      int m = 0, keep[VCN_LC_MAXF];
      for (int a = 0; a < nt; a++) if (used[a]) keep[m++] = a;
      double *C2 = (double *)malloc(sizeof(double) * m * m), cb2[VCN_LC_MAXF], mf2[VCN_LC_MAXF];
      int id2[VCN_LC_MAXF];
      for (int a = 0; a < m; a++) {
        cb2[a] = cb[keep[a]], mf2[a] = mf[keep[a]], id2[a] = id[keep[a]];
        for (int k = 0; k < m; k++) C2[a * m + k] = C[keep[a] * nt + keep[k]];
      }
      memcpy(C, C2, sizeof(double) * m * m), memcpy(cb, cb2, sizeof(double) * m), memcpy(mf, mf2, sizeof(double) * m);
      memcpy(id, id2, sizeof(int) * m);
      nt = m;
      free(R), free(C2);
    }
  return nt;
}
static void vcn_lc_quantize(const double *C0, const double *cb, int32_t *q, int nt) {
      /* 1/256 steps; the rounding keeps the sum of the weights (the predictor's gain) */
      double sw = 0, fr[VCN_LC_MAXF];
      long sq = 0;
      for (int k = 0; k < nt; k++) {
        double v = cb[k] * 256.0;
        v = v < -32000 ? -32000 : v > 32000 ? 32000 : v;
        sw += v, q[k] = (int32_t)floor(v + 0.5), sq += q[k], fr[k] = v - q[k];
      }
      const long tgt = (long)floor(sw + 0.5);
      while (sq != tgt) {
        const int dir = tgt > sq ? 1 : -1;
        int bk = 0;
        for (int k = 1; k < nt; k++)
          if (dir * fr[k] > dir * fr[bk]) bk = k;
        q[bk] += dir, fr[bk] -= dir, sq += dir;
      }
      vcn_lc_qopt(C0, cb, q, nt);
}
static int vcn_lc_fit(const uint8_t *b, const vcn_lc_taps *t, int zlo, int zhi, int stride, int slice_only, int nsel, int32_t *wq, int32_t *bias, const uint8_t *feature, const uint8_t *sample_mask,ng_fit_session*session) {
  uint64_t sample_counts[3];
  uint8_t* selected_samples=ng_fit_nonzero_sample_mask(sample_mask,
      VCN_LC_D,VCN_LC_Y,VCN_LC_Z,VCN_LC_XP,zlo,zhi,stride,
      sample_counts);
  if(selected_samples){sample_mask=selected_samples;stride=1;}
  int id[VCN_LC_MAXF], nt = 0;
  for (int k = 0; k < t->nt; k++)
    if ((!slice_only || t->dz[k] == 0)) id[nt++] = k;
  if(feature)id[nt++]=t->nt;
  const int n = nt + 1;
  /* normal equations in exact integers: the samples' values as int16 columns (taps, the constant 1, the
     target), batches of VCN_LC_SB samples, every column pair's dot product in int32 per batch (at most
     VCN_LC_SB * 255 * 255 < 2^31) summed into int64; the sums equal the double ones bit for bit */
  const int ncol = nt + 2, ncp = (ncol + 3) & ~3, TC = nt + 1;
  ng_fit_bank_cache* cache=session&&feature&&!sample_mask?session->banks+slice_only:NULL;
  const int reuse=cache&&cache->count&&cache->nt==nt&&cache->ncp==ncp;
  double *A = (double *)calloc((size_t)n * n, sizeof(double)), *bv = (double *)calloc(n, sizeof(double));
  int16_t *F = (int16_t *)malloc((size_t)ncp * VCN_LC_SBP * sizeof(int16_t));
  int64_t *G = (int64_t *)calloc((size_t)ncp * ncp, sizeof(int64_t));
  memset(wq, 0, sizeof(int32_t) * (t->nt+(feature!=NULL)));
  *bias = 0;
  if (!A || !bv || !F || !G) {
    free(A), free(bv), free(F), free(G);
    free(selected_samples);
    wq[0] = 256;
    return 0;
  }
  long cnt = 0;
  int ns = 0;
  memset(F, 0, (size_t)ncp * VCN_LC_SBP * sizeof(int16_t));
  if(cache&&!reuse){
    ng_fit_bank_clear(cache);cache->nt=nt;cache->ncol=ncol;cache->ncp=ncp;
    cache->capacity=(((size_t)(zhi-zlo)*VCN_LC_Y*((VCN_LC_D+stride-1)/stride)+VCN_LC_SB-1)/VCN_LC_SB)*VCN_LC_SB;
    cache->columns=malloc((size_t)ncol*cache->capacity*sizeof(int16_t));
    cache->positions=malloc(cache->capacity*sizeof(size_t));
    cache->gram=malloc((size_t)ncp*ncp*sizeof(int64_t));
    if(!cache->columns||!cache->positions||!cache->gram){ng_fit_bank_clear(cache);cache=NULL;}
  }
  if(reuse){
    memcpy(G,cache->gram,(size_t)ncp*ncp*sizeof(int64_t));cnt=(long)cache->count;
    int16_t*column=cache->columns+(size_t)(nt-1)*cache->capacity;
    for(size_t i=0;i<cache->count;i++)column[i]=feature[cache->positions[i]];
    memset(column+cache->count,0,(cache->stored-cache->count)*sizeof(int16_t));
    for(int k=0;k<ncol;k++){
      int a=k>nt-1?k:nt-1,c=k<nt-1?k:nt-1;
      G[(size_t)a*ncp+c]=ng_fit_dot(column,cache->columns+(size_t)k*cache->capacity,cache->stored);
    }
  }else{
  for (int z = zlo; z < zhi; z++)
    for (int y = 0; y < VCN_LC_Y; y++) {
      for (int x = VCN_LC_XP + ng_fit_sample_phase(y,z,stride); x < VCN_LC_D; x += stride) {
        if(sample_mask&&!sample_mask[((size_t)z*VCN_LC_Y+y)*VCN_LC_D+x])continue;
        const size_t i = VCN_LC_IDX(z, y, x);
        if(cache)cache->positions[cnt]=((size_t)z*VCN_LC_Y+y)*VCN_LC_D+x;
        unsigned extra=feature?feature[((size_t)z*VCN_LC_Y+y)*VCN_LC_D+x]:0;
        for (int k = 0; k < nt; k++) {
          int value=id[k]<t->nt?b[i+t->off[id[k]]]:(int)extra;
          F[(size_t)k*VCN_LC_SBP+ns]=(int16_t)value;
        }
        F[(size_t)nt * VCN_LC_SBP + ns] = 1;
        F[(size_t)TC * VCN_LC_SBP + ns] = b[i];
        cnt++;
        if (++ns == VCN_LC_SB) {
          vcn_lc_gram(F,ncp,G);
          if(cache)ng_fit_store_batch(cache,F);
          ns=0;
        }
      }
    }
  if (ns) {
    for (int k = 0; k < ncp; k++) memset(F + (size_t)k * VCN_LC_SBP + ns, 0, (size_t)(VCN_LC_SB - ns) * sizeof(int16_t));
    vcn_lc_gram(F,ncp,G);
    if(cache)ng_fit_store_batch(cache,F);
  }
  if(cache){cache->count=cnt;memcpy(cache->gram,G,(size_t)ncp*ncp*sizeof(int64_t));}
  }
  for (int a = 0; a < n; a++) {
    for (int k = 0; k <= a; k++) A[(size_t)a * n + k] = (double)G[(size_t)a * ncp + k];
    bv[a] = (double)G[(size_t)TC * ncp + a];
  }
  free(F), free(G);
  int ok = 0;
  if (cnt >= 4 * n) {
    /* Adding lambda*bias^2 or lambda all-zero observations changes only the
     * intercept diagonal. Eliminate that intercept analytically; keep the
     * normalization by real sample mass, so both formulations are identical. */
    double *C = (double *)malloc(sizeof(double) * nt * nt), cb[VCN_LC_MAXF], mf[VCN_LC_MAXF], cn = A[(size_t)nt*n+nt], tr = 0;
    double intercept_ratio=1.+ng_fit_intercept_weight;
    double intercept_mass=cn*intercept_ratio, mt=bv[nt]/intercept_mass;
    for (int a = 0; a < nt; a++) mf[a] = A[(size_t)nt * n + a] / intercept_mass;
    for (int a = 0; a < nt; a++) {
      cb[a] = bv[a] / cn - mf[a] * mt * intercept_ratio;
      for (int k = 0; k < nt; k++) {
        const double s = a >= k ? A[(size_t)a * n + k] : A[(size_t)k * n + a];
        C[a * nt + k] = s / cn - mf[a] * mf[k] * intercept_ratio;
      }
      tr += C[a * nt + a];
    }
    for (int a = 0; a < nt; a++) C[a * nt + a] += (1e-6 * tr / nt + 1e-9);
    nt = vcn_lc_select(C,cb,mf,id,nt,nsel,tr);
    double *C0 = (double *)malloc(sizeof(double) * nt * nt);
    if (C0) memcpy(C0, C, sizeof(double) * nt * nt);
    if (C0 && vcn_solve(C, cb, nt) == 0) {
      int32_t q[VCN_LC_MAXF];
      vcn_lc_quantize(C0, cb, q, nt);
      double bb = mt;
      for (int k = 0; k < nt; k++) wq[id[k]] = q[k], bb -= q[k] / 256.0 * mf[k];
      *bias = (int32_t)(uint16_t)(int32_t)floor(bb * 256.0 + 0.5);
      ok = 1;
    }
    free(C), free(C0);
  }
  if (!ok) {
    wq[0] = 256; /* W */
  }
  free(A), free(bv);
  free(selected_samples);
  return ok;
}
