/* Encoder-only prediction, symbol generation and wavefront rANS writing. */
#pragma once
#include "ng_kernel_fit.h"
/* the encoder's row kernels. pass 1: P = pre + the in-row taps, |the bias-free residual|; symbols: from
   P with its class bias, the zigzag of the residual (sign flipped when P rounds up), |residual| */
static void vcn_lc_pass1_span(const uint16_t *restrict pre, const uint8_t *restrict br, const uint16_t *w, uint16_t *restrict Q,
                           uint8_t *restrict er,int count) {
  for (int x = 0; x < count; x++) {
    const uint16_t P = (uint16_t)(pre[x] + (uint16_t)(w[0] * br[x - 1]) + (uint16_t)(w[1] * br[x - 2]) + (uint16_t)(w[2] * br[x - 3]));
    const int r = (int)(int8_t)(uint8_t)(br[x] - VCN_LC_PP(P));
    Q[x] = P;
    er[x] = (uint8_t)(r < 0 ? -r : r);
  }
}
VCN_MAYBE_UNUSED static void vcn_lc_pass1_c(const uint16_t *restrict pre,const uint8_t *restrict br,const uint16_t*w,uint16_t*restrict Q,uint8_t*restrict er){vcn_lc_pass1_span(pre,br,w,Q,er,VCN_LC_D);}
#include "ng_symbol_mapping.h"
VCN_MAYBE_UNUSED static void vcn_lc_syms_enc_c(const uint16_t *restrict Q, const uint8_t *restrict vr, uint8_t *restrict sr, uint8_t *restrict er) {
  for (int x = 0; x < VCN_LC_D; x++) {
    const unsigned sy = ng_prediction_symbol(vr[x],Q[x]);
    sr[x] = (uint8_t)sy, er[x] = (uint8_t)((sy + 1) >> 1);
  }
}
#if VCN_HAVE_AVX2
__attribute__((target("avx2"))) static void vcn_lc_pass1_avx2(const uint16_t *restrict pre, const uint8_t *restrict br, const uint16_t *w,
                                                              uint16_t *restrict Q, uint8_t *restrict er) {
  const __m256i w0 = _mm256_set1_epi16((short)w[0]), w1 = _mm256_set1_epi16((short)w[1]), w2 = _mm256_set1_epi16((short)w[2]),
                k128 = _mm256_set1_epi16(128);
  for (int x = 0; x < VCN_LC_D; x += 16) {
#define VCN_LC_L8(q) _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(q)))
    const __m256i P = _mm256_add_epi16(_mm256_add_epi16(_mm256_loadu_si256((const __m256i *)(pre + x)), _mm256_mullo_epi16(VCN_LC_L8(br + x - 1), w0)),
                                       _mm256_add_epi16(_mm256_mullo_epi16(VCN_LC_L8(br + x - 2), w1), _mm256_mullo_epi16(VCN_LC_L8(br + x - 3), w2)));
    const __m256i r = _mm256_sub_epi16(VCN_LC_L8(br + x), _mm256_srli_epi16(_mm256_add_epi16(P, k128), 8));
    const __m256i a = _mm256_abs_epi16(_mm256_srai_epi16(_mm256_slli_epi16(r, 8), 8));
    _mm256_storeu_si256((__m256i *)(Q + x), P);
    _mm_storeu_si128((__m128i *)(er + x), _mm_packus_epi16(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1)));
  }
}
__attribute__((target("avx2"))) static void vcn_lc_syms_enc_avx2(const uint16_t *restrict Q, const uint8_t *restrict vr, uint8_t *restrict sr,
                                                                 uint8_t *restrict er) {
  const __m256i k128 = _mm256_set1_epi16(128), k1 = _mm256_set1_epi16(1);
  for (int x = 0; x < VCN_LC_D; x += 16) {
    const __m256i P = _mm256_loadu_si256((const __m256i *)(Q + x));
    const __m256i d = _mm256_sub_epi16(VCN_LC_L8(vr + x), _mm256_srli_epi16(_mm256_add_epi16(P, k128), 8));
    const __m256i ev = _mm256_srai_epi16(_mm256_slli_epi16(d, 8), 8);
    const __m256i fm = _mm256_srai_epi16(_mm256_slli_epi16(P, 8), 15); /* -1 when P rounds up */
    const __m256i sg = _mm256_srai_epi16(_mm256_slli_epi16(_mm256_sub_epi16(_mm256_xor_si256(ev, fm), fm), 8), 8);
    const __m256i sy = _mm256_xor_si256(_mm256_slli_epi16(sg, 1), _mm256_srai_epi16(sg, 15));
    const __m256i sy8 = _mm256_and_si256(sy, _mm256_set1_epi16(255));
    const __m256i ea = _mm256_srli_epi16(_mm256_add_epi16(sy8, k1), 1);
    _mm_storeu_si128((__m128i *)(sr + x), _mm_packus_epi16(_mm256_castsi256_si128(sy8), _mm256_extracti128_si256(sy8, 1)));
    _mm_storeu_si128((__m128i *)(er + x), _mm_packus_epi16(_mm256_castsi256_si128(ea), _mm256_extracti128_si256(ea, 1)));
#undef VCN_LC_L8
  }
}
#define VCN_LC_PASS1(pre, br, w, Q, er) (vcn_use_avx2() ? vcn_lc_pass1_avx2(pre, br, w, Q, er) : vcn_lc_pass1_c(pre, br, w, Q, er))
#define VCN_LC_SYMS(Q, vr, sr, er) (vcn_use_avx2() ? vcn_lc_syms_enc_avx2(Q, vr, sr, er) : vcn_lc_syms_enc_c(Q, vr, sr, er))
#elif VCN_HAVE_NEON
static void vcn_lc_pass1_neon(const uint16_t *restrict pre, const uint8_t *restrict br, const uint16_t *w, uint16_t *restrict Q,
                              uint8_t *restrict er) {
  const uint16x8_t k128 = vdupq_n_u16(128);
  for (int x = 0; x < VCN_LC_D; x += 16) {
    const uint8x16_t b1 = vld1q_u8(br + x - 1), b2 = vld1q_u8(br + x - 2), b3 = vld1q_u8(br + x - 3), b0 = vld1q_u8(br + x);
    uint16x8_t Pl = vld1q_u16(pre + x), Ph = vld1q_u16(pre + x + 8);
    Pl = vmlaq_n_u16(vmlaq_n_u16(vmlaq_n_u16(Pl, vmovl_u8(vget_low_u8(b1)), w[0]), vmovl_u8(vget_low_u8(b2)), w[1]), vmovl_u8(vget_low_u8(b3)), w[2]);
    Ph = vmlaq_n_u16(vmlaq_n_u16(vmlaq_n_u16(Ph, vmovl_high_u8(b1), w[0]), vmovl_high_u8(b2), w[1]), vmovl_high_u8(b3), w[2]);
    /* the residual modulo 256 as int8, then its magnitude (|-128| = 128, as the C form) */
    const uint8x16_t pp = vcombine_u8(vshrn_n_u16(vaddq_u16(Pl, k128), 8), vshrn_n_u16(vaddq_u16(Ph, k128), 8));
    vst1q_u16(Q + x, Pl), vst1q_u16(Q + x + 8, Ph);
    vst1q_u8(er + x, vreinterpretq_u8_s8(vabsq_s8(vreinterpretq_s8_u8(vsubq_u8(b0, pp)))));
  }
}
static void vcn_lc_syms_enc_neon(const uint16_t *restrict Q, const uint8_t *restrict vr, uint8_t *restrict sr, uint8_t *restrict er) {
  const uint16x8_t k128 = vdupq_n_u16(128);
  for (int x = 0; x < VCN_LC_D; x += 16) {
    const uint16x8_t Pl = vld1q_u16(Q + x), Ph = vld1q_u16(Q + x + 8);
    const uint8x16_t pp = vcombine_u8(vshrn_n_u16(vaddq_u16(Pl, k128), 8), vshrn_n_u16(vaddq_u16(Ph, k128), 8));
    const int8x16_t ev = vreinterpretq_s8_u8(vsubq_u8(vld1q_u8(vr + x), pp));
    /* -1 where P rounds up (bit 7 of P), else 0; the sign flip and the zigzag in int8, wrapping as the C form */
    const int8x16_t fm = vshrq_n_s8(vreinterpretq_s8_u8(vcombine_u8(vmovn_u16(Pl), vmovn_u16(Ph))), 7);
    const int8x16_t sg = vsubq_s8(veorq_s8(ev, fm), fm);
    const uint8x16_t sy = vreinterpretq_u8_s8(veorq_s8(vshlq_n_s8(sg, 1), vshrq_n_s8(sg, 7)));
    vst1q_u8(sr + x, sy);
    vst1q_u8(er + x, vrshrq_n_u8(sy, 1)); /* (sy + 1) >> 1 without overflow */
  }
}
#define VCN_LC_PASS1(pre, br, w, Q, er) vcn_lc_pass1_neon(pre, br, w, Q, er)
#define VCN_LC_SYMS(Q, vr, sr, er) vcn_lc_syms_enc_neon(Q, vr, sr, er)
#else
#define VCN_LC_PASS1(pre, br, w, Q, er) vcn_lc_pass1_c(pre, br, w, Q, er)
#define VCN_LC_SYMS(Q, vr, sr, er) vcn_lc_syms_enc_c(Q, vr, sr, er)
#endif

/* encode one chunk: fit; one pass in raster order for the bias-free predictions, the activity histogram
   and the residual sums (row-parallel); thresholds and biases; a second raster pass for the classes and
   symbols; then rANS (reciprocal form) over the decoder's order */
typedef struct {
  int r0, r1, r2, K, tb, nsel, stride;
  uint16_t *export_prediction;
  uint16_t *export_activity; /* Optional causal activity, reused by entropy-only tuning. */
  uint32_t *export_activity_histogram;
  /* Already prepared symbols/classes: share the native entropy encoder. */
  const uint8_t *import_sym, *import_cls, *import_params;
  size_t import_params_n;
  const uint8_t *fit_feature;
  int32_t *export_feature_weights;
  const uint8_t *fit_mask; /* optional membership mask for model-bank regression */
  ng_fit_session *fit_session; /* optional same-chunk independent-fit reuse */
} vcn_lc_cfg;
typedef struct { uint32_t xmax, rcp, bias; uint16_t cmpl, shift; } vcn_lc_esym;
static inline void vcn_lc_init_esym(vcn_lc_esym*E,uint32_t f,uint32_t cum,uint32_t M,int tb){
  E->xmax=((VCN_LC_RL>>tb)<<16)*f;E->cmpl=(uint16_t)(M-f);
  if(f<2){E->rcp=~0u;E->shift=32;E->bias=cum+M-1;}
  else{uint32_t sh=0;while(f>(1u<<sh))sh++;E->rcp=(uint32_t)(((1ull<<(sh+31))+f-1)/f);E->shift=(uint16_t)(sh-1+32);E->bias=cum;}
}
static inline uint32_t vcn_lc_put_esym(uint32_t xs,const vcn_lc_esym*E,uint16_t*words,size_t*nw){
  unsigned fl=xs>=E->xmax;words[*nw]=(uint16_t)xs;*nw+=fl;xs>>=fl<<4;
  uint32_t q=(uint32_t)(((uint64_t)xs*E->rcp)>>E->shift);return xs+E->bias+q*E->cmpl;
}
#if VCN_HAVE_AVX2
/* Sixteen independent states, same reverse renormalization order as scalar.
   Compact the emitted words only after all state transitions are evaluated. */
__attribute__((target("avx512f,avx512bw,avx2")))
static size_t ng_rans_encode_row_wide(const uint8_t*cr,const uint8_t*sr,int width,
    const vcn_lc_esym*table,uint32_t*S,uint16_t*words,size_t nw){
  const __m512i reverse=_mm512_setr_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
  for(int x=width;x>0;x-=16){
    const int lane=(x-16)&(VCN_LC_NS-1);
    __m512i c=_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(cr+x-16)));
    __m512i s=_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(sr+x-16)));
    __m512i idx=_mm512_permutexvar_epi32(reverse,_mm512_slli_epi32(_mm512_add_epi32(_mm512_slli_epi32(c,8),s),2));
    const uint32_t*t=(const uint32_t*)table;
    __m512i xmax=_mm512_i32gather_epi32(idx,t,4),rcp=_mm512_i32gather_epi32(idx,t+1,4);
    __m512i bias=_mm512_i32gather_epi32(idx,t+2,4),cs=_mm512_i32gather_epi32(idx,t+3,4);
    __m512i old=_mm512_permutexvar_epi32(reverse,_mm512_loadu_si512(S+lane));
    __mmask16 flush=_mm512_cmpge_epu32_mask(old,xmax);
    __m512i v=_mm512_mask_srli_epi32(old,flush,old,16);
    __m512i even=_mm512_srli_epi64(_mm512_mul_epu32(v,rcp),32);
    __m512i odd=_mm512_mul_epu32(_mm512_srli_epi64(v,32),_mm512_srli_epi64(rcp,32));
    __m512i high=_mm512_mask_blend_epi32(0xaaaa,even,odd);
    __m512i q=_mm512_srlv_epi32(high,_mm512_sub_epi32(_mm512_srli_epi32(cs,16),_mm512_set1_epi32(32)));
    __m512i next=_mm512_add_epi32(_mm512_add_epi32(v,bias),_mm512_mullo_epi32(q,_mm512_and_si512(cs,_mm512_set1_epi32(65535))));
    _mm512_storeu_si512(S+lane,_mm512_permutexvar_epi32(reverse,next));
    __m512i emitted=_mm512_maskz_compress_epi32(flush,old);
    _mm256_storeu_si256((__m256i*)(words+nw),_mm512_cvtepi32_epi16(emitted));
    nw+=(unsigned)__builtin_popcount((unsigned)flush);
  }
  return nw;
}
#endif
/* Shared native model serialization, also used by fit-only experiments. */
static size_t vcn_lc_writepar(const vcn_lc_par*p,const vcn_lc_cfg*cf,uint8_t*hdr){
  size_t pos=0;hdr[pos++]=VCN_LC_VERSION;hdr[pos++]=cf->r0;hdr[pos++]=cf->r1;hdr[pos++]=cf->r2;hdr[pos++]=p->K;hdr[pos++]=p->tb;
  for(int b=0;b<2;b++){
    for(int k=0;k<p->t.nt;k++){vcn_wr16(hdr+pos,(uint16_t)(int16_t)p->w[b][k]);pos+=2;}
    vcn_wr16(hdr+pos,(uint16_t)p->bias[b]);pos+=2;
  }
  for(int k=0;k<p->K-1;k++){vcn_wr16(hdr+pos,p->thr[k]);pos+=2;}
  for(int k=0;k<p->K;k++){vcn_wr16(hdr+pos,(uint16_t)(int16_t)p->cb[k]);pos+=2;}
  return pos;
}
static vcn_status vcn_lc_encode_ex(const uint8_t *src, const vcn_lc_cfg *cf, uint8_t *out, size_t cap, size_t *out_n,
                                  uint8_t *export_sym, uint8_t *export_cls) {
  const size_t NV = (size_t)VCN_LC_D * VCN_LC_Y * VCN_LC_Z;
  int32_t feature_weights[2]={0,0};
  const int K = cf->K, tb = cf->tb;
  const uint32_t M = 1u << tb;
  vcn_lc_par *p = (vcn_lc_par *)calloc(1, sizeof(vcn_lc_par));
  const int imported=cf->import_sym!=NULL;
  uint8_t *b = imported?NULL:cf->fit_session&&cf->fit_session->padded?cf->fit_session->padded:(uint8_t *)malloc(VCN_LC_NB), *e = imported?NULL:(uint8_t *)malloc(VCN_LC_NB);
  if(cf->fit_session)cf->fit_session->padded=b;
  uint16_t *Pa = imported?NULL:(uint16_t *)malloc(2 * NV);
  uint32_t(*h)[VCN_LC_AMAX] = imported?NULL:(uint32_t(*)[VCN_LC_AMAX])calloc(1, sizeof *h);
  int64_t(*sa)[VCN_LC_AMAX] = imported?NULL:(int64_t(*)[VCN_LC_AMAX])calloc(1, sizeof *sa);
  uint8_t *sym = imported?NULL:export_sym?export_sym:(uint8_t *)malloc(NV);
  const int row_classes=export_sym&&!export_cls;
  uint8_t *cls = imported?NULL:export_cls?export_cls:(uint8_t *)malloc(row_classes?VCN_LC_D:NV);
  const uint8_t *coded_sym=imported?cf->import_sym:sym, *coded_cls=imported?cf->import_cls:cls;
  uint32_t(*cnt)[256] = export_sym?NULL:(uint32_t(*)[256])calloc(K, sizeof *cnt);
  vcn_lc_esym(*es)[256] = export_sym?NULL:(vcn_lc_esym(*)[256])malloc(K * sizeof *es);
  uint16_t *words = export_sym?NULL:(uint16_t *)malloc(2 * (NV + 64));
  uint8_t *hdr = (uint8_t *)malloc(9 + 2 * (2 * VCN_LC_MAXT + 4) + 4 * VCN_LC_MAXK + K * (2 + 2 * 256));
  vcn_lc_sched sc = {0, NULL, NULL};
  vcn_status st = VCN_OK;
  if (!p || (!imported && (!b || !e || !Pa || !h || !sa || !sym || !cls)) || !coded_sym || !coded_cls || (!export_sym&&(!cnt || !es || !words)) || !hdr) {
    st = VCN_ERR_NOMEM;
    goto done;
  }
  size_t pos = 0;
  vcn_lc_build(&p->t, cf->r0, cf->r1, cf->r2);
  p->K = K, p->tb = tb;
  const vcn_lc_taps *t = &p->t;
  if (cf->import_sym) {
    for(size_t i=0;i<NV;i++) {
      if(coded_cls[i]>=K){st=VCN_ERR_CORRUPT;goto done;}
      cnt[coded_cls[i]][coded_sym[i]]++;
    }
    pos=cf->import_params_n;
    if(pos>9+2*(2*VCN_LC_MAXT+4)+4*VCN_LC_MAXK){st=VCN_ERR_CAP;goto done;}
    memcpy(hdr,cf->import_params,pos);
    goto entropy_ready;
  }
  /* the padded volume in its final state, for the fit */
  if(!cf->fit_session||!cf->fit_session->padded_ready){
  memset(b, 128, VCN_LC_NB);
  for (int z = 0; z < VCN_LC_Z; z++)
    for (int y = 0; y < VCN_LC_Y; y++) {
      vcn_lc_row_begin(b, z, y, 128);
      memcpy(b + VCN_LC_IDX(z, y, 0), src + ((size_t)z * VCN_LC_Y + y) * VCN_LC_D, VCN_LC_D);
      vcn_lc_row_end(b, z, y);
    }
  if(cf->fit_session)cf->fit_session->padded_ready=1;
  }
  if(cf->fit_feature){
    for(int bank=0;bank<2;bank++){
      int32_t weights[VCN_LC_MAXF]={0};
      vcn_lc_fit(b,t,bank?0:1,bank?1:VCN_LC_Z,bank?(cf->stride==1?1:2):cf->stride,bank,bank?cf->nsel/2:cf->nsel,
          weights,&p->bias[bank],cf->fit_feature,cf->fit_mask,cf->fit_session);
      memcpy(p->w[bank],weights,t->nt*sizeof(*weights));
      feature_weights[bank]=weights[t->nt];
      if(cf->export_feature_weights)cf->export_feature_weights[bank]=feature_weights[bank];
    }
  }else if(cf->fit_mask){
    for(int bank=0;bank<2;bank++)
      vcn_lc_fit(b,t,bank?0:1,bank?1:VCN_LC_Z,bank?(cf->stride==1?1:2):cf->stride,bank,bank?cf->nsel/2:cf->nsel,
          p->w[bank],&p->bias[bank],NULL,cf->fit_mask,NULL);
  }else{
    vcn_lc_fit(b,t,1,VCN_LC_Z,cf->stride,0,cf->nsel,p->w[0],&p->bias[0],NULL,NULL,NULL);
    vcn_lc_fit(b,t,0,1,cf->stride==1?1:2,1,cf->nsel/2,p->w[1],&p->bias[1],NULL,NULL,NULL);
  }
  memset(h,0,sizeof(*h));memset(sa,0,sizeof(*sa));
  if(cnt)memset(cnt,0,(size_t)K*sizeof(*cnt));
  if(cf->export_feature_weights)for(int bank=0;bank<2;bank++)cf->export_feature_weights[bank]=feature_weights[bank];
  {
    vcn_lc_plist pl[2];
    vcn_lc_mkplist(t, p->w[0], &pl[0]), vcn_lc_mkplist(t, p->w[1], &pl[1]);
    _Alignas(32) uint16_t pre[VCN_LC_D], ap[VCN_LC_D];
    if(!cf->fit_session)memset(b,128,VCN_LC_NB);
    memset(e, 0, VCN_LC_NB);
    for (int z = 0; z < VCN_LC_Z; z++) {
      const int s0 = z == 0;
      uint16_t w[3] = {0, 0, 0};
      for (int k = 0; k < t->nrow; k++) w[k] = (uint16_t)p->w[s0][k];
      for (int y = 0; y < VCN_LC_Y; y++) {
        vcn_lc_row_begin(b, z, y, 128), vcn_lc_row_begin(e, z, y, 0);
        const size_t ri = VCN_LC_IDX(z, y, 0), o = ((size_t)z * VCN_LC_Y + y) * VCN_LC_D;
        uint8_t *restrict br = b + ri, *restrict er = e + ri;
        if(!cf->fit_session)memcpy(br, src + o, VCN_LC_D);

        VCN_LC_PRE(br, &pl[s0], (uint16_t)p->bias[s0], pre);
         vcn_lc_apre(e, ri, ap);
        uint16_t *restrict Q = Pa + o;
        // Add optional inputs before the native SIMD predictor/residual pass.
        // Only low16 bits survive; modular additions commute exactly. This
        // avoids computing residuals twice and a scalar64-bit multiply/voxel.
        if(cf->fit_feature)for(int x=0;x<VCN_LC_D;x++){
          pre[x]=(uint16_t)(pre[x]+(unsigned)(uint16_t)feature_weights[s0]*cf->fit_feature[o+x]);
        }
        VCN_LC_PASS1(pre, br, w, Q, er);
        /* activity histogram and the sums of 256 v - P per activity value */
        for (int x = 0; x < VCN_LC_D; x++) {
          /* Calibrate only on fitted targets; bypassed blocks must not bias the correction. */
          if (cf->fit_mask && !cf->fit_mask[o+x]) continue;
          const int a = ap[x], q = 0;
          h[q][a]++, sa[q][a] += (int16_t)(uint16_t)(br[x] * 256u - Q[x]);
        }
        vcn_lc_row_end(b, z, y), vcn_lc_row_end(e, z, y);
      }
    }
  }
  /* classes: quantiles of the bias-free activity; biases: mean of 256 v - P per class (exact integer sums,
     so the order of the sums does not matter) */
  {
    uint64_t acc = 0, calibration_count=NV;
    if(cf->fit_mask){
      calibration_count=0;
      for(int a=0;a<VCN_LC_AMAX;a++)calibration_count+=h[0][a];
    }

    int k = 0;
    for (int v = 0; v < VCN_LC_AMAX && k < K - 1; v++) {
      acc += h[0][v];
      while (k < K - 1 && acc * (uint64_t)K >= (uint64_t)(k + 1) * calibration_count) p->thr[k++] = (uint16_t)v;
    }
    while (k < K - 1) p->thr[k++] = VCN_LC_AMAX - 1;
    vcn_lc_mklut(p);
    int64_t sum[VCN_LC_MAXK] = {0}, n[VCN_LC_MAXK] = {0};
    for (int a = 0; a < VCN_LC_AMAX; a++) sum[p->lut[a]] += sa[0][a], n[p->lut[a]] += h[0][a];
    for (int c = 0; c < K; c++) {
      double m = n[c] ? floor((double)sum[c] / (double)n[c] + 0.5) : 0.0;
      m = m < -32768 ? -32768 : m > 32767 ? 32767 : m;
      p->cb[c] = (int32_t)m;
    }
  }
  /* symbols, in raster order: the class from the activity of the biased residual magnitudes of earlier
     rows (the decoder has them as soon as it has those rows' symbols: |e| = (symbol + 1) >> 1) */
  {
    memset(e, 0, VCN_LC_NB);
    if(cf->export_activity_histogram)memset(cf->export_activity_histogram,0,VCN_LC_AMAX*sizeof(uint32_t));
    for (int z = 0; z < VCN_LC_Z; z++)
      for (int y = 0; y < VCN_LC_Y; y++) {
        vcn_lc_row_begin(e, z, y, 0);
        const size_t ri = VCN_LC_IDX(z, y, 0), o = ((size_t)z * VCN_LC_Y + y) * VCN_LC_D;
        uint8_t *restrict er = e + ri, *restrict cr = cls + (row_classes?0:o), *restrict sr = sym + o;
        const uint8_t *restrict vr = src + o;
        uint16_t *restrict Q = Pa + o; /* += the class bias */

         if(cf->export_activity){
#if VCN_HAVE_AVX2
          if(vcn_use_avx2() && p->K<=16)ng_cls_export_avx2(e,ri,p,Q,cr,cf->export_activity+o);
          else
#endif
          {
          uint16_t*a=cf->export_activity+o;vcn_lc_apre(e,ri,a);
          for(int x=0;x<VCN_LC_D;x++){
            cr[x]=p->lut[a[x]];Q[x]=(uint16_t)(Q[x]+p->cb[cr[x]]);
          }
          }
        }else{
          if(cf->export_activity)vcn_lc_apre(e,ri,cf->export_activity+o);
          VCN_LC_CLS(e, ri, p, Q, cr);
        }
        if(cf->export_activity_histogram)for(int x=0;x<VCN_LC_D;x++)cf->export_activity_histogram[cf->export_activity[o+x]]++;
        if (cf->export_prediction) memcpy(cf->export_prediction+o,Q,2*VCN_LC_D);
        VCN_LC_SYMS(Q, vr, sr, er);
        if (!export_sym) for (int x = 0; x < VCN_LC_D; x++) cnt[cr[x]][sr[x]]++;
        vcn_lc_row_end(e, z, y);
      }
  }
  pos=vcn_lc_writepar(p,cf,hdr);
  /* Optional entropy-backend handoff: same fitted predictor and coded symbols.
     Stop before writing any rANS probability tables. */
  if (export_sym) {
    if (pos > cap) st = VCN_ERR_CAP;
    else {
      memcpy(out,hdr,pos); *out_n=pos;
      /* Caller-owned symbol/class storage is filled directly above. */
    }
    goto done;
  }
entropy_ready:
  for (int k = 0; k < K; k++) {
    uint16_t fr[256];
    vcn_lc_norm(cnt[k], fr, tb);
    for(unsigned s=0;s<256;s++,pos+=2)vcn_wr16(hdr+pos,fr[s]);
    uint32_t cum = 0;
    for (unsigned s = 0; s < 256; s++) {
      const uint32_t f = fr[s];
      vcn_lc_esym *E = &es[k][s]; /* reciprocal form (ryg_rans), exact for states below 2^31 */
      vcn_lc_init_esym(E,f,cum,M,tb);
      cum += f;
    }
  }
  if (vcn_lc_mksched(vcn_lc_delta(t), &sc)) {
    st = VCN_ERR_NOMEM;
    goto done;
  }
  {
    size_t nw = 0;
    uint32_t S[VCN_LC_NS];
    for (int k = 0; k < VCN_LC_NS; k++) S[k] = VCN_LC_RL;
    for (int g = sc.ng; g-- > 0;)
      for (int j = sc.n[g]; j-- > 0;) {
        const size_t o = (size_t)sc.row[g][j] * VCN_LC_D;
        const uint8_t *restrict cr = coded_cls + o, *restrict sr = coded_sym + o;
#if VCN_HAVE_AVX2
        if(VCN_LC_D%16==0 && ng_use_avx512() && __builtin_cpu_supports("avx512bw")){
          nw=ng_rans_encode_row_wide(cr,sr,VCN_LC_D,&es[0][0],S,words,nw);
          continue;
        }
#endif
        for (int x = VCN_LC_D; x-- > 0;) {
          const vcn_lc_esym *E = &es[cr[x]][sr[x]];
          S[x & (VCN_LC_NS - 1)] = vcn_lc_put_esym(S[x & (VCN_LC_NS - 1)],E,words,&nw);
        }
      }
    const size_t total = pos + 4 * VCN_LC_NS + 2 * nw;
    if (total > cap) st = VCN_ERR_CAP;
    else {
      memcpy(out, hdr, pos);
      for (int k = 0; k < VCN_LC_NS; k++) vcn_wr32(out + pos + 4 * k, S[k]);
      uint8_t *o = out + pos + 4 * VCN_LC_NS;
      for (size_t j = 0; j < nw; j++) vcn_wr16(o + 2 * j, words[nw - 1 - j]);
      *out_n = total;
    }
  }
done:
  free(sc.row), free(sc.n);
  if(!cf->fit_session)free(b);
  if(!export_sym)free(sym);
  if(!export_cls)free(cls);
  free(hdr), free(words), free(es), free(cnt), free(h), free(sa), free(Pa), free(e), free(p);
  return st;
}
