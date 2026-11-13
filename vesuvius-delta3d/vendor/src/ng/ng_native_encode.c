#include "residual_bridge.h"
#include "ng_kernel_encode.h"
/* Retained encoder entrypoints; shared arithmetic, no duplicate kernels. */
#define NG_DECLARE_FIXED(d) \
int ng_fixed##d##_encode_ex(const uint8_t*,const void*,uint8_t*,size_t,size_t*,uint8_t*,uint8_t*); \
int ng_fixed##d##_activity_values(const uint8_t*,uint16_t*,uint32_t*);
NG_FIXED_SIZES(NG_DECLARE_FIXED)
typedef struct {
    int (*encode)(const uint8_t*,const void*,uint8_t*,size_t,size_t*,uint8_t*,uint8_t*);
    int (*activity_values)(const uint8_t*,uint16_t*,uint32_t*);
} ng_fixed_kernels;
static const ng_fixed_kernels* ng_fixed_current(void) {
    if(NG_X!=NG_Y || NG_X!=NG_Z)return NULL;
    switch(NG_X){
#define NG_CASE(d) case d:{static const ng_fixed_kernels k={ng_fixed##d##_encode_ex,ng_fixed##d##_activity_values};return &k;}
    NG_FIXED_SIZES(NG_CASE)
#undef NG_CASE
    default:return NULL;
    }
}
static vcn_status ng_encode_core(const uint8_t* src,const vcn_lc_cfg* cf,uint8_t* out,size_t cap,size_t* n,uint8_t* sy,uint8_t* cl) {
    const ng_fixed_kernels* fixed=ng_fixed_current();
    return fixed ? fixed->encode(src,cf,out,cap,n,sy,cl) : vcn_lc_encode_ex(src,cf,out,cap,n,sy,cl);
}

static int ng_config(const int* a,vcn_lc_cfg* cf) {
    if (!a || a[0]<1 || a[0]>3 || a[1]<-1 || a[1]>3 || a[2]<-1 || a[2]>3 ||
        a[3]<2 || a[3]>32 || a[4]<2 || a[4]>160 || a[5]<1 || a[5]>128) return -1;
    *cf=(vcn_lc_cfg){a[0],a[1],a[2],a[3],VCN_LC_TB,a[4],a[5]};return 0;
}

void* ng_fit_session_create(void){return calloc(1,sizeof(ng_fit_session));}
void ng_fit_session_destroy(void*ptr){
    ng_fit_session*s=ptr;if(!s)return;
    for(int i=0;i<2;i++)ng_fit_bank_clear(s->banks+i);
    free(s->padded);free(s);
}
/* One preparation path; cached fitting and fused activity remain optional
 * outputs of the same kernel, not separate encoder implementations. */
int ng_prepare(const uint8_t*src,const int*config,const uint8_t*feature,const uint8_t*mask,void*session,int32_t*weights,
    uint8_t*params,size_t cap,size_t*n,uint8_t*sy,uint8_t*cl,uint16_t*predictions,uint16_t*activity,uint32_t*histogram){
    ng_fit_session*s=session;vcn_lc_cfg cf;
    if(ng_config(config,&cf)||!src||(feature&&!weights)||
       (s&&(!feature||mask||activity))||(activity&&(!feature||mask||!histogram)))return -1;
    if(s){
        const int dims[3]={NG_X,NG_Y,NG_Z};
        if(s->source&&(s->source!=src||memcmp(s->config,config,sizeof(s->config))||memcmp(s->dims,dims,sizeof(dims))))return -1;
        s->source=src;memcpy(s->config,config,sizeof(s->config));memcpy(s->dims,dims,sizeof(dims));
    }
    cf.fit_feature=feature;cf.fit_mask=mask;cf.export_feature_weights=feature?weights:NULL;
    cf.export_prediction=predictions;cf.fit_session=s;
    cf.export_activity=activity;cf.export_activity_histogram=histogram;
    return ng_encode_core(src,&cf,params,cap,n,sy,cl);
}

double ng_hist_cost_logs(const uint32_t* counts,int tb,const double* logs) {
    uint16_t freq[256];vcn_lc_norm(counts,freq,tb);
    unsigned nn=256;while(nn>1 && !freq[nn-1])nn--;
    double bits=16;
    for(unsigned s=0;s<256;s++) {
        if(s<nn){unsigned f=freq[s];do{bits+=8;f>>=7;}while(f);}
        if(counts[s])bits+=(double)counts[s]*(tb-logs[freq[s]]);
    }
    return bits;
}

int ng_wave_encode(const uint8_t* params,size_t n,const uint8_t* sy,const uint8_t* cl,
    uint8_t* out,size_t cap,size_t* size) {
    const uint8_t* cursor=params;vcn_lc_par* p=NULL;
    if(vcn_lc_readpar(&cursor,n,&p)!=VCN_OK)return -1;
    if(cursor!=params+n){free(p);return -1;}
    vcn_lc_cfg cf={0};cf.r0=p->t.r[0];cf.r1=p->t.r[1];cf.r2=p->t.r[2];cf.K=p->K;cf.tb=p->tb;
    cf.import_sym=sy;cf.import_cls=cl;cf.import_params=params;cf.import_params_n=n;
    int rc=ng_encode_core(NULL,&cf,out,cap,size,NULL,NULL);free(p);return rc;
}


/* Multiple independently fitted models, all driven by the same actual selected
 * residual history. Fixed candidates precede the model candidates. Reuses the
 * native class and symbol kernels, including fractional Q8 sign mapping. */
#include "ng_bank_symbols.h"
#define NG_BANK_ARGS unsigned c,const uint8_t*const*p,const size_t*n,const uint16_t*const*b,const uint8_t*s,const uint8_t*x,uint8_t*const*v,uint8_t*y,uint16_t*a
#define NG_DECLARE_BANK(d) int ng_fixed##d##_bank_symbols(NG_BANK_ARGS,int);
NG_FIXED_SIZES(NG_DECLARE_BANK)
#undef NG_DECLARE_BANK
static int ng_bank_dispatch(NG_BANK_ARGS,int selected){
    if(NG_X==NG_Y&&NG_X==NG_Z){
        switch(NG_X){
#define NG_CASE(d) case d:return ng_fixed##d##_bank_symbols(c,p,n,b,s,x,v,y,a,selected);
        NG_FIXED_SIZES(NG_CASE)
#undef NG_CASE
        }
    }
    return ng_bank_symbols_impl(c,p,n,b,s,x,v,y,a,selected);
}
int ng_bank_symbols(NG_BANK_ARGS){return ng_bank_dispatch(c,p,n,b,s,x,v,y,a,0);}
int ng_bank_selected_symbols(NG_BANK_ARGS){return ng_bank_dispatch(c,p,n,b,s,x,v,y,a,1);}
#undef NG_BANK_ARGS

/* Contexts for an already-filtered byte stream: same causal activity stencil
   as NG, but quantiles fitted on this stream rather than the pre-filter one. */
#include "ng_activity_values.h"
static int ng_activity_dispatch(const uint8_t*s,uint16_t*a,uint32_t*h){
    const ng_fixed_kernels*f=ng_fixed_current();
    return f?f->activity_values(s,a,h):ng_activity_values_impl(s,a,h);
}

int ng_activity_symbol_values(const uint8_t* symbols,uint16_t* activity) {return ng_activity_dispatch(symbols,activity,NULL);}
int ng_activity_symbol_histogram(const uint8_t* symbols,uint16_t* activity,uint32_t* histogram) {
    if(!symbols||!activity||!histogram)return -1;
    return ng_activity_dispatch(symbols,activity,histogram);
}
