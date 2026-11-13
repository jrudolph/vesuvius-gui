/* Public preset dispatch. Save all scoped fitter settings for nested callers. */
#include "residual_bridge.h"
#include "ng_frame.h"
#include <vc_delta3d/codec.h>
#include <stdlib.h>
#include <string.h>
/* reach, taps, stride, intercept, models, rounds: see vc_ng_tuning. */
static const vc_ng_tuning faster={1,40,128,0.,1,0};    /* the fast preset of 0.3.x */
static const vc_ng_tuning fast={2,56,128,0.,1,0};
static const vc_ng_tuning balanced={3,72,32,0.,1,0};
static const vc_ng_tuning small={2,72,32,0.,2,1};
int vc_ng_tuned_encode(const uint8_t *s,unsigned d,const vc_ng_tuning *t,uint8_t *out,size_t cap,size_t *n)
{
    if(!t||t->reach<1||t->reach>3||t->taps<1||t->taps>160||t->stride<1||t->stride>128||
       !t->models||t->models>4||(t->models>1&&!t->rounds))return -1;
    double saved=ng_fit_intercept_weight;
    ng_fit_intercept_weight=t->intercept;
    /* Several models only replace the one-option frame when they are smaller:
     * data that a fixed predictor codes best (sparse or piecewise constant
     * volumes) and chunks with too few occupied blocks keep the one option. */
    int rc=vc_ng_fast_encode(s,d,t,out,cap,n);
    if(!rc&&t->models>1&&*n>1){
        size_t m=0;uint8_t *alt=malloc(*n-1);
        if(alt&&!vc_ng_split_encode(s,d,t,alt,*n-1,&m)){memcpy(out,alt,m);*n=m;}
        free(alt);
    }
    ng_fit_intercept_weight=saved;return rc;
}
int vc_ng_preset_encode(const uint8_t *s,unsigned d,unsigned preset,uint8_t *out,size_t cap,size_t *n)
{
    const vc_ng_tuning *t=preset==VC_FASTER?&faster:preset==VC_FAST?&fast:preset==VC_BALANCED?&balanced:
        preset==VC_SMALL?&small:NULL;
    if(t)return vc_ng_tuned_encode(s,d,t,out,cap,n);
    if(preset!=VC_MEDIUM&&preset!=VC_SLOW&&preset!=VC_SLOWER)return -1;
    double saved=ng_fit_intercept_weight;
    ng_fit_intercept_weight=preset==VC_MEDIUM?0.:100.;
    int rc=preset==VC_MEDIUM?vc_ng_medium_encode(s,d,out,cap,n):vc_ng_slow_encode(s,d,preset==VC_SLOWER,out,cap,n);
    ng_fit_intercept_weight=saved;return rc;
}
