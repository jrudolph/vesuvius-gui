#include "ng_search.h"
#include "ng_screen.h"
#include "ng_thresholds.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

int vc_ng_scan(vc_ng_occupancy *o,const uint8_t *src,unsigned d)
{
    o->blocks=(size_t)(d/4)*(d/4)*(d/4);
    o->empty=malloc(o->blocks);
    if(!o->empty)return 0;
    o->count=vc_ng_zero_scan(src,d,o->empty);
    if(o->count){
        o->mask=malloc((size_t)d*d*d);
        if(!o->mask)return 0;
        vc_ng_zero_mask(d,o->empty,o->mask);
    }
    return 1;
}

void vc_ng_top_two(const double rank[12],unsigned top[2])
{
    top[0]=0;
    for(unsigned m=1;m<12;++m)if(rank[m]<rank[top[0]])top[0]=m;
    top[1]=top[0]?0:1;
    for(unsigned m=0;m<12;++m)if(m!=top[0]&&rank[m]<rank[top[1]])top[1]=m;
}

int vc_ng_search_thresholds(const uint8_t *symbols,const uint16_t *activity,size_t n,
    const uint8_t *mask,const uint32_t *blocks,const uint8_t *choices,size_t choice_count,
    unsigned groups,unsigned tune,const uint32_t *ready_histogram,uint16_t out[15])
{
    uint8_t *ss=NULL;uint16_t *aa=NULL;uint32_t *bb=NULL;
    uint32_t histogram[2560]={0};
    int ok=0;
    if(mask){
        ss=malloc(n);aa=malloc(n*sizeof(*aa));
        if(groups>1)bb=malloc(n*sizeof(*bb));
        if(!ss||!aa||(groups>1&&!bb))goto done;
        size_t count=0;
        for(size_t i=0;i<n;++i)if(mask[i]){
            if(activity[i]>=2560)goto done;
            ss[count]=symbols[i];aa[count]=activity[i];
            ++histogram[activity[i]];
            if(groups>1)bb[count]=blocks[i];
            ++count;
        }
        symbols=ss;activity=aa;blocks=bb;n=count;
        ready_histogram=histogram;
    }else if(!ready_histogram){
        for(size_t i=0;i<n;++i){if(activity[i]>=2560)goto done;++histogram[activity[i]];}
        ready_histogram=histogram;
    }
    if(!vc_ng_quantiles(ready_histogram,16,out))goto done;
    if(tune){
        uint16_t refined[15];
        if(!vc_ng_refine_thresholds(symbols,activity,n,out,16,blocks,choices,choice_count,
            groups,tune==2,refined))goto done;
        memcpy(out,refined,sizeof(refined));
    }
    ok=1;
done:
    free(ss);free(aa);free(bb);return ok;
}

void vc_ng_distribution_finish(vc_ng_distribution *s)
{
    for(unsigned c=0;c<16;++c){
        double total=128;
        for(unsigned j=0;j<256;++j)total+=s->histogram[c][j];
        for(unsigned j=0;j<256;++j)s->cost[c][j]=-log2((s->histogram[c][j]+.5)/total);
    }
}

int vc_ng_distribution_fit(vc_ng_distribution *s,const uint8_t *symbols,
    const uint16_t *activity,size_t n,const uint8_t *mask,unsigned tune)
{
    uint8_t *classes=malloc(n);
    int ok=0;
    if(!classes) return 0;
    if(!vc_ng_search_thresholds(symbols,activity,n,mask,NULL,NULL,0,1,tune,NULL,s->threshold)||
       !vc_ng_classify(activity,n,s->threshold,16,classes))goto done;
    memset(s->histogram,0,sizeof(s->histogram));
    for(size_t i=0;i<n;++i)if(!mask||mask[i])++s->histogram[classes[i]][symbols[i]];
    vc_ng_distribution_finish(s);ok=1;
done:
    free(classes);return ok;
}

double vc_ng_distribution_score(const vc_ng_distribution *s)
{
    double cost=0;
    for(unsigned c=0;c<16;++c)cost+=vc_ng_histogram_cost(s->histogram[c],12);
    return cost/8.;
}
