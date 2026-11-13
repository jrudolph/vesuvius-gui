/* Preserve candidate order, strict tie comparisons and floating-point sum order
 * of the original threshold search. No approximate logarithms or reassociation. */
#include "ng_thresholds.h"
#include "residual_bridge.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

double vc_ng_histogram_cost(const uint32_t *counts,int bits) {
    static _Thread_local double logs[4097];
    static _Thread_local int ready;
    if(!ready) { for(unsigned i=1;i<=4096;++i)logs[i]=log2((double)i);ready=1; }
    return ng_hist_cost_logs(counts,bits,logs);
}
int vc_ng_quantiles(const uint32_t hist[2560],unsigned K,uint16_t *out) {
    if(!hist || !out || K<2 || K>2560)return 0;
    uint64_t total=0,sum=0;for(unsigned v=0;v<2560;++v)total+=hist[v];
    if(!total)return 0;
    unsigned count=0;
    for(unsigned v=0;v<2560&&count+1<K;++v) {
        sum+=hist[v];while(count+1<K&&sum*K>=(count+1)*total)out[count++]=(uint16_t)v;
    }
    return count==K-1;
}
static unsigned lower(const uint16_t *values,unsigned count,unsigned v) {
    unsigned first=0;
    while(first<count) {unsigned mid=first+(count-first)/2;if(values[mid]<v)first=mid+1;else count=mid;}
    return first;
}
int vc_ng_classify(const uint16_t *a,size_t n,const uint16_t *thr,unsigned K,uint8_t *out) {
    if(!a||!thr||!out||K<2||K>256)return 0;
    uint8_t lut[2560];for(unsigned v=0;v<2560;++v)lut[v]=(uint8_t)lower(thr,K-1,v);
    for(size_t i=0;i<n;++i) {if(a[i]>=2560)return 0;out[i]=lut[a[i]];}return 1;
}
typedef struct {uint32_t *prefix;double *cache;unsigned edges,groups;} costs;
static double interval(costs *c,unsigned l,unsigned r) {
    double *result=&c->cache[(size_t)l*(c->edges+1)+r];
    if(*result>=0)return *result;
    *result=0;
    for(unsigned g=0;g<c->groups;++g) {
        uint32_t hist[256];
        const uint32_t *a=c->prefix+((size_t)l*c->groups+g)*256,*b=c->prefix+((size_t)r*c->groups+g)*256;
        for(unsigned s=0;s<256;++s)hist[s]=b[s]-a[s];
        *result+=vc_ng_histogram_cost(hist,12);
    }
    return *result;
}
static int compare(const void *a,const void *b) {return (int)*(const uint16_t*)a-(int)*(const uint16_t*)b;}
int vc_ng_refine_thresholds(const uint8_t *raw,const uint16_t *a,size_t n,const uint16_t *old,unsigned K,
    const uint32_t *block,const uint8_t *choices,size_t choice_count,unsigned groups,
    int global,uint16_t *out) {
    if(!raw||!a||!old||!out||K<2||K>256||!groups||groups>256||(groups>1&&(!block||!choices)))return 0;
    uint16_t *edges=NULL;uint32_t *prefix=NULL;double *cache=NULL,*dp=NULL;unsigned *prev=NULL;
    int ok=0;unsigned E=0;
    const unsigned grid=64;
    size_t capacity=grid+K;
    edges=malloc(capacity*sizeof(*edges));if(!edges)goto done;
    {
        uint32_t hist[2560]={0};for(size_t i=0;i<n;++i) {if(a[i]>=2560)goto done;++hist[a[i]];}
        if(!vc_ng_quantiles(hist,grid,edges))goto done;E=grid-1;
    }
    for(unsigned k=0;k<K-1;++k) {if(old[k]>2559)goto done;edges[E++]=old[k];}
    edges[E++]=2559;qsort(edges,E,sizeof(*edges),compare);
    unsigned unique=0;for(unsigned i=0;i<E;++i)if(!i||edges[i]!=edges[i-1])edges[unique++]=edges[i];E=unique;
    size_t bins=(size_t)E+1,pc=bins*groups*256,cc=bins*bins;
    prefix=calloc(pc,sizeof(*prefix));cache=malloc(cc*sizeof(*cache));if(!prefix||!cache)goto done;
    for(size_t i=0;i<cc;++i)cache[i]=-1;
    unsigned bin[2560];for(unsigned v=0;v<2560;++v)bin[v]=lower(edges,E,v);
    for(size_t i=0;i<n;++i) {
        if(a[i]>=2560)goto done;
        unsigned g=0;if(groups>1) {if(block[i]>=choice_count)goto done;g=choices[block[i]];if(g>=groups)goto done;}
        unsigned value=raw[i];
        ++prefix[((size_t)(bin[a[i]]+1)*groups+g)*256+value];
    }
    for(unsigned e=1;e<=E;++e)for(unsigned g=0;g<groups;++g)for(unsigned s=0;s<256;++s)
        prefix[((size_t)e*groups+g)*256+s]+=prefix[((size_t)(e-1)*groups+g)*256+s];
    costs cost={prefix,cache,E,groups};
    if(global) {
        size_t cells=(K+1)*bins;dp=malloc(cells*sizeof(*dp));prev=calloc(cells,sizeof(*prev));if(!dp||!prev)goto done;
        for(size_t i=0;i<cells;++i)dp[i]=INFINITY;dp[0]=0;
        for(unsigned k=1;k<=K;++k)for(unsigned b=1;b<=E;++b)for(unsigned l=k==1?0:1;l<=b;++l) {
            double before=dp[(k-1)*bins+l];if(!isfinite(before))continue;
            double score=before+interval(&cost,l,b);
            if(score<dp[k*bins+b]) {dp[k*bins+b]=score;prev[k*bins+b]=l;}
        }
        unsigned b=E;
        for(unsigned k=K;k>1;--k) {b=prev[k*bins+b];if(!b)goto done;out[k-2]=edges[b-1];}
    } else {
        unsigned cuts[257];cuts[0]=0;cuts[K]=E;
        for(unsigned k=0;k<K-1;++k)cuts[k+1]=lower(edges,E,old[k])+1;
        for(unsigned pass=0;pass<2;++pass)for(unsigned k=1;k<K;++k) {
            unsigned best=cuts[k];double score=interval(&cost,cuts[k-1],best)+interval(&cost,best,cuts[k+1]);
            for(unsigned v=cuts[k-1]>1?cuts[k-1]:1;v<=cuts[k+1];++v) {
                double s=interval(&cost,cuts[k-1],v)+interval(&cost,v,cuts[k+1]);if(s<score) {score=s;best=v;}
            }
            cuts[k]=best;
        }
        for(unsigned k=1;k<K;++k)out[k-1]=edges[cuts[k]-1];
    }
    ok=1;
done:
    free(prev);free(dp);free(cache);free(prefix);free(edges);return ok;
}
