/* Medium: direct, plain NG, and one independently fitted screened predictor.
 * Their entropy distributions remain separate. One observed-distribution
 * reselection and the screened singleton trial reproduce the retained preset. */
#include "ng_fit.h"
#include "ng_frame.h"
#include "ng_predict.h"
#include "ng_screen.h"
#include "ng_search.h"
#include "ng_thresholds.h"
#include "residual_bridge.h"
#include <stdlib.h>
#include <string.h>
#include <float.h>

typedef struct {
    unsigned d, third, assisted, ids[3];
    size_t n, blocks;
    const uint8_t *src;
    vc_ng_occupancy zero;
    vc_ng_fit fits[2];
    uint8_t *candidate[36], *choices, *canonical, *symbols, *classes;
    uint16_t *activity;
    uint32_t *block;
    vc_ng_distribution stats[3];
} medium_state;

static int evaluate(medium_state *s,int selected_only)
{
    const uint8_t *params[2];size_t sizes[2];const uint16_t *bases[2];
    uint8_t *outputs[36]={0};
    for(unsigned k=0;k<2;++k){
        const vc_ng_fit *f=&s->fits[k&&s->assisted?1:0];
        params[k]=f->params;sizes[k]=f->param_size;bases[k]=f->base;
    }
    for(unsigned k=0;k<3;++k)outputs[s->ids[k]]=s->candidate[s->ids[k]];
    for(size_t b=0;b<s->blocks;++b)
        s->canonical[b]=s->zero.count&&s->zero.empty[b]?0:(uint8_t)s->ids[s->choices[b]];
    return (selected_only?ng_bank_selected_symbols:ng_bank_symbols)(2,params,sizes,bases,
        s->canonical,s->src,outputs,s->symbols,s->activity)==0;
}

static int select_blocks(medium_state *s)
{
    double *cost=calloc(s->blocks*3,sizeof(*cost));
    if(!cost)return 0;
    for(unsigned k=0;k<3;++k){
        if(!vc_ng_classify(s->activity,s->n,s->stats[k].threshold,16,s->classes)){free(cost);return 0;}
        for(size_t i=0;i<s->n;++i)if(!s->zero.mask||s->zero.mask[i])
            cost[3*s->block[i]+k]+=s->stats[k].cost[s->classes[i]][s->candidate[s->ids[k]][i]];
    }
    for(size_t b=0;b<s->blocks;++b){
        unsigned best=0;
        for(unsigned k=1;k<3;++k)if(cost[3*b+k]<cost[3*b+best])best=k;
        s->choices[b]=(uint8_t)(s->zero.count&&s->zero.empty[b]?1:best);
    }
    free(cost);return 1;
}

static int thresholds(medium_state *s,unsigned groups,unsigned tune,uint16_t out[15])
{
    return vc_ng_search_thresholds(s->symbols,s->activity,s->n,s->zero.mask,s->block,
        s->choices,s->blocks,groups,tune,NULL,out);
}

static int refresh(medium_state *s)
{
    uint16_t t[15];
    if(!thresholds(s,3,1,t)||!vc_ng_classify(s->activity,s->n,t,16,s->classes))return 0;
    for(unsigned k=0;k<3;++k){
        memcpy(s->stats[k].threshold,t,sizeof(t));
        memset(s->stats[k].histogram,0,sizeof(s->stats[k].histogram));
    }
    for(size_t i=0;i<s->n;++i)if(!s->zero.mask||s->zero.mask[i])
        ++s->stats[s->choices[s->block[i]]].histogram[s->classes[i]][s->symbols[i]];
    for(unsigned k=0;k<3;++k)vc_ng_distribution_finish(&s->stats[k]);
    return 1;
}

static int screen(medium_state *s,unsigned *singleton)
{
    const ng_geometry saved=ng_current_geometry;
    const unsigned d=32;const size_t n=32*32*32;
    const int cfg[6]={3,3,1,16,40,4};
    uint8_t *proxy=malloc(n),*fixed[12]={0};
    uint16_t *activity=malloc(n*sizeof(*activity));
    vc_ng_occupancy zeros={0};
    vc_ng_fit plain={0},fitted={0};
    vc_ng_distribution stats;
    double rank[12],best=DBL_MAX;unsigned top[2]={0,0};int ok=0;
    if(!proxy||!activity||vc_ng_screen_proxy(s->src,s->d,s->zero.count?s->zero.empty:NULL,proxy))goto done;
    ng_current_geometry=(ng_geometry){d,d,d};
    if(s->zero.count&&!vc_ng_scan(&zeros,proxy,d))goto done;
    for(unsigned m=0;m<12;++m){fixed[m]=malloc(n);if(!fixed[m])goto done;}
    vc_ng_fixed_candidates(proxy,fixed,d,4095);
    for(unsigned m=0;m<12;++m){
        if(ng_activity_symbol_values(fixed[m],activity)||
            !vc_ng_distribution_fit(&stats,fixed[m],activity,n,zeros.mask,0))goto done;
        rank[m]=vc_ng_distribution_score(&stats);
    }
    vc_ng_top_two(rank,top);
    if(!vc_ng_fit_input(&plain,proxy,cfg,NULL,zeros.mask,NULL,NULL,NULL)||
       ng_activity_symbol_values(plain.symbols,activity)||
       !vc_ng_distribution_fit(&s->stats[0],plain.symbols,activity,n,zeros.mask,0)||
       ng_activity_symbol_values(fixed[0],activity)||
       !vc_ng_distribution_fit(&s->stats[1],fixed[0],activity,n,zeros.mask,0))goto done;
    for(unsigned k=0;k<2;++k){
        unsigned m=top[k];if(!m)continue;
        if(ng_activity_symbol_values(fixed[m],activity)||
            !vc_ng_distribution_fit(&stats,fixed[m],activity,n,zeros.mask,0))goto done;
        double cost=vc_ng_distribution_score(&stats)*s->n/n;
        if(cost<best){best=cost;s->third=m;s->assisted=0;s->stats[2]=stats;}
        if(!vc_ng_fit_input(&fitted,proxy,cfg,fixed[m],zeros.mask,NULL,NULL,NULL)||
           ng_activity_symbol_values(fitted.symbols,activity)||
           !vc_ng_distribution_fit(&stats,fitted.symbols,activity,n,zeros.mask,0))goto done;
        cost=vc_ng_distribution_score(&stats)*s->n/n+fitted.param_size+8;
        if(cost<best){best=cost;s->third=m;s->assisted=1;s->stats[2]=stats;}
        vc_ng_fit_free(&fitted);
    }
    double costs[3]={vc_ng_distribution_score(&s->stats[0])*s->n/n+plain.param_size,
                    vc_ng_distribution_score(&s->stats[1])*s->n/n,best};
    *singleton=0;
    for(unsigned k=1;k<3;++k)if(costs[k]<costs[*singleton])*singleton=k;
    ok=1;
done:
    vc_ng_fit_free(&plain);vc_ng_fit_free(&fitted);
    for(unsigned m=0;m<12;++m)free(fixed[m]);
    free(proxy);free(activity);free(zeros.empty);free(zeros.mask);
    ng_current_geometry=saved;return ok;
}

int vc_ng_medium_encode(const uint8_t *src,unsigned d,uint8_t *out,size_t cap,size_t *written)
{
    if(!src||!out||!written||!ng_geometry_valid(d,d,d))return -1;
    const ng_geometry saved=ng_current_geometry;
    const unsigned old_edge=ng_choice_edge;
    ng_current_geometry=(ng_geometry){(int)d,(int)d,(int)d};ng_choice_edge=4;
    medium_state *s=calloc(1,sizeof(*s));
    uint8_t *frame=NULL,*trial=NULL;size_t frame_size=0,trial_size=0;
    uint16_t t[15];unsigned singleton=0;int rc=-1;
    if(!s)goto done;
    s->src=src;s->d=d;s->n=(size_t)d*d*d;s->blocks=s->n/64;
    if(!vc_ng_scan(&s->zero,src,d))goto done;
    if(s->zero.count==s->blocks){
        if(!vc_ng_frame_all_zero(d,&frame,&frame_size))goto done;
        goto finish;
    }
    s->choices=calloc(s->blocks,1);
    if(!s->choices)goto done;
    if(!screen(s,&singleton))goto done;
    s->candidate[0]=malloc(s->n);s->candidate[s->third]=malloc(s->n);
    if(!s->candidate[0]||!s->candidate[s->third])goto done;
    vc_ng_fixed_candidates(src,s->candidate,d,1u|(1u<<s->third));
    const int cfg[6]={3,3,1,16,40,128};
    if(!vc_ng_fit_input(&s->fits[0],src,cfg,NULL,s->zero.mask,NULL,NULL,NULL))goto done;
    if(s->assisted&&!vc_ng_fit_input(&s->fits[1],src,cfg,s->candidate[s->third],s->zero.mask,NULL,NULL,NULL))goto done;
    s->ids[0]=12;s->ids[1]=0;s->ids[2]=s->assisted?24+s->third:s->third;
    s->candidate[12]=malloc(s->n);
    if(s->assisted)s->candidate[24+s->third]=malloc(s->n);
    s->canonical=malloc(s->blocks);s->symbols=malloc(s->n);s->classes=malloc(s->n);
    s->activity=malloc(s->n*sizeof(*s->activity));s->block=malloc(s->n*sizeof(*s->block));
    if(!s->candidate[12]||(s->assisted&&!s->candidate[24+s->third])||!s->canonical||
        !s->symbols||!s->classes||!s->activity||!s->block)goto done;
    for(unsigned z=0;z<d;++z)for(unsigned y=0;y<d;++y)for(unsigned x=0;x<d;++x)
        s->block[((size_t)z*d+y)*d+x]=((z/4)*(d/4)+y/4)*(d/4)+x/4;
    if(!evaluate(s,0))goto done;
    for(unsigned k=0;k<3;++k)
        if(!vc_ng_distribution_fit(&s->stats[k],s->candidate[s->ids[k]],s->activity,s->n,s->zero.mask,1))goto done;
    if(!select_blocks(s)||!evaluate(s,1)||!evaluate(s,0)||!refresh(s)||!select_blocks(s)||!evaluate(s,1))goto done;
    if(!thresholds(s,3,1,t)||!vc_ng_classify(s->activity,s->n,t,16,s->classes))goto done;
    vc_ng_model models[2]={{s->fits[0].params,s->fits[0].param_size},{s->fits[1].params,s->fits[1].param_size}};
    vc_ng_choice menu[3]={{0,0,0,{0,0}},{0,255,1,{0,0}},{s->third,s->assisted?1u:255u,2,{0,0}}};
    if(s->assisted)memcpy(menu[2].weights,s->fits[1].weights,sizeof(menu[2].weights));
    if(!vc_ng_frame_encode_zeros(d,models,s->assisted?2:1,menu,3,s->choices,s->symbols,t,
        s->classes,s->zero.count?s->zero.empty:NULL,&frame,&frame_size))goto done;
    /* The screen chooses one singleton trial; no additional full menu search. */
    memset(s->choices,singleton,s->blocks);
    if(!evaluate(s,1)||!thresholds(s,1,2,t)||!vc_ng_classify(s->activity,s->n,t,16,s->classes))goto done;
    vc_ng_choice single={singleton==2?s->third:0,255,0,{0,0}};
    unsigned nm=0;
    if(singleton==0||(singleton==2&&s->assisted)){
        const vc_ng_fit *f=&s->fits[singleton==0?0:1];
        models[0]=(vc_ng_model){f->params,f->param_size};single.model=0;nm=1;
        if(singleton==2)memcpy(single.weights,f->weights,sizeof(single.weights));
    }
    memset(s->choices,0,s->blocks);
    if(!vc_ng_frame_encode_zeros(d,models,nm,&single,1,s->choices,s->symbols,t,s->classes,
        s->zero.count?s->zero.empty:NULL,&trial,&trial_size))goto done;
    if(trial_size<frame_size){free(frame);frame=trial;frame_size=trial_size;trial=NULL;}
finish:
    if(frame_size>cap)goto done;
    memcpy(out,frame,frame_size);*written=frame_size;rc=0;
done:
    if(s){
        for(unsigned k=0;k<2;++k)vc_ng_fit_free(&s->fits[k]);
        for(unsigned m=0;m<36;++m)free(s->candidate[m]);
        free(s->zero.empty);free(s->zero.mask);free(s->choices);free(s->canonical);
        free(s->symbols);free(s->classes);free(s->activity);free(s->block);free(s);
    }
    free(frame);free(trial);ng_current_geometry=saved;ng_choice_edge=old_edge;return rc;
}
