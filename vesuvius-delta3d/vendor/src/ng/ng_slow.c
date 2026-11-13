/* Slow/slower share the same beam menu search; only the NG tap configuration
 * differs. Fit finalists once at full resolution and retain the smallest of
 * the one-, two- and three-choice frames. No streaming/adaptive experiments. */
#include "buffer.h"
#include "ng_fit.h"
#include "ng_frame.h"
#include "ng_predict.h"
#include "ng_screen.h"
#include "ng_search.h"
#include "ng_select.h"
#include "ng_thresholds.h"
#include "residual_bridge.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

enum { CANDIDATES=25, MODELS=13, SAMPLE_N=32768, SAMPLE_BLOCKS=512 };
typedef struct {
    double cost;
    unsigned menu[3],count,seed;
    uint8_t choices[SAMPLE_BLOCKS];
} plan;
typedef struct {
    unsigned predictor[CANDIDATES];
    vc_ng_fit fit[MODELS];
    uint8_t *symbols[CANDIDATES],*classes[CANDIDATES];
    vc_ng_distribution stats[CANDIDATES];
    double uniform[CANDIDATES];
    uint32_t *block;
    uint8_t *mask;
    unsigned scale;
} search;

static void plus_one_cost(vc_ng_distribution *s)
{
    for(unsigned c=0;c<16;++c){
        double total=256;
        for(unsigned j=0;j<256;++j)total+=s->histogram[c][j];
        for(unsigned j=0;j<256;++j)s->cost[c][j]=log2(total)-log2(s->histogram[c][j]+1.);
    }
}

static double model_cost(const search *s,const plan *p)
{
    vc_ng_model models[3];vc_ng_choice menu[3];unsigned nm=0,remap[MODELS];
    for(unsigned k=0;k<MODELS;k++)remap[k]=255;
    for(unsigned j=0;j<p->count;j++){
        unsigned id=p->menu[j];menu[j]=(vc_ng_choice){s->predictor[id],255,j,{0,0}};
        if(id>=12){unsigned k=id-12;
            if(remap[k]==255){remap[k]=nm;models[nm++]=(vc_ng_model){s->fit[k].params,s->fit[k].param_size};}
            menu[j].model=remap[k];memcpy(menu[j].weights,s->fit[k].weights,sizeof(menu[j].weights));
        }
    }
    return vc_ng_model_cost(models,nm,menu,p->count);
}

static double estimate(const search *s,const plan *p)
{
    uint32_t hist[3][16][256]={{{0}}},selectors[256]={0};
    const uint8_t *classes=s->classes[p->seed];
    for(size_t i=0;i<SAMPLE_N;++i)if(!s->mask||s->mask[i]){
        unsigned k=p->choices[s->block[i]];
        hist[k][classes[i]][s->symbols[p->menu[k]][i]]+=s->scale;
    }
    for(size_t b=0;b<SAMPLE_BLOCKS;++b)selectors[p->choices[b]]+=s->scale;
    double cost=vc_ng_histogram_cost(selectors,12);
    for(unsigned k=0;k<p->count;++k)for(unsigned c=0;c<16;++c)
        cost+=vc_ng_histogram_cost(hist[k][c],12);
    return cost/8.;
}

static int score_plan(const search *s,plan *p)
{
    unsigned seed=0;
    for(unsigned k=1;k<p->count;++k)if(s->uniform[p->menu[k]]<s->uniform[p->menu[seed]])seed=k;
    p->seed=p->menu[seed];memset(p->choices,seed,SAMPLE_BLOCKS);
    if(p->count>1){
        const uint8_t *symbols[3],*classes[3];const double *costs[3];
        for(unsigned k=0;k<p->count;++k){
            unsigned id=p->menu[k];symbols[k]=s->symbols[id];classes[k]=s->classes[id];costs[k]=&s->stats[id].cost[0][0];
        }
        /* The second selector pass uses the first pass's selector prior. */
        for(unsigned pass=0;pass<2;++pass)
            if(!vc_ng_select(symbols,classes,costs,p->count,s->block,SAMPLE_N,p->choices,SAMPLE_BLOCKS))return 0;
    }
    double metadata=model_cost(s,p);if(metadata<0)return 0;
    p->cost=estimate(s,p)+metadata;return 1;
}

static int rank_plan(const void *aa,const void *bb)
{
    const plan *a=aa,*b=bb;
    if(a->cost!=b->cost)return a->cost<b->cost?-1:1;
    unsigned n=a->count<b->count?a->count:b->count;
    for(unsigned k=0;k<n;++k)if(a->menu[k]!=b->menu[k])return a->menu[k]<b->menu[k]?-1:1;
    return a->count<b->count?-1:a->count>b->count;
}

static int screen(search *s,const uint8_t *src,unsigned d,const vc_ng_occupancy *zeros,
                  const int *config,plan plans[3])
{
    ng_geometry old=ng_current_geometry;
    uint8_t *proxy=malloc(SAMPLE_N),*fixed[12]={0};uint16_t *activity=malloc(SAMPLE_N*sizeof(*activity));
    vc_ng_occupancy occupancy={0};void *session=NULL;int ok=0;
    unsigned top[MODELS]={0},top_count=2;
    double rank[12];vc_ng_distribution temporary;
    plan *beam=calloc(200,sizeof(*beam)),*next=calloc(200,sizeof(*next));
    s->block=malloc(SAMPLE_N*sizeof(*s->block));
    if(!proxy||!activity||!beam||!next||!s->block)goto done;
    if(vc_ng_screen_proxy(src,d,zeros->count?zeros->empty:NULL,proxy))goto done;
    ng_current_geometry=(ng_geometry){32,32,32};
    if(zeros->count&&!vc_ng_scan(&occupancy,proxy,32))goto done;
    for(unsigned m=0;m<12;++m){fixed[m]=malloc(SAMPLE_N);if(!fixed[m])goto done;s->predictor[m]=m;}
    vc_ng_fixed_candidates(proxy,fixed,32,4095);
    for(unsigned m=0;m<12;++m){
        if(ng_activity_symbol_values(fixed[m],activity)||
            !vc_ng_distribution_fit(&temporary,fixed[m],activity,SAMPLE_N,occupancy.mask,0))goto done;
        rank[m]=vc_ng_distribution_score(&temporary);
    }
    vc_ng_top_two(rank,top);
    for(unsigned m=0;m<12;++m)if(m!=top[0]&&m!=top[1])top[top_count++]=m;
    top[top_count++]=0;
    int cfg[6];memcpy(cfg,config,sizeof(cfg));if(d>32)cfg[5]=4;
    session=ng_fit_session_create();if(!session)goto done;
    for(unsigned k=0;k<MODELS;++k){
        s->predictor[12+k]=top[k];
        if(!vc_ng_fit_input(&s->fit[k],proxy,cfg,k==12?NULL:fixed[top[k]],occupancy.mask,
            k==12||occupancy.count?NULL:session,NULL,NULL))goto done;
    }
    ng_fit_session_destroy(session);session=NULL;
    s->scale=(unsigned)((size_t)d*d*d/SAMPLE_N);
    if(occupancy.count){s->mask=malloc(SAMPLE_N);if(!s->mask)goto done;}
    /* Preserve the original octant sample order, including selector IDs. */
    for(unsigned id=0;id<CANDIDATES;++id){
        const uint8_t *raw=id<12?fixed[id]:s->fit[id-12].symbols;
        s->symbols[id]=malloc(SAMPLE_N);s->classes[id]=malloc(SAMPLE_N);
        uint8_t *classes=malloc(SAMPLE_N);
        if(!s->symbols[id]||!s->classes[id]||!classes){free(classes);goto done;}
        if(ng_activity_symbol_values(raw,activity)||
           !vc_ng_distribution_fit(&s->stats[id],raw,activity,SAMPLE_N,occupancy.mask,2)||
           !vc_ng_classify(activity,SAMPLE_N,s->stats[id].threshold,16,classes)){free(classes);goto done;}
        plus_one_cost(&s->stats[id]);
        size_t i=0;
        for(unsigned oz=0;oz<2;++oz)for(unsigned oy=0;oy<2;++oy)for(unsigned ox=0;ox<2;++ox)
            for(unsigned z=0;z<16;++z)for(unsigned y=0;y<16;++y)for(unsigned x=0;x<16;++x,++i){
                size_t q=((size_t)(16*oz+z)*32+16*oy+y)*32+16*ox+x;
                s->symbols[id][i]=raw[q];s->classes[id][i]=classes[q];
                if(id==0){s->block[i]=((oz*2+oy)*2+ox)*64+(z/4*4+y/4)*4+x/4;if(s->mask)s->mask[i]=occupancy.mask[q];}
            }
        free(classes);
        plan p={.menu={id},.count=1,.seed=id};
        s->uniform[id]=estimate(s,&p);
        if(!score_plan(s,&p))goto done;
        beam[id]=p;
    }
    qsort(beam,CANDIDATES,sizeof(*beam),rank_plan);plans[0]=beam[0];unsigned beam_count=8;
    for(unsigned depth=2;depth<=3;++depth){
        uint32_t seen[200];unsigned count=0;
        for(unsigned b=0;b<beam_count;++b)for(unsigned id=0;id<CANDIDATES;++id){
            uint32_t mask=0;for(unsigned k=0;k<beam[b].count;++k)mask|=1u<<beam[b].menu[k];
            if(mask&(1u<<id))continue;mask|=1u<<id;
            unsigned j=0;while(j<count&&seen[j]!=mask)++j;if(j<count)continue;
            seen[count]=mask;plan *p=&next[count];memset(p,0,sizeof(*p));
            for(unsigned k=0;k<CANDIDATES;++k)if(mask&(1u<<k))p->menu[p->count++]=k;
            if(!score_plan(s,p))goto done;++count;
        }
        qsort(next,count,sizeof(*next),rank_plan);plans[depth-1]=next[0];
        beam_count=count<8?count:8;memcpy(beam,next,beam_count*sizeof(*beam));
    }
    ok=1;
done:
    if(session)ng_fit_session_destroy(session);
    for(unsigned m=0;m<12;++m)free(fixed[m]);
    free(proxy);free(activity);free(occupancy.empty);free(occupancy.mask);free(beam);free(next);
    ng_current_geometry=old;return ok;
}

typedef struct {
    size_t n,blocks;unsigned count,ids[3],model_count;
    const uint8_t *src;const vc_ng_occupancy *zero;
    vc_ng_fit *fit[3];
    uint8_t *candidate[48],*symbols,*classes[3],*local,*canonical;
    uint16_t *activity;uint32_t *block;
    vc_ng_distribution stats[3];
} full_state;

static int evaluate(full_state *s,int selected_only)
{
    if(!s->model_count){
        for(size_t i=0;i<s->n;++i)s->symbols[i]=s->candidate[s->canonical[s->block[i]]][i];
        return ng_activity_symbol_values(s->symbols,s->activity)==0;
    }
    const uint8_t *params[3];const uint16_t *bases[3];size_t lengths[3];
    for(unsigned k=0;k<s->model_count;++k){
        params[k]=s->fit[k]->params;lengths[k]=s->fit[k]->param_size;bases[k]=s->fit[k]->base;
    }
    return (selected_only?ng_bank_selected_symbols:ng_bank_symbols)(s->model_count,params,lengths,
        bases,s->canonical,s->src,s->candidate,s->symbols,s->activity)==0;
}

static int select_full(full_state *s)
{
    const uint8_t *symbols[3],*classes[3];const double *costs[3];
    for(unsigned k=0;k<s->count;++k){
        symbols[k]=s->candidate[s->ids[k]];
        if(!vc_ng_distribution_fit(&s->stats[k],symbols[k],s->activity,s->n,s->zero->mask,2)||
           !vc_ng_classify(s->activity,s->n,s->stats[k].threshold,16,s->classes[k]))return 0;
        plus_one_cost(&s->stats[k]);classes[k]=s->classes[k];costs[k]=&s->stats[k].cost[0][0];
    }
    if(!vc_ng_select(symbols,classes,costs,s->count,s->block,s->n,s->local,s->blocks))return 0;
    for(size_t b=0;b<s->blocks;++b)s->canonical[b]=s->zero->count&&s->zero->empty[b]?0:(uint8_t)s->ids[s->local[b]];
    return 1;
}

static int encode_plan(const uint8_t *src,unsigned d,const vc_ng_occupancy *zero,
    search *search,const unsigned *order,unsigned order_count,const plan *p,uint8_t **out,size_t *size)
{
    full_state *s=calloc(1,sizeof(*s));int ok=0;
    if(!s)return 0;
    s->n=(size_t)d*d*d;s->blocks=s->n/64;s->src=src;s->zero=zero;s->count=p->count;
    unsigned map[MODELS],used[MODELS]={0};
    for(unsigned k=0;k<p->count;++k)if(p->menu[k]>=12)used[p->menu[k]-12]=1;
    for(unsigned k=0;k<order_count;++k)if(used[order[k]]){
        map[order[k]]=s->model_count;s->fit[s->model_count++]=&search->fit[order[k]];
    }
    unsigned mask=zero->count?1:0;
    for(unsigned k=0;k<p->count;++k){
        unsigned id=p->menu[k],pred=search->predictor[id];
        s->ids[k]=id<12?pred:12+12*map[id-12]+pred;
        if(id<12)mask|=1u<<pred;
        else {s->candidate[s->ids[k]]=malloc(s->n);if(!s->candidate[s->ids[k]])goto done;
              memcpy(s->candidate[s->ids[k]],search->fit[id-12].symbols,s->n);}
    }
    for(unsigned m=0;m<12;++m)if(mask&(1u<<m)){s->candidate[m]=malloc(s->n);if(!s->candidate[m])goto done;}
    vc_ng_fixed_candidates(src,s->candidate,d,mask);
    s->symbols=malloc(s->n);s->activity=malloc(s->n*sizeof(*s->activity));s->block=malloc(s->n*sizeof(*s->block));
    s->local=malloc(s->blocks);s->canonical=malloc(s->blocks);
    for(unsigned k=0;k<3;++k){s->classes[k]=malloc(s->n);if(!s->classes[k])goto done;}
    if(!s->symbols||!s->activity||!s->block||!s->local||!s->canonical)goto done;
    unsigned seed=0;while(p->menu[seed]!=p->seed)++seed;
    memset(s->canonical,s->ids[seed],s->blocks);
    for(size_t b=0;b<s->blocks;++b)s->local[b]=p->choices[b%SAMPLE_BLOCKS];
    for(unsigned z=0;z<d;++z)for(unsigned y=0;y<d;++y)for(unsigned x=0;x<d;++x)
        s->block[((size_t)z*d+y)*d+x]=((z/4)*(d/4)+y/4)*(d/4)+x/4;
    if(!s->model_count||(s->model_count==1&&p->seed>=12)){
        memcpy(s->symbols,s->candidate[s->ids[seed]],s->n);
        if(ng_activity_symbol_values(s->symbols,s->activity))goto done;
    }else if(!evaluate(s,0))goto done;
    if(s->count>1){
        if(s->model_count&&!evaluate(s,0))goto done;
        if(!select_full(s))goto done;
    }else for(size_t b=0;b<s->blocks;++b)s->canonical[b]=zero->count&&zero->empty[b]?0:(uint8_t)s->ids[0];
    int changed=0;for(size_t b=0;b<s->blocks;++b)changed|=s->canonical[b]!=s->ids[seed];
    if(changed&&!evaluate(s,1))goto done;
    if(s->count>1){
        if(s->model_count&&!evaluate(s,0))goto done;
        if(!select_full(s)||!evaluate(s,1))goto done;
    }
    uint16_t t[15];
    if(!vc_ng_search_thresholds(s->symbols,s->activity,s->n,zero->mask,s->block,s->local,
        s->blocks,s->count,2,NULL,t)||!vc_ng_classify(s->activity,s->n,t,16,s->classes[0]))goto done;
    /* Retain model order across finalist menus, but omit unreferenced models. */
    unsigned live[3]={0},remap[3],nm=0;
    for(size_t b=0;b<s->blocks;++b)if(s->canonical[b]>=12)live[(s->canonical[b]-12)/12]=1;
    vc_ng_model models[3];
    for(unsigned k=0;k<s->model_count;++k)if(live[k]){
        remap[k]=nm;models[nm++]=(vc_ng_model){s->fit[k]->params,s->fit[k]->param_size};
    }
    vc_ng_choice menu[5];unsigned nc=0,ids[48];for(unsigned i=0;i<48;++i)ids[i]=255;
    for(size_t b=0;b<s->blocks;++b){
        if(s->count>1&&zero->count&&zero->empty[b]){s->canonical[b]=0;continue;}
        unsigned id=s->canonical[b];
        if(ids[id]==255){
            unsigned k=id>=12?(id-12)/12:255;
            menu[nc]=(vc_ng_choice){id%12,k==255?255:remap[k],s->count>1?nc:0,{0,0}};
            if(k!=255&&id%12)memcpy(menu[nc].weights,s->fit[k]->weights,sizeof(menu[nc].weights));
            ids[id]=nc++;
        }
        s->canonical[b]=(uint8_t)ids[id];
    }
    if(!vc_ng_frame_encode_zeros(d,models,nm,menu,nc,s->canonical,s->symbols,t,s->classes[0],
        zero->count?zero->empty:NULL,out,size))goto done;
    ok=1;
done:
    for(unsigned m=0;m<48;++m)free(s->candidate[m]);
    for(unsigned k=0;k<3;++k)free(s->classes[k]);
    free(s->symbols);free(s->activity);free(s->block);free(s->local);free(s->canonical);free(s);return ok;
}

int vc_ng_slow_encode(const uint8_t *src,unsigned d,unsigned preset,uint8_t *out,size_t cap,size_t *written)
{
    if(!src||!out||!written||preset>1||!ng_geometry_valid(d,d,d))return -1;
    ng_geometry old=ng_current_geometry;unsigned old_edge=ng_choice_edge;
    ng_choice_edge=4;
    ng_current_geometry=(ng_geometry){(int)d,(int)d,(int)d};
    search *s=calloc(1,sizeof(*s));vc_ng_occupancy zeros={0};plan plans[3];
    uint8_t *frame=NULL,*best=NULL;size_t size=0,best_size=0;int rc=-1;
    int cfg[6]={3,3,preset?3:1,16,preset?160:40,1};
    if(!s||!vc_ng_scan(&zeros,src,d))goto done;
    if(zeros.count==zeros.blocks){
        if(!vc_ng_frame_all_zero(d,&best,&best_size))goto done;goto finish;
    }
    if(!screen(s,src,d,&zeros,cfg,plans))goto done;
    unsigned order[MODELS],count=0,seen[MODELS]={0};
    for(unsigned p=0;p<3;++p)for(unsigned j=0;j<plans[p].count;++j)if(plans[p].menu[j]>=12){
        unsigned k=plans[p].menu[j]-12;if(!seen[k]){order[count++]=k;seen[k]=1;}
    }
    if(d>32){
        for(unsigned j=0;j<count;++j){
            unsigned k=order[j],pred=s->predictor[12+k];uint8_t *fixed[12]={0};
            if(k!=12){fixed[pred]=malloc((size_t)d*d*d);if(!fixed[pred])goto done;
                vc_ng_fixed_candidates(src,fixed,d,1u<<pred);}
            vc_ng_fit_free(&s->fit[k]);
            int ok=vc_ng_fit_input(&s->fit[k],src,cfg,k==12?NULL:fixed[pred],zeros.mask,NULL,NULL,NULL);
            free(fixed[pred]);if(!ok)goto done;
        }
    }
    for(unsigned p=0;p<3;++p){
        if(!encode_plan(src,d,&zeros,s,order,count,&plans[p],&frame,&size))goto done;
        if(!best||size<best_size){free(best);best=frame;best_size=size;frame=NULL;}
        free(frame);frame=NULL;
    }
finish:
    if(best_size>cap)goto done;
    memcpy(out,best,best_size);*written=best_size;rc=0;
done:
    if(s){for(unsigned k=0;k<MODELS;++k)vc_ng_fit_free(&s->fit[k]);
        for(unsigned k=0;k<CANDIDATES;++k){free(s->symbols[k]);free(s->classes[k]);}
        free(s->block);free(s->mask);free(s);}
    free(zeros.empty);free(zeros.mask);free(best);free(frame);
    ng_current_geometry=old;ng_choice_edge=old_edge;return rc;
}
