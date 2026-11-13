#include "buffer.h"
/* Retained NCF1 encoder. Model, selector and entropy sections share the same
 * native fitting and wavefront entropy kernels as the retained presets.
 * Fixed block edges 2,4,8,16,32 are supported. Experimental adaptive/tree frames
 * are deliberately not accepted by the reduced codec. */
#include "residual_bridge.h"
#include "rans.h"
#include "ng_frame.h"
#include "selector.h"
#include "header.h"
#include <stdlib.h>
#include <string.h>

static int metadata(vc_buffer*h,unsigned x,unsigned y,unsigned z,unsigned edge,unsigned mode,
    const vc_ng_model*models,unsigned nm,const vc_ng_choice*menu,unsigned nc,size_t selector_bytes){
    if(!vc_buffer_put(h,x,4)||!vc_buffer_put(h,y,4)||!vc_buffer_put(h,z,4)||
       !vc_buffer_put(h,mode|(edge<<8),4)||!vc_buffer_put(h,nm,4)||!vc_buffer_put(h,nc,4)||!vc_buffer_put(h,selector_bytes,4))return 0;
    for(unsigned m=0;m<nm;m++)
        if(models[m].size<6||!models[m].data||ng_model_size(models[m].data)!=models[m].size||
           !vc_buffer_append(h,models[m].data,models[m].size))return 0;
    for(unsigned i=0;i<nc;i++){
        const vc_ng_choice*c=menu+i;
        if(!vc_buffer_put(h,c->predictor,1)||!vc_buffer_put(h,c->model,1)||!vc_buffer_put(h,c->entropy,1)||!vc_buffer_put(h,0,1)||
           !vc_buffer_put(h,(uint32_t)c->weights[0],4)||!vc_buffer_put(h,(uint32_t)c->weights[1],4))return 0;
    }return 1;
}
/* Screening charges actual model/menu storage using the same header codec.
 * Entropy table cost is accounted separately by the histogram estimator. */
double vc_ng_model_cost(const vc_ng_model*models,unsigned nm,const vc_ng_choice*menu,unsigned nc){
    vc_buffer h={0};uint8_t*z=NULL;size_t n=0;double cost=-1;
    if(metadata(&h,32,32,32,4,1,models,nm,menu,nc,2)&&vc_header_encode(h.data,h.size,&z,&n))cost=(double)n+8;
    free(h.data);free(z);return cost;
}

int vc_ng_frame_all_zero(unsigned d,uint8_t **out,size_t *size)
{
    uint8_t *selectors=calloc((size_t)(d/4)*(d/4)*(d/4),1);
    vc_ng_choice zero={12,255,0,{0,0}};uint16_t t[15]={0};
    int ok=selectors&&vc_ng_frame_encode(d,4,1,NULL,0,&zero,1,selectors,NULL,t,16,NULL,out,size);
    free(selectors);return ok;
}

int vc_ng_frame_encode_zeros(unsigned d,const vc_ng_model *models,unsigned nm,
    const vc_ng_choice *menu,unsigned nc,const uint8_t *selectors,const uint8_t *symbols,
    const uint16_t *thresholds,const uint8_t *classes,const uint8_t *empty,uint8_t **out,size_t *size)
{
    if(!empty)return vc_ng_frame_encode(d,4,1,models,nm,menu,nc,selectors,symbols,thresholds,16,classes,out,size);
    if(!nc||nc>=60)return 0;
    vc_ng_choice extended[60];memcpy(extended,menu,nc*sizeof(*menu));
    unsigned bank=0;
    for(unsigned i=0;i<nc;++i)if(bank<=menu[i].entropy)bank=menu[i].entropy+1;
    if(bank>=4)return 0;
    extended[nc]=(vc_ng_choice){12,255,bank,{0,0}};
    size_t count=(size_t)(d/4)*(d/4)*(d/4);
    uint8_t *selected=malloc(count);if(!selected)return 0;
    for(size_t i=0;i<count;++i)selected[i]=empty[i]?(uint8_t)nc:selectors[i];
    int ok=vc_ng_frame_encode(d,4,1,models,nm,extended,nc+1,selected,symbols,thresholds,16,classes,out,size);
    free(selected);return ok;
}

int vc_ng_frame_encode(unsigned d,unsigned edge,unsigned mode,const vc_ng_model *models,unsigned nm,
    const vc_ng_choice *menu,unsigned nc,const uint8_t *selectors,const uint8_t *symbols,
    const uint16_t *thresholds,unsigned K,const uint8_t *classes,uint8_t **out,size_t *out_size) {
    return vc_ng_frame_encode_shape(d,d,d,edge,mode,models,nm,menu,nc,selectors,symbols,thresholds,K,classes,out,out_size);
}
int vc_ng_frame_encode_shape(unsigned x,unsigned y,unsigned z,unsigned edge,unsigned mode,const vc_ng_model *models,unsigned nm,
    const vc_ng_choice *menu,unsigned nc,const uint8_t *selectors,const uint8_t *symbols,
    const uint16_t *thresholds,unsigned K,const uint8_t *classes,uint8_t **out,size_t *out_size) {
    if(!out || !out_size) return 0;
    *out=NULL;*out_size=0;
    if(!ng_geometry_valid(x,y,z) || !vc_ng_edge_valid(edge) || mode>1 ||
       nm>4 || (nm&&!models) || !nc || nc>60 || !menu || !selectors || K<2 || K>32) return 0;
    vc_buffer header={0},params={0},result={0};
    uint8_t *cls=NULL,*entropy=NULL,*compressed=NULL,*packed=NULL;
    int ok=0;unsigned banks=1;
    size_t count=(size_t)(x/edge)*(y/edge)*(z/edge),volume=(size_t)x*y*z;
    size_t selector_bytes=0;
    if(!vc_selector_encode(selectors,count,&packed,&selector_bytes)||selector_bytes>UINT32_MAX)goto done;
    for(unsigned i=0;i<nc;++i) {
        const vc_ng_choice *c=&menu[i];
        if(!vc_ng_choice_valid(c->predictor,c->model,c->entropy,c->weights[0],c->weights[1],nm)) goto done;
        if(banks<=c->entropy) banks=c->entropy+1;
    }
    if(K*banks>128) goto done;
    if(!metadata(&header,x,y,z,edge,mode,models,nm,menu,nc,selector_bytes))goto done;
    int all_zero=1,max_delta=-1;unsigned chosen=UINT32_MAX;
    for(size_t i=0;i<count;++i) {
        unsigned id=selectors[i];if(id>=nc) goto done;
        all_zero&=menu[id].predictor==12;unsigned m=menu[id].model;if(m==255) continue;
        const vc_ng_model *p=&models[m];if(p->size<6) goto done;
        int delta=(int8_t)p->data[2];if(delta<(int8_t)p->data[3]) delta=(int8_t)p->data[3];
        if(delta>max_delta) { max_delta=delta;chosen=m; }
    }
    size_t payload_size=0,entropy_header=0;
    if(!all_zero) {
        if(!symbols || !classes || !thresholds)goto done;
        if(chosen!=UINT32_MAX) { if(!vc_buffer_append(&params,models[chosen].data,models[chosen].size))goto done; }
        else {
            uint8_t empty[4096];size_t size=0;
            if(ng_bank_empty_model(empty,sizeof(empty),&size)||!vc_buffer_append(&params,empty,size))goto done;
        }
        if(params.size<6 || params.data[4]<2 || params.data[4]>64)goto done;
        unsigned tail=4u*params.data[4]-2;if(params.size<tail)goto done;
        /* Entropy traversal needs tap geometry, not predictor coefficients. */
        memset(params.data+6,0,params.size-tail-6);
        params.size-=tail;params.data[4]=(uint8_t)(K*banks);
        for(unsigned i=0;i<K-1;++i)if(!vc_buffer_put(&params,thresholds[i],2))goto done;
        for(unsigned i=K-1;i<K*banks-1;++i)if(!vc_buffer_put(&params,0,2))goto done;
        for(unsigned i=0;i<K*banks;++i)if(!vc_buffer_put(&params,0,2))goto done;
        cls=malloc(volume);entropy=malloc(3*volume+131072);if(!cls||!entropy)goto done;
        for(unsigned iz=0;iz<z;++iz)for(unsigned iy=0;iy<y;++iy) {
            size_t b=((size_t)(iz/edge)*(y/edge)+iy/edge)*(x/edge);
            for(unsigned ix=0;ix<x;ix+=edge,++b) {
                unsigned bank=K*menu[selectors[b]].entropy;size_t i=((size_t)iz*y+iy)*x+ix;
                for(unsigned j=0;j<edge;++j) { if(classes[i+j]>=K)goto done;cls[i+j]=(uint8_t)(classes[i+j]+bank); }
            }
        }
        ng_geometry previous=ng_current_geometry;ng_current_geometry=(ng_geometry){(int)x,(int)y,(int)z};
        int rc=ng_wave_encode(params.data,params.size,symbols,cls,entropy,3*volume+131072,&payload_size);
        ng_current_geometry=previous;
        if(rc || ng_entropy_header_size(entropy,payload_size,&entropy_header) || entropy_header>payload_size ||
           !vc_buffer_append(&header,entropy,entropy_header))goto done;
    }
    if(header.size>UINT32_MAX)goto done;
    size_t bytes=0;
    if(!vc_header_encode(header.data,header.size,&compressed,&bytes)||bytes>UINT32_MAX||!vc_buffer_append(&result,"NCF1",4)||
       !vc_buffer_put(&result,(uint32_t)bytes,4)||!vc_buffer_append(&result,compressed,bytes)||
       !vc_buffer_append(&result,packed,selector_bytes)||
       (payload_size>entropy_header&&!vc_buffer_append(&result,entropy+entropy_header,payload_size-entropy_header)))goto done;
    *out=result.data;*out_size=result.size;result.data=NULL;ok=1;
done:
    free(result.data);free(params.data);free(header.data);free(cls);free(entropy);free(compressed);free(packed);return ok;
}
