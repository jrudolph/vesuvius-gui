/* Compact schema + one numerical pool. The schema walk is shared by encode
 * and decode. Numerical coding knows only word-array boundaries, not models
 * or frequency tables. No compatibility representations, XOR, RLE or packing
 * alternatives. Exact log ranges retain all low bits. */
#include "header.h"
#include "ng_frame.h"
#include "buffer.h"
#include "ng_stream.h"
#include "residual_bridge.h"

#include "reader.h"
typedef struct {vc_buffer words;size_t at[134],len[134];unsigned count;} arrays;
static int record(arrays*a,const uint8_t*p,size_t n){
    if(a->count>=134||n%2||n>8192)return 0;
    a->at[a->count]=a->words.size;a->len[a->count++]=n;
    if(p)return vc_buffer_append(&a->words,p,n);
    if(!vc_reserve(&a->words.data,&a->words.capacity,a->words.size+n,1))return 0;
    memset(a->words.data+a->words.size,0,n);a->words.size+=n;return 1;
}
/* Optional within-record wrapping word delta, shared by all numeric records.
 * It acts on pairs of log labels; an odd last label remains unchanged. */
static void delta(uint8_t*p,size_t n,int inverse){
    unsigned prev=0;for(size_t i=0;i+1<n;i+=2){unsigned v=p[i]|(unsigned)p[i+1]<<8;
        unsigned q=inverse?(v+prev)&65535:(v-prev)&65535;
        prev=inverse?q:v;p[i]=(uint8_t)q;p[i+1]=(uint8_t)(q>>8);}
}
static int pool_encode(const arrays*a,vc_buffer*out){
    vc_buffer counts={0},labels={0},bits={0},best={0};int ok=0;
    size_t lengths[134];uint64_t acc=0;unsigned held=0;
    for(unsigned k=0;k<a->count;k++){
        const uint8_t*p=a->words.data+a->at[k];size_t n=a->len[k]/2;
        while(n&&!p[2*n-2]&&!p[2*n-1])n--;
        lengths[k]=n;if(!vc_varput(&counts,(uint32_t)n))goto done;
        for(size_t i=0;i<n;i++){
            unsigned v=p[2*i]|(unsigned)p[2*i+1]<<8,nb=0,s=v;
            if(v>=16){for(unsigned q=v;q>>=1;)nb++;s=12+nb;v-=1u<<nb;}
            if(!vc_buffer_put(&labels,s,1))goto done;
            if(nb){acc|=(uint64_t)v<<held;held+=nb;}
            while(held>=8){if(!vc_buffer_put(&bits,acc&255,1))goto done;acc>>=8;held-=8;}
        }
    }
    if(held&&!vc_buffer_put(&bits,acc,1))goto done;
    /* Only raw versus entropy-coded labels, with/without the one delta op. */
    for(unsigned d=0;d<2;d++){
        if(d){size_t at=0;for(unsigned k=0;k<a->count;k++){if(lengths[k])delta(labels.data+at,lengths[k],0);at+=lengths[k];}}
        uint8_t*z=NULL;size_t zn=0;
        if(labels.size&&!vc_ng_stream_fixed_encode(labels.data,labels.size,d,&z,&zn))goto done;
        int coded=labels.size&&zn<labels.size;
        vc_buffer trial={0};
        if(!vc_buffer_put(&trial,d*4+coded,1)||!vc_varput(&trial,(uint32_t)(coded?zn:labels.size))||
           !vc_buffer_append(&trial,coded?z:labels.data,coded?zn:labels.size)){
            free(z);free(trial.data);goto done;}
        free(z);if(!best.data||trial.size<best.size){free(best.data);best=trial;}else free(trial.data);
    }
    ok=vc_buffer_append(out,counts.data,counts.size)&&vc_buffer_append(out,best.data,best.size)&&vc_buffer_append(out,bits.data,bits.size);
done:free(counts.data);free(labels.data);free(bits.data);free(best.data);return ok;
}
static int pool_decode(vc_reader*r,arrays*a){
    size_t lengths[134],total=0;uint8_t*labels=NULL;int ok=0;
    for(unsigned k=0;k<a->count;k++){lengths[k]=vc_varget(r,UINT32_MAX);if(!r->ok||lengths[k]>a->len[k]/2)goto done;total+=lengths[k];}
    unsigned tag=vc_get(r,1);size_t zn=vc_varget(r,UINT32_MAX);const uint8_t*z=vc_take(r,zn);
    if(!r->ok||tag>6||(tag&3)==3||!(labels=malloc(total?total:1)))goto done;
    if((tag&3)==1){if(!total||!vc_ng_stream_fixed_decode(z,zn,tag>>2,labels,total))goto done;}
    else if((tag&3)==2){if(!total||!vc_ng_stream_decode(z,zn,labels,total))goto done;}
    else{if(zn!=total)goto done;if(total)memcpy(labels,z,total);}
    size_t at=0;uint64_t acc=0;unsigned held=0;
    for(unsigned k=0;k<a->count;k++){
        if(tag&4)delta(labels+at,lengths[k],1);
        uint8_t*p=a->words.data+a->at[k];
        for(size_t i=0;i<lengths[k];i++){
            unsigned s=labels[at++],v=s;if(s>27)goto done;
            if(s>=16){unsigned nb=s-12;while(held<nb){acc|=(uint64_t)vc_get(r,1)<<held;held+=8;}
                v=(1u<<nb)+(unsigned)(acc&((1u<<nb)-1));acc>>=nb;held-=nb;}
            p[2*i]=(uint8_t)v;p[2*i+1]=(uint8_t)(v>>8);
        }
    }
    ok=r->ok&&r->i==r->n&&!acc;
done:free(labels);return ok;
}

typedef struct {vc_reader input;vc_buffer output;arrays*a;int decode;unsigned record;} walk;
static uint32_t field(walk*w,unsigned n){
    uint32_t v=w->decode?vc_varget(&w->input,UINT32_MAX):vc_get(&w->input,n);
    if((n<4&&v>=(1u<<(8*n)))||!(w->decode?vc_buffer_put(&w->output,v,n):vc_varput(&w->output,v)))w->input.ok=0;
    return v;
}
static int numerical(walk*w,size_t n){
    if(w->decode){
        if(w->record>=w->a->count||w->a->len[w->record]!=n)return 0;
        size_t at=w->a->at[w->record++];return vc_buffer_append(&w->output,w->a->words.data+at,n);
    }
    const uint8_t*p=vc_take(&w->input,n);return w->input.ok&&record(w->a,p,n);
}
/* A single standard model: version is an internal constant, never transmitted.
 * Geometry determines the coefficient count; classes determine threshold/bias
 * counts. Do not transmit lengths the decoder can compute. */
static int model(walk*w,int shapes){
    uint8_t cfg[6]={3,0,0,0,0,0};
    if(w->decode){if(!vc_buffer_put(&w->output,3,1))return 0;}
    else if(vc_get(&w->input,1)!=3)return 0;
    for(unsigned i=1;i<6;i++)cfg[i]=(uint8_t)field(w,1);
    size_t n=ng_model_size(cfg);if(!w->input.ok||n<6)return 0;
    if(shapes)return record(w->a,NULL,n-6);
    return numerical(w,n-6);
}
static int schema(walk*w,int shapes){
    unsigned x=field(w,4),y=field(w,4),z=field(w,4),mode=field(w,4),nm=field(w,4),nc=field(w,4);
    (void)field(w,4);unsigned edge=mode>>8;
    if(!w->input.ok||!ng_geometry_valid(x,y,z)||(mode&255)>1||nm>4||!nc||nc>60||
       !vc_ng_edge_valid(edge))return 0;
    for(unsigned m=0;m<nm;m++)if(!model(w,shapes))return 0;
    unsigned banks=1;
    for(unsigned i=0;i<nc;i++){
        unsigned p=field(w,1),m=field(w,1),b=field(w,1);
        if(w->decode){if(!vc_buffer_put(&w->output,0,1))return 0;}
        else if(vc_get(&w->input,1))return 0;
        unsigned a=field(w,4),c=field(w,4);
        if(!vc_ng_choice_valid(p,m,b,a,c,nm))return 0;
        if(banks<=b)banks=b+1;
    }
    unsigned entropy=w->decode?vc_varget(&w->input,UINT32_MAX):w->input.i<w->input.n;
    if(entropy>1||!w->input.ok)return 0;
    if(!w->decode&&!vc_varput(&w->output,entropy))return 0;
    if(entropy){
        /* Entropy needs geometry, thresholds and tables, not another copy of
         * fitted predictor weights. The native kernel still accepts its
         * ordinary parameter struct; reconstruct unused fields as zeros. */
        uint8_t cfg[6]={3,0,0,0,0,0};
        if(w->decode){if(!vc_buffer_put(&w->output,3,1))return 0;}
        else if(vc_get(&w->input,1)!=3)return 0;
        for(unsigned j=1;j<6;j++)cfg[j]=(uint8_t)field(w,1);
        size_t pn=ng_model_size(cfg);unsigned K=cfg[4];
        if(!w->input.ok||!pn)return 0;
        if(K%banks||K/banks<2||K/banks>32)return 0;
        size_t prefix=pn-6-(4u*K-2),thresholds=2*(K/banks-1),suffix=4u*K-2-thresholds;
        if(w->decode){
            for(size_t j=0;j<prefix;j++)if(!vc_buffer_put(&w->output,0,1))return 0;
            if(shapes){if(!record(w->a,NULL,thresholds))return 0;}
            else if(!numerical(w,thresholds))return 0;
            for(size_t j=0;j<suffix;j++)if(!vc_buffer_put(&w->output,0,1))return 0;
        }else{
            const uint8_t*p=vc_take(&w->input,prefix);if(!w->input.ok)return 0;
            for(size_t j=0;j<prefix;j++)if(p[j])return 0;
            if(!numerical(w,thresholds))return 0;
            p=vc_take(&w->input,suffix);if(!w->input.ok)return 0;
            for(size_t j=0;j<suffix;j++)if(p[j])return 0;
        }
        for(unsigned k=0;k<K;k++){
            if(shapes){if(!record(w->a,NULL,512))return 0;}
            else{
                size_t at=w->decode?w->output.size:w->input.i;
                if(!numerical(w,512))return 0;
                const uint8_t*p=(w->decode?w->output.data:w->input.p)+at;unsigned sum=0;
                for(unsigned j=0;j<256;j++)sum+=p[2*j]|(unsigned)p[2*j+1]<<8;
                if(sum!=(1u<<cfg[5]))return 0;
            }
        }
    }
    return w->input.ok;
}
int vc_header_encode(const uint8_t*p,size_t n,uint8_t**out,size_t*size){
    if(!out||!size)return 0;*out=NULL;*size=0;if(!p||n>1048576)return 0;
    arrays a={0};walk w={{p,n,0,1},{0},&a,0,0};int ok=0;
    if(schema(&w,0)&&w.input.i==n&&pool_encode(&a,&w.output)){
        *out=w.output.data;*size=w.output.size;w.output.data=NULL;ok=1;}
    free(w.output.data);free(a.words.data);return ok;
}
int vc_header_decode(const uint8_t*p,size_t n,uint8_t**out,size_t*size){
    if(!out||!size)return 0;*out=NULL;*size=0;if(!p||n>1048576)return 0;
    arrays a={0};walk w={{p,n,0,1},{0},&a,1,0};int ok=0;
    if(!schema(&w,1))goto done;
    size_t controls=w.input.i;
    if(!pool_decode(&w.input,&a))goto done;
    w.input=(vc_reader){p,controls,0,1};w.output.size=0;w.record=0;
    if(!schema(&w,0)||w.input.i!=controls||w.record!=a.count)goto done;
    *out=w.output.data;*size=w.output.size;w.output.data=NULL;ok=1;
done:free(w.output.data);free(a.words.data);return ok;
}
