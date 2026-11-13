#include "indexed_internal.h"
#include <vc_delta3d/codec.h>
#include "reader.h"
static int plan_tree(domain root,const vc_partition_node*plan,size_t nodes,node**out,size_t*count) {
    domain*stack=NULL;node*t=NULL;size_t cap=0,tc=0,pending=0,used=0,step=0;
    if(!vc_reserve(&stack,&cap,1,sizeof(*stack)))goto fail;stack[pending++]=root;
    while(pending){
        domain d=stack[--pending];vc_partition_node p={4,0};
        if(nodes){if(step>=nodes)goto fail;p=plan[step++];}
        else if(d.channels>1){p.axis=8;p.cut=d.channels;}
        else if(!cube(d.shape)||!leaf_shape(d.shape)){
            p.axis=0;for(unsigned a=1;a<3;a++)if(d.shape[a]>d.shape[p.axis])p.axis=a;
            p.cut=(d.shape[p.axis]/64)*32;
        }
        if(p.axis>8||!vc_reserve(&t,&tc,used+1,sizeof(*t)))goto fail;
        node n={p.axis==4?8:p.axis>4?p.axis-1:p.axis,p.cut,0};t[used++]=n;
        if(n.tag==8){
            if(p.cut||d.channels!=1||!leaf_shape(d.shape)||!cube(d.shape))goto fail;
        }else{
            domain a,b;uint32_t k;if(!children(d,&n,&a,&b,&k))goto fail;
            if(!vc_reserve(&stack,&cap,pending+2,sizeof(*stack)))goto fail;
            if(n.tag<4)stack[pending++]=b;stack[pending++]=a;
        }
    }
    if((nodes&&step!=nodes)||!finish_tree(t,used))goto fail;
    free(stack);*out=t;*count=used;return 1;
fail:free(stack);free(t);return 0;
}
int vc_indexed_encode(const uint8_t*input,size_t input_size,const uint32_t shape[3],uint32_t channels,
    unsigned layout,unsigned preset,const vc_partition_node*plan,size_t nodes,uint8_t**output,size_t*output_size) {
    if(!output||!output_size)return 0;*output=NULL;*output_size=0;
    uint64_t field;
    if(!input||!shape||!channels||channels>INT32_MAX||layout>1||preset==VC_DELTA||!vc_preset_valid(preset)||
       (nodes&&!plan)||!voxels(shape,&field)||field>SIZE_MAX/channels||field*channels!=input_size)return 0;
    domain root={0};memcpy(root.shape,shape,sizeof(root.shape));root.channels=channels;
    node*t=NULL;size_t nt=0,nl=0;vc_indexed_leaf*leaves=NULL;
    vc_buffer payload={0},header={0};uint8_t*scratch=NULL;size_t scratch_cap=0;
    if(!plan_tree(root,plan,nodes,&t,&nt)||!expand(t,root,input_size/32768,&leaves,&nl))goto fail;
    for(size_t i=0;i<nl;i++){
        vc_indexed_leaf*l=leaves+i;uint64_t vox;if(!voxels(l->shape,&vox)||vox>SIZE_MAX)goto fail;
        size_t n=(size_t)vox;const uint8_t*src;
        if(!layout&&full(l,shape))src=input+(size_t)l->channel*n;
        else{
            if(!vc_reserve(&scratch,&scratch_cap,n,1))goto fail;size_t p=0;
            for(uint32_t z=0;z<l->shape[0];z++)for(uint32_t y=0;y<l->shape[1];y++){
                size_t q=((size_t)(l->origin[0]+z)*shape[1]+l->origin[1]+y)*shape[2]+l->origin[2];
                if(!layout)memcpy(scratch+p,input+q+(size_t)l->channel*field,l->shape[2]);
                else for(uint32_t x=0;x<l->shape[2];x++)scratch[p+x]=input[(q+x)*channels+l->channel];
                p+=l->shape[2];
            }src=scratch;
        }
        if(payload.size>SIZE_MAX-131072||n>(SIZE_MAX-payload.size-131072)/4||
           !vc_reserve(&payload.data,&payload.capacity,payload.size+4*n+131072,1))goto fail;
        size_t bytes=0;uint8_t*dst=payload.data+payload.size;
        int rc=vc_ng_preset_encode(src,l->shape[0],preset,dst,payload.capacity-payload.size,&bytes);
        if(rc||bytes<8||bytes>payload.capacity-payload.size||memcmp(dst,"NCF1",4))goto fail;
        /* Embedded leaves share the container's codec identity. */
        memmove(dst,dst+4,bytes-4);l->length=bytes-4;payload.size+=bytes-4;
    }
    if(!vc_buffer_append(&header,VC_INDEX_MAGIC,4)||!vc_buffer_put(&header,0,4)||!vc_buffer_put(&header,layout,1))goto fail;
    for(unsigned a=0;a<3;a++)if(!vc_varput(&header,shape[a]/32-1))goto fail;
    if(!vc_varput(&header,channels-1))goto fail;
    for(size_t i=0;i<nt;i++){
        if(!vc_buffer_put(&header,t[i].tag,1))goto fail;
        if(t[i].tag!=8){uint32_t v=t[i].value;if(t[i].tag<3)v/=32;
            if(!vc_varput(&header,v-1))goto fail;}
    }
    for(size_t i=0;i<nl;i++)if(!vc_varput(&header,leaves[i].length))goto fail;
    if(header.size>UINT32_MAX)goto fail;write_le(header.data+4,header.size,4);
    if(!vc_buffer_append(&header,payload.data,payload.size))goto fail;
    free(t);free(leaves);free(scratch);free(payload.data);*output=header.data;*output_size=header.size;return 1;
fail:free(t);free(leaves);free(scratch);free(payload.data);free(header.data);return 0;
}
