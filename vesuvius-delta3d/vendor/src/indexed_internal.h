#pragma once
#include "buffer.h"
#include "ng/ng_frame.h"
/* NGI1: one preorder template, consecutive lengths, consecutive payloads.
 * Repeat nodes share geometry, never prediction state. All traversals are
 * iterative and expansion is bounded by the physically present length table. */
#include <vc_delta3d/indexed.h>
#include <stdlib.h>
#include <string.h>
#define VC_INDEX_MAGIC "NGI1"
#define VC_INDEX_PREFIX 8u /* magic + u32 total header length */
typedef struct { uint32_t origin[3],shape[3],channel,channels; } domain;
typedef struct { unsigned tag; uint32_t value; size_t end; } node;
typedef struct { domain d; size_t node; } job;
static uint64_t read_le(const uint8_t*p,unsigned n) {
    uint64_t v=0;for(unsigned i=0;i<n;i++)v|=(uint64_t)p[i]<<(8*i);return v;
}
static void write_le(uint8_t*p,uint64_t v,unsigned n) {
    for(unsigned i=0;i<n;i++)p[i]=(uint8_t)(v>>(8*i));
}
static int voxels(const uint32_t s[3],uint64_t*n) {
    uint64_t v=1;for(unsigned a=0;a<3;a++){
        if(!s[a]||s[a]%32||v>UINT64_MAX/s[a])return 0;v*=s[a];
    }*n=v;return 1;
}
static int leaf_shape(const uint32_t s[3]) {
    uint64_t n;return voxels(s,&n)&&s[2]<=1024&&(uint64_t)s[0]*s[1]<=65536;
}
static int cube(const uint32_t s[3]) {return s[0]==s[1]&&s[1]==s[2];}
static int full(const vc_indexed_leaf*l,const uint32_t s[3]) {
    return !l->origin[0]&&!l->origin[1]&&!l->origin[2]&&!memcmp(l->shape,s,3*sizeof(*s));
}
/* Domain subdivision shared by planner and reader. Repeat partitions an axis
 * into equal regions; the child's cuts are relative to each such region. */
static int children(domain d,const node*n,domain*first,domain*second,uint32_t*count) {
    unsigned a=n->tag&3;uint32_t extent=a==3?d.channels:d.shape[a],v=n->value;
    *first=*second=d;
    if(n->tag<4){
        if(!v||v>=extent||(a<3&&v%32))return 0;*count=2;
        if(a==3){first->channels=v;second->channel+=v;second->channels-=v;}
        else{first->shape[a]=v;second->origin[a]+=v;second->shape[a]-=v;}
    }else{
        if(v<2||extent%v)return 0;uint32_t part=extent/v;
        if(!part||(a<3&&part%32))return 0;*count=v;
        if(a==3)first->channels=part;else first->shape[a]=part;
        *second=*first;
    }return 1;
}
/* Compute subtree ends in reverse preorder, without recursion. */
static int finish_tree(node*t,size_t n) {
    for(size_t i=n;i--;){
        if(t[i].tag==8)t[i].end=i+1;
        else{
            if(i+1>=n)return 0;size_t end=t[i+1].end;
            if(t[i].tag<4){if(end>=n)return 0;end=t[end].end;}
            if(end>n)return 0;t[i].end=end;
        }
    }return n&&t[0].end==n;
}
static int expand(const node*t,domain root,size_t limit,vc_indexed_leaf**out,size_t*count) {
    job*stack=NULL;size_t cap=0,pending=0,lc=0;vc_indexed_leaf*leaves=NULL;size_t lcap=0;
    if(!limit||!vc_reserve(&stack,&cap,1,sizeof(*stack)))goto fail;
    stack[pending++]=(job){root,0};
    while(pending){
        job j=stack[--pending];const node*n=t+j.node;
        if(n->tag==8){
            if(j.d.channels!=1||!leaf_shape(j.d.shape)||lc>=limit||
               !vc_reserve(&leaves,&lcap,lc+1,sizeof(*leaves)))goto fail;
            vc_indexed_leaf*l=leaves+lc++;memset(l,0,sizeof(*l));
            l->channel=j.d.channel;memcpy(l->origin,j.d.origin,sizeof(l->origin));memcpy(l->shape,j.d.shape,sizeof(l->shape));
            l->codec=VC_LEAF_CHOICE;continue;
        }
        domain first,second;uint32_t k;
        if(!children(j.d,n,&first,&second,&k)||pending>limit-lc||k>limit-lc-pending||
           !vc_reserve(&stack,&cap,pending+k,sizeof(*stack)))goto fail;
        if(n->tag<4){stack[pending++]=(job){second,t[j.node+1].end};stack[pending++]=(job){first,j.node+1};}
        else for(uint32_t r=k;r--;){
            domain d=first;unsigned a=n->tag&3;
            if(a==3)d.channel+=r*d.channels;else d.origin[a]+=r*d.shape[a];
            stack[pending++]=(job){d,j.node+1};
        }
    }
    free(stack);*out=leaves;*count=lc;return 1;
fail:free(stack);free(leaves);return 0;
}
