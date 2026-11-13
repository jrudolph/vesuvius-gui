#include "indexed_internal.h"
#include "reader.h"
uint32_t vc_indexed_header_size(const uint8_t*p,size_t n) {
    if(!p||n<VC_INDEX_PREFIX||memcmp(p,VC_INDEX_MAGIC,4))return 0;
    /* Smallest index: prefix, flags, three dimensions, channels, leaf, length. */
    uint32_t size=(uint32_t)read_le(p+4,4);return size>=VC_INDEX_PREFIX+7?size:0;
}
void vc_index_free(vc_chunk_index*index) {
    if(index){free(index->leaves);memset(index,0,sizeof(*index));}
}
int vc_index_read(const uint8_t*data,size_t size,uint64_t object,vc_chunk_index*result) {
    if(!result)return 0;memset(result,0,sizeof(*result));
    uint32_t header=vc_indexed_header_size(data,size);if(!header||header>size||header>object)return 0;
    vc_chunk_index index={0};index.header_bytes=header;index.encoded_bytes=object;
    vc_reader r={data,header,VC_INDEX_PREFIX,1};
    index.layout=vc_get(&r,1);if(!r.ok||index.layout>1)return 0;
    for(unsigned a=0;a<3;a++)index.shape[a]=(uint32_t)(vc_varget(&r,UINT32_MAX/32-1)+1)*32;
    index.channels=(uint32_t)vc_varget(&r,INT32_MAX-1)+1;
    if(!r.ok||!voxels(index.shape,&index.decoded_bytes)||index.decoded_bytes>SIZE_MAX/index.channels)return 0;
    index.decoded_bytes*=index.channels;
    node*t=NULL;size_t cap=0,nt=0,needed=1;
    while(needed){
        if(!vc_reserve(&t,&cap,nt+1,sizeof(*t)))goto fail;
        unsigned tag=vc_get(&r,1);uint64_t value=0;
        if(!r.ok||tag>8)goto fail;
        if(tag<8){
            value=vc_varget(&r,UINT32_MAX-1)+1;if(tag<3)value*=32;
            if(!r.ok||value>UINT32_MAX)goto fail;
        }
        t[nt++]=(node){tag,(uint32_t)value,0};needed--;
        if(tag<4)needed+=2;else if(tag<8)needed++;
        if(needed>r.n-r.i)goto fail;
    }
    if(!finish_tree(t,nt))goto fail;
    /* Every leaf length takes at least one byte, which bounds expansion. */
    domain root={0};memcpy(root.shape,index.shape,sizeof(root.shape));root.channels=index.channels;
    if(!expand(t,root,r.n-r.i,&index.leaves,&index.leaf_count))goto fail;
    uint64_t next=header;
    for(size_t i=0;i<index.leaf_count;i++){
        vc_indexed_leaf*l=index.leaves+i;l->offset=next;l->length=vc_varget(&r,UINT64_MAX);
        if(!r.ok||l->length<4||next>object||l->length>object-next)goto fail;next+=l->length;
    }
    if(r.i!=header||next!=object)goto fail;free(t);*result=index;return 1;
fail:free(t);vc_index_free(&index);return 0;
}
int vc_indexed_leaf_decode(const uint8_t*src,size_t size,const vc_indexed_leaf*l,uint8_t*out,size_t count) {
    uint64_t n;
    if(!src||!out||!l||size<4||size!=l->length||l->codec!=VC_LEAF_CHOICE||
       !leaf_shape(l->shape)||!voxels(l->shape,&n)||n!=count)return 0;
    return vc_ng_leaf_decode(src,size,l->shape,out,count)==0;
}
int vc_indexed_decode(const uint8_t *src,size_t size,uint8_t *out,size_t count) {
    vc_chunk_index index;
    if(!out || !vc_index_read(src,size,size,&index)) return 0;
    if(index.decoded_bytes!=count) { vc_index_free(&index); return 0; }
    uint8_t *scratch=NULL; size_t capacity=0;
    for(size_t i=0;i<index.leaf_count;++i) {
        const vc_indexed_leaf *l=&index.leaves[i]; uint64_t n; voxels(l->shape,&n);
        if(!index.layout && full(l,index.shape)) {
            if(!vc_indexed_leaf_decode(src+l->offset,(size_t)l->length,l,out+(size_t)l->channel*(size_t)n,(size_t)n)) goto fail;
        } else {
            if(n>capacity) {
                uint8_t *grown=realloc(scratch,(size_t)n); if(!grown) goto fail;
                scratch=grown; capacity=(size_t)n;
            }
            if(!vc_indexed_leaf_decode(src+l->offset,(size_t)l->length,l,scratch,(size_t)n)) goto fail;
            size_t p=0,field=(size_t)(index.decoded_bytes/index.channels);
            for(uint32_t z=0;z<l->shape[0];++z) for(uint32_t y=0;y<l->shape[1];++y) {
                size_t q=((size_t)(l->origin[0]+z)*index.shape[1]+l->origin[1]+y)*index.shape[2]+l->origin[2];
                if(!index.layout) memcpy(out+q+(size_t)l->channel*field,scratch+p,l->shape[2]);
                else for(uint32_t x=0;x<l->shape[2];++x) out[(q+x)*index.channels+l->channel]=scratch[p+x];
                p+=l->shape[2];
            }
        }
    }
    free(scratch); vc_index_free(&index); return 1;
fail:
    free(scratch); vc_index_free(&index); return 0;
}
