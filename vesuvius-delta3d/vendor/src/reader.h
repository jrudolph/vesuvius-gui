#ifndef VC_READER_H
#define VC_READER_H
#include "buffer.h"
/* Bounded cursor shared by leaf, metadata and selector parsing. Failed reads
 * never cross the buffer; ok remains false until the caller discards it. */
typedef struct {const uint8_t*p;size_t n,i;int ok;} vc_reader;
static inline const uint8_t* vc_take(vc_reader*r,size_t n){
    if(!r->ok||r->i>r->n||n>r->n-r->i){r->ok=0;return NULL;}
    const uint8_t*p=r->p+r->i;r->i+=n;return p;
}
static inline uint32_t vc_get(vc_reader*r,unsigned n){
    if(n>4){r->ok=0;return 0;}
    const uint8_t*p=vc_take(r,n);uint32_t v=0;
    if(p)for(unsigned j=0;j<n;j++)v|=(uint32_t)p[j]<<(8*j);
    return v;
}
static inline uint64_t vc_varget(vc_reader*r,uint64_t limit){
    uint64_t n=0;
    for(unsigned shift=0;shift<64&&(limit>>shift);shift+=7){
        unsigned v=vc_get(r,1),low=v&127;
        if(!r->ok||low>(limit>>shift))break;
        n|=(uint64_t)low<<shift;
        if(!(v&128))return n;
    }
    r->ok=0;return 0;
}
static inline int vc_varput(vc_buffer*b,uint64_t v){
    do{unsigned c=v&127;v>>=7;if(!vc_buffer_put(b,c|(v?128:0),1))return 0;}while(v);
    return 1;
}
#endif
