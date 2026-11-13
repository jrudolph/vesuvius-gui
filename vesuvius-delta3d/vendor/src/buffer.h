#ifndef VC_BUFFER_H
#define VC_BUFFER_H
/* Internal checked storage shared by frame/container writers. Append sources
 * must not alias the destination allocation (which can move during growth). */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

static inline int vc_reserve(void *storage,size_t *capacity,size_t count,size_t element) {
    if(count<=*capacity)return 1;
    if(!element || count>SIZE_MAX/element)return 0;
    size_t next=*capacity<=SIZE_MAX/2?*capacity*2:SIZE_MAX;
    if(next<count)next=count;
    if(next>SIZE_MAX/element)next=count;
    /* memcpy permits callers to pass T** without pointer-aliasing violations. */
    void *old;memcpy(&old,storage,sizeof(old));
    void *grown=realloc(old,next*element);if(!grown)return 0;
    memcpy(storage,&grown,sizeof(grown));*capacity=next;return 1;
}
typedef struct {uint8_t *data;size_t size,capacity;} vc_buffer;
static inline int vc_buffer_append(vc_buffer *b,const void *src,size_t size) {
    if((size&&!src) || size>SIZE_MAX-b->size || !vc_reserve(&b->data,&b->capacity,b->size+size,1))return 0;
    if(size)memcpy(b->data+b->size,src,size);
    b->size+=size;return 1;
}
static inline int vc_buffer_put(vc_buffer *b,uint64_t value,unsigned bytes) {
    if(bytes>8)return 0;
    uint8_t data[8];for(unsigned i=0;i<bytes;++i)data[i]=(uint8_t)(value>>(8*i));
    return vc_buffer_append(b,data,bytes);
}
#endif
