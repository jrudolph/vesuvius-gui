#ifndef VC_NG_SELECTOR_BITS_H
#define VC_NG_SELECTOR_BITS_H
#include <stddef.h>
#include <stdint.h>
#include <string.h>
/* Shared current selector bit layout. Callers validate sizes and IDs. */
static inline unsigned vc_selector_bits(unsigned choices) {
    unsigned bits=0;while((1u<<bits)<choices)++bits;return bits;
}
static inline void vc_selector_pack(const uint8_t *src,size_t count,unsigned bits,uint8_t *dst) {
    memset(dst,0,(count*bits+7)/8);
    for(size_t i=0;i<count;++i)for(unsigned b=0;b<bits;++b)
        dst[(i*bits+b)/8]|=((src[i]>>b)&1u)<<((i*bits+b)%8);
}
static inline void vc_selector_unpack(const uint8_t *src,size_t count,unsigned bits,uint8_t *dst) {
    memset(dst,0,count);
    for(size_t i=0;i<count;++i)for(unsigned b=0;b<bits;++b)
        dst[i]|=((src[(i*bits+b)/8]>>((i*bits+b)%8))&1u)<<b;
}
#endif
