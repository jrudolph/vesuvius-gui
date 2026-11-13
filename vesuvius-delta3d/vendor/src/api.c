#include <vc_delta3d/codec.h>
#include "legacy.h"
#include "delta_filter.h"
#include <stdlib.h>
#include <string.h>
#ifdef VC_DELTA3D_MODERN
#include <vc_delta3d/indexed.h>
#include "ng/ng_frame.h"
#endif
static const uint8_t magic[4]={'D','3','D','1'};
vc_encode_options vc_default_options(void) {
    vc_encode_options opt={{0,0,0},0,0,1,1,VC_FAST}; return opt;
}
void vc_free(void *p) { free(p); }
static size_t frame_bytes(const uint8_t *src,size_t size) {
#ifdef VC_DELTA3D_MODERN
    unsigned shape[3];
    if(vc_ng_frame_shape(src,size,shape))return (size_t)shape[0]*shape[1]*shape[2];
#else
    (void)src; (void)size;
#endif
    return 0;
}
int vc_compress(const void *data,size_t size,const vc_encode_options *opt,uint8_t **out,size_t *out_size) {
    if(!out || !out_size) return 0;
    *out=NULL; *out_size=0;
    if(!data || !opt || (opt->element_size!=1 && opt->element_size!=2) ||
       opt->layout>1 || !opt->quantization || opt->quantization>255 || opt->channels>INT32_MAX ||
       !vc_preset_valid(opt->preset)) return 0;
    size_t expected=opt->element_size; int aligned=1;
    for(unsigned a=0;a<3;++a) {
        uint32_t d=opt->shape[a]; if(!d || expected>SIZE_MAX/d) return 0;
        expected*=d; aligned&=d%32==0;
    }
    uint32_t count=opt->channels?opt->channels:1;
    if(expected>SIZE_MAX/count || expected*count!=size) return 0;
    const uint8_t *src=data;
    if(opt->preset==VC_DELTA || opt->element_size!=1 || opt->quantization!=1 || !aligned)
        return vc_legacy_encode(src,size,opt->shape,opt->element_size,opt->quantization,opt->channels,opt->layout,out,out_size);
#ifdef VC_DELTA3D_MODERN
    if(opt->channels || opt->shape[0]!=opt->shape[1] || opt->shape[1]!=opt->shape[2] || opt->shape[2]>256)
        return vc_indexed_encode(src,size,opt->shape,count,opt->layout,opt->preset,NULL,0,out,out_size);
    if(size>(SIZE_MAX-131072)/4) return 0;
    size_t cap=size*4+131072,bytes=0; uint8_t *result=malloc(cap); if(!result) return 0;
    if(vc_ng_preset_encode(src,opt->shape[2],opt->preset,result,cap,&bytes) || bytes>cap) { free(result); return 0; }
    *out=result; *out_size=bytes; return 1;
#else
    return 0;
#endif
}
int vc_frame_inspect(const uint8_t *src,size_t size,vc_frame_info *out) {
    if(!out) return 0;
    memset(out,0,sizeof(*out)); out->delta_mask=-1;
#ifdef VC_DELTA3D_MODERN
    if(vc_indexed_header_size(src,size)) {
        vc_chunk_index index;
        if(!vc_index_read(src,size,size,&index)) return 0;
        out->decoded_bytes=(size_t)index.decoded_bytes; out->channels=index.channels; out->quantization=1;
        vc_index_free(&index); return 1;
    }
#endif
    size_t bytes=frame_bytes(src,size);
    if(bytes) { out->decoded_bytes=bytes; out->channels=1; out->quantization=1; return 1; }
    vc_legacy_info info;
    if(!vc_legacy_probe(src,size,magic,&info)) return 0;
    out->decoded_bytes=info.bytes; out->channels=info.channels; out->quantization=info.quantization;
    out->legacy_scalar=info.version==1;
    if(out->legacy_scalar && info.has_mask) out->delta_mask=(int)info.mask;
    return 1;
}
int vc_decompress_with_magic(const uint8_t *src,size_t size,const uint8_t expected[4],void *out,size_t count) {
    if(!src || !expected || !out) return 0;
#ifdef VC_DELTA3D_MODERN
    if(!memcmp(expected,magic,4)) {
        if(vc_indexed_header_size(src,size)) return vc_indexed_decode(src,size,out,count);
        if(size>=4 && !memcmp(src,"NCF1",4))
            return vc_ng_preset_decode(src,size,out,count)==0;
    }
#endif
    return vc_legacy_decode(src,size,expected,out,count);
}
int vc_decompress(const uint8_t *src,size_t size,void *out,size_t count) {
    return vc_decompress_with_magic(src,size,magic,out,count);
}
int vc_quantize(void *data,size_t size,unsigned elem,unsigned width) {
    if(!data || (elem!=1 && elem!=2) || size%elem || !width || width>255) return 0;
    if(width>1) {
        if(elem==1) vc_delta_quantize_u8(data,size,(int)width);
        else vc_delta_quantize_u16(data,size/2,(int)width);
    }
    return 1;
}
