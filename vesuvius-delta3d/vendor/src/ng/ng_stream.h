#ifndef VC_NG_STREAM_H
#define VC_NG_STREAM_H
#include <stdint.h>
#include <stddef.h>
int vc_ng_stream_encode(const uint8_t*,size_t,uint8_t**,size_t*);
int vc_ng_stream_decode(const uint8_t*,size_t,uint8_t*,size_t);
int vc_ng_stream_fixed_encode(const uint8_t*,size_t,unsigned,uint8_t**,size_t*);
int vc_ng_stream_fixed_decode(const uint8_t*,size_t,unsigned,uint8_t*,size_t);
#endif
