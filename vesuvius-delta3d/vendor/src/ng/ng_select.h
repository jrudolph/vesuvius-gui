#ifndef VC_NG_SELECT_H
#define VC_NG_SELECT_H
#include <stdint.h>
#include <stddef.h>
int vc_ng_select(const uint8_t*const*,const uint8_t*const*,const double*const*,unsigned,
    const uint32_t*,size_t,uint8_t*,size_t);
#endif
