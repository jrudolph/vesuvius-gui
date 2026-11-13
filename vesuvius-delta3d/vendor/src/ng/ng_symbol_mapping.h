#pragma once
#include <stdint.h>
/* Shared exact Q8 rounding/sign convention for scalar NG and sparse proposals. */
static inline uint8_t ng_prediction_symbol(uint8_t value,uint16_t q){
    unsigned prediction=(uint16_t)(q+128u)>>8;
    int e=(int8_t)(uint8_t)(value-prediction);
    int s=((q>>7)&1u)?(int8_t)(uint8_t)(-e):e;
    return (uint8_t)(s>=0?2*s:-2*s-1);
}
