#ifndef VC_NG_FIT_H
#define VC_NG_FIT_H
#include <stdint.h>
#include <stddef.h>
typedef struct {
    uint8_t *params, *symbols;
    size_t param_size;
    int32_t weights[2];
    uint16_t *base;
} vc_ng_fit;
void vc_ng_fit_free(vc_ng_fit *fit);
/* Fit one shared NG model with the fixed predictor as its extra input
 * (NULL selects a plain NG fit).
 * Optional activity output fuses the final causal activity pass. */
int vc_ng_fit_input(vc_ng_fit*,const uint8_t*,const int*,const uint8_t *fixed,
    const uint8_t *mask,void *session,uint16_t *activity,uint32_t *histogram);
#endif
