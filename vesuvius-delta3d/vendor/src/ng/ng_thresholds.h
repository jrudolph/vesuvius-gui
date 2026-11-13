#ifndef VC_NG_THRESHOLDS_H
#define VC_NG_THRESHOLDS_H
#include <stdint.h>
#include <stddef.h>
int vc_ng_quantiles(const uint32_t histogram[2560],unsigned classes,uint16_t *thresholds);
int vc_ng_classify(const uint16_t*,size_t,const uint16_t*,unsigned classes,uint8_t*);
double vc_ng_histogram_cost(const uint32_t*,int table_bits);
/* Exact original DP64/coordinate threshold search. Empty classes are allowed.
 * groups>1 routes each sample through choices[block[i]], otherwise both may be NULL. */
int vc_ng_refine_thresholds(const uint8_t*,const uint16_t*,size_t,const uint16_t*,unsigned classes,
    const uint32_t *block,const uint8_t *choices,size_t choice_count,unsigned groups,
    int global,uint16_t *out);
#endif
