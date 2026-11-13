#ifndef VC_NG_SEARCH_H
#define VC_NG_SEARCH_H
#include <stdint.h>
#include <stddef.h>
typedef struct {
    uint8_t *empty, *mask;
    size_t count, blocks;
} vc_ng_occupancy;
int vc_ng_scan(vc_ng_occupancy*,const uint8_t*,unsigned);
/* Stable predictor-ID tie breaking, shared by all screens. */
void vc_ng_top_two(const double[12],unsigned[2]);
typedef struct {
    uint16_t threshold[15];
    uint32_t histogram[16][256];
    double cost[16][256];
} vc_ng_distribution;
/* tune: 0 quantiles, 1 coordinate refinement, 2 global DP64. */
int vc_ng_distribution_fit(vc_ng_distribution*,const uint8_t*,const uint16_t*,
    size_t,const uint8_t *mask,unsigned tune);
void vc_ng_distribution_finish(vc_ng_distribution*);
double vc_ng_distribution_score(const vc_ng_distribution*);
/* Shared threshold fitting; optional histogram reuses fused activity work
 * when no masking is necessary. */
int vc_ng_search_thresholds(const uint8_t*,const uint16_t*,size_t,const uint8_t*,
    const uint32_t*,const uint8_t*,size_t,unsigned,unsigned,const uint32_t*,uint16_t[15]);
#endif
