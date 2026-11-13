/* Fast preset: the original distributed screen, one full-chunk fit, and DP64
 * entropy thresholds. Candidate ordering and strict comparisons are deliberate:
 * they preserve the existing format bytes, including tied screening choices. */
#include "ng_fit.h"
#include "ng_frame.h"
#include "ng_predict.h"
#include "ng_screen.h"
#include "ng_search.h"
#include "ng_thresholds.h"
#include "residual_bridge.h"
#include <stdlib.h>
#include <string.h>


static int statistics(const uint8_t *symbols, const uint8_t *mask, size_t n,
                      uint16_t *activity, uint8_t *classes, uint16_t thresholds[15],
                      uint32_t counts[16][256], uint32_t histogram[2560])
{
    if (ng_activity_symbol_histogram(symbols, activity, histogram)) return 0;
    if (mask) {
        memset(histogram, 0, 2560 * sizeof(*histogram));
        for (size_t i = 0; i < n; ++i) if (mask[i]) ++histogram[activity[i]];
    }
    if (!vc_ng_quantiles(histogram, 16, thresholds) ||
        !vc_ng_classify(activity, n, thresholds, 16, classes)) return 0;
    memset(counts, 0, 16 * 256 * sizeof(uint32_t));
    for (size_t i = 0; i < n; ++i)
        if (!mask || mask[i]) ++counts[classes[i]][symbols[i]];
    return 1;
}

static double uniform_cost(uint32_t counts[16][256], unsigned scale, size_t blocks)
{
    uint32_t selectors[256] = {0};
    selectors[0] = (uint32_t)(blocks * scale);
    double cost = vc_ng_histogram_cost(selectors,12);
    for (unsigned k = 0; k < 16; ++k) {
        uint32_t scaled[256];
        for (unsigned j = 0; j < 256; ++j) scaled[j] = counts[k][j] * scale;
        cost += vc_ng_histogram_cost(scaled,12);
    }
    return cost / 8.;
}

static double metadata_cost(const vc_ng_fit *fit)
{
    vc_ng_model m={fit?fit->params:NULL,fit?fit->param_size:0};
    vc_ng_choice c={1,fit?0:255,0,{0,0}};
    if(fit)memcpy(c.weights,fit->weights,sizeof(c.weights));
    return vc_ng_model_cost(&m,fit?1:0,&c,1);
}

int vc_ng_fast_encode(const uint8_t *src, unsigned d, const vc_ng_tuning *tuning, uint8_t *out, size_t cap, size_t *written)
{
    if (!src || !tuning || !out || !written || !ng_geometry_valid(d,d,d)) return -1;
    const ng_geometry saved_geometry = ng_current_geometry;
    const unsigned saved_edge = ng_choice_edge;
    const size_t full_n = (size_t)d * d * d;
    const unsigned sd = d > 32 ? 32 : d;
    const size_t n = (size_t)sd * sd * sd;
    const unsigned scale = (unsigned)(full_n / n);
    vc_ng_occupancy full = {0}, screen = {0};
    vc_ng_fit fits[2] = {{0}}, final_fit = {0};
    uint8_t *proxy = NULL, *candidate[12] = {0}, *classes = NULL, *symbols = NULL;
    uint8_t *selected = NULL, *frame = NULL;
    uint16_t *activity = NULL;
    void *session = NULL;
    size_t frame_size = 0;
    uint32_t counts[16][256], histogram[2560];
    uint16_t thresholds[15], final_thresholds[15];
    double screening[12], costs[5];
    unsigned top[2] = {0,0}, options[5], option_count = 1, chosen = 0;
    int rc = -1, activity_ready = 0;
    ng_current_geometry = (ng_geometry){(int)d,(int)d,(int)d};
    ng_choice_edge = 4;
    if (!vc_ng_scan(&full,src,d)) goto done;
    if (full.count == full.blocks) {
        if (!vc_ng_frame_all_zero(d,&frame,&frame_size)) goto done;
        goto finish;
    }
    selected = calloc(full.blocks,1);
    if (!selected) goto done;
    const uint8_t *sample = src;
    if (d > 32) {
        proxy = malloc(n);
        if (!proxy || vc_ng_screen_proxy(src,d,full.count?full.empty:NULL,proxy)) goto done;
        sample = proxy;
    }
    ng_current_geometry = (ng_geometry){(int)sd,(int)sd,(int)sd};
    if (full.count && !vc_ng_scan(&screen,sample,sd)) goto done;
    classes = malloc(full_n);
    activity = malloc(full_n * sizeof(*activity));
    if (!classes || !activity) goto done;
    for (unsigned m = 0; m < 12; ++m) {
        candidate[m] = malloc(n);
        if (!candidate[m]) goto done;
    }
    vc_ng_fixed_candidates(sample,candidate,sd,4095);
    for (unsigned m = 0; m < 12; ++m) {
        if (!statistics(candidate[m],screen.mask,n,activity,classes,thresholds,counts,histogram)) goto done;
        double cost = 0;
        for (unsigned k = 0; k < 16; ++k) cost += vc_ng_histogram_cost(counts[k],12);
        screening[m] = cost;
    }
    /* Stable ascending predictor ID resolves equal histogram costs. */
    vc_ng_top_two(screening,top);
    int cfg[6] = {3,3,tuning->reach,16,tuning->taps,d>32?4:128};
    session = ng_fit_session_create();
    if (!session) goto done;
    for (unsigned k = 0; k < 2; ++k)
        if (!vc_ng_fit_input(&fits[k],sample,cfg,candidate[top[k]],screen.mask,
                            screen.count?NULL:session,NULL,NULL)) goto done;
    ng_fit_session_destroy(session);session = NULL;
    options[0] = 0;
    for (unsigned k = 0; k < 2; ++k) {
        unsigned ids[2] = {top[k],12+12*k+top[k]};
        for (unsigned j = 0; j < 2; ++j) {
            unsigned p = 0;
            while (p < option_count && options[p] != ids[j]) ++p;
            if (p == option_count) options[option_count++] = ids[j];
        }
    }
    for (unsigned j = 0; j < option_count; ++j) {
        unsigned id = options[j];
        const vc_ng_fit *fit = id >= 12 ? &fits[(id-12)/12] : NULL;
        const uint8_t *s = fit ? fit->symbols : candidate[id];
        if (!statistics(s,screen.mask,n,activity,classes,thresholds,counts,histogram)) goto done;
        double metadata = metadata_cost(fit);
        if (metadata < 0) goto done;
        costs[j] = uniform_cost(counts,scale,n/64) + metadata;
        if (costs[j] < costs[chosen]) chosen = j;
    }
    chosen = options[chosen];
    unsigned predictor = chosen % 12;
    const int learned = chosen >= 12;
    if (d > 32) {
        ng_current_geometry = (ng_geometry){(int)d,(int)d,(int)d};
        for (unsigned m = 0; m < 12; ++m) {free(candidate[m]);candidate[m]=NULL;}
        unsigned fixed_mask = (1u<<predictor) | (full.count?1u:0u);
        for (unsigned m = 0; m < 12; ++m) if (fixed_mask & (1u<<m)) {
            candidate[m] = malloc(full_n);
            if (!candidate[m]) goto done;
        }
        vc_ng_fixed_candidates(src,candidate,d,fixed_mask);
        if (learned) {
            cfg[5] = tuning->stride;
            if (!vc_ng_fit_input(&final_fit,src,cfg,candidate[predictor],full.mask,NULL,
                                full.count?NULL:activity,full.count?NULL:histogram)) goto done;
            activity_ready = !full.count;
        }
    } else if (learned) {
        unsigned k = (chosen-12)/12;
        final_fit = fits[k];memset(&fits[k],0,sizeof(fits[k]));
    }
    symbols = malloc(full_n);
    if (!symbols) goto done;
    memcpy(symbols,learned?final_fit.symbols:candidate[predictor],full_n);
    if (!activity_ready && ng_activity_symbol_histogram(symbols,activity,histogram)) goto done;
    unsigned seed = learned ? 12+predictor : predictor;
    memset(selected,seed,full.blocks);
    if (full.count) {
        for (size_t b = 0; b < full.blocks; ++b) if (full.empty[b]) selected[b] = 0;
        if (learned) {
            const uint8_t *params[]={final_fit.params};size_t sizes[]={final_fit.param_size};const uint16_t *bases[]={final_fit.base};
            uint8_t *outputs[24] = {0};
            for (unsigned m = 0; m < 12; ++m) {
                outputs[m]=candidate[m];
            }
            outputs[12+predictor]=final_fit.symbols;
            if (ng_bank_selected_symbols(1,params,sizes,bases,selected,src,outputs,symbols,activity)) goto done;
        } else {
            for (size_t i = 0; i < full_n; ++i) if (!full.mask[i]) symbols[i]=0;
            if (ng_activity_symbol_values(symbols,activity)) goto done;
        }
    }
    if (!vc_ng_search_thresholds(symbols,activity,full_n,full.mask,NULL,NULL,0,1,2,histogram,final_thresholds) ||
        !vc_ng_classify(activity,full_n,final_thresholds,16,classes)) goto done;
    vc_ng_model model = {final_fit.params,final_fit.param_size};
    vc_ng_choice menu[3];unsigned remap[24],count=0;
    for (unsigned j=0;j<24;++j) remap[j]=255;
    for (size_t b=0;b<full.blocks;++b) {
        unsigned id=selected[b];
        if (remap[id]==255) {
            unsigned m=id>=12?0:255;
            menu[count]=(vc_ng_choice){id%12,m,0,{0,0}};
            if (m!=255 && id%12) memcpy(menu[count].weights,final_fit.weights,sizeof(final_fit.weights));
            remap[id]=count++;
        }
        selected[b]=(uint8_t)remap[id];
    }
    if (full.count) {
        menu[count]=(vc_ng_choice){12,255,1,{0,0}};
        for (size_t b=0;b<full.blocks;++b) if (full.empty[b]) selected[b]=(uint8_t)count;
        ++count;
    }
    if (!vc_ng_frame_encode(d,4,1,&model,learned?1:0,menu,count,selected,symbols,final_thresholds,16,classes,&frame,&frame_size)) goto done;
finish:
    if (frame_size>cap) goto done;
    memcpy(out,frame,frame_size);*written=frame_size;rc=0;
done:
    if (session) ng_fit_session_destroy(session);
    for (unsigned k=0;k<2;++k) vc_ng_fit_free(&fits[k]);
    vc_ng_fit_free(&final_fit);
    for (unsigned m=0;m<12;++m) free(candidate[m]);
    free(full.empty);free(full.mask);free(screen.empty);free(screen.mask);
    free(proxy);free(classes);free(symbols);free(activity);free(selected);
    free(frame);
    ng_current_geometry=saved_geometry;ng_choice_edge=saved_edge;
    return rc;
}
