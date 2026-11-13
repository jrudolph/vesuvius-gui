/* Block modes: several learned models, each fitted on the blocks that select it.
 *
 * One model is fitted on the whole chunk and the 4^3 blocks are split by the
 * size of its residuals. Each round then (1) fits every model on its own
 * blocks and (2) moves every block to the model that codes it most cheaply.
 * The models share the neighbourhood and differ in their weights; decoding
 * costs one more sum over the earlier rows per extra model. */
#include "ng_fit.h"
#include "ng_frame.h"
#include "ng_predict.h"
#include "ng_search.h"
#include "ng_thresholds.h"
#include "residual_bridge.h"
#include <stdlib.h>
#include <string.h>

enum { MAX_MODES = 4, MIN_BLOCKS = 64 };

typedef struct {
    unsigned d, modes;
    int cfg[6];
    size_t n, blocks;
    const uint8_t *src;
    vc_ng_occupancy zero;
    vc_ng_fit fits[MAX_MODES];
    uint8_t *candidate[MAX_MODES], *raw, *choices, *canonical, *symbols, *classes, *mask;
    uint16_t *activity;
    uint32_t *block;
    vc_ng_distribution stats[MAX_MODES];
} split_state;

/* Symbols of every model (or only of the selected ones) under the activity
 * that the current block choices produce. */
static int evaluate(split_state *s, int selected_only)
{
    const uint8_t *params[MAX_MODES];
    size_t sizes[MAX_MODES];
    const uint16_t *bases[MAX_MODES];
    uint8_t *outputs[12 + 12 * MAX_MODES] = {0};
    for (unsigned k = 0; k < s->modes; ++k) {
        params[k] = s->fits[k].params;
        sizes[k] = s->fits[k].param_size;
        bases[k] = s->fits[k].base;
        outputs[12 + 12 * k] = s->candidate[k];
    }
    outputs[0] = s->raw;
    for (size_t b = 0; b < s->blocks; ++b)
        s->canonical[b] = s->zero.count && s->zero.empty[b] ? 0 : (uint8_t)(12 + 12 * s->choices[b]);
    return (selected_only ? ng_bank_selected_symbols : ng_bank_symbols)(s->modes, params, sizes, bases,
        s->canonical, s->src, outputs, s->symbols, s->activity) == 0;
}

/* Voxels of the non-empty blocks that currently choose `mode`; returns the block count. */
static size_t mode_mask(split_state *s, unsigned mode)
{
    size_t count = 0;
    for (size_t b = 0; b < s->blocks; ++b)
        count += s->choices[b] == mode && !(s->zero.count && s->zero.empty[b]);
    for (size_t i = 0; i < s->n; ++i) {
        size_t b = s->block[i];
        s->mask[i] = s->choices[b] == mode && !(s->zero.count && s->zero.empty[b]);
    }
    return count;
}

/* Split the non-empty blocks into equally sized groups by residual size. */
static int initial_split(split_state *s, const uint8_t *symbols)
{
    uint32_t *energy = calloc(s->blocks, sizeof(*energy));
    uint32_t histogram[64 * 128 + 1] = {0};
    if (!energy) return 0;
    for (size_t i = 0; i < s->n; ++i) energy[s->block[i]] += (symbols[i] + 1u) >> 1;
    size_t occupied = 0;
    for (size_t b = 0; b < s->blocks; ++b)
        if (!(s->zero.count && s->zero.empty[b])) { ++histogram[energy[b]]; ++occupied; }
    uint32_t cut[MAX_MODES];
    size_t seen = 0;
    unsigned next = 1;
    for (uint32_t e = 0; e <= 64 * 128 && next < s->modes; ++e) {
        seen += histogram[e];
        while (next < s->modes && seen * s->modes >= occupied * next) cut[next++ - 1] = e;
    }
    while (next < s->modes) cut[next++ - 1] = 64 * 128;
    for (size_t b = 0; b < s->blocks; ++b) {
        unsigned mode = 0;
        while (mode + 1 < s->modes && energy[b] > cut[mode]) ++mode;
        s->choices[b] = (uint8_t)mode;
    }
    free(energy);
    return 1;
}

static int fit_modes(split_state *s)
{
    for (unsigned k = 0; k < s->modes; ++k) {
        /* A mode that lost nearly all its blocks keeps its previous model. */
        if (mode_mask(s, k) < MIN_BLOCKS && s->fits[k].params) continue;
        vc_ng_fit fit;
        if (!vc_ng_fit_input(&fit, s->src, s->cfg, NULL, s->mask, NULL, NULL, NULL)) return 0;
        vc_ng_fit_free(&s->fits[k]);
        s->fits[k] = fit;
    }
    return 1;
}

/* Symbol statistics of each mode on the blocks that choose it. */
static int mode_statistics(split_state *s)
{
    for (unsigned k = 0; k < s->modes; ++k) {
        mode_mask(s, k);
        if (!vc_ng_distribution_fit(&s->stats[k], s->candidate[k], s->activity, s->n, s->mask, 1)) return 0;
    }
    return 1;
}

static int select_blocks(split_state *s)
{
    double *cost = calloc(s->blocks * s->modes, sizeof(*cost));
    if (!cost) return 0;
    for (unsigned k = 0; k < s->modes; ++k) {
        if (!vc_ng_classify(s->activity, s->n, s->stats[k].threshold, 16, s->classes)) { free(cost); return 0; }
        const uint8_t *symbols = s->candidate[k];
        for (size_t i = 0; i < s->n; ++i)
            cost[s->modes * s->block[i] + k] += s->stats[k].cost[s->classes[i]][symbols[i]];
    }
    for (size_t b = 0; b < s->blocks; ++b) {
        unsigned best = 0;
        for (unsigned k = 1; k < s->modes; ++k)
            if (cost[s->modes * b + k] < cost[s->modes * b + best]) best = k;
        s->choices[b] = (uint8_t)(s->zero.count && s->zero.empty[b] ? 0 : best);
    }
    free(cost);
    return 1;
}

int vc_ng_split_encode(const uint8_t *src, unsigned d, const vc_ng_tuning *tuning,
    uint8_t *out, size_t cap, size_t *written)
{
    if (!src || !tuning || !out || !written || !ng_geometry_valid(d, d, d)) return -1;
    const unsigned modes = tuning->models, rounds = tuning->rounds;
    if (modes < 2 || modes > MAX_MODES || !rounds) return -1;
    const ng_geometry saved = ng_current_geometry;
    const unsigned old_edge = ng_choice_edge;
    ng_current_geometry = (ng_geometry){(int)d, (int)d, (int)d};
    ng_choice_edge = 4;
    split_state *s = calloc(1, sizeof(*s));
    uint8_t *frame = NULL;
    size_t frame_size = 0;
    uint16_t t[15];
    int rc = -1;
    if (!s) goto done;
    s->src = src; s->d = d; s->n = (size_t)d * d * d; s->blocks = s->n / 64; s->modes = modes;
    memcpy(s->cfg, (int[6]){3, 3, tuning->reach, 16, tuning->taps, tuning->stride}, sizeof(s->cfg));
    if (!vc_ng_scan(&s->zero, src, d)) goto done;
    /* The zero predictor takes one of the four entropy banks. */
    if (s->zero.count && s->modes > 3) s->modes = 3;
    if (s->zero.count == s->blocks) {
        if (!vc_ng_frame_all_zero(d, &frame, &frame_size)) goto done;
        goto finish;
    }
    if (s->blocks - s->zero.count < (size_t)MIN_BLOCKS * s->modes) goto done;
    s->choices = calloc(s->blocks, 1); s->canonical = malloc(s->blocks);
    s->symbols = malloc(s->n); s->classes = malloc(s->n); s->mask = malloc(s->n);
    s->activity = malloc(s->n * sizeof(*s->activity)); s->block = malloc(s->n * sizeof(*s->block));
    if (!s->choices || !s->canonical || !s->symbols || !s->classes || !s->mask || !s->activity || !s->block) goto done;
    for (unsigned k = 0; k < s->modes; ++k) if (!(s->candidate[k] = malloc(s->n))) goto done;
    if (s->zero.count) {
        uint8_t *fixed[12] = {0};
        if (!(s->raw = fixed[0] = malloc(s->n))) goto done;
        vc_ng_fixed_candidates(src, fixed, d, 1u);
    }
    for (unsigned z = 0; z < d; ++z) for (unsigned y = 0; y < d; ++y) for (unsigned x = 0; x < d; ++x)
        s->block[((size_t)z * d + y) * d + x] = ((z / 4) * (d / 4) + y / 4) * (d / 4) + x / 4;
    {
        vc_ng_fit whole;
        if (!vc_ng_fit_input(&whole, src, s->cfg, NULL, s->zero.mask, NULL, NULL, NULL)) goto done;
        int ok = initial_split(s, whole.symbols);
        vc_ng_fit_free(&whole);
        if (!ok) goto done;
    }
    for (unsigned round = 0; round < rounds; ++round)
        if (!fit_modes(s) || !evaluate(s, 0) || !mode_statistics(s) || !select_blocks(s)) goto done;
    if (!evaluate(s, 1) ||
        !vc_ng_search_thresholds(s->symbols, s->activity, s->n, s->zero.mask, s->block, s->choices, s->blocks,
            s->modes, 1, NULL, t) ||
        !vc_ng_classify(s->activity, s->n, t, 16, s->classes)) goto done;
    {
        vc_ng_model models[MAX_MODES];
        vc_ng_choice menu[MAX_MODES];
        for (unsigned k = 0; k < s->modes; ++k) {
            models[k] = (vc_ng_model){s->fits[k].params, s->fits[k].param_size};
            menu[k] = (vc_ng_choice){0, k, k, {0, 0}};
        }
        if (!vc_ng_frame_encode_zeros(d, models, s->modes, menu, s->modes, s->choices, s->symbols, t, s->classes,
            s->zero.count ? s->zero.empty : NULL, &frame, &frame_size)) goto done;
    }
finish:
    if (frame_size > cap) goto done;
    memcpy(out, frame, frame_size); *written = frame_size; rc = 0;
done:
    if (s) {
        for (unsigned k = 0; k < MAX_MODES; ++k) { vc_ng_fit_free(&s->fits[k]); free(s->candidate[k]); }
        free(s->zero.empty); free(s->zero.mask); free(s->raw); free(s->choices); free(s->canonical);
        free(s->symbols); free(s->classes); free(s->mask); free(s->activity); free(s->block); free(s);
    }
    free(frame); ng_current_geometry = saved; ng_choice_edge = old_edge;
    return rc;
}
