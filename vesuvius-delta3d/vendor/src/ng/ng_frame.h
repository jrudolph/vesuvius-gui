#ifndef VC_NG_FRAME_H
#define VC_NG_FRAME_H
#include <stdint.h>
#include <stddef.h>
/* Encoder settings of the single-pass policies. `reach` is the window radius
 * two slices back (1: 3x3, 3: 7x7); `taps` is how many candidate neighbours
 * the model keeps; the full-chunk fit samples every `stride`-th voxel of a
 * row; `intercept` is the fit's intercept weight; `models` > 1 fits that many
 * models, each refitted on the blocks that select it for `rounds` rounds. */
typedef struct { int reach,taps,stride; double intercept; unsigned models,rounds; } vc_ng_tuning;
/* The encoder policies, shared by scalar/indexed dispatch. */
int vc_ng_fast_encode(const uint8_t*,unsigned,const vc_ng_tuning*,uint8_t*,size_t,size_t*);
int vc_ng_split_encode(const uint8_t*,unsigned,const vc_ng_tuning*,uint8_t*,size_t,size_t*);
/* Menu-search policies of 0.3.x: medium, and slow (0) or slower (1). */
int vc_ng_medium_encode(const uint8_t*,unsigned,uint8_t*,size_t,size_t*);
int vc_ng_slow_encode(const uint8_t*,unsigned,unsigned,uint8_t*,size_t,size_t*);
/* Any tuning (benchmarks); the presets call this with their fixed settings. */
int vc_ng_tuned_encode(const uint8_t*,unsigned,const vc_ng_tuning*,uint8_t*,size_t,size_t*);
/* `preset` is a public vc_preset other than delta. */
int vc_ng_preset_encode(const uint8_t*,unsigned,unsigned,uint8_t*,size_t,size_t*);
typedef struct { const uint8_t *data; size_t size; } vc_ng_model;
typedef struct { unsigned predictor,model,entropy; int32_t weights[2]; } vc_ng_choice;
/* Shared wire limits, independent of the current encoder's preset menus. */
static inline int vc_ng_edge_valid(unsigned edge) {
    return edge==2||edge==4||edge==8||edge==16||edge==32;
}
static inline int vc_ng_choice_valid(unsigned p,unsigned m,unsigned b,uint32_t w0,uint32_t w1,unsigned nm) {
    return p<=12&&(m==255||m<nm)&&b<16&&(p!=12||(m==255&&!w0&&!w1));
}
double vc_ng_model_cost(const vc_ng_model*,unsigned,const vc_ng_choice*,unsigned);
int vc_ng_frame_all_zero(unsigned,uint8_t**,size_t*);
/* Already fitted models/selected blocks: serialize native NCF1, without search.
 * Result belongs to caller (free). Zero predictor=12; no model=255. */
int vc_ng_frame_encode(unsigned dimension,unsigned edge,unsigned mode,
    const vc_ng_model*,unsigned models,const vc_ng_choice*,unsigned choices,
    const uint8_t *selectors,const uint8_t *symbols,const uint16_t *thresholds,
    unsigned classes,const uint8_t *class_ids,uint8_t **out,size_t *size);
int vc_ng_preset_decode(const uint8_t*,size_t,uint8_t*,size_t);
/* Indexed leaves omit magic, retain the header length; expected shape is ZYX.
 * Decode byte count must match exactly, as in the public C API. */
int vc_ng_leaf_decode(const uint8_t*,size_t,const uint32_t[3],uint8_t*,size_t);
/* Append the ordinary zero predictor for predetermined empty blocks. */
int vc_ng_frame_encode_zeros(unsigned,const vc_ng_model*,unsigned,const vc_ng_choice*,unsigned,
    const uint8_t*,const uint8_t*,const uint16_t*,const uint8_t*,const uint8_t*,uint8_t**,size_t*);
int vc_ng_frame_shape(const uint8_t*,size_t,unsigned shape[3]);
int vc_ng_frame_encode_shape(unsigned x,unsigned y,unsigned z,unsigned edge,unsigned mode,
    const vc_ng_model*,unsigned,const vc_ng_choice*,unsigned,const uint8_t*,const uint8_t*,
    const uint16_t*,unsigned,const uint8_t*,uint8_t**,size_t*);
#endif
