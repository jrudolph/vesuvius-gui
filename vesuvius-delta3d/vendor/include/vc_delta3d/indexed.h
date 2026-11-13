#ifndef VC_DELTA3D_INDEXED_H
#define VC_DELTA3D_INDEXED_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* One leaf representation for both cubic and rectangular shapes. */
enum { VC_LEAF_CHOICE=16 };
typedef struct {
    uint32_t channel, origin[3], shape[3];
    unsigned codec;
    uint64_t offset, length;
} vc_indexed_leaf;
typedef struct {
    uint32_t shape[3], channels, header_bytes;
    unsigned layout;
    uint64_t encoded_bytes, decoded_bytes;
    vc_indexed_leaf *leaves;
    size_t leaf_count;
} vc_chunk_index;
/* Preorder partition: axes 0/1/2 spatial, 3 channel, 4 leaf;
 * 5/6/7/8 repeat Z/Y/X/channel. Repeat cut is the repetition count (>=2),
 * followed by ONE subtree reused in equally sized, disjoint regions.
 * Spatial cuts must be positive multiples of 32. An empty plan selects the
 * canonical channel/cubic partition. Explicit encoder leaves must be cubic;
 * decoding still supports rectangular leaves. */
typedef struct { unsigned axis; uint32_t cut; } vc_partition_node;
int vc_indexed_encode(const uint8_t*, size_t, const uint32_t shape[3], uint32_t channels,
                      unsigned layout, unsigned preset, const vc_partition_node*, size_t nodes,
                      uint8_t**, size_t*);
/* Read >=8 bytes, then fetch the remaining header (no payload required). */
uint32_t vc_indexed_header_size(const uint8_t*, size_t);
/* On success the index owns its leaves. Release with vc_index_free. */
int vc_index_read(const uint8_t*, size_t, uint64_t object_bytes, vc_chunk_index*);
void vc_index_free(vc_chunk_index*);
int vc_indexed_leaf_decode(const uint8_t*, size_t, const vc_indexed_leaf*, uint8_t*, size_t);
int vc_indexed_decode(const uint8_t*, size_t, uint8_t*, size_t);
#ifdef __cplusplus
}
#endif
#endif
