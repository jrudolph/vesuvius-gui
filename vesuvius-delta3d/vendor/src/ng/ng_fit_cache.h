#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
/* One source/configuration per session, owned by one encode call. No global
 * cache, locking, persistent storage, or cross-chunk state. */
typedef struct {
    int ncp,ncol,nt;
    size_t capacity,stored,count;
    int16_t* columns;
    size_t* positions;
    int64_t* gram;
} ng_fit_bank_cache;
typedef struct {
    ng_fit_bank_cache banks[2];
    uint8_t* padded;
    int padded_ready;
    const uint8_t* source;
    int config[6],dims[3];
} ng_fit_session;
/* Original deterministic phase; unchanged sampling positions. */
static inline unsigned ng_fit_sample_phase(unsigned y,unsigned z,unsigned stride){
    return (y*7+z*3)%stride;
}
/* Preserve the unmasked sparse sample budget, redistributed deterministically
 * over eligible mask entries. Stratify by rank, without replacement. An empty
 * or undersized bank uses all available targets; no samples cross fit banks.
 * NULL means retain the original traversal (dense or no excluded targets).
 * This selects targets only: source/predictor-neighbor values never change. */
static inline uint8_t* ng_fit_nonzero_sample_mask(const uint8_t*mask,int X,int Y,int Z,
 int xlo,int zlo,int zhi,int stride,uint64_t stats[3]){
    stats[0]=stats[1]=stats[2]=0;
    if(!mask||stride<=1)return NULL;
    for(int z=zlo;z<zhi;z++)for(int y=0;y<Y;y++){
        int start=xlo+(int)ng_fit_sample_phase(y,z,stride);
        if(start<X)stats[0]+=(X-1-start)/stride+1;
        for(int x=xlo;x<X;x++)stats[1]+=mask[((size_t)z*Y+y)*X+x]!=0;
    }
    if(stats[1]==(uint64_t)(zhi-zlo)*Y*(X-xlo))return NULL;
    uint64_t n=stats[0]<stats[1]?stats[0]:stats[1];
    uint8_t* selected=(uint8_t*)calloc((size_t)X*Y*Z,1);
    if(!selected)return NULL;
    if(!n)return selected;
    uint64_t rank=0,k=0,next=stats[1]/(2*n);
    for(int z=zlo;z<zhi;z++)for(int y=0;y<Y;y++)for(int x=xlo;x<X;x++){
        size_t i=((size_t)z*Y+y)*X+x;if(!mask[i])continue;
        if(rank==next){selected[i]=1;k++;if(k<n)next=((2*k+1)*stats[1])/(2*n);}
        rank++;
    }
    stats[2]=k;return selected;
}
static inline void ng_fit_bank_clear(ng_fit_bank_cache*c){
    free(c->columns);free(c->positions);free(c->gram);memset(c,0,sizeof(*c));
}
static inline int64_t ng_fit_dot_c(const int16_t*a,const int16_t*b,size_t n){
    int64_t v=0;for(size_t i=0;i<n;i++)v+=(int32_t)a[i]*b[i];return v;
}
