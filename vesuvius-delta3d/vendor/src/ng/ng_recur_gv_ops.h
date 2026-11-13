/* Portable recurrence operations (GCC/Clang vector extensions): 8 rows per vector. */
#define NG_RATTR __attribute__((always_inline))
#define NG_RV ng_gv
#define NG_RHALVES 4
#define NG_RSTEP 8
#define NG_RSET(X) ((ng_gv){0}+(uint16_t)(X))
#define NG_RLOAD(P) (*(const ng_gvu*)(const void*)(P))
#define NG_RSTORE(P,V) (*(ng_gvu*)(void*)(P)=(V))
#define NG_RAND(A,B) ((A)&(B))
#define NG_REQ(A,B) ((ng_gv)((A)==(B)))
#define NG_RBLEND(A,B,M) ng_gv_sel(M,A,B)
#define NG_RMASK(V) (ng_gv_all(V)?-1:0)
#define NG_RADD(A,B) ((A)+(B))
#define NG_RMUL(A,B) ((A)*(B))
#define NG_RMIN ng_gv_min
#define NG_RMAX ng_gv_max
#define NG_RSRL(A,N) ((A)>>(N))
#define NG_RSRA(A,N) ((ng_gv)((ng_gs)(A)>>(N)))
#define NG_RSLL(A,N) ((A)<<(N))
#define NG_RSUB(A,B) ((A)-(B))
#define NG_RXOR(A,B) ((A)^(B))
#define NG_ROUT(P,V) ng_gv_out(P,V)
