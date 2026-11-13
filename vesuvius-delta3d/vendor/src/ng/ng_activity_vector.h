NG_AATTR static inline NG_AV NG_ANAME(const uint8_t*e,size_t ri){
  const uint8_t*n=e+ri-VCN_LC_RP,*u=e+ri-VCN_LC_SP;
#define VCN_ACT_LOAD(q) NG_ALOAD(q)
  NG_AV a=NG_ASHIFT(NG_AADD(VCN_ACT_LOAD(n),VCN_ACT_LOAD(u)),1);
  a=NG_AADD(a,NG_AADD(VCN_ACT_LOAD(n-1),VCN_ACT_LOAD(n+1)));
  a=NG_AADD(a,NG_AADD(VCN_ACT_LOAD(u-1),VCN_ACT_LOAD(u+1)));
  a=NG_AADD(a,NG_AADD(VCN_ACT_LOAD(u-VCN_LC_RP),VCN_ACT_LOAD(u+VCN_LC_RP)));
  a=NG_AADD(a,NG_AADD(VCN_ACT_LOAD(n-VCN_LC_RP),VCN_ACT_LOAD(u-VCN_LC_SP)));
#undef VCN_ACT_LOAD
  return a;
}
#undef NG_AATTR
#undef NG_AV
#undef NG_ANAME
#undef NG_ALOAD
#undef NG_AADD
#undef NG_ASHIFT
