/* Included with vector operations defined: exact shared digit-pair arithmetic. */
NG_PATTR static void NG_PNAME(const uint8_t *row, const vcn_lc_plist *pl, uint16_t bias, uint16_t *restrict pre,unsigned needed
#ifdef NG_PFUSED
,const uint8_t*e,size_t ri,const uint32_t*table,uint8_t*classes
#endif
) {
  const int n1 = pl->nlev[0], n2 = pl->nlev[1], n3 = pl->npair;
  for (int x = 0; x < VCN_LC_D; x += NG_PWIDTH) {
    if(!(needed & ((NG_PWIDTH==64?3u:1u)<<(x>>5)))) {memset(pre+x,0,NG_PWIDTH*sizeof(*pre));continue;}
    const uint8_t *r = row + x;
    NG_PV lo0 = NG_PSET((short)bias), hi0 = lo0, lo1 = NG_PZERO(), hi1 = lo1, lo2 = lo1, hi2 = lo1;
    int i = 0;
    for (; i < n1; i++) {
      const NG_PV a = NG_PLOAD((const NG_PV *)(r + pl->po[i][0])), c = NG_PLOAD((const NG_PV *)(r + pl->po[i][1]));
      const NG_PV d0 = NG_PSET(pl->pd[i][0]);
      lo0 = NG_PADD(lo0, NG_PMADD(NG_PLO(a, c), d0));
      hi0 = NG_PADD(hi0, NG_PMADD(NG_PHI(a, c), d0));
    }
    for (; i < n2; i++) {
      const NG_PV a = NG_PLOAD((const NG_PV *)(r + pl->po[i][0])), c = NG_PLOAD((const NG_PV *)(r + pl->po[i][1]));
      const NG_PV ul = NG_PLO(a, c), uh = NG_PHI(a, c);
      const NG_PV d0 = NG_PSET(pl->pd[i][0]), d1 = NG_PSET(pl->pd[i][1]);
      lo0 = NG_PADD(lo0, NG_PMADD(ul, d0)), hi0 = NG_PADD(hi0, NG_PMADD(uh, d0));
      lo1 = NG_PADD(lo1, NG_PMADD(ul, d1)), hi1 = NG_PADD(hi1, NG_PMADD(uh, d1));
    }
    for (; i < n3; i++) {
      const NG_PV a = NG_PLOAD((const NG_PV *)(r + pl->po[i][0])), c = NG_PLOAD((const NG_PV *)(r + pl->po[i][1]));
      const NG_PV ul = NG_PLO(a, c), uh = NG_PHI(a, c);
      const NG_PV d0 = NG_PSET(pl->pd[i][0]), d1 = NG_PSET(pl->pd[i][1]), d2 = NG_PSET(pl->pd[i][2]);
      lo0 = NG_PADD(lo0, NG_PMADD(ul, d0)), hi0 = NG_PADD(hi0, NG_PMADD(uh, d0));
      lo1 = NG_PADD(lo1, NG_PMADD(ul, d1)), hi1 = NG_PADD(hi1, NG_PMADD(uh, d1));
      lo2 = NG_PADD(lo2, NG_PMADD(ul, d2)), hi2 = NG_PADD(hi2, NG_PMADD(uh, d2));
    }
    lo0 = NG_PADD(lo0, NG_PADD(NG_PSHIFT(lo1, 7), NG_PSHIFT(lo2, 14)));
    hi0 = NG_PADD(hi0, NG_PADD(NG_PSHIFT(hi1, 7), NG_PSHIFT(hi2, 14)));
    NG_PSTORE(pre+x,lo0,hi0);
  }
}
#undef NG_PATTR
#undef NG_PNAME
#undef NG_PV
#undef NG_PWIDTH
#undef NG_PSTORE
#undef NG_PSET
#undef NG_PZERO
#undef NG_PLOAD
#undef NG_PADD
#undef NG_PMADD
#undef NG_PLO
#undef NG_PHI
#undef NG_PSHIFT
