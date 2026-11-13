/* Exact integer Gram tiles, shared by AVX2 and AVX-512 instantiations.
 * AVX2 has only 16 vector registers: use 2x4 instead of spilling a 4x4
 * accumulator tile plus its inputs. AVX-512 retains the 4x4 tile. */
NG_GATTR static void NG_GNAME(const int16_t*F,int ncp,int64_t*G){
  for(int a0=0;a0<ncp;a0+=NG_GROWS)for(int k0=0;k0<=a0;k0+=4){
    NG_GV acc[NG_GROWS][4];
    for(int i=0;i<NG_GROWS;i++)for(int j=0;j<4;j++)acc[i][j]=NG_GZERO();
    const int16_t*fa=F+(size_t)a0*VCN_LC_SBP,*fk=F+(size_t)k0*VCN_LC_SBP;
    for(int v=0;v<VCN_LC_SB;v+=NG_GSTEP){
      NG_GV a[NG_GROWS],b[4];
      for(int i=0;i<NG_GROWS;i++)a[i]=NG_GLOAD((const NG_GV*)(fa+(size_t)i*VCN_LC_SBP+v));
      for(int i=0;i<4;i++)b[i]=NG_GLOAD((const NG_GV*)(fk+(size_t)i*VCN_LC_SBP+v));
      for(int i=0;i<NG_GROWS;i++)for(int j=0;j<4;j++)acc[i][j]=NG_GADD(acc[i][j],NG_GMADD(a[i],b[j]));
    }
    for(int i=0;i<NG_GROWS;i++)for(int j=0;j<4;j++)G[(size_t)(a0+i)*ncp+k0+j]+=NG_GSUM(acc[i][j]);
  }
}
#undef NG_GATTR
#undef NG_GROWS
#undef NG_GNAME
#undef NG_GV
#undef NG_GZERO
#undef NG_GSTEP
#undef NG_GLOAD
#undef NG_GADD
#undef NG_GMADD
#undef NG_GSUM
