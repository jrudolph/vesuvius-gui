/* Shared generic/fixed-geometry activity construction. */
static int ng_activity_values_impl(const uint8_t *raw,uint16_t *activity,uint32_t *histogram) {
    uint8_t *e=calloc(VCN_LC_NB,1);
    if(!e)return -1;
    uint32_t bins[4][2560];
    if(histogram)memset(bins,0,sizeof(bins));
    for(int z=0;z<VCN_LC_Z;z++)for(int y=0;y<VCN_LC_Y;y++) {
        size_t ri=VCN_LC_IDX(z,y,0),o=((size_t)z*VCN_LC_Y+y)*VCN_LC_D;
        vcn_lc_row_begin(e,z,y,0);vcn_lc_apre(e,ri,activity+o);
        for(int x=0;x<VCN_LC_D;x++) {
            e[ri+x]=(raw[o+x]+1u)>>1;
        }
        if(histogram)for(int x=0;x<VCN_LC_D;x++)bins[x&3][activity[o+x]]++;
        vcn_lc_row_end(e,z,y);
    }
    if(histogram)for(unsigned a=0;a<2560;a++)histogram[a]=bins[0][a]+bins[1][a]+bins[2][a]+bins[3][a];
    free(e);return 0;
}
