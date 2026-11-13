/* Shared candidate evaluation; specialized by the existing geometry builds and,
 * through ng_bank_symbols_impl, by the selector edge. */
__attribute__((always_inline)) static inline int ng_bank_symbols_edge(unsigned count,const uint8_t* const* params,const size_t* sizes,
    const uint16_t* const* base,const uint8_t* choices,const uint8_t* src,
    uint8_t* const* candidates,uint8_t* symbols,uint16_t* activity,int selected_only,const unsigned edge) {
    if(count<1||count>4)return -1;
    vcn_lc_par* p[4]={0};uint8_t* e=calloc(VCN_LC_NB,1);int rc=-1;
    if(!e)return -1;
    for(unsigned m=0;m<count;m++){const uint8_t* cur=params[m];
        if(!cur&&!sizes[m])continue;
        if(vcn_lc_readpar(&cur,sizes[m],&p[m])!=VCN_OK||cur!=params[m]+sizes[m])goto done;}
    int shared=1;
    for(unsigned m=1;m<count;m++)if(sizes[m]!=sizes[0]||memcmp(params[m],params[0],sizes[0]))shared=0;
    for(int z=0;z<VCN_LC_Z;z++)for(int y=0;y<VCN_LC_Y;y++){
        size_t ri=VCN_LC_IDX(z,y,0),o=((size_t)z*VCN_LC_Y+y)*VCN_LC_D;
        vcn_lc_row_begin(e,z,y,0);vcn_lc_apre(e,ri,activity+o);
        uint8_t row_choices[VCN_LC_D];
        const size_t choice_row=((z/edge)*(VCN_LC_Y/edge)+y/edge)*(VCN_LC_D/edge);
        for(unsigned x=0;x<VCN_LC_D;x+=edge)memset(row_choices+x,choices[choice_row+x/edge],edge);

        uint16_t bias[VCN_LC_D];uint8_t shared_class[VCN_LC_D];
        if(shared){memset(bias,0,sizeof(bias));VCN_LC_CLS(e,ri,p[0],bias,shared_class);}
        uint8_t needed[48]={0};
        if(selected_only)for(unsigned x=0;x<VCN_LC_D;x+=edge){
            unsigned m=row_choices[x];
            if(m>=12+12*count)goto done;
            if(m>=12)needed[m-12]=1;
        }
        for(unsigned m=0;m<12*count;m++){
            unsigned k=m/12;
            if(!p[k])continue;
            // A compact screened menu need not materialize unused inputs.
            if(!candidates[12+m])continue;
            if(selected_only&&!needed[m])continue;
            if(m&&shared&&base[k]==base[0]&&candidates[12+m]==candidates[12]&&(!selected_only||needed[0]))continue;
            uint16_t q[VCN_LC_D];uint8_t cl[VCN_LC_D],err[VCN_LC_D];
            if(shared){for(int x=0;x<VCN_LC_D;x++)q[x]=(uint16_t)(base[k][o+x]+bias[x]);}
            else {memcpy(q,base[k]+o,sizeof(q));VCN_LC_CLS(e,ri,p[k],q,cl);}
            VCN_LC_SYMS(q,src+o,candidates[12+m]+o,err);
        }
        for(unsigned x=0;x<VCN_LC_D;x+=edge){
            unsigned m=row_choices[x];
            if(m>=12+12*count||!candidates[m]||(m>=12&&!p[(m-12)/12]))goto done;
            memcpy(symbols+o+x,candidates[m]+o+x,edge);
            for(unsigned j=0;j<edge;j++)e[ri+x+j]=(symbols[o+x+j]+1u)>>1;
        }
        vcn_lc_row_end(e,z,y);
    }
    rc=0;
done:free(e);for(unsigned m=0;m<count;m++)free(p[m]);return rc;
}
/* Edge 4, the encoders' only choice so far, gets its own copy: the per-block
 * memset/memcpy of the selector row become fixed 4-byte stores (~9% of medium
 * encode). Other edges run the same code with the run-time value. */
static int ng_bank_symbols_impl(unsigned count,const uint8_t* const* params,const size_t* sizes,
    const uint16_t* const* base,const uint8_t* choices,const uint8_t* src,
    uint8_t* const* candidates,uint8_t* symbols,uint16_t* activity,int selected_only) {
    if(ng_choice_edge==4)return ng_bank_symbols_edge(count,params,sizes,base,choices,src,candidates,symbols,activity,selected_only,4);
    return ng_bank_symbols_edge(count,params,sizes,base,choices,src,candidates,symbols,activity,selected_only,ng_choice_edge);
}
