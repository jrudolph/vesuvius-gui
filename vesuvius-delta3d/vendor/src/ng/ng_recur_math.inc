            NG_RV q;
            if(simple){
                NG_RV first=homogeneous&&!x?w0:wx;
                q=NG_RADD(NG_BASE_PRE,NG_RADD(NG_RMUL(first,p0),NG_RADD(NG_RMUL(w1,p1),NG_RMUL(w2,p2))));
            }else if(xyz){
                NG_RV bounds=NG_BASE_BOUNDS;
                NG_RV lo=NG_RMIN(NG_RAND(bounds,bytes),p0),hi=NG_RMAX(NG_RSRL(bounds,8),p0);
                NG_RV base=NG_RMIN(NG_RMAX(NG_RADD(NG_BASE_CONST,p0),lo),hi);
                q=NG_RADD(NG_BASE_PRE,NG_RADD(NG_RMUL(f,base),NG_RADD(NG_RMUL(w0,p0),NG_RADD(NG_RMUL(w1,p1),NG_RMUL(w2,p2)))));
            }else if(pair){
                NG_RV bounds=NG_BASE_BOUNDS;
                NG_RV lo=NG_RMIN(NG_RAND(bounds,bytes),p0),hi=NG_RMAX(NG_RSRL(bounds,8),p0);
                NG_RV base=NG_RADD(NG_BASE_CONST,p0);
                base=NG_RBLEND(base,NG_RMIN(NG_RMAX(base,lo),hi),clamp);
                NG_RV dot=NG_RADD(NG_RMUL(w0,p0),NG_RADD(NG_RMUL(w1,p1),NG_RMUL(w2,p2)));
                q=NG_RADD(NG_BASE_PRE,NG_RADD(dot,NG_RMUL(f,NG_RAND(base,bytes))));
            }else{
            NG_RV bounds=NG_BASE_BOUNDS;
            NG_RV base=NG_RADD(NG_BASE_CONST,NG_RMASKVAL(left,p0));
            NG_RV lo=NG_RAND(bounds,bytes),hi=NG_RSRL(bounds,8);
            lo=NG_RBLEND(lo,NG_RMIN(lo,p0),left);
            hi=NG_RBLEND(hi,NG_RMAX(hi,p0),left);
            base=NG_RBLEND(base,NG_RMIN(NG_RMAX(base,lo),hi),clamp);
            NG_RV dot=NG_RADD(NG_RMUL(w0,p0),NG_RADD(NG_RMUL(w1,p1),NG_RMUL(w2,p2)));
            q=NG_RADD(NG_BASE_PRE,NG_RADD(NG_RMASKVAL(ng,dot),NG_RMUL(factor,NG_RAND(base,bytes))));
            }
            NG_RV flip=NG_RSRA(NG_RSLL(q,8),15);
            NG_RV raw;
#ifdef NG_RPACKED
            raw=_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)raw_bytes[x]));
#else
            raw=NG_RSRL(NG_COMPACT_LOAD(flags),8);
#endif
            NG_RV residual=NG_RSUB(NG_RXOR(raw,flip),flip);
            NG_RV v=NG_RAND(bytes,NG_RADD(NG_RSRL(NG_RADD(q,NG_RSET(128)),8),residual));
            NG_ROUT(result+x*32+NG_RSTEP*h,v);
            p2=p1;p1=p0;p0=v;
#undef NG_BASE_CONST
#undef NG_BASE_BOUNDS
#undef NG_BASE_PRE
