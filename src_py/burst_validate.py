#!/usr/bin/env python3
"""
burst_validate.py — do our bursts behave like order splits, and does the result depend on
using the midpoint as "the price"?

(A) ORDER-FLOW LONG MEMORY. Signed order flow has Hurst ~0.7-0.8, and Lillo-Farmer and
    Toth et al. attribute that almost entirely to order splitting: a parent order emits many
    same-side children, so the sign series is persistent. The falsifiable implication is that
    if a burst definition correctly aggregates a parent's children into ONE event, flow
    measured at the burst level should have markedly shorter memory than flow measured at the
    trade level. If the bursts are arbitrary time windows, H barely moves. Hurst is estimated
    by aggregated variance: var of block means against block size, slope = 2H-2.

(B) SQUARE-ROOT IMPACT LAW. Metaorder impact scales as sqrt(size/ADV). If bursts reproduce
    that exponent they are behaving like parent orders. We regress log|impact| on log(size)
    across bursts and report the exponent; ~0.5 supports the metaorder reading, ~1.0 or ~0
    does not.

(C) PRICE-PROXY ROBUSTNESS. The midpoint moves when makers cancel, so it partly reflects
    quoting posture rather than transacted value. We therefore recompute the aggressive-hidden
    markout under four definitions of "the price":
        mid        (bid+ask)/2                       -- the paper's convention
        micro      queue-weighted (bid*askSz + ask*bidSz)/(bidSz+askSz)
        last       last traded price                 -- immune to cancellation, but carries
                                                        bid-ask bounce (Roll 1984)
        vwap5      5-second trade VWAP               -- bounce-damped
    If the footprint survives all four, it is not an artifact of quoting posture.

Output: ticker,date,h_trade,h_burst,n_tr,n_bu,sqrt_exp,sqrt_r2,mk_mid,mk_micro,mk_last,mk_vwap
"""
import argparse, os, re, sys
import numpy as np, pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import burst_alt as BA
RTH0, RTH1 = 34200.0, 57600.0
NA = "{t},{d}" + ",nan" * 10

def wm(x, cap=1000.0):
    x=np.asarray(x,float); x=x[np.isfinite(x)]; x=x[np.abs(x)<=cap]
    return float(np.mean(x)) if len(x) else np.nan

def hurst(x):
    """Aggregated-variance estimator. Returns H, or nan."""
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    n=len(x)
    if n<400 or np.std(x)<1e-12: return np.nan
    ms=[m for m in (2,4,8,16,32,64,128) if n//m>=20]
    if len(ms)<4: return np.nan
    v=[]
    for m in ms:
        k=n//m
        blk=x[:k*m].reshape(k,m).mean(axis=1)
        v.append(np.var(blk))
    v=np.array(v); ok=v>0
    if ok.sum()<4: return np.nan
    s=np.polyfit(np.log(np.array(ms)[ok]), np.log(v[ok]), 1)[0]
    return float(s/2.0 + 1.0)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--msg",required=True); ap.add_argument("--ticker",required=True)
    a=ap.parse_args()
    m=re.search(r"(\d{4})-(\d{2})-(\d{2})",os.path.basename(a.msg))
    d=int(m.group(1)+m.group(2)+m.group(3)) if m else 0
    try:
        bt,bm,bb,ba,bbsz,basz,ofi,trades=BA.reconstruct(a.msg)
        df=pd.read_csv(a.msg,header=None,usecols=[0,1,3,4,5],names=["t","ty","sz","px","dr"])
        df=df[(df.t>=RTH0)&(df.t<RTH1)]
        if len(df)<1000 or len(bt)<100: print(NA.format(t=a.ticker,d=d)); return
        T=df.t.to_numpy(float); TY=df.ty.to_numpy(int)
        SZ=df.sz.to_numpy(float); PX=df.px.to_numpy(float)/BA.SCALE; DR=df.dr.to_numpy(int)
        v=TY==4; tv,sv,pv,av=T[v],SZ[v],PX[v],-DR[v]      # native aggressor sign
        if len(tv)<500: print(NA.format(t=a.ticker,d=d)); return

        # ---- (A) long memory: trade-level vs burst-level signed flow ----
        h_tr=hurst(av.astype(float))
        # bursts on ARRIVAL TIMING ONLY (price-free), native signs, >=3 same-side
        ends=[];dirs=[];sizes=[];starts=[]
        i=0
        while i<len(tv):
            j=i
            while j+1<len(tv) and av[j+1]==av[i] and (tv[j+1]-tv[j])<1.0: j+=1
            if j-i+1>=3:
                starts.append(tv[i]); ends.append(tv[j]); dirs.append(int(av[i])); sizes.append(sv[i:j+1].sum())
            i=j+1
        dirs=np.array(dirs); sizes=np.array(sizes,float)
        starts=np.array(starts,float); ends=np.array(ends,float)
        h_bu=hurst(dirs.astype(float)) if len(dirs)>=400 else np.nan

        # ---- (B) square-root law: |impact| vs burst size ----
        sq_e=sq_r=np.nan
        if len(ends)>=40:
            p0=BA.mid_at(bt,bm,starts); p1=BA.mid_at(bt,bm,ends)
            with np.errstate(invalid="ignore",divide="ignore"):
                imp=np.abs(dirs*(p1-p0)/p0*1e4)
            k=np.isfinite(imp)&(imp>0)&(sizes>0)&(imp<1000)
            if k.sum()>=40:
                X=np.log(sizes[k]); Y=np.log(imp[k])
                b=np.polyfit(X,Y,1); sq_e=float(b[0])
                yh=np.polyval(b,X); ss=((Y-Y.mean())**2).sum()
                sq_r=float(1-((Y-yh)**2).sum()/ss) if ss>0 else np.nan

        # ---- (C) markout under four price proxies ----
        h5=TY==5
        th,ph=T[h5],PX[h5]
        mk=[np.nan]*4
        if len(th)>=20:
            mid0=BA.mid_at(bt,bm,th)
            ok=np.isfinite(mid0)&(mid0>0)&(th<RTH1-200.0)
            th,ph,mid0=th[ok],ph[ok],mid0[ok]
            pb,pa=BA.bbo_at(bt,bb,ba,th-1e-3)
            q=np.zeros(len(th),int)
            q[np.isfinite(pa)&(ph>pa)]=1; q[np.isfinite(pb)&(ph<pb)]=-1   # unambiguous only
            s=q!=0
            if s.sum()>=10:
                ts,qs=th[s],q[s]
                def proxy(times):
                    lo,hi=BA.bbo_at(bt,bb,ba,times)
                    ls,hs=BA.bbo_at(bt,bbsz,basz,times)
                    mid=(lo+hi)/2.0
                    with np.errstate(invalid="ignore",divide="ignore"):
                        micro=(lo*hs+hi*ls)/np.maximum(ls+hs,1e-9)
                    idx=np.clip(np.searchsorted(tv,times,side="right")-1,0,len(tv)-1)
                    last=pv[idx]
                    vw=np.empty(len(times))
                    for k2,tt in enumerate(times):
                        a2=np.searchsorted(tv,tt-5.0); b2=np.searchsorted(tv,tt,side="right")
                        vw[k2]=(pv[a2:b2]*sv[a2:b2]).sum()/sv[a2:b2].sum() if b2>a2 and sv[a2:b2].sum()>0 else last[k2]
                    return mid,micro,last,vw
                P0=proxy(ts); P1=proxy(ts+180.0)
                for z in range(4):
                    with np.errstate(invalid="ignore",divide="ignore"):
                        mk[z]=wm(qs*(P1[z]-P0[z])/P0[z]*1e4)
        f=lambda x:("%.5f"%x) if np.isfinite(x) else "nan"
        print("%s,%d,%s,%s,%d,%d,%s,%s,%s,%s,%s,%s"%(a.ticker,d,f(h_tr),f(h_bu),len(tv),len(ends),
              f(sq_e),f(sq_r),f(mk[0]),f(mk[1]),f(mk[2]),f(mk[3])))
    except Exception as e:
        print(f"{a.ticker},{d},ERR,{e}",file=sys.stderr); print(NA.format(t=a.ticker,d=d))

if __name__=="__main__": main()
