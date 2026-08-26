#!/usr/bin/env python3
"""
idea_zoo2.py — remaining ideas, plus the test that decides the volatility result.

The one live finding from sweep 1 was burst intensity forecasting future |return| at
IC ~ 0.39. But volatility clusters, and burst intensity is correlated with CONTEMPORANEOUS
volatility, so that IC may be a restatement of "volatility is autocorrelated" rather than new
information. The decisive test is incremental: regress future realized volatility on lagged
realized volatility (the HAR benchmark) and then ask whether burst intensity adds anything.
That is the same incrementality discipline applied to the footprint against visible OFI, now
applied to a second moment.

Also implements the remaining single-name ideas from the list: adverse-selection avoidance,
time-to-fill, volume-profile deviation, closing-auction imbalance, jump arrival, and
entropy-defined bursts.

Long format: ticker,date,idea,family,param,n,v3,v10,v30,net3
"""
import argparse, os, re, sys
import numpy as np, pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import burst_alt as BA
RTH0, RTH1 = 34200.0, 57600.0

def wm(x, cap=1e9):
    x=np.asarray(x,float); x=x[np.isfinite(x)]; x=x[np.abs(x)<=cap]
    return float(np.mean(x)) if len(x) else np.nan

def ic(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); k=np.isfinite(x)&np.isfinite(y)
    if k.sum()<20 or np.std(x[k])<1e-12 or np.std(y[k])<1e-12: return np.nan
    return float(np.corrcoef(x[k],y[k])[0,1])

def row(tk,d,idea,fam,p,n,a,b,c,e=np.nan):
    f=lambda v: ("%.5f"%v) if np.isfinite(v) else "nan"
    print("%s,%s,%s,%s,%s,%d,%s,%s,%s,%s"%(tk,d,idea,fam,p,int(n),f(a),f(b),f(c),f(e)))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--msg",required=True); ap.add_argument("--ticker",required=True)
    a=ap.parse_args(); tk=a.ticker
    m=re.search(r"(\d{4})-(\d{2})-(\d{2})",os.path.basename(a.msg))
    d=int(m.group(1)+m.group(2)+m.group(3)) if m else 0
    try:
        bt,bm,bb,ba,bbsz,basz,ofi,trades=BA.reconstruct(a.msg)
        df=pd.read_csv(a.msg,header=None,usecols=[0,1,3,4,5],names=["t","ty","sz","px","dr"])
        df=df[(df.t>=RTH0)&(df.t<RTH1)]
        if len(df)<500 or len(bt)<100: return
        T=df.t.to_numpy(float); TY=df.ty.to_numpy(int); SZ=df.sz.to_numpy(float); DR=df.dr.to_numpy(int)
        tv,sv,av=T[TY==4],SZ[TY==4],-DR[TY==4]; th=T[TY==5]; sh=SZ[TY==5]

        # ---- HAR INCREMENTALITY: does burst intensity beat lagged realized vol? ----
        for W in (60.0,300.0):
            edges=np.arange(RTH0,RTH1-W,W)
            if len(edges)<40: continue
            mg=BA.mid_at(bt,bm,edges)
            with np.errstate(invalid="ignore",divide="ignore"):
                r=np.diff(mg)/mg[:-1]*1e4
            rv=np.abs(r)                                   # bucket realized |return|
            cnt=np.histogram(tv,bins=np.append(edges,edges[-1]+W))[0].astype(float)[:len(rv)]
            hcn=np.histogram(th,bins=np.append(edges,edges[-1]+W))[0].astype(float)[:len(rv)]
            L=6
            if len(rv)<L+30: continue
            y=rv[L:]
            lag1=rv[L-1:-1]                                # immediately prior bucket
            lagK=np.array([rv[i-L:i].mean() for i in range(L,len(rv))])   # trailing average
            xb=cnt[L:]; xh=hcn[L:]
            k=np.isfinite(y)&np.isfinite(lag1)&np.isfinite(lagK)&np.isfinite(xb)
            if k.sum()<40: continue
            def r2(cols):
                X=np.column_stack([np.ones(k.sum())]+[c[k] for c in cols])
                bta,*_=np.linalg.lstsq(X,y[k],rcond=None); res=y[k]-X@bta
                ss=((y[k]-y[k].mean())**2).sum()
                return (1-(res**2).sum()/ss) if ss>0 else np.nan, bta, X, res
            r2_har,_,_,_=r2([lag1,lagK])
            r2_all,bta,X,res=r2([lag1,lagK,xb,xh])
            # t on the burst-count coefficient
            n_,p_=X.shape; s2=(res**2).sum()/max(n_-p_,1)
            try: se=np.sqrt(np.diag(s2*np.linalg.pinv(X.T@X)))
            except Exception: se=np.full(p_,np.nan)
            tb=bta[3]/se[3] if se[3]>0 else np.nan
            row(tk,d,"i19b_HAR_incremental","VOL",int(W),k.sum(),r2_har,r2_all,r2_all-r2_har,tb)

        # ---- i21 jump arrival: does burst intensity predict a >3 sigma move? ----
        for W in (60.0,300.0):
            edges=np.arange(RTH0,RTH1-W,W)
            if len(edges)<40: continue
            mg=BA.mid_at(bt,bm,edges)
            with np.errstate(invalid="ignore",divide="ignore"):
                r=np.diff(mg)/mg[:-1]*1e4
            sd=np.nanstd(r)
            if not np.isfinite(sd) or sd<=0: continue
            cnt=np.histogram(tv,bins=np.append(edges,edges[-1]+W))[0].astype(float)[:len(r)]
            jump=(np.abs(r)>3*sd).astype(float)
            row(tk,d,"i21_jump","VOL",int(W),len(r),ic(cnt[:-1],jump[1:]),np.nan,np.nan)

        # ---- i09 adverse-selection avoidance: which moments are toxic to quote into ----
        for W in (60.0,300.0):
            edges=np.arange(RTH0,RTH1-600.0,W)
            if len(edges)<30: continue
            hcn=np.histogram(th,bins=np.append(edges,edges[-1]+W))[0].astype(float)
            m0=BA.mid_at(bt,bm,edges+W); m1=BA.mid_at(bt,bm,edges+W+300.0)
            with np.errstate(invalid="ignore",divide="ignore"):
                adv=np.abs((m1-m0)/m0)*1e4      # magnitude of the move a quoter would eat
            row(tk,d,"i09_advsel","LIQ",int(W),len(edges),ic(hcn,adv),np.nan,np.nan)

        # ---- i10 time-to-fill proxy: burst state vs subsequent trade count at the touch ----
        for W in (60.0,):
            edges=np.arange(RTH0,RTH1-600.0,W)
            if len(edges)<30: continue
            cnt=np.histogram(tv,bins=np.append(edges,edges[-1]+W))[0].astype(float)
            nxt=np.histogram(tv,bins=np.append(edges+300.0,edges[-1]+300.0+W))[0].astype(float)
            row(tk,d,"i10_timetofill","LIQ",int(W),len(edges),ic(cnt,nxt),np.nan,np.nan)

        # ---- i12 volume-profile deviation -> rest-of-day volume ----
        edges=np.arange(RTH0,RTH1-1800.0,300.0)
        if len(edges)>20:
            cum=np.array([sv[tv<e+300.0].sum() for e in edges])
            rest=np.array([sv[tv>=e+300.0].sum() for e in edges])
            frac=cum/max(sv.sum(),1)
            row(tk,d,"i12_volprofile","FLOW",300,len(edges),ic(frac,rest),np.nan,np.nan)

        # ---- i15 closing-auction: late imbalance -> move into the close ----
        for W in (600.0,1800.0,3600.0):
            t0=RTH1-W
            k=tv>=t0
            if k.sum()<20: continue
            imb=(av[k]*sv[k]).sum()/max(sv[k].sum(),1)
            m0=BA.mid_at(bt,bm,np.array([t0]))[0]; m1=BA.mid_at(bt,bm,np.array([RTH1-1.0]))[0]
            with np.errstate(invalid="ignore",divide="ignore"):
                mv=np.sign(imb)*(m1-m0)/m0*1e4
            row(tk,d,"i15_auction","DIR",int(W),1,mv,np.nan,np.nan)

        # ---- i27 entropy bursts: surprise in the message-type mix ----
        for W in (60.0,300.0):
            edges=np.arange(RTH0,RTH1-600.0,W)
            if len(edges)<30: continue
            H=[]
            for e in edges:
                seg=TY[(T>=e)&(T<e+W)]
                if len(seg)<5: H.append(np.nan); continue
                _,c=np.unique(seg,return_counts=True); pr=c/c.sum()
                H.append(float(-(pr*np.log(pr)).sum()))
            H=np.array(H)
            m0=BA.mid_at(bt,bm,edges+W); m1=BA.mid_at(bt,bm,edges+W+300.0)
            with np.errstate(invalid="ignore",divide="ignore"):
                absr=np.abs((m1-m0)/m0)*1e4
            row(tk,d,"i27_entropy","VOL",int(W),len(edges),ic(H,absr),np.nan,np.nan)
    except Exception as ex:
        print(f"{tk},{d},ERR,ERR,{ex},0,nan,nan,nan,nan",file=sys.stderr)

if __name__=="__main__": main()
