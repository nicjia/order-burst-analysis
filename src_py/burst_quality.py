#!/usr/bin/env python3
"""
burst_quality.py — rank burst definitions by how well they aggregate order splits.

Every definition in this project was scored on whether it produced a significant markout.
None was scored on whether it is a GOOD DEFINITION. Lillo-Farmer and Toth et al. attribute
the long memory of signed order flow (Hurst ~0.75) to order splitting, so a definition that
correctly collapses a parent's children into one event should destroy that memory. Definitions
can therefore be ranked by how much they reduce H.

CONFOUND: coarser aggregation reduces H mechanically. A definition emitting one burst per day
would drive H to nothing while capturing nothing. We therefore compare each definition against
a COARSENESS-MATCHED PLACEBO: the same trade sequence cut into the same number of blocks with
the same size distribution, but at RANDOM boundaries. The quality metric is

    dH = H(placebo) - H(real)

which is positive only if the definition captures structure beyond its own granularity.

Definitions swept (all price-free in formation, native ITCH signs):
    sil_X      silence threshold X seconds, runs of same-side trades
    hwk_B_L    decaying counter, decay B, threshold L
    run_K      minimum run length K at a 1s gap
    vol_F      volume clock, bucket = F of daily volume

Output: ticker,date,defn,n_bursts,mean_size,h_trade,h_real,h_placebo,dH
"""
import argparse, os, re, sys
import numpy as np, pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import burst_alt as BA
RTH0, RTH1 = 34200.0, 57600.0

def hurst(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]; n=len(x)
    if n<400 or np.std(x)<1e-12: return np.nan
    ms=[m for m in (2,4,8,16,32,64,128) if n//m>=20]
    if len(ms)<4: return np.nan
    v=[]
    for m in ms:
        k=n//m; v.append(np.var(x[:k*m].reshape(k,m).mean(axis=1)))
    v=np.array(v); ok=v>0
    if ok.sum()<4: return np.nan
    return float(np.polyfit(np.log(np.array(ms)[ok]),np.log(v[ok]),1)[0]/2.0+1.0)

def sign_series(seg_sizes, av):
    """Collapse consecutive blocks of the given sizes into net signs."""
    out=[]; i=0
    for s in seg_sizes:
        s=int(s)
        if s<=0 or i+s>len(av): break
        net=av[i:i+s].sum()
        out.append(np.sign(net) if net!=0 else 0.0); i+=s
    return np.array(out,float)

def runs(tv, av, gap, minrun, same_side=True):
    idx=[]; i=0; n=len(tv)
    while i<n:
        j=i
        while j+1<n and (tv[j+1]-tv[j])<gap and ((av[j+1]==av[i]) if same_side else True): j+=1
        if j-i+1>=minrun: idx.append((i,j))
        i=j+1
    return idx

def counter(tv, av, beta, lam, minrun=3):
    """Decaying counter: intensity *= exp(-beta*gap) + 1; burst ends when it drops below lam."""
    idx=[]; i=0; n=len(tv)
    while i<n:
        j=i; inten=1.0
        while j+1<n:
            g=tv[j+1]-tv[j]
            if inten*np.exp(-beta*g) < lam or av[j+1]!=av[i]: break
            inten=inten*np.exp(-beta*g)+1.0; j+=1
        if j-i+1>=minrun: idx.append((i,j))
        i=j+1
    return idx

def emit(name, idx, tv, av, sv, h_tr, rng, tk, d):
    if len(idx)<400:
        return
    sizes=np.array([b-a+1 for a,b in idx],float)
    sgn=np.array([np.sign(av[a:b+1].sum()) for a,b in idx],float)
    h_real=hurst(sgn)
    # coarseness-matched placebo: same block-size multiset, random order, from position 0
    perm=rng.permutation(sizes)
    h_pl=hurst(sign_series(perm, av))
    dH=(h_pl-h_real) if (np.isfinite(h_pl) and np.isfinite(h_real)) else np.nan
    f=lambda x:("%.5f"%x) if np.isfinite(x) else "nan"
    print("%s,%d,%s,%d,%.1f,%s,%s,%s,%s"%(tk,d,name,len(idx),sizes.mean(),
          f(h_tr),f(h_real),f(h_pl),f(dH)))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--msg",required=True); ap.add_argument("--ticker",required=True)
    a=ap.parse_args(); tk=a.ticker
    m=re.search(r"(\d{4})-(\d{2})-(\d{2})",os.path.basename(a.msg))
    d=int(m.group(1)+m.group(2)+m.group(3)) if m else 0
    try:
        bt,bm,bb,ba,bbsz,basz,ofi,trades=BA.reconstruct(a.msg)
        df=pd.read_csv(a.msg,header=None,usecols=[0,1,3,4,5],names=["t","ty","sz","px","dr"])
        df=df[(df.t>=RTH0)&(df.t<RTH1)&(df.ty==4)]
        if len(df)<3000: return
        tv=df.t.to_numpy(float); sv=df.sz.to_numpy(float); av=(-df.dr.to_numpy(int)).astype(float)
        h_tr=hurst(av)
        rng=np.random.default_rng((d+abs(hash(tk)))%(2**32))
        for g in (0.25,0.5,1.0,2.0,5.0):
            emit("sil_%g"%g, runs(tv,av,g,3), tv,av,sv,h_tr,rng,tk,d)
        for B in (0.5,1.0,2.0,5.0):
            for L in (0.3,0.5,0.8):
                emit("hwk_%g_%g"%(B,L), counter(tv,av,B,L), tv,av,sv,h_tr,rng,tk,d)
        for K in (2,3,5,10):
            emit("run_%d"%K, runs(tv,av,1.0,K), tv,av,sv,h_tr,rng,tk,d)
        cum=np.cumsum(sv); tot=cum[-1]
        for F in (0.0005,0.001,0.002,0.005):
            b=(cum/(tot*F)).astype(int)
            ch=np.flatnonzero(np.diff(b))+1
            idx=list(zip(np.r_[0,ch], np.r_[ch-1,len(tv)-1]))
            idx=[(x,y) for x,y in idx if y>=x+2]
            emit("vol_%g"%F, idx, tv,av,sv,h_tr,rng,tk,d)
    except Exception as e:
        print(f"{tk},{d},ERR,ERR,0,nan,nan,nan,nan",file=sys.stderr)

if __name__=="__main__": main()
