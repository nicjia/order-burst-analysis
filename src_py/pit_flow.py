#!/usr/bin/env python3
"""
pit_flow.py — daily flow panel for a point-in-time universe (delisted names included).

Emits one row per ticker-day with NATIVELY SIGNED visible flow (ITCH Direction on type-4
messages, no price inference, so none of the circularity that invalidated the hidden-print
signing) alongside cleanly-signed hidden flow (outside-the-pre-print-quote only) for
comparison. This is the input to a reversal test on a universe that includes names which
later delisted.

Output: ticker,date,n_vis,vol_vis,signed_vis,dvol_vis,n_hid,vol_hid,signed_hid,open_mid,close_mid,halfsp
"""
import argparse, os, re, sys
import numpy as np, pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import burst_alt as BA
RTH0, RTH1 = 34200.0, 57600.0
NA = "{t},{d}" + ",nan" * 10

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True); ap.add_argument("--ticker", required=True)
    a = ap.parse_args()
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", os.path.basename(a.msg))
    d = int(m.group(1)+m.group(2)+m.group(3)) if m else 0
    try:
        bt, bm, bb, ba, bbsz, basz, ofi, trades = BA.reconstruct(a.msg)
        df = pd.read_csv(a.msg, header=None, usecols=[0,1,3,4,5], names=["t","ty","sz","px","dr"])
        df = df[(df.t>=RTH0)&(df.t<RTH1)]
        if len(df)<200 or len(bt)<50:
            print(NA.format(t=a.ticker,d=d)); return
        T=df.t.to_numpy(float); TY=df.ty.to_numpy(int)
        SZ=df.sz.to_numpy(float); PX=df.px.to_numpy(float)/BA.SCALE; DR=df.dr.to_numpy(int)
        v=TY==4; tv,sv,pv,av=T[v],SZ[v],PX[v],-DR[v]
        h=TY==5; th,sh,ph=T[h],SZ[h],PX[h]
        sgn_h=0.0; vol_h=float(sh.sum()); n_h=int(h.sum())
        if n_h>0:
            pb,pa=BA.bbo_at(bt,bb,ba,th-1e-3)
            s=np.zeros(n_h)
            s[np.isfinite(pa)&(ph>pa)]=1; s[np.isfinite(pb)&(ph<pb)]=-1
            sgn_h=float((s*sh).sum())
        g=np.arange(RTH0,RTH1,60.0); gm=BA.mid_at(bt,bm,g); gl,gh=BA.bbo_at(bt,bb,ba,g)
        with np.errstate(invalid="ignore",divide="ignore"):
            hs=0.5*(gh-gl)/gm*1e4
        hs=hs[np.isfinite(hs)&(np.abs(hs)<500)]
        om=BA.mid_at(bt,bm,np.array([RTH0+60.0]))[0]
        cm=BA.mid_at(bt,bm,np.array([RTH1-60.0]))[0]
        f=lambda x:("%.5f"%x) if np.isfinite(x) else "nan"
        print("%s,%d,%d,%.0f,%.0f,%.0f,%d,%.0f,%.0f,%s,%s,%s"%(
            a.ticker,d,int(v.sum()),sv.sum(),(av*sv).sum(),(sv*pv).sum(),
            n_h,vol_h,sgn_h,f(om),f(cm),f(np.mean(hs) if len(hs) else np.nan)))
    except Exception as e:
        print(f"{a.ticker},{d},ERR,{e}",file=sys.stderr); print(NA.format(t=a.ticker,d=d))

if __name__=="__main__": main()
