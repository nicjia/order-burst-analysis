#!/usr/bin/env python3
"""
intraday_backtest.py — Reviewer R1: tradable intraday burst-continuation test.

Trades EVERY geometry-gated burst in its own direction and exits after a fixed
intraday horizon (1/3/5/10 min) or at the close. Crucially, NO D_b / kappa gating
is used (kappa=0): D_b is computed from forward mid-prices, so applying it would be
look-ahead bias for a strategy that exits inside that same forward window. Only the
non-anticipating geometry gates (fractional-ADV volume, directional consistency,
volume ratio) are applied.

Execution / cost model, per burst:
  EndMid = (EndBid+EndAsk)/2                       (mark)
  entry_exec = EndAsk if buy else EndBid           (cross the spread to enter)
  full_spread_bps = (EndAsk-EndBid)/EndMid*1e4
  gross_mid[h]  = Dir*(Mid_h - EndMid)/EndMid*1e4          (mid->mid, no friction)
  net_entry[h]  = Dir*(Mid_h - entry_exec)/entry_exec*1e4  (cross entry only, exit at mid)
  net_rtrip[h]  = gross_mid[h] - full_spread_bps           (cross both sides)
  minus an optional flat fee (bps) on top.

Aggregation: each (day) forms an equal-weight portfolio over that day's bursts
(mean bps); the daily series gives an annualized Sharpe (sqrt(252)). Cluster
bootstrap over dates for the mean-markout CI.
"""
import argparse, os, sys, glob, json
import numpy as np, pandas as pd
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
from silence_optimized_sweep import compute_trailing_adv, classify_and_filter

HORIZ = [("1m","Mid_1m"),("3m","Mid_3m"),("5m","Mid_5m"),("10m","Mid_10m"),("close","CloseMid")]
NEED = ["Date","Direction","Volume","BuyCount","SellCount","BuyVolume","SellVolume","D_b",
        "StartPrice","EndPrice","EndBid","EndAsk","Mid_1m","Mid_3m","Mid_5m","Mid_10m","CloseMid"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", required=True, help="comma list or @file")
    ap.add_argument("--burst-dir", default="results/")
    ap.add_argument("--suffix", default="baseline_unfiltered")  # UNFILTERED: kappa=0 only
    ap.add_argument("--params", default="results/optuna_regression/universal_median_params.json")
    ap.add_argument("--fee-bps", type=float, default=0.0, help="extra flat one-way fee per trade (bps)")
    ap.add_argument("--start-date", type=int, default=20220101)
    ap.add_argument("--end-date", type=int, default=20261231)
    ap.add_argument("--nboot", type=int, default=1000)
    ap.add_argument("--out", default="results/research/intraday_backtest_2026")
    args = ap.parse_args()

    P = json.load(open(args.params)); VF,DT,VR = P["vol_frac"],P["dir_thresh"],P["vol_ratio"]
    tickers = ([l.split()[0] for l in open(args.tickers[1:]) if l.strip() and not l.startswith("#")]
               if args.tickers.startswith("@") else [t.strip() for t in args.tickers.split(",") if t.strip()])

    daily = []   # per (ticker, day): mean bps per horizon under each cost model
    n_bursts = 0
    for tk in tickers:
        path = os.path.join(args.burst_dir, f"bursts_{tk}_{args.suffix}.csv")
        if not os.path.exists(path): continue
        try:
            avail = pd.read_csv(path, nrows=0).columns
            df = pd.read_csv(path, usecols=[c for c in NEED if c in avail])
        except Exception: continue
        gate_cols = ["Volume","BuyCount","SellCount","BuyVolume","SellVolume","D_b"]
        if df.empty or any(c not in df.columns for c in gate_cols): continue
        try: df["Date"] = df["Date"].astype(int)
        except (ValueError,TypeError): df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y%m%d").astype(int)
        df = df[(df["Date"]>=args.start_date)&(df["Date"]<=args.end_date)]
        if df.empty: continue
        adv = compute_trailing_adv(df, window=14, stock_folder=f"data/{tk}")
        mvpb = (VF*df["Date"].map(adv)).reindex(df.index)
        try:
            g = classify_and_filter(df, min_vol=0, dir_thresh=DT, vol_ratio=VR, kappa=0.0,
                                    require_directional=True, min_vol_per_burst=mvpb)
        except Exception as e:
            print(f"  skip {tk}: {e}"); continue
        if g.empty: continue
        n_bursts += len(g)
        d = g["Direction"].astype(float).values
        eb = g["EndBid"].astype(float).values; ea = g["EndAsk"].astype(float).values
        em = (eb+ea)/2.0; bad = ~(em>0); em = np.where(bad, g["EndPrice"].astype(float).values, em)
        entry_exec = np.where(d>0, ea, eb); entry_exec = np.where(entry_exec>0, entry_exec, em)
        spread_bps = np.where(em>0, (ea-eb)/em*1e4, 0.0)
        rec = {"Date": g["Date"].values}
        for h,col in HORIZ:
            if col not in g.columns: continue
            mh = g[col].astype(float).values
            gross = d*(mh-em)/em*1e4
            net_e = d*(mh-entry_exec)/entry_exec*1e4 - args.fee_bps
            net_rt = gross - spread_bps - 2*args.fee_bps
            rec[f"gross_{h}"]=gross; rec[f"nete_{h}"]=net_e; rec[f"netrt_{h}"]=net_rt
        dd = pd.DataFrame(rec).replace([np.inf,-np.inf],np.nan)
        daily.append(dd.groupby("Date").mean())
    if not daily: print("no data"); sys.exit(1)
    # universe daily portfolio: average across ticker-days per date
    allp = pd.concat(daily)                       # index=Date (dup across tickers)
    port = allp.groupby(level=0).mean()           # equal-weight names each day
    port.to_csv(f"{args.out}_daily.csv")
    print(f"gated bursts={n_bursts:,}  trading days={len(port)}  ticker-days={len(allp):,}")

    rng = np.random.default_rng(7)
    def boot_ci(series):
        v = series.dropna().values
        if len(v)<30: return (np.nan,np.nan)
        bm = [np.nanmean(v[rng.integers(0,len(v),len(v))]) for _ in range(args.nboot)]
        return np.nanpercentile(bm,2.5), np.nanpercentile(bm,97.5)

    print(f"\n===== INTRADAY BURST-CONTINUATION (geometry-gated, kappa=0, fee={args.fee_bps}bps one-way) =====")
    print(f"{'hz':5s} | {'mid->mid':>22s} | {'cross entry':>22s} | {'round-trip spread':>22s}")
    print(f"{'':5s} | {'bps  Sharpe  hit%':>22s} | {'bps  Sharpe  hit%':>22s} | {'bps  Sharpe  hit%':>22s}")
    for h,_ in HORIZ:
        line=f"{h:5s} |"
        for pre in ["gross","nete","netrt"]:
            c=f"{pre}_{h}"
            if c not in port.columns: line+=f" {'--':>22s} |"; continue
            s=port[c].dropna()
            mean=s.mean(); shp=s.mean()/s.std()*np.sqrt(252) if s.std()>0 else np.nan
            hit=100*(s>0).mean()
            line+=f" {mean:>6.2f} {shp:>6.2f} {hit:>4.0f}% |"
        print(line)
    # bootstrap CI on the headline (3m, cross-entry)
    if "nete_3m" in port.columns:
        lo,hi=boot_ci(port["nete_3m"]); print(f"\n3m cross-entry mean 95% CI (date bootstrap): [{lo:+.2f}, {hi:+.2f}] bps/day")
    print(f"saved daily portfolio: {args.out}_daily.csv")


if __name__ == "__main__":
    main()
