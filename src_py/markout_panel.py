#!/usr/bin/env python3
"""
markout_panel.py — Reviewer R1/R4/R5: full-universe multi-horizon markout panel.

Directional post-burst markout (gross of costs) at fixed forward horizons from the
BURST-TERMINATION mid (not StartPrice): this isolates the *predictive* post-burst
drift and removes contamination by the burst's own residual impact (Reviewer R1).

For each burst:  markout_bps(h) = Direction * (mid_h - EndMid) / EndMid * 1e4
  EndMid = (EndBid + EndAsk)/2   (fallback EndPrice)
Horizons from C++ mid snapshots: 1m, 3m, 5m, 10m, tCLOSE (CloseMid).
Overnight (CLOP) / next-close (CLCL) are reverse-engineered from the permanence
labels (arcsinh(Volume*Direction*(exit-entry))), consistent with naive_baseline_markout.

Inference: naive per-burst t-stats are inflated (overlapping same-day bursts are
not independent). We aggregate to (ticker, day) means and CLUSTER-BOOTSTRAP over
trading DATES (resample dates w/ replacement, 1000 reps) for honest 95% CIs.

Also reports the StartPrice-entry markout for the intraday horizons to quantify the
own-impact contamination the referee flagged.

Usage:
  python3 src_py/markout_panel.py --tickers <csv or @file> --burst-dir results/ \
      --suffix baseline_filtered --out results/research/markout_panel_2026
"""
import argparse, os, sys, glob
import numpy as np
import pandas as pd

HORIZ_MID = [("1m","Mid_1m"),("3m","Mid_3m"),("5m","Mid_5m"),("10m","Mid_10m"),("tCLOSE","CloseMid")]
PERM_HORIZ = [("CLOP","Perm_CLOP"),("CLCL","Perm_CLCL")]
NEED = ["Date","Direction","Volume","StartPrice","EndPrice","EndBid","EndAsk",
        "Mid_1m","Mid_3m","Mid_5m","Mid_10m","CloseMid","Perm_CLOP","Perm_CLCL"]


def end_mid(df):
    m = (df["EndBid"].astype(float) + df["EndAsk"].astype(float)) / 2.0
    bad = ~(m > 0)
    m = m.where(~bad, df["EndPrice"].astype(float))
    return m


def markout_from(df, ref, exit_col):
    ref = ref.astype(float); ex = df[exit_col].astype(float); d = df["Direction"].astype(float)
    ok = (ref > 0) & ex.notna() & (d != 0)
    out = pd.Series(np.nan, index=df.index)
    out[ok] = d[ok] * (ex[ok] - ref[ok]) / ref[ok] * 1e4
    return out


def perm_markout(df, perm_col):
    # bps = Direction*(exit-entry)/entry*1e4 recovered from arcsinh permanence.
    raw = np.sinh(df[perm_col].astype(float))
    entry = df["StartPrice"].astype(float); vol = df["Volume"].astype(float)
    ok = (entry > 0) & (vol > 0) & df[perm_col].notna()
    out = pd.Series(np.nan, index=df.index)
    out[ok] = (raw[ok] / (vol[ok] * entry[ok]) * 1e4).clip(-500, 500)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", required=True, help="comma list, or @path to newline file")
    ap.add_argument("--burst-dir", default="results/")
    ap.add_argument("--suffix", default="baseline_filtered")
    ap.add_argument("--out", default="results/research/markout_panel_2026")
    ap.add_argument("--start-date", type=int, default=20220101)
    ap.add_argument("--end-date", type=int, default=20261231)
    ap.add_argument("--nboot", type=int, default=1000)
    args = ap.parse_args()

    if args.tickers.startswith("@"):
        tickers = [l.split()[0] for l in open(args.tickers[1:]) if l.strip() and not l.startswith("#")]
    else:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    all_h = [h for h,_ in HORIZ_MID] + [h for h,_ in PERM_HORIZ]
    panel_rows = []          # (Ticker, Date, n, endmid markouts..., start markouts intraday...)
    n_bursts_total = 0
    for tk in tickers:
        path = os.path.join(args.burst_dir, f"bursts_{tk}_{args.suffix}.csv")
        if not os.path.exists(path):
            continue
        try:
            avail = pd.read_csv(path, nrows=0).columns
            df = pd.read_csv(path, usecols=[c for c in NEED if c in avail])
        except Exception:
            continue
        if df.empty or "Direction" not in df.columns:
            continue
        try:
            df["Date"] = df["Date"].astype(int)
        except (ValueError, TypeError):
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y%m%d").astype(int)
        df = df[(df["Date"] >= args.start_date) & (df["Date"] <= args.end_date)]
        df = df[df["Direction"].fillna(0) != 0]
        if df.empty:
            continue
        n_bursts_total += len(df)
        em = end_mid(df)
        cols = {"Ticker": tk, "Date": df["Date"].values, "n": 1}
        d = pd.DataFrame({"Date": df["Date"].values})
        for h, col in HORIZ_MID:
            if col in df.columns:
                d[f"em_{h}"] = markout_from(df, em, col).values
                d[f"sp_{h}"] = markout_from(df, df["StartPrice"], col).values
        for h, col in PERM_HORIZ:
            if col in df.columns:
                d[f"em_{h}"] = perm_markout(df, col).values  # overnight: StartPrice-based (own-impact negligible vs gap)
        g = d.groupby("Date").mean(numeric_only=True)
        g["Ticker"] = tk; g["n"] = d.groupby("Date").size().values
        panel_rows.append(g.reset_index())

    if not panel_rows:
        print("ERROR: no data loaded"); sys.exit(1)
    P = pd.concat(panel_rows, ignore_index=True)
    P.to_csv(f"{args.out}.csv", index=False)
    print(f"panel: {len(P):,} ticker-days over {P['Ticker'].nunique()} names; total gated bursts={n_bursts_total:,}")

    dates = P["Date"].values
    uniq_dates = np.unique(dates)
    rng = np.random.default_rng(42)

    def cluster_boot(colvals):
        # resample DATES with replacement; mean over all ticker-days on sampled dates
        by_date = pd.Series(colvals, index=dates).dropna()
        gmean = by_date.groupby(level=0).mean()
        d_idx = gmean.index.values; d_val = gmean.values
        means = np.empty(args.nboot)
        for b in range(args.nboot):
            samp = rng.choice(len(d_idx), size=len(d_idx), replace=True)
            means[b] = np.nanmean(d_val[samp])
        return np.nanpercentile(means, 2.5), np.nanpercentile(means, 97.5)

    print("\n================ MULTI-HORIZON MARKOUT PANEL (gated bursts, gross of costs) ================")
    print(f"{'horizon':8s} {'mean(bps)':>10s} {'95% CI (cluster-boot)':>26s} {'%pos days':>10s} {'naive per-burst t*':>18s}")
    summary=[]
    for h in all_h:
        c=f"em_{h}"
        if c not in P.columns: continue
        v=P[c].dropna().values
        if len(v)<50: continue
        mean=v.mean()
        lo,hi=cluster_boot(P[c].values)
        pos=100*(v>0).mean()
        # naive t treats each ticker-day as independent (still conservative vs per-burst)
        t=mean/(v.std(ddof=1)/np.sqrt(len(v)))
        sig = "" if (lo<=0<=hi) else "  (excl 0)"
        print(f"{h:8s} {mean:>10.2f} {('['+format(lo,'+.2f')+', '+format(hi,'+.2f')+']'):>26s} {pos:>9.0f}% {t:>18.1f}{sig}")
        summary.append((h,mean,lo,hi,pos,t))
    pd.DataFrame(summary,columns=["horizon","mean_bps","ci_lo","ci_hi","pct_pos_days","td_t"]).to_csv(f"{args.out}_summary.csv",index=False)

    print("\n--- Own-impact contamination check (intraday): EndMid-entry vs StartPrice-entry ---")
    for h,_ in HORIZ_MID:
        ce,cs=f"em_{h}",f"sp_{h}"
        if ce in P.columns and cs in P.columns:
            print(f"  {h:6s} EndMid={P[ce].mean():+7.2f} bps   StartPrice={P[cs].mean():+7.2f} bps   contamination={P[cs].mean()-P[ce].mean():+7.2f} bps")
    print(f"\nsaved: {args.out}.csv  and  {args.out}_summary.csv")


if __name__ == "__main__":
    main()
