#!/usr/bin/env python3
"""
tod_coi_test.py — Referee B9 (time-of-day stratification) + B11 (count vs volume COI).

Sharded single pass over the master burst files. For each burst it uses the
start time (seconds from midnight), direction, volume, and the 3-minute markout
(Direction*(Mid_3m - M_end)/M_end, kappa=0). Emits:
  B9: per intraday-window burst counts + summed 3-min markout (justifies the
      3:50 pm dead-zone: bursts cluster near the close and their forward window
      cannot mature).
  B11: per ticker-day count-COI = mean(Direction) and volume-COI =
       sum(Dir*Vol)/sum(Vol), for a downstream IC-vs-CLOP comparison.

Usage: tod_coi_test.py shard <id> <n>  |  tod_coi_test.py merge <n>
"""
import glob, sys, os, math, numpy as np, pandas as pd
from scipy import stats

MASTER = "results/bursts_*_baseline_unfiltered.csv"
OUT = "results/research/todcoi"
# intraday windows (seconds from midnight): 9:30=34200 ... 16:00=57600
WIN = [("open_0930_1000", 34200, 36000), ("mid_1000_1530", 36000, 55800),
       ("pre_1530_1550", 55800, 57000), ("deadzone_1550_1600", 57000, 57600)]


def process_shard(sid, n):
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(MASTER))[sid - 1::n]
    wrows = []          # per-ticker window aggregates
    coirows = []        # per ticker-day COI
    for f in files:
        tk = f.split('/')[-1].replace('bursts_', '').replace('_baseline_unfiltered.csv', '')
        try:
            d = pd.read_csv(f, usecols=["Date", "StartTime", "Direction", "Volume",
                                        "EndBid", "EndAsk", "Mid_3m"])
        except Exception:
            continue
        if d.empty: continue
        d = d[(d["EndBid"] > 0) & (d["EndAsk"] > 0)].copy()
        mend = (d["EndBid"].values + d["EndAsk"].values) / 2.0
        mk = d["Direction"].values * (d["Mid_3m"].values - mend) / mend * 1e4
        st = d["StartTime"].values
        for name, lo, hi in WIN:
            m = (st >= lo) & (st < hi)
            v = mk[m]; v = v[np.isfinite(v)]
            wrows.append((tk, name, int(m.sum()), float(np.nansum(v)), int(np.isfinite(v).sum())))
        # per ticker-day COI
        d["di"] = pd.to_datetime(d["Date"]).dt.strftime("%Y%m%d").astype(int)
        g = d.groupby("di").apply(lambda x: pd.Series({
            "count_coi": np.sign(x["Direction"]).mean(),
            "vol_coi": (x["Direction"] * x["Volume"]).sum() / max(x["Volume"].sum(), 1),
            "nb": len(x)}))
        g = g.reset_index(); g["tk"] = tk
        coirows.append(g)
    pd.DataFrame(wrows, columns=["tk", "window", "n_bursts", "mk_sum", "mk_n"]
                 ).to_csv(f"{OUT}/win_{sid}.csv", index=False)
    (pd.concat(coirows, ignore_index=True) if coirows else pd.DataFrame()
     ).to_csv(f"{OUT}/coi_{sid}.csv", index=False)
    print(f"[shard {sid}/{n}] done: {len(files)} files")


def merge(n):
    W = pd.concat([pd.read_csv(f"{OUT}/win_{i}.csv") for i in range(1, n + 1)
                   if os.path.exists(f"{OUT}/win_{i}.csv")], ignore_index=True)
    agg = W.groupby("window").agg(n_bursts=("n_bursts", "sum"), mk_sum=("mk_sum", "sum"),
                                  mk_n=("mk_n", "sum")).reindex([w[0] for w in WIN])
    tot = agg["n_bursts"].sum()
    print("=== B9: burst distribution + 3-min markout by intraday window ===")
    print(f"{'window':22s} {'%of bursts':>11s} {'mean 3m markout (bps)':>22s}")
    for w in agg.index:
        share = 100 * agg.loc[w, "n_bursts"] / tot
        mk = agg.loc[w, "mk_sum"] / max(agg.loc[w, "mk_n"], 1)
        print(f"{w:22s} {share:>10.1f}% {mk:>+22.3f}")
    dz = 100 * agg.loc["deadzone_1550_1600", "n_bursts"] / tot
    print(f"  -> {dz:.1f}% of all bursts start in the 15:50-16:00 dead-zone; their 10-min $D_b$")
    print(f"     window cannot mature before the close, justifying the 3:50 pm cutoff.")

    C = pd.concat([pd.read_csv(f"{OUT}/coi_{i}.csv") for i in range(1, n + 1)
                   if os.path.exists(f"{OUT}/coi_{i}.csv")], ignore_index=True)
    close = pd.read_csv("close_all.csv", index_col="date"); close.index = close.index.astype(int)
    opn = pd.read_csv("open_all.csv", index_col="date"); opn.index = opn.index.astype(int)
    tdays = np.array(sorted(close.index))

    def clop(r):
        tk = r["tk"]; dt = int(r["di"])
        if tk not in close.columns or tk not in opn.columns: return np.nan
        j = np.searchsorted(tdays, dt, side="right")
        if j >= len(tdays): return np.nan
        c = close[tk].get(dt, np.nan); o = opn[tk].get(int(tdays[j]), np.nan)
        return (o - c) / c if (np.isfinite(c) and np.isfinite(o) and c > 0) else np.nan
    C["CLOP"] = C.apply(clop, axis=1)
    M = C.dropna(subset=["CLOP", "count_coi", "vol_coi"])
    print("\n=== B11: count-based vs volume-based COI (corr + IC vs next-day CLOP) ===")
    r_cv = M["count_coi"].corr(M["vol_coi"])
    print(f"  corr(count-COI, volume-COI) across {len(M):,} name-days: {r_cv:+.3f}")
    for lab, col in [("count-COI", "count_coi"), ("volume-COI", "vol_coi")]:
        ics = []
        for tk, g in M.groupby("tk"):
            if len(g) >= 30 and g[col].std() > 0:
                ics.append(stats.spearmanr(g[col], g["CLOP"]).correlation)
        ics = np.array([i for i in ics if np.isfinite(i)])
        # date-clustered pooled
        dic = []
        for di, g in M.groupby("di"):
            if len(g) >= 10 and g[col].std() > 0 and g["CLOP"].std() > 0:
                r = stats.spearmanr(g[col], g["CLOP"]).correlation
                if np.isfinite(r): dic.append(r)
        dic = np.array(dic)
        t = dic.mean() / (dic.std() / math.sqrt(len(dic))) if len(dic) > 1 else np.nan
        print(f"  {lab:11s}: per-name mean IC={ics.mean():+.4f} | date-clustered IC={dic.mean():+.4f} t={t:+.2f}")
    C.to_csv("results/research/todcoi_daily.csv", index=False)
    print("\nsaved: results/research/todcoi_daily.csv")


if __name__ == "__main__":
    if sys.argv[1] == "shard": process_shard(int(sys.argv[2]), int(sys.argv[3]))
    elif sys.argv[1] == "merge": merge(int(sys.argv[2]))
