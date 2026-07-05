#!/usr/bin/env python3
"""
m4_closemid_target.py — Referee M4.  (sharded for a job array)

The paper's VSI target (VSI = arcsinh(Q·side·(P_tau - m_{t_b}))) is measured from
the BURST-START mid m_{t_b}. For the reg_clop (overnight) target that base mixes in
the burst's own contemporaneous impact + the burst->close drift, both realized
BEFORE the MOC entry and therefore UNEARNABLE, so Table 8's rho (0.088-0.199)
overstates association with the *tradable* close-to-open return.

Fix (M4): recompute the predictive metric against the tradable close-to-open return
measured from the CLOSE mid (CloseMid) instead of the burst-start mid.

Predictor (non-anticipating, known at burst end; transparent stand-in for the SGD's
end-of-burst features): the burst's realized directional end-impact
    I_b = side·(EndPrice - StartPrice).
Targets:
    INFLATED (paper, from burst-start):  side·(Open_next - StartPrice)
    TRADABLE (from close mid):           side·(Open_next - CloseMid)   [= code Perm_CLOP base]
VSI-form (arcsinh, Q-scaled) Spearman for both, + level decomposition.

Usage:
    python m4_closemid_target.py shard <id> <nshard>   # writes results/research/m4_part_<id>.{csv,daily.csv}
    python m4_closemid_target.py merge <nshard>        # prints summary from all parts
"""
import glob, sys, math, os, numpy as np, pandas as pd
from scipy.stats import spearmanr

MASTER = "results/bursts_*_baseline_unfiltered.csv"
OUT = "results/research"


def process_shard(shard_id, nshard):
    close = pd.read_csv("close_all.csv", index_col="date"); close.index = close.index.astype(int)
    open_ = pd.read_csv("open_all.csv", index_col="date"); open_.index = open_.index.astype(int)
    tdays = np.array(sorted(close.index), dtype=np.int64)
    files = sorted(glob.glob(MASTER))[shard_id - 1::nshard]

    per_name, daily_rows = [], []
    for f in files:
        tk = f.split('/')[-1].replace('bursts_', '').replace('_baseline_unfiltered.csv', '')
        if tk not in open_.columns: continue
        try:
            d = pd.read_csv(f, usecols=["Date", "Direction", "BurstVolume",
                                        "StartPrice", "EndPrice", "CloseMid"])
        except Exception:
            continue
        if d.empty: continue
        d = d[(d["StartPrice"] > 0) & (d["CloseMid"] > 0)].copy()
        di = pd.to_datetime(d["Date"]).dt.strftime("%Y%m%d").astype(int).values
        pos = np.searchsorted(tdays, di, side="right")
        ok = pos < len(tdays)
        d = d[ok].copy(); di = di[ok]; nd = tdays[pos[ok]]
        opx = open_[tk].reindex(nd).values
        good = np.isfinite(opx) & (opx > 0)
        d = d[good]; di = di[good]; opx = opx[good]
        if len(d) < 50: continue
        side = d["Direction"].values
        pred = side * (d["EndPrice"].values - d["StartPrice"].values)
        r_infl = side * (opx / d["StartPrice"].values - 1.0) * 1e4
        r_trad = side * (opx / d["CloseMid"].values - 1.0) * 1e4
        r_pre  = side * (d["CloseMid"].values / d["StartPrice"].values - 1.0) * 1e4
        Q = d["BurstVolume"].values
        vsi_infl = np.arcsinh(Q * side * (opx - d["StartPrice"].values))
        vsi_trad = np.arcsinh(Q * side * (opx - d["CloseMid"].values))

        def sp(a, b):
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 50 or np.std(a[m]) == 0 or np.std(b[m]) == 0: return (np.nan, np.nan)
            return spearmanr(a[m], b[m])
        rI, pI = sp(pred, vsi_infl); rT, pT = sp(pred, vsi_trad)
        per_name.append(dict(tk=tk, n=len(d), rho_infl=rI, p_infl=pI, rho_trad=rT, p_trad=pT,
                             mean_pre_bps=np.nanmean(r_pre), mean_trad_bps=np.nanmean(r_trad),
                             mean_infl_bps=np.nanmean(r_infl)))
        tmp = pd.DataFrame({"di": di, "pred": pred, "infl": vsi_infl, "trad": vsi_trad})
        g = tmp.groupby("di").mean(); g["tk"] = tk
        daily_rows.append(g.reset_index())

    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(per_name).to_csv(f"{OUT}/m4_part_{shard_id}.csv", index=False)
    (pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()
     ).to_csv(f"{OUT}/m4_daily_{shard_id}.csv", index=False)
    print(f"[shard {shard_id}/{nshard}] {len(per_name)} names -> parts written")


def merge(nshard):
    R = pd.concat([pd.read_csv(f"{OUT}/m4_part_{i}.csv")
                   for i in range(1, nshard + 1) if os.path.exists(f"{OUT}/m4_part_{i}.csv")],
                  ignore_index=True)
    R.to_csv(f"{OUT}/m4_closemid_target.csv", index=False)
    A = pd.concat([pd.read_csv(f"{OUT}/m4_daily_{i}.csv")
                   for i in range(1, nshard + 1) if os.path.exists(f"{OUT}/m4_daily_{i}.csv")],
                  ignore_index=True)

    def summ(col, plab):
        c = R[col].dropna()
        sigpos = ((R[col] > 0) & (R[plab] < 0.05)).mean() * 100
        return (f"mean={c.mean():+.3f} median={c.median():+.3f} "
                f"%pos={100*(c>0).mean():.0f}% %sig+={sigpos:.0f}%")

    print(f"=== M4: predictive Spearman rho, {len(R)} names — VSI(pred, target) ===")
    print("  predictor = burst realized directional end-impact (non-anticipating)")
    print(f"  INFLATED  target (paper, from burst-START mid):  {summ('rho_infl','p_infl')}")
    print(f"  TRADABLE  target (from CLOSE mid, close->open):  {summ('rho_trad','p_trad')}")

    def day_ic(sig, tgt):
        ics = []
        for di, grp in A.groupby("di"):
            m = grp[sig].notna() & grp[tgt].notna()
            if m.sum() >= 10 and grp.loc[m, sig].std() > 0 and grp.loc[m, tgt].std() > 0:
                ics.append(spearmanr(grp.loc[m, sig], grp.loc[m, tgt])[0])
        ics = np.array(ics)
        return ics.mean(), ics.mean()/(ics.std(ddof=1)/math.sqrt(len(ics))), len(ics)
    mi, ti, ni = day_ic("pred", "infl"); mt, tt, nt = day_ic("pred", "trad")
    print(f"\n  Date-clustered pooled IC (cross-name, per day; effective N=#days):")
    print(f"    INFLATED: mean IC={mi:+.4f}  t={ti:+.2f}  ({ni} days)")
    print(f"    TRADABLE: mean IC={mt:+.4f}  t={tt:+.2f}  ({nt} days)")

    print(f"\n  Target level decomposition (directional bps, per-name mean):")
    print(f"    pre-entry drift side*(CloseMid-StartPrice):  {R['mean_pre_bps'].mean():+.1f} bps  (UNEARNABLE)")
    print(f"    tradable overnight side*(Open-CloseMid):      {R['mean_trad_bps'].mean():+.1f} bps  (earnable)")
    print(f"    burst-start->open side*(Open-StartPrice):     {R['mean_infl_bps'].mean():+.1f} bps  (paper base)")
    share = 100 * abs(R['mean_pre_bps'].mean()) / max(1e-9, abs(R['mean_infl_bps'].mean()))
    print(f"    -> pre-entry (unearnable) piece = {share:.0f}% of the burst-start->open target level")
    print(f"\nsaved: {OUT}/m4_closemid_target.csv")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "shard":
        process_shard(int(sys.argv[2]), int(sys.argv[3]))
    elif mode == "merge":
        merge(int(sys.argv[2]))
