#!/usr/bin/env python3
"""
m10_sign_audit.py — Referee M10: sign-convention audit + net-short tilt.

Referee: "informational" flow is sell-signed on ~81% of name-days in a bull
market; LOBSTER Direction = side of the RESTING order (execution vs bid = market
SELL). If our aggressor sign is right, the CONTEMPORANEOUS correlation
    corr(daily signed burst flow_day_t, same-day return_t)
should be strongly POSITIVE nearly everywhere (net buying pushes price up the
same day = mechanical price impact). If it is ~0 or negative, the 81% short tilt
and the "anti-calibrated" IC have a mundane sign cause.

Also documents the panel sign-flip: results/regime/regime_classifications.csv
FlipSign=-1 names (mean-reverting cluster) have their COI inverted in the Table 11
panel — this must appear in the table notes.
"""
import glob, numpy as np, pandas as pd
from scipy.stats import pearsonr

GLOB = "results/sgd_backtests_oos/*_reg_clop_b1p0_i0p5_debug_trades.csv"


def load_daily_flow():
    rows = []
    for f in glob.glob(GLOB):
        tk = f.split('/')[-1].split('_reg_clop')[0]
        try:
            d = pd.read_csv(f, usecols=["day", "flow_signal", "side"])
        except Exception:
            continue
        if d.empty: continue
        d["di"] = pd.to_datetime(d["day"]).dt.strftime("%Y%m%d").astype(int)
        g = d.groupby("di").agg(flow=("flow_signal", "sum"), side=("side", "sum"))
        g["net_dir"] = np.sign(g["side"]); g["tk"] = tk
        rows.append(g.reset_index()[["di", "tk", "flow", "net_dir"]])
    return pd.concat(rows, ignore_index=True)


def main():
    A = load_daily_flow()
    close = pd.read_csv("close_all.csv", index_col="date"); close.index = close.index.astype(int)
    open_ = pd.read_csv("open_all.csv", index_col="date"); open_.index = open_.index.astype(int)
    names = sorted(set(A["tk"]) & set(close.columns) & set(open_.columns))

    cc = close.pct_change(fill_method=None)                  # prev close -> di close
    oc = (close / open_ - 1.0)                                # same-day open -> close (intraday)

    recs = []
    for tk in names:
        a = A[A["tk"] == tk].set_index("di")
        cc_t = cc[tk].reindex(a.index); oc_t = oc[tk].reindex(a.index)
        def cor(x, y):
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 30 or np.nanstd(x[m]) == 0: return (np.nan, np.nan)
            return pearsonr(x[m], y[m])
        r_oc, p_oc = cor(a["flow"].values, oc_t.values)
        r_cc, p_cc = cor(a["flow"].values, cc_t.values)
        r_dir, _   = cor(a["net_dir"].values.astype(float), oc_t.values)
        recs.append(dict(tk=tk, n=len(a), flow_oc=r_oc, p_oc=p_oc, flow_cc=r_cc,
                         dir_oc=r_dir, mean_dir=a["net_dir"].mean(),
                         frac_short=(a["net_dir"] < 0).mean()))
    R = pd.DataFrame(recs)
    R.to_csv("results/research/m10_sign_audit.csv", index=False)

    def summ(col, plab):
        c = R[col].dropna()
        line = (f"  {col:10s}: mean={c.mean():+.3f} median={c.median():+.3f} "
                f"%pos={100*(c>0).mean():.0f}%")
        if plab and plab in R.columns:
            sigpos = ((R[col] > 0) & (R[plab] < 0.05)).mean() * 100
            signeg = ((R[col] < 0) & (R[plab] < 0.05)).mean() * 100
            line += f"  %sig+={sigpos:.0f}% %sig-={signeg:.0f}%"
        print(line)

    print(f"=== M10 sign audit: contemporaneous corr(signed flow, same-day return), {len(R)} names ===")
    print("  (sign convention CORRECT  <=>  strongly POSITIVE nearly everywhere)")
    summ("flow_oc", "p_oc")   # flow vs same-day open->close (intraday) — the cleanest contemporaneous test
    summ("flow_cc", "p_cc")   # flow vs same-day close->close
    summ("dir_oc", None)      # direction-only vs same-day open->close

    print("\n=== 81% net-short tilt audit ===")
    nd = A["net_dir"]
    print(f"  overall E[net_dir] = {nd.mean():+.3f}; fraction of name-days net SHORT = "
          f"{100*(nd<0).mean():.0f}% (referee cites ~81%)")
    print(f"  per-name mean(net_dir): median={R['mean_dir'].median():+.3f}, "
          f"%names net-short = {100*(R['mean_dir']<0).mean():.0f}%")

    # panel sign-flip disclosure
    reg = pd.read_csv("results/regime/regime_classifications.csv")
    flipped = reg.loc[reg["FlipSign"] == -1, "Ticker"].astype(str).tolist()
    print("\n=== Panel sign-flip disclosure (for Table 11 notes) ===")
    print(f"  {len(flipped)} of {len(reg)} names have FlipSign=-1 (data-driven mean-reverting "
          f"cluster, regime_classifier.py: Spearman(burst dir, NEXT-day ret)<0).")
    print(f"  Their signed COI is multiplied by -1 in the panel (--regime-csv). Names (first 20): "
          f"{', '.join(sorted(flipped)[:20])} ...")
    print("  => MUST be stated in Table 11 notes: the panel applies a data-driven per-name sign flip.")
    print("\nsaved: results/research/m10_sign_audit.csv")


if __name__ == "__main__":
    main()
