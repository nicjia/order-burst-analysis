#!/usr/bin/env python3
"""
m5m6_inference.py — Referee M5 (honest inference) + M6 (OOS factor alpha).

Reuses the deployed reversal machinery (m7_reversal_baseline) and the existing
paper statistics helpers:
  * multiple_testing_correction.run_pnl_inference  -> Lo(2002) SE, Deflated Sharpe
    (Bailey-Lopez de Prado, deflated for the config search), circular block
    bootstrap CI (block = date cluster).
  * panel_regression.factor_adjust_long_short      -> FF5+MOM Newey-West alpha.

Produces, for the tick-constrained burst-flow reversal (bottom-100, H=20):
  M5(iii) DSR-z + Lo(2002) t for BOTH the full-sample and the walk-forward OOS series.
  M6      FF5+MOM alpha on the FULL-SAMPLE (in-sample) vs the WALK-FORWARD OOS series
          -> the OOS alpha (wider CI) is the number that should headline.
  M5(i)   pseudo-replication demo: naive pooled per-name-day IC t  vs  date-clustered
          IC t (effective N = #days), on flow -> next-day return.
"""
import os, glob, math, numpy as np, pandas as pd
import m7_reversal_baseline as m7
from multiple_testing_correction import run_pnl_inference, lo_sharpe_se, deflated_sharpe_ratio
from panel_regression import factor_adjust_long_short

N_TRIALS = 75          # size of the reversal config search (FINDINGS 4b)
FF5 = "data_factors/F-F_Research_Data_5_Factors_2x3_daily.csv"
MOM = "data_factors/F-F_Momentum_Factor_daily.csv"
MERGED_FACTORS = "results/research/ff5_mom_merged.csv"


def parse_french(path, names):
    """parse a Ken French daily CSV (skip preamble); rows keyed by 8-digit date.
    returns DataFrame indexed by int Date, values converted to DECIMAL (/100)."""
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 2: continue
            d = parts[0].strip()
            if len(d) == 8 and d.isdigit():
                try:
                    vals = [float(x) for x in parts[1:1 + len(names)]]
                except ValueError:
                    continue
                if len(vals) == len(names):
                    rows.append([int(d)] + vals)
    df = pd.DataFrame(rows, columns=["Date"] + names).set_index("Date")
    return df / 100.0


def build_factor_csv():
    ff5 = parse_french(FF5, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
    mom = parse_french(MOM, ["Mom"])
    merged = ff5.join(mom, how="inner").reset_index()
    merged.to_csv(MERGED_FACTORS, index=False)
    print(f"[factors] merged FF5+MOM {merged['Date'].min()}..{merged['Date'].max()} "
          f"({len(merged)} days) -> {MERGED_FACTORS}")
    return MERGED_FACTORS


def report_factor(tag, series, factor_csv):
    r = factor_adjust_long_short(series, factor_csv)
    if r is None:
        print(f"  [{tag}] factor regression could not run"); return
    a_bps = r["beta"][0] * 1e4
    a_t = r["t_stat"][0]
    print(f"\n  --- FF5+MOM alpha: {tag} ---")
    print(f"    alpha = {a_bps:+.2f} bps/day  (t={a_t:+.2f}, NW lags={r['nw_lags']}, "
          f"n={r['n_obs']}, ann={r['alpha_ann']*100:+.1f}%/yr, IR={r['info_ratio']:+.2f}, "
          f"R2={r['r_squared']:.3f})")
    for nm, b, t in zip(r["names"][1:], r["beta"][1:], r["t_stat"][1:]):
        print(f"      {nm:<7s} beta={b:+.3f}  t={t:+.2f}")


def pseudo_replication_demo(FL, R, dis, cols):
    """M5(i): naive pooled name-day IC t vs date-clustered IC t (effective N=#days)."""
    Zf = m7.zscore(FL)
    Rn = R.shift(-1)                       # flow on day di -> next-day close-to-close return
    x = Zf.values.ravel(); y = Rn.values.ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    r_pool = np.corrcoef(x, y)[0, 1]; Nnd = len(x)
    t_pool = r_pool * math.sqrt((Nnd - 2) / max(1e-12, 1 - r_pool**2))
    # date-clustered: one IC per day, t over #days
    ics = []
    for di in dis:
        a = m7.zscore(FL.loc[[di]]).iloc[0]; b = R.shift(-1).loc[di]
        j = a.notna() & b.notna()
        if j.sum() >= 10:
            ics.append(np.corrcoef(a[j], b[j])[0, 1])
    ics = np.array(ics); Nd = len(ics)
    t_day = ics.mean() / (ics.std(ddof=1) / math.sqrt(Nd))
    print("\n" + "=" * 100)
    print("  M5(i) PSEUDO-REPLICATION: flow -> next-day return IC")
    print("=" * 100)
    print(f"  Naive pooled (treats {Nnd:,} name-days independent): "
          f"r={r_pool:+.4f}, t={t_pool:+.2f}")
    print(f"  Date-clustered (effective N = {Nd} days):            "
          f"mean IC={ics.mean():+.4f}, t={t_day:+.2f}")
    print(f"  -> pooled |t| inflated {abs(t_pool)/max(abs(t_day),1e-9):.0f}x by pseudo-replication.")


def main():
    os.makedirs("results/research", exist_ok=True)
    dis, cols, FL, PR, SD, R, cpx, close = m7.load_panels()
    Zflow = m7.zscore(FL)
    Mfull = m7.full_sample_universe(cpx, dis, cols, m7.BOTTOM_K)
    Mwf   = m7.walkforward_universe(cpx, dis, cols, m7.BOTTOM_K)
    oos_days = [d for d in dis if d >= dis[m7.BURN]]

    full = m7.strat_returns(Zflow, R, Mfull)
    full = full[full != 0].dropna()
    oos = m7.strat_returns(Zflow, R, Mwf).loc[oos_days]
    oos = oos[oos != 0].dropna()

    # save series (day, daily_pnl) so run_pnl_inference can load them
    pd.DataFrame({"day": full.index, "daily_pnl": full.values}).to_csv(
        "results/research/reversal_full_pnl.csv", index=False)
    pd.DataFrame({"day": oos.index, "daily_pnl": oos.values}).to_csv(
        "results/research/reversal_oos_pnl.csv", index=False)

    print("############ M5(iii): DSR / Lo(2002) — FULL-SAMPLE (in-sample selection) ############")
    run_pnl_inference("results/research/reversal_full_pnl.csv", N_TRIALS)
    print("\n############ M5(iii): DSR / Lo(2002) — WALK-FORWARD OOS (2023-2026) ############")
    run_pnl_inference("results/research/reversal_oos_pnl.csv", N_TRIALS)

    factor_csv = build_factor_csv()
    print("\n############ M6: FF5+MOM factor alpha ############")
    report_factor("FULL-SAMPLE (in-sample, = current Table 13)", full, factor_csv)
    report_factor("WALK-FORWARD OOS (M6 primary number)", oos, factor_csv)

    pseudo_replication_demo(FL, R, dis, cols)


if __name__ == "__main__":
    main()
