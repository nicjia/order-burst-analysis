#!/usr/bin/env python3
"""forecast-eval-v1: how much of the burst forecast of the decision-to-close move is burst information?

Per trade burst, target = signed mid move from T_dec to the close (d_close, bps), raw and in excess of the
equal-weight intraday market over the same interval (results/market_index). Nested feature sets:
  CTRL   known non-burst predictors: move since the open (intraday reversal), 30-min pre-move, time of day, spread
  IMPACT the burst's own price path: peak impact, displacement at 1 and 10 min, mean displacement, retained ratio
  STRUCT burst structure and the new ideas: size/ADV, children, duration, fingerprint, program score, linkage,
         hidden/truncated share, yesterday's repeated-clip programs
Gradient boosting (depth 3, fixed) on the DEV-winsorized target, trained on DEV 2017-19, scored on VAL 2020-21 and
TEST 2022-25. Metrics: daily rank IC (NW t), OOS R^2 vs a zero forecast, sign hit rate, gross mid-to-mid
top-minus-bottom decile spread (NW t), and the paired daily IC gain of each richer set over CTRL. No costs.
"""
import sys as _sys, pathlib as _pl  # shared project modules live in src_py/
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "src_py"))
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import p4_analyze as PA
import fp_multiday_h1 as H
import burst279_daily as BD

ROOT = Path(__file__).resolve().parents[3]
MI = ROOT / "results" / "burst_forecasting" / "market_index"
CTRL = ["own_open_dec_bps", "pre30_bps", "tod", "spread_dec"]
IMPACT = ["peak_bps", "d60_bps", "d600_bps", "dmean_bps", "ratio"]
STRUCT = ["log_q_adv", "log_n", "log_dur", "mode_share", "mode_nonround", "program_score", "link_back",
          "truncated_share", "hidden_share", "prog_same_prev", "prog_opp_prev"]
SETS = {"CTRL": CTRL, "IMPACT": IMPACT, "STRUCT": STRUCT, "CTRL+IMPACT": CTRL + IMPACT, "FULL": CTRL + IMPACT + STRUCT}


def add_excess(d, cell):
    p = MI / ("%s_mkt.npz" % cell)
    if not p.exists():
        d["d_close_x"] = np.nan
        return d
    z = np.load(p); idx = dict(zip(z["dates"], z["idx"]))
    k = np.clip(np.ceil((d.t_dec.to_numpy() - 34200.0) / 60.0).astype(int) - 1, 0, 389)
    mk = np.array([idx[dt][389] - idx[dt][kk] if dt in idx else np.nan for dt, kk in zip(d.date.to_numpy(), k)])
    d["d_close_x"] = d.d_close - d.side * mk * 1e4
    return d


def metrics(df, pred, y):
    t = pd.DataFrame(dict(date=df.date.to_numpy(), p=pred, y=df[y].to_numpy())).dropna()
    ic, spread, rows = [], [], []
    for d, g in t.groupby("date"):
        if len(g) < 30:
            continue
        ic.append(g.p.rank().corr(g.y.rank()))
        r = g.p.rank(method="first"); k = max(1, len(g) // 10)
        spread.append(g.y[r > len(g) - k].mean() - g.y[r <= k].mean())
    ic, spread = np.array(ic), np.array(spread)
    r2 = 1 - np.sum((t.y - t.p) ** 2) / np.sum(t.y ** 2)
    return dict(ic=PA.nw_t(ic), spread=PA.nw_t(spread), r2_oos=float(r2), hit=float((np.sign(t.p) == np.sign(t.y)).mean()),
                _ic_series=pd.Series(ic))


def main():
    t0 = time.time(); cal = H.calendar()
    tr = add_excess(BD.load("DEV", cal, 400000), "DEV")
    cells = {c: add_excess(BD.load(c, cal, 1500000), c) for c in ("VAL", "TEST")}
    res = {}
    for y in ("d_close", "d_close_x"):
        if tr[y].notna().sum() < 1000:
            print("skip", y, "(market index not available)"); continue
        t = tr[tr[y].notna()]
        lo, hi = t[y].quantile([0.01, 0.99])
        models = {k: HistGradientBoostingRegressor(max_depth=3, max_iter=250, learning_rate=0.05, min_samples_leaf=200,
                                                   early_stopping=False, random_state=1).fit(t[c].to_numpy(float), t[y].clip(lo, hi).to_numpy())
                  for k, c in SETS.items()}
        print("\n#### target %s (%s)  trained (%.0fs)" % (y, "raw" if y == "d_close" else "market excess", time.time() - t0), flush=True)
        for cell, te in cells.items():
            te = te[te[y].notna()].copy(); te[y] = te[y].clip(lo, hi)
            out = {k: metrics(te, m.predict(te[SETS[k]].to_numpy(float)), y) for k, m in models.items()}
            print("  %s  bursts %d" % (cell, len(te)))
            print("    %-12s %8s %6s %9s %8s %9s %6s" % ("feature set", "IC", "t", "R2 oos", "hit", "D10-D1", "t"))
            for k, v in out.items():
                print("    %-12s %+8.4f %6.2f %+9.5f %8.4f %+9.2f %6.2f" % (k, v["ic"]["mean"], v["ic"]["t"], v["r2_oos"], v["hit"], v["spread"]["mean"], v["spread"]["t"]))
            base = out["CTRL"]["_ic_series"]
            for k in ("CTRL+IMPACT", "FULL", "IMPACT", "STRUCT"):
                dlt = out[k]["_ic_series"] - base
                g = PA.nw_t(dlt.to_numpy())
                print("    IC gain %-12s over CTRL: %+.4f (t %.2f)" % (k, g["mean"], g["t"]))
                out[k]["ic_gain_over_ctrl"] = g
            dlt = out["FULL"]["_ic_series"] - out["CTRL+IMPACT"]["_ic_series"]
            g = PA.nw_t(dlt.to_numpy()); out["FULL"]["ic_gain_over_ctrl_impact"] = g
            print("    IC gain FULL over CTRL+IMPACT (structure + new ideas): %+.4f (t %.2f)" % (g["mean"], g["t"]))
            for v in out.values():
                v.pop("_ic_series", None)
            res["%s|%s" % (y, cell)] = out
    (ROOT / "results" / "burst_forecasting" / "burst279_v1" / "forecast_eval.json").write_text(json.dumps(res, indent=1, default=float))
    print("done %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
