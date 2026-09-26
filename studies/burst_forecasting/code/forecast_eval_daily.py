#!/usr/bin/env python3
"""forecast-eval-daily: does anything about today's bursts forecast TOMORROW's close-to-close market-excess return?

Name-day level, trade family. Everything known at today's close. Nested sets:
  CTRL  today's return (reversal), 5-day return, 20-day volatility and turnover, log cap, closing spread
  FLOW  order-flow imbalance of ALL signed trade packets up to 15:50, (buy - sell) / (buy + sell)
  BURST burst flow / ADV to 15:50: all bursts, large bursts, informative (kappa 0.5), non-informative large,
        log burst count
Target: next day's CRSP return minus the value-weighted universe (bps); raw next-day return also reported.
Gradient boosting trained on DEV 2017-19, scored on VAL 2020-21 and TEST 2022-25; daily rank IC (NW t),
decile spread, and the paired IC gain of each richer set.
"""
import sys as _sys, pathlib as _pl  # shared project modules live in src_py/
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "src_py"))
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import p4_analyze as PA
import earnings_flow as E
import forecast_eval as F

ROOT = Path(__file__).resolve().parents[3]
AGG = ROOT / "results" / "p4_revisit_v1" / "agg"
CTRL = ["dlyret_bps", "ret_lag5", "sigma20", "turn20", "log_cap", "spread_close"]
FLOW = ["ofi_all"]
BURST = ["x_all", "x_large", "x_info", "x_non", "log_nb"]
SETS = {"CTRL": CTRL, "CTRL+FLOW": CTRL + FLOW, "CTRL+FLOW+BURST": CTRL + FLOW + BURST, "BURST": BURST, "FLOW": FLOW}


def load(cell):
    cols = ["permno", "date", "family", "S_all_1550", "S_large_1550", "S_info_1550_k50", "n_bursts", "buy_1550",
            "sell_1550", "adv20", "sigma20", "turn20", "ret_lag5", "cap_lag1", "dlyret", "spread_close",
            "ret_next"]
    d = pd.read_csv(AGG / cell / ("nameday_%s.csv.gz" % cell), usecols=cols, dtype={"date": str})
    d = d[d.family == "T"].drop(columns="family")
    for c in ("S_all_1550", "S_large_1550", "S_info_1550_k50"):
        d[c] = d[c].fillna(0.0)
    d["x_all"] = d.S_all_1550 / d.adv20; d["x_large"] = d.S_large_1550 / d.adv20
    d["x_info"] = d.S_info_1550_k50 / d.adv20; d["x_non"] = (d.S_large_1550 - d.S_info_1550_k50) / d.adv20
    d["log_nb"] = np.log1p(d.n_bursts)
    tot = d.buy_1550 + d.sell_1550
    d["ofi_all"] = np.where(tot > 0, (d.buy_1550 - d.sell_1550) / tot, np.nan)
    d["dlyret_bps"] = d.dlyret * 1e4; d["log_cap"] = np.log(d.cap_lag1)
    years = sorted(d.date.str[:4].astype(int).unique())
    c, _ = E.returns(years + [max(years) + 1])
    c = c.sort_values(["permno", "date"])
    c["ar_next"] = c.groupby("permno").ar.shift(-1) * 1e4
    d = d.merge(c[["permno", "date", "ar_next"]], on=["permno", "date"], how="left")
    d["ret_next_bps"] = d.ret_next * 1e4
    return d.replace([np.inf, -np.inf], np.nan)


def main():
    tr = load("DEV"); cells = {c: load(c) for c in ("VAL", "TEST")}
    res = {}
    for y in ("ar_next", "ret_next_bps"):
        t = tr[tr[y].notna()]; lo, hi = t[y].quantile([0.01, 0.99])
        models = {k: HistGradientBoostingRegressor(max_depth=3, max_iter=250, learning_rate=0.05, min_samples_leaf=200,
                                                   early_stopping=False, random_state=1).fit(t[c].to_numpy(float), t[y].clip(lo, hi).to_numpy())
                  for k, c in SETS.items()}
        print("\n#### target %s (%s)" % (y, "next-day market excess" if y == "ar_next" else "next-day raw"))
        for cell, te in cells.items():
            te = te[te[y].notna()].copy(); te[y] = te[y].clip(lo, hi)
            out = {k: F.metrics(te, m.predict(te[SETS[k]].to_numpy(float)), y) for k, m in models.items()}
            print("  %s  name-days %d" % (cell, len(te)))
            print("    %-16s %8s %6s %9s %8s %9s %6s" % ("feature set", "IC", "t", "R2 oos", "hit", "D10-D1", "t"))
            for k, v in out.items():
                print("    %-16s %+8.4f %6.2f %+9.5f %8.4f %+9.2f %6.2f" % (k, v["ic"]["mean"], v["ic"]["t"], v["r2_oos"], v["hit"], v["spread"]["mean"], v["spread"]["t"]))
            for a, b in (("CTRL+FLOW", "CTRL"), ("CTRL+FLOW+BURST", "CTRL+FLOW")):
                g = PA.nw_t((out[a]["_ic_series"] - out[b]["_ic_series"]).to_numpy())
                print("    IC gain %s over %s: %+.4f (t %.2f)" % (a, b, g["mean"], g["t"]))
                out[a]["gain_over_" + b] = g
            for v in out.values():
                v.pop("_ic_series", None)
            res["%s|%s" % (y, cell)] = out
    (ROOT / "results" / "burst_forecasting" / "burst279_v1" / "forecast_eval_daily.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
