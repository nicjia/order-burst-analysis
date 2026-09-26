#!/usr/bin/env python3
"""forecast-power-v1: forecasting power of real-time burst features — no trading, no fills, no costs.

(1) What carries the forecast. For the strongest definitions and every horizon, gradient boosting on all v3 features;
    group ablation (drop one feature group, refit) and the IC lost; out-of-sample R^2 against a zero forecast and
    against the training mean; sign hit rate. Evaluated on the 2024 test stocks and on every 2025 stock.
(2) Volatility. The same features forecasting the SIZE of the next move, |signed move|, at +60 s, +5 min, +30 min —
    burst features vs controls + book only.
(3) Stability. IC by year (2024 / 2025), by spread tercile (tick constraint), by time of day, and the share of
    stocks with a positive IC.
Train: the train stocks, 2022-23 (v3 panel). Gradient boosting depth 3, 200 iterations, fixed.
"""
import sys as _sys, pathlib as _pl  # shared project modules live in src_py/
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "src_py"))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import p4_analyze as PA
import burst_defs_raw2_model as M
import burst_defs_raw3_model as V
import confirm_2025 as C

GROUPS = {"controls": V.CTRL,
          "book": V.BOOK,
          "burst price path": ["move_during", "pre60", "opp_consumed", "spread_change"],
          "burst structure": ["log_n", "log_dur", "log_q_adv"],
          "size/timing regularity": ["mode_share", "nonround", "size_cv", "size_to_depth", "iat_cv", "iat_med", "phase_R"]}
ALL = sum(GROUPS.values(), [])
DEFS = ("early5", "run0.01", "levelclear", "cancel", "hidden", "hawkesfix")
HOR = ("r10", "r60", "r300", "r1800_x", "r_close_x")


def r2(y, p, mean):
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    sse = np.sum((y - p) ** 2)
    return 1 - sse / np.sum(y ** 2), 1 - sse / np.sum((y - mean) ** 2)


def main():
    train = set(pd.read_csv(M.ROOT / "results" / "p4_revisit_v1" / "jobs" / "DEFS_RAW_train_names.txt", header=None)[0])
    a = C.prep(M.D / "burst_defs_raw3" / "RAW3_events.csv.gz")
    b = C.prep(M.D / "burst_defs_raw3" / "Y2025_events.csv.gz")
    tr_all = a[a.permno.isin(train) & a.date.str[:4].isin(["2022", "2023"])]
    tests = {"2024 test stocks": a[~a.permno.isin(train) & (a.date.str[:4] == "2024")], "2025 all stocks": b[b.date.str[:4] == "2025"]}
    rows, abl, vol, stab = [], [], [], []
    out = M.D / "burst_defs_raw3"
    for dn in DEFS:
        tr = tr_all[tr_all.defn == dn]
        if len(tr) < 3000:
            print(dn, "not in the v3 panel, skipped", flush=True)
            continue
        for y in HOR:
            m, lo, hi = M.hgb(tr, ALL, y)
            ymean = float(tr[y].clip(lo, hi).mean())
            ablated = {g: M.hgb(tr, [c for c in ALL if c not in cols], y)[0] for g, cols in GROUPS.items()}
            for tn, te_all in tests.items():
                e = te_all[(te_all.defn == dn) & te_all[y].notna()]
                yy = e[y].clip(lo, hi).to_numpy(); p = m.predict(e[ALL].to_numpy(float))
                s, ic, sp = M.daily_ic(e.date.to_numpy(), p, yy)
                r2z, r2m = r2(yy, p, ymean)
                rows.append(dict(defn=dn, horizon=y, test=tn, n=int(len(e)), ic=ic["mean"], t=ic["t"], r2_vs_zero=r2z,
                                 r2_vs_mean=r2m, hit=float(np.mean(np.sign(p) == np.sign(yy))), d10_d1_bps=sp["mean"]))
                for g, mg in ablated.items():
                    cols = [c for c in ALL if c not in GROUPS[g]]
                    s2, ic2, _ = M.daily_ic(e.date.to_numpy(), mg.predict(e[cols].to_numpy(float)), yy)
                    d = PA.nw_t((s - s2).to_numpy())
                    abl.append(dict(defn=dn, horizon=y, test=tn, dropped=g, ic_full=ic["mean"], ic_without=ic2["mean"],
                                    ic_lost=d["mean"], t_lost=d["t"]))
                if tn == "2025 all stocks":
                    e2 = e.assign(p=p, yy=yy)
                    e2["spread_terc"] = pd.qcut(e2.spread_dec.rank(method="first"), 3, labels=["tight", "mid", "wide"])
                    e2["tod_b"] = pd.cut(e2.tod, [0, 0.25, 0.75, 1.01], labels=["open-11:07", "midday", "14:52-close"])
                    for col in ("spread_terc", "tod_b"):
                        for k, g in e2.groupby(col, observed=True):
                            _, ic3, _ = M.daily_ic(g.date.to_numpy(), g.p.to_numpy(), g.yy.to_numpy())
                            stab.append(dict(defn=dn, horizon=y, split=col, bucket=str(k), ic=ic3["mean"], t=ic3["t"]))
                    per_stock = e2.groupby("permno")[["p", "yy"]].apply(lambda g: g.p.rank().corr(g.yy.rank()) if len(g) > 100 else np.nan).dropna()
                    stab.append(dict(defn=dn, horizon=y, split="stocks", bucket="share with IC > 0 (%d stocks)" % len(per_stock),
                                     ic=float((per_stock > 0).mean()), t=np.nan))
        # volatility: size of the next move
        for y in ("r60", "r300", "r1800_x"):
            tv = tr.assign(v=tr[y].abs()); lo_, hi_ = tv.v.quantile([0.01, 0.99])
            full, _, _ = M.hgb(tv, ALL, "v"); base, _, _ = M.hgb(tv, V.CTRL + V.BOOK, "v")
            for tn, te_all in tests.items():
                e = te_all[(te_all.defn == dn) & te_all[y].notna()].assign(v=lambda x: x[y].abs())
                _, icf, _ = M.daily_ic(e.date.to_numpy(), full.predict(e[ALL].to_numpy(float)), e.v.clip(lo_, hi_).to_numpy())
                _, icb, _ = M.daily_ic(e.date.to_numpy(), base.predict(e[V.CTRL + V.BOOK].to_numpy(float)), e.v.clip(lo_, hi_).to_numpy())
                vol.append(dict(defn=dn, target="|%s|" % y, test=tn, ic_controls_book=icb["mean"], ic_all=icf["mean"], t_all=icf["t"]))
        R, A, VO, S = pd.DataFrame(rows), pd.DataFrame(abl), pd.DataFrame(vol), pd.DataFrame(stab)
        for df, nm in ((R, "power"), (A, "ablation"), (VO, "volatility"), (S, "stability")):
            df.to_csv(out / ("forecast_power_%s.csv" % nm), index=False)       # saved after every definition
        print(dn, "done", flush=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print("\n### FORECASTING POWER (IC, NW t, out-of-sample R^2, hit rate, top-minus-bottom decile at the mid)")
    print(R.round(4).to_string(index=False))
    print("\n### WHAT CARRIES IT: IC lost when a feature group is removed (2025, all stocks)")
    print(A[A.test == "2025 all stocks"].pivot_table(index=["defn", "horizon"], columns="dropped", values="ic_lost").round(4).to_string())
    print("\n### VOLATILITY: forecasting the size of the next move")
    print(VO.round(4).to_string(index=False))
    print("\n### STABILITY (2025)")
    print(S.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
