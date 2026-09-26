#!/usr/bin/env python3
"""Pre-registered 2025 confirmation (docs/CONFIRMATION_2025.md). Models trained ONLY on the train stocks, 2022-23
(v3 panel); applied once to every 2025 event of all 112 raw-pass stocks. Rule: top 20% |forecast| (training 80th
percentile), side = sign(forecast), exit at the closing mid, cost = half spread + 1 bp. Primary cells:
levelclear <= 5 bps, run0.01 <= 5 bps, hawkesfix 5-10 bps; gate net > 0 with t > 2. Also reports the mid P&L
(forecast) at every horizon for every definition."""
import sys as _sys, pathlib as _pl  # shared project modules live in src_py/
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "src_py"))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import p4_analyze as PA
import burst_defs_raw2_model as M
import burst_defs_raw3_model as V

F = V.CTRL + V.BOOK + V.BURST
PRIMARY = {("levelclear", "<=5 bps"), ("run0.01", "<=5 bps"), ("hawkesfix", "5-10 bps")}


def prep(path):
    d = pd.read_csv(path, dtype={"date": str})
    nd = pd.read_csv(M.ROOT / "results" / "p4_revisit_v1" / "agg" / "TEST" / "nameday_TEST.csv.gz", dtype={"date": str},
                     usecols=["permno", "date", "family", "adv20"])
    d = d.merge(nd[nd.family == "T"][["permno", "date", "adv20"]], on=["permno", "date"], how="left")
    d["log_n"] = np.log(d.n_used); d["log_dur"] = np.log1p(d.dur); d["log_q_adv"] = np.log(d.vol_used / d.adv20)
    d = d.replace([np.inf, -np.inf], np.nan); d = d[(d.spread_dec > 0) & (d.spread_dec < 500)]
    for c in ("r10", "r60", "r300", "r1800", "r_close"):
        d.loc[d[c].abs() > 1000, c] = np.nan
    return M.add_market(d, "t_dec", None)


def main():
    train = set(pd.read_csv(M.ROOT / "results" / "p4_revisit_v1" / "jobs" / "DEFS_RAW_train_names.txt", header=None)[0])
    tr_all = prep(M.D / "burst_defs_raw3" / "RAW3_events.csv.gz")
    tr_all = tr_all[tr_all.permno.isin(train) & tr_all.date.str[:4].isin(["2022", "2023"])]
    te_all = prep(M.D / "burst_defs_raw3" / "Y2025_events.csv.gz")
    te_all = te_all[te_all.date.str[:4] == "2025"]
    print("train events %d (2022-23, train stocks); 2025 events %d on %d stocks, %d days"
          % (len(tr_all), len(te_all), te_all.permno.nunique(), te_all.date.nunique()))
    rows, fc = [], []
    for dn in sorted(te_all.defn.unique()):
        tr, te = tr_all[tr_all.defn == dn], te_all[te_all.defn == dn]
        if len(tr) < 3000 or len(te) < 1000:
            continue
        for y in ("r10", "r60", "r300", "r1800_x", "r_close_x"):
            m, lo, hi = M.hgb(tr, F, y)
            e = te[te[y].notna()].copy(); p = m.predict(e[F].to_numpy(float))
            s, ic, sp = M.daily_ic(e.date.to_numpy(), p, e[y].clip(lo, hi).to_numpy())
            fc.append(dict(defn=dn, horizon=y, ic=ic["mean"], t=ic["t"], d10_d1_bps=sp["mean"]))
            if y != "r_close_x":
                continue
            thr = np.quantile(np.abs(m.predict(tr[F].to_numpy(float))), 0.8)
            k = np.abs(p) >= thr
            e, p = e[k], p[k]
            e["pnl"] = np.sign(p) * e.r_close_x; e["net"] = e.pnl - (e.spread_dec / 2 + 1.0)
            for lab, msk in (("<=5 bps", e.spread_dec <= 5), ("5-10 bps", (e.spread_dec > 5) & (e.spread_dec <= 10)), ("all", e.spread_dec > 0)):
                x = e[msk]
                if len(x) < 100:
                    continue
                day = x.groupby("date").net.mean()
                t = PA.nw_t(day.to_numpy())["t"]
                prim = (dn, lab) in PRIMARY
                rows.append(dict(defn=dn, spread=lab, primary=prim, trades=int(len(x)), mid_bps=float(x.pnl.mean()),
                                 net_bps=float(x.net.mean()), t_net=t, passes=bool(prim and x.net.mean() > 0 and t > 2)))
        print("  %s done" % dn, flush=True)
    R, FC = pd.DataFrame(rows), pd.DataFrame(fc)
    out = M.D / "burst_defs_raw3"
    R.to_csv(out / "confirm_2025_trades.csv", index=False); FC.to_csv(out / "confirm_2025_forecasts.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 300)
    print("\n=== PRIMARY CELLS (pre-registered) ===")
    print(R[R.primary].round(3).to_string(index=False))
    print("\n=== all to-close cells ===")
    print(R.round(3).to_string(index=False))
    print("\n=== 2025 forecasting at the mid (IC, NW t, top-minus-bottom decile) ===")
    print(FC.pivot_table(index="defn", columns="horizon", values="ic").round(4).to_string())


if __name__ == "__main__":
    main()
