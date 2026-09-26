#!/usr/bin/env python3
"""burst-defs-raw-v3 stage 2: the new definitions, deciding during the burst, latency decay, new book features and
the continuation target.

For every v3 definition and horizon (+10 s, +60 s, +300 s, +30 min market-excess, close market-excess):
  IC of CTRL+BOOK (controls + queue imbalance, OFI, trade-flow imbalance, microprice gap, opposite cancels)
  IC of everything (adds burst path, structure, regularity, opposite-touch consumption, spread change)
  paired daily gain of the burst features over CTRL+BOOK, and the top-minus-bottom decile at the mid.
Latency: the same full model scored on outcomes that start 1 s and 10 s after the decision (r10_L1, r10_L10,
r60_L1, r60_L10) -- how fast the forecast must be acted on.
Continuation (D8): AUC for "another same-side event of this definition starts within 60 s", and for early3 / early5
"the run grows past k" -- burst-continuation forecasting. For early3/5, n_total and vol_total are the FULL burst and
are never features.
Train 56 stocks 2022-23, test 56 different stocks 2024. Gradient boosting depth 3, fixed.
"""
import sys as _sys, pathlib as _pl  # shared project modules live in src_py/
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "src_py"))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import p4_analyze as PA
import burst_defs_raw2_model as M

CTRL = ["since_open", "pre30m", "tod", "spread_dec"]
BOOK = ["imb_first", "imb_last", "qofi_pre60", "qofi_during", "tfi_pre60", "micro_gap", "canc_opp", "canc_same"]
BURST = ["move_during", "pre60", "log_n", "log_dur", "log_q_adv", "mode_share", "nonround", "size_cv", "size_to_depth",
         "iat_cv", "iat_med", "phase_R", "opp_consumed", "spread_change"]
HOR = ["r10", "r60", "r300", "r1800_x", "r_close_x"]
LAT = ["r10_L1", "r10_L10", "r60_L1", "r60_L10"]


def main():
    train = set(pd.read_csv(M.ROOT / "results" / "p4_revisit_v1" / "jobs" / "DEFS_RAW_train_names.txt", header=None)[0])
    d = pd.read_csv(M.D / "burst_defs_raw3" / "RAW3_events.csv.gz", dtype={"date": str})
    nd = pd.read_csv(M.ROOT / "results" / "p4_revisit_v1" / "agg" / "TEST" / "nameday_TEST.csv.gz", dtype={"date": str},
                     usecols=["permno", "date", "family", "adv20"])
    d = d.merge(nd[nd.family == "T"][["permno", "date", "adv20"]], on=["permno", "date"], how="left")
    d["log_n"] = np.log(d.n_used); d["log_dur"] = np.log1p(d.dur); d["log_q_adv"] = np.log(d.vol_used / d.adv20)
    d = d.replace([np.inf, -np.inf], np.nan); d = d[(d.spread_dec > 0) & (d.spread_dec < 500)]
    for c in HOR[:3] + ["r1800", "r_close"] + LAT:
        d.loc[d[c].abs() > 1000, c] = np.nan
    d = M.add_market(d, "t_dec", None)
    tr_all = d[d.permno.isin(train) & d.date.str[:4].isin(["2022", "2023"])]
    te_all = d[~d.permno.isin(train) & (d.date.str[:4] == "2024")]
    print("events: train %d, test %d; definitions %d" % (len(tr_all), len(te_all), d.defn.nunique()))
    rows, lat_rows, cont_rows = [], [], []
    for defn in sorted(d.defn.unique()):
        tr, te = tr_all[tr_all.defn == defn], te_all[te_all.defn == defn]
        if len(tr) < 3000 or len(te) < 1500:
            print("  %s: too few events (train %d, test %d)" % (defn, len(tr), len(te))); continue
        for y in HOR:
            ics = {}
            for k, cols in (("CTRL+BOOK", CTRL + BOOK), ("ALL", CTRL + BOOK + BURST)):
                m, lo, hi = M.hgb(tr, cols, y)
                e = te[te[y].notna()]
                s, ic, sp = M.daily_ic(e.date.to_numpy(), m.predict(e[cols].to_numpy(float)), e[y].clip(lo, hi).to_numpy())
                ics[k] = (s, ic, sp)
                if k == "ALL" and y in ("r10", "r60"):
                    for L in LAT:
                        if L.startswith(y + "_"):
                            e2 = te[te[L].notna()]
                            _, ic2, _ = M.daily_ic(e2.date.to_numpy(), m.predict(e2[cols].to_numpy(float)), e2[L].to_numpy())
                            lat_rows.append(dict(defn=defn, horizon=y, start_delay=L.split("_L")[1] + " s", ic=ic2["mean"], t=ic2["t"],
                                                 ic_at_decision=ic["mean"]))
            g = PA.nw_t((ics["ALL"][0] - ics["CTRL+BOOK"][0]).to_numpy())
            rows.append(dict(defn=defn, horizon=y, n_test=int(te[y].notna().sum()), ic_ctrl_book=ics["CTRL+BOOK"][1]["mean"],
                             ic_all=ics["ALL"][1]["mean"], t_all=ics["ALL"][1]["t"], gain_burst=g["mean"], t_gain=g["t"],
                             d10_d1_bps=ics["ALL"][2]["mean"]))
        targets = [("cont60_same", "cont60_same"), ("cont60_opp", "cont60_opp")] + ([("grew", "grew")] if defn.startswith("early") else [])
        for name, col in targets:
            t_ = tr[tr[col].notna()]; e_ = te[te[col].notna()]
            if t_[col].nunique() < 2 or e_[col].nunique() < 2:
                continue
            clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, min_samples_leaf=200,
                                                 early_stopping=False, random_state=1).fit(t_[CTRL + BOOK + BURST].to_numpy(float), t_[col].astype(int))
            p = clf.predict_proba(e_[CTRL + BOOK + BURST].to_numpy(float))[:, 1]
            cont_rows.append(dict(defn=defn, target=name, base_rate=float(e_[col].mean()), auc=float(roc_auc_score(e_[col].astype(int), p))))
        print("  %s done" % defn, flush=True)
    R, L, C = pd.DataFrame(rows), pd.DataFrame(lat_rows), pd.DataFrame(cont_rows)
    out = M.D / "burst_defs_raw3"
    R.to_csv(out / "forecast_v3.csv", index=False); L.to_csv(out / "latency_v3.csv", index=False); C.to_csv(out / "continuation_v3.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    for y in HOR:
        print("\n### %s  (IC; gain = burst features over controls + book, NW t)" % y)
        s = R[R.horizon == y].sort_values("gain_burst", ascending=False)
        print(s[["defn", "n_test", "ic_ctrl_book", "ic_all", "t_all", "gain_burst", "t_gain", "d10_d1_bps"]].round(4).to_string(index=False))
    print("\n### latency: IC of the full model when the outcome window starts later")
    print(L.round(4).to_string(index=False))
    print("\n### continuation targets (AUC)")
    print(C.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
