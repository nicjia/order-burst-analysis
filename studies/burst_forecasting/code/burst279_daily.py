#!/usr/bin/env python3
"""burst279-daily: (1) how well the model predicts permanence; (2) the once-per-day strategies of the old pipeline,
rebuilt on corrected data and driven by the burst model.

(1) Labels, per burst: permanence from burst start, phi_x >= 0.5 for x in {close, next open, next close}
(the old 279 permanence, which overlaps the move already seen by T_dec), and post-decision continuation d_x > 0
(the tradable part). HGB classifiers trained on DEV; AUC, Brier score, calibration and lift by predicted decile
on VAL and TEST.
(2) Daily signals per name-day from bursts decided by the clock (15:30 for tCLOSE, 15:50 for CLOP / CLCL):
S_model = sum side * q * (p - mean_DEV(p)), q = burst size / ADV, p = the horizon-matched continuation model;
S_flow = sum side * q (model-free flow, the old phase-3 style). Daily decile sort, long top / short bottom,
equal-weight, dollar-neutral. Outcomes: tCLOSE = mid 15:30 -> close mid; CLOP = close -> next open; CLCL = close
-> next close. Costs per day on the long-short spread: CLOP / CLCL 8 bps (2 bps per side per leg at the
auctions); tCLOSE the two legs' half-spreads at 15:30 plus 4 bps at the close. The per-burst sample is capped at
10 bursts per name-day, so S is an estimate of the full-day signal.
Usage: burst279_daily.py
"""
import sys as _sys, pathlib as _pl  # shared project modules live in src_py/
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "src_py"))
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
import p4_analyze as PA
import fp_multiday_h1 as H
import burst279_model as B

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "burst_forecasting" / "burst279_v1"
FEATS = B.BASE + B.NEW
LABELS = {"perm_close": ("phi_close", 0.5, "from start"), "perm_open": ("phi_open", 0.5, "from start"),
          "perm_cc": ("phi_cc", 0.5, "from start"), "cont_close": ("d_close", 0.0, "after decision"),
          "cont_open": ("d_open", 0.0, "after decision"), "cont_cc": ("d_cc", 0.0, "after decision")}


def load(cell, cal, n=None):
    d = pd.read_csv(B.AGG / cell / ("sample_%s.csv.gz" % cell), dtype={"date": str},
                    usecols=list(dict.fromkeys(B.COLS + ["d_open", "d_cc", "phi_open", "phi_cc"])))
    d = d[d.family == "T"].drop(columns="family")
    if n and len(d) > n:
        d = d.sample(n, random_state=0)
    d["log_n"] = np.log1p(d.n); d["log_dur"] = np.log1p(d.t_e - d.t_b)
    d["t_dec"] = np.maximum(d.t_b + 600, d.t_e + 10)
    p = B.prior_day_programs(cell, cal)
    d = d.merge(p.rename(columns={"cnt": "prog_same_prev"}), on=["permno", "date", "side"], how="left")
    d = d.merge(p.assign(side=-p.side).rename(columns={"cnt": "prog_opp_prev"}), on=["permno", "date", "side"], how="left")
    d[["prog_same_prev", "prog_opp_prev"]] = d[["prog_same_prev", "prog_opp_prev"]].fillna(0.0)
    return d.replace([np.inf, -np.inf], np.nan)


def calib(y, p):
    q = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False)
    t = pd.DataFrame(dict(y=y, p=p, q=q)).groupby("q").agg(pred=("p", "mean"), real=("y", "mean"))
    return [dict(decile=int(k) + 1, predicted=round(float(r.pred), 4), realized=round(float(r.real), 4)) for k, r in t.iterrows()]


def portfolio(day, sig, ret, cost_fn):
    rows = []
    for d, g in day[day[sig] != 0].groupby("date"):
        g = g.dropna(subset=[ret])
        if len(g) < 20:
            continue
        r = g[sig].rank(method="first"); k = max(1, len(g) // 10)
        top, bot = g[r > len(g) - k], g[r <= k]
        gross = (top[ret].mean() - bot[ret].mean()) * 1e4
        rows.append((d, gross, gross - cost_fn(top, bot)))
    p = pd.DataFrame(rows, columns=["date", "gross", "net"])
    if p.empty:
        return None
    g, n = PA.nw_t(p.gross.to_numpy()), PA.nw_t(p.net.to_numpy())
    sr = lambda s: float(s.mean() / s.std() * np.sqrt(252)) if s.std() > 0 else None
    return dict(days=int(len(p)), gross=float(p.gross.mean()), gross_t=g["t"], net=float(p.net.mean()), net_t=n["t"],
                sr_gross=sr(p.gross), sr_net=sr(p.net), fade_net=float((-p.gross - (p.gross - p.net)).mean()))


def main():
    t0 = time.time(); cal = H.calendar(); OUT.mkdir(parents=True, exist_ok=True)
    tr = load("DEV", cal, 400000)
    models, pbar = {}, {}
    for lab, (col, thr, _) in LABELS.items():
        t = tr[tr[col].notna()]
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.05, min_samples_leaf=200,
                                           early_stopping=False, random_state=1).fit(t[FEATS].to_numpy(float), (t[col] >= thr).astype(int) if thr else (t[col] > 0).astype(int))
        models[lab] = m
        pbar[lab] = float(m.predict_proba(t[FEATS].to_numpy(float))[:, 1].mean())
    print("trained 6 permanence/continuation models on DEV (%.0fs)" % (time.time() - t0), flush=True)
    res = {}
    for cell in ("VAL", "TEST"):
        te = load(cell, cal)
        X = te[FEATS].to_numpy(float)
        r = dict(bursts=int(len(te)), quality={})
        print("\n=== %s  bursts %d  names %d  days %d" % (cell, len(te), te.permno.nunique(), te.date.nunique()))
        print("  %-11s %-15s %7s %7s %7s %8s %8s %8s" % ("label", "measured", "base", "AUC", "Brier", "top-dec", "bot-dec", "lift"))
        for lab, (col, thr, how) in LABELS.items():
            ok = te[col].notna().to_numpy()
            y = ((te[col] >= thr) if thr else (te[col] > 0)).astype(int).to_numpy()[ok]
            p = models[lab].predict_proba(X[ok])[:, 1]
            if lab.startswith("cont"):
                te.loc[ok, "p_" + lab] = p
            c = calib(y, p)
            q = dict(base=float(y.mean()), auc=float(roc_auc_score(y, p)), brier=float(brier_score_loss(y, p)),
                     brier_base=float(brier_score_loss(y, np.full(len(y), y.mean()))), top=c[-1]["realized"], bottom=c[0]["realized"],
                     calibration=c)
            r["quality"][lab] = q
            print("  %-11s %-15s %7.3f %7.4f %7.4f %8.3f %8.3f %8.2f" % (lab, how, q["base"], q["auc"], q["brier"], q["top"], q["bottom"], q["top"] / q["base"]))
        # daily once-per-day strategies
        nd = pd.read_csv(B.AGG / cell / ("nameday_%s.csv.gz" % cell), dtype={"date": str},
                         usecols=["permno", "date", "family", "clop", "ret_next", "tclose", "spread_1530"])
        nd = nd[nd.family == "T"].drop(columns="family")
        te["q"] = np.exp(te.log_q_adv)
        books = {}
        for strat, clock, ret, lab in (("tCLOSE 15:30->close", 55800, "tclose", "cont_close"),
                                       ("CLOP close->next open", 57000, "clop", "cont_open"),
                                       ("CLCL close->next close", 57000, "ret_next", "cont_cc")):
            b = te[(te.t_dec <= clock) & te["p_" + lab].notna()]
            s = b.assign(S_model=b.side * b.q * (b["p_" + lab] - pbar[lab]), S_flow=b.side * b.q)
            s = s.groupby(["permno", "date"])[["S_model", "S_flow"]].sum().reset_index().merge(nd, on=["permno", "date"])
            if ret == "tclose":
                cost = lambda top, bot: top.spread_1530.mean() / 2 + bot.spread_1530.mean() / 2 + 4.0
            else:
                cost = lambda top, bot: 8.0
            for sig in ("S_model", "S_flow"):
                books["%s | %s" % (strat, sig)] = portfolio(s, sig, ret, cost)
        r["daily"] = books
        print("  once-per-day long-short decile books (bps/day):")
        print("  %-42s %6s %8s %7s %8s %7s %8s %8s" % ("strategy | signal", "days", "gross", "t", "net", "t", "SR net", "fade net"))
        for k, v in books.items():
            if v:
                print("  %-42s %6d %8.2f %7.2f %8.2f %7.2f %8.2f %8.2f" % (k, v["days"], v["gross"], v["gross_t"], v["net"], v["net_t"], v["sr_net"] or 0, v["fade_net"]))
        res[cell] = r
    (OUT / "daily_and_quality.json").write_text(json.dumps(res, indent=1, default=float))
    print("\ndone %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
