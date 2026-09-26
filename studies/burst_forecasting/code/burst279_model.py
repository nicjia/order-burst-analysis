#!/usr/bin/env python3
"""burst279-v1: the original MATH 279 pipeline, rebuilt leak-free with the new burst features.

  burst -> label its price impact as permanent or not (label may look ahead) -> model predicts that label from
  information available at the decision time -> trade on the prediction, entering after the decision time.

Decision time T_dec = max(t_b + 600 s, t_e + 10 s) (p4-revisit-v1). Every feature below is known by T_dec:
  BASE  burst shape and price path so far: size/ADV, children, duration, time of day, peak impact over
        [t_b, t_e+10s), mid displacement at 60 s and 600 s, mean displacement, D/peak ratio, spreads, the
        30-minute pre-burst move and the open-to-decision move.
  NEW   the ideas explored since: identical-size fingerprint (modal-size share, non-round clip), program score
        (size-free; includes inter-arrival regularity, book imbalance and execution depth measured inside the
        burst), same-side same-size bursts in the previous 30 minutes (link_back), truncated and hidden shares,
        and the multi-day fingerprint: yesterday's count of repeated-clip programs on this side and the other.
  EXCLUDED as leaky: link_same / link_opp (they count bursts up to 30 minutes AFTER this one).
Direction is not predicted: trade bursts carry the aggressor's side from native ITCH signs.

Labels: y_cont = post-decision continuation, d_close > 0 (the tradable question);
        y_perm = impact retained from burst start to close, phi_close >= 0.5 (the classic "permanent impact"
        label; it overlaps the decision window, so it is shown only to demonstrate the overlap trap).
Split: train on DEV 2017-19 (group-0 names); test on VAL 2020-21 and TEST 2022-25 (groups 1-2): disjoint names
and years. Hyperparameters fixed in advance; trading thresholds are DEV quantiles.
Trading: at T_dec cross half the quoted spread, exit at the closing auction for 1 bp; each trade earns
d_close (follow) or -d_close (fade) minus cost. Daily equal-weight PnL, Newey-West(10).
Usage: burst279_model.py [--family T] [--train-rows 500000]
"""
import sys as _sys, pathlib as _pl  # shared project modules live in src_py/
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "src_py"))
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import p4_analyze as PA
import fp_multiday_h1 as H

ROOT = Path(__file__).resolve().parents[3]
AGG = ROOT / "results" / "p4_revisit_v1" / "agg"
OUT = ROOT / "results" / "burst_forecasting" / "burst279_v1"
BASE = ["log_q_adv", "log_n", "log_dur", "tod", "peak_bps", "dmean_bps", "d60_bps", "d600_bps", "ratio",
        "spread_b", "spread_dec", "pre30_bps", "own_open_dec_bps"]
NEW = ["mode_share", "mode_nonround", "program_score", "link_back", "truncated_share", "hidden_share",
       "prog_same_prev", "prog_opp_prev"]
COLS = ["permno", "date", "family", "side", "t_b", "t_e", "n", "log_q_adv", "mode_share", "mode_nonround",
        "peak_bps", "dmean_bps", "d60_bps", "d600_bps", "ratio", "spread_b", "spread_dec", "pre30_bps",
        "own_open_dec_bps", "tod", "link_back", "truncated_share", "hidden_share", "program_score",
        "d_close", "phi_close"]
EXIT_COST_BPS = 1.0


def prior_day_programs(cell, cal):
    """Per (permno, date, side): number of repeated-clip program keys (>= 3 one-sided bursts) on the PREVIOUS
    trading day. Uses only the prior day, so it is known before the session opens."""
    fp = pd.read_csv(H.T / ("%s_fp.csv.gz" % cell), dtype={"date": str})
    fp = fp[fp["size"] % 100 != 0]
    w = fp.pivot_table(index=["permno", "size", "date"], columns="side", values="nb", aggfunc="sum", fill_value=0)
    w.columns = ["B" if c == 1 else "S" for c in w.columns]
    w = w.reset_index()
    rows = []
    for s, o, sgn in (("B", "S", 1), ("S", "B", -1)):
        q = w[(w[s] >= 3) & (w[o] == 0)].groupby(["permno", "date"]).size().rename("cnt").reset_index()
        q["side"] = sgn
        rows.append(q)
    p = pd.concat(rows, ignore_index=True)
    inv = {v: k for k, v in cal.items()}
    p["date"] = [inv.get(cal[d] + 1) for d in p.date]          # becomes a feature of the NEXT trading day
    return p.dropna(subset=["date"])


def load(cell, family, cal, max_rows=None, seed=0):
    d = pd.read_csv(AGG / cell / ("sample_%s.csv.gz" % cell), usecols=COLS, dtype={"date": str})
    d = d[d.family == family].drop(columns="family")
    if max_rows and len(d) > max_rows:
        d = d.sample(max_rows, random_state=seed)
    d["log_n"] = np.log1p(d.n); d["log_dur"] = np.log1p(d.t_e - d.t_b)
    p = prior_day_programs(cell, cal)
    same = p.rename(columns={"cnt": "prog_same_prev"})
    opp = p.assign(side=-p.side).rename(columns={"cnt": "prog_opp_prev"})
    d = d.merge(same, on=["permno", "date", "side"], how="left").merge(opp, on=["permno", "date", "side"], how="left")
    d[["prog_same_prev", "prog_opp_prev"]] = d[["prog_same_prev", "prog_opp_prev"]].fillna(0.0)
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["d_close", "spread_dec"])
    d = d[(d.spread_dec > 0) & (d.spread_dec < 500)]
    d["y_cont"] = (d.d_close > 0).astype(int)
    d["y_perm"] = (d.phi_close >= 0.5).astype(int)
    return d


def clip_by_train(train, test, cols):
    lo, hi = train[cols].quantile(0.01), train[cols].quantile(0.99)
    return train[cols].clip(lo, hi, axis=1), test[cols].clip(lo, hi, axis=1)


def fit_models(tr, cols, label, seed=20260924, shuffle=False):
    y = tr[label].to_numpy()
    if shuffle:  # placebo: labels permuted within date, so date-level structure is kept but burst information is not
        y = tr.groupby("date")[label].transform(lambda s: s.sample(frac=1.0, random_state=seed).to_numpy()).to_numpy()
    hgb = HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.05, min_samples_leaf=200,
                                         early_stopping=False, random_state=seed).fit(tr[cols].to_numpy(float), y)
    Xc, _ = clip_by_train(tr, tr, cols)
    mu, sd = Xc.mean(), Xc.std().replace(0, 1)
    lr = LogisticRegression(max_iter=500, C=1.0).fit(((Xc - mu) / sd).fillna(0).to_numpy(), y)
    return dict(hgb=hgb, lr=lr, mu=mu, sd=sd, lo=tr[cols].quantile(0.01), hi=tr[cols].quantile(0.99), cols=cols)


def predict(m, df, kind):
    if kind == "hgb":
        return m["hgb"].predict_proba(df[m["cols"]].to_numpy(float))[:, 1]
    X = df[m["cols"]].clip(m["lo"], m["hi"], axis=1)
    return m["lr"].predict_proba(((X - m["mu"]) / m["sd"]).fillna(0).to_numpy())[:, 1]


def daily_ic(df, score):
    ics = []
    for _, g in df.assign(s=score).groupby("date"):
        if len(g) >= 30:
            ics.append(g.s.rank().corr(g.d_close.rank()))
    return PA.nw_t(np.array(ics))


def book(df, mask, sign, label):
    t = df[mask]
    gross = sign * t.d_close
    net = gross - t.spread_dec / 2 - EXIT_COST_BPS
    daily_g = gross.groupby(t.date).mean(); daily_n = net.groupby(t.date).mean()
    g, n = PA.nw_t(daily_g.to_numpy()), PA.nw_t(daily_n.to_numpy())
    sr = lambda s: float(s.mean() / s.std() * np.sqrt(252)) if s.std() > 0 else float("nan")
    return dict(strategy=label, trades=int(len(t)), trades_per_day=float(len(t) / max(daily_g.size, 1)),
                gross_bps=float(gross.mean()), gross_t=g["t"], net_bps=float(net.mean()), net_t=n["t"],
                half_spread_bps=float((t.spread_dec / 2).mean()), sharpe_gross=sr(daily_g), sharpe_net=sr(daily_n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="T"); ap.add_argument("--train-rows", type=int, default=500000)
    ap.add_argument("--test-rows", type=int, default=1500000)
    ap.add_argument("--cells", default="VAL,TEST")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); cal = H.calendar()
    tr = load("DEV", a.family, cal, a.train_rows)
    print("train DEV rows %d  cont-rate %.3f  perm-rate %.3f  (%.0fs)" % (len(tr), tr.y_cont.mean(), tr.y_perm.mean(), time.time() - t0))
    feats = {"BASE": BASE, "NEW only": NEW, "FULL": BASE + NEW}
    models = {(k, lab): fit_models(tr, cols, lab) for k, cols in feats.items() for lab in ("y_cont", "y_perm")}
    placebo = fit_models(tr, BASE + NEW, "y_cont", shuffle=True)
    # DEV thresholds for trading (deciles of the in-sample score, fixed before any test row is scored)
    dev_score = predict(models[("FULL", "y_cont")], tr, "hgb")
    q10, q90 = np.quantile(dev_score, [0.10, 0.90])
    dev_pl = predict(placebo, tr, "hgb"); p10, p90 = np.quantile(dev_pl, [0.10, 0.90])
    spread_med = float(tr.spread_dec.median())   # added after the VAL read; TEST is its first out-of-sample look
    res = dict(family=a.family, train_rows=int(len(tr)), thresholds=dict(q10=float(q10), q90=float(q90)))
    for cell in a.cells.split(","):
        if not (AGG / cell / ("sample_%s.csv.gz" % cell)).exists():
            print(cell, "sample not available"); continue
        te = load(cell, a.family, cal, a.test_rows)
        r = dict(rows=int(len(te)), names=int(te.permno.nunique()), days=int(te.date.nunique()),
                 cont_rate=float(te.y_cont.mean()), perm_rate=float(te.y_perm.mean()), auc={}, ic={})
        print("\n=== %s  rows %d  names %d  days %d" % (cell, r["rows"], r["names"], r["days"]))
        for (k, lab), m in models.items():
            for kind in ("hgb", "lr"):
                s = predict(m, te, kind)
                auc = roc_auc_score(te[lab], s)
                r["auc"]["%s|%s|%s" % (k, lab, kind)] = float(auc)
                if lab == "y_cont" and kind == "hgb":
                    r["ic"][k] = daily_ic(te, s)
        for k in feats:
            print("  %-9s AUC continuation (tradable) hgb %.4f lr %.4f | AUC permanence-from-start (overlaps) hgb %.4f | daily IC %+.4f (t %.2f)"
                  % (k, r["auc"]["%s|y_cont|hgb" % k], r["auc"]["%s|y_cont|lr" % k], r["auc"]["%s|y_perm|hgb" % k],
                     r["ic"][k]["mean"], r["ic"][k]["t"]))
        s = predict(models[("FULL", "y_cont")], te, "hgb"); pl = predict(placebo, te, "hgb")
        r["auc"]["placebo"] = float(roc_auc_score(te.y_cont, pl))
        print("  placebo (labels shuffled within date) AUC %.4f" % r["auc"]["placebo"])
        books = [book(te, np.ones(len(te), bool), +1, "follow every burst"),
                 book(te, np.ones(len(te), bool), -1, "fade every burst"),
                 book(te, s >= q90, +1, "FOLLOW top decile of P(continue)"),
                 book(te, s <= q10, -1, "FADE bottom decile of P(continue)"),
                 book(te, (s <= q10) & (te.spread_dec <= spread_med), -1, "FADE bottom decile, tight spread"),
                 book(te, (s >= q90) & (te.spread_dec <= spread_med), +1, "FOLLOW top decile, tight spread"),
                 book(te, pl >= p90, +1, "placebo follow top decile"),
                 book(te, pl <= p10, -1, "placebo fade bottom decile")]
        r["books"] = books
        print("  %-36s %8s %8s %9s %7s %9s %7s %8s %8s" % ("strategy", "trades", "per day", "gross bps", "t", "net bps", "t", "SR gross", "SR net"))
        for b in books:
            print("  %-36s %8d %8.1f %9.2f %7.2f %9.2f %7.2f %8.2f %8.2f" % (b["strategy"], b["trades"], b["trades_per_day"],
                  b["gross_bps"], b["gross_t"], b["net_bps"], b["net_t"], b["sharpe_gross"], b["sharpe_net"]))
        # which features carry the prediction (permutation importance on a test subsample, AUC drop)
        sub = te.sample(min(150000, len(te)), random_state=1)
        base_auc = roc_auc_score(sub.y_cont, predict(models[("FULL", "y_cont")], sub, "hgb"))
        imp = {}
        rng = np.random.default_rng(2)
        for c in BASE + NEW:
            x = sub.copy(); x[c] = rng.permutation(x[c].to_numpy())
            imp[c] = float(base_auc - roc_auc_score(sub.y_cont, predict(models[("FULL", "y_cont")], x, "hgb")))
        r["importance_auc_drop"] = dict(sorted(imp.items(), key=lambda kv: -kv[1]))
        print("  permutation importance (AUC drop):", ", ".join("%s %.4f" % kv for kv in list(r["importance_auc_drop"].items())[:8]))
        res[cell] = r
    (OUT / ("results_%s.json" % a.family)).write_text(json.dumps(res, indent=1, default=float))
    print("\ndone in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
