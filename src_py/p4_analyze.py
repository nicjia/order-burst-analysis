#!/usr/bin/env python3
"""P4 revisit v1, stage 3: pre-registered tests Q1, Q2(a) and Q4 on one cell (P4_REVISIT_DESIGN.md section 6).

Inputs: nameday_<cell>.csv.gz and strata_<cell>.csv.gz from p4_aggregate.py.
Protocol (design section 7.5): each run appends to analysis/access_log.txt with input checksums; TEST refuses to
run until analysis/VAL_primary.json exists, ERA2 until analysis/TEST_primary.json exists.

Q1   per family and kappa: name-day mean d_close of informative-large bursts minus the mean over their informative
     pseudo-bursts (primary, kappa 0.5), and minus non-informative large bursts (also d_open, d_cc); daily
     cross-name means, Newey-West(10); name-bootstrap 95% CI for the primary statistic.
Q2a  stratified difference (name-day x hour x size quintile cells) in same-side and opposite-side link rates,
     informative minus non-informative large bursts, weights n_i n_n / (n_i + n_n); name bootstrap.
Q4   daily cross-sectional OLS (regressors winsorized 1/99% per day) of CLOP, CLCL (15:50 signals) and tCLOSE
     (15:30 signals) in bps on S_info/ADV20 with S_large/ADV20, S_all/ADV20 and the design controls; NW(10) on
     slopes; Holm across the 6 primary cells; secondary S_pseudo in place of S_info and the kappa grid;
     decile long-short portfolios with costs and the deflated Sharpe probability.
"""
import argparse
import datetime as _dt
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# P4_ANALYSIS_DIR lets the cluster copy (whose code sits under results/p4_revisit_v1/code) use the same directory
ANALYSIS = Path(os.environ.get("P4_ANALYSIS_DIR", ROOT / "results" / "p4_revisit_v1" / "analysis"))
NW_LAGS = 10
KAPPAS = ("k25", "k50", "k75")
TRIAL_COUNT = 110 + 75 + 66 + 60   # legacy ledger subtotals (DEFINITIONS_TRIED.md) + this design's Q4 cells, conservative


def nw_t(x, lags=NW_LAGS):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return dict(mean=None, t=None, n=int(n))
    m = x.mean(); e = x - m
    v = e @ e / n
    for l in range(1, min(lags, n - 1) + 1):
        v += 2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / n
    return dict(mean=float(m), t=float(m / math.sqrt(v / n)) if v > 0 else None, n=int(n))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def check_protocol(cell, inputs):
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    need = {"TEST": "VAL", "ERA2": "TEST"}.get(cell)
    if need and not (ANALYSIS / ("%s_primary.json" % need)).exists():
        raise SystemExit("protocol: %s cannot be read before %s_primary.json exists" % (cell, need))
    with open(ANALYSIS / "access_log.txt", "a") as fh:
        fh.write("%s %s %s\n" % (_dt.datetime.now().isoformat(timespec="seconds"), cell,
                                " ".join("%s=%s" % (Path(p).name, sha(p)) for p in inputs)))


# ---------------------------------------------------------------------------------------------
# Q1

def safe_mean(s, n):
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def q1(nd, rng, n_boot=1000):
    out = {}
    for fam, g in nd.groupby("family"):
        g = g[g.has_large.astype(bool)]
        res = {}
        for k in KAPPAS:
            info = safe_mean(g["sum_dclose_info_" + k], g["n_dclose_info_" + k])
            pseudo = safe_mean(g["sum_dclose_pseudo_" + k], g["n_dclose_pseudo_" + k])
            non = safe_mean(g["sum_dclose_non_" + k], g["n_dclose_non_" + k])
            frame = pd.DataFrame(dict(date=g.date.to_numpy(), permno=g.permno.to_numpy(),
                                      d_pseudo=info - pseudo, d_non=info - non, info=info, pseudo=pseudo, non=non))
            for o in ("dopen", "dcc"):
                frame["d_non_" + o] = (safe_mean(g["sum_%s_info_%s" % (o, k)], g["n_%s_info_%s" % (o, k)])
                                       - safe_mean(g["sum_%s_non_%s" % (o, k)], g["n_%s_non_%s" % (o, k)]))
                frame["info_" + o] = safe_mean(g["sum_%s_info_%s" % (o, k)], g["n_%s_info_%s" % (o, k)])
                frame["non_" + o] = safe_mean(g["sum_%s_non_%s" % (o, k)], g["n_%s_non_%s" % (o, k)])
            r = {}
            for col in [c for c in frame.columns if c not in ("date", "permno")]:
                daily = frame.groupby("date")[col].mean()
                r[col] = nw_t(daily.to_numpy())
            if k == "k50":
                r["d_pseudo_ci95"] = name_bootstrap(frame, "d_pseudo", rng, n_boot)
            r["namedays_with_both"] = int(np.isfinite(frame.d_pseudo).sum())
            res[k] = r
        for base in ("large", "all"):
            m = safe_mean(g["sum_dclose_" + base], g["n_dclose_" + base])
            res["level_" + base] = nw_t(pd.Series(m).groupby(g.date.to_numpy()).mean().to_numpy())
        out[fam] = res
    return out


def name_bootstrap(frame, col, rng, n_boot):
    """Mean over days of the daily cross-name mean, with names resampled (count weights)."""
    f = frame[["date", "permno", col]].dropna()
    if f.empty:
        return None
    X = f.pivot_table(index="permno", columns="date", values=col, aggfunc="mean")
    M = X.notna().to_numpy(float); V = np.nan_to_num(X.to_numpy(float))
    n = len(X)
    stats = []
    for _ in range(n_boot):
        c = rng.multinomial(n, np.full(n, 1.0 / n)).astype(float)
        num = c @ V; den = c @ M
        ok = den > 0
        stats.append(float((num[ok] / den[ok]).mean()))
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


# ---------------------------------------------------------------------------------------------
# Q2(a)

def q2a_contributions(st):
    """Per stratum weight w = n_i n_n / (n_i + n_n) and weighted rate differences; strata need both classes."""
    piv = st.pivot_table(index=["permno", "date", "hour", "quint"], columns="info",
                         values=["n", "same", "opp"], aggfunc="sum").dropna()
    if piv.empty:
        return None
    ni, nn = piv[("n", 1)], piv[("n", 0)]
    w = ni * nn / (ni + nn)
    c = pd.DataFrame(dict(permno=piv.index.get_level_values("permno"), w=w.to_numpy(),
                          same=(w * (piv[("same", 1)] / ni - piv[("same", 0)] / nn)).to_numpy(),
                          opp=(w * (piv[("opp", 1)] / ni - piv[("opp", 0)] / nn)).to_numpy()))
    return c.groupby("permno")[["w", "same", "opp"]].sum()


def q2a(strata, rng, n_boot=1000):
    out = {}
    for fam, st in strata.groupby("family"):
        c = q2a_contributions(st)
        if c is None:
            out[fam] = None
            continue
        W, S, O = (c[k].to_numpy(float) for k in ("w", "same", "opp"))
        est = dict(same=float(S.sum() / W.sum()), opp=float(O.sum() / W.sum()))
        est["directional"] = est["same"] - est["opp"]
        est["names"] = int(len(c)); est["weight"] = float(W.sum())
        n = len(c)
        boots = {"same": [], "opp": [], "directional": []}
        for _ in range(n_boot):
            k = rng.multinomial(n, np.full(n, 1.0 / n)).astype(float)
            wsum = k @ W
            if wsum <= 0:
                continue
            a, b = (k @ S) / wsum, (k @ O) / wsum
            boots["same"].append(a); boots["opp"].append(b); boots["directional"].append(a - b)
        est["ci95"] = {kk: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for kk, v in boots.items()}
        est["pass_directional"] = bool(est["directional"] > 0 and est["ci95"]["directional"][0] > 0)   # design Q2(a) gate
        out[fam] = est
    return out


# ---------------------------------------------------------------------------------------------
# Q4

def winsor_day(df, cols):
    def w(g):
        g = g.copy()
        for c in cols:
            lo, hi = g[c].quantile([0.01, 0.99])
            g[c] = g[c].clip(lo, hi)
        return g
    return df.groupby("date", group_keys=False)[df.columns.tolist()].apply(w)


def design_frame(g, clock, kappa, signal="info"):
    adv = g.adv20
    x = pd.DataFrame(dict(date=g.date, permno=g.permno))
    x["S_sig"] = g["S_%s_%s_%s" % (signal, clock, kappa)] / adv
    x["S_large"] = g["S_large_" + clock] / adv
    x["S_all"] = g["S_all_" + clock] / adv
    x["packet_flow"] = (g["buy_" + clock] - g["sell_" + clock]) / adv
    x["own_open"] = g["own_open_" + clock]
    x["ret_lag1"] = g.ret_lag1
    x["ret_lag5"] = g.ret_lag5
    with np.errstate(divide="ignore", invalid="ignore"):
        x["log_cap"] = np.log(g.cap_lag1)
        x["log_dvol"] = np.log(g.dvol20)
    x["sigma20"] = g.sigma20
    x["spread"] = g["spread_" + clock]
    x["turn20"] = g.turn20
    return x


REGS = ["S_sig", "S_large", "S_all", "packet_flow", "own_open", "ret_lag1", "ret_lag5", "log_cap", "log_dvol",
        "sigma20", "spread", "turn20"]
TARGETS = {"CLOP": ("1550", "clop"), "CLCL": ("1550", "ret_next"), "tCLOSE": ("1530", "tclose")}


def fm_cell(g, target, kappa="k50", signal="info", min_obs=30):
    clock, ycol = TARGETS[target]
    x = design_frame(g, clock, kappa, signal)
    x["y"] = g[ycol].to_numpy() * 1e4
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    x = winsor_day(x, REGS)
    slopes = []
    for d, h in x.groupby("date"):
        if len(h) < max(min_obs, len(REGS) + 5):
            continue
        X = np.column_stack([np.ones(len(h))] + [h[c].to_numpy(float) for c in REGS])
        if np.linalg.matrix_rank(X) < X.shape[1]:
            continue
        b = np.linalg.lstsq(X, h.y.to_numpy(float), rcond=None)[0]
        slopes.append(dict(date=d, **{c: b[i + 1] for i, c in enumerate(REGS)}))
    s = pd.DataFrame(slopes)
    if s.empty:
        return None
    res = {c: nw_t(s[c].to_numpy()) for c in REGS}
    sd = x.groupby("date").S_sig.std().mean()
    res["S_sig_per_sd_bps"] = float(res["S_sig"]["mean"] * sd) if res["S_sig"]["mean"] is not None else None
    res["days"] = int(len(s)); res["namedays"] = int(len(x))
    return res


def holm(pvals, alpha=0.05):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    passed = [False] * len(pvals)
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (len(pvals) - rank):
            passed[i] = True
        else:
            break
    return passed


def p_two_sided(t):
    return math.erfc(abs(t) / math.sqrt(2)) if t is not None else 1.0


def norm_ppf(p):
    """Inverse standard normal by bisection on math.erf (exact to 1e-12; avoids hand-typed constants)."""
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * math.erfc(-mid / math.sqrt(2)) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def deflated_sharpe_prob(sr_daily, n_days, skew, kurt, n_trials, sr_var_trials=None):
    """Bailey and Lopez de Prado (2014): P(true Sharpe > expected maximum Sharpe of n_trials null strategies).

    sr_daily is the per-period Sharpe; the variance of Sharpe across trials defaults to 1 / n_days; kurt is raw
    (non-excess) kurtosis."""
    if n_days < 30 or not np.isfinite(sr_daily):
        return None
    emc = 0.5772156649015329
    var = sr_var_trials if sr_var_trials is not None else 1.0 / n_days
    sr0 = math.sqrt(var) * ((1 - emc) * norm_ppf(1 - 1.0 / n_trials) + emc * norm_ppf(1 - 1.0 / (n_trials * math.e)))
    denom = math.sqrt(max(1 - skew * sr_daily + (kurt - 1) / 4 * sr_daily ** 2, 1e-12))
    stat = (sr_daily - sr0) * math.sqrt(n_days - 1) / denom
    return 0.5 * math.erfc(-stat / math.sqrt(2))


def portfolio(g, target, kappa="k50", cost_side_bps=2.0):
    clock, ycol = TARGETS[target]
    sig = g["S_info_%s_%s" % (clock, kappa)] / g.adv20
    f = pd.DataFrame(dict(date=g.date, sig=sig, y=g[ycol] * 1e4, spread=g["spread_" + clock]))
    f = f.replace([np.inf, -np.inf], np.nan).dropna()
    f = f[f.sig != 0]
    rows = []
    for d, h in f.groupby("date"):
        if len(h) < 50:
            continue
        lo, hi = h.sig.quantile([0.1, 0.9])
        L, S = h[h.sig >= hi], h[h.sig <= lo]
        if target == "tCLOSE":
            cost = (L.spread.mean() / 2 + cost_side_bps) + (S.spread.mean() / 2 + cost_side_bps)
        else:
            cost = 4 * cost_side_bps
        rows.append(dict(date=d, gross=L.y.mean() - S.y.mean(), net=L.y.mean() - S.y.mean() - cost))
    p = pd.DataFrame(rows)
    if p.empty:
        return None
    out = {}
    for col in ("gross", "net"):
        r = p[col].to_numpy()
        sr = r.mean() / r.std(ddof=1) if r.std(ddof=1) > 0 else float("nan")
        sk = float(pd.Series(r).skew()); ku = float(pd.Series(r).kurt() + 3)
        out[col] = dict(mean_bps=float(r.mean()), sharpe_ann=float(sr * math.sqrt(252)), nw=nw_t(r),
                        deflated_prob=deflated_sharpe_prob(sr, len(r), sk, ku, TRIAL_COUNT))
    out["days"] = int(len(p))
    return out


def q4(nd):
    out = {"primary": {}, "secondary": {}}
    cells, pvals = [], []
    for fam, g in nd.groupby("family"):
        g = g[g.has_large.astype(bool)]
        for target in TARGETS:
            r = fm_cell(g, target)
            key = "%s_%s" % (fam, target)
            out["primary"][key] = r
            cells.append(key)
            pvals.append(p_two_sided(r["S_sig"]["t"]) if r and r["S_sig"]["t"] is not None else 1.0)
            sec = {}
            sec["pseudo"] = fm_cell(g, target, signal="pseudo")
            for k in ("k25", "k75"):
                sec["kappa_" + k] = fm_cell(g, target, kappa=k)
            sec["portfolio_2bps"] = portfolio(g, target, cost_side_bps=2.0)
            sec["portfolio_1bp"] = portfolio(g, target, cost_side_bps=1.0)
            out["secondary"][key] = sec
    passed = holm(pvals)
    out["holm"] = {c: dict(p=p, holm_pass=ps, t_gt_3=bool(out["primary"][c] and out["primary"][c]["S_sig"]["t"] is not None
                                                          and abs(out["primary"][c]["S_sig"]["t"]) > 3))
                   for c, p, ps in zip(cells, pvals, passed)}
    return out


# ---------------------------------------------------------------------------------------------
# Phase I descriptives (sample) and Q5 heterogeneity (name-day table)

def phase1_tables(sample):
    """Mean d_close, d_open (bps) and median phi_close by deciles of burst characteristics, per family."""
    out = {}
    for fam, g in sample.groupby("family"):
        g = g.replace([np.inf, -np.inf], np.nan)
        g = g.assign(duration=g.t_e - g.t_b)
        res = {}
        for col in ("duration", "log_q_adv", "peak_bps", "ratio", "tod", "spread_b"):
            x = g[col]
            ok = x.notna()
            if ok.sum() < 100:
                continue
            dec = pd.qcut(x[ok].rank(method="first"), 10, labels=False)
            h = g[ok].assign(dec=dec)
            t = h.groupby("dec").agg(lo=(col, "min"), hi=(col, "max"), n=("d_close", "size"),
                                     d_close=("d_close", "mean"), d_open=("d_open", "mean"),
                                     phi_close_median=("phi_close", "median"), info_share=("info_k50", "mean"))
            res[col] = t.round(4).reset_index().to_dict(orient="list")
        out[fam] = res
    return out


def q5(nd):
    """Q1 primary statistic and the Q4 primary coefficient by listing exchange, coverage, size, tick constraint, year."""
    out = {}
    for fam, g in nd.groupby("family"):
        g = g[g.has_large.astype(bool)].copy()
        g["exchange"] = g.primaryexch.map({"N": "NYSE", "Q": "NASDAQ", "A": "AMEX"})
        g["coverage"] = (g.buy_1550 + g.sell_1550 + g.unsigned_1550) / g.dlyvol
        g["year"] = g.date.str[:4]
        splits = {"exchange": g.exchange, "year": g.year}
        for col, name in (("coverage", "coverage_tercile"), ("cap_lag1", "size_tercile")):
            ranks = g.groupby("date")[col].rank(pct=True)
            splits[name] = pd.cut(ranks, [0, 1 / 3, 2 / 3, 1.0], labels=["low", "mid", "high"])
        splits["tick_constrained"] = np.where(g.spread_1550 / 1e4 * g.mid_1550 <= 0.015, "yes", "no")
        res = {}
        for name, lab in splits.items():
            res[name] = {}
            for level, h in g.groupby(np.asarray(lab)):
                info = safe_mean(h.sum_dclose_info_k50, h.n_dclose_info_k50)
                pseudo = safe_mean(h.sum_dclose_pseudo_k50, h.n_dclose_pseudo_k50)
                daily = pd.Series(info - pseudo).groupby(h.date.to_numpy()).mean()
                entry = dict(namedays=int(len(h)), q1_d_pseudo=nw_t(daily.to_numpy()))
                for target in ("CLOP", "CLCL", "tCLOSE"):
                    r = fm_cell(h, target, min_obs=20)
                    entry["q4_" + target] = r["S_sig"] if r else None
                res[name][str(level)] = entry
        out[fam] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=["DEV", "VAL", "TEST", "ERA2"])
    ap.add_argument("--dir", required=True, help="directory with nameday_<cell>.csv.gz and strata_<cell>.csv.gz")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--sample", action="store_true", help="also write Phase I tables from sample_<cell>.csv.gz")
    ap.add_argument("--q5", action="store_true", help="also write Q5 heterogeneity (VAL/TEST)")
    args = ap.parse_args()
    nd_path = Path(args.dir) / ("nameday_%s.csv.gz" % args.cell)
    st_path = Path(args.dir) / ("strata_%s.csv.gz" % args.cell)
    check_protocol(args.cell, [nd_path, st_path])
    rng = np.random.default_rng(20260915)
    nd = pd.read_csv(nd_path, dtype={"date": str})
    strata = pd.read_csv(st_path, dtype={"date": str})
    res = dict(cell=args.cell, inputs={nd_path.name: sha(nd_path), st_path.name: sha(st_path)},
               namedays=int(nd.groupby("family").size().max()), names=int(nd.permno.nunique()),
               Q1=q1(nd, rng, args.boot), Q2a=q2a(strata, rng, args.boot), Q4=q4(nd))
    out = ANALYSIS / ("%s_primary.json" % args.cell)
    out.write_text(json.dumps(res, indent=1, default=float))
    print("wrote", out)
    if args.sample:
        sp = Path(args.dir) / ("sample_%s.csv.gz" % args.cell)
        tables = phase1_tables(pd.read_csv(sp, dtype={"date": str}))
        (ANALYSIS / ("%s_phase1.json" % args.cell)).write_text(json.dumps(tables, indent=1, default=float))
    if args.q5:
        (ANALYSIS / ("%s_q5.json" % args.cell)).write_text(json.dumps(q5(nd), indent=1, default=float))


if __name__ == "__main__":
    main()
