#!/usr/bin/env python3
"""P4 revisit v1, stage 3: Q2(b)-(e) external institutional and retail proxies (P4_REVISIT_DESIGN.md section 6).

Signals per name-day (15:50 clock, kappa 0.5): x_info = S_info / ADV20 and x_other = (S_large - S_info) / ADV20.
(b) 13F: dIO_q = 13F shares / shares outstanding at quarter end q minus the same at q-1 (WRDS SEC Analytics sums,
    p4_external.py). A quarter enters only when its cross-sectional median filer count is at least 100: the
    structured filings are thin before 2013Q2 (median 8-17) and the newest quarter is still incomplete. Pooled OLS of dIO on quarter sums of x_info and x_other plus prior-quarter return, log size and
    mean turnover, quarter fixed effects, standard errors clustered by PERMNO. Pass: b_info - b_other > 0, t > 2.
(c) CRSP mutual funds: dMF_q from portfolios reporting at both quarter ends (split-adjusted with the CRSP price
    factor), and flow-induced trading FIT_q (Lou 2012); same regression and pass rule, reported separately.
(d) S&P 500 and Nasdaq-100 changes: z = (mean x over E-5..E-1 - mean over E-60..E-21) / s.d. over E-60..E-21 for
    x_info and x_other; additions minus deletions of (z_info - z_other), Welch t. Pass: > 0, t > 2.
(e) BJZZ retail: daily cross-sectional corr(x_info, retail imbalance) - corr(x_other, retail imbalance), NW(10).
    Fail if > 0 with t > 2.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import p4_analyze as PA

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "p4"


def signals(nd, family):
    g = nd[(nd.family == family) & nd.has_large.astype(bool)].copy()
    g["x_info"] = g.S_info_1550_k50 / g.adv20
    g["x_other"] = (g.S_large_1550 - g.S_info_1550_k50) / g.adv20
    g["dt"] = pd.to_datetime(g.date)
    g["quarter"] = g.dt.dt.to_period("Q")
    return g.replace([np.inf, -np.inf], np.nan)


def crsp_daily(years):
    frames = [pd.read_csv(DATA / ("crsp_%d.csv.gz" % y), dtype={"date": str}) for y in years
              if (DATA / ("crsp_%d.csv.gz" % y)).exists()]
    c = pd.concat(frames, ignore_index=True).drop_duplicates(["permno", "date"])
    c["dt"] = pd.to_datetime(c.date)
    for col in ("dlyret", "shrout", "dlycap", "dlycumfacpr", "dlyvol"):
        c[col] = pd.to_numeric(c[col], errors="coerce")
    return c


def quarter_end_state(c):
    """Per PERMNO and calendar quarter: last shares outstanding, cap, price factor, compounded return."""
    c = c.sort_values(["permno", "dt"]).copy()
    c["quarter"] = c.dt.dt.to_period("Q")
    agg = c.groupby(["permno", "quarter"]).agg(shrout=("shrout", "last"), cap=("dlycap", "last"),
                                                fac=("dlycumfacpr", "last"),
                                                ret=("dlyret", lambda r: float(np.prod(1 + r.dropna()) - 1)))
    return agg.reset_index()


def cluster_ols(y, X, groups):
    """OLS with cluster-robust covariance (CR1)."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    uniq, inv = np.unique(groups, return_inverse=True)
    scores = np.zeros((len(uniq), X.shape[1]))
    np.add.at(scores, inv, X * e[:, None])
    meat = scores.T @ scores
    n, k, G = len(y), X.shape[1], len(uniq)
    adj = G / (G - 1) * (n - 1) / (n - k) if G > 1 and n > k else 1.0
    V = adj * XtX_inv @ meat @ XtX_inv
    return b, V


def fe_regression(df, ycol, xcols, fe="quarter", cluster="permno"):
    d = df[[ycol, fe, cluster] + xcols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 50:
        return None
    for col in [ycol] + xcols:
        lo, hi = d[col].quantile([0.01, 0.99])
        d[col] = d[col].clip(lo, hi)
    dm = d.groupby(fe)[[ycol] + xcols].transform(lambda s: s - s.mean())
    b, V = cluster_ols(dm[ycol].to_numpy(float), dm[xcols].to_numpy(float), d[cluster].to_numpy())
    out = {x: dict(b=float(b[i]), t=float(b[i] / math.sqrt(V[i, i])) if V[i, i] > 0 else None) for i, x in enumerate(xcols)}
    i, j = xcols.index("q_info"), xcols.index("q_other")
    var = V[i, i] + V[j, j] - 2 * V[i, j]
    out["info_minus_other"] = dict(b=float(b[i] - b[j]), t=float((b[i] - b[j]) / math.sqrt(var)) if var > 0 else None)
    out["pass"] = bool(out["info_minus_other"]["t"] is not None and out["info_minus_other"]["b"] > 0
                       and out["info_minus_other"]["t"] > 2)
    out["n"] = int(len(d)); out["names"] = int(d[cluster].nunique()); out["quarters"] = int(d[fe].nunique())
    return out


def quarterly_panel(g, state):
    q = g.groupby(["permno", "quarter"]).agg(q_info=("x_info", "sum"), q_other=("x_other", "sum"),
                                              turn=("turn20", "mean"), days=("x_info", "count")).reset_index()
    s = state.set_index(["permno", "quarter"])
    prev = state.copy(); prev["quarter"] = prev.quarter + 1
    prev = prev.set_index(["permno", "quarter"])
    q = q.join(s[["shrout", "fac"]], on=["permno", "quarter"])
    q = q.join(prev[["ret", "cap", "shrout", "fac"]].rename(columns={"ret": "ret_prev", "cap": "cap_prev",
                                                                     "shrout": "shrout_prev", "fac": "fac_prev"}),
               on=["permno", "quarter"])
    q["log_cap_prev"] = np.log(q.cap_prev)
    return q[q.days >= 20]


MIN_MEDIAN_FILERS = 100   # amendment A6: 13F structured data is thin before 2013Q2 and incomplete in the newest quarter


def test_13f(q, cusip_hist):
    rows = []
    dropped = []
    ch = cusip_hist.dropna(subset=["cusip"]).copy()
    ch["start"] = pd.to_datetime(ch.secinfostartdt); ch["end"] = pd.to_datetime(ch.secinfoenddt.fillna("2099-12-31"))
    ch["cusip8"] = ch.cusip.str[:8]
    quarters = sorted(q.quarter.unique())
    f13 = {}
    for per in quarters:
        for p in (per - 1, per):
            if p in f13:
                continue
            path = DATA / "f13" / ("f13_%s.csv.gz" % p.end_time.strftime("%Y%m%d"))
            if not path.exists():
                f13[p] = None
                continue
            d = pd.read_csv(path, dtype={"cusip8": str})
            if len(d) and d.n_filers.median() >= MIN_MEDIAN_FILERS:
                f13[p] = d.set_index("cusip8").shares
            else:
                f13[p] = None
                dropped.append((str(p), int(len(d)), float(d.n_filers.median()) if len(d) else 0.0))
    for r in q.itertuples(index=False):
        a, b = f13.get(r.quarter), f13.get(r.quarter - 1)
        if a is None or b is None:
            continue
        def io(per, series, shrout):
            end = per.end_time.normalize()
            m = ch[(ch.permno == r.permno) & (ch.start <= end) & (ch.end >= end)]
            if m.empty or not np.isfinite(shrout) or shrout <= 0:
                return np.nan
            return series.get(m.cusip8.iloc[0], 0.0) / (shrout * 1000.0)
        rows.append(dict(permno=r.permno, quarter=str(r.quarter),
                         dIO=io(r.quarter, a, r.shrout) - io(r.quarter - 1, b, r.shrout_prev)))
    d = q.assign(quarter=q.quarter.astype(str)).merge(pd.DataFrame(rows), on=["permno", "quarter"])
    res = fe_regression(d, "dIO", ["q_info", "q_other", "ret_prev", "log_cap_prev", "turn"])
    if res is not None:
        res["quarters_dropped_incomplete"] = sorted(set(dropped))
    return res


def test_mf(q, years):
    out_rows = []
    for y in years:
        hp = DATA / ("mf_holdings_%d.csv.gz" % y)
        if not hp.exists():
            continue
        h = pd.read_csv(hp)
        h["report_dt"] = pd.to_datetime(h.report_dt)
        h["quarter"] = h.report_dt.dt.to_period("Q")
        pivot = h.groupby(["crsp_portno", "permno", "quarter"]).nbr_shares.sum()
        for per in sorted(h.quarter.unique()):
            if per - 1 not in set(h.quarter.unique()):
                continue
            now = pivot.xs(per, level="quarter"); before = pivot.xs(per - 1, level="quarter")
            ports = set(now.index.get_level_values(0)) & set(before.index.get_level_values(0))
            nb = now[now.index.get_level_values(0).isin(ports)].groupby(level="permno").sum()
            bb = before[before.index.get_level_values(0).isin(ports)].groupby(level="permno").sum()
            both = pd.concat([nb.rename("now"), bb.rename("before")], axis=1).fillna(0.0)
            both["quarter"] = str(per)
            out_rows.append(both.reset_index())
    if not out_rows:
        return None
    mf = pd.concat(out_rows, ignore_index=True)
    d = q.assign(quarter=q.quarter.astype(str)).merge(mf, on=["permno", "quarter"])
    # CRSP price factor puts share counts on a common basis: shares_adj = shares * fac
    d["dMF"] = (d.now * d.fac - d.before * d.fac_prev) / (d.shrout * 1000.0 * d.fac)
    return fe_regression(d, "dMF", ["q_info", "q_other", "ret_prev", "log_cap_prev", "turn"])


def fit_panel(years):
    """Flow-induced trading (Lou 2012) per PERMNO-quarter: sum over portfolios of prior-quarter-end holdings x the
    portfolio's quarterly flow (TNA_q - TNA_p (1 + r)) / TNA_p, flows winsorized 1/99%; shares on the CRSP price-factor
    basis, divided by shares outstanding on the same basis (program-evidence-v1 construction)."""
    rows = []
    for y in years:
        paths = [DATA / ("%s_%d.csv.gz" % (k, y)) for k in ("mf_holdings", "mf_portno_map", "mf_monthly")]
        if not all(p.exists() for p in paths):
            continue
        h, mp, mo = (pd.read_csv(p) for p in paths)
        h["report_dt"] = pd.to_datetime(h.report_dt)
        mo["caldt"] = pd.to_datetime(mo.caldt); mp["begdt"] = pd.to_datetime(mp.begdt); mp["enddt"] = pd.to_datetime(mp.enddt)
        mo = mo.merge(mp, on="crsp_fundno")
        mo = mo[(mo.caldt >= mo.begdt) & (mo.caldt <= mo.enddt)]
        mo["mret"] = pd.to_numeric(mo.mret, errors="coerce"); mo["mtna"] = pd.to_numeric(mo.mtna, errors="coerce")
        mo["month"] = mo.caldt.dt.to_period("M")
        mo["wret"] = mo.mret * mo.mtna
        port = mo.groupby(["crsp_portno", "month"]).agg(tna=("mtna", "sum"), wret=("wret", "sum"),
                                                        nret=("mret", "count"), n=("mret", "size")).reset_index()
        port["ret"] = np.where((port.tna > 0) & (port.nret == port.n), port.wret / port.tna, np.nan)
        qe = sorted(h.report_dt.unique())
        for P, Q in zip(qe[:-1], qe[1:]):
            P, Q = pd.Timestamp(P), pd.Timestamp(Q)
            mP = P.to_period("M"); months = [Q.to_period("M") - k for k in (2, 1, 0)]
            a = port[port.month == mP].set_index("crsp_portno").tna
            b = port[port.month == months[-1]].set_index("crsp_portno").tna
            r = port[port.month.isin(months)].groupby("crsp_portno").ret.apply(lambda v: np.prod(1 + v) - 1 if v.notna().sum() == 3 else np.nan)
            f = pd.concat([a.rename("tna_p"), b.rename("tna_q"), r.rename("r")], axis=1).dropna()
            f = f[f.tna_p > 1]
            f["flow"] = (f.tna_q - f.tna_p * (1 + f.r)) / f.tna_p
            lo, hi = f.flow.quantile([0.01, 0.99]); f["flow"] = f.flow.clip(lo, hi)
            hp = h[h.report_dt == P][["crsp_portno", "permno", "nbr_shares"]].merge(f.flow.reset_index(), on="crsp_portno")
            s = (hp.nbr_shares * hp.flow).groupby(hp.permno).sum().rename("fit_sh").reset_index()
            s["quarter"] = str(Q.to_period("Q"))
            rows.append(s)
    return pd.concat(rows, ignore_index=True).drop_duplicates(["permno", "quarter"]) if rows else None


def test_fit(q, years):
    fit = fit_panel(years)
    if fit is None:
        return None
    d = q.assign(quarter=q.quarter.astype(str)).merge(fit, on=["permno", "quarter"])
    d["FIT"] = d.fit_sh * d.fac_prev / (d.shrout * 1000.0 * d.fac)
    return fe_regression(d, "FIT", ["q_info", "q_other", "ret_prev", "log_cap_prev", "turn"])


def test_index(g, events):
    res = []
    ev = events.copy(); ev["dt"] = pd.to_datetime(ev.date, format="%Y%m%d")
    by = {p: h.sort_values("dt").reset_index(drop=True) for p, h in g.groupby("permno")}
    for e in ev.itertuples(index=False):
        h = by.get(e.permno)
        if h is None:
            continue
        k = int(np.searchsorted(h.dt.to_numpy(), np.datetime64(e.dt)))
        if k < 60 or k >= len(h):
            continue
        base = h.iloc[k - 60:k - 20]; pre = h.iloc[k - 5:k]
        z = {}
        for col in ("x_info", "x_other"):
            sd = base[col].std()
            z[col] = (pre[col].mean() - base[col].mean()) / sd if sd and np.isfinite(sd) and sd > 0 else np.nan
        res.append(dict(index=e.index, kind=e.kind, permno=e.permno, date=e.date, z_info=z["x_info"], z_other=z["x_other"]))
    r = pd.DataFrame(res).dropna()
    if r.empty or r.kind.nunique() < 2:
        return dict(events=int(len(r)))
    r["diff"] = r.z_info - r.z_other
    out = dict(events=int(len(r)), additions=int((r.kind == "add").sum()), deletions=int((r.kind == "delete").sum()))
    for col in ("z_info", "z_other", "diff"):
        a, b = r[r.kind == "add"][col], r[r.kind == "delete"][col]
        se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) if len(a) > 1 and len(b) > 1 else float("nan")
        out[col] = dict(add_minus_delete=float(a.mean() - b.mean()), welch_t=float((a.mean() - b.mean()) / se) if se > 0 else None)
    out["pass"] = bool(out["diff"]["welch_t"] is not None and out["diff"]["add_minus_delete"] > 0 and out["diff"]["welch_t"] > 2)
    return out


def test_retail(g, years):
    frames = []
    for y in years:
        p = DATA / ("iid_%d.csv.gz" % y)
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        return None
    iid = pd.concat(frames, ignore_index=True)
    iid["date"] = pd.to_datetime(iid.date).dt.strftime("%Y%m%d")
    tot = iid.buyvol_retail + iid.sellvol_retail
    iid["RI"] = np.where(tot > 0, (iid.buyvol_retail - iid.sellvol_retail) / tot, np.nan)
    c = crsp_daily(years)[["permno", "date", "ticker"]]
    iid = iid.merge(c, left_on=["date", "sym_root"], right_on=["date", "ticker"])
    d = g[["permno", "date", "x_info", "x_other"]].merge(iid[["permno", "date", "RI"]], on=["permno", "date"]).dropna()
    diffs = []
    for day, h in d.groupby("date"):
        if len(h) < 30:
            continue
        ci = h[["x_info", "RI"]].corr(method="pearson").iloc[0, 1]
        co = h[["x_other", "RI"]].corr(method="pearson").iloc[0, 1]
        if np.isfinite(ci) and np.isfinite(co):
            diffs.append(ci - co)
    r = PA.nw_t(np.array(diffs))
    return dict(corr_info_minus_other=r, fail=bool(r["t"] is not None and r["mean"] > 0 and r["t"] > 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=["DEV", "VAL", "TEST", "ERA2"])
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    nd_path = Path(args.dir) / ("nameday_%s.csv.gz" % args.cell)
    PA.check_protocol(args.cell, [nd_path])
    nd = pd.read_csv(nd_path, dtype={"date": str})
    years = sorted(nd.date.str[:4].astype(int).unique())
    c = crsp_daily(sorted(set(years) | {min(years) - 1}))
    state = quarter_end_state(c)
    cusip_hist = pd.read_csv(DATA / "cusip_hist.csv.gz", dtype={"cusip": str})
    events = pd.read_csv(DATA / "index_events.csv", dtype={"date": str})
    events = events[events.date.str[:4].astype(int).isin(years)]
    out = dict(cell=args.cell, inputs={nd_path.name: PA.sha(nd_path)})
    for fam in sorted(nd.family.unique()):
        g = signals(nd, fam)
        q = quarterly_panel(g, state)
        out[fam] = dict(b_13F=test_13f(q, cusip_hist), c_MF=test_mf(q, years), c_FIT=test_fit(q, years),
                        d_index=test_index(g, events),
                        e_retail=test_retail(g, years))
    path = PA.ANALYSIS / ("%s_q2ext.json" % args.cell)
    path.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", path)


if __name__ == "__main__":
    main()
