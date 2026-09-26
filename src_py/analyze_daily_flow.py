#!/usr/bin/env python3
"""program-evidence-v1 modules D1-D3 and E1-E3 on a contiguous daily-flow panel (one year/group).

Inputs: collected daily_flow.py rows for the group, CRSP daily file, WRDS Intraday Indicators,
CRSP mutual-fund holdings/TNA, ETF Global flow demand. See PROGRAM_EVIDENCE_DESIGN.md (modules D, E
and the 2026-09-14 amendments). Fama-MacBeth slopes use Newey-West (10 lags) on the daily series.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WR = ROOT / "data" / "wrds"
NW_LAGS = 10
BOOT = 1000


def nw_mean_t(x, lags=NW_LAGS):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return dict(mean=None, t=None, n=int(n))
    m = x.mean(); e = x - m
    v = e @ e / n
    for l in range(1, min(lags, n - 1) + 1):
        v += 2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / n
    return dict(mean=float(m), t=float(m / np.sqrt(v / n)) if v > 0 else None, n=int(n))


def fm(panel, y, xs, min_obs=30):
    """Daily cross-sectional OLS; returns per-date slope frame."""
    rows = []
    for d, g in panel.groupby("date"):
        g = g[[y] + xs].replace([np.inf, -np.inf], np.nan).dropna()
        if len(g) < max(min_obs, len(xs) + 5):
            continue
        X = np.column_stack([np.ones(len(g))] + [g[x].to_numpy(float) for x in xs])
        b = np.linalg.lstsq(X, g[y].to_numpy(float), rcond=None)[0]
        rows.append(dict(date=d, **{x: b[i + 1] for i, x in enumerate(xs)}))
    return pd.DataFrame(rows)


def load_panel(flow_path, year):
    f = pd.read_csv(flow_path)
    f["date"] = f.date.astype(str)
    tot = f.buy_vol + f.sell_vol
    f = f[tot > 0].copy(); tot = tot[tot > 0]
    pb, ps = f.program_buy_vol, f.program_sell_vol
    ob, os_ = f.buy_vol - pb, f.sell_vol - ps
    f["PI"] = (pb - ps) / tot; f["NPI"] = (ob - os_) / tot; f["OI"] = (f.buy_vol - f.sell_vol) / tot
    f["PIR"] = np.where(pb + ps > 0, (pb - ps) / (pb + ps), np.nan)
    f["NPIR"] = np.where(ob + os_ > 0, (ob - os_) / (ob + os_), np.nan)
    f["program_share"] = (pb + ps) / tot
    c = pd.read_csv(WR / ("crsp_dsf_pit_%d.csv.gz" % year))
    c["date"] = pd.to_datetime(c.date).dt.strftime("%Y%m%d")
    c = c.sort_values(["permno", "date"])
    c["ret"] = pd.to_numeric(c.ret, errors="coerce")
    c["prc"] = c.prc.abs(); c["openprc"] = c.openprc.abs()
    c["r_oc"] = c.prc / c.openprc - 1
    c["mcap"] = c.prc * c.shrout * 1000
    g = c.groupby("permno")
    c["ret_next"] = g.ret.shift(-1); c["r_oc_next"] = g.r_oc.shift(-1)
    c["ret_next5"] = np.exp(sum(np.log1p(g.ret.shift(-k)) for k in range(1, 6))) - 1
    c["log_mcap_lag"] = np.log(g.mcap.shift(1))
    c["date_next"] = g.date.shift(-1)
    p = f.merge(c[["permno", "date", "ret", "ret_next", "r_oc_next", "ret_next5", "log_mcap_lag", "date_next", "prc",
                   "shrout", "cfacshr", "vol"]], on=["permno", "date"], how="left")
    p = p[p.date.str[:4] == str(year)]
    # next trading day's flow for persistence (must be the next CRSP trading day)
    nxt = p[["permno", "date", "PI", "NPI", "PIR", "NPIR", "OI"]].rename(
        columns={"date": "date_next", "PI": "PI_next", "NPI": "NPI_next", "PIR": "PIR_next", "NPIR": "NPIR_next", "OI": "OI_next"})
    p = p.merge(nxt, on=["permno", "date_next"], how="left")
    return p


def coverage(p, year):
    pit = pd.read_csv(ROOT / "data" / "evidence" / ("pit_%d.csv" % year))
    grp = pit[pit.group == 0] if year == 2024 else pit[pit.group != 0]
    have = p.groupby("permno").size()
    c = pd.read_csv(WR / ("crsp_dsf_pit_%d.csv.gz" % year))
    c["date"] = pd.to_datetime(c.date); c = c[c.date.dt.year == year]
    c["ret"] = pd.to_numeric(c.ret, errors="coerce")
    annual = c.groupby("permno").ret.apply(lambda r: float(np.exp(np.log1p(r.dropna()).sum()) - 1))
    days = c.groupby("permno").size()
    covered = grp.permno[grp.permno.isin(have.index[have >= 0.8 * days.reindex(have.index).fillna(252)])]
    uncovered = grp.permno[~grp.permno.isin(covered)]
    return dict(group_names=int(len(grp)), names_with_flow=int(p.permno.nunique()),
                names_covered_80pct=int(len(covered)), name_days=int(len(p)),
                mean_annual_return_covered=float(annual.reindex(covered).mean()),
                mean_annual_return_not_covered=float(annual.reindex(uncovered).mean()),
                median_program_volume_share=float(p.program_share.median()))


def e_tests(p):
    out = {}
    s1 = fm(p, "PI_next", ["PI", "NPI"]); s2 = fm(p, "NPI_next", ["PI", "NPI"])
    m = s1.merge(s2, on="date", suffixes=("_toPI", "_toNPI"))
    out["E1"] = dict(PI_to_PI=nw_mean_t(m.PI_toPI), NPI_to_NPI=nw_mean_t(m.NPI_toNPI), NPI_to_PI=nw_mean_t(m.NPI_toPI),
                     PI_to_NPI=nw_mean_t(m.PI_toNPI), difference=nw_mean_t(m.PI_toPI - m.NPI_toNPI))
    s1r = fm(p, "PIR_next", ["PIR", "NPIR"]); s2r = fm(p, "NPIR_next", ["PIR", "NPIR"])
    mr = s1r.merge(s2r, on="date", suffixes=("_toPIR", "_toNPIR"))
    out["E1_within_type"] = dict(PIR_to_PIR=nw_mean_t(mr.PIR_toPIR), NPIR_to_NPIR=nw_mean_t(mr.NPIR_toNPIR),
                                 difference=nw_mean_t(mr.PIR_toPIR - mr.NPIR_toNPIR))
    diff_t = out["E1"]["difference"]["t"]
    out["E1_gate_stat_t"] = diff_t
    e2 = {}
    q = p.copy()
    for col in ("ret_next", "r_oc_next", "ret_next5"):
        q[col + "_bps"] = q[col] * 1e4
    q["ret_bps"] = q.ret * 1e4
    for y in ("ret_next_bps", "r_oc_next_bps", "ret_next5_bps"):
        s = fm(q, y, ["PI", "NPI", "ret_bps", "log_mcap_lag"])
        e2[y] = {x: nw_mean_t(s[x]) for x in ("PI", "NPI", "ret_bps")}
        s_std = fm(q.assign(PIz=q.groupby("date").PI.transform(lambda v: (v - v.mean()) / v.std()),
                            NPIz=q.groupby("date").NPI.transform(lambda v: (v - v.mean()) / v.std())),
                   y, ["PIz", "NPIz", "ret_bps", "log_mcap_lag"])
        e2[y]["PI_per_sd_bps"] = nw_mean_t(s_std.PIz); e2[y]["NPI_per_sd_bps"] = nw_mean_t(s_std.NPIz)
    out["E2"] = e2
    ts = [abs(e2[y]["PI"]["t"]) for y in e2 if e2[y]["PI"]["t"] is not None]
    out["E2_max_abs_t_PI"] = float(max(ts)) if ts else None
    return out


def d2_retail(p, year):
    iid = pd.read_csv(WR / ("iid_%d.csv.gz" % year))
    iid["date"] = pd.to_datetime(iid.date).dt.strftime("%Y%m%d")
    rv = iid.buyvol_retail + iid.sellvol_retail; iv = iid.buyvol_inst50k + iid.sellvol_inst50k
    iid["RI"] = np.where(rv > 0, (iid.buyvol_retail - iid.sellvol_retail) / rv, np.nan)
    iid["I50"] = np.where(iv > 0, (iid.buyvol_inst50k - iid.sellvol_inst50k) / iv, np.nan)
    q = p.merge(iid[["date", "sym_root", "RI", "I50"]], left_on=["date", "ticker"], right_on=["date", "sym_root"], how="inner")
    res = dict(name_days=int(len(q)))
    for a in ("PI", "NPI", "PIR", "NPIR"):
        for b in ("RI", "I50"):
            cors = q.groupby("date").apply(lambda g: g[a].corr(g[b], method="spearman") if g[[a, b]].dropna().shape[0] >= 30 else np.nan)
            res["corr_%s_%s" % (a, b)] = nw_mean_t(cors)
    for b in ("RI", "I50"):
        d = q.groupby("date").apply(lambda g: (g["PIR"].corr(g[b], method="spearman") - g["NPIR"].corr(g[b], method="spearman"))
                                    if g[["PIR", "NPIR", b]].dropna().shape[0] >= 30 else np.nan)
        res["PIR_minus_NPIR_corr_with_%s" % b] = nw_mean_t(d)
    return res


def d3_etf(p, year):
    e = pd.read_csv(WR / ("etf_flow_demand_%d.csv.gz" % year))
    e["date"] = pd.to_datetime(e.as_of_date).dt.strftime("%Y%m%d")
    q = p.merge(e[["date", "constituent_ticker", "flow_demand_usd"]], left_on=["date", "ticker"],
                right_on=["date", "constituent_ticker"], how="left")
    q["flow_demand_usd"] = q.flow_demand_usd.fillna(0.0)
    mc = np.exp(q.log_mcap_lag)
    q["ETFD"] = q.flow_demand_usd / mc
    q = q.sort_values(["permno", "date"])
    q["ETFD_lead"] = q.groupby("permno").ETFD.shift(-1)
    for col in ("ETFD", "ETFD_lead"):
        q[col + "_z"] = q.groupby("date")[col].transform(lambda v: (v - v.mean()) / v.std() if v.std() > 0 else v * np.nan)
    res = {}
    for y in ("PI", "NPI", "PIR", "NPIR"):
        s = fm(q, y, ["ETFD_z", "ETFD_lead_z"])
        res[y] = {x: nw_mean_t(s[x]) for x in ("ETFD_z", "ETFD_lead_z")}
    for x in ("ETFD_z", "ETFD_lead_z"):
        a = fm(q, "PIR", ["ETFD_z", "ETFD_lead_z"]); b = fm(q, "NPIR", ["ETFD_z", "ETFD_lead_z"])
        m = a.merge(b, on="date", suffixes=("_p", "_n"))
        res["PIR_minus_NPIR_" + x] = nw_mean_t(m[x + "_p"] - m[x + "_n"])
        a = fm(q, "PI", ["ETFD_z", "ETFD_lead_z"]); b = fm(q, "NPI", ["ETFD_z", "ETFD_lead_z"])
        m = a.merge(b, on="date", suffixes=("_p", "_n"))
        res["PI_minus_NPI_" + x] = nw_mean_t(m[x + "_p"] - m[x + "_n"])
    res["name_days_with_etf_demand"] = int((q.flow_demand_usd != 0).sum())
    return res


def d1_mutual_funds(p, year, rng):
    h = pd.read_csv(WR / ("mf_holdings_%d.csv.gz" % year)); fil = pd.read_csv(WR / ("mf_filers_%d.csv.gz" % year))
    c = pd.read_csv(WR / ("crsp_dsf_pit_%d.csv.gz" % year)); c["date"] = pd.to_datetime(c.date)
    qe = sorted(pd.to_datetime(fil.report_dt.unique()))
    h["report_dt"] = pd.to_datetime(h.report_dt); fil["report_dt"] = pd.to_datetime(fil.report_dt)
    h = h.dropna(subset=["permno", "nbr_shares"]).copy(); h["permno"] = h.permno.astype(np.int64)
    c["permno"] = c.permno.astype(np.int64)
    cf = c[["permno", "date", "cfacshr", "shrout"]].dropna().sort_values("date")

    def asof(permno_dates):
        k = permno_dates.sort_values("report_dt")
        return pd.merge_asof(k, cf.rename(columns={"date": "report_dt"}), on="report_dt", by="permno", direction="backward")
    h = asof(h); h["adj_shares"] = h.nbr_shares * h.cfacshr
    rows = []
    for P, Q in zip(qe[:-1], qe[1:]):
        both = set(fil[fil.report_dt == P].crsp_portno) & set(fil[fil.report_dt == Q].crsp_portno)
        hp = h[(h.report_dt == P) & h.crsp_portno.isin(both)].groupby("permno").adj_shares.sum()
        hq = h[(h.report_dt == Q) & h.crsp_portno.isin(both)].groupby("permno").adj_shares.sum()
        d = pd.concat([hp.rename("p"), hq.rename("q")], axis=1).fillna(0.0)
        d["quarter"] = Q.strftime("%Y%m%d"); d["P"] = P
        rows.append(d.reset_index())
    mf = pd.concat(rows)
    # quarter flows from the daily panel (shares, split-adjusted), coverage >= 80% and scaled up
    q = p.copy(); q["d"] = pd.to_datetime(q.date)
    q["adj"] = q.cfacshr
    q["np_sh"] = (q.program_buy_vol - q.program_sell_vol) * q.adj
    q["nnp_sh"] = ((q.buy_vol - q.program_buy_vol) - (q.sell_vol - q.program_sell_vol)) * q.adj
    q["tot_sh"] = (q.buy_vol + q.sell_vol) * q.adj
    cal = c[c.date.dt.year == year].groupby("permno").date
    out_rows = []
    for P, Q in zip(qe[:-1], qe[1:]):
        w = q[(q.d > P) & (q.d <= Q)]
        tdays = c[(c.date > P) & (c.date <= Q)].groupby("permno").size()
        agg = w.groupby("permno").agg(np_sh=("np_sh", "sum"), nnp_sh=("nnp_sh", "sum"), tot_sh=("tot_sh", "sum"), n=("np_sh", "size"))
        agg["tdays"] = tdays.reindex(agg.index)
        agg = agg[agg.n >= 0.8 * agg.tdays]
        scale = agg.tdays / agg.n
        for col in ("np_sh", "nnp_sh", "tot_sh"):
            agg[col] = agg[col] * scale
        last = c[(c.date <= Q)].sort_values("date").groupby("permno").last()
        agg["adj_shrout"] = (last.shrout * 1000 * last.cfacshr).reindex(agg.index)
        agg["quarter"] = Q.strftime("%Y%m%d")
        out_rows.append(agg.reset_index())
    fl = pd.concat(out_rows)
    m = fl.merge(mf, on=["permno", "quarter"], how="inner")
    m["dMF"] = (m.q - m.p) / m.adj_shrout
    m["NP"] = m.np_sh / m.adj_shrout; m["NNP"] = m.nnp_sh / m.adj_shrout
    for col in ("dMF", "NP", "NNP"):
        lo, hi = m[col].quantile([0.01, 0.99]); m[col] = m[col].clip(lo, hi)
    res = dict(name_quarters=int(len(m)), names=int(m.permno.nunique()))
    res["dMF_on_NP_NNP"] = pooled_fe(m, "dMF", ["NP", "NNP"], rng)
    # flow-induced trading
    fit = flow_induced_trading(year, qe, h)
    if fit is not None:
        m2 = fl.merge(fit, on=["permno", "quarter"], how="inner")
        m2["NP"] = m2.np_sh / m2.adj_shrout; m2["NNP"] = m2.nnp_sh / m2.adj_shrout
        for col in ("NP", "NNP", "FIT"):
            lo, hi = m2[col].quantile([0.01, 0.99]); m2[col] = m2[col].clip(lo, hi)
        res["NP_on_FIT"] = pooled_fe(m2, "NP", ["FIT"], rng)
        res["NNP_on_FIT"] = pooled_fe(m2, "NNP", ["FIT"], rng)
        res["fit_name_quarters"] = int(len(m2))
    return res


def pooled_fe(m, y, xs, rng):
    d = m[[y, "quarter", "permno"] + xs].dropna()
    Y = d[y] - d.groupby("quarter")[y].transform("mean")
    X = np.column_stack([d[x] - d.groupby("quarter")[x].transform("mean") for x in xs])
    beta = np.linalg.lstsq(X, Y.to_numpy(), rcond=None)[0]
    names = d.permno.unique(); idx = {k: np.flatnonzero(d.permno.to_numpy() == k) for k in names}
    boots = []
    for _ in range(BOOT):
        pick = np.concatenate([idx[names[i]] for i in rng.integers(0, len(names), len(names))])
        boots.append(np.linalg.lstsq(X[pick], Y.to_numpy()[pick], rcond=None)[0])
    boots = np.array(boots); se = boots.std(0)
    out = {x: dict(coef=float(beta[i]), se=float(se[i]), t=float(beta[i] / se[i])) for i, x in enumerate(xs)}
    if len(xs) == 2:
        dd = boots[:, 0] - boots[:, 1]
        out["difference"] = dict(coef=float(beta[0] - beta[1]), se=float(dd.std()), t=float((beta[0] - beta[1]) / dd.std()))
    out["n"] = int(len(d))
    return out


def flow_induced_trading(year, qe, h):
    mp = pd.read_csv(WR / ("mf_portno_map_%d.csv.gz" % year)); mo = pd.read_csv(WR / ("mf_monthly_%d.csv.gz" % year))
    mo["caldt"] = pd.to_datetime(mo.caldt); mp["begdt"] = pd.to_datetime(mp.begdt); mp["enddt"] = pd.to_datetime(mp.enddt)
    mo = mo.merge(mp[["crsp_fundno", "crsp_portno", "begdt", "enddt"]], on="crsp_fundno", how="inner")
    mo = mo[(mo.caldt >= mo.begdt) & (mo.caldt <= mo.enddt)]
    mo["mret"] = pd.to_numeric(mo.mret, errors="coerce"); mo["mtna"] = pd.to_numeric(mo.mtna, errors="coerce")
    mo["month"] = mo.caldt.dt.to_period("M")
    port = mo.groupby(["crsp_portno", "month"]).apply(
        lambda g: pd.Series(dict(tna=g.mtna.sum(), ret=np.average(g.mret, weights=g.mtna) if g.mtna.sum() > 0 and g.mret.notna().all() else np.nan)))
    port = port.reset_index()
    rows = []
    for P, Q in zip(qe[:-1], qe[1:]):
        mP, months = P.to_period("M"), [Q.to_period("M") - k for k in (2, 1, 0)]
        a = port[port.month == mP].set_index("crsp_portno")
        b = port[port.month == months[-1]].set_index("crsp_portno")
        r = port[port.month.isin(months)].groupby("crsp_portno").ret.apply(lambda v: np.prod(1 + v) - 1 if len(v) == 3 else np.nan)
        f = pd.concat([a.tna.rename("tna_p"), b.tna.rename("tna_q"), r.rename("r")], axis=1).dropna()
        f = f[f.tna_p > 1]
        f["flow"] = (f.tna_q - f.tna_p * (1 + f.r)) / f.tna_p
        lo, hi = f.flow.quantile([0.01, 0.99]); f["flow"] = f.flow.clip(lo, hi)
        hp = h[h.report_dt == P][["crsp_portno", "permno", "adj_shares"]]
        x = hp.merge(f.flow.reset_index(), on="crsp_portno", how="inner")
        x["fit_sh"] = x.adj_shares * x.flow
        s = x.groupby("permno").fit_sh.sum().rename("fit_sh").reset_index()
        s["quarter"] = Q.strftime("%Y%m%d"); rows.append(s)
    if not rows:
        return None
    fit = pd.concat(rows)
    c = pd.read_csv(WR / ("crsp_dsf_pit_%d.csv.gz" % year)); c["date"] = pd.to_datetime(c.date)
    last = []
    for Q in qe[1:]:
        l = c[c.date <= Q].sort_values("date").groupby("permno").last()
        last.append(pd.DataFrame(dict(permno=l.index, quarter=Q.strftime("%Y%m%d"), adj_shrout_fit=(l.shrout * 1000 * l.cfacshr).to_numpy())))
    fit = fit.merge(pd.concat(last), on=["permno", "quarter"], how="left")
    fit["FIT"] = fit.fit_sh / fit.adj_shrout_fit
    return fit[["permno", "quarter", "FIT"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(20260914)
    p = load_panel(args.flows, args.year)
    res = dict(year=args.year, coverage=coverage(p, args.year))
    res.update(e_tests(p))
    res["D2"] = d2_retail(p, args.year)
    res["D3"] = d3_etf(p, args.year)
    res["D1"] = d1_mutual_funds(p, args.year, rng)
    Path(args.out).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps(dict(coverage=res["coverage"], E1=res["E1"]["difference"], E2_max_abs_t_PI=res["E2_max_abs_t_PI"]), indent=1))


if __name__ == "__main__":
    main()
