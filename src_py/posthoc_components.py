#!/usr/bin/env python3
"""POST HOC (2026-09-14, after program-evidence-v1 D1/D4 were read): which component of NASDAQ signed flow
carries mutual-fund trading and index-event demand?

Components of daily signed volume: program bursts (top-quintile score), middle bursts, bottom bursts
(bottom quintile: large, book-sweeping children), and packets outside any run/60 burst. Quarterly net buying
per component (shares / shares outstanding, split-adjusted, coverage >= 80%) is regressed jointly on the
change in mutual-fund holdings, with quarter fixed effects and a name bootstrap; flow-induced trading (FIT) is
regressed on each component. Exploratory: not pre-registered.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_daily_flow as AD

ROOT = Path(__file__).resolve().parents[1]
COMP = ("program", "middle", "bottom", "other")


def main():
    rng = np.random.default_rng(20260914)
    out = {}
    for year, g in ((2024, "contig_explore_2024"), (2021, "contig_confirm_2021")):
        p = AD.load_panel(ROOT / "results" / "program_evidence_v1" / ("flows_%s.csv" % g), year)
        wr = AD.WR
        h = pd.read_csv(wr / ("mf_holdings_%d.csv.gz" % year)); fil = pd.read_csv(wr / ("mf_filers_%d.csv.gz" % year))
        c = pd.read_csv(wr / ("crsp_dsf_pit_%d.csv.gz" % year)); c["date"] = pd.to_datetime(c.date)
        h["report_dt"] = pd.to_datetime(h.report_dt); fil["report_dt"] = pd.to_datetime(fil.report_dt)
        h = h.dropna(subset=["permno", "nbr_shares"]).copy(); h["permno"] = h.permno.astype(np.int64)
        c["permno"] = c.permno.astype(np.int64)
        cf = c[["permno", "date", "cfacshr", "shrout"]].dropna().sort_values("date")
        h = pd.merge_asof(h.sort_values("report_dt"), cf.rename(columns={"date": "report_dt"}), on="report_dt", by="permno", direction="backward")
        h["adj_shares"] = h.nbr_shares * h.cfacshr
        qe = sorted(pd.to_datetime(fil.report_dt.unique()))
        rows = []
        for P, Q in zip(qe[:-1], qe[1:]):
            both = set(fil[fil.report_dt == P].crsp_portno) & set(fil[fil.report_dt == Q].crsp_portno)
            hp = h[(h.report_dt == P) & h.crsp_portno.isin(both)].groupby("permno").adj_shares.sum()
            hq = h[(h.report_dt == Q) & h.crsp_portno.isin(both)].groupby("permno").adj_shares.sum()
            dd = pd.concat([hp.rename("p"), hq.rename("q")], axis=1).fillna(0.0); dd["quarter"] = Q.strftime("%Y%m%d")
            rows.append(dd.reset_index())
        mf = pd.concat(rows)
        q = p.copy(); q["d"] = pd.to_datetime(q.date)
        for comp in COMP:
            q[comp + "_sh"] = (q[comp + "_buy_vol"] - q[comp + "_sell_vol"]) * q.cfacshr
        frames = []
        for P, Q in zip(qe[:-1], qe[1:]):
            w = q[(q.d > P) & (q.d <= Q)]
            tdays = c[(c.date > P) & (c.date <= Q)].groupby("permno").size()
            agg = w.groupby("permno")[[k + "_sh" for k in COMP]].sum()
            agg["n"] = w.groupby("permno").size()
            agg["tdays"] = tdays.reindex(agg.index)
            agg = agg[agg.n >= 0.8 * agg.tdays]
            for k in COMP:
                agg[k + "_sh"] *= agg.tdays / agg.n
            last = c[c.date <= Q].sort_values("date").groupby("permno").last()
            agg["adj_shrout"] = (last.shrout * 1000 * last.cfacshr).reindex(agg.index)
            agg["quarter"] = Q.strftime("%Y%m%d")
            frames.append(agg.reset_index())
        fl = pd.concat(frames)
        m = fl.merge(mf, on=["permno", "quarter"], how="inner")
        m["dMF"] = (m.q - m.p) / m.adj_shrout
        for k in COMP:
            m[k] = m[k + "_sh"] / m.adj_shrout
        for col in ("dMF",) + COMP:
            lo, hi = m[col].quantile([0.01, 0.99]); m[col] = m[col].clip(lo, hi)
        res = dict(name_quarters=int(len(m)))
        res["dMF_on_components_joint"] = AD.pooled_fe(m, "dMF", list(COMP), rng)
        fit = AD.flow_induced_trading(year, qe, h)
        m2 = fl.merge(fit, on=["permno", "quarter"], how="inner")
        for k in COMP:
            m2[k] = m2[k + "_sh"] / m2.adj_shrout
        for col in COMP + ("FIT",):
            lo, hi = m2[col].quantile([0.01, 0.99]); m2[col] = m2[col].clip(lo, hi)
        res["component_on_FIT"] = {k: AD.pooled_fe(m2, k, ["FIT"], rng)["FIT"] for k in COMP}
        # component volume shares, for scale
        tot = p.buy_vol + p.sell_vol
        res["median_volume_share"] = {k: float(((p[k + "_buy_vol"] + p[k + "_sell_vol"]) / tot).median()) for k in COMP}
        out[str(year)] = res
    # index events by component (z over E-5..E-1, additions minus deletions)
    ev = pd.read_csv(AD.WR / "index_events_2021_2024.csv")
    cal = sorted(pd.read_csv(AD.WR / "crsp_dsi.csv.gz").date.astype(str).str.replace("-", ""))
    flows = pd.concat([pd.read_csv(ROOT / "results" / "program_evidence_v1" / f) for f in
                       ("flows_events.csv", "flows_contig_explore_2024.csv", "flows_contig_confirm_2021.csv")], ignore_index=True)
    flows["date"] = flows.date.astype(str); flows = flows.drop_duplicates(["permno", "date"])
    tot = flows.buy_vol + flows.sell_vol
    for k in COMP:
        flows[k + "_imb"] = np.where(tot > 0, (flows[k + "_buy_vol"] - flows[k + "_sell_vol"]) / tot, np.nan)
    by = {kk: gg.set_index("date") for kk, gg in flows.groupby("permno")}
    zs = []
    for r in ev.itertuples():
        d = r.date.replace("-", "")
        pos = next(i for i, x in enumerate(cal) if (x >= d if r.kind == "add" else x > d))
        gg = by.get(r.permno)
        if gg is None:
            continue
        rel = {cal[pos + k]: k for k in range(-45, 11) if 0 <= pos + k < len(cal)}
        w = gg[gg.index.isin(rel)].copy(); w["k"] = [rel[x] for x in w.index]
        base = w[(w.k >= -40) & (w.k <= -11)]; pre = w[(w.k >= -5) & (w.k <= -1)]
        if len(base) < 20 or len(pre) < 3:
            continue
        row = dict(kind=r.kind)
        for k in COMP:
            sd = base[k + "_imb"].std()
            row[k] = (pre[k + "_imb"].mean() - base[k + "_imb"].mean()) / sd if sd > 0 else np.nan
        zs.append(row)
    z = pd.DataFrame(zs)
    from scipy import stats
    out["index_events"] = {k: dict(add=float(z[z.kind == "add"][k].mean()), delete=float(z[z.kind == "delete"][k].mean()),
                                   welch_t=float(stats.ttest_ind(z[z.kind == "add"][k].dropna(), z[z.kind == "delete"][k].dropna(), equal_var=False).statistic))
                           for k in COMP}
    out["index_events"]["n_add"] = int((z.kind == "add").sum()); out["index_events"]["n_delete"] = int((z.kind == "delete").sum())
    (ROOT / "results" / "program_evidence_v1" / "posthoc_components.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
