#!/usr/bin/env python3
"""P4 revisit v1: archive coverage of the requested name-days (P4_REVISIT_DESIGN.md section 2).

A name-day is covered when any status line for its (date, PERMNO) says ok (main pass or renamed-ticker retry).
Coverage is reported overall, by year, by CRSP listing exchange on that date, by the name's dollar-volume rank
decile in the universe, and by the name's prior-calendar-year return quintile (information dated before the
sample year, so no sample outcome is used).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "p4"


def prior_year_returns(years):
    out = {}
    for y in years:
        p = DATA / ("crsp_%d.csv.gz" % (y - 1))
        if not p.exists():
            continue
        c = pd.read_csv(p, usecols=["permno", "date", "dlyret"], dtype={"date": str})
        c = c[c.date.str[:4] == str(y - 1)]
        c["dlyret"] = pd.to_numeric(c.dlyret, errors="coerce")
        out[y] = c.groupby("permno").dlyret.apply(lambda r: float(np.prod(1 + r.dropna()) - 1))
    return out


def coverage(cell, jobs_path, status_paths):
    jobs = pd.read_csv(jobs_path, sep=" ", header=None, names=["date", "ticker", "permno"], dtype={"date": str})
    st = pd.concat([pd.read_csv(p, sep=" ", header=None, names=["date", "ticker", "permno", "status"], dtype={"date": str})
                    for p in status_paths], ignore_index=True)
    ok = st[st.status == "ok"][["date", "permno"]].drop_duplicates()
    ok["covered"] = 1
    j = jobs.merge(ok, on=["date", "permno"], how="left")
    j["covered"] = j.covered.fillna(0)
    j["year"] = j.date.str[:4].astype(int)
    frames = []
    for y in sorted(j.year.unique()):
        u = pd.read_csv(DATA / ("universe_%d.csv" % y))
        c = pd.read_csv(DATA / ("crsp_%d.csv.gz" % y), usecols=["permno", "date", "primaryexch"], dtype={"date": str})
        part = j[j.year == y].merge(c, on=["permno", "date"], how="left").merge(u[["permno", "rank"]], on="permno", how="left")
        frames.append(part)
    j = pd.concat(frames, ignore_index=True)
    j["rank_decile"] = ((j["rank"] - 1) // 100 + 1).astype("Int64")
    pr = prior_year_returns(sorted(j.year.unique()))
    j["prior_ret"] = [pr.get(y, pd.Series(dtype=float)).get(p, np.nan) for y, p in zip(j.year, j.permno)]
    j["prior_ret_q"] = j.groupby("year").prior_ret.transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1
                                                             if s.notna().sum() > 10 else np.nan)
    res = dict(cell=cell, requested=int(len(j)), covered=int(j.covered.sum()), share=float(j.covered.mean()),
               names_requested=int(j.permno.nunique()), names_with_any=int(j[j.covered == 1].permno.nunique()))
    for key in ("year", "primaryexch", "rank_decile", "prior_ret_q"):
        t = j.groupby(key).covered.agg(["size", "mean"])
        res["by_" + key] = {str(k): dict(requested=int(r["size"]), share=round(float(r["mean"]), 4)) for k, r in t.iterrows()}
    renamed = st[(st.status == "ok")].merge(jobs, on=["date", "permno"], suffixes=("", "_job"))
    res["covered_via_later_ticker"] = int((renamed.ticker != renamed.ticker_job).sum())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--status", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = coverage(args.cell, args.jobs, args.status)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("by_")}, indent=1))
    for k in ("by_year", "by_primaryexch", "by_prior_ret_q"):
        print(k, {kk: vv["share"] for kk, vv in res[k].items()})


if __name__ == "__main__":
    main()
