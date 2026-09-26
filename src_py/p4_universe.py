#!/usr/bin/env python3
"""P4 revisit v1: point-in-time universes, CRSP v2 daily data and extraction job lists.

P4_REVISIT_DESIGN.md section 2. Year Y: CRSP v2 common stocks (sharetype NS, securitytype EQTY,
securitysubtype COM, usincflg Y, issuertype ACOR/CORP, primaryexch N/Q/A), top 1,000 by average daily
dollar volume over October-December of Y-1 with at least 40 valid days. Name split on PERMNO:
sha256("p4-revisit-v1|" + permno), first 8 hex digits, mod 3. Cells: DEV 2017-2019 group 0, VAL
2020-2021 groups 1-2, TEST 2022-2025 groups 1-2, ERA2 2012-2016 all groups.

Writes (licensed-data derivatives, gitignored): data/p4/universe_<Y>.csv, data/p4/crsp_<Y>.csv.gz
(universe PERMNOs, mid-November of Y-1 to mid-January of Y+1) and results/p4_revisit_v1/jobs/<cell>.txt
with one "date ticker permno" line per requested name-day. When two universe PERMNOs carry the same
ticker on a date, only the one with the higher ranking dollar volume is requested.
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd

import wrds_access as W

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "p4"
JOBS = ROOT / "results" / "p4_revisit_v1" / "jobs"
TOP = 1000
FILTER = """sharetype = 'NS' and securitytype = 'EQTY' and securitysubtype = 'COM' and usincflg = 'Y'
            and issuertype in ('ACOR', 'CORP') and primaryexch in ('N', 'Q', 'A')"""
CELLS = {"DEV": (range(2017, 2020), {0}), "VAL": (range(2020, 2022), {1, 2}),
         "TEST": (range(2022, 2026), {1, 2}), "ERA2": (range(2012, 2017), {0, 1, 2})}


def split_group(permno):
    return int(hashlib.sha256(("p4-revisit-v1|%d" % int(permno)).encode()).hexdigest()[:8], 16) % 3


def rank(year, conn):
    liq = W.query("""
        select permno, avg(abs(dlyprc) * dlyvol) as dvol, count(*) as n_days
        from crsp.dsf_v2
        where dlycaldt between %s and %s and dlyprc is not null and dlyvol is not null and """ + FILTER + """
        group by permno""", ("%d-10-01" % (year - 1), "%d-12-31" % (year - 1)), conn=conn)
    liq = liq[liq.n_days >= 40].sort_values("dvol", ascending=False).head(TOP).reset_index(drop=True)
    liq["rank"] = liq.index + 1
    liq["group"] = liq.permno.map(split_group)
    return liq


def daily(year, permnos, conn):
    frames = []
    plist = sorted(int(p) for p in permnos)
    for i in range(0, len(plist), 250):
        frames.append(W.query("""
            select permno, dlycaldt as date, ticker, primaryexch, sharetype, securitytype, securitysubtype,
                   dlyopen, dlyclose, dlyprc, dlyret, dlyretx, dlyvol, dlyprcvol, shrout, dlycap,
                   dlycumfacpr, dlyfacprc, dlydelflg
            from crsp.dsf_v2
            where permno in %s and dlycaldt between %s and %s""",
            (tuple(plist[i:i + 250]), "%d-11-15" % (year - 1), "%d-01-15" % (year + 1)), conn=conn))
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out.date).dt.strftime("%Y%m%d")
    return out.sort_values(["permno", "date"]).reset_index(drop=True)


def job_lines(year, liq, days, groups):
    sub = days[(days.date.str[:4] == str(year)) & days.permno.isin(liq.permno[liq.group.isin(groups)])]
    sub = sub.dropna(subset=["ticker"]).merge(liq[["permno", "dvol"]], on="permno")
    # one PERMNO per (date, ticker), among the whole universe, not only the requested groups
    allday = days[days.date.str[:4] == str(year)].dropna(subset=["ticker"]).merge(liq[["permno", "dvol"]], on="permno")
    best = allday.sort_values("dvol", ascending=False).drop_duplicates(["date", "ticker"])[["date", "ticker", "permno"]]
    sub = sub.merge(best, on=["date", "ticker", "permno"])
    return sub[["date", "ticker", "permno", "dvol"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2012-2025")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.years.split("-"))
    DATA.mkdir(parents=True, exist_ok=True); JOBS.mkdir(parents=True, exist_ok=True)
    conn = W.connect()
    lines = {cell: [] for cell in CELLS}
    try:
        for year in range(lo, hi + 1):
            liq = rank(year, conn)
            liq.to_csv(DATA / ("universe_%d.csv" % year), index=False)
            path = DATA / ("crsp_%d.csv.gz" % year)
            if path.exists():
                days = pd.read_csv(path, dtype={"date": str})
            else:
                days = daily(year, liq.permno, conn)
                days.to_csv(path, index=False, compression="gzip")
            for cell, (years, groups) in CELLS.items():
                if year in years:
                    jl = job_lines(year, liq, days, groups)
                    lines[cell].append(jl)
                    print(year, cell, "names", jl.permno.nunique(), "name-days", len(jl), flush=True)
    finally:
        conn.close()
    for cell, frames in lines.items():
        if frames:
            out = pd.concat(frames, ignore_index=True).sort_values(["dvol", "permno", "date"], ascending=[False, True, True])
            out[["date", "ticker", "permno"]].to_csv(JOBS / ("%s.txt" % cell), sep=" ", header=False, index=False)
            print(cell, "total name-days", len(out), "names", out.permno.nunique())


if __name__ == "__main__":
    main()
