#!/usr/bin/env python3
"""Point-in-time universes for program-evidence-v1 contiguous panels (modules D and E).

Year Y: CRSP common stocks (share codes 10, 11), top 500 by average daily dollar volume over
October-December of Y-1, ranked with information available at the end of Y-1. Every trading day of
Y is requested under the ticker CRSP assigns on that day, so names that change ticker or delist
keep their history until lobster2 stops. Split: sha256("fingerprint-v1|" + ticker) mod 3 on the
ticker at the start of Y; 0 is exploration (used in 2024), 1-2 confirmation (used in 2021).

Writes data/evidence/pit_<Y>.csv (licensed-data derivative, gitignored) and, per group, one
"date ticker" job file per name under results/program_evidence_v1/contig_<group>/jobs/.
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd

import wrds_access as W

ROOT = Path(__file__).resolve().parents[1]


def split_group(ticker):
    """fingerprint-v1's rule exactly: first 8 hex digits of sha256("fingerprint-v1|" + ticker), mod 3."""
    return int(hashlib.sha256(("fingerprint-v1|" + ticker).encode()).hexdigest()[:8], 16) % 3


def build(year, conn, top=500):
    rank_lo, rank_hi = "%d-10-01" % (year - 1), "%d-12-31" % (year - 1)
    liq = W.query("""
        select d.permno, avg(abs(d.prc) * d.vol) as dvol, count(*) as n_days
        from crsp.dsf d
        join crsp.dsenames n on n.permno = d.permno and d.date between n.namedt and coalesce(n.nameendt, '9999-12-31')
        where d.date between %s and %s and n.shrcd in (10, 11) and d.prc is not null and d.vol is not null
        group by d.permno""", (rank_lo, rank_hi), conn=conn)
    liq = liq[liq.n_days >= 40].sort_values("dvol", ascending=False).head(top).reset_index(drop=True)
    liq["rank"] = liq.index + 1
    permnos = tuple(int(p) for p in liq.permno)
    days = W.query("""
        select d.permno, d.date, n.ticker
        from crsp.dsf d
        join crsp.dsenames n on n.permno = d.permno and d.date between n.namedt and coalesce(n.nameendt, '9999-12-31')
        where d.date between %s and %s and d.permno in %s""", ("%d-01-01" % year, "%d-12-31" % year, permnos), conn=conn)
    days["date"] = pd.to_datetime(days.date).dt.strftime("%Y%m%d")
    days = days.dropna(subset=["ticker"]).sort_values(["permno", "date"])
    start = W.query("""
        select n.permno, n.ticker from crsp.dsenames n
        where %s between n.namedt and coalesce(n.nameendt, '9999-12-31') and n.permno in %s""",
                    ("%d-12-31" % (year - 1), permnos), conn=conn).drop_duplicates("permno")
    liq = liq.merge(start.rename(columns={"ticker": "ticker_start"}), on="permno", how="left")
    liq["group"] = liq.ticker_start.fillna("").map(split_group)
    return liq, days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2021,2024")
    args = ap.parse_args()
    out = ROOT / "data" / "evidence"; out.mkdir(parents=True, exist_ok=True)
    conn = W.connect()
    try:
        for year in [int(y) for y in args.years.split(",")]:
            liq, days = build(year, conn)
            liq.to_csv(out / ("pit_%d.csv" % year), index=False)
            days.to_csv(out / ("pit_%d_days.csv.gz" % year), index=False)
            group = "explore_2024" if year == 2024 else "confirm_2021"
            keep = liq[liq.group == 0] if year == 2024 else liq[liq.group != 0]
            jobs = ROOT / "results" / "program_evidence_v1" / ("contig_" + group) / "jobs"
            jobs.mkdir(parents=True, exist_ok=True)
            names = []
            for permno in keep.permno:
                sub = days[days.permno == permno]
                if sub.empty:
                    continue
                (jobs / ("%d.txt" % permno)).write_text("".join("%s %s\n" % r for r in zip(sub.date, sub.ticker)))
                names.append(str(permno))
            (jobs.parent / "universe.txt").write_text("\n".join(names) + "\n")
            print(year, "universe", len(liq), "group names", len(names), "name-days", int(days.permno.isin(keep.permno).sum()),
                  "missing start ticker", int(liq.ticker_start.isna().sum()))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
