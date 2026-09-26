#!/usr/bin/env python3
"""P4 revisit v1: retry list for name-days whose archive file is filed under a later ticker of the same company.

The lobster2 archive names some historical files by a company's current ticker (META.7z on 2022-01-03, when CRSP's
ticker was FB). For each name-day recorded as missing, the candidates are the PERMNO's CRSP tickers whose
validity starts after the date, most recent change first, excluding any ticker that CRSP assigns to a different
security on that date. Output: one "date candidate permno" line per candidate (the shard skips a name-day once one
candidate succeeds, because retries run in candidate order within one shard).
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", required=True, help="concatenated status lines: date ticker permno status")
    ap.add_argument("--tickers", default="data/p4/ticker_hist_all.csv.gz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    st = pd.read_csv(args.status, sep=" ", header=None, names=["date", "ticker", "permno", "status"], dtype={"date": str})
    done = set(map(tuple, st[st.status == "ok"][["date", "permno"]].to_numpy()))
    miss = st[st.status == "missing"].drop_duplicates(["date", "permno"])
    miss = miss[[(d, p) not in done for d, p in zip(miss.date, miss.permno)]]
    th = pd.read_csv(args.tickers)
    th["start"] = pd.to_datetime(th.secinfostartdt); th["end"] = pd.to_datetime(th.secinfoenddt)
    by_permno = {p: g for p, g in th.groupby("permno")}
    by_ticker = {t: g for t, g in th.groupby("ticker")}
    lines = []
    for r in miss.itertuples(index=False):
        g = by_permno.get(int(r.permno))
        if g is None:
            continue
        day = pd.Timestamp(r.date)
        later = g[(g.start > day) & (g.ticker != r.ticker)].sort_values("start", ascending=False)
        seen = set()
        for t in later.ticker:
            if t in seen:
                continue
            seen.add(t)
            holders = by_ticker.get(t)
            other = holders[(holders.permno != r.permno) & (holders.start <= day) & (holders.end >= day)]
            if len(other):
                continue
            lines.append("%s %s %d" % (r.date, t, int(r.permno)))
    Path(args.out).write_text("".join(l + "\n" for l in lines))
    print("missing name-days", len(miss), "candidate lines", len(lines),
          "name-days with candidates", len({(l.split()[0], l.split()[2]) for l in lines}))


if __name__ == "__main__":
    main()
