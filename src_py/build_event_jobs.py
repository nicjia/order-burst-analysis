#!/usr/bin/env python3
"""Job files for module D4: packet extraction windows around S&P 500 and Nasdaq-100 changes.

Window: trading days E-45 .. E+10 around each effective date E (CRSP calendar), under the ticker CRSP
assigns on each day. Name-days already requested by the contiguous panels are skipped; the event
analysis reads packets from both places.
"""
from pathlib import Path

import pandas as pd

import wrds_access as W

ROOT = Path(__file__).resolve().parents[1]
PRE, POST = 45, 10


def main():
    ev = pd.read_csv(ROOT / "data" / "wrds" / "index_events_2021_2024.csv")
    cal = pd.read_csv(ROOT / "data" / "wrds" / "crsp_dsi.csv.gz").date.astype(str).str.replace("-", "").tolist()
    cal = sorted(cal)
    perms = tuple(int(p) for p in ev.permno.unique())
    names = W.query("""select permno, namedt, nameendt, ticker from crsp.dsenames
                       where permno in %s and nameendt >= '2020-06-01'""", (perms,))
    names.to_csv(ROOT / "data" / "wrds" / "crsp_names_events.csv.gz", index=False)
    names["namedt"] = pd.to_datetime(names.namedt).dt.strftime("%Y%m%d")
    names["nameendt"] = pd.to_datetime(names.nameendt.fillna("2099-12-31")).dt.strftime("%Y%m%d")
    have = set()
    for g in ("contig_explore_2024", "contig_confirm_2021"):
        for f in (ROOT / "results" / "program_evidence_v1" / g / "jobs").glob("*.txt"):
            for line in f.read_text().split("\n"):
                if line.strip():
                    have.add((int(f.stem), line.split()[0]))
    out = ROOT / "results" / "program_evidence_v1" / "events"
    (out / "jobs").mkdir(parents=True, exist_ok=True)
    wanted = {}
    for r in ev.itertuples():
        e = r.date.replace("-", "")
        pos = next(i for i, d in enumerate(cal) if d >= e)
        for d in cal[max(0, pos - PRE): pos + POST + 1]:
            if (r.permno, d) in have:
                continue
            m = names[(names.permno == r.permno) & (names.namedt <= d) & (names.nameendt >= d)]
            if len(m) and isinstance(m.ticker.iloc[0], str):
                wanted.setdefault(int(r.permno), {})[d] = m.ticker.iloc[0]
    ids = []
    for permno, days in sorted(wanted.items()):
        (out / "jobs" / ("%d.txt" % permno)).write_text("".join("%s %s\n" % (d, t) for d, t in sorted(days.items())))
        ids.append(str(permno))
    (out / "universe.txt").write_text("\n".join(ids) + "\n")
    print("event names", len(ids), "name-days", sum(len(v) for v in wanted.values()))


if __name__ == "__main__":
    main()
