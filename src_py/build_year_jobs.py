#!/usr/bin/env python3
"""Job files for program-evidence-v1 module J (descriptive trend and Tick Size Pilot).

J1: fingerprint-v1 tickers on 10 adjacent date pairs per year (2013, 2016, 2019), calendar-matched
    to the 2024 fingerprint pairs, plus the complementary name groups for 2021 and 2024 (so every
    year covers the same 474 tickers).
J2: Tick Size Pilot names (TAQ master, 2016-10-31) present in lobster2; 10 adjacent pairs before
    (April-September 2016) and 10 during (November 2016-June 2017), at most two per month, chosen
    for the number of pilot names present on both days (a coverage choice made before any outcome).
"""
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "program_evidence_v1"
V1_2024 = [("0119", "0122"), ("0226", "0227"), ("0404", "0405"), ("0507", "0508"), ("0613", "0614"),
           ("0723", "0724"), ("0823", "0826"), ("0930", "1001"), ("1104", "1105"), ("1210", "1211")]


def split_group(ticker):
    """fingerprint-v1's rule exactly: first 8 hex digits of sha256("fingerprint-v1|" + ticker), mod 3."""
    return int(hashlib.sha256(("fingerprint-v1|" + ticker).encode()).hexdigest()[:8], 16) % 3


def lob_dates():
    dates = {}
    for f in ("lob_counts_1319.txt", "lob_counts_1617.txt"):
        for line in (ROOT / "data" / "evidence" / f).read_text().splitlines():
            d = line.split()[0]
            dates.setdefault(d[:4], []).append(d)
    return {y: sorted(v) for y, v in dates.items()}


def matched_pairs(year, cal):
    pairs = []
    for a, _b in V1_2024:
        target = "%d%s" % (year, a)
        i = next(k for k, d in enumerate(cal) if d >= target)
        pairs.append((cal[i], cal[i + 1]))
    return pairs


def write_group(name, tickers, pairs):
    g = OUT / name
    (g / "jobs").mkdir(parents=True, exist_ok=True)
    dates = [d for p in pairs for d in p]
    for tk in tickers:
        (g / "jobs" / (tk + ".txt")).write_text("".join("%s %s\n" % (d, tk) for d in dates))
    (g / "universe.txt").write_text("\n".join(tickers) + "\n")
    (g / "dates.txt").write_text("\n".join(dates) + "\n")
    (g / "pairs.txt").write_text("".join("%s %s\n" % p for p in pairs))
    print(name, "tickers", len(tickers), "dates", len(dates))


def main():
    names = (ROOT / "results" / "fingerprint_v1" / "universe_474.txt").read_text().split()
    cal = lob_dates()
    for y in ("2013", "2016", "2019"):
        write_group("years_%s" % y, names, matched_pairs(int(y), cal[y]))
    # complementary groups for 2021 and 2024 reuse fingerprint-v1 calendars
    for y, fp_group in (("2021", "confirm_2021"), ("2024", "explore_2024")):
        base = ROOT / "results" / "fingerprint_v1" / fp_group
        pairs = [tuple(x.split()) for x in (base / "pairs.txt").read_text().splitlines() if x.strip()]
        done = set((base / "universe.txt").read_text().split())
        write_group("years_%s" % y, [t for t in names if t not in done], pairs)
    # J2: Tick Size Pilot
    tsp = pd.read_csv(ROOT / "data" / "wrds" / "tsp_groups_20161031.csv.gz")
    tsp_names = set(tsp.sym_root)
    cov = {}
    for line in (ROOT / "data" / "evidence" / "lob_tickers_1617_all.txt").read_text().splitlines():
        parts = line.split()
        cov[parts[0]] = set(parts[1:]) & tsp_names
    days = sorted(cov)
    cands = [(days[i], days[i + 1], len(cov[days[i]] & cov[days[i + 1]])) for i in range(len(days) - 1)]

    def choose(lo, hi, n=10, per_month=2):
        c = sorted([x for x in cands if lo <= x[0] <= hi and x[1] <= hi], key=lambda x: -x[2])
        picked, used, months = [], set(), {}
        for a, b, k in c:
            m = a[:6]
            if a in used or b in used or months.get(m, 0) >= per_month:
                continue
            picked.append((a, b)); used |= {a, b}; months[m] = months.get(m, 0) + 1
            if len(picked) == n:
                break
        return sorted(picked)
    pre = choose("20160401", "20160930"); during = choose("20161101", "20170630")
    present = set().union(*[cov[d] for p in pre + during for d in p])
    write_group("tsp_pre", sorted(present), pre)
    write_group("tsp_during", sorted(present), during)
    tsp[tsp.sym_root.isin(present)].to_csv(OUT / "tsp_names_groups.csv", index=False)
    print("tsp names by group", tsp[tsp.sym_root.isin(present)].grp.value_counts().to_dict(),
          "min pair coverage pre", min(x[2] for x in cands if (x[0], x[1]) in pre),
          "during", min(x[2] for x in cands if (x[0], x[1]) in during))


if __name__ == "__main__":
    main()
