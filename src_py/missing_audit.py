#!/usr/bin/env python3
"""
missing_audit.py — classify MISSING rows from the 2017-2021 sweep.

Two modes:
  --partial  : fast early-warning (no CRSP). MISSING%/real%/zero% by YEAR, plus
               MISSING% for a hardcoded set of continuously-traded NYSE names
               (should be ~0) and NASDAQ mega-caps (expected high pre-2022 archive
               gap). ALERTS if a continuous name spikes -> dropped connection.
  (default)  : full classification using CRSP annual snapshots (Yearly/), splitting
               every MISSING into:
                 pre-IPO         (date-year < first CRSP-listed year)
                 post-delist     (date-year > last CRSP-listed year)
                 ARCHIVE-GAP     (CRSP-listed that year but lobster lacked the .7z)
               and reports the archive coverage boundary (which names/years).

Input: a consolidated rows CSV (ticker,date,netflow,n_bursts,buy,sell) where failed
pulls are the literal token MISSING. Usage:
  python3 src_py/missing_audit.py [--partial] results/hist_flow/all_rows.csv
"""
import glob, os, sys
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONT_NYSE = ["F", "T", "BAC", "JPM", "XOM", "GE", "C", "WFC", "PFE", "KO", "JNJ",
             "PG", "CVX", "HD", "WMT", "DIS", "MCD", "IBM", "MMM", "CAT"]
NASDAQ_MEGA = ["AAPL", "AMZN", "NVDA", "TSLA", "MSFT", "GOOGL", "GOOG", "META",
               "INTC", "CSCO", "CMCSA", "PEP", "COST", "ADBE", "NFLX"]


def load(path):
    d = pd.read_csv(path, header=None, names=["ticker", "date", "netflow", "n_bursts", "buy", "sell"],
                    dtype=str, on_bad_lines="skip")
    d = d[d["date"].str.fullmatch(r"\d{8}", na=False)]
    d["year"] = d["date"].str[:4].astype(int)
    d["is_missing"] = d["netflow"] == "MISSING"
    d["is_real"] = (~d["is_missing"]) & (pd.to_numeric(d["netflow"], errors="coerce").fillna(0) != 0)
    d["is_zero"] = (~d["is_missing"]) & (~d["is_real"])
    return d


def partial(d):
    print("=== MISSING/REAL/ZERO by YEAR ===")
    g = d.groupby("year").agg(n=("ticker", "size"), miss=("is_missing", "sum"),
                              real=("is_real", "sum"), zero=("is_zero", "sum"))
    for y, r in g.iterrows():
        print("  %d: n=%-7d MISSING=%5.1f%%  REAL=%5.1f%%  ZERO=%4.1f%%" %
              (y, r.n, 100*r.miss/r.n, 100*r.real/r.n, 100*r.zero/r.n))
    print("\n=== continuously-traded NYSE names (expect ~0%% MISSING) ===")
    alert = []
    for tk in CONT_NYSE:
        sub = d[d.ticker == tk]
        if not len(sub): continue
        by = sub.groupby("year")["is_missing"].mean() * 100
        hot = by[by > 15]
        tag = "  <-- ALERT" if len(hot) else ""
        if len(hot): alert.append((tk, hot.round(0).to_dict()))
        print("  %-5s miss%% by yr: %s%s" % (tk, {int(y): round(v) for y, v in by.items()}, tag))
    print("\n=== NASDAQ mega-caps (high pre-2022 = expected ARCHIVE GAP, not an error) ===")
    for tk in NASDAQ_MEGA:
        sub = d[d.ticker == tk]
        if not len(sub): continue
        by = sub.groupby("year")["is_missing"].mean() * 100
        print("  %-5s miss%% by yr: %s" % (tk, {int(y): round(v) for y, v in by.items()}))
    if alert:
        print("\n*** %d continuously-traded name(s) show MISSING spikes -> investigate a dropped"
              " connection / lobster2 staging gap for those year(s):" % len(alert))
        for tk, h in alert: print("   ", tk, h)
    else:
        print("\nOK: no continuously-traded NYSE name shows a MISSING spike (no obvious staging failure).")


def crsp_listed_years():
    listed = {}
    for y in sorted(glob.glob(os.path.join(REPO, "Yearly", "20*"))):
        fs = sorted(glob.glob(os.path.join(y, "*.csv")))
        if fs:
            listed[int(os.path.basename(y))] = set(pd.read_csv(fs[0])["ticker"].dropna().astype(str))
    return listed


def full(d):
    listed = crsp_listed_years()
    yrs = sorted(listed)
    first = {}; last = {}
    for tk in d.ticker.unique():
        yl = [y for y in yrs if tk in listed[y]]
        if yl: first[tk], last[tk] = min(yl), max(yl)
    miss = d[d.is_missing].copy()
    def classify(row):
        tk, y = row.ticker, row.year
        if tk not in first: return "not-in-CRSP"          # never in CRSP snapshots (ticker-map issue)
        if y < first[tk]: return "pre-IPO"
        if y > last[tk]: return "post-delist"
        if y in listed and tk in listed[y]: return "ARCHIVE-GAP"   # CRSP-listed but lobster lacked file
        return "ARCHIVE-GAP"
    miss["cls"] = miss.apply(classify, axis=1)
    print("=== MISSING classification (CRSP-verified) ===")
    print(miss["cls"].value_counts().to_string())
    print("\n=== ARCHIVE-GAP rate by year (the lobster coverage boundary) ===")
    ag = miss[miss.cls == "ARCHIVE-GAP"]
    tot_by_yr = d.groupby("year").size()
    for y in sorted(tot_by_yr.index):
        n_ag = int((ag.year == y).sum())
        print("  %d: archive-gap name-days=%-7d (%.1f%% of all %d name-days that year)" %
              (y, n_ag, 100*n_ag/tot_by_yr[y], tot_by_yr[y]))
    print("\n=== names with the most ARCHIVE-GAP (CRSP-listed, lobster-missing) ===")
    print(ag.groupby("ticker").size().sort_values(ascending=False).head(20).to_string())
    print("\n=> Report boundary: NASDAQ names absent from lobster pre-2022 are ARCHIVE-GAP, not")
    print("   pre-IPO; pre-IPO/post-delist are legitimately excluded point-in-time.")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else "results/hist_flow/all_rows.csv"
    d = load(path)
    print("loaded %d rows | MISSING=%d REAL=%d ZERO=%d\n" %
          (len(d), d.is_missing.sum(), d.is_real.sum(), d.is_zero.sum()))
    partial(d)
    if "--partial" not in sys.argv:
        print(); full(d)


if __name__ == "__main__":
    main()
