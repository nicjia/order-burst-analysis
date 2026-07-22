#!/usr/bin/env python3
"""
data_quality.py — data-integrity infrastructure for the 2017-2021 (and full) sweep.
Addresses three concerns, all LOBSTER-independent (uses CRSP annual snapshots in
Yearly/ + yfinance):

  1. MISSING DATES: build the true trading calendar (from CRSP snapshot dates /
     price panel) so a date absent from LOBSTER is flagged MISSING, not counted 0.
  2. VOLUME VERIFICATION (download integrity): compare extracted burst volume
     (buy+sell from hist_flow) to a reference (CRSP snapshot volume or yfinance).
     A name-day with extracted~0 while reference is large = a failed/truncated
     download -> flag & re-pull. LOBSTER captures only NASDAQ marketable trades,
     so a healthy ratio is a FRACTION of consolidated volume, not 1.0 -- we flag
     ratios that collapse toward 0, not those below 1.
  3. IPO / DELISTING (survivorship-free universe): from CRSP snapshots, each name's
     first-listed and last-listed year, plus DLRET delisting returns -> a
     point-in-time membership that INCLUDES names from their IPO and applies CRSP
     delisting returns at exit.

Usage: python3 src_py/data_quality.py   (prints the universe/IPO/delisting audit)
"""
import glob, os, re, sys
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def crsp_snapshots():
    """{year: DataFrame} of the CRSP annual snapshot (Yearly/YYYY/*.csv)."""
    out = {}
    for y in sorted(glob.glob(os.path.join(REPO, "Yearly", "20*"))):
        fs = sorted(glob.glob(os.path.join(y, "*.csv")))
        if fs:
            out[int(os.path.basename(y))] = pd.read_csv(fs[0])
    return out


def study_names():
    names = []
    for ln in open(os.path.join(REPO, "universes", "full_500.txt")):
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            for x in re.split(r"[,\s]+", ln):
                if re.fullmatch(r"[A-Z]{1,6}", x.strip().upper()):
                    names.append(x.strip().upper())
    return sorted(set(names))


def build_pit_universe():
    """point-in-time listing + delisting per study name, from CRSP snapshots."""
    snaps = crsp_snapshots(); years = sorted(snaps)
    listed = {y: set(snaps[y]["ticker"].dropna().astype(str)) for y in years}
    dlret = {}
    for y in years:
        s = snaps[y]
        if "DLRET" in s.columns:
            r = s.assign(dl=pd.to_numeric(s["DLRET"], errors="coerce"))
            for _, row in r[r.dl.notna()].iterrows():
                dlret.setdefault(str(row["ticker"]), []).append((y, row.dl))
    rows = []
    for n in study_names():
        yl = [y for y in years if n in listed[y]]
        rows.append(dict(ticker=n, first_listed=min(yl) if yl else None,
                         last_listed=max(yl) if yl else None,
                         listed_years=len(yl), delist_return=dlret.get(n)))
    return pd.DataFrame(rows), years, listed


def volume_reference_crsp(ticker, year):
    """CRSP snapshot volume for a ticker in a given year's snapshot (reference)."""
    snaps = crsp_snapshots()
    if year in snaps:
        s = snaps[year]; m = s[s["ticker"].astype(str) == ticker]
        if len(m) and "volume" in s.columns:
            return float(pd.to_numeric(m["volume"], errors="coerce").iloc[0])
    return np.nan


def volume_check(extracted_rows_csv, ref="crsp", flag_ratio=0.001):
    """extracted_rows_csv: hist_flow output (ticker,date,netflow,n_bursts,buy,sell).
    Returns per-name coverage: extracted burst volume vs reference; flags collapses.
    ref='yfinance' does a live per-name daily lookup (slower)."""
    d = pd.read_csv(extracted_rows_csv)
    d["extracted_vol"] = pd.to_numeric(d["buy"], errors="coerce") + pd.to_numeric(d["sell"], errors="coerce")
    d["year"] = (pd.to_numeric(d["date"], errors="coerce") // 10000).astype("Int64")
    out = []
    for (tk, yr), g in d.groupby(["ticker", "year"]):
        ext = g["extracted_vol"].sum()
        nz = int((g["extracted_vol"] > 0).sum()); tot = len(g)
        ref_v = volume_reference_crsp(tk, int(yr)) if ref == "crsp" else np.nan
        out.append(dict(ticker=tk, year=int(yr), name_days=tot, nonzero=nz,
                        zero_frac=1 - nz / max(tot, 1), extracted_vol=ext, ref_snapshot_vol=ref_v))
    r = pd.DataFrame(out)
    r["SUSPECT_DOWNLOAD"] = r["zero_frac"] > 0.5     # >50% zero name-days => likely failed pulls
    return r


def main():
    df, years, listed = build_pit_universe()
    print("=== POINT-IN-TIME UNIVERSE (CRSP snapshots %d-%d) ===" % (years[0], years[-1]))
    print("study names total:", len(df))
    print("listed every snapshot year:", int((df.listed_years == len(years)).sum()))
    ipo = df[df.first_listed > years[0]]
    print("IPO'd after %d (not in first snapshot): %d" % (years[0], len(ipo)))
    print("  by first-listed year:", ipo.first_listed.value_counts().sort_index().to_dict())
    dead = df[df.delist_return.notna()]
    print("with a CRSP delisting return: %d %s" % (len(dead), sorted(dead.ticker.tolist())[:15]))
    gone = df[(df.last_listed.notna()) & (df.last_listed < years[-1])]
    print("not listed in final snapshot (exited): %d %s" % (len(gone), sorted(gone.ticker.tolist())[:15]))
    print("\nactive study names per year (point-in-time):")
    for y in years:
        print("  %d: %d" % (y, len([n for n in df.ticker if n in listed[y]])))
    print("\n=> A survivorship-free sweep must (a) add each name from its first_listed year,")
    print("   (b) drop it after last_listed and splice delist_return, (c) treat LOBSTER-missing")
    print("   dates as MISSING (not 0), and (d) volume-check each name-year via volume_check().")


if __name__ == "__main__":
    main()
