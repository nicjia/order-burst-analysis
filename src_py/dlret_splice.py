#!/usr/bin/env python3
"""
dlret_splice.py — CRSP delisting-return splicing with a with/without TOGGLE.

Goal: at each delisted name's delisting date, splice the terminal delisting return
(distress/bankruptcy drop) into the return panel, so the reversal is tested both
WITH and WITHOUT that terminal move -- proving the Sharpe isn't just capturing
extreme delisting gaps.

Delisting date: the last valid daily price in close_all.csv for a name whose CRSP
listing ends before the sample end (from data_quality.build_pit_universe()).
Delisting return: use the actual CRSP DLRET if a daily delisting file is supplied
(DLRET_FILE, columns ticker,date,dlret); otherwise fall back to the Shumway (1997)
convention (configurable), the referee-standard when exact DLRET is unavailable:
  NYSE/AMEX performance delist ~ -0.30 ; NASDAQ ~ -0.55 ; bankruptcy ~ -1.00.

Functions:
  build_delist_table(cols) -> DataFrame[ticker, delist_date, dlret]
  splice_returns(R, table, on=True) -> R' with the terminal return placed on the
      delist date (and NaN afterwards). on=False returns R unchanged (the toggle).
"""
import os, glob
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DLRET_FILE = os.path.join(REPO, "data_factors", "crsp_dlret_daily.csv")   # optional, if present
SHUMWAY = {"NYSE": -0.30, "AMEX": -0.30, "NASDAQ": -0.55, "DEFAULT": -0.55}


def _crsp_listed_years():
    out = {}
    for y in sorted(glob.glob(os.path.join(REPO, "Yearly", "20*"))):
        fs = sorted(glob.glob(os.path.join(y, "*.csv")))
        if fs:
            d = pd.read_csv(fs[0])
            ex = dict(zip(d["ticker"].astype(str), d.get("PRIMEXCH", pd.Series(["N"]*len(d)))))
            out[int(os.path.basename(y))] = (set(d["ticker"].dropna().astype(str)), ex)
    return out


def build_delist_table(cols, sample_end=20211231, default_conv=None):
    """names in `cols` whose CRSP listing ends before the sample end -> delist row."""
    ly = _crsp_listed_years(); yrs = sorted(ly)
    close = pd.read_csv(os.path.join(REPO, "close_all.csv"), index_col="date")
    close.index = close.index.astype(int)
    actual = None
    if os.path.exists(DLRET_FILE):
        actual = pd.read_csv(DLRET_FILE)
        actual = dict(zip(zip(actual.ticker.astype(str), actual.date.astype(int)),
                          pd.to_numeric(actual.dlret, errors="coerce")))
    exch = {}
    for y in yrs:
        for tk, ex in ly[y][1].items(): exch.setdefault(tk, ex)
    rows = []
    for tk in cols:
        listed = [y for y in yrs if tk in ly[y][0]]
        if not listed or max(listed) >= yrs[-1]:
            continue                                        # still listed at sample end -> not delisted
        if tk not in close.columns:
            continue
        s = close[tk].dropna(); s = s[s.index <= sample_end]
        if not len(s):
            continue
        dd = int(s.index[-1])                               # last traded date ~ delist date
        ex = "NASDAQ" if str(exch.get(tk, "")).upper().startswith(("Q", "NASDAQ")) else "NYSE"
        dl = (actual.get((tk, dd)) if actual else None)
        if dl is None or not np.isfinite(dl):
            dl = default_conv if default_conv is not None else SHUMWAY.get(ex, SHUMWAY["DEFAULT"])
        rows.append(dict(ticker=tk, delist_date=dd, exch=ex, dlret=float(dl),
                         source=("CRSP" if actual and (tk, dd) in actual else "Shumway")))
    return pd.DataFrame(rows)


def splice_returns(R, table, on=True):
    """place the delisting return on each name's delist date (and NaN after). The
    toggle: on=False -> unchanged R."""
    if not on or table is None or not len(table):
        return R
    R2 = R.copy()
    for _, r in table.iterrows():
        tk, dd, dl = r["ticker"], int(r["delist_date"]), r["dlret"]
        if tk not in R2.columns:
            continue
        idx = R2.index
        on_or_before = idx[idx <= dd]
        if len(on_or_before):
            R2.loc[on_or_before[-1], tk] = dl               # terminal delisting return
        R2.loc[idx[idx > dd], tk] = np.nan                  # gone afterwards
    return R2


if __name__ == "__main__":
    # demo: build the delist table for the extraction universe
    import glob as g
    cols = sorted({os.path.basename(f).split("_")[0] for f in
                   g.glob("/private/tmp/*/scratchpad/all_rows.csv")}) or None
    t = build_delist_table(pd.read_csv(os.path.join(REPO, "close_all.csv"), nrows=0).columns[1:])
    print("delisted names with a spliceable terminal return:", len(t))
    if len(t):
        print(t.head(20).to_string())
        print("\nby source:", t.source.value_counts().to_dict(),
              "| mean dlret:", round(t.dlret.mean(), 3))
