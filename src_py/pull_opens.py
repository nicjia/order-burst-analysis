#!/usr/bin/env python3
"""Pull split/div-adjusted Open+Close for the 493 panel names (2017-2021) from
yfinance, cache to scratchpad as opens.parquet / closes.parquet. Batched + resumable."""
import os, time, sys
import numpy as np, pandas as pd, yfinance as yf

SP = "/private/tmp/claude-502/-Users-nick-order-burst-analysis/6dc069bf-0ebb-4f0f-a5f4-147298fce374/scratchpad"


def panel_tickers():
    d = pd.read_csv(SP + "/all_rows.csv", header=None,
                    names=["ticker", "date", "netflow", "n_bursts", "buy", "sell"],
                    dtype=str, on_bad_lines="skip")
    d = d[d["date"].str.fullmatch(r"\d{8}", na=False)]
    d["nf"] = pd.to_numeric(d["netflow"], errors="coerce")
    real = d[d["nf"].notna() & (d["nf"] != 0)]
    return sorted(real["ticker"].unique())


def main():
    tks = panel_tickers()
    print("panel names:", len(tks))
    op_path, cl_path = SP + "/opens.parquet", SP + "/closes.parquet"
    opens, closes = {}, {}
    if os.path.exists(op_path):
        opens = pd.read_parquet(op_path).to_dict("series")
        closes = pd.read_parquet(cl_path).to_dict("series")
        print("resuming; already have", len(opens))
    todo = [t for t in tks if t not in opens]
    B = 80
    for i in range(0, len(todo), B):
        batch = todo[i:i + B]
        try:
            df = yf.download(batch, start="2017-01-01", end="2021-12-31",
                             auto_adjust=True, progress=False, threads=True)
        except Exception as e:
            print("batch fail", i, e); time.sleep(5); continue
        if df.empty:
            print("empty batch", i); continue
        O = df["Open"] if isinstance(df.columns, pd.MultiIndex) else df[["Open"]]
        C = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
        for t in batch:
            if t in O.columns:
                s = O[t].dropna()
                if len(s) > 200:
                    opens[t] = s; closes[t] = C[t].reindex(s.index)
        print("batch %d/%d done, have %d" % (i // B + 1, (len(todo) + B - 1) // B, len(opens)))
        pd.DataFrame(opens).to_parquet(op_path)
        pd.DataFrame(closes).to_parquet(cl_path)
        time.sleep(1)
    print("FINAL: opens for %d names -> %s" % (len(opens), op_path))


if __name__ == "__main__":
    main()
