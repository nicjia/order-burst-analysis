#!/usr/bin/env python3
"""
pull_earnings.py — fetch earnings announcement dates for the 474-name hidden panel so the
permanent-footprint measurement can be re-run excluding information-event windows.

This addresses the news-clustering confound: aggressive hidden bursts plausibly cluster
around earnings, and price moves permanently because of the announcement, with the burst a
correlate rather than a cause. Writes ticker,earn_date for 2022-07..2025-06 (a margin
around the 2023-24 sample so windows at the edges are covered).
"""
import sys, time
import pandas as pd, yfinance as yf

OUT = "measurements/data/earnings_dates.csv"


def main():
    names = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    rows, miss = [], []
    for i, t in enumerate(names):
        try:
            e = yf.Ticker(t).get_earnings_dates(limit=60)
            if e is None or len(e) == 0:
                miss.append(t); continue
            idx = pd.to_datetime(e.index).tz_localize(None)
            for d in idx:
                if pd.Timestamp("2022-07-01") <= d <= pd.Timestamp("2025-06-30"):
                    rows.append((t, int(d.strftime("%Y%m%d"))))
        except Exception:
            miss.append(t)
        if (i + 1) % 25 == 0:
            print("  %d/%d  rows=%d  miss=%d" % (i + 1, len(names), len(rows), len(miss)),
                  flush=True)
        time.sleep(0.15)
    df = pd.DataFrame(rows, columns=["ticker", "earn_date"]).drop_duplicates()
    df.to_csv(OUT, index=False)
    print("wrote %s: %d dates, %d names, %d missing"
          % (OUT, len(df), df.ticker.nunique(), len(miss)))
    if miss:
        print("missing:", ",".join(miss[:40]))


if __name__ == "__main__":
    main()
