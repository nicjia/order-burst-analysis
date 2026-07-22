#!/usr/bin/env python3
"""
explore_vol_trade.py — the strongest, most robust signal in the panel is NON-directional:
burst intensity (n_bursts) forecasts next-day realized volatility (FM t~4.9). Direction
is ~a coin flip (reversal Sharpe ~0.45, = the standard ST_Rev style). So the honest
'better way to trade' is to trade VOLATILITY / DISPERSION, not direction.

This script asks the questions that decide whether the vol edge is real and monetizable:
  (1) INCREMENTAL? Does intensity forecast next-day |ret| AFTER controlling for the
      obvious predictors (today's |ret|, 20d realized vol)? A vol forecaster is only
      tradeable if it beats vol-persistence.
  (2) BEST PREDICTOR? intensity (n_bursts) vs turnover (buy+sell) vs |flow| vs today |ret|.
  (3) TRADEABLE SPREAD: cross-sectional long hi-intensity / short lo-intensity, measured
      in realized |ret| and realized variance -> the dispersion/variance-swap P&L the
      forecast powers, with a day-level t-stat.
  (4) STABILITY: year-by-year incremental t.
All day-level Fama-MacBeth + Newey-West.
"""
import math, os
import numpy as np, pandas as pd

SP = "/private/tmp/claude-502/-Users-nick-order-burst-analysis/6dc069bf-0ebb-4f0f-a5f4-147298fce374/scratchpad"
REPO = "/Users/nick/order-burst-analysis"


def nw(x, L=10):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; T = len(x)
    if T < 20: return (np.nan, np.nan, T)
    m = x.mean(); e = x - m; v = (e @ e) / T
    for l in range(1, L + 1):
        w = 1 - l / (L + 1); v += 2 * w * (e[l:] @ e[:-l]) / T
    return (m, m / np.sqrt(v / T), T)


def zrows(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-9, axis=0).clip(-4, 4)


def load():
    d = pd.read_csv(SP + "/all_rows.csv", header=None,
                    names=["ticker", "date", "netflow", "n_bursts", "buy", "sell"],
                    dtype=str, on_bad_lines="skip")
    d = d[d["date"].str.fullmatch(r"\d{8}", na=False)]
    for c in ["date", "netflow", "n_bursts", "buy", "sell"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["date"] = d["date"].astype("Int64")
    piv = lambda v: d.pivot_table(index="date", columns="ticker", values=v)
    FL, CNT, BUY, SELL = piv("netflow"), piv("n_bursts"), piv("buy"), piv("sell")
    close = pd.read_csv(REPO + "/close_all.csv", index_col="date"); close.index = close.index.astype(int)
    dates = sorted(x for x in FL.index if 20170101 <= x <= 20211231)
    cols = [c for c in FL.columns if c in close.columns]
    FL, CNT, BUY, SELL = (X.reindex(index=dates, columns=cols) for X in (FL, CNT, BUY, SELL))
    R = close.reindex(dates)[cols].pct_change(fill_method=None)
    return dates, cols, FL, CNT, BUY, SELL, R


def fm(y, Xs, dates):
    """Fama-MacBeth: cross-sectional multivariate reg each day, NW on the slope series.
       Xs: dict name->DataFrame. Returns dict name->(mean_slope, t)."""
    names = list(Xs)
    slopes = {nm: [] for nm in names}
    for dd in dates:
        cols = pd.DataFrame({"y": y.loc[dd], **{nm: Xs[nm].loc[dd] for nm in names}}).dropna()
        if len(cols) < 20: continue
        X = np.column_stack([np.ones(len(cols))] + [cols[nm].values for nm in names])
        b, *_ = np.linalg.lstsq(X, cols["y"].values, rcond=None)
        for k, nm in enumerate(names): slopes[nm].append(b[k + 1])
    return {nm: nw(slopes[nm]) for nm in names}, slopes


def main():
    dates, cols, FL, CNT, BUY, SELL, R = load()
    print("panel %d names x %d dates\n" % (len(cols), len(dates)))

    absr = R.abs()
    tgt = absr.shift(-1)                       # next-day |ret| = realized vol proxy
    rv20 = R.rolling(20, min_periods=10).std() # trailing realized vol
    cnt_z, tvol_z, absr_z = zrows(CNT), zrows(rv20), zrows(absr)
    turn_z = zrows(BUY + SELL)
    aflow_z = zrows(FL.abs())

    print("=== 1) INCREMENTAL vol forecast: next|ret| ~ intensity + controls (FM) ===")
    res, _ = fm(tgt, {"intensity": cnt_z, "today|ret|": absr_z, "trail_vol": tvol_z}, dates)
    for nm, (m, t, n) in res.items():
        print("  %-12s slope=%+.3f  t=%+.2f" % (nm, m, t))
    print("  -> if intensity t>3 with today|ret| & trail_vol in the model, it's a genuine")
    print("     INCREMENTAL vol forecaster (beats vol-persistence), i.e. monetizable.")

    print("\n=== 2) HORSE RACE: which single var forecasts next|ret| best? (univariate FM) ===")
    for nm, X in [("intensity(n_bursts)", cnt_z), ("turnover(buy+sell)", turn_z),
                  ("|netflow|", aflow_z), ("today|ret|", absr_z), ("trail_vol20", tvol_z)]:
        (m, t, n), _ = fm(tgt, {nm: X}, dates)[0][nm], None
        # fm returns dict; adapt:
    # redo cleanly:
    for nm, X in [("intensity(n_bursts)", cnt_z), ("turnover(buy+sell)", turn_z),
                  ("|netflow|", aflow_z), ("today|ret|", absr_z), ("trail_vol20", tvol_z)]:
        res, _ = fm(tgt, {nm: X}, dates)
        m, t, n = res[nm]
        print("  %-22s slope=%+.3f  t=%+.2f" % (nm, m, t))

    print("\n=== 3) TRADEABLE DISPERSION SPREAD: long hi-intensity / short lo-intensity ===")
    #  each day rank by intensity; long top quintile, short bottom quintile; 'return' =
    #  next-day realized |ret| (a delta-hedged straddle / variance-swap proxy).
    rank = CNT.rank(axis=1, pct=True)
    hi = rank >= 0.8; lo = rank <= 0.2
    spread = []
    for i, dd in enumerate(dates[:-1]):
        nd = dates[i + 1]
        h = absr.loc[nd][hi.loc[dd].reindex(absr.columns).fillna(False)].mean()
        l = absr.loc[nd][lo.loc[dd].reindex(absr.columns).fillna(False)].mean()
        if np.isfinite(h) and np.isfinite(l):
            spread.append((dd, h - l))
    sp = pd.Series(dict(spread))
    m, t, n = nw(sp.values)
    print("  next|ret| hi-intensity minus lo-intensity: %+.1f bps/day  t=%+.2f (days=%d)"
          % (m * 1e4, t, n))
    # variance version
    var = (R ** 2)
    tgtv = var.shift(-1)
    spv = []
    for i, dd in enumerate(dates[:-1]):
        nd = dates[i + 1]
        h = tgtv.loc[dd][hi.loc[dd].reindex(var.columns).fillna(False)].mean()
        l = tgtv.loc[dd][lo.loc[dd].reindex(var.columns).fillna(False)].mean()
        if np.isfinite(h) and np.isfinite(l): spv.append(h - l)
    mv, tv, nv = nw(spv)
    ann_ratio = (np.nanmean(spv) / (np.nanstd(spv) + 1e-12)) * math.sqrt(252)
    print("  next-day VARIANCE spread hi-lo: mean %+.2e t=%+.2f  (info-ratio %.2f)" % (mv, tv, ann_ratio))

    print("\n=== 4) YEAR-BY-YEAR incremental intensity t (controls: today|ret|, trail_vol) ===")
    for y in (2017, 2018, 2019, 2020, 2021):
        ds = [d for d in dates if y * 10000 <= d < (y + 1) * 10000]
        res, _ = fm(tgt, {"intensity": cnt_z, "abs": absr_z, "tv": tvol_z}, ds)
        m, t, n = res["intensity"]
        print("  %d: intensity slope=%+.3f t=%+.2f" % (y, m, t))


if __name__ == "__main__":
    main()
