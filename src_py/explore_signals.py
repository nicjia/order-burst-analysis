#!/usr/bin/env python3
"""
explore_signals.py — broad but honest search for a stronger tradeable signal in the
2017-2021 burst panel. Everything is day-level P&L with Newey-West t-stats and a 1bp
per-side cost; no pseudo-replication. We test a MENU of variants and print them all
(so multiple testing is visible), flagging which clear the Harvey-Liu-Zhu t>=3 bar.

Panel columns: netflow (= buy - sell), n_bursts, buy, sell. Derived:
  imbalance = netflow / (buy+sell)         scale-free order imbalance (cross-name comparable)
  turnover  = buy + sell                    burst participation
  flow_z    = cross-sectional z of netflow  (raw, size-tilted)
  imb_z     = cross-sectional z of imbalance
  cnt_z     = cross-sectional z of n_bursts (intensity/conviction)
  tsz(x)    = per-name time-series z (own-history surprise), 60d
"""
import glob, math, os, sys
import numpy as np, pandas as pd

SP = "/private/tmp/claude-502/-Users-nick-order-burst-analysis/6dc069bf-0ebb-4f0f-a5f4-147298fce374/scratchpad"
REPO = "/Users/nick/order-burst-analysis"
COST = 1.0  # bps per side


def nw(x, L=10):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; T = len(x)
    if T < 20: return (np.nan, np.nan, T)
    m = x.mean(); e = x - m; v = (e @ e) / T
    for l in range(1, L + 1):
        w = 1 - l / (L + 1); v += 2 * w * (e[l:] @ e[:-l]) / T
    return (m, m / np.sqrt(v / T), T)


def sharpe_nw(ret):
    r = np.asarray(ret, float); r = r[np.isfinite(r)]
    if len(r) < 40: return (np.nan, np.nan, len(r))
    ann = r.mean() / (r.std() + 1e-12) * math.sqrt(252)
    _, t, n = nw(r)
    return (ann, t, n)


def zrows(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-9, axis=0).clip(-4, 4)


def tsz(df, win=60):
    m = df.rolling(win, min_periods=20).mean()
    s = df.rolling(win, min_periods=20).std()
    return ((df - m) / (s + 1e-9)).clip(-4, 4)


def load():
    d = pd.read_csv(SP + "/all_rows.csv", header=None,
                    names=["ticker", "date", "netflow", "n_bursts", "buy", "sell"],
                    dtype=str, on_bad_lines="skip")
    d = d[d["date"].str.fullmatch(r"\d{8}", na=False)]
    for c in ["date", "netflow", "n_bursts", "buy", "sell"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["date"] = d["date"].astype("Int64")
    FL = d.pivot_table(index="date", columns="ticker", values="netflow")
    CNT = d.pivot_table(index="date", columns="ticker", values="n_bursts")
    BUY = d.pivot_table(index="date", columns="ticker", values="buy")
    SELL = d.pivot_table(index="date", columns="ticker", values="sell")
    close = pd.read_csv(REPO + "/close_all.csv", index_col="date"); close.index = close.index.astype(int)
    dates = sorted(x for x in FL.index if 20170101 <= x <= 20211231)
    cols = [c for c in FL.columns if c in close.columns]
    FL, CNT, BUY, SELL = (X.reindex(index=dates, columns=cols) for X in (FL, CNT, BUY, SELL))
    R = close.reindex(dates)[cols].pct_change(fill_method=None)
    cpx = close.reindex(dates)[cols]
    return dates, cols, FL, CNT, BUY, SELL, R, cpx


def book(sig, R, H=1, direction=-1, cheap=None, cpx=None, K=None, scale=None):
    """Generic dollar-neutral backtest.
       sig      : signal DataFrame (higher = more buy pressure)
       direction: -1 reversal (fade), +1 continuation
       H        : holding/averaging window (days)
       K        : if set, restrict to the K cheapest names each day (tick-constrained)
       scale    : optional per-name multiplicative weight (e.g. 1/vol) applied before norm
    """
    W = direction * sig
    if scale is not None:
        W = W * scale
    if K is not None and cpx is not None:
        rank = cpx.rank(axis=1)  # 1 = cheapest
        W = W.where(rank <= K)
    if H > 1:
        W = W.rolling(H, min_periods=1).mean()
    W = W.shift(1)  # form on t, earn t+1
    W = W.sub(W.mean(axis=1), axis=0)
    g = W.abs().sum(axis=1).replace(0, np.nan)
    W = W.div(g, axis=0).fillna(0.0)
    turn = (W - W.shift(1)).abs().sum(axis=1)
    ret = (W * R).sum(axis=1) - (COST / 1e4) * turn
    return sharpe_nw(ret), ret


def line(tag, res):
    (s, t, n), _ = res
    star = "  <== t>=3" if (np.isfinite(t) and abs(t) >= 3) else ("  * t>=2" if (np.isfinite(t) and abs(t) >= 2) else "")
    print("  %-52s Sharpe %+5.2f  t=%+5.2f  (n=%d)%s" % (tag, s, t, n, star))
    return (s, t)


def main():
    dates, cols, FL, CNT, BUY, SELL, R, cpx = load()
    TOT = BUY + SELL
    imbalance = FL / TOT.replace(0, np.nan)
    flow_z = zrows(FL)
    imb_z = zrows(imbalance)
    cnt_z = zrows(CNT)
    imb_ts = tsz(imbalance)
    print("panel: %d names x %d dates, %d real name-days\n" %
          (len(cols), len(dates), int((FL.notna() & (FL != 0)).sum().sum())))

    print("=== 1) REVERSAL: signal choice (H=5, full universe) ===")
    line("fade raw flow_z", book(flow_z, R, H=5))
    line("fade imbalance_z (scale-free)", book(imb_z, R, H=5))
    line("fade imbalance_ts (own-history surprise)", book(imb_ts, R, H=5))
    line("fade sign(flow)", book(np.sign(FL), R, H=5))

    print("\n=== 2) HOLDING-PERIOD sweep (fade imbalance_z, full universe) ===")
    for H in (1, 2, 3, 5, 10, 20):
        line("H=%2d" % H, book(imb_z, R, H=H))

    print("\n=== 3) DIRECTION at H=1 (continuation vs reversal, imbalance_z) ===")
    line("H=1 reversal (fade)", book(imb_z, R, H=1, direction=-1))
    line("H=1 continuation (follow)", book(imb_z, R, H=1, direction=+1))

    print("\n=== 4) TICK-CONSTRAINED (cheap names) vs full (fade imbalance_z, H=5) ===")
    for K in (30, 50, 100, 200):
        line("K=%3d cheapest" % K, book(imb_z, R, H=5, K=K, cpx=cpx))
    line("full universe", book(imb_z, R, H=5))

    print("\n=== 5) CONVICTION-WEIGHTED (fade imbalance_z * intensity, H=5) ===")
    conv = cnt_z.clip(lower=0)  # only names with above-median burst intensity get weight
    line("fade imb_z, weight by intensity+", book(imb_z, R, H=5, scale=conv))
    hi = (cnt_z > 0.5).astype(float).replace(0, np.nan)
    line("fade imb_z on high-intensity subset only", book(imb_z, R, H=5, scale=hi))

    print("\n=== 6) VOL-MANAGED reversal (Moreira-Muir: gross ~ 1/recent realized vol) ===")
    rv = R.rolling(20, min_periods=10).std()
    inv = (1.0 / (rv + 1e-6))
    line("fade imb_z, per-name 1/vol scale, H=5", book(imb_z, R, H=5, scale=inv))
    # time-series vol-managed on the base reversal p&l
    (_, _, _), base = book(imb_z, R, H=5)
    tvol = pd.Series(base, index=dates).rolling(20, min_periods=10).std().shift(1)
    vm = pd.Series(base, index=dates) * (base.std() if hasattr(base, "std") else np.nanstd(base)) / (tvol + 1e-9)
    print("  %-52s Sharpe %+5.2f  t=%+5.2f" % ("time-series vol-managed base reversal",
          *(lambda a, b, c: (a, b))(*sharpe_nw(vm.values))))

    print("\n=== 7) CAMPAIGN long-only reversion (>=3 same-sign, hold 5d) ===")
    idx = list(FL.index)
    for want, lab in [(+1, "BUY-campaign  (short after)"), (-1, "SELL-campaign (long after)")]:
        pnl = {}
        for name in cols:
            s = np.sign(FL[name].values); r = R[name].values; n = len(s); i = 0
            while i < n:
                if not np.isfinite(s[i]) or s[i] == 0: i += 1; continue
                j = i
                while j + 1 < n and s[j + 1] == s[i]: j += 1
                if j - i + 1 >= 3 and s[i] == want:
                    for tt in range(j + 1, min(j + 6, n)):
                        if np.isfinite(r[tt]):
                            pnl.setdefault(idx[tt], []).append(-want * r[tt])  # fade the campaign
                i = j + 1
        ser = pd.Series({d: np.mean(v) for d, v in pnl.items()}).sort_index()
        s, t, nn = sharpe_nw(ser.values)
        print("  %-52s Sharpe %+5.2f  t=%+5.2f (days=%d)" % ("fade " + lab, s, t, nn))

    print("\n=== 8) COMBO: blended cross-sectional score (fade imb_z + intensity tilt), H=5 ===")
    # overreaction hypothesis: fade imbalance more where intensity (overreaction) is high
    combo = imb_z * (1 + 0.5 * cnt_z.clip(-2, 2))
    line("fade [imb_z * (1+0.5 cnt_z)]", book(combo, R, H=5))

    print("\nNOTE: menu search => multiple testing. Treat t>=3 (HLZ) as the bar for a")
    print("genuine tradeable claim; 2<=t<3 is suggestive; report the whole menu, not the max.")


if __name__ == "__main__":
    main()
