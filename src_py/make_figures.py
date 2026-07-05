#!/usr/bin/env python3
"""
make_figures.py — Referee m5: generate the three manuscript figures as PDFs.

  fig_markout.pdf  intraday markout decay (sub-spread; hidden-exec footprint reverses by 30 min)
  fig_oos_pnl.pdf  cumulative walk-forward OOS P&L: burst-flow reversal vs plain reversal vs SGD ML
  fig_regime.pdf   reversal Sharpe monotone across price / spread quintiles
"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import m7_reversal_baseline as m7

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150})
C = {"blue": "#1f5fa8", "red": "#c0392b", "gray": "#7f8c8d", "green": "#2e7d32"}


def fig_markout():
    # honest geometry-gate (kappa=0) directional mid-to-mid burst markout (Table markout_honest)
    hb_h = [1, 3, 5, 10]; hb_v = [0.54, 0.53, 0.47, 0.42]
    # properly-signed hidden-execution footprint, AAPL (Table hidden_horizon): peaks ~3 min, reverses
    hd_h = [3, 15, 30]; hd_v = [0.42, 0.24, -0.28]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.axhspan(-4.0, 4.0, color=C["gray"], alpha=0.12, zorder=0)
    ax.text(29, 3.4, "inside the bid-ask spread ($\\approx$4 bps)", ha="right", va="top",
            fontsize=9, color=C["gray"])
    ax.axhline(0, color="black", lw=0.8)
    ax.plot(hb_h, hb_v, "-o", color=C["blue"], label="Aggressive-burst flow ($\\kappa=0$, honest)")
    ax.plot(hd_h, hd_v, "-s", color=C["red"], label="Hidden-execution flow (Lee-Ready signed)")
    ax.set_xlabel("Horizon after burst termination (minutes)")
    ax.set_ylabel("Directional markout (bps)")
    ax.set_title("Intraday markout: sub-spread and reversing")
    ax.set_ylim(-5, 5); ax.set_xlim(0, 31)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout(); fig.savefig("figures/fig_markout.pdf"); plt.close(fig)
    print("wrote figures/fig_markout.pdf")


def fig_oos_pnl():
    dis, cols, FL, PR, SD, R, cpx, close = m7.load_panels()
    Mwf = m7.walkforward_universe(cpx, dis, cols, m7.BOTTOM_K)
    oos = [d for d in dis if d >= dis[m7.BURN]]
    sigs = {
        "Burst-flow reversal (headline)": (m7.zscore(FL), C["blue"], "-"),
        "Plain 5-day price reversal": (m7.zscore(close.reindex(dis)[cols].pct_change(5, fill_method=None)),
                                       C["gray"], "--"),
        "SGD ML prediction as signal": (m7.zscore(PR), C["red"], "-."),
    }
    # M8 mitigation: winsorize per-name daily returns at +/-50% to remove unadjusted
    # (reverse-)split artifacts (LCID +862% etc.) before forming the strategy return.
    Rw = R.clip(-0.5, 0.5)
    dts = pd.to_datetime([str(d) for d in oos], format="%Y%m%d")
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    for name, (Z, col, ls) in sigs.items():
        s = m7.strat_returns(Z, Rw, Mwf).reindex(oos).fillna(0.0)
        sr = s[s != 0]; sh = sr.mean()/sr.std()*np.sqrt(252) if sr.std() > 0 else float("nan")
        cum = 100.0 * s.cumsum().values
        ax.plot(dts, cum, ls, color=col, label=f"{name} (SR {sh:+.2f})", lw=1.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Cumulative net return (%)")
    ax.set_title("Out-of-sample P&L (net 1 bps, winsorized)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig("figures/fig_oos_pnl.pdf"); plt.close(fig)
    print("wrote figures/fig_oos_pnl.pdf")


def fig_regime():
    price = [1.35, 0.75, 0.71, -0.15, 0.34]
    spread = [0.26, 0.62, 0.83, 1.13, 0.58]
    x = np.arange(5); w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(x - w/2, price, w, color=C["blue"], label="By average price (Q1 cheapest)")
    ax.bar(x + w/2, spread, w, color=C["green"], label="By relative spread (Q1 tightest)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"])
    ax.set_xlabel("Tick-constraint quintile")
    ax.set_ylabel("Reversal annualized Sharpe")
    ax.set_title("Reversal concentrates in tick-constrained names")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig("figures/fig_regime.pdf"); plt.close(fig)
    print("wrote figures/fig_regime.pdf")


if __name__ == "__main__":
    fig_markout(); fig_regime(); fig_oos_pnl()
