#!/usr/bin/env python3
"""Figures for burst-defs-raw-v4: the term structure of forecasting power (IC by horizon, real bursts vs matched
random-time events; and the IC the burst features add over controls + book). Reads v4_term_structure.csv."""
import sys as _sys, pathlib as _pl  # shared project modules live in src_py/
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "src_py"))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import burst_defs_raw4_model as V4

LAB = ["1 s", "2 s", "5 s", "10 s", "30 s", "1 min", "2 min", "5 min", "10 min", "30 min*", "60 min*", "close*",
       "next open*", "next close*"]
SHOW = [("early5 real", "act at the 5th child", "#1f4e79", "-"), ("run0.01 real", "run, 10 ms gap", "#2e75b6", "-"),
        ("levelclear real", "level-clearing run", "#70ad47", "-"), ("cancel real", "cancellation burst", "#c55a11", "-"),
        ("hidden real", "hidden-heavy run", "#7030a0", "-"), ("run60 real", "run, 60 s gap", "#7f7f7f", "-"),
        ("run0.1 real 10:00-15:30", "run 0.1 s, 10:00-15:30", "#000000", "-"),
        ("run0.1 pseudo 10:00-15:30", "random time, same duration and side", "#000000", "--")]


def main():
    ts = pd.read_csv(V4.D4 / "v4_term_structure.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    for j, tn in enumerate(V4.TESTS):
        x = ts[ts.test == tn]
        for i, (val, title) in enumerate((("ic", "rank IC of the forecast"), ("burst_gain", "IC added by burst features over controls + book"))):
            ax = axes[i, j]
            for g, lab, col, ls in SHOW:
                y = x[x.group == g].set_index("horizon")[val].reindex(V4.HS)
                if y.notna().any():
                    ax.plot(range(len(V4.HS)), y.to_numpy(), ls, color=col, marker="o", ms=3, lw=1.4, label=lab)
            ax.axhline(0, color="#999", lw=0.8)
            ax.set_title("%s — %s" % (title, tn), fontsize=10)
            ax.grid(alpha=0.25)
    for ax in axes[1]:
        ax.set_xticks(range(len(V4.HS))); ax.set_xticklabels(LAB, rotation=45, ha="right", fontsize=8)
    axes[0, 0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Forecasting the signed mid move after a burst (decision = the moment the burst is known; * market-excess)\n"
                 "trained on 56 stocks 2022-23; tested on 56 different stocks in 2024 and on all 96 stocks in 2025", fontsize=10.5)
    fig.tight_layout()
    out = V4.D4 / "fig_term_structure.png"
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
