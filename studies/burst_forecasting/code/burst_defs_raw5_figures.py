#!/usr/bin/env python3
"""Figures for burst-defs-raw-v5: IC by horizon for bursts vs single orders vs random times, for the mid and for the
far-side quote (which the refill of the side the event hit does not move), and the IC the event's own features add
over the burst-blind GENERIC + DEPTH model. Reads v5_placebo.csv."""
import sys as _sys, pathlib as _pl  # shared project modules live in src_py/
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "src_py"))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import burst_defs_raw5_model as V5

LAB = {"1": "1 s", "10": "10 s", "60": "1 min", "300": "5 min", "1800": "30 min*", "_close": "close*"}
SHOW = [(("early5", "real"), "burst, act at 5th order", "#1f4e79", "-"), (("run0.1", "real"), "burst, run 0.1 s", "#2e75b6", "-"),
        (("run0.5", "real"), "burst, run 0.5 s", "#70ad47", "-"), (("pair", "real"), "two orders", "#c55a11", "-"),
        (("iso_large", "single"), "one large isolated order", "#7030a0", "--"), (("iso", "single"), "one isolated order", "#7f7f7f", "--"),
        (("any", "single"), "any marketable order", "#bfbfbf", "--"), (("run0.1", "pseudo"), "random time", "#000000", ":")]


def main(test="2025 test names"):
    P = pd.read_csv(V5.D5 / "v5_placebo.csv")
    P = P[P.test == test]
    hs = list(LAB)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    for j, fam in enumerate(("mid", "far quote")):
        for i, (val, title) in enumerate((("ic", "rank IC, GENERIC + DEPTH + EVENT"), ("gain", "IC added by the event's own features"))):
            ax = axes[i, j]
            for (dn, kind), lab, col, ls in SHOW:
                y = P[(P.defn == dn) & (P.kind == kind) & (P.target == fam)].set_index("horizon")[val].reindex(hs)
                if y.notna().any():
                    ax.plot(range(len(hs)), y.to_numpy(), ls, color=col, marker="o", ms=3, lw=1.4, label=lab)
            ax.axhline(0, color="#999", lw=0.8); ax.grid(alpha=0.25)
            ax.set_title("%s — %s" % (title, fam), fontsize=10)
    for ax in axes[1]:
        ax.set_xticks(range(len(hs))); ax.set_xticklabels([LAB[h] for h in hs], fontsize=8)
    axes[0, 0].legend(fontsize=7.5)
    fig.suptitle("v5, %s (676-stock universe; trained on other stocks 2022-23; * market-excess)\n"
                 "far quote = the side the event did not trade against (unaffected by the refill of the side it hit)" % test, fontsize=10.5)
    fig.tight_layout()
    out = V5.D5 / "fig_v5_placebo.png"
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main(*_sys.argv[1:])
