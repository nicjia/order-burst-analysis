#!/usr/bin/env python3
"""
fig_tugofwar.py — figures/fig_tugofwar.pdf: cumulative P&L of the three session legs
for all three panels. Left/middle: Samples A and B, where the overnight and intraday
legs run opposite. Right: Sample C, the paper's primary 2022-2026 aggressive panel,
where every leg is flat. The point of the exhibit is the contrast.
"""
import math, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = "/private/tmp/claude-502/-Users-nick-order-burst-analysis/6dc069bf-0ebb-4f0f-a5f4-147298fce374/scratchpad"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def z(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-9, axis=0).clip(-4, 4)


def ii(df):
    df = df.copy(); df.index = [int(pd.Timestamp(x).strftime("%Y%m%d")) for x in df.index]; return df


def book(sig, RET):
    W = sig.sub(sig.mean(axis=1), axis=0)
    g = W.abs().sum(axis=1).replace(0, np.nan)
    W = W.div(g, axis=0).fillna(0)
    return (W * RET).sum(axis=1)


def legs(FL, O, C):
    dates = sorted(set(FL.index) & set(O.index))
    cols = [c for c in FL.columns if c in O.columns]
    FL = FL.reindex(dates, columns=cols); O = O.reindex(dates, columns=cols); C = C.reindex(dates, columns=cols)
    ON = O.shift(-1) / C - 1.0
    ID = C.shift(-1) / O.shift(-1) - 1.0
    CC = C.shift(-1) / C - 1.0
    s = z(FL)
    out = pd.DataFrame({"on": book(s, ON), "id": book(s, ID), "cc": book(s, CC)}, index=dates).dropna()
    return out


def sample_a():
    d = pd.read_csv(SP + "/all_rows.csv", header=None,
                    names=["ticker", "date", "netflow", "n_bursts", "buy", "sell"], dtype=str, on_bad_lines="skip")
    d = d[d["date"].str.fullmatch(r"\d{8}", na=False)]
    for c in ["date", "netflow"]: d[c] = pd.to_numeric(d[c], errors="coerce")
    d["date"] = d["date"].astype("Int64")
    FL = d.pivot_table(index="date", columns="ticker", values="netflow")
    return legs(FL, ii(pd.read_parquet(SP + "/opens.parquet")), ii(pd.read_parquet(SP + "/closes.parquet")))


def sample_b():
    h = pd.read_csv(os.path.join(REPO, "results/research/hidden_xsec_daily.csv")); h["date"] = h["date"].astype(int)
    FL = h.assign(nf=h.buy - h.sell).pivot_table(index="date", columns="ticker", values="nf")
    return legs(FL, ii(pd.read_parquet(SP + "/opens24.parquet")), ii(pd.read_parquet(SP + "/closes24.parquet")))


def sample_c():
    d = pd.read_csv(SP + "/coi_panel_ungated_2026.csv"); d["Date"] = d["Date"].astype(int)
    FL = (d.pivot_table(index="Date", columns="Ticker", values="buy_vol")
          - d.pivot_table(index="Date", columns="Ticker", values="sell_vol"))
    return legs(FL, ii(pd.read_parquet(SP + "/opens26.parquet")), ii(pd.read_parquet(SP + "/closes26.parquet")))


def xdates(idx):
    return [pd.Timestamp(str(int(x))) for x in idx]


def main():
    panels = [("A: 2017--2021 aggressive", sample_a()),
              ("B: 2023--2024 hidden", sample_b()),
              ("C: 2022--2026 aggressive (primary)", sample_c())]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), sharey=True)
    for ax, (lab, df) in zip(axes, panels):
        x = xdates(df.index)
        ax.plot(x, df["on"].cumsum() * 100, lw=1.6, color="#1f4e9c", label="overnight (close$\\to$open)")
        ax.plot(x, df["id"].cumsum() * 100, lw=1.6, color="#0f8a5f", label="intraday (open$\\to$close)")
        ax.plot(x, df["cc"].cumsum() * 100, lw=1.4, color="#777777", ls="--", label="close-to-close")
        ax.axhline(0, color="black", lw=0.6, alpha=0.5)
        ax.set_title(lab.replace("--", "–"), fontsize=10)
        ax.tick_params(axis="x", labelsize=8, rotation=30)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("cumulative return (%), dollar-neutral", fontsize=9)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    out = os.path.join(REPO, "figures", "fig_tugofwar.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)
    for lab, df in panels:
        print("%-38s overnight %+6.1f%%  intraday %+6.1f%%  close-close %+6.1f%%  (n=%d)" %
              (lab, df["on"].sum() * 100, df["id"].sum() * 100, df["cc"].sum() * 100, len(df)))


if __name__ == "__main__":
    main()
