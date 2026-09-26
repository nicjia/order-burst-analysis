#!/usr/bin/env python3
"""Figures for program-evidence-v1 (PNG for review, PDF for the manuscript)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from report_fingerprint import GRID, INK, INK2, SERIES, SURFACE, style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "program_evidence_v1"
YEARS = (("explore_2024", "2024", "-"), ("confirm_2021", "2021", "--"))


def save(fig, out):
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    plt.close(fig)


def legend(ax, **kw):
    leg = ax.legend(frameon=False, fontsize=8.5, **kw)
    for text in leg.get_texts():
        text.set_color(INK2)


def timing(out):
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=160); fig.patch.set_facecolor(SURFACE); style(ax)
    deltas = [1, 2, 5, 10, 20, 50]
    for lab, year, ls in YEARS:
        r = json.loads((RES / ("abi_%s.json" % lab)).read_text())["A"]
        for k, (key, name) in enumerate((("a1", "same day vs other day"), ("a2", "identical vs different size"))):
            y = [r["delta_%dms" % d][key]["median"] for d in deltas]
            lo = [r["delta_%dms" % d][key]["ci95"][0] for d in deltas]; hi = [r["delta_%dms" % d][key]["ci95"][1] for d in deltas]
            ax.fill_between(deltas, lo, hi, color=SERIES[k], alpha=0.10, linewidth=0)
            ax.plot(deltas, y, ls, color=SERIES[k], linewidth=2, label="%s, %s" % (name, year))
            ax.plot(deltas, y, "o", color=SERIES[k], markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.axhline(1.0, color=INK2, linewidth=1)
    ax.set_xscale("log"); ax.set_xticks(deltas); ax.set_xticklabels([str(d) for d in deltas])
    ax.set_xlabel("window around a whole-second lag (± ms)"); ax.set_ylabel("phase concentration ratio")
    ax.set_title("Timer fingerprint: same-side lags lock to whole seconds", color=INK, fontsize=11, loc="left")
    legend(ax, loc="upper right")
    save(fig, out)


def sides(out):
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=160); fig.patch.set_facecolor(SURFACE); style(ax)
    for lab, year, ls in YEARS:
        r = json.loads((RES / ("abi_%s.json" % lab)).read_text())["B"]
        edges = np.array(r["lag_edges"]); mids = np.sqrt(np.maximum(edges[:-1], 0.25) * edges[1:])
        for k, (rel, name) in enumerate((("same_side", "same side"), ("opposite_side", "opposite side"))):
            cur = r["u_nonround"]["ratio_curves"][rel]
            y = np.array([c["median"] if c["median"] is not None else np.nan for c in cur])
            ax.plot(mids, y, ls, color=SERIES[k], linewidth=2, label="%s, %s" % (name, year))
            ax.plot(mids, y, "o", color=SERIES[k], markersize=4, markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.axhline(1.0, color=INK2, linewidth=1)
    ax.set_xscale("log")
    ticks = [1, 10, 60, 600, 3600, 14400]
    ax.set_xticks(ticks); ax.set_xticklabels(["1s", "10s", "1m", "10m", "1h", "4h"])
    ax.set_xlabel("lag between the two packets"); ax.set_ylabel("identical-size matches / chance")
    ax.set_title("Same-size flow is directional and persists for hours", color=INK, fontsize=11, loc="left")
    legend(ax, loc="upper right")
    save(fig, out)


def markouts(out):
    fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=160); fig.patch.set_facecolor(SURFACE); style(ax)
    groups = (("program", "program-like bursts"), ("bottom", "least program-like bursts"), ("no_burst", "outside bursts"))
    for lab, year, ls in YEARS:
        r = json.loads((RES / ("stage2_%s.json" % lab)).read_text())["F"]["mean_markout_bps_by_group"]
        hz = [1, 10, 60, 300]
        for k, (g, name) in enumerate(groups):
            y = [r["%ds" % h][g] for h in hz]
            ax.plot(hz, y, ls, color=SERIES[k], linewidth=2, label="%s, %s" % (name, year))
            ax.plot(hz, y, "o", color=SERIES[k], markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.axhline(0.0, color=INK2, linewidth=1)
    ax.set_xscale("log"); ax.set_xticks([1, 10, 60, 300]); ax.set_xticklabels(["1s", "10s", "1m", "5m"])
    ax.set_xlabel("horizon after the execution"); ax.set_ylabel("liquidity-provider markout (bps)")
    ax.set_title("Who pays: providers earn against program-like flow", color=INK, fontsize=11, loc="left")
    legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    save(fig, out)


def sync(out):
    fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=160); fig.patch.set_facecolor(SURFACE); style(ax)
    keys = (("same_side_program_i", "same side, program-like"), ("same_side_other_i", "same side, other"),
            ("opposite_side", "opposite side"))
    deltas = [0.1, 1, 10]
    for lab, year, ls in YEARS:
        r = json.loads((RES / ("stage2_%s.json" % lab)).read_text())["G"]
        for k, (key, name) in enumerate(keys):
            y = [r["sync_ratio_%gms" % d][key] for d in deltas]
            ax.plot(deltas, y, ls, color=SERIES[k], linewidth=2, label="%s, %s" % (name, year))
            ax.plot(deltas, y, "o", color=SERIES[k], markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.axhline(1.0, color=INK2, linewidth=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(deltas); ax.set_xticklabels(["0.1", "1", "10"])
    ax.set_xlabel("coincidence window across stocks (± ms)"); ax.set_ylabel("coincidences / shifted-time baseline")
    ax.set_title("Basket execution: packets in different stocks coincide", color=INK, fontsize=11, loc="left")
    legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    save(fig, out)


def events(out):
    path = RES / "events_2021_2024.json"
    r = json.loads(path.read_text())
    import pandas as pd
    prof = pd.DataFrame(r["profile"])
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160); fig.patch.set_facecolor(SURFACE); style(ax)
    for k, (var, name) in enumerate((("PI", "program imbalance"), ("NPI", "other imbalance"))):
        for kind, ls in (("add", "-"), ("delete", "--")):
            g = prof[(prof["var"] == var) & (prof["kind"] == kind)].sort_values("k")
            g = g[(g.k >= -20) & (g.k <= 10)]
            ax.plot(g.k, g.z, ls, color=SERIES[k], linewidth=2, label="%s, %s" % (name, "additions" if kind == "add" else "deletions"))
    ax.axhline(0.0, color=INK2, linewidth=1); ax.axvline(0.0, color=GRID, linewidth=1.5)
    ax.axvspan(-5, -1, color=GRID, alpha=0.5, linewidth=0)
    ax.set_xlabel("trading days relative to the effective date"); ax.set_ylabel("z-score against own days E-40..E-11")
    ax.set_title("Index changes: signed flow around the effective date", color=INK, fontsize=11, loc="left")
    legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    save(fig, out)


def passive(out):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.8), dpi=160); fig.patch.set_facecolor(SURFACE)
    labels = (("program", "program-like"), ("middle", "middle"), ("bottom", "least program-like"), ("placebo", "random time"))
    width = 0.38
    for ax, (key, title) in zip(axes, (("markout_60s_per_fill", "markout per fill, 60 s (bps)"),
                                        ("pnl_60s_per_posting", "P&L per posting, 60 s (bps)"))):
        style(ax)
        for j, (grp, year) in enumerate((("explore_2024", "2024"), ("confirm_2021", "2021"))):
            r = json.loads((ROOT / "results" / "metaorder_v1" / ("m5_%s.json" % grp)).read_text())["by_label"]
            vals = [r[k][key] for k, _ in labels]
            x = np.arange(len(labels)) + (j - 0.5) * width
            ax.bar(x, vals, width=width * 0.95, color=SERIES[j], label=year)
        ax.axhline(0, color=INK2, linewidth=1)
        ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels([n for _, n in labels], rotation=20, ha="right", fontsize=8)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    legend(axes[1], loc="lower right")
    save(fig, out)


def trend(out):
    r = json.loads((RES / "years_tsp.json").read_text())["J1"]
    years = ["2016", "2019", "2021", "2024"]
    fig, ax = plt.subplots(figsize=(6.0, 3.8), dpi=160); fig.patch.set_facecolor(SURFACE); style(ax)
    for k, (key, name) in enumerate((("ratio_05_2", "0.5-2 s"), ("ratio_2_10", "2-10 s"))):
        y = [r[yy]["balanced"][key]["median"] for yy in years]
        lo = [r[yy]["balanced"][key]["ci95"][0] for yy in years]; hi = [r[yy]["balanced"][key]["ci95"][1] for yy in years]
        xs = [int(yy) for yy in years]
        ax.fill_between(xs, lo, hi, color=SERIES[k], alpha=0.12, linewidth=0)
        ax.plot(xs, y, "-o", color=SERIES[k], linewidth=2, markersize=5, markeredgecolor=SURFACE, label="identical-size repeats, %s" % name)
    ax.axhline(1.0, color=INK2, linewidth=1)
    ax.set_xticks([2016, 2019, 2021, 2024]); ax.set_ylabel("observed / chance (median name)")
    ax.set_title("The size fingerprint has weakened since 2016", color=INK, fontsize=11, loc="left")
    legend(ax, loc="upper right")
    save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / "report"))
    ap.add_argument("--figs", default="timing,sides,markouts,sync")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for name in args.figs.split(","):
        globals()[name](out / ("fig_program_%s" % name))
        print("wrote", name)


if __name__ == "__main__":
    main()
