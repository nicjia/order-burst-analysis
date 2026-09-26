#!/usr/bin/env python3
"""Figures and markdown tables for fingerprint-v1 summaries (exploration and confirmation)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # validated all-pairs for three series (light)
NULL_LABEL = {"cross_day_depth_matched": "Cross-day, depth-matched (primary)",
              "cross_day": "Cross-day, unmatched", "within_day_long_lag": "Same day, 15-30 min lags"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID); ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=1, linestyle="-"); ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2); ax.yaxis.label.set_color(INK2)


def curve_figure(summary, cls, out):
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=160); fig.patch.set_facecolor(SURFACE); style(ax)
    for k, null in enumerate(("cross_day_depth_matched", "cross_day", "within_day_long_lag")):
        rows = pd.DataFrame(summary["excess_curves"]["%s/%s/same_side" % (null, cls)])
        rows = rows[rows.lag_hi <= 900]
        x = np.sqrt(np.maximum(rows.lag_lo, 0.025) * rows.lag_hi)
        y = rows.name_median_ratio.to_numpy(float)
        lo = [c[0] if c and c[0] is not None else np.nan for c in rows.name_median_ratio_ci95]
        hi = [c[1] if c and c[1] is not None else np.nan for c in rows.name_median_ratio_ci95]
        ax.fill_between(x, lo, hi, color=SERIES[k], alpha=0.10, linewidth=0)
        ax.plot(x, y, color=SERIES[k], linewidth=2, solid_capstyle="round", label=NULL_LABEL[null])
        ax.plot(x, y, "o", color=SERIES[k], markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.axhline(1.0, color=INK2, linewidth=1)
    ax.set_xscale("log"); ax.set_xlabel("Lag between same-side packets (seconds, log scale)")
    ax.set_ylabel("Identical-size matches ÷ chance\n(median across names)")
    ax.set_title("Same-origin fingerprint: identical child sizes recur at short lags", color=INK, fontsize=11, loc="left")
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    for text in leg.get_texts():
        text.set_color(INK2)
    fig.tight_layout(); fig.savefig(out, facecolor=SURFACE); fig.savefig(Path(out).with_suffix(".pdf"), facecolor=SURFACE); plt.close(fig)


def roc_figure(summary, null, out, selected):
    table = pd.DataFrame(summary["burst_definitions"])
    t = table[(table.null == null) & (table.size_class == "u_nonround") & (table.min_packets == 3)]
    fig, ax = plt.subplots(figsize=(5.6, 4.6), dpi=160); fig.patch.set_facecolor(SURFACE); style(ax)
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1)
    labels = {"run": "run (ends at any opposite-side trade)", "stream": "stream (same side only)", "timing": "timing (sign-blind)"}
    for k, rule in enumerate(("run", "stream", "timing")):
        r = t[t.rule == rule].sort_values("gap_s")
        z = 4 if rule == "run" else 2
        ax.plot(r.name_mean_fpr, r.name_mean_tpr, color=SERIES[k], linewidth=2, label=labels[rule], zorder=z)
        ax.plot(r.name_mean_fpr, r.name_mean_tpr, "o", color=SERIES[k], markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=z + 0.5)
        for _, row in r.iterrows():
            if row.gap_s in (1.0, 10.0, 60.0) and rule != "run":
                ax.annotate("%gs" % row.gap_s, (row.name_mean_fpr, row.name_mean_tpr), xytext=(6, -10),
                            textcoords="offset points", fontsize=7.5, color=INK2)
    if selected:
        s = t[(t.rule == selected["rule"]) & (t.gap_s == selected["gap_s"])].iloc[0]
        ax.plot([s.name_mean_fpr], [s.name_mean_tpr], "o", markersize=12, markerfacecolor="none",
                markeredgecolor=INK, markeredgewidth=1.5)
        ax.annotate("selected: %s, %gs gap" % (selected["rule"], selected["gap_s"]), (s.name_mean_fpr, s.name_mean_tpr),
                    xytext=(-95, 28), textcoords="offset points", fontsize=8.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
    ax.set_xlim(0, 0.5); ax.set_ylim(0.3, 0.8)
    ax.set_xlabel("Share of all same-side pairs inside one burst (≈ false-positive rate)")
    ax.set_ylabel("Share of excess identical-size pairs kept together")
    ax.set_title("Burst definitions against the fingerprint", color=INK, fontsize=11, loc="left")
    leg = ax.legend(title="burst rule (gap 0.1s to 300s along each line)", frameon=False, fontsize=8, title_fontsize=8, loc="lower right")
    for text in leg.get_texts():
        text.set_color(INK2)
    leg.get_title().set_color(INK2)
    fig.tight_layout(); fig.savefig(out, facecolor=SURFACE); fig.savefig(Path(out).with_suffix(".pdf"), facecolor=SURFACE); plt.close(fig)


def md_tables(summary, null, state=None):
    out = []
    e1 = summary["e1_existence"]
    out.append("| class | lag | names | name-median ratio [95% CI] | names with ratio > 1 | pooled ratio |")
    out.append("|---|---|---:|---|---:|---:|")
    for cls in ("u_nonround", "u_visible", "u_rare", "u_round", "truncated"):
        for lag in ("0.5-2s", "2-10s"):
            r = e1.get("%s/%s/%s" % (null, cls, lag))
            if not r:
                continue
            ci = r["name_median_ratio_ci95"]
            out.append("| %s | %s | %d | %.3f [%s, %s] | %s | %.3f |" % (
                cls, lag, r["names_eligible"], r["name_median_ratio"],
                "%.3f" % ci[0] if ci[0] is not None else "—", "%.3f" % ci[1] if ci[1] is not None else "—",
                "%.0f%%" % (100 * r["names_ratio_above_1"]) if r["names_ratio_above_1"] is not None else "—",
                r["pooled_ratio"]))
    out.append("")
    sel = summary["selection"][null][:8]
    out.append("| rank | rule | gap (s) | mean per-name J [95% CI] | TPR | FPR | names |")
    out.append("|---:|---|---:|---|---:|---:|---:|")
    for r in sel:
        ci = r["name_mean_j_ci95"]
        out.append("| %d | %s | %g | %.3f [%.3f, %.3f] | %.3f | %.3f | %d |" % (
            r["rank"], r["rule"], r["gap_s"], r["name_mean_j"], ci[0], ci[1], r["name_mean_tpr"], r["name_mean_fpr"],
            r["names_eligible"]))
    out.append("")
    if state is not None:
        out.append("| lag | feature | names | name-median matched ÷ control [95% CI] | names below 1 | pooled |")
        out.append("|---|---|---:|---|---:|---:|")
        for key, r in state["ranges"].items():
            lag, feat = key.split("/")
            ci = r["name_median_ratio_ci95"] or [None, None]
            out.append("| %s | %s | %d | %.3f [%.3f, %.3f] | %.0f%% | %.3f |" % (
                lag, feat, r["names_eligible"], r["name_median_ratio"], ci[0], ci[1],
                100 * r["names_ratio_below_1"], r["pooled_ratio"]))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--state", help="aggregate_fingerprint_state.py output (corrected E3)")
    args = ap.parse_args()
    s = json.loads(Path(args.summary).read_text())
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    curve_figure(s, "u_nonround", out / ("fingerprint_curve_%s.png" % args.tag))
    roc_figure(s, "cross_day_depth_matched", out / ("fingerprint_roc_%s.png" % args.tag), s.get("selected_definition"))
    state = json.loads(Path(args.state).read_text()) if args.state else None
    (out / ("fingerprint_tables_%s.md" % args.tag)).write_text(md_tables(s, "cross_day_depth_matched", state) + "\n")
    print("written", out)


if __name__ == "__main__":
    main()
