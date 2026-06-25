#!/usr/bin/env python3
"""
aggregate_results.py — Post-HPC Cross-Sectional Aggregator

Runs after the 500-ticker HPC data/permanence phases complete. It:

  1. Concatenates every per-ticker burst CSV into one panel
     (results/aggregate/burst_panel_all.csv)
  2. Reports universe coverage (tickers found / missing, rows, date span)
  3. Computes the cross-sectional IC distribution: per-ticker burst-return
     correlation (the four-point illustration in the original paper becomes
     a 500-point distribution — Reviewer M9 / R6)
  4. Produces (or consumes) the regime classification and runs the COI-based
     Fama-MacBeth + quintile panel regression over the found universe
  5. Writes machine-readable summary.json + human SUMMARY.md

This is the glue that turns "interesting case study on four names" into a
"publishable cross-sectional alpha study" (Addendum R6 / B1).

Reuses, rather than re-implements:
  - regime_classifier.load_burst_returns / classify_regimes
  - panel_regression.compute_daily_coi / build_quintile_portfolios /
    fama_macbeth_regression

Usage:
    python3 src_py/aggregate_results.py \
        --results-dir results/ \
        --universe-file universes/full_500.txt \
        --open-csv open_all.csv --close-csv close_all.csv \
        --regime-csv results/regime/regime_classifications.csv \
        --factor-csv data/ff5_mom_daily.csv
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.absolute()))
import regime_classifier as rc
import panel_regression as pr


def read_universe(universe_file=None, tickers_arg=None):
    """Read the ticker universe from a file (one per line, '#' comments)
    or a comma-separated --tickers string."""
    if tickers_arg:
        return [t.strip() for t in tickers_arg.split(",") if t.strip()]
    tickers = []
    if universe_file and os.path.exists(universe_file):
        with open(universe_file) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if line:
                    tickers.append(line)
    # de-dup, preserve order
    seen, out = set(), []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def build_panel(results_dir, tickers, suffix, start_int, end_int):
    """Concatenate per-ticker burst CSVs into one panel; return
    (panel_df, found, missing)."""
    frames, found, missing = [], [], []
    for ticker in tickers:
        path = os.path.join(results_dir, f"bursts_{ticker}_{suffix}.csv")
        if not os.path.exists(path):
            missing.append(ticker)
            continue
        try:
            df = pd.read_csv(path)
        except Exception:  # noqa: BLE001
            missing.append(ticker)
            continue
        if df.empty:
            missing.append(ticker)
            continue
        try:
            df["Date"] = df["Date"].astype(int)
        except (ValueError, TypeError):
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y%m%d").astype(int)
        if "Ticker" not in df.columns:
            df["Ticker"] = ticker
        if start_int is not None:
            df = df[df["Date"] >= start_int]
        if end_int is not None:
            df = df[df["Date"] <= end_int]
        if df.empty:
            missing.append(ticker)
            continue
        frames.append(df)
        found.append(ticker)
    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return panel, found, missing


def main():
    ap = argparse.ArgumentParser(description="Post-HPC cross-sectional aggregator")
    ap.add_argument("--results-dir", default="results/")
    ap.add_argument("--universe-file", default=None)
    ap.add_argument("--tickers", default=None,
                    help="Comma-separated tickers (overrides --universe-file)")
    ap.add_argument("--open-csv", default="open_all.csv")
    ap.add_argument("--close-csv", default="close_all.csv")
    ap.add_argument("--regime-csv", default=None,
                    help="Existing regime CSV; if absent it is generated here")
    ap.add_argument("--factor-csv", default=None)
    ap.add_argument("--suffix", default="baseline_unfiltered")
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    ap.add_argument("--out-dir", default="results/aggregate")
    ap.add_argument("--run-panel-regression", action="store_true",
                    help="Also shell out to panel_regression.py for the full "
                         "FM + quintile + FF report (captured to a log)")
    args = ap.parse_args()

    start_int = int(pd.to_datetime(args.start_date).strftime("%Y%m%d")) if args.start_date else None
    end_int = int(pd.to_datetime(args.end_date).strftime("%Y%m%d")) if args.end_date else None

    tickers = read_universe(args.universe_file, args.tickers)
    if not tickers:
        print("ERROR: empty universe — provide --universe-file or --tickers.")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"  CROSS-SECTIONAL RESULT AGGREGATION")
    print(f"  Universe size: {len(tickers)}  |  results: {args.results_dir}")
    print(f"  Suffix: {args.suffix}")
    print(f"{'='*80}")

    # ── Step 1: build the burst panel ──
    panel, found, missing = build_panel(
        args.results_dir, tickers, args.suffix, start_int, end_int)

    coverage = {
        "universe_size": len(tickers),
        "found": len(found),
        "missing": len(missing),
        "missing_tickers": missing[:50],  # cap for readability
        "total_bursts": int(len(panel)),
    }
    print(f"\n  Coverage: {len(found)}/{len(tickers)} tickers with data "
          f"({len(missing)} missing)")
    if missing:
        print(f"  Missing (first 20): {missing[:20]}")
    if panel.empty:
        print("\n  No burst data found for any ticker. Nothing to aggregate.")
        with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
            json.dump({"coverage": coverage}, f, indent=2)
        return

    coverage["date_min"] = int(panel["Date"].min())
    coverage["date_max"] = int(panel["Date"].max())
    print(f"  Bursts: {len(panel):,}  |  Date span: {coverage['date_min']}–{coverage['date_max']}")

    panel_path = os.path.join(args.out_dir, "burst_panel_all.csv")
    panel.to_csv(panel_path, index=False)
    print(f"  Wrote panel: {panel_path}")

    # ── Step 2: cross-sectional IC distribution (Reviewer M9 / R6) ──
    print(f"\n  Computing per-ticker burst-return correlations (cross-sectional IC)...")
    feat = rc.load_burst_returns(args.results_dir, found, args.close_csv, args.suffix)
    ic_stats = {}
    if not feat.empty:
        ic = feat["BurstReturnCorr"].dropna()
        ic_stats = {
            "n": int(len(ic)),
            "mean_ic": float(ic.mean()),
            "median_ic": float(ic.median()),
            "std_ic": float(ic.std()),
            "pct_positive": float((ic > 0).mean() * 100.0),
            "ir_proxy": float(ic.mean() / ic.std() * np.sqrt(len(ic))) if ic.std() > 0 else 0.0,
        }
        print(f"    N tickers: {ic_stats['n']}  |  mean IC: {ic_stats['mean_ic']:+.4f}  "
              f"|  median: {ic_stats['median_ic']:+.4f}  |  % positive: {ic_stats['pct_positive']:.1f}%")
        # Grinold-Kahn breadth proxy: portfolio IR ≈ mean_IC * sqrt(breadth)
        print(f"    Breadth-amplified IR proxy (IC·√N): {ic_stats['ir_proxy']:.2f}")
        feat.to_csv(os.path.join(args.out_dir, "cross_sectional_ic.csv"), index=False)

    # ── Step 3: regime classification (generate if not supplied) ──
    regime_csv = args.regime_csv
    if (not regime_csv or not os.path.exists(regime_csv)) and not feat.empty:
        try:
            classified = rc.classify_regimes(feat, n_clusters=min(3, max(2, feat["BurstReturnCorr"].nunique())))
            regime_csv = os.path.join(args.out_dir, "regime_classifications.csv")
            classified.to_csv(regime_csv, index=False)
            counts = classified["Regime"].value_counts().to_dict()
            print(f"\n  Generated regime classification → {regime_csv}")
            print(f"    Regime counts: {counts}")
        except Exception as exc:  # noqa: BLE001
            print(f"  Regime classification skipped: {exc}")
            regime_csv = None

    # ── Step 4: COI panel + Fama-MacBeth over the found universe ──
    fm_summary = {}
    print(f"\n  Building daily COI panel + Fama-MacBeth (R_i,t+1 ~ COI)...")
    coi_daily = pr.compute_daily_coi(panel)
    # Apply sign-conditional flip from the regime CSV (Reviewer R3)
    if regime_csv and os.path.exists(regime_csv):
        rdf = pd.read_csv(regime_csv)
        if {"Ticker", "FlipSign"}.issubset(rdf.columns):
            flip = set(rdf.loc[rdf["FlipSign"] == -1, "Ticker"].astype(str))
            mask = coi_daily["Ticker"].isin(flip)
            coi_daily.loc[mask, "COI"] *= -1.0
            print(f"    Applied sign flip to {int(mask.sum())} stock-days "
                  f"({len(flip & set(found))} tickers)")

    # forward CLOP returns
    try:
        close_px = pd.read_csv(args.close_csv, index_col="date")
        close_px.index = pd.Index(close_px.index).astype(int)
        open_px = pd.read_csv(args.open_csv, index_col="date")
        open_px.index = pd.Index(open_px.index).astype(int)
        trading_days = np.array(sorted(close_px.index.tolist()), dtype=np.int64)
        recs = []
        for ticker in found:
            if ticker not in close_px.columns:
                continue
            cl = close_px[ticker].dropna()
            op = open_px[ticker].dropna() if ticker in open_px.columns else pd.Series(dtype=float)
            for date_int in cl.index:
                idx = np.searchsorted(trading_days, date_int, side="right")
                if idx >= len(trading_days):
                    continue
                nd = int(trading_days[idx])
                tc, no = cl.get(date_int, np.nan), op.get(nd, np.nan)
                if np.isnan(tc) or np.isnan(no) or tc <= 0:
                    continue
                recs.append({"Date": date_int, "Ticker": ticker, "fwd_return": (no - tc) / tc})
        returns_df = pd.DataFrame(recs)
        merged = coi_daily.merge(returns_df, on=["Date", "Ticker"], how="inner")
        print(f"    Merged COI/return panel: {len(merged):,} obs")
        if len(merged) >= 10:
            fm = None
            if pr._HAS_LINEARMODELS:
                fm = pr.fama_macbeth_linearmodels(merged, "fwd_return", ["COI"])
            if fm is None:
                fm = pr.fama_macbeth_regression(merged, "fwd_return", ["COI"])
            if fm and "COI" in fm:
                fm_summary = {k: {kk: vv for kk, vv in v.items()} for k, v in fm.items()}
                c = fm["COI"]
                print(f"    Fama-MacBeth COI: coef={c['mean']:+.6f}  "
                      f"t={c['t_stat']:.2f}  p={c['p_value']:.4f}")
    except Exception as exc:  # noqa: BLE001
        print(f"  COI/return merge skipped: {exc}")

    # ── Step 5: optional full panel_regression report ──
    if args.run_panel_regression:
        log_path = os.path.join(args.out_dir, "panel_regression_full.log")
        cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "panel_regression.py"),
               "--burst-dir", args.results_dir,
               "--tickers", ",".join(found),
               "--open-csv", args.open_csv, "--close-csv", args.close_csv,
               "--suffix", args.suffix]
        if regime_csv:
            cmd += ["--regime-csv", regime_csv]
        if args.factor_csv:
            cmd += ["--factor-csv", args.factor_csv]
        if args.start_date:
            cmd += ["--start-date", args.start_date]
        if args.end_date:
            cmd += ["--end-date", args.end_date]
        print(f"\n  Running full panel_regression.py → {log_path}")
        with open(log_path, "w") as lf:
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=False)

    # ── Step 6: write summaries ──
    summary = {"coverage": coverage, "cross_sectional_ic": ic_stats,
               "fama_macbeth": fm_summary, "regime_csv": regime_csv}
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    md = [f"# Aggregate Results Summary", ""]
    md.append(f"- Universe: **{coverage['found']}/{coverage['universe_size']}** tickers with data")
    md.append(f"- Total bursts: **{coverage['total_bursts']:,}**")
    if "date_min" in coverage:
        md.append(f"- Date span: {coverage['date_min']}–{coverage['date_max']}")
    if ic_stats:
        md.append(f"- Cross-sectional IC: mean **{ic_stats['mean_ic']:+.4f}**, "
                  f"median {ic_stats['median_ic']:+.4f}, "
                  f"{ic_stats['pct_positive']:.1f}% positive (N={ic_stats['n']})")
        md.append(f"- Breadth-amplified IR proxy (IC·√N): **{ic_stats['ir_proxy']:.2f}**")
    if fm_summary.get("COI"):
        c = fm_summary["COI"]
        md.append(f"- Fama-MacBeth COI coef: {c['mean']:+.6f} "
                  f"(t={c['t_stat']:.2f}, p={c['p_value']:.4f})")
    md.append("")
    md.append(f"Panel written to `{panel_path}`.")
    with open(os.path.join(args.out_dir, "SUMMARY.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"\n  Wrote: {os.path.join(args.out_dir, 'summary.json')}")
    print(f"  Wrote: {os.path.join(args.out_dir, 'SUMMARY.md')}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
