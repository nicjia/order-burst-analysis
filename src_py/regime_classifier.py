#!/usr/bin/env python3
"""
regime_classifier.py — Automated Microstructural Regime Classification

Replaces the hardcoded mean_revert_tickers list with a data-driven approach.
For each stock in the universe, computes:
  1. Rolling 60-day beta to SPY (market sensitivity)
  2. Burst-return correlation (does a buy burst predict +/- return?)
  3. K-means clustering into 3 regimes: momentum, mean-reverting, neutral

Outputs:
  results/regime/regime_classifications.csv
  Columns: Ticker, Regime, BurstBeta, BurstReturnCorr, FlipSign

Usage:
    python3 src_py/regime_classifier.py \
        --burst-dir results/ \
        --close-csv close_all.csv \
        --tickers AAPL,NVDA,JPM,GS,...
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def load_burst_returns(burst_dir, tickers, close_csv, suffix="baseline_unfiltered"):
    """
    For each ticker, compute the burst-return correlation:
    correlation between Direction (+1/-1) and next-day return.
    """
    close_px = pd.read_csv(close_csv, index_col="date")
    close_px.index = pd.Index(close_px.index).astype(int)

    records = []
    for ticker in tickers:
        path = os.path.join(burst_dir, f"bursts_{ticker}_{suffix}.csv")
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        if df.empty or "Direction" not in df.columns:
            continue

        # Convert dates
        try:
            df["Date"] = df["Date"].astype(int)
        except (ValueError, TypeError):
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y%m%d").astype(int)

        # Filter to directional bursts only
        directional = df[df["Direction"] != 0].copy()
        if len(directional) < 50:
            continue

        # Compute daily burst imbalance
        daily = directional.groupby("Date").agg(
            net_dir=("Direction", "mean"),
            burst_count=("Direction", "count"),
            total_vol=("Volume", "sum") if "Volume" in directional.columns else ("Direction", "count"),
        ).reset_index()

        # Merge with next-day returns
        if ticker not in close_px.columns:
            continue

        cl = close_px[ticker].dropna()
        ret = cl.pct_change().shift(-1)  # next-day return
        ret.name = "fwd_return"

        daily = daily.set_index("Date")
        merged = daily.join(ret, how="inner").dropna(subset=["fwd_return", "net_dir"])

        if len(merged) < 30:
            continue

        # Burst-return correlation
        corr, pval = spearmanr(merged["net_dir"], merged["fwd_return"])

        # Rolling beta to SPY (if SPY is in the close matrix)
        beta = np.nan
        if "SPY" in close_px.columns:
            spy_ret = close_px["SPY"].dropna().pct_change()
            stock_ret = cl.pct_change()
            aligned = pd.DataFrame({"stock": stock_ret, "spy": spy_ret}).dropna()
            if len(aligned) > 60:
                # Rolling 60-day OLS beta
                recent = aligned.tail(252)  # last year
                cov = np.cov(recent["stock"], recent["spy"])
                if cov[1, 1] > 0:
                    beta = cov[0, 1] / cov[1, 1]

        records.append({
            "Ticker": ticker,
            "BurstReturnCorr": corr,
            "BurstReturnPval": pval,
            "BurstBeta": beta,
            "N_BurstDays": len(merged),
        })

    return pd.DataFrame(records)


def classify_regimes(df, n_clusters=3):
    """
    K-means clustering on [BurstReturnCorr, BurstBeta] to identify:
      - Momentum stocks: positive burst-return correlation
      - Mean-reverting stocks: negative burst-return correlation
      - Neutral stocks: near-zero correlation
    """
    features = df[["BurstReturnCorr"]].copy()
    features = features.fillna(0)

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    df = df.copy()
    df["ClusterLabel"] = labels

    # Identify which cluster is which by mean burst-return correlation
    cluster_means = df.groupby("ClusterLabel")["BurstReturnCorr"].mean()

    # Sort clusters by their mean burst-return correlation
    sorted_clusters = cluster_means.sort_values()
    regime_map = {}

    if n_clusters == 3:
        regime_map[sorted_clusters.index[0]] = "mean_reverting"
        regime_map[sorted_clusters.index[1]] = "neutral"
        regime_map[sorted_clusters.index[2]] = "momentum"
    elif n_clusters == 2:
        regime_map[sorted_clusters.index[0]] = "mean_reverting"
        regime_map[sorted_clusters.index[1]] = "momentum"
    else:
        for i, idx in enumerate(sorted_clusters.index):
            if sorted_clusters[idx] < -0.02:
                regime_map[idx] = "mean_reverting"
            elif sorted_clusters[idx] > 0.02:
                regime_map[idx] = "momentum"
            else:
                regime_map[idx] = "neutral"

    df["Regime"] = df["ClusterLabel"].map(regime_map)

    # FlipSign: -1 for mean-reverting (COI inverted), +1 for momentum/neutral
    df["FlipSign"] = df["Regime"].map({
        "mean_reverting": -1,
        "neutral": 1,
        "momentum": 1,
    })

    return df


def main():
    ap = argparse.ArgumentParser(
        description="Automated microstructural regime classification")
    ap.add_argument("--burst-dir", required=True,
                    help="Directory containing burst CSV files")
    ap.add_argument("--close-csv", required=True,
                    help="Close price matrix CSV")
    ap.add_argument("--tickers", required=True,
                    help="Comma-separated list of tickers")
    ap.add_argument("--suffix", default="baseline_unfiltered",
                    help="Burst CSV suffix (default: baseline_unfiltered)")
    ap.add_argument("--n-clusters", type=int, default=3,
                    help="Number of regime clusters (default: 3)")
    ap.add_argument("--output-dir", default="results/regime",
                    help="Output directory for regime classification")
    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    print(f"\n{'='*70}")
    print(f"  AUTOMATED REGIME CLASSIFICATION")
    print(f"  Tickers: {len(tickers)}")
    print(f"  Burst dir: {args.burst_dir}")
    print(f"  Clusters: {args.n_clusters}")
    print(f"{'='*70}")

    # Step 1: Compute burst-return features
    print(f"\n  Computing burst-return correlations...")
    features_df = load_burst_returns(
        args.burst_dir, tickers, args.close_csv, args.suffix
    )

    if features_df.empty:
        print("ERROR: No valid burst-return data found.")
        sys.exit(1)

    print(f"  Valid tickers with data: {len(features_df)}")
    print(f"  Mean burst-return correlation: {features_df['BurstReturnCorr'].mean():.4f}")
    print(f"  Std burst-return correlation:  {features_df['BurstReturnCorr'].std():.4f}")

    # Step 2: Cluster into regimes
    print(f"\n  Running K-means clustering ({args.n_clusters} clusters)...")
    classified = classify_regimes(features_df, n_clusters=args.n_clusters)

    # Step 3: Report
    print(f"\n  {'='*70}")
    print(f"  REGIME CLASSIFICATION RESULTS")
    print(f"  {'='*70}")

    regime_summary = classified.groupby("Regime").agg(
        count=("Ticker", "count"),
        mean_corr=("BurstReturnCorr", "mean"),
        mean_beta=("BurstBeta", "mean"),
    )
    for regime, row in regime_summary.iterrows():
        flip = "×(-1)" if regime == "mean_reverting" else "×(+1)"
        print(f"  {regime:<16} N={int(row['count']):>3}  "
              f"mean_corr={row['mean_corr']:>+.4f}  "
              f"mean_beta={row['mean_beta']:>6.2f}  "
              f"signal_flip={flip}")

    # Print per-ticker detail
    print(f"\n  Per-Ticker Classification:")
    print(f"  {'Ticker':<8} {'Regime':<16} {'BurstCorr':>10} {'Beta':>8} {'FlipSign':>10}")
    print(f"  {'-'*60}")
    for _, row in classified.sort_values("BurstReturnCorr").iterrows():
        print(f"  {row['Ticker']:<8} {row['Regime']:<16} "
              f"{row['BurstReturnCorr']:>+10.4f} "
              f"{row['BurstBeta']:>8.2f} "
              f"{row['FlipSign']:>+10d}")

    # Step 4: Save
    os.makedirs(args.output_dir, exist_ok=True)
    output_csv = os.path.join(args.output_dir, "regime_classifications.csv")
    classified.to_csv(output_csv, index=False)
    print(f"\n  Saved: {output_csv}")

    # Also output a mean_revert_tickers.txt compatible with run_pipeline.sh
    mr_tickers = classified[classified["Regime"] == "mean_reverting"]["Ticker"].tolist()
    mr_file = os.path.join(args.output_dir, "mean_revert_tickers_auto.txt")
    with open(mr_file, "w") as f:
        f.write("# Auto-generated by regime_classifier.py\n")
        for t in sorted(mr_tickers):
            f.write(f"{t}\n")
    print(f"  Saved: {mr_file} ({len(mr_tickers)} tickers)")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
