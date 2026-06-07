#!/usr/bin/env python3
"""
poisson_baseline_test.py — Reviewer B2: Poisson Null-Model Structural Test

Proves that detected bursts are NOT just random order-arrival clustering
by comparing the observed burst arrival process against a homogeneous
Poisson baseline.

Tests:
  1. Kolmogorov-Smirnov test of inter-burst arrival times vs Exponential(λ)
  2. Dispersion Index: Var(N) / E(N) — Poisson ≡ 1.0
  3. Coefficient of Variation of inter-arrival times — Poisson ≡ 1.0
  4. Fano Factor per-day: windowed count variance / mean

Usage:
    python3 src_py/poisson_baseline_test.py results/bursts_NVDA_baseline_unfiltered.csv
    python3 src_py/poisson_baseline_test.py results/bursts_NVDA_baseline_unfiltered.csv --filtered results/bursts_NVDA_baseline_filtered.csv
"""

import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats


def compute_inter_arrival_times(burst_df):
    """Compute per-day inter-burst arrival times in seconds."""
    all_iats = []
    for date, group in burst_df.groupby("Date"):
        start_times = group["StartTime"].sort_values().values
        if len(start_times) < 2:
            continue
        iats = np.diff(start_times)
        all_iats.extend(iats[iats > 0])
    return np.array(all_iats)


def compute_daily_counts(burst_df):
    """Compute per-day burst counts."""
    return burst_df.groupby("Date").size().values


def run_poisson_tests(iats, daily_counts, label=""):
    """Run full Poisson null-model test battery."""
    print(f"\n{'='*70}")
    print(f"  POISSON NULL-MODEL TEST: {label}")
    print(f"{'='*70}")

    # ── 1. Summary Statistics ──
    print(f"\n  Sample Size:")
    print(f"    Total inter-arrival intervals:  {len(iats):,}")
    print(f"    Total trading days:             {len(daily_counts):,}")
    print(f"    Total bursts:                   {daily_counts.sum():,}")

    mean_iat = np.mean(iats)
    std_iat = np.std(iats)
    median_iat = np.median(iats)
    mean_daily = np.mean(daily_counts)
    var_daily = np.var(daily_counts)

    print(f"\n  Inter-Arrival Time Statistics:")
    print(f"    Mean:                           {mean_iat:.4f} seconds")
    print(f"    Median:                         {median_iat:.4f} seconds")
    print(f"    Std Dev:                        {std_iat:.4f} seconds")
    print(f"    Min:                            {np.min(iats):.6f} seconds")
    print(f"    Max:                            {np.max(iats):.4f} seconds")

    # ── 2. Kolmogorov-Smirnov Test ──
    # Under Poisson null: inter-arrival times ~ Exponential(1/mean)
    rate_lambda = 1.0 / mean_iat if mean_iat > 0 else 1.0
    ks_stat, ks_pval = stats.kstest(iats, "expon", args=(0, 1.0 / rate_lambda))

    print(f"\n  Test 1: Kolmogorov-Smirnov vs Exponential(λ={rate_lambda:.4f})")
    print(f"    KS Statistic:                   {ks_stat:.6f}")
    print(f"    P-Value:                        {ks_pval:.2e}")
    if ks_pval < 0.01:
        print(f"    → REJECT Poisson null at 1% level (p < 0.01)")
    elif ks_pval < 0.05:
        print(f"    → REJECT Poisson null at 5% level (p < 0.05)")
    else:
        print(f"    → FAIL to reject Poisson null")

    # ── 3. Dispersion Index (Index of Dispersion) ──
    # For Poisson: Var(N) / E(N) = 1.0
    # Over-dispersed (clustered) → index > 1
    dispersion_index = var_daily / mean_daily if mean_daily > 0 else np.nan
    # Chi-squared test for dispersion
    n_days = len(daily_counts)
    chi2_stat = (n_days - 1) * dispersion_index
    chi2_pval = 1.0 - stats.chi2.cdf(chi2_stat, df=n_days - 1)

    print(f"\n  Test 2: Dispersion Index (Var/Mean of daily burst counts)")
    print(f"    E[N]:                           {mean_daily:.2f} bursts/day")
    print(f"    Var[N]:                         {var_daily:.2f}")
    print(f"    Dispersion Index:               {dispersion_index:.4f}")
    print(f"    (Poisson null = 1.0)")
    if dispersion_index > 1.0:
        print(f"    → OVER-DISPERSED: bursts are clustered beyond Poisson")
    else:
        print(f"    → UNDER-DISPERSED or consistent with Poisson")
    print(f"    χ² Statistic:                   {chi2_stat:.2f}")
    print(f"    χ² P-Value (over-dispersion):   {chi2_pval:.2e}")

    # ── 4. Coefficient of Variation ──
    # For Exponential (Poisson IATs): CV = 1.0
    # CV > 1 → more clustered than Poisson
    cv = std_iat / mean_iat if mean_iat > 0 else np.nan

    print(f"\n  Test 3: Coefficient of Variation of Inter-Arrival Times")
    print(f"    CV:                             {cv:.4f}")
    print(f"    (Poisson/Exponential null = 1.0)")
    if cv > 1.0:
        print(f"    → CV > 1: inter-arrival times are MORE variable than Poisson")
        print(f"      (consistent with bursty/clustered arrivals)")
    else:
        print(f"    → CV ≤ 1: inter-arrival times are LESS variable than Poisson")
        print(f"      (consistent with regular/periodic arrivals)")

    # ── 5. Anderson-Darling Test (more sensitive to tails) ──
    ad_result = stats.anderson(iats, dist="expon")
    print(f"\n  Test 4: Anderson-Darling vs Exponential")
    print(f"    AD Statistic:                   {ad_result.statistic:.4f}")
    for sl, cv_val in zip(ad_result.significance_level, ad_result.critical_values):
        reject = "REJECT" if ad_result.statistic > cv_val else "accept"
        print(f"    {sl:5.1f}% level: critical={cv_val:.4f}  → {reject}")

    # ── 6. Quantile Summary ──
    print(f"\n  Inter-Arrival Time Quantiles:")
    quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    for q in quantiles:
        print(f"    {q*100:5.1f}th percentile:          {np.quantile(iats, q):.4f} s")

    print(f"\n{'='*70}")

    return {
        "label": label,
        "n_iats": len(iats),
        "n_days": n_days,
        "total_bursts": int(daily_counts.sum()),
        "mean_iat": mean_iat,
        "median_iat": median_iat,
        "cv": cv,
        "ks_stat": ks_stat,
        "ks_pval": ks_pval,
        "dispersion_index": dispersion_index,
        "chi2_pval": chi2_pval,
        "ad_stat": ad_result.statistic,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Poisson null-model test for burst arrival process (Reviewer B2)")
    ap.add_argument("unfiltered_csv",
                    help="Path to unfiltered bursts CSV (all raw bursts from C++)")
    ap.add_argument("--filtered", default=None,
                    help="Optional: path to filtered bursts CSV (after Optuna params applied)")
    ap.add_argument("--ticker", default=None,
                    help="Filter to a specific ticker (if CSV contains multiple)")
    ap.add_argument("--start-date", default=None,
                    help="Inclusive start date (YYYY-MM-DD or YYYYMMDD)")
    ap.add_argument("--end-date", default=None,
                    help="Inclusive end date (YYYY-MM-DD or YYYYMMDD)")
    args = ap.parse_args()

    # ── Load unfiltered bursts ──
    print(f"Loading unfiltered bursts from {args.unfiltered_csv}...")
    df_raw = pd.read_csv(args.unfiltered_csv)

    # Normalize date column
    try:
        df_raw["Date"] = df_raw["Date"].astype(int)
    except (ValueError, TypeError):
        df_raw["Date"] = pd.to_datetime(df_raw["Date"]).dt.strftime("%Y%m%d").astype(int)

    if args.ticker:
        df_raw = df_raw[df_raw["Ticker"] == args.ticker].copy()

    if args.start_date:
        start_int = int(pd.to_datetime(args.start_date).strftime("%Y%m%d"))
        df_raw = df_raw[df_raw["Date"] >= start_int].copy()
    if args.end_date:
        end_int = int(pd.to_datetime(args.end_date).strftime("%Y%m%d"))
        df_raw = df_raw[df_raw["Date"] <= end_int].copy()

    if len(df_raw) < 10:
        print(f"ERROR: Only {len(df_raw)} bursts after filtering. Need at least 10.")
        sys.exit(1)

    print(f"  Loaded {len(df_raw):,} unfiltered bursts across "
          f"{df_raw['Date'].nunique()} trading days")

    if "StartTime" not in df_raw.columns:
        print("ERROR: CSV must contain 'StartTime' column (seconds past midnight)")
        sys.exit(1)

    # ── Run tests on unfiltered bursts ──
    iats_raw = compute_inter_arrival_times(df_raw)
    counts_raw = compute_daily_counts(df_raw)

    if len(iats_raw) < 10:
        print("ERROR: Too few inter-arrival intervals to test.")
        sys.exit(1)

    results = []
    results.append(run_poisson_tests(iats_raw, counts_raw, label="ALL RAW BURSTS (Unfiltered)"))

    # ── Optionally run on filtered bursts ──
    if args.filtered:
        print(f"\nLoading filtered bursts from {args.filtered}...")
        df_filt = pd.read_csv(args.filtered)

        try:
            df_filt["Date"] = df_filt["Date"].astype(int)
        except (ValueError, TypeError):
            df_filt["Date"] = pd.to_datetime(df_filt["Date"]).dt.strftime("%Y%m%d").astype(int)

        if args.ticker:
            df_filt = df_filt[df_filt["Ticker"] == args.ticker].copy()
        if args.start_date:
            df_filt = df_filt[df_filt["Date"] >= start_int].copy()
        if args.end_date:
            df_filt = df_filt[df_filt["Date"] <= end_int].copy()

        print(f"  Loaded {len(df_filt):,} filtered bursts across "
              f"{df_filt['Date'].nunique()} trading days")

        if len(df_filt) >= 10 and "StartTime" in df_filt.columns:
            iats_filt = compute_inter_arrival_times(df_filt)
            counts_filt = compute_daily_counts(df_filt)
            if len(iats_filt) >= 10:
                results.append(
                    run_poisson_tests(iats_filt, counts_filt,
                                     label="FILTERED BURSTS (Post-Optuna)"))

    # ── Summary comparison ──
    if len(results) > 1:
        print(f"\n{'='*70}")
        print(f"  COMPARISON: Raw vs Filtered Burst Arrival Process")
        print(f"{'='*70}")
        print(f"  {'Metric':<30} {'Raw':<20} {'Filtered':<20}")
        print(f"  {'-'*70}")
        for key in ["total_bursts", "n_days", "mean_iat", "cv", "ks_pval",
                     "dispersion_index"]:
            raw_val = results[0][key]
            filt_val = results[1][key]
            if isinstance(raw_val, float):
                print(f"  {key:<30} {raw_val:<20.6f} {filt_val:<20.6f}")
            else:
                print(f"  {key:<30} {str(raw_val):<20} {str(filt_val):<20}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
