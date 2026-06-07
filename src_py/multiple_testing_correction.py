#!/usr/bin/env python3
"""
multiple_testing_correction.py — Sharpe Inference & Multiple Testing Correction

Because Optuna tests hundreds of parameter combinations, the reviewers demand
correction for multiple hypothesis testing to prove results aren't a product
of data mining.

Applies:
  1. Bonferroni correction (conservative family-wise error rate)
  2. Benjamini-Hochberg FDR (false discovery rate, less conservative)
  3. Holm-Bonferroni (step-down, tighter than Bonferroni)

Usage:
    python3 src_py/multiple_testing_correction.py results/optuna_regression/NVDA/
    python3 src_py/multiple_testing_correction.py results/optuna_regression/ --all-tickers
"""

import argparse
import sys
import os
import json
import glob

import numpy as np
import pandas as pd


def load_optuna_results(directory, all_tickers=False):
    """Load all Optuna result JSONs from a directory tree."""
    results = []

    if all_tickers:
        pattern = os.path.join(directory, "*/best_regression_params_*.json")
    else:
        pattern = os.path.join(directory, "best_regression_params_*.json")

    json_files = sorted(glob.glob(pattern))

    if not json_files:
        # Try nested structure
        pattern = os.path.join(directory, "**/*.json")
        json_files = sorted(glob.glob(pattern, recursive=True))

    for jf in json_files:
        try:
            with open(jf) as f:
                data = json.load(f)

            result = {
                "file": os.path.basename(jf),
                "ticker": data.get("ticker", "unknown"),
                "target": data.get("target", "unknown"),
                "hawkes_tag": data.get("hawkes_tag", "unknown"),
                "score": data.get("score", np.nan),
                "raw_spearman": data.get("raw_spearman", np.nan),
                "p_value": data.get("p_value", np.nan),
                "n_test": data.get("n_test", 0),
                "n_train": data.get("n_train", 0),
                "n_total": data.get("n_total", 0),
                "confidence": data.get("confidence", np.nan),
                "vol_frac": data.get("vol_frac", np.nan),
                "dir_thresh": data.get("dir_thresh", np.nan),
                "vol_ratio": data.get("vol_ratio", np.nan),
                "kappa": data.get("kappa", np.nan),
                "n_trials": 100,  # Default Optuna trial count
            }
            results.append(result)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: Skipping {jf}: {e}")

    return results


def bonferroni_correction(p_values, alpha=0.05):
    """Apply Bonferroni correction."""
    n = len(p_values)
    adjusted = np.minimum(np.array(p_values) * n, 1.0)
    significant = adjusted < alpha
    return adjusted, significant


def holm_bonferroni_correction(p_values, alpha=0.05):
    """Apply Holm-Bonferroni step-down correction."""
    n = len(p_values)
    p_arr = np.array(p_values)
    sorted_idx = np.argsort(p_arr)
    sorted_p = p_arr[sorted_idx]

    adjusted = np.zeros(n)
    for i in range(n):
        adjusted[i] = sorted_p[i] * (n - i)

    # Enforce monotonicity
    for i in range(1, n):
        adjusted[i] = max(adjusted[i], adjusted[i - 1])
    adjusted = np.minimum(adjusted, 1.0)

    # Map back to original order
    result = np.zeros(n)
    result[sorted_idx] = adjusted

    significant = result < alpha
    return result, significant


def benjamini_hochberg_fdr(p_values, alpha=0.05):
    """Apply Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    p_arr = np.array(p_values)
    sorted_idx = np.argsort(p_arr)
    sorted_p = p_arr[sorted_idx]

    adjusted = np.zeros(n)
    for i in range(n):
        adjusted[i] = sorted_p[i] * n / (i + 1)

    # Enforce monotonicity (step up)
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    adjusted = np.minimum(adjusted, 1.0)

    # Map back
    result = np.zeros(n)
    result[sorted_idx] = adjusted

    significant = result < alpha
    return result, significant


def compute_haircut_sharpe(observed_sharpe, n_trials, n_observations):
    """
    Harvey-Liu-Zhu (2016) Sharpe ratio haircut.
    Adjusts observed Sharpe for multiple testing bias.

    Haircut = expected maximum of n_trials independent standard normal variables.
    """
    if n_trials <= 1:
        return observed_sharpe

    # Expected max of n standard normals ≈ sqrt(2 * ln(n_trials))
    expected_max_z = np.sqrt(2.0 * np.log(n_trials))

    # Adjusted Sharpe = observed - haircut
    # But we need to convert Sharpe to z-score first
    z_score = observed_sharpe * np.sqrt(n_observations / 252.0)
    z_adjusted = z_score - expected_max_z

    # Convert back to annualized Sharpe
    adjusted_sharpe = z_adjusted / np.sqrt(n_observations / 252.0)

    return adjusted_sharpe


def main():
    ap = argparse.ArgumentParser(
        description="Multiple testing correction for Optuna p-values")
    ap.add_argument("results_dir", help="Directory containing Optuna result JSONs")
    ap.add_argument("--all-tickers", action="store_true",
                    help="Search subdirectories for all tickers")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Significance level (default: 0.05)")
    ap.add_argument("--n-trials", type=int, default=100,
                    help="Number of Optuna trials per ticker (for Sharpe haircut)")
    args = ap.parse_args()

    # ── Load results ──
    results = load_optuna_results(args.results_dir, args.all_tickers)

    if not results:
        print(f"ERROR: No Optuna result JSONs found in {args.results_dir}")
        sys.exit(1)

    print(f"\n{'='*100}")
    print(f"  MULTIPLE TESTING CORRECTION FOR OPTUNA PARAMETER SEARCH")
    print(f"  Results directory: {args.results_dir}")
    print(f"  Total result files: {len(results)}")
    print(f"  Significance level: α = {args.alpha}")
    print(f"{'='*100}")

    # ── Extract p-values ──
    df = pd.DataFrame(results)
    valid_mask = df["p_value"].notna() & (df["p_value"] > 0)
    df_valid = df[valid_mask].copy()

    if len(df_valid) == 0:
        print("ERROR: No valid p-values found in results.")
        sys.exit(1)

    p_values = df_valid["p_value"].values
    n_tests = len(p_values)

    # The TOTAL number of hypotheses tested is n_optuna_trials × n_tickers × n_targets
    total_hypotheses = args.n_trials * len(df_valid)

    print(f"\n  Valid results with p-values:     {n_tests}")
    print(f"  Total hypotheses (estimated):    {total_hypotheses}")
    print(f"  (n_trials={args.n_trials} × n_results={n_tests})")

    # ── Apply corrections (using total hypotheses count) ──
    # For the individual best-result p-values, adjust by total trial count
    adjusted_p_bonferroni = np.minimum(p_values * total_hypotheses, 1.0)
    _, sig_bonferroni = bonferroni_correction(p_values * args.n_trials, args.alpha)

    adjusted_p_holm, sig_holm = holm_bonferroni_correction(
        p_values * args.n_trials, args.alpha)

    adjusted_p_bh, sig_bh = benjamini_hochberg_fdr(
        p_values * args.n_trials, args.alpha)

    df_valid["p_bonferroni"] = adjusted_p_bonferroni
    df_valid["p_holm"] = adjusted_p_holm
    df_valid["p_bh_fdr"] = adjusted_p_bh
    df_valid["sig_bonferroni"] = sig_bonferroni
    df_valid["sig_holm"] = sig_holm
    df_valid["sig_bh_fdr"] = sig_bh

    # ── Print results table ──
    print(f"\n  {'Ticker':<8} {'Target':<12} {'ρ':>8} {'Raw p':>10} "
          f"{'Bonf. p':>10} {'Holm p':>10} {'BH-FDR p':>10} "
          f"{'Bonf.':>6} {'Holm':>6} {'FDR':>6}")
    print(f"  {'-'*100}")

    for _, row in df_valid.iterrows():
        bonf_sig = "✓" if row["sig_bonferroni"] else "✗"
        holm_sig = "✓" if row["sig_holm"] else "✗"
        bh_sig = "✓" if row["sig_bh_fdr"] else "✗"

        print(f"  {row['ticker']:<8} {row['target']:<12} "
              f"{row['raw_spearman']:>8.4f} "
              f"{row['p_value']:>10.2e} "
              f"{row['p_bonferroni']:>10.2e} "
              f"{row['p_holm']:>10.2e} "
              f"{row['p_bh_fdr']:>10.2e} "
              f"{bonf_sig:>6} {holm_sig:>6} {bh_sig:>6}")

    # ── Summary statistics ──
    print(f"\n  Summary of Multiple Testing Corrections:")
    print(f"  {'Method':<25} {'# Significant':>15} {'% Surviving':>15}")
    print(f"  {'-'*55}")
    print(f"  {'Raw (uncorrected)':<25} {(p_values < args.alpha).sum():>15} "
          f"{100.0 * (p_values < args.alpha).sum() / n_tests:>14.1f}%")
    print(f"  {'Bonferroni':<25} {sig_bonferroni.sum():>15} "
          f"{100.0 * sig_bonferroni.sum() / n_tests:>14.1f}%")
    print(f"  {'Holm-Bonferroni':<25} {sig_holm.sum():>15} "
          f"{100.0 * sig_holm.sum() / n_tests:>14.1f}%")
    print(f"  {'Benjamini-Hochberg FDR':<25} {sig_bh.sum():>15} "
          f"{100.0 * sig_bh.sum() / n_tests:>14.1f}%")

    # ── Harvey-Liu-Zhu Sharpe Haircut ──
    if "score" in df_valid.columns:
        print(f"\n  Harvey-Liu-Zhu (2016) Sharpe Ratio Haircut:")
        print(f"  (Adjusts for data mining when testing {args.n_trials} parameter combinations)")
        print(f"  {'Ticker':<8} {'Target':<12} {'Obs. Score':>12} "
              f"{'N_test':>8} {'Adjusted':>12} {'Survives?':>10}")
        print(f"  {'-'*70}")

        for _, row in df_valid.iterrows():
            observed = row["score"] if not np.isnan(row["score"]) else 0.0
            n_obs = row["n_test"] if row["n_test"] > 0 else 252
            adjusted = compute_haircut_sharpe(observed, args.n_trials, n_obs)
            survives = "YES" if adjusted > 0 else "NO"
            print(f"  {row['ticker']:<8} {row['target']:<12} "
                  f"{observed:>12.6f} {n_obs:>8} "
                  f"{adjusted:>12.6f} {survives:>10}")

    print(f"\n{'='*100}")


if __name__ == "__main__":
    main()
