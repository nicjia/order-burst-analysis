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


# ─────────────────────────────────────────────────────────────────────────
# Sharpe inference on a realized daily-PnL series (Reviewer M3 / Rec #4)
# ─────────────────────────────────────────────────────────────────────────

def lo_sharpe_se(daily_returns, q=None):
    """
    Lo (2002) autocorrelation-adjusted standard error of the annualized
    Sharpe ratio.

        SE(SR_ann) = sqrt(252) * sqrt((1 + 2*Σ_{k=1}^{q} rho_k) / T)

    where rho_k is the lag-k autocorrelation of the (per-period) return
    series.  ``q`` defaults to the Newey-West optimal bandwidth.

    Returns (sharpe_ann, se_ann, q).
    """
    r = np.asarray(daily_returns, dtype=float)
    r = r[np.isfinite(r)]
    T = len(r)
    if T < 3 or r.std(ddof=1) == 0:
        return 0.0, np.nan, 0

    sharpe_ann = (r.mean() / r.std(ddof=1)) * np.sqrt(252.0)

    if q is None:
        q = max(1, int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0))))

    correction = 1.0
    for k in range(1, q + 1):
        if T > k:
            rho_k = np.corrcoef(r[k:], r[:-k])[0, 1]
            if np.isfinite(rho_k):
                correction += 2.0 * rho_k
    correction = max(correction, 0.01)  # floor to avoid negative variance

    se_ann = np.sqrt(252.0) * np.sqrt(correction / T)
    return sharpe_ann, se_ann, q


def deflated_sharpe_ratio(sharpe_ann, n_trials, n_obs, skew=0.0, kurt=3.0):
    """
    Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Probability that the observed Sharpe exceeds the expected maximum
    Sharpe under the null of zero skill across ``n_trials`` independent
    trials, accounting for non-normal returns (skewness γ3, kurtosis γ4)
    and the finite sample length.

    Returns (dsr_probability, expected_max_sharpe_ann).
    """
    from scipy.stats import norm

    # Work in per-period (non-annualized) Sharpe.
    sr = sharpe_ann / np.sqrt(252.0)

    if n_trials <= 1 or n_obs <= 2:
        return float("nan"), float("nan")

    # Variance of the Sharpe estimator (Mertens / Lo, with higher moments).
    var_sr = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2) / (n_obs - 1.0)
    var_sr = max(var_sr, 1e-12)
    sigma_sr = np.sqrt(var_sr)

    # Expected maximum of N i.i.d. Sharpe estimates via extreme-value theory
    # (Bailey-LdP). The EVT term is in standard-normal z-units; scale by the
    # Sharpe estimator's std to convert to a Sharpe (per-period) threshold.
    gamma = 0.5772156649  # Euler-Mascheroni
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    e_max_z = (1.0 - gamma) * z1 + gamma * z2
    sr_star = sigma_sr * e_max_z  # expected max Sharpe under the null (per-period)

    dsr = norm.cdf((sr - sr_star) / sigma_sr)
    return float(dsr), float(sr_star * np.sqrt(252.0))


def block_bootstrap_ci(daily_pnl, n_boot=2000, block_len=None,
                       alpha=0.05, seed=42):
    """
    Circular block bootstrap (Politis-Romano style) confidence intervals
    for (i) cumulative PnL and (ii) mean per-period PnL, preserving the
    autocorrelation structure of the daily series.

    Returns dict with cum_lo/cum_hi/cum_point and mean_lo/mean_hi/mean_point.
    """
    x = np.asarray(daily_pnl, dtype=float)
    x = x[np.isfinite(x)]
    T = len(x)
    if T < 5:
        return None

    if block_len is None:
        block_len = max(1, int(round(T ** (1.0 / 3.0))))  # ~T^(1/3)

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block_len))

    cum_samples = np.empty(n_boot)
    mean_samples = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, T, size=n_blocks)
        idx = (starts[:, None] + np.arange(block_len)[None, :]).ravel() % T
        idx = idx[:T]
        sample = x[idx]
        cum_samples[b] = sample.sum()
        mean_samples[b] = sample.mean()

    lo_q, hi_q = 100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)
    return {
        "block_len": block_len,
        "n_boot": n_boot,
        "cum_point": float(x.sum()),
        "cum_lo": float(np.percentile(cum_samples, lo_q)),
        "cum_hi": float(np.percentile(cum_samples, hi_q)),
        "mean_point": float(x.mean()),
        "mean_lo": float(np.percentile(mean_samples, lo_q)),
        "mean_hi": float(np.percentile(mean_samples, hi_q)),
    }


def _load_daily_pnl(pnl_csv):
    """Load a daily PnL series from either a debug-trades CSV (columns
    'day' + 'net_raw') or a generic CSV with a single numeric PnL column."""
    df = pd.read_csv(pnl_csv)
    if "day" in df.columns and "net_raw" in df.columns:
        df["day"] = pd.to_datetime(df["day"]).dt.strftime("%Y%m%d").astype(int)
        return df.groupby("day")["net_raw"].sum().sort_index()
    # Fallback: a column literally named pnl / daily_pnl / net
    for col in ("daily_pnl", "pnl", "net", "net_raw"):
        if col in df.columns:
            return df[col].astype(float)
    # Last resort: the last numeric column
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        return pd.Series(dtype=float)
    return num.iloc[:, -1].astype(float)


def run_pnl_inference(pnl_csv, n_trials):
    """Print Lo-SE, Deflated-Sharpe, and block-bootstrap inference for a
    realized daily-PnL series (Reviewer M3 / Rec #4)."""
    from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurt

    daily = _load_daily_pnl(pnl_csv)
    daily = daily[np.isfinite(daily)]
    if len(daily) < 5:
        print(f"  PnL inference skipped: only {len(daily)} daily observations in {pnl_csv}")
        return

    r = daily.values.astype(float)
    T = len(r)

    sharpe_ann, se_ann, q = lo_sharpe_se(r)
    z = sharpe_ann / se_ann if (se_ann and np.isfinite(se_ann) and se_ann > 0) else np.nan
    g3 = float(scipy_skew(r)) if T > 2 else 0.0
    g4 = float(scipy_kurt(r, fisher=False)) if T > 3 else 3.0
    dsr, e_max_sr = deflated_sharpe_ratio(sharpe_ann, n_trials, T, skew=g3, kurt=g4)
    boot = block_bootstrap_ci(r)

    print(f"\n{'='*100}")
    print(f"  REALIZED-PnL SHARPE INFERENCE  (Reviewer M3 / Rec #4)")
    print(f"  Source: {pnl_csv}")
    print(f"  Trading days: {T}")
    print(f"{'='*100}")
    print(f"  Annualized Sharpe:            {sharpe_ann:.3f}")
    print(f"  Lo (2002) SE (q={q} lags):     {se_ann:.3f}")
    if np.isfinite(z):
        ci_lo = sharpe_ann - 1.96 * se_ann
        ci_hi = sharpe_ann + 1.96 * se_ann
        print(f"  Sharpe 95% CI:                [{ci_lo:.3f}, {ci_hi:.3f}]   (z={z:.2f})")
    print(f"  Return skew / kurtosis:       {g3:+.3f} / {g4:.3f}")
    print(f"  Expected max Sharpe (N={n_trials}):  {e_max_sr:.3f}")
    print(f"  Deflated Sharpe (prob real):  {dsr:.4f}")
    if np.isfinite(dsr):
        verdict = "SURVIVES" if dsr > 0.95 else "does NOT survive"
        print(f"  → DSR {verdict} multiple-testing deflation at the 95% level")
    if boot:
        print(f"\n  Circular block bootstrap (block_len={boot['block_len']}, "
              f"n_boot={boot['n_boot']}):")
        print(f"    Cumulative PnL:  {boot['cum_point']:.2f}  "
              f"95% CI [{boot['cum_lo']:.2f}, {boot['cum_hi']:.2f}]")
        print(f"    Mean daily PnL:  {boot['mean_point']:.4f}  "
              f"95% CI [{boot['mean_lo']:.4f}, {boot['mean_hi']:.4f}]")
        spans_zero = boot['cum_lo'] <= 0 <= boot['cum_hi']
        print(f"    → Cumulative-PnL CI {'INCLUDES' if spans_zero else 'excludes'} zero")
    print(f"{'='*100}")


def main():
    ap = argparse.ArgumentParser(
        description="Multiple testing correction + Sharpe inference (M3/M4)")
    ap.add_argument("results_dir", nargs="?", default=None,
                    help="Directory containing Optuna result JSONs (optional "
                         "if --pnl-csv is given)")
    ap.add_argument("--all-tickers", action="store_true",
                    help="Search subdirectories for all tickers")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Significance level (default: 0.05)")
    ap.add_argument("--n-trials", type=int, default=100,
                    help="Number of Optuna trials per ticker (for Sharpe haircut/DSR)")
    ap.add_argument("--pnl-csv", default=None,
                    help="Optional daily-PnL or debug-trades CSV: run Lo-SE, "
                         "Deflated-Sharpe, and block-bootstrap inference on it")
    args = ap.parse_args()

    # ── Realized-PnL Sharpe inference path (Reviewer M3 / Rec #4) ──
    if args.pnl_csv:
        run_pnl_inference(args.pnl_csv, args.n_trials)

    if args.results_dir is None:
        if not args.pnl_csv:
            print("ERROR: provide a results_dir and/or --pnl-csv.")
            sys.exit(1)
        return

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
