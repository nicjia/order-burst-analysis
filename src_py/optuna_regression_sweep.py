#!/usr/bin/env python3
"""
optuna_regression_sweep.py

Regression-based Optuna sweep that mirrors the exact Phase-III execution environment.
Addresses three critical traps:
  1. Sparsity Exploit: score is confidence-scaled by min(1, n_test / CONFIDENCE_N)
  2. Fat-Tail MSE: Uses Spearman rank correlation (rank-based, outlier-immune)
  3. Model Mismatch: Uses SGDRegressor + StandardScaler — exact backtest mirror

Usage:
    python3 src_py/optuna_regression_sweep.py \
        --ticker NVDA --target reg_clop --trials 100
"""

import argparse
import sys
import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from scipy.stats import spearmanr

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append(str(Path(__file__).parent.absolute()))
from silence_optimized_sweep import compute_trailing_adv, classify_and_filter

# ─────────────────────────────────────────────────────────────────────────
# ANTI-SPARSITY CONSTANTS
# ─────────────────────────────────────────────────────────────────────────
MIN_TOTAL_BURSTS = 200    # Hard floor: prune if fewer total bursts survive
CONFIDENCE_N     = 500    # Test set size for full confidence (no penalty)

# ─────────────────────────────────────────────────────────────────────────
# FEATURE COLUMNS — exact mirror of online_sgd_backtest.py
# ─────────────────────────────────────────────────────────────────────────
BASE_FEATURE_COLS = [
    'Direction', 'BurstVolume', 'TradeCount', 'Duration',
    'PeakImpact', 'D_b', 'AvgTradeSize', 'PriceChange',
]
EXTENDED_FEATURE_COLS = BASE_FEATURE_COLS + [
    'TimeOfDay', 'LogVolume', 'LogPeakImpact', 'ImpactPerShare',
    'RecentBurstCount', 'RecentBurstVol',
    'Dir_x_Volume', 'Dir_x_Impact', 'Dir_x_Db',
    'Volume_x_Impact', 'Volume_x_Duration',
    'Impact_x_Db', 'Impact_x_TradeCount',
    'AvgSize_x_Impact', 'AvgSize_x_Db',
    'ImpactPerTrade', 'VolumePerSec',
    'DbSquared', 'ImpactSquared',
    'Volume_qrank', 'Impact_qrank', 'Db_qrank',
    'RecentBurstCountOpp', 'RecentBurstVolOpp',
    'NetRecentFlow', 'BurstDensity5m',
    'TimeOfDaySin', 'TimeOfDayCos', 'IsOpen15', 'IsClose15', 'HourOfDay',
    'PriceLevel', 'VolPerDollar',
    # ── VWAP/TWAP Fingerprinting ──
    'TradeSizeVariance', 'RoundLotPct', 'LogTradeSizeVariance',
    # ── Hawkes Process ──
    'HawkesPeakIntensity', 'LogHawkesIntensity',
    # ── Pre-Burst Quote Depletion ──
    'PreBurstCancelRate',
    # ── Cross-path interactions ──
    'Variance_x_Volume', 'CancelRate_x_Impact', 'Hawkes_x_Volume',
]

# Regression targets
TARGET_MAP = {
    'reg_close': 'Perm_tCLOSE',
    'reg_clop':  'Perm_CLOP',
    'reg_clcl':  'Perm_CLCL',
}

# ─────────────────────────────────────────────────────────────────────────
# CACHES
# ─────────────────────────────────────────────────────────────────────────
df_cache = {}
adv_cache = {}


def get_features(df):
    """Extract feature matrix — same columns as backtest."""
    feat_available = [c for c in EXTENDED_FEATURE_COLS if c in df.columns]
    X = df[feat_available].copy()
    X.fillna(0, inplace=True)
    return X.values, feat_available


def objective(trial, ticker, target_key, fixed_hawkes_tag):
    """
    Optuna objective: confidence-scaled Spearman correlation using SGDRegressor.
    
    Anti-Sparsity Defense:
        score = spearman_rho * min(1.0, n_test / CONFIDENCE_N)
        If Optuna tries to shrink the dataset to 3 bursts for a "perfect" correlation,
        the confidence factor (3/500 = 0.006) annihilates the score.
    """
    # ── 1. Suggest Physical Parameters (same search space as classification sweep) ──
    hawkes_tag = fixed_hawkes_tag
    vol_frac   = trial.suggest_float("vol_frac", 0.00001, 0.005, log=True)
    dir_thresh = trial.suggest_float("dir_thresh", 0.5, 0.95)
    vol_ratio  = trial.suggest_float("vol_ratio", 0.01, 0.6)
    kappa      = trial.suggest_float("kappa", 0.0, 2.0)

    # ── 2. Load cached data & apply filters ──
    base_df    = df_cache[hawkes_tag]
    adv_series = adv_cache[hawkes_tag]
    burst_adv  = base_df["Date"].map(adv_series)
    min_vol_per_burst = (vol_frac * burst_adv).reindex(base_df.index)

    filtered = classify_and_filter(
        base_df, min_vol=0, dir_thresh=dir_thresh,
        vol_ratio=vol_ratio, kappa=kappa,
        require_directional=False, min_vol_per_burst=min_vol_per_burst
    )

    # ── TRAP 1 DEFENSE: hard floor on dataset size ──
    if len(filtered) < MIN_TOTAL_BURSTS:
        raise optuna.exceptions.TrialPruned()

    filtered = filtered.reset_index(drop=True)

    # ── 3. Build regression target ──
    col = TARGET_MAP[target_key]
    if col not in filtered.columns:
        raise optuna.exceptions.TrialPruned()

    y = filtered[col].values.copy()
    # Winsorize at 1st/99th percentile to reduce outlier influence on training
    lo, hi = np.nanpercentile(y, [1, 99])
    y = np.clip(y, lo, hi)

    # Drop rows where target is NaN
    valid_mask = ~np.isnan(y)
    if valid_mask.sum() < MIN_TOTAL_BURSTS:
        raise optuna.exceptions.TrialPruned()

    # ── 4. Get features ──
    X, feat_names = get_features(filtered)
    X = X[valid_mask]
    y = y[valid_mask]

    # ── 5. 70/30 chronological split ──
    filtered_valid = filtered.loc[valid_mask].copy()
    filtered_valid['Month'] = pd.to_datetime(
        filtered_valid['Date'].astype(str)
    ).dt.to_period('M')
    months = sorted(filtered_valid['Month'].unique())

    if len(months) < 4:
        raise optuna.exceptions.TrialPruned()

    split_idx    = int(len(months) * 0.7)
    train_months = months[:split_idx]
    test_months  = months[split_idx:]

    month_arr   = filtered_valid['Month'].values
    train_mask  = np.isin(month_arr, train_months)
    test_mask   = np.isin(month_arr, test_months)

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask],  y[test_mask]

    if len(X_te) < 30 or len(X_tr) < 50:
        raise optuna.exceptions.TrialPruned()

    # ── 6. Clean NaN/Inf ──
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

    # ── 7. StandardScaler (exact backtest mirror) ──
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # ── 8. SGDRegressor (exact backtest mirror — Trap 3 fix) ──
    model = SGDRegressor(
        loss='huber',             # Same as backtest
        epsilon=1.35,             # Same as backtest
        penalty='l2',             # Same as backtest
        alpha=0.001,              # Same as backtest
        learning_rate='adaptive', # Same as backtest
        eta0=0.001,               # Same as backtest
        max_iter=1000,            # Same as backtest
        random_state=42           # Same as backtest
    )
    model.fit(X_tr, y_tr)

    # ── 9. Predict on test set ──
    y_pred = model.predict(X_te)

    # ── 10. Spearman rank correlation (Trap 2 defense — immune to fat tails) ──
    rho, pval = spearmanr(y_te, y_pred)
    if np.isnan(rho):
        raise optuna.exceptions.TrialPruned()

    # ── 11. Confidence scaling (Trap 1 defense — anti-sparsity) ──
    confidence = min(1.0, len(X_te) / CONFIDENCE_N)
    score = rho * confidence

    # Store diagnostics
    trial.set_user_attr("raw_spearman", float(rho))
    trial.set_user_attr("p_value", float(pval))
    trial.set_user_attr("n_test", int(len(X_te)))
    trial.set_user_attr("n_train", int(len(X_tr)))
    trial.set_user_attr("n_total", int(len(filtered)))
    trial.set_user_attr("confidence", float(confidence))

    return score


def main():
    parser = argparse.ArgumentParser(
        description="Regression-based Optuna sweep (SGDRegressor mirror)"
    )
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--target", required=True,
                        help="Regression target: reg_close, reg_clop, reg_clcl")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--hawkes-tag", default="b1p0_i0p5",
                        help="Fixed Hawkes tag to evaluate")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    args = parser.parse_args()

    if args.target not in TARGET_MAP:
        print(f"ERROR: target must be one of {list(TARGET_MAP.keys())}")
        sys.exit(1)

    print(f"========== OPTUNA REGRESSION SWEEP ==========")
    print(f"Ticker:     {args.ticker}")
    print(f"Target:     {args.target}")
    print(f"Hawkes Tag: {args.hawkes_tag}")
    print(f"Trials:     {args.trials}")
    print(f"Dates:      {args.start_date} -> {args.end_date}")
    print(f"Model:      SGDRegressor(huber, l2, alpha=0.001) [exact backtest mirror]")
    print(f"Metric:     Spearman ρ × confidence(n_test/{CONFIDENCE_N})")
    print(f"Anti-Sparsity Floor: {MIN_TOTAL_BURSTS} minimum bursts")
    print(f"=============================================\n")

    start_ts = pd.to_datetime(args.start_date)
    end_ts   = pd.to_datetime(args.end_date)

    # Preload cache for the specified hawkes tag
    tag  = args.hawkes_tag
    path = f"results/{args.ticker}_params/bursts_{args.ticker}_{tag}_filtered.csv"

    if not os.path.exists(path):
        # Fall back to baseline unfiltered
        path = f"results/bursts_{args.ticker}_baseline_unfiltered.csv"

    if not os.path.exists(path):
        print(f"ERROR: Cannot find data at {path}")
        sys.exit(1)

    print(f"Loading data from {path}...")

    cols = pd.read_csv(path, nrows=0).columns
    float_cols = [c for c in cols if c not in ('Date', 'Time', 'Ticker')]
    dtype_dict = {c: 'float32' for c in float_cols}

    df = pd.read_csv(path, dtype=dtype_dict, low_memory=True)
    
    # Handle both integer (20230106) and string (2023-01-06) date formats
    try:
        df['Date'] = df['Date'].astype(int)
    except (ValueError, TypeError):
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y%m%d').astype(int)

    # Date filter
    date_ints = df['Date'].values
    start_int = int(start_ts.strftime("%Y%m%d"))
    end_int   = int(end_ts.strftime("%Y%m%d"))
    df = df[(date_ints >= start_int) & (date_ints <= end_int)].copy()

    print(f"Loaded {len(df):,} bursts in date range.")

    # Compute trailing ADV
    adv_series = compute_trailing_adv(df, window=14, stock_folder=f"data/{args.ticker}")

    df_cache[tag]  = df
    adv_cache[tag] = adv_series

    # Run Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, args.ticker, args.target, tag),
        n_trials=args.trials,
        show_progress_bar=True,
    )

    # ── Report Results ──
    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"BEST TRIAL (#{best.number})")
    print(f"{'='*60}")
    print(f"  Confidence-Scaled Spearman: {best.value:.6f}")
    print(f"  Raw Spearman ρ:             {best.user_attrs.get('raw_spearman', 'N/A'):.6f}")
    print(f"  P-value:                    {best.user_attrs.get('p_value', 'N/A'):.6f}")
    print(f"  Test samples:               {best.user_attrs.get('n_test', 'N/A')}")
    print(f"  Train samples:              {best.user_attrs.get('n_train', 'N/A')}")
    print(f"  Total filtered bursts:      {best.user_attrs.get('n_total', 'N/A')}")
    print(f"  Confidence factor:          {best.user_attrs.get('confidence', 'N/A'):.4f}")
    print(f"  Parameters:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # Save best params
    out_dir = f"results/optuna_regression/{args.ticker}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/best_regression_params_{args.target}_{tag}.json"

    result = {
        "ticker": args.ticker,
        "target": args.target,
        "hawkes_tag": tag,
        "score": best.value,
        "raw_spearman": best.user_attrs.get("raw_spearman"),
        "p_value": best.user_attrs.get("p_value"),
        "n_test": best.user_attrs.get("n_test"),
        "n_train": best.user_attrs.get("n_train"),
        "n_total": best.user_attrs.get("n_total"),
        "confidence": best.user_attrs.get("confidence"),
    }
    result.update(best.params)

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to {out_path}")

    # Print top 5 trials
    print(f"\nTop 5 Trials:")
    print(f"{'Trial':>6} {'Score':>10} {'Raw ρ':>10} {'n_test':>8} {'vol_frac':>12} {'dir':>8} {'vr':>8} {'κ':>8}")
    for t in sorted(study.trials, key=lambda t: t.value if t.value is not None else -999, reverse=True)[:5]:
        if t.value is None:
            continue
        print(f"{t.number:>6} {t.value:>10.6f} {t.user_attrs.get('raw_spearman',0):>10.6f} "
              f"{t.user_attrs.get('n_test',0):>8} {t.params.get('vol_frac',0):>12.8f} "
              f"{t.params.get('dir_thresh',0):>8.4f} {t.params.get('vol_ratio',0):>8.4f} "
              f"{t.params.get('kappa',0):>8.4f}")


if __name__ == "__main__":
    main()
