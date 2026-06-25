#!/usr/bin/env python3
"""
ablation_study.py — Feature Ablation: Direction Dominance Test

The main referee report criticized the dominance of the "Direction" feature.
This script runs the SGDRegressor pipeline with and without the Direction
feature (and all Direction-interaction terms) to prove that other engineered
features carry independent predictive weight.

Outputs:
  - Full model Spearman ρ, PPT, and Sharpe
  - Ablated model (no Direction) Spearman ρ, PPT, and Sharpe
  - Feature importance ranking for both models

Usage:
    python3 src_py/ablation_study.py results/bursts_NVDA_baseline_unfiltered.csv --ticker NVDA
"""

import argparse
import sys
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append(str(Path(__file__).parent.absolute()))
from silence_optimized_sweep import compute_trailing_adv, classify_and_filter

# All features (mirrors online_sgd_backtest.py)
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
    'TradeSizeVariance', 'RoundLotPct', 'LogTradeSizeVariance',
    'HawkesPeakIntensity', 'LogHawkesIntensity',
    'PreBurstCancelRate',
    'Variance_x_Volume', 'CancelRate_x_Impact', 'Hawkes_x_Volume',
]

# Features that directly involve Direction
DIRECTION_FEATURES = {
    'Direction',
    'Dir_x_Volume', 'Dir_x_Impact', 'Dir_x_Db',
}

TARGET_MAP = {
    'reg_clop': 'Perm_CLOP',
    'reg_clcl': 'Perm_CLCL',
    'reg_close': 'Perm_tCLOSE',
}


def _spearman_score(y_true, y_pred):
    """Scorer used for permutation importance: higher = better."""
    rho, _ = spearmanr(y_true, y_pred)
    return rho if np.isfinite(rho) else 0.0


def run_model(X_train, y_train, X_test, y_test, feat_names, label,
              compute_perm_importance=False):
    """Train SGDRegressor and evaluate.

    When ``compute_perm_importance`` is True, also compute sklearn
    permutation importance on the held-out test fold (Reviewer Rec #2),
    scored by Spearman ρ — a model-agnostic complement to the raw
    coefficient magnitude (which can mislead under correlated features).
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    model = SGDRegressor(
        loss='huber', epsilon=1.35, penalty='l2',
        alpha=0.001, learning_rate='adaptive', eta0=0.001,
        max_iter=1000, random_state=42
    )
    model.fit(X_tr, y_train)

    y_pred = model.predict(X_te)

    rho, pval = spearmanr(y_test, y_pred)

    # Feature importance (coefficient magnitude)
    coef_importance = np.abs(model.coef_)
    sorted_idx = np.argsort(coef_importance)[::-1]

    print(f"\n  {'='*60}")
    print(f"  MODEL: {label}")
    print(f"  {'='*60}")
    print(f"  Features:              {len(feat_names)}")
    print(f"  Train samples:         {len(X_train):,}")
    print(f"  Test samples:          {len(X_test):,}")
    print(f"  Spearman ρ:            {rho:.6f}")
    print(f"  P-value:               {pval:.2e}")

    # BPS markout from predictions
    pred_correct = np.sign(y_pred) == np.sign(y_test)
    accuracy = pred_correct.mean() * 100.0
    print(f"  Sign accuracy:         {accuracy:.1f}%")

    # Top 10 features by coefficient magnitude
    print(f"\n  Top {min(10, len(feat_names))} Features (|coef| after scaling):")
    for rank, idx in enumerate(sorted_idx[:10], 1):
        name = feat_names[idx] if idx < len(feat_names) else f"feat_{idx}"
        print(f"    {rank:2d}. {name:<30}  |coef|={coef_importance[idx]:.6f}")

    perm_ranking = None
    if compute_perm_importance and len(feat_names) > 1:
        # Permutation importance against the Spearman scorer on the test fold.
        from sklearn.metrics import make_scorer
        scorer = make_scorer(_spearman_score)
        try:
            perm = permutation_importance(
                model, X_te, y_test, scoring=scorer,
                n_repeats=10, random_state=42, n_jobs=1,
            )
            perm_idx = np.argsort(perm.importances_mean)[::-1]
            print(f"\n  Top 10 Features (permutation importance, Δ Spearman ρ):")
            for rank, idx in enumerate(perm_idx[:10], 1):
                name = feat_names[idx] if idx < len(feat_names) else f"feat_{idx}"
                print(f"    {rank:2d}. {name:<30}  "
                      f"Δρ={perm.importances_mean[idx]:+.6f} ± {perm.importances_std[idx]:.6f}")
            perm_ranking = [
                (feat_names[i], float(perm.importances_mean[i]),
                 float(perm.importances_std[i]))
                for i in perm_idx[:10]
            ]
        except Exception as exc:  # noqa: BLE001
            print(f"  (permutation importance skipped: {exc})")

    return {
        "label": label,
        "n_features": len(feat_names),
        "spearman_rho": rho,
        "p_value": pval,
        "sign_accuracy": accuracy,
        "top_features": [(feat_names[i], coef_importance[i]) for i in sorted_idx[:10]],
        "perm_importance": perm_ranking,
    }


def main():
    ap = argparse.ArgumentParser(description="Direction feature ablation study")
    ap.add_argument("data_csv", help="Path to bursts CSV (unfiltered or filtered)")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--target", default="reg_clop", choices=list(TARGET_MAP.keys()))
    ap.add_argument("--vol-frac", type=float, default=0.001)
    ap.add_argument("--dir-thresh", type=float, default=0.7)
    ap.add_argument("--vol-ratio", type=float, default=0.4)
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--start-date", default="2023-01-01")
    ap.add_argument("--end-date", default="2024-12-31")
    args = ap.parse_args()

    print(f"{'='*70}")
    print(f"  ABLATION STUDY: Direction Feature Dominance Test")
    print(f"  Ticker: {args.ticker}  |  Target: {args.target}")
    print(f"  Date Range: {args.start_date} → {args.end_date}")
    print(f"{'='*70}")

    # ── Load data ──
    df = pd.read_csv(args.data_csv)
    try:
        df["Date"] = df["Date"].astype(int)
    except (ValueError, TypeError):
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y%m%d").astype(int)

    start_int = int(pd.to_datetime(args.start_date).strftime("%Y%m%d"))
    end_int = int(pd.to_datetime(args.end_date).strftime("%Y%m%d"))
    df = df[(df["Date"] >= start_int) & (df["Date"] <= end_int)].copy()

    print(f"  Loaded {len(df):,} bursts in date range")

    # ── Apply physical filters ──
    adv_series = compute_trailing_adv(df, window=14, stock_folder=f"data/{args.ticker}")
    burst_adv = df["Date"].map(adv_series)
    min_vol_per_burst = (args.vol_frac * burst_adv).reindex(df.index)

    filtered = classify_and_filter(
        df, min_vol=0, dir_thresh=args.dir_thresh,
        vol_ratio=args.vol_ratio, kappa=args.kappa,
        require_directional=False, min_vol_per_burst=min_vol_per_burst
    ).reset_index(drop=True)

    print(f"  After filtering: {len(filtered):,} bursts")

    # ── Build target ──
    target_col = TARGET_MAP[args.target]
    if target_col not in filtered.columns:
        print(f"ERROR: Target column '{target_col}' not found in data.")
        sys.exit(1)

    y = filtered[target_col].values.copy()
    lo, hi = np.nanpercentile(y, [1, 99])
    y = np.clip(y, lo, hi)

    valid = ~np.isnan(y)
    filtered = filtered[valid].reset_index(drop=True)
    y = y[valid]

    if len(y) < 200:
        print(f"ERROR: Only {len(y)} valid samples. Need at least 200.")
        sys.exit(1)

    # ── Chronological 70/30 split ──
    filtered["Month"] = pd.to_datetime(filtered["Date"].astype(str)).dt.to_period("M")
    months = sorted(filtered["Month"].unique())
    split_idx = int(len(months) * 0.7)
    train_months = months[:split_idx]
    test_months = months[split_idx:]

    month_arr = filtered["Month"].values
    train_mask = np.isin(month_arr, train_months)
    test_mask = np.isin(month_arr, test_months)

    # ── Build feature matrices ──
    # FULL model: all features
    full_feats = [c for c in EXTENDED_FEATURE_COLS if c in filtered.columns]
    X_full = filtered[full_feats].fillna(0).values

    # ABLATED model: remove Direction and interaction terms
    ablated_feats = [c for c in full_feats if c not in DIRECTION_FEATURES]
    X_ablated = filtered[ablated_feats].fillna(0).values

    print(f"\n  Full model features:     {len(full_feats)}")
    print(f"  Ablated model features:  {len(ablated_feats)}")
    print(f"  Removed features:        {sorted(DIRECTION_FEATURES & set(full_feats))}")
    print(f"  Train months:            {len(train_months)}")
    print(f"  Test months:             {len(test_months)}")

    # ── Direction-only model (Reviewer M2 / Rec #2) ──
    # The single most important ablation the referee asked for: if a
    # one-feature Direction model recovers most of the full-model ρ, the
    # contribution simplifies to "filtered burst direction is a tradable
    # overnight signal" — a cleaner (still publishable) framing.
    if "Direction" in full_feats:
        X_dironly = filtered[["Direction"]].fillna(0).values
    else:
        X_dironly = np.zeros((len(filtered), 1))

    # ── Run the three models ──
    result_full = run_model(
        X_full[train_mask], y[train_mask],
        X_full[test_mask], y[test_mask],
        full_feats, "FULL MODEL (All Features)",
        compute_perm_importance=True,
    )

    result_ablated = run_model(
        X_ablated[train_mask], y[train_mask],
        X_ablated[test_mask], y[test_mask],
        ablated_feats, "ABLATED MODEL (No Direction)"
    )

    result_dironly = run_model(
        X_dironly[train_mask], y[train_mask],
        X_dironly[test_mask], y[test_mask],
        ["Direction"], "DIRECTION-ONLY MODEL (1 Feature)"
    )

    # ── Summary comparison ──
    print(f"\n{'='*90}")
    print(f"  ABLATION SUMMARY")
    print(f"{'='*90}")
    print(f"  {'Metric':<24} {'Full':<20} {'No Direction':<20} {'Direction-only':<20}")
    print(f"  {'-'*90}")

    full_rho = result_full["spearman_rho"]
    abl_rho = result_ablated["spearman_rho"]
    dir_rho = result_dironly["spearman_rho"]
    rho_drop = full_rho - abl_rho
    rho_pct_drop = (rho_drop / abs(full_rho) * 100.0) if full_rho != 0 else 0.0
    # Fraction of the full-model ρ that Direction alone recovers.
    dir_share = (dir_rho / full_rho * 100.0) if full_rho != 0 else 0.0

    print(f"  {'Spearman ρ':<24} {full_rho:<20.6f} {abl_rho:<20.6f} {dir_rho:<20.6f}")
    print(f"  {'Sign Accuracy (%)':<24} {result_full['sign_accuracy']:<20.1f} "
          f"{result_ablated['sign_accuracy']:<20.1f} {result_dironly['sign_accuracy']:<20.1f}")
    print(f"  {'P-value':<24} {result_full['p_value']:<20.2e} "
          f"{result_ablated['p_value']:<20.2e} {result_dironly['p_value']:<20.2e}")
    print(f"  {'# Features':<24} {result_full['n_features']:<20d} "
          f"{result_ablated['n_features']:<20d} {result_dironly['n_features']:<20d}")

    print(f"\n  Direction removal impact on ρ:    {rho_drop:+.6f} ({rho_pct_drop:+.1f}%)")
    print(f"  Direction-only recovers:          {dir_share:.1f}% of full-model ρ")

    if abs(abl_rho) > 0 and result_ablated["p_value"] < 0.05:
        print(f"  → Non-direction features carry INDEPENDENT predictive weight (p < 0.05)")
    elif abs(abl_rho) > 0:
        print(f"  → Non-direction features show signal but not statistically significant")
    else:
        print(f"  → Direction appears to be the sole predictive feature")

    if abs(dir_share) >= 80.0:
        print(f"  → Direction-only explains ≥80% of ρ: framing simplifies to "
              f"'filtered burst direction is a tradable overnight signal' (M2)")
    else:
        print(f"  → Direction-only explains <80% of ρ: high-dimensional framing is "
              f"supported — engineered features add material predictive content")

    print(f"{'='*90}")


if __name__ == "__main__":
    main()
