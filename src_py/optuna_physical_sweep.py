#!/usr/bin/env python3
"""
optuna_physical_sweep.py

Uses Optuna to find the best physical burst-definition parameters:
- hawkes_tag (b1p0_i0p3, b1p0_i0p5, b1p0_i0p8)
- min_vol_frac (fraction of 14d trailing ADV)
- dir_thresh (directional trade threshold)
- vol_ratio (volume ratio max allowable)

For a given stock and target, this preloads the precomputed unfiltered CSVs in memory,
then slices them instantly during trials, applies walk-forward validation with a
Random Forest classifier (captures non-linear VWAP/TWAP signals), and returns
out-of-sample AUC to Optuna.

Usage:
    python3 src_py/optuna_physical_sweep.py \
        --ticker NVDA --target cls_close --trials 100
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Allow importing local modules
sys.path.append(str(Path(__file__).parent.absolute()))

from silence_optimized_sweep import compute_trailing_adv, classify_and_filter
from train_model_zoo import build_target, TARGET_MAP

# ─────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING LOGIC (Extracted from train_model_zoo.py)
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
    # ── Path 1: VWAP/TWAP Fingerprinting ──
    'TradeSizeVariance', 'RoundLotPct', 'LogTradeSizeVariance',
    # ── Path 2: Hawkes Process ──
    'HawkesPeakIntensity', 'LogHawkesIntensity',
    # ── Path 3: Pre-Burst Quote Depletion ──
    'PreBurstCancelRate',
    # ── Cross-path interactions ──
    'Variance_x_Volume', 'CancelRate_x_Impact', 'Hawkes_x_Volume',
]
DB_TAINTED_FEATURES = {
    'D_b', 'Dir_x_Db', 'Impact_x_Db', 'AvgSize_x_Db', 'DbSquared', 'Db_qrank',
}

# Targets whose horizon is ≤ 10m.  Since we removed short-horizon targets,
# this set is now empty.  Keep the structure so older code paths don't break.
DB_LEAKY_TARGETS = set()

def _time_ordered_train_val_split(df, min_train_months=3):
    months = sorted(df['Month'].unique())
    splits = []
    
    # Needs a minimum number of months to even split
    if len(months) < min_train_months + 1:
        return splits
        
    for i in range(min_train_months, len(months)):
        train_months = months[:i]
        test_month = months[i]
        
        train_mask = df['Month'].isin(train_months)
        test_mask = df['Month'] == test_month
        
        # Require absolute minimum counts
        if train_mask.sum() < 50 or test_mask.sum() < 5:
            continue
            
        splits.append((test_month, train_mask, test_mask))
    return splits


# ─────────────────────────────────────────────────────────────────────────
# CACHES
# ─────────────────────────────────────────────────────────────────────────
df_cache = {}
adv_cache = {}

def get_features(df, target_key):
    feat_cols = list(EXTENDED_FEATURE_COLS)
    if target_key in DB_LEAKY_TARGETS:
        feat_cols = [c for c in feat_cols if c not in DB_TAINTED_FEATURES]
    feat_available = [c for c in feat_cols if c in df.columns]
    
    X = df[feat_available].copy()
    X.fillna(0, inplace=True)
    return X, feat_available


def objective(trial, ticker, target_key, fixed_hawkes_tag, min_rows_thresh):
    # ── 1. Suggest Physical Parameters ──
    hawkes_tag = fixed_hawkes_tag
    
    # 0.00001 (0.001%) to 0.005 (0.5%) of 14d trailing ADV
    vol_frac = trial.suggest_float("vol_frac", 0.00001, 0.005, log=True)
    
    # 0.5 to 0.95
    dir_thresh = trial.suggest_float("dir_thresh", 0.5, 0.95)
    
    # 0.01 to 0.50
    vol_ratio = trial.suggest_float("vol_ratio", 0.01, 0.6)
    
    kappa = 0.0
    if target_key not in DB_LEAKY_TARGETS:
        kappa = trial.suggest_float("kappa", 0.0, 2.0)
        
    # ── 2. Load cached data & ADV ──
    base_df = df_cache[hawkes_tag]
    adv_series = adv_cache[hawkes_tag]
    
    burst_adv = base_df["Date"].map(adv_series)
    min_vol_per_burst = (vol_frac * burst_adv).reindex(base_df.index)
    
    # ── 3. Apply post-filters ──
    filtered = classify_and_filter(
        base_df,
        min_vol=0,
        dir_thresh=dir_thresh,
        vol_ratio=vol_ratio,
        kappa=kappa,
        require_directional=False,
        min_vol_per_burst=min_vol_per_burst
    )
    
    if len(filtered) < min_rows_thresh:
        raise optuna.exceptions.TrialPruned()
        
    filtered = filtered.reset_index(drop=True)
        
    # ── 4. Build Targets and Features ──
    try:
        y, task_type, meta = build_target(filtered, target_key)
        X, feat_available = get_features(filtered, target_key)
        
        if task_type != 'binary':
            raise ValueError("Only binary classification is supported for this sweep.")
            
    except Exception as e:
        raise optuna.exceptions.TrialPruned()
        
    if len(X) < min_rows_thresh:
        raise optuna.exceptions.TrialPruned()
        
    # ── 5. Generate Walk Forward Splits ──
    splits = _time_ordered_train_val_split(filtered, min_train_months=3)
    if not splits:
        raise optuna.exceptions.TrialPruned()
        
    # ── 6. Evaluate with RandomForestClassifier ──
    # Random Forest captures the non-linear VWAP/TWAP "variance == 0" signal
    # that linear models cannot learn.
    y_true_all = []
    y_pred_all = []
    
    X_vals = X.values
    y_vals = y if isinstance(y, np.ndarray) else y.values
    
    for month_label, train_mask, test_mask in splits:
        train_idx = train_mask.values if hasattr(train_mask, 'values') else train_mask
        test_idx = test_mask.values if hasattr(test_mask, 'values') else test_mask
        
        X_tr, y_tr = X_vals[train_idx], y_vals[train_idx]
        X_te, y_te = X_vals[test_idx], y_vals[test_idx]
        
        if len(np.unique(y_tr)) < 2:
            continue

        # Replace NaN/Inf in features with 0
        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
        X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

        # RandomForest — non-linear, captures TWAP variance=0 splits natively
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_leaf=50,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr)
        
        y_pred = model.predict_proba(X_te)[:, 1]
        
        y_true_all.extend(y_te)
        y_pred_all.extend(y_pred)
        
    if len(y_true_all) < min_rows_thresh or len(np.unique(y_true_all)) < 2:
        raise optuna.exceptions.TrialPruned()
        
    auc = roc_auc_score(y_true_all, y_pred_all)
    return auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--target", required=True,
                        help="Target key, e.g. cls_close, cls_clop, cls_clcl")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    args = parser.parse_args()
    
    print(f"========== OPTUNA PHYSICAL SWEEP ==========")
    print(f"Ticker: {args.ticker}")
    print(f"Target: {args.target}")
    print(f"Trials: {args.trials}")
    print(f"Date window: {args.start_date} -> {args.end_date}")
    print(f"Model: RandomForestClassifier (non-linear VWAP/TWAP capture)")
    print(f"===========================================\n")

    start_ts = pd.to_datetime(args.start_date)
    end_ts = pd.to_datetime(args.end_date)
    
    # Preload caches
    tags = ["b1p0_i0p3", "b1p0_i0p5", "b1p0_i0p8"]
    for tag in tags:
        path = f"results/{args.ticker}_params/bursts_{args.ticker}_{tag}_filtered.csv"
        
        if not os.path.exists(path):
            print(f"ERROR: Cannot find precomputed cache at {path}")
            sys.exit(1)
            
        print(f"Loading {tag} cached data from {path}...")
        
        # Read columns first to build dtype dict to prevent OOM DURING read_csv
        cols = pd.read_csv(path, nrows=0).columns
        float_cols = [c for c in cols if c not in ('Date', 'Time', 'Ticker')]
        dtypes = {c: 'float32' for c in float_cols}
        
        df = pd.read_csv(path, dtype=dtypes)
        
        # Safely fill NaNs in new behavioral columns
        for col in ('TradeSizeVariance', 'RoundLotPct', 'HawkesPeakIntensity', 'PreBurstCancelRate'):
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        df['DateCol'] = pd.to_datetime(df['Date'])
        df = df[(df['DateCol'] >= start_ts) & (df['DateCol'] <= end_ts)].copy()
        if df.empty:
            print(
                f"ERROR: No rows left after date filter for {tag} "
                f"in [{args.start_date}, {args.end_date}]"
            )
            sys.exit(1)
        df['Month'] = df['DateCol'].dt.strftime('%Y-%m')
        
        df_cache[tag] = df
        
        print(f"Computing 14d trailing ADV for {tag}...")
        adv_cache[tag] = compute_trailing_adv(df, window=14)
        
        print(f" -> {tag} loaded: {len(df):,} total unfiltered bursts.")
        
    print(f"\nStarting Bayesian Optimization with RandomForest evaluator...")
    
    outdir = Path(f"results/optuna_physical/{args.ticker}")
    outdir.mkdir(parents=True, exist_ok=True)
    
    for tag in tags:
        print(f"\n--- OPTIMIZING {args.target} exactly for {tag} ---")
        study = optuna.create_study(direction="maximize", study_name=f"{args.ticker}_phys_sweep_{tag}")
        obj = lambda trial: objective(trial, args.ticker, args.target, tag, min_rows_thresh=100)
        
        # Turn off progress bar so individual [I 2026...] trial strings go safely to .out logs
        study.optimize(obj, n_trials=args.trials, show_progress_bar=False)
        
        res = {
            "ticker": args.ticker,
            "target": args.target,
            "model": "RandomForestClassifier",
            "best_auc": float(study.best_value),
            "best_params": study.best_params
        }
        # Safely insert the fixed hawkes tag into the payload for the JSON parser
        res["best_params"]["hawkes_tag"] = tag
        
        out_file = outdir / f"best_physical_params_{args.target}_{tag}.json"
        with open(out_file, "w") as f:
            json.dump(res, f, indent=4)
            
        print(f"[{tag}] Best AUC: {study.best_value:.4f}")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")
            
    print(f"\nOptimization Complete for {args.ticker} -> {args.target} over all hawkes bounds!")

if __name__ == "__main__":
    main()
