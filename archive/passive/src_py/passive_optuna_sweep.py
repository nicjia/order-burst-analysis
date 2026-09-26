#!/usr/bin/env python3
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

MIN_TOTAL_BURSTS = 200    
CONFIDENCE_N     = 500    

PASSIVE_FEATURE_COLS = [
    'Volume', 'SubmissionCount', 'BidSubCount', 'AskSubCount',
    'BidSubVolume', 'AskSubVolume', 'BidRatio', 'AskRatio', 'MinMaxVolRatio',
    'PeakImpact', 'Spread', 'BidVolBest', 'AskVolBest',
    'BidDepth5', 'AskDepth5', 'BookImbalance', 'Volatility60s',
    'Momentum5s', 'Momentum30s', 'Momentum60s', 'TradeCount5m', 'TradeVolume5m',
    'SubmissionSizeVariance', 'RoundLotPct', 'HawkesPeakIntensity',
    'CancelCount', 'CancelVolume', 'BidCancelCount', 'AskCancelCount',
    'BidCancelVolume', 'AskCancelVolume', 'CancelRatio', 'PreBurstCancelRate'
]

TARGET_MAP = {
    'reg_close': 'Perm_tCLOSE',
    'reg_clop':  'Perm_CLOP',
    'reg_clcl':  'Perm_CLCL',
}

df_cache = {}
adv_cache = {}

def get_features(df):
    feat_available = [c for c in PASSIVE_FEATURE_COLS if c in df.columns]
    X = df[feat_available].copy()
    X.fillna(0, inplace=True)
    return X.values, feat_available

def classify_and_filter(df, vol_frac, dir_thresh, vol_ratio, max_cancel_ratio, burst_adv):
    min_vol_per_burst = vol_frac * burst_adv
    
    # Volume filter
    mask = df['Volume'] >= min_vol_per_burst
    
    # Direction filter
    # Either bid ratio or ask ratio > dir_thresh
    dir_mask = (df['BidRatio'] >= dir_thresh) | (df['AskRatio'] >= dir_thresh)
    mask = mask & dir_mask
    
    # Volume ratio filter
    mask = mask & (df['MinMaxVolRatio'] <= vol_ratio)
    
    # Quote Stability filter
    if 'CancelRatio' in df.columns:
        mask = mask & (df['CancelRatio'] <= max_cancel_ratio)
        
    return df[mask].copy()

def compute_trailing_adv(df, window=14):
    daily_vol = df.groupby('Date')['Volume'].sum()
    adv = daily_vol.rolling(window=window, min_periods=1).mean()
    return adv

def objective(trial, ticker, target_key):
    vol_frac = trial.suggest_float("vol_frac", 0.00001, 0.005, log=True)
    dir_thresh = trial.suggest_float("dir_thresh", 0.5, 0.95)
    vol_ratio = trial.suggest_float("vol_ratio", 0.01, 0.6)
    max_cancel_ratio = trial.suggest_float("max_cancel_ratio", 0.1, 0.9)
    
    base_df = df_cache[ticker]
    adv_series = adv_cache[ticker]
    burst_adv = base_df["Date"].map(adv_series)
    
    filtered = classify_and_filter(base_df, vol_frac, dir_thresh, vol_ratio, max_cancel_ratio, burst_adv)
    
    if len(filtered) < MIN_TOTAL_BURSTS:
        raise optuna.exceptions.TrialPruned()
        
    filtered = filtered.reset_index(drop=True)
    
    col = TARGET_MAP[target_key]
    if col not in filtered.columns:
        raise optuna.exceptions.TrialPruned()
        
    y = filtered[col].values.copy()
    lo, hi = np.nanpercentile(y, [1, 99])
    y = np.clip(y, lo, hi)
    
    valid_mask = ~np.isnan(y)
    if valid_mask.sum() < MIN_TOTAL_BURSTS:
        raise optuna.exceptions.TrialPruned()
        
    X, _ = get_features(filtered)
    X = X[valid_mask]
    y = y[valid_mask]
    
    filtered_valid = filtered.loc[valid_mask].copy()
    filtered_valid['Month'] = pd.to_datetime(filtered_valid['Date'].astype(str)).dt.to_period('M')
    months = sorted(filtered_valid['Month'].unique())
    
    if len(months) < 4:
        raise optuna.exceptions.TrialPruned()
        
    split_idx = int(len(months) * 0.7)
    train_months = months[:split_idx]
    test_months = months[split_idx:]
    
    month_arr = filtered_valid['Month'].values
    train_mask = np.isin(month_arr, train_months)
    test_mask = np.isin(month_arr, test_months)
    
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask], y[test_mask]
    
    if len(X_te) < 30 or len(X_tr) < 50:
        raise optuna.exceptions.TrialPruned()
        
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)
    
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    
    model = SGDRegressor(
        loss='huber', epsilon=1.35, penalty='l2', alpha=0.001,
        learning_rate='adaptive', eta0=0.001, max_iter=1000, random_state=42
    )
    model.fit(X_tr, y_tr)
    
    y_pred = model.predict(X_te)
    
    rho, pval = spearmanr(y_te, y_pred)
    if np.isnan(rho):
        raise optuna.exceptions.TrialPruned()
        
    confidence = min(1.0, len(X_te) / CONFIDENCE_N)
    score = rho * confidence
    
    trial.set_user_attr("raw_spearman", float(rho))
    trial.set_user_attr("n_test", int(len(X_te)))
    return score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()
    
    path = f"results/passive/passive_bursts_{args.ticker}_raw_filtered.csv"
    if not os.path.exists(path):
        print(f"ERROR: Cannot find data at {path}")
        sys.exit(1)
        
    df = pd.read_csv(path, low_memory=True)
    df['Date'] = df['Date'].astype(str).str.replace('-', '').astype(int)
    
    print(f"Loaded {len(df):,} passive bursts for {args.ticker}.")
    
    adv_series = compute_trailing_adv(df, window=14)
    df_cache[args.ticker] = df
    adv_cache[args.ticker] = adv_series
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, args.ticker, args.target), n_trials=args.trials)
    
    best = study.best_trial
    print(f"\nBEST TRIAL:")
    print(f"Score: {best.value:.6f}")
    print(f"Raw Spearman: {best.user_attrs.get('raw_spearman'):.6f}")
    print(f"Params: {best.params}")
    
    out_dir = f"results/optuna_passive"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/best_params_{args.ticker}_{args.target}.json"
    
    result = {"ticker": args.ticker, "target": args.target, "score": best.value}
    result.update(best.params)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
