#!/usr/bin/env python3
import argparse
import sys
import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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
    'reg_clop': 'Perm_CLOP',
    'reg_clcl': 'Perm_CLCL',
    'reg_close': 'Perm_tCLOSE'
}

def get_features(df):
    feat_available = [c for c in PASSIVE_FEATURE_COLS if c in df.columns]
    X = df[feat_available].copy()
    X.fillna(0, inplace=True)
    return X.values, feat_available

def compute_trailing_adv(df, window=14):
    daily_vol = df.groupby('Date')['Volume'].sum()
    adv = daily_vol.rolling(window=window, min_periods=1).mean()
    return adv

def classify_and_filter(df, vol_frac, dir_thresh, vol_ratio, max_cancel_ratio, burst_adv):
    min_vol_per_burst = vol_frac * burst_adv
    mask = df['Volume'] >= min_vol_per_burst
    dir_mask = (df['BidRatio'] >= dir_thresh) | (df['AskRatio'] >= dir_thresh)
    mask = mask & dir_mask
    mask = mask & (df['MinMaxVolRatio'] <= vol_ratio)
    if 'CancelRatio' in df.columns:
        mask = mask & (df['CancelRatio'] <= max_cancel_ratio)
    return df[mask].copy()

def train_eval_sgd(X_tr, y_tr, X_te, y_te):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    model = SGDRegressor(
        loss='huber', epsilon=1.35, penalty='l2', alpha=0.001,
        learning_rate='adaptive', eta0=0.001, max_iter=1000, random_state=42
    )
    model.fit(X_tr_s, y_tr)
    y_pred = model.predict(X_te_s)
    
    return y_pred, model, scaler

def walk_forward_simulation(df, target_col):
    df['Month'] = pd.to_datetime(df['Date'].astype(str)).dt.to_period('M')
    months = sorted(df['Month'].unique())
    
    if len(months) < 3:
        print("Not enough months for walk-forward validation.")
        return None
        
    all_preds = []
    
    for i in range(2, len(months)):
        test_month = months[i]
        train_months = months[:i]
        
        train_mask = df['Month'].isin(train_months)
        test_mask = df['Month'] == test_month
        
        train_df = df[train_mask]
        test_df = df[test_mask]
        
        if len(train_df) < 50 or len(test_df) < 10:
            all_preds.extend([0] * len(test_df))
            continue
            
        y_tr = train_df[target_col].values
        y_te = test_df[target_col].values
        
        X_tr, _ = get_features(train_df)
        X_te, _ = get_features(test_df)
        
        y_pred, model, scaler = train_eval_sgd(X_tr, y_tr, X_te, y_te)
        all_preds.extend(y_pred)
        
    df_eval = df[df['Month'] >= months[2]].copy()
    df_eval['Pred'] = all_preds
    
    return df_eval

def run_backtest(df_eval, target_col):
    df_eval = df_eval[df_eval['Pred'] != 0].copy()
    
    p25 = df_eval['Pred'].quantile(0.25)
    p75 = df_eval['Pred'].quantile(0.75)
    
    # Enter if prediction is extreme
    enter_mask = (df_eval['Pred'] <= p25) | (df_eval['Pred'] >= p75)
    df_trades = df_eval[enter_mask].copy()
    
    # For passive bursts, the return target is the price drift.
    # The actual return we capture is the sign of the prediction * target return * direction.
    # Wait, the target is arcsinh(side * return * 10000). So return = sinh(target) / (side * 10000).
    # Since we trade based on the prediction of the target:
    # If pred > 0, we bet the target will be positive.
    # Target is already signed by the burst direction.
    
    # Let's simplify: the actual return we capture is proportional to:
    # sign(Pred) * Target
    # Because if Pred > 0, we take the side of the burst.
    # Subtract 3.0 bps round-trip MOC/MOO cost
    cost_bps = 3.0
    gate = 3.0 if target_col in ['Perm_CLOP', 'reg_clop'] else 1.0
    
    # Cost-Aware Gate: Daily net directional imbalance must exceed Expected friction
    df_trades = df_trades[df_trades['Pred'].abs() > gate].copy()
    if df_trades.empty:
        print("--- Passive Backtest Results ---")
        print("No trades triggered the cost-aware gate.")
        return

    df_trades['Capture'] = np.sign(df_trades['Pred']) * df_trades[target_col] - cost_bps
    
    print(f"--- Passive Backtest Results ---")
    print(f"Total entries: {len(df_trades)} out of {len(df_eval)} evaluated bursts")
    print(f"Mean Capture (Target Units): {df_trades['Capture'].mean():.4f}")
    print(f"Sum Capture: {df_trades['Capture'].sum():.4f}")
    print(f"Win Rate: {(df_trades['Capture'] > 0).mean():.4f}")
    
    sharpe = df_trades['Capture'].mean() / df_trades['Capture'].std() * np.sqrt(252 * (len(df_trades)/len(df_eval['Date'].unique())))
    print(f"Annualized Sharpe: {sharpe:.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--params-file", required=True)
    args = parser.parse_args()
    
    path = f"results/passive/passive_bursts_{args.ticker}_raw_filtered.csv"
    if not os.path.exists(path):
        print(f"ERROR: Cannot find data at {path}")
        sys.exit(1)
        
    with open(args.params_file, 'r') as f:
        params = json.load(f)
        
    df = pd.read_csv(path, low_memory=True)
    df['Date'] = df['Date'].astype(str).str.replace('-', '').astype(int)
    
    adv_series = compute_trailing_adv(df, window=14)
    burst_adv = df["Date"].map(adv_series)
    
    filtered = classify_and_filter(
        df, 
        params.get('passive_adv_multiplier', params.get('vol_frac', 1e-6)), 
        params.get('dir_thresh', 0.6), 
        params.get('vol_ratio', 0.5), 
        params.get('max_cancel_ratio', 0.8), 
        burst_adv
    )
    
    print(f"Filtered to {len(filtered)} passive bursts for {args.ticker}.")
    
    col = TARGET_MAP[args.target]
    filtered = filtered.dropna(subset=[col])
    
    df_eval = walk_forward_simulation(filtered, col)
    if df_eval is not None and not df_eval.empty:
        run_backtest(df_eval, col)

if __name__ == "__main__":
    main()
