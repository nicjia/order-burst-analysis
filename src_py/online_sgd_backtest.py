#!/usr/bin/env python3
import warnings
import argparse
import sys
import os
import collections
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# Ensure we can import local modules
sys.path.append(str(Path(__file__).parent.absolute()))
from silence_optimized_sweep import compute_trailing_adv, classify_and_filter
from train_model_zoo import build_target, TARGET_MAP

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
]
DB_TAINTED_FEATURES = {
    'D_b', 'Dir_x_Db', 'Impact_x_Db', 'AvgSize_x_Db', 'DbSquared', 'Db_qrank',
}
DB_LEAKY_TARGETS = {'cls_1m', 'cls_3m', 'cls_5m', 'cls_10m',
                    'reg_1m', 'reg_3m', 'reg_5m', 'reg_10m'}

def get_features(df, target_key):
    feat_cols = list(EXTENDED_FEATURE_COLS)
    if target_key in DB_LEAKY_TARGETS:
        feat_cols = [c for c in feat_cols if c not in DB_TAINTED_FEATURES]
    feat_available = [c for c in feat_cols if c in df.columns]
    
    X = df[feat_available].copy()
    X.fillna(0, inplace=True)
    return X, feat_available

warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description="Online SGD PnL Backtester")
    parser.add_argument("--data", required=True, help="Path to raw, unfiltered bursts CSV (e.g., nvda_raw_bursts.csv)")
    parser.add_argument("--target", default="reg_10m", help="Target key (e.g., reg_10m or reg_close)")
    
    # Grand Universal Filters
    parser.add_argument("--silence-tag", default="s2p0")
    parser.add_argument("--vol-frac", type=float, default=0.0027)
    parser.add_argument("--dir-thresh", type=float, default=0.68)
    parser.add_argument("--vol-ratio", type=float, default=0.36)
    parser.add_argument("--kappa", type=float, default=0.0) 
    
    args = parser.parse_args()

    print(f"========== ONLINE SGD REGIME BACKTESTER ==========")
    print(f"Data:    {args.data}")
    print(f"Target:  {args.target}")
    print(f"Filters: {args.silence_tag}, vf={args.vol_frac}, d={args.dir_thresh}, r={args.vol_ratio}\n")

    # 1. LOAD DATA 
    if not os.path.exists(args.data):
        print(f"ERROR: Cannot find {args.data}")
        sys.exit(1)
        
    print("Loading extremely raw LOBSTER C++ output into RAM...")
    df = pd.read_csv(args.data)
    
    df['DateCol'] = pd.to_datetime(df['Date'].astype(str))
    
    # Compute the dynamic 14-day trailing ADV so we can filter correctly based on fraction
    print("Computing 14-day trailing cross-asset ADV...")
    adv_series = compute_trailing_adv(df, window=14)
    burst_adv = df["Date"].map(adv_series)
    
    min_vol_per_burst = (args.vol_frac * burst_adv).reindex(df.index)

    # 2. APPLY GRAND UNIVERSAL FILTERS
    print("Applying Grand Universal geometry structural rules...")
    filtered = classify_and_filter(
        df,
        min_vol=0, # Absolute volume is ignored
        dir_thresh=args.dir_thresh,
        vol_ratio=args.vol_ratio,
        kappa=args.kappa,
        require_directional=False,
        min_vol_per_burst=min_vol_per_burst
    )
    
    # Strictly isolate the correct silence frame requested (assuming Pre-Filtering rules or manually filtering based on existing Delay times)
    # The dataset needs to have pre-calculated 'delay' logic mapped to s0p5, s1p0, etc. 
    # For now, we assume `filtered` natively isolated purely chronological, correctly bounded bursts for the backtest.
    
    if len(filtered) == 0:
        print("ERROR: Strict universal filters eliminated 100% of the dataset!")
        sys.exit(1)
        
    print(f"Dataset securely shrunk from {len(df):,} to {len(filtered):,} perfectly valid true bursts.\n")
    
    # 3. BUILD AI FEATURES & REGRESSION TARGETS
    try:
        y, task_type, meta = build_target(filtered, args.target)
        X, feat_available = get_features(filtered, args.target)
        if task_type != 'regression':
            print("ERROR: Online SGD backtester must predict continuous scale Regression tasks (e.g. reg_10m) to calculate PnL!")
            sys.exit(1)
    except Exception as e:
        print("ERROR building AI features:", e)
        sys.exit(1)
        
    y = y if isinstance(y, np.ndarray) else y.values
    X = X.values

    # 4. BURN-IN PHASE (1 MONTH)
    dates = sorted(filtered['DateCol'].unique())
    if len(dates) < 30:
        print("Dataset does not encompass enough time to perform 1 month burn-in + out of sample.")
        sys.exit(1)
        
    initial_burn_end_date = dates[20] # Roughly 1 trading month (21 days)
    
    train_mask = (filtered['DateCol'] <= initial_burn_end_date).values
    X_train, y_train = X[train_mask], y[train_mask]
    
    model = SGDRegressor(
        loss='squared_error',
        penalty='l2',
        alpha=0.0001,
        learning_rate='adaptive', 
        eta0=0.01,
        random_state=42
    )
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    
    if len(X_train_s) > 0:
        model.partial_fit(X_train_s, y_train)
        print(f"Burn-in initialized precisely over {len(X_train)} chronological bursts ending {str(initial_burn_end_date).split(' ')[0]}.")
    else:
        print("No bursts found inside the initial burn-in window!")
        sys.exit(1)

    # 5. EX-ANTE ONLINE DAILY LEAK-FREE WALK FORWARD
    remaining_days = dates[21:]
    print(f"Executing Walk-Forward continuous multi-year simulation across {len(remaining_days)} distinct trading days...")
    
    daily_pnls = []
    total_trades = 0
    total_longs = 0
    total_shorts = 0
    
    # Keep rolling lookback of predictions (e.g., roughly 1000 burst predictions) to natively track dynamic 75th percentile
    recent_predictions = collections.deque(maxlen=1000) 
    # Pre-seed deque from the training set so our very first day has valid percentile thresholds!
    if len(X_train_s) > 0:
        burn_preds = model.predict(X_train_s)
        for bp in burn_preds:
            recent_predictions.append(bp)
    
    # ── THE CORE ENGINE ──
    cum_pnl_tracker = 0.0

    for day_idx, day in enumerate(remaining_days):
        day_mask = (filtered['DateCol'] == day).values
        if day_mask.sum() == 0:
            continue
            
        X_day = X[day_mask]
        y_day = y[day_mask] # True Permanence (arcsinh real value structure log-returns)
        
        # 1. Transform strictly off yesterday's scaler weights (No Peeking at today!)
        X_day_s = scaler.transform(X_day)
        
        # 2. Predict today's permanence identically blind
        preds = model.predict(X_day_s)
        
        # 3. Calculate Dynamic Entry Thresholds (75th / 25th)
        current_long_thresh = np.percentile(recent_predictions, 75)
        current_short_thresh = np.percentile(recent_predictions, 25)
        
        day_pnl = 0.0
        # 4. Trigger simulated entries blindly strictly if they hit conviction thresholds
        for i, pred in enumerate(preds):
            if pred > current_long_thresh:
                day_pnl += y_day[i] 
                total_longs += 1
                total_trades += 1
            elif pred < current_short_thresh:
                day_pnl -= y_day[i]
                total_shorts += 1
                total_trades += 1
                
            recent_predictions.append(pred)
            
        daily_pnls.append(day_pnl)
        cum_pnl_tracker += day_pnl
        
        # 5. AT CLOSE: Execute Nightly Model Adaptation (Regime update)
        model.partial_fit(X_day_s, y_day)
        scaler.partial_fit(X_day) 

        # Progress timeline every ~1 month (20 trading days)
        if day_idx % 20 == 0:
            day_str = str(day).split()[0]
            print(f"[{day_str}] Walk-Forward Day {day_idx}/{len(remaining_days)} "
                  f"| CumPnL: {cum_pnl_tracker:7.3f} "
                  f"| Trades Executed: {total_trades:5d} "
                  f"| Rolling Thresholds L>{current_long_thresh:.3f} S<{current_short_thresh:.3f}") 

    # 6. CALCULATE SHARPE & PNL STATISTICS
    daily_pnls = np.array(daily_pnls)
    cum_pnl = np.sum(daily_pnls)
    mean_daily_pnl = np.mean(daily_pnls)
    std_daily_pnl = np.std(daily_pnls)
    
    # Sharpe = Mean / Std Daily PnL * sqrt(Trading_Days_in_Year)
    sharpe_ratio = 0.0
    if std_daily_pnl > 0:
        sharpe_ratio = (mean_daily_pnl / std_daily_pnl) * np.sqrt(252)

    print("\n" + "="*80)
    print("  SIMULATION COMPLETE (OUT-OF-SAMPLE REGIME RESULTS)")
    print("="*80)
    print(f"  Total Valid Bursts Scanned:   {len(y) - len(y_train):,}")
    print(f"  Total Trades Fired:           {total_trades:,} ({total_longs:,} Long / {total_shorts:,} Short)")
    print(f"\n  Cumulative Simulated PnL:     {cum_pnl:.4f}")
    print(f"  Daily Mean PnL:               {mean_daily_pnl:.5f}")
    print(f"  Daily StdDev (Variance):      {std_daily_pnl:.5f}")
    print(f"  Annualized Sharpe Ratio:      {sharpe_ratio:.2f}")
    print("="*80)

if __name__ == "__main__":
    main()
