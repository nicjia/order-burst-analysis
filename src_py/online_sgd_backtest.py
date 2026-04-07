#!/usr/bin/env python3
import warnings
import argparse
import sys
import os
import collections
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# Ensure we can import local modules
sys.path.append(str(Path(__file__).parent.absolute()))
from silence_optimized_sweep import compute_trailing_adv, classify_and_filter

TARGET_MAP = {
    'cls_1m':     ('Perm_t1m',     'binary',  0.0),
    'cls_3m':     ('Perm_t3m',     'binary',  0.0),
    'cls_5m':     ('Perm_t5m',     'binary',  0.0),
    'cls_10m':    ('Perm_t10m',    'binary',  0.0),
    'cls_close':  ('Perm_tCLOSE', 'binary',  0.0),
    'cls_clop':   ('Perm_CLOP',   'binary',  0.0),
    'cls_clcl':   ('Perm_CLCL',   'binary',  0.0),
    'reg_close':  ('Perm_tCLOSE', 'regression', None),
    'reg_clop':   ('Perm_CLOP',   'regression', None),
    'reg_clcl':   ('Perm_CLCL',   'regression', None),
    'reg_1m':     ('Perm_t1m',    'regression', None),
    'reg_3m':     ('Perm_t3m',    'regression', None),
    'reg_5m':     ('Perm_t5m',    'regression', None),
    'reg_10m':    ('Perm_t10m',   'regression', None),
}

def build_target(df, target_key):
    col, task, threshold = TARGET_MAP[target_key]
    if col not in df.columns:
        raise ValueError(f"Target column '{col}' not in data.")
    vals = df[col].values.copy()
    if task == 'regression':
        lo = np.nanpercentile(vals, 1)
        hi = np.nanpercentile(vals, 99)
        y = np.clip(vals, lo, hi)
        return y, 'regression', {'lo': lo, 'hi': hi}
    else:
        raise ValueError("SGD requires a regression target!")

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


def infer_exit_spread_col(target_key):
    horizon_map = {
        "reg_1m": "Spread_1m",
        "reg_3m": "Spread_3m",
        "reg_5m": "Spread_5m",
        "reg_10m": "Spread_10m",
        "cls_1m": "Spread_1m",
        "cls_3m": "Spread_3m",
        "cls_5m": "Spread_5m",
        "cls_10m": "Spread_10m",
        "reg_close": "Spread_close",
        "cls_close": "Spread_close",
        "reg_clop": "Spread_clop",
        "cls_clop": "Spread_clop",
        "reg_clcl": "Spread_clcl",
        "cls_clcl": "Spread_clcl",
    }
    return horizon_map.get(target_key)


def infer_horizon_minutes(target_key):
    horizon_map = {
        "reg_1m": 1,
        "reg_3m": 3,
        "reg_5m": 5,
        "reg_10m": 10,
        "cls_1m": 1,
        "cls_3m": 3,
        "cls_5m": 5,
        "cls_10m": 10,
    }
    return horizon_map.get(target_key, 10)

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
    parser.add_argument("--spread-col", default="Spread",
                        help="Column name for bid-ask spread in price units")
    parser.add_argument("--spread-multiplier", type=float, default=0.5,
                        help="Entry-side spread multiplier (0.5=half spread at entry)")
    parser.add_argument("--spread-exit-col", default="",
                        help="Optional horizon spread column; if missing/empty, fallback uses entry spread column")
    parser.add_argument("--spread-exit-multiplier", type=float, default=0.5,
                        help="Exit-side spread multiplier (0.5=half spread at exit)")
    parser.add_argument("--execution-mode", choices=["label_proxy", "burst_stream"], default="burst_stream",
                        help="label_proxy uses permanence labels for realized edge; burst_stream uses event-time round-trip fills")
    parser.add_argument("--mid-col", default="EndPrice",
                        help="Proxy mid-price column for burst_stream fills (default EndPrice)")
    parser.add_argument("--horizon-minutes", type=float, default=None,
                        help="Hold horizon for burst_stream mode. If omitted, inferred from target (e.g., reg_10m -> 10)")
    parser.add_argument("--position-size-mult", type=float, default=1.0,
                        help="Fraction of BurstVolume traded per signal (1.0 = full burst volume)")
    parser.add_argument("--pnl-space", choices=["raw", "transformed"], default="raw",
                        help="Space used for reported PnL/Sharpe. raw is dollar-like; transformed uses arcsinh compression")
    parser.add_argument("--adaptive-scaler", action="store_true",
                        help="Update StandardScaler online after each day (off by default for scale stability)")
    
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

    use_spread_cost = args.spread_col in filtered.columns
    inferred_exit_col = infer_exit_spread_col(args.target)
    exit_spread_col = args.spread_exit_col.strip() if args.spread_exit_col.strip() else inferred_exit_col
    use_exit_spread_col = bool(exit_spread_col) and (exit_spread_col in filtered.columns)

    if use_spread_cost:
        if use_exit_spread_col:
            print(
                f"Execution model: round-trip spread-aware using entry='{args.spread_col}' and exit='{exit_spread_col}' "
                f"(mult={args.spread_multiplier}+{args.spread_exit_multiplier})"
            )
        else:
            print(
                f"Execution model: round-trip spread-aware using entry='{args.spread_col}' and exit~entry fallback "
                f"(mult={args.spread_multiplier}+{args.spread_exit_multiplier})"
            )
    else:
        print(f"Execution model: no spread cost (column '{args.spread_col}' not found)")
    print(f"Position size multiplier: {args.position_size_mult}")
    print(f"Reported PnL space: {args.pnl_space}")
    print(f"Execution mode: {args.execution_mode}")
    print(f"Scaler mode: {'adaptive' if args.adaptive_scaler else 'fixed-after-burn-in'}")
        
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

    # Build per-burst event timestamps for stream execution.
    time_col = "EndTime" if "EndTime" in filtered.columns else "StartTime"
    if time_col in filtered.columns:
        event_ts = filtered["DateCol"] + pd.to_timedelta(filtered[time_col], unit="s")
    else:
        event_ts = filtered["DateCol"]
    event_ts_np = event_ts.to_numpy()

    if args.execution_mode == "burst_stream" and args.mid_col not in filtered.columns:
        print(f"ERROR: --mid-col '{args.mid_col}' not found for burst_stream mode")
        sys.exit(1)

    # 4. BURN-IN PHASE (1 MONTH)
    dates = sorted(filtered['DateCol'].unique())
    if len(dates) < 30:
        print("Dataset does not encompass enough time to perform 1 month burn-in + out of sample.")
        sys.exit(1)
        
    initial_burn_end_date = dates[20] # Roughly 1 trading month (21 days)
    
    train_mask = (filtered['DateCol'] <= initial_burn_end_date).values
    X_train, y_train = X[train_mask], y[train_mask]
    
    model = SGDRegressor(
        loss='huber',           # FIX 1: Robust to massive market outliers
        epsilon=1.35,           # Standard boundary for where Huber kicks in
        penalty='l2',
        alpha=0.001,            # Slightly stronger regularization
        learning_rate='adaptive', 
        eta0=0.001,             # FIX 2: Lower initial step size (0.01 was too aggressive)
        max_iter=1000,
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
    total_spread_cost_raw = 0.0
    total_entry_spread_cost_raw = 0.0
    total_exit_spread_cost_raw = 0.0
    
    # Keep rolling lookback of predictions (e.g., roughly 1000 burst predictions) to natively track dynamic 75th percentile
    recent_predictions = collections.deque(maxlen=1000) 
    # Pre-seed deque from the training set so our very first day has valid percentile thresholds!
    if len(X_train_s) > 0:
        burn_preds = model.predict(X_train_s)
        for bp in burn_preds:
            recent_predictions.append(bp)
    
    # ── THE CORE ENGINE ──
    cum_pnl_tracker = 0.0
    cum_pnl_raw_tracker = 0.0

    for day_idx, day in enumerate(remaining_days):
        day_mask = (filtered['DateCol'] == day).values
        if day_mask.sum() == 0:
            continue
            
        X_day = X[day_mask]
        y_day = y[day_mask] # True Permanence (arcsinh real value structure log-returns)
        day_df = filtered.loc[day_mask]
        vol_day = day_df['BurstVolume'].to_numpy(dtype=float)
        day_idx_arr = np.flatnonzero(day_mask)
        event_ts_day = event_ts_np[day_idx_arr]
        mid_day = day_df[args.mid_col].to_numpy(dtype=float) if args.execution_mode == "burst_stream" else None
        spread_entry_day = filtered.loc[day_mask, args.spread_col].to_numpy(dtype=float) if use_spread_cost else None
        spread_exit_day = (
            filtered.loc[day_mask, exit_spread_col].to_numpy(dtype=float)
            if use_spread_cost and use_exit_spread_col
            else spread_entry_day
        )
        
        # 1. Transform strictly off yesterday's scaler weights (No Peeking at today!)
        X_day_s = scaler.transform(X_day)
        
        # 2. Predict today's permanence identically blind
        preds = model.predict(X_day_s)
        
        # 3. Calculate Dynamic Entry Thresholds (75th / 25th)
        current_long_thresh = np.percentile(recent_predictions, 75)
        current_short_thresh = np.percentile(recent_predictions, 25)
        
        day_pnl = 0.0
        day_pnl_raw = 0.0
        if day_idx == 0:
            open_trades = collections.deque()
            horizon_minutes = args.horizon_minutes if args.horizon_minutes is not None else infer_horizon_minutes(args.target)
            hold_delta = np.timedelta64(int(round(horizon_minutes * 60)), "s")

        # 4. Trigger simulated entries blindly strictly if they hit conviction thresholds
        for i, pred in enumerate(preds):
            current_ts = event_ts_day[i]

            # Close all due trades at current burst proxy prices (round-trip simulation mode).
            if args.execution_mode == "burst_stream":
                while open_trades and current_ts >= open_trades[0]["due_ts"]:
                    tr = open_trades.popleft()
                    exit_mid = float(mid_day[i])
                    exit_spread_val = max(0.0, float(spread_exit_day[i])) if use_spread_cost else 0.0
                    exit_cost_raw = tr["qty"] * args.spread_exit_multiplier * exit_spread_val
                    gross_mid_move_raw = tr["side"] * tr["qty"] * (exit_mid - tr["entry_mid"])
                    net_edge_raw = gross_mid_move_raw - tr["entry_cost_raw"] - exit_cost_raw
                    day_pnl_raw += net_edge_raw
                    total_exit_spread_cost_raw += exit_cost_raw
                    total_spread_cost_raw += exit_cost_raw
                    if args.pnl_space == "raw":
                        day_pnl += net_edge_raw
                    else:
                        day_pnl += np.arcsinh(net_edge_raw)

            side = 0
            if pred > current_long_thresh:
                side = 1
                total_longs += 1
                total_trades += 1
            elif pred < current_short_thresh:
                side = -1
                total_shorts += 1
                total_trades += 1

            if side != 0:
                if args.execution_mode == "label_proxy":
                    gross_edge_raw = np.sinh(y_day[i])
                    signed_edge_raw = gross_edge_raw if side > 0 else -gross_edge_raw
                    signed_edge_raw *= args.position_size_mult

                    spread_cost_raw = 0.0
                    if use_spread_cost:
                        spread_entry_val = max(0.0, float(spread_entry_day[i]))
                        spread_exit_val = max(0.0, float(spread_exit_day[i]))
                        entry_cost_raw = (
                            args.spread_multiplier * args.position_size_mult * float(vol_day[i]) * spread_entry_val
                        )
                        exit_cost_raw = (
                            args.spread_exit_multiplier * args.position_size_mult * float(vol_day[i]) * spread_exit_val
                        )
                        spread_cost_raw = entry_cost_raw + exit_cost_raw
                        total_entry_spread_cost_raw += entry_cost_raw
                        total_exit_spread_cost_raw += exit_cost_raw

                    net_edge_raw = signed_edge_raw - spread_cost_raw
                    day_pnl_raw += net_edge_raw
                    total_spread_cost_raw += spread_cost_raw
                    if args.pnl_space == "raw":
                        day_pnl += net_edge_raw
                    else:
                        day_pnl += np.arcsinh(net_edge_raw)
                else:
                    # Open a round-trip trade now; realize PnL when a later burst reaches horizon.
                    entry_mid = float(mid_day[i])
                    qty = args.position_size_mult * float(vol_day[i])
                    spread_entry_val = max(0.0, float(spread_entry_day[i])) if use_spread_cost else 0.0
                    entry_cost_raw = qty * args.spread_multiplier * spread_entry_val
                    total_entry_spread_cost_raw += entry_cost_raw
                    total_spread_cost_raw += entry_cost_raw
                    open_trades.append({
                        "due_ts": current_ts + hold_delta,
                        "entry_mid": entry_mid,
                        "qty": qty,
                        "side": side,
                        "entry_cost_raw": entry_cost_raw,
                    })
                
            recent_predictions.append(pred)
            
        daily_pnls.append(day_pnl)
        cum_pnl_raw_tracker += day_pnl_raw
        cum_pnl_tracker += day_pnl
        
        # 5. AT CLOSE: Execute Nightly Model Adaptation (Regime update)
        if args.adaptive_scaler:
            scaler.partial_fit(X_day)
            X_day_for_fit = scaler.transform(X_day)
        else:
            X_day_for_fit = X_day_s
        model.partial_fit(X_day_for_fit, y_day)

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

    if args.execution_mode == "burst_stream" and open_trades:
        print(f"  Open trades not yet due at end of data: {len(open_trades):,} (excluded from realized PnL)")

    print(f"  Total Valid Bursts Scanned:   {len(y) - len(y_train):,}")
    print(f"  Total Trades Fired:           {total_trades:,} ({total_longs:,} Long / {total_shorts:,} Short)")
    print(f"  Entry Spread Cost (raw):      {total_entry_spread_cost_raw:.4f}")
    print(f"  Exit Spread Cost (raw):       {total_exit_spread_cost_raw:.4f}")
    print(f"  Total Spread Cost (raw):      {total_spread_cost_raw:.4f}")
    print(f"\n  Cumulative Simulated PnL ({args.pnl_space}): {cum_pnl:.4f}")
    print(f"  Cumulative Sim PnL (raw):     {cum_pnl_raw_tracker:.4f}")
    print(f"  Daily Mean PnL ({args.pnl_space}):         {mean_daily_pnl:.5f}")
    print(f"  Daily StdDev ({args.pnl_space}):           {std_daily_pnl:.5f}")
    print(f"  Annualized Sharpe Ratio:      {sharpe_ratio:.2f}")
    print("="*80)

if __name__ == "__main__":
    main()
