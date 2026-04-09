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

PATCH_VERSION = "nan-guard-v4-20260409"

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
        raise ValueError(
            "SGD backtester currently supports regression targets only: "
            "reg_1m/reg_3m/reg_5m/reg_10m/reg_close/reg_clop/reg_clcl"
        )

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


def is_close_style_target(target_key):
    return target_key in {
        "reg_close", "reg_clop", "reg_clcl",
        "cls_close", "cls_clop", "cls_clcl",
    }

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
    parser.add_argument("--signal-mode", choices=["percentile", "cost_aware", "direction"], default="percentile",
                        help="Trade trigger mode: percentile thresholds, cost-aware edge gating, or sign(prediction) direction")
    parser.add_argument("--cost-buffer-mult", type=float, default=1.0,
                        help="Safety multiplier for spread cost in cost_aware mode (1.0 means edge must exceed estimated costs)")
    parser.add_argument("--mid-col", default="EndPrice",
                        help="Proxy mid-price column for burst_stream fills (default EndPrice)")
    parser.add_argument("--entry-bid-col", default="EndBid",
                        help="Bid column used for quote-based burst_stream entry")
    parser.add_argument("--entry-ask-col", default="EndAsk",
                        help="Ask column used for quote-based burst_stream entry")
    parser.add_argument("--exit-bid-col", default="EndBid",
                        help="Bid column used for quote-based burst_stream exit")
    parser.add_argument("--exit-ask-col", default="EndAsk",
                        help="Ask column used for quote-based burst_stream exit")
    parser.add_argument("--horizon-minutes", type=float, default=None,
                        help="Hold horizon for burst_stream mode. If omitted, inferred from target (e.g., reg_10m -> 10)")
    parser.add_argument("--position-size-mult", type=float, default=1.0,
                        help="Fraction of BurstVolume traded per signal (1.0 = full burst volume)")
    parser.add_argument("--position-mode", choices=["fraction", "shares"], default="fraction",
                        help="Position sizing: fraction of burst volume or fixed shares per trade")
    parser.add_argument("--shares-per-trade", type=float, default=1.0,
                        help="Fixed shares per trade when --position-mode=shares")
    parser.add_argument("--pnl-space", choices=["raw", "transformed"], default="raw",
                        help="Space used for reported PnL/Sharpe. raw is dollar-like; transformed uses arcsinh compression")
    parser.add_argument("--adaptive-scaler", action="store_true",
                        help="Update StandardScaler online after each day (off by default for scale stability)")
    parser.add_argument("--debug-trades-out", default="",
                        help="Optional CSV path to write realized trade-level diagnostics")
    parser.add_argument("--debug-signals-out", default="",
                        help="Optional CSV path to write per-burst signal/gate diagnostics")
    
    args = parser.parse_args()

    print(f"========== ONLINE SGD REGIME BACKTESTER ==========")
    print(f"Patch:   {PATCH_VERSION}")
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
    if args.position_mode == "shares":
        print(f"Position sizing: fixed shares per trade = {args.shares_per_trade}")
    else:
        print(f"Position sizing: fraction of burst volume = {args.position_size_mult}")
    print(f"Reported PnL space: {args.pnl_space}")
    print(f"Execution mode: {args.execution_mode}")
    print(f"Signal mode: {args.signal_mode}")
    print(f"Scaler mode: {'adaptive' if args.adaptive_scaler else 'fixed-after-burn-in'}")

    close_style_target = is_close_style_target(args.target)
    if args.execution_mode == "burst_stream":
        if close_style_target:
            print("Hold horizon: close-style target (positions exit at end of same trading day)")
        else:
            horizon_minutes = args.horizon_minutes if args.horizon_minutes is not None else infer_horizon_minutes(args.target)
            print(f"Hold horizon: {horizon_minutes:.2f} minutes")
        
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

    # Hard guard against residual NaN/Inf in targets/features (can happen for
    # overnight targets on sparse/missing daily print sequences).
    finite_y = np.isfinite(y)
    finite_X = np.all(np.isfinite(X), axis=1)
    finite_mask = finite_y & finite_X
    dropped_rows = int((~finite_mask).sum())
    if dropped_rows > 0:
        print(f"Dropping {dropped_rows:,} bursts with non-finite feature/target values before simulation.")
        filtered = filtered.loc[finite_mask].copy()
        y = y[finite_mask]
        X = X[finite_mask]

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

    stream_quote_mode = (
        args.execution_mode == "burst_stream"
        and args.entry_bid_col in filtered.columns
        and args.entry_ask_col in filtered.columns
        and args.exit_bid_col in filtered.columns
        and args.exit_ask_col in filtered.columns
    )
    if args.execution_mode == "burst_stream":
        if stream_quote_mode:
            print(
                f"Burst-stream fill mode: quote-based "
                f"(entry bid/ask={args.entry_bid_col}/{args.entry_ask_col}, "
                f"exit bid/ask={args.exit_bid_col}/{args.exit_ask_col})"
            )
        else:
            print("Burst-stream fill mode: mid+spread proxy (quote columns not fully available)")

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
    total_gross_move_raw = 0.0

    side_stats = {
        1: {"name": "long", "trades": 0, "wins": 0, "gross": 0.0, "cost": 0.0, "net": 0.0},
        -1: {"name": "short", "trades": 0, "wins": 0, "gross": 0.0, "cost": 0.0, "net": 0.0},
    }

    signal_evals = 0
    signal_pass_long = 0
    signal_pass_short = 0
    signal_reject = 0

    trade_rows = []
    signal_rows = []
    
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
    open_trades = collections.deque()
    hold_delta = None
    if args.execution_mode == "burst_stream" and not close_style_target:
        horizon_minutes = args.horizon_minutes if args.horizon_minutes is not None else infer_horizon_minutes(args.target)
        hold_delta = np.timedelta64(int(round(horizon_minutes * 60)), "s")

    for day_idx, day in enumerate(remaining_days):
        day_mask = (filtered['DateCol'] == day).values
        if day_mask.sum() == 0:
            continue
            
        X_day = X[day_mask]
        y_day = y[day_mask] # True Permanence (arcsinh real value structure log-returns)
        day_df = filtered.loc[day_mask]

        # Secondary per-day guard used only for nightly model updates.
        day_finite = np.isfinite(y_day) & np.all(np.isfinite(X_day), axis=1)
        X_day_fit = X_day[day_finite]
        y_day_fit = y_day[day_finite]
        vol_day = day_df['BurstVolume'].to_numpy(dtype=float)
        day_idx_arr = np.flatnonzero(day_mask)
        event_ts_day = event_ts_np[day_idx_arr]
        day_end_ts = event_ts_day[-1]
        mid_day = day_df[args.mid_col].to_numpy(dtype=float) if args.execution_mode == "burst_stream" else None
        entry_bid_day = day_df[args.entry_bid_col].to_numpy(dtype=float) if stream_quote_mode else None
        entry_ask_day = day_df[args.entry_ask_col].to_numpy(dtype=float) if stream_quote_mode else None
        exit_bid_day = day_df[args.exit_bid_col].to_numpy(dtype=float) if stream_quote_mode else None
        exit_ask_day = day_df[args.exit_ask_col].to_numpy(dtype=float) if stream_quote_mode else None
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
        
        # 3. Calculate dynamic thresholds only for percentile trigger mode.
        current_long_thresh = np.percentile(recent_predictions, 75)
        current_short_thresh = np.percentile(recent_predictions, 25)
        
        day_pnl = 0.0
        day_pnl_raw = 0.0

        # 4. Trigger simulated entries blindly strictly if they hit conviction thresholds
        for i, pred in enumerate(preds):
            current_ts = event_ts_day[i]

            # Close all due trades at current burst proxy prices (round-trip simulation mode).
            if args.execution_mode == "burst_stream":
                while open_trades and current_ts >= open_trades[0]["due_ts"]:
                    tr = open_trades.popleft()
                    if stream_quote_mode:
                        exit_px = float(exit_bid_day[i]) if tr["side"] > 0 else float(exit_ask_day[i])
                        gross_edge_raw = tr["side"] * tr["qty"] * (exit_px - tr["entry_px"])
                        exit_cost_raw = 0.0
                        net_edge_raw = gross_edge_raw
                    else:
                        exit_mid = float(mid_day[i])
                        exit_spread_val = max(0.0, float(spread_exit_day[i])) if use_spread_cost else 0.0
                        exit_cost_raw = tr["qty"] * args.spread_exit_multiplier * exit_spread_val
                        gross_mid_move_raw = tr["side"] * tr["qty"] * (exit_mid - tr["entry_mid"])
                        gross_edge_raw = gross_mid_move_raw
                        net_edge_raw = gross_mid_move_raw - tr["entry_cost_raw"] - exit_cost_raw
                    gross_edge = gross_edge_raw if args.pnl_space == "raw" else np.arcsinh(gross_edge_raw)
                    cost_edge = (tr["entry_cost_raw"] + exit_cost_raw) if args.pnl_space == "raw" else np.arcsinh(tr["entry_cost_raw"] + exit_cost_raw)
                    net_edge = net_edge_raw if args.pnl_space == "raw" else np.arcsinh(net_edge_raw)

                    side_stats[tr["side"]]["trades"] += 1
                    side_stats[tr["side"]]["wins"] += int(net_edge_raw > 0)
                    side_stats[tr["side"]]["gross"] += gross_edge
                    side_stats[tr["side"]]["cost"] += cost_edge
                    side_stats[tr["side"]]["net"] += net_edge

                    if args.debug_trades_out:
                        hold_seconds = (current_ts - tr["entry_ts"]) / np.timedelta64(1, "s")
                        trade_rows.append({
                            "day": str(day),
                            "execution_mode": "burst_stream",
                            "entry_ts": str(tr["entry_ts"]),
                            "exit_ts": str(current_ts),
                            "hold_seconds": float(hold_seconds),
                            "side": int(tr["side"]),
                            "qty": float(tr["qty"]),
                            "entry_cost_raw": float(tr["entry_cost_raw"]),
                            "exit_cost_raw": float(exit_cost_raw),
                            "gross_raw": float(gross_edge_raw),
                            "net_raw": float(net_edge_raw),
                            "pred": float(tr["pred"]),
                            "pred_raw": float(np.sinh(tr["pred"])),
                            "pred_move_per_share": float(np.sinh(tr["pred"]) / max(float(tr["burst_vol"]), 1e-12)),
                            "gate": "" if np.isnan(tr["gate"]) else float(tr["gate"]),
                            "burst_volume": float(tr["burst_vol"]),
                        })
                    day_pnl_raw += net_edge_raw
                    total_exit_spread_cost_raw += exit_cost_raw
                    total_spread_cost_raw += exit_cost_raw
                    if args.pnl_space == "raw":
                        day_pnl += net_edge_raw
                    else:
                        day_pnl += np.arcsinh(net_edge_raw)

            burst_vol_i = float(vol_day[i])
            qty = args.shares_per_trade if args.position_mode == "shares" else (args.position_size_mult * burst_vol_i)
            pred_raw = np.sinh(pred)
            pred_move_per_share = pred_raw / max(burst_vol_i, 1e-12)
            signal_evals += 1

            side = 0
            gate = np.nan
            if args.signal_mode == "percentile":
                if pred > current_long_thresh:
                    side = 1
                elif pred < current_short_thresh:
                    side = -1
            elif args.signal_mode == "cost_aware":
                # Cost-aware trigger: use predicted per-share move and require it to clear estimated round-trip spread costs.
                spread_entry_val = max(0.0, float(spread_entry_day[i])) if use_spread_cost else 0.0
                spread_exit_val = max(0.0, float(spread_exit_day[i])) if use_spread_cost else 0.0
                per_share_cost = (
                    args.spread_multiplier * spread_entry_val
                    + args.spread_exit_multiplier * spread_exit_val
                )
                gate = args.cost_buffer_mult * per_share_cost
                if pred_move_per_share > gate:
                    side = 1
                elif pred_move_per_share < -gate:
                    side = -1
            else:
                # Direction-only trigger: trade purely on predicted sign.
                if pred > 0:
                    side = 1
                elif pred < 0:
                    side = -1

            if side == 1:
                signal_pass_long += 1
            elif side == -1:
                signal_pass_short += 1
            else:
                signal_reject += 1

            if args.debug_signals_out:
                signal_rows.append({
                    "day": str(day),
                    "ts": str(current_ts),
                    "target": args.target,
                    "pred": float(pred),
                    "pred_raw": float(pred_raw),
                    "pred_move_per_share": float(pred_move_per_share),
                    "gate": "" if np.isnan(gate) else float(gate),
                    "signal_side": int(side),
                    "burst_volume": float(burst_vol_i),
                    "qty": float(qty),
                })

            if side != 0:
                if side > 0:
                    total_longs += 1
                else:
                    total_shorts += 1
                total_trades += 1

            if side != 0:
                if args.execution_mode == "label_proxy":
                    # permanence label encodes approx. burst_volume * price_move; convert to per-share move.
                    gross_edge_raw = np.sinh(y_day[i])
                    edge_per_share = gross_edge_raw / max(burst_vol_i, 1e-12)
                    signed_edge_raw = side * qty * edge_per_share

                    spread_cost_raw = 0.0
                    if use_spread_cost:
                        spread_entry_val = max(0.0, float(spread_entry_day[i]))
                        spread_exit_val = max(0.0, float(spread_exit_day[i]))
                        entry_cost_raw = (
                            args.spread_multiplier * qty * spread_entry_val
                        )
                        exit_cost_raw = (
                            args.spread_exit_multiplier * qty * spread_exit_val
                        )
                        spread_cost_raw = entry_cost_raw + exit_cost_raw
                        total_entry_spread_cost_raw += entry_cost_raw
                        total_exit_spread_cost_raw += exit_cost_raw

                    net_edge_raw = signed_edge_raw - spread_cost_raw
                    gross_edge = signed_edge_raw if args.pnl_space == "raw" else np.arcsinh(signed_edge_raw)
                    cost_edge = spread_cost_raw if args.pnl_space == "raw" else np.arcsinh(spread_cost_raw)
                    net_edge = net_edge_raw if args.pnl_space == "raw" else np.arcsinh(net_edge_raw)

                    side_stats[side]["trades"] += 1
                    side_stats[side]["wins"] += int(net_edge_raw > 0)
                    side_stats[side]["gross"] += gross_edge
                    side_stats[side]["cost"] += cost_edge
                    side_stats[side]["net"] += net_edge

                    if args.debug_trades_out:
                        trade_rows.append({
                            "day": str(day),
                            "execution_mode": "label_proxy",
                            "entry_ts": str(current_ts),
                            "exit_ts": str(current_ts),
                            "hold_seconds": 0.0,
                            "side": int(side),
                            "qty": float(qty),
                            "entry_cost_raw": float(spread_cost_raw),
                            "exit_cost_raw": 0.0,
                            "gross_raw": float(signed_edge_raw),
                            "net_raw": float(net_edge_raw),
                            "pred": float(pred),
                            "pred_raw": float(pred_raw),
                            "pred_move_per_share": float(pred_move_per_share),
                            "gate": "" if np.isnan(gate) else float(gate),
                            "burst_volume": float(burst_vol_i),
                        })

                    day_pnl_raw += net_edge_raw
                    total_spread_cost_raw += spread_cost_raw
                    if args.pnl_space == "raw":
                        day_pnl += net_edge_raw
                    else:
                        day_pnl += np.arcsinh(net_edge_raw)
                else:
                    # Open a round-trip trade now; realize PnL when a later burst reaches horizon.
                    if stream_quote_mode:
                        entry_px = float(entry_ask_day[i]) if side > 0 else float(entry_bid_day[i])
                        entry_cost_raw = 0.0
                    else:
                        entry_mid = float(mid_day[i])
                        spread_entry_val = max(0.0, float(spread_entry_day[i])) if use_spread_cost else 0.0
                        entry_cost_raw = qty * args.spread_multiplier * spread_entry_val
                        total_entry_spread_cost_raw += entry_cost_raw
                        total_spread_cost_raw += entry_cost_raw

                    due_ts = day_end_ts if close_style_target else (current_ts + hold_delta)
                    # If there is no later event to close against, skip this entry.
                    if due_ts <= current_ts:
                        continue

                    open_trades.append({
                        "due_ts": due_ts,
                        "entry_mid": entry_mid if not stream_quote_mode else 0.0,
                        "entry_px": entry_px if stream_quote_mode else 0.0,
                        "qty": qty,
                        "side": side,
                        "entry_cost_raw": entry_cost_raw,
                        "entry_ts": current_ts,
                        "pred": pred,
                        "gate": gate,
                        "burst_vol": burst_vol_i,
                    })
                
            recent_predictions.append(pred)
            
        daily_pnls.append(day_pnl)
        cum_pnl_raw_tracker += day_pnl_raw
        cum_pnl_tracker += day_pnl
        
        # 5. AT CLOSE: Execute Nightly Model Adaptation (Regime update)
        if len(X_day_fit) > 0:
            if args.adaptive_scaler:
                scaler.partial_fit(X_day_fit)
                X_day_for_fit = scaler.transform(X_day_fit)
            else:
                X_day_for_fit = scaler.transform(X_day_fit)
            fit_finite = np.isfinite(y_day_fit) & np.all(np.isfinite(X_day_for_fit), axis=1)
            if not np.all(fit_finite):
                bad = int((~fit_finite).sum())
                print(f"WARNING: Skipping {bad} non-finite fit rows on {str(day).split()[0]} before partial_fit.")
            if np.any(fit_finite):
                model.partial_fit(X_day_for_fit[fit_finite], y_day_fit[fit_finite])

        # Progress timeline every ~1 month (20 trading days)
        if day_idx % 20 == 0:
            day_str = str(day).split()[0]
            if args.signal_mode == "percentile":
                trigger_info = f"Rolling Thresholds L>{current_long_thresh:.3f} S<{current_short_thresh:.3f}"
            elif args.signal_mode == "cost_aware":
                trigger_info = f"CostAware buffer={args.cost_buffer_mult:.2f}"
            else:
                trigger_info = "Direction sign gate"
            print(f"[{day_str}] Walk-Forward Day {day_idx}/{len(remaining_days)} "
                  f"| CumPnL: {cum_pnl_tracker:7.3f} "
                  f"| Trades Executed: {total_trades:5d} "
                  f"| {trigger_info}") 

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

    if signal_evals > 0:
        print("\n  Signal Diagnostics")
        print(f"  Signals evaluated:            {signal_evals:,}")
        print(f"  Signals passed long:          {signal_pass_long:,} ({100.0 * signal_pass_long / signal_evals:.2f}%)")
        print(f"  Signals passed short:         {signal_pass_short:,} ({100.0 * signal_pass_short / signal_evals:.2f}%)")
        print(f"  Signals rejected:             {signal_reject:,} ({100.0 * signal_reject / signal_evals:.2f}%)")

    print("\n  Side Diagnostics")
    for side_key in [1, -1]:
        stats = side_stats[side_key]
        trades = int(stats["trades"])
        win_rate = (100.0 * stats["wins"] / trades) if trades > 0 else 0.0
        avg_net = (stats["net"] / trades) if trades > 0 else 0.0
        print(
            f"  {stats['name'].capitalize():<6} trades={trades:>7,} "
            f"win_rate={win_rate:>6.2f}% "
            f"gross={stats['gross']:>10.4f} "
            f"cost={stats['cost']:>10.4f} "
            f"net={stats['net']:>10.4f} "
            f"avg_net/trade={avg_net:>10.6f}"
        )

    if args.debug_trades_out:
        pd.DataFrame(trade_rows).to_csv(args.debug_trades_out, index=False)
        print(f"\n  Wrote trade diagnostics:       {args.debug_trades_out}")

    if args.debug_signals_out:
        pd.DataFrame(signal_rows).to_csv(args.debug_signals_out, index=False)
        print(f"  Wrote signal diagnostics:      {args.debug_signals_out}")

    print("="*80)

if __name__ == "__main__":
    main()
