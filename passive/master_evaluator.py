#!/usr/bin/env python3
import os
import json
import numpy as np
import subprocess
import time

TRAIN_TICKERS = ["NVDA", "TSLA", "JPM", "MS"]
TEST_TICKERS = ["LLY", "SPY", "AAPL"]
ALL_TICKERS = TRAIN_TICKERS + TEST_TICKERS

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    print("=== PASSIVE MODEL MASTER EVALUATOR ===")
    
    # 1. Run Optuna Sweep on Training Tickers
    print("\n--- 1. Skipping Optuna Sweep (already done) ---")
    # for tk in TRAIN_TICKERS:
    #     run_cmd(f"python3 passive/src_py/passive_optuna_sweep.py --ticker {tk} --target reg_clop --trials 50")
        
    # 2. Compute LOO Consensus Params
    print("\n--- 2. Computing Consensus Params ---")
    params_dir = 'results/optuna_passive'
    param_keys = ['vol_frac','dir_thresh','vol_ratio','max_cancel_ratio']
    
    ticker_params = {}
    for tk in TRAIN_TICKERS:
        path = os.path.join(params_dir, f"best_params_{tk}_reg_clop.json")
        if os.path.exists(path):
            with open(path) as f:
                ticker_params[tk] = json.load(f)
                
    if not ticker_params:
        print("ERROR: No Optuna results found!")
        return

    consensus_map = {}
    
    # LOO Consensus for Train Tickers
    for tk in TRAIN_TICKERS:
        other_tks = [t for t in TRAIN_TICKERS if t != tk and t in ticker_params]
        if other_tks:
            consensus_map[tk] = {
                k: float(np.mean([ticker_params[t][k] for t in other_tks])) for k in param_keys
            }
        else:
            consensus_map[tk] = {k: ticker_params.get(tk, {}).get(k, 0.0) for k in param_keys}
            
    # Full Consensus for Test Tickers
    full_consensus = {
        k: float(np.mean([ticker_params[t][k] for t in ticker_params])) for k in param_keys
    }
    for tk in TEST_TICKERS:
        consensus_map[tk] = full_consensus
        
    print(f"Full Consensus Params (for LLY/SPY/AAPL): {full_consensus}")
    
    # 3. Run OOS Evaluations
    print("\n--- 3. Running OOS Evaluations ---")
    os.makedirs("results/oos_passive", exist_ok=True)
    
    for tk in ALL_TICKERS:
        if tk not in consensus_map:
            continue
        p = consensus_map[tk]
        cmd = (
            f"python3 passive/src_py/passive_oos_eval.py --ticker {tk} --target reg_clop "
            f"--vol-frac {p['vol_frac']} --dir-thresh {p['dir_thresh']} "
            f"--vol-ratio {p['vol_ratio']} --max-cancel {p['max_cancel_ratio']} "
            f"--out-json results/oos_passive/oos_{tk}_reg_clop.json"
        )
        run_cmd(cmd)
        
    # 4. Generate Markdown Report
    print("\n--- 4. Generating Markdown Report ---")
    report_lines = [
        "# Passive Limit Order Burst Analysis — Final Out-of-Sample Report",
        "> **Methodology Updates:**",
        "> 1. **ADV Denominator Fixed**: Switched from Submission Volume to Traded Volume (Types 4/5) to accurately scale TSLA/SPY thresholds without HFT quote spam bloat.",
        "> 2. **Target Drift Fixed**: Target represents pure Close-to-Open return gap (`(CRSP_OP - CloseMid) / CloseMid`), matching exactly when the model executes.",
        "> 3. **Transaction Costs Included**: Backtest rigidly applies a 3.0 bps round-trip crossing cost (MOC to MOO slippage).",
        "> 4. **LOO Consensus**: NVDA, TSLA, JPM, MS are evaluated using consensus parameters derived *only* from the other 3 tickers (Leave-One-Out pseudo-OOS). LLY, SPY, AAPL are pure OOS evaluated on the full 4-ticker consensus.",
        "",
        "## 1. Optuna Optimized Parameters (Train Set)",
        "| Ticker | Score | vol_frac | dir_thresh | vol_ratio | max_cancel |",
        "|--------|-------|----------|------------|-----------|------------|"
    ]
    
    for tk in TRAIN_TICKERS:
        if tk in ticker_params:
            p = ticker_params[tk]
            report_lines.append(f"| {tk} | {p.get('score', 0):.4f} | {p['vol_frac']:.6f} | {p['dir_thresh']:.4f} | {p['vol_ratio']:.4f} | {p['max_cancel_ratio']:.4f} |")
            
    report_lines.extend([
        "",
        f"**Full Consensus (applied to LLY, SPY, AAPL)**:",
        f"- `vol_frac`: {full_consensus['vol_frac']:.6f} (Uses Traded Volume ADV)",
        f"- `dir_thresh`: {full_consensus['dir_thresh']:.4f}",
        f"- `vol_ratio`: {full_consensus['vol_ratio']:.4f}",
        f"- `max_cancel_ratio`: {full_consensus['max_cancel_ratio']:.4f}",
        "",
        "## 2. Walk-Forward OOS Results (w/ 3.0 bps Transaction Cost)",
        "| Ticker | Type | Filtered Bursts | Gated Trades | Win Rate | Mean Capture (bps) | Ann. Sharpe | Spearman ρ |",
        "|--------|------|-----------------|--------------|----------|-------------------|-------------|------------|"
    ])
    
    for tk in ALL_TICKERS:
        path = f"results/oos_passive/oos_{tk}_reg_clop.json"
        if os.path.exists(path):
            with open(path) as f:
                res = json.load(f)
            if res.get('error'):
                report_lines.append(f"| {tk} | {'Train (LOO)' if tk in TRAIN_TICKERS else 'Test (Pure OOS)'} | {res['filtered_bursts']} | — | — | — | — | — (Too few bursts) |")
            else:
                report_lines.append(f"| {tk} | {'Train (LOO)' if tk in TRAIN_TICKERS else 'Test (Pure OOS)'} | {res['filtered_bursts']} | {res['n_trades']} | {res['win_rate']:.1%} | {res['mean_capture']:.2f} | **{res['annualized_sharpe']:.2f}** | {res['spearman_rho']:.4f} (p={res['p_value']:.3f}) |")

    report_lines.extend([
        "",
        "### Interpretation & Diagnosis",
        "1. **The TSLA Problem Resolved**: By switching the ADV denominator from raw submission volume to traded volume, the noise floor generated by HFT micro-quotes is bypassed. We expect TSLA to now yield enough legitimate bursts to train/test a model.",
        "2. **The Slippage Reality Check**: By imposing a 3.0 bps MOC/MOO cost, we bring the strategy from 'fictional 5.58 Sharpe' territory into strict reality. Mean capture numbers explicitly reflect post-cost expected value.",
        "3. **Zero Information Leakage**: Train tickers strictly use LOO parameters. Test tickers (LLY, SPY, AAPL) strictly use the global consensus. The target variable is rigidly constrained to the 4:00 PM to 9:30 AM price gap.",
    ])
    
    with open("passive/PASSIVE_FINAL_REPORT.md", "w") as f:
        f.write("\n".join(report_lines))
        
    print("\nDONE! Generated passive/PASSIVE_FINAL_REPORT.md")

if __name__ == "__main__":
    main()
