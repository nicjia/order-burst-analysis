#!/usr/bin/env python3
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

JOB_ID = "12890943"
ROOT = Path("hoffman_pull_20260409_overnight")
LOG_DIR = ROOT / "results" / "overnight_backtests"
OUT_DIR = ROOT / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fnum(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    return f"{float(x):,.6f}"


def parse_run_log(log_path: Path):
    txt = log_path.read_text(errors="ignore")

    def m1(pat, cast=str, default=None):
        m = re.search(pat, txt, re.M)
        if not m:
            return default
        try:
            return cast(m.group(1).replace(",", ""))
        except Exception:
            return default

    row = {
        "run_file": log_path.name,
        "ticker": m1(r"Data:\s+results/bursts_([A-Z]+)_"),
        "target": m1(r"Target:\s+(reg_[a-z0-9]+)"),
        "patch": m1(r"Patch:\s*(.+)"),
        "raw_bursts_total": m1(r"Dataset securely shrunk from\s*([0-9,]+)\s*to", int),
        "filtered_bursts": m1(r"Dataset securely shrunk from\s*[0-9,]+\s*to\s*([0-9,]+)", int),
        "dropped_nonfinite": m1(r"Dropping\s*([0-9,]+)\s*bursts", int, 0),
        "valid_bursts_scanned": m1(r"Total Valid Bursts Scanned:\s*([0-9,]+)", int),
        "trades": m1(r"Total Trades Fired:\s*([0-9,]+)", int),
        "longs": m1(r"Total Trades Fired:\s*[0-9,]+\s*\(([0-9,]+)\s*Long", int),
        "shorts": m1(r"Total Trades Fired:\s*[0-9,]+\s*\([0-9,]+\s*Long\s*/\s*([0-9,]+)\s*Short\)", int),
        "cum_pnl_raw": m1(r"Cumulative Simulated PnL \(raw\):\s*([-0-9,.]+)", float),
        "sharpe": m1(r"Annualized Sharpe Ratio:\s*([-0-9.]+)", float),
        "signals_eval": m1(r"Signals evaluated:\s*([0-9,]+)", int),
        "signals_long": m1(r"Signals passed long:\s*([0-9,]+)", int),
        "signals_short": m1(r"Signals passed short:\s*([0-9,]+)", int),
        "signals_rej": m1(r"Signals rejected:\s*([0-9,]+)", int),
    }

    run_stem = log_path.stem
    row["trades_csv"] = str((LOG_DIR / f"{run_stem}_debug_trades.csv").resolve())
    row["signals_csv"] = str((LOG_DIR / f"{run_stem}_debug_signals.csv").resolve())

    return row


def summarize_csvs(df_runs):
    extra = []
    for _, r in df_runs.iterrows():
        sig_p = Path(r["signals_csv"])
        trd_p = Path(r["trades_csv"])

        gate_med = np.nan
        pred_med = np.nan
        pred_abs_max = np.nan
        gate_abs_max = np.nan
        long_pass_rate = np.nan
        short_pass_rate = np.nan
        traded_signal_rate = np.nan

        if sig_p.exists():
            try:
                sig = pd.read_csv(sig_p)
            except EmptyDataError:
                sig = pd.DataFrame()
            if len(sig) > 0:
                pm = pd.to_numeric(sig.get("pred_move_per_share"), errors="coerce")
                gate = pd.to_numeric(sig.get("gate"), errors="coerce")
                side = pd.to_numeric(sig.get("signal_side"), errors="coerce").fillna(0)
                gate_med = float(np.nanmedian(gate)) if gate.notna().any() else np.nan
                pred_med = float(np.nanmedian(pm)) if pm.notna().any() else np.nan
                pred_abs_max = float(np.nanmax(np.abs(pm))) if pm.notna().any() else np.nan
                gate_abs_max = float(np.nanmax(np.abs(gate))) if gate.notna().any() else np.nan
                long_pass_rate = float((side > 0).mean())
                short_pass_rate = float((side < 0).mean())
                traded_signal_rate = float((side != 0).mean())

        trades_n = 0
        qty_mean = np.nan
        qty_p95 = np.nan
        qty_max = np.nan
        net_mean = np.nan
        net_p95 = np.nan
        net_max = np.nan
        net_min = np.nan
        if trd_p.exists():
            try:
                trd = pd.read_csv(trd_p)
            except EmptyDataError:
                trd = pd.DataFrame()
            trades_n = int(len(trd))
            if trades_n > 0:
                qty = pd.to_numeric(trd.get("qty"), errors="coerce")
                net = pd.to_numeric(trd.get("net_raw"), errors="coerce")
                qty_mean = float(np.nanmean(qty)) if qty.notna().any() else np.nan
                qty_p95 = float(np.nanpercentile(qty.dropna(), 95)) if qty.notna().any() else np.nan
                qty_max = float(np.nanmax(qty)) if qty.notna().any() else np.nan
                net_mean = float(np.nanmean(net)) if net.notna().any() else np.nan
                net_p95 = float(np.nanpercentile(net.dropna(), 95)) if net.notna().any() else np.nan
                net_max = float(np.nanmax(net)) if net.notna().any() else np.nan
                net_min = float(np.nanmin(net)) if net.notna().any() else np.nan

        extra.append({
            "run_file": r["run_file"],
            "trade_rate_vs_signals": traded_signal_rate,
            "long_pass_rate": long_pass_rate,
            "short_pass_rate": short_pass_rate,
            "gate_median": gate_med,
            "pred_move_median": pred_med,
            "pred_move_abs_max": pred_abs_max,
            "gate_abs_max": gate_abs_max,
            "trades_csv_rows": trades_n,
            "qty_mean": qty_mean,
            "qty_p95": qty_p95,
            "qty_max": qty_max,
            "net_mean": net_mean,
            "net_p95": net_p95,
            "net_max": net_max,
            "net_min": net_min,
        })

    return pd.DataFrame(extra)


def main():
    run_logs = sorted(LOG_DIR.glob("*.log"))
    runs = [parse_run_log(p) for p in run_logs]
    df_runs = pd.DataFrame(runs)
    df_extra = summarize_csvs(df_runs)
    df = df_runs.merge(df_extra, on="run_file", how="left")

    # derived diagnostics
    df["trade_rate_vs_valid_bursts"] = df["trades"] / df["valid_bursts_scanned"]
    df["filter_keep_rate"] = df["filtered_bursts"] / df["raw_bursts_total"]

    out_csv = OUT_DIR / f"overnight_{JOB_ID}_deep_diagnostics.csv"
    df.to_csv(out_csv, index=False)

    # ticker-level unique raw burst counts
    by_ticker_base = (
        df.sort_values(["ticker", "target"]) 
          .groupby("ticker", as_index=False)
          .first()[["ticker", "raw_bursts_total"]]
    )

    md = []
    md.append(f"# Overnight Deep Diagnostics ({JOB_ID})")
    md.append("")
    md.append("## Key Questions")
    md.append("- Why NVDA can show very large PnL with modest trade count")
    md.append("- Why TSLA has zero trades")
    md.append("- Why JPM reg_clop has very few trades")
    md.append("- How many bursts exist and what percent trigger trades")
    md.append("")

    md.append("## Burst Coverage (base files)")
    md.append(by_ticker_base.to_markdown(index=False))
    md.append("")

    show_cols = [
        "ticker", "target", "raw_bursts_total", "filtered_bursts", "filter_keep_rate", "valid_bursts_scanned",
        "signals_eval", "trades", "trade_rate_vs_valid_bursts", "trade_rate_vs_signals", "cum_pnl_raw", "sharpe"
    ]
    md.append("## Run Coverage + Trade Rates")
    md.append(df[show_cols].to_markdown(index=False, floatfmt=".6f"))
    md.append("")

    md.append("## Gate vs Prediction Scale")
    md.append(df[["ticker", "target", "gate_median", "pred_move_median", "pred_move_abs_max", "gate_abs_max", "long_pass_rate", "short_pass_rate"]]
              .to_markdown(index=False, floatfmt=".6f"))
    md.append("")

    md.append("## Position Size / PnL per Trade Diagnostics")
    md.append(df[["ticker", "target", "trades_csv_rows", "qty_mean", "qty_p95", "qty_max", "net_mean", "net_p95", "net_max", "net_min"]]
              .to_markdown(index=False, floatfmt=".6f"))
    md.append("")

    md.append("## Direct Findings")
    # NVDA note
    nv = df[(df["ticker"] == "NVDA") & (df["target"] == "reg_clcl")]
    if len(nv):
        r = nv.iloc[0]
        md.append(
            f"- NVDA `reg_clcl`: trades={int(r['trades'])}, cum_pnl_raw={r['cum_pnl_raw']:.2f}. "
            f"Mean qty={r['qty_mean']:.2f}, p95 qty={r['qty_p95']:.2f}, max qty={r['qty_max']:.2f}. "
            "Large size (volume-linked qty with position_size_mult=1.0) amplifies per-trade PnL."
        )
    ts = df[df["ticker"] == "TSLA"]
    if len(ts):
        for _, r in ts.iterrows():
            md.append(
                f"- TSLA `{r['target']}`: trades={int(r['trades'])}, trade_rate_vs_signals={r['trade_rate_vs_signals']:.6f}. "
                f"Pred abs max={r['pred_move_abs_max']:.6f} vs gate median={r['gate_median']:.6f}; "
                "cost-aware gate rejected all signals in this parameter regime."
            )
    jp = df[(df["ticker"] == "JPM") & (df["target"] == "reg_clop")]
    if len(jp):
        r = jp.iloc[0]
        md.append(
            f"- JPM `reg_clop`: trades={int(r['trades'])} out of signals={int(r['signals_eval'])} "
            f"(rate={r['trade_rate_vs_signals']:.6f}). Gate/prediction scale mismatch heavily suppresses entries."
        )
    md.append("- NaN rows are from non-finite targets/features after overnight label construction around missing/unaligned print windows; these are now dropped defensively before fit.")

    out_md = OUT_DIR / f"overnight_{JOB_ID}_deep_diagnostics.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
    print(df[["ticker", "target", "trades", "trade_rate_vs_signals", "cum_pnl_raw", "qty_mean", "qty_max"]].to_string(index=False))


if __name__ == "__main__":
    main()
