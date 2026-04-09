#!/usr/bin/env python3
import re
from pathlib import Path

import pandas as pd

JOB_ID = "12890943"
PULL_ROOT = Path("hoffman_pull_20260409_overnight")
JOB_LOG_DIR = PULL_ROOT / "logs"
RESULT_DIR = PULL_ROOT / "results" / "overnight_backtests"
ANALYSIS_DIR = PULL_ROOT / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def _to_float(s: str):
    s = s.replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _to_int(s: str):
    s = s.replace(",", "").strip()
    try:
        return int(s)
    except ValueError:
        return None


def parse_result_log(path: Path) -> dict:
    txt = path.read_text(errors="ignore")
    row = {
        "file": path.name,
        "ticker": None,
        "target": None,
        "patch": None,
        "dropped_rows": 0,
        "total_valid_bursts": None,
        "total_trades": None,
        "longs": None,
        "shorts": None,
        "cum_pnl_raw": None,
        "daily_mean_pnl_raw": None,
        "daily_std_pnl_raw": None,
        "sharpe": None,
        "signals_evaluated": None,
        "signals_long": None,
        "signals_short": None,
        "signals_rejected": None,
        "long_win_rate_pct": None,
        "short_win_rate_pct": None,
    }

    m = re.search(r"Patch:\s*(.+)", txt)
    if m:
        row["patch"] = m.group(1).strip()

    m = re.search(r"Target:\s*(reg_[a-z0-9]+)", txt)
    if m:
        row["target"] = m.group(1)

    m = re.search(r"Data:\s+results/bursts_([A-Z]+)_", txt)
    if m:
        row["ticker"] = m.group(1)

    m = re.search(r"Dropping\s+([0-9,]+)\s+bursts", txt)
    if m:
        row["dropped_rows"] = _to_int(m.group(1)) or 0

    m = re.search(r"Total Valid Bursts Scanned:\s*([0-9,]+)", txt)
    if m:
        row["total_valid_bursts"] = _to_int(m.group(1))

    m = re.search(r"Total Trades Fired:\s*([0-9,]+)\s*\(([0-9,]+) Long / ([0-9,]+) Short\)", txt)
    if m:
        row["total_trades"] = _to_int(m.group(1))
        row["longs"] = _to_int(m.group(2))
        row["shorts"] = _to_int(m.group(3))

    m = re.search(r"Cumulative Simulated PnL \(raw\):\s*([-0-9.,]+)", txt)
    if m:
        row["cum_pnl_raw"] = _to_float(m.group(1))

    m = re.search(r"Daily Mean PnL \(raw\):\s*([-0-9.,]+)", txt)
    if m:
        row["daily_mean_pnl_raw"] = _to_float(m.group(1))

    m = re.search(r"Daily StdDev \(raw\):\s*([-0-9.,]+)", txt)
    if m:
        row["daily_std_pnl_raw"] = _to_float(m.group(1))

    m = re.search(r"Annualized Sharpe Ratio:\s*([-0-9.]+)", txt)
    if m:
        row["sharpe"] = _to_float(m.group(1))

    m = re.search(r"Signals evaluated:\s*([0-9,]+)", txt)
    if m:
        row["signals_evaluated"] = _to_int(m.group(1))

    m = re.search(r"Signals passed long:\s*([0-9,]+)", txt)
    if m:
        row["signals_long"] = _to_int(m.group(1))

    m = re.search(r"Signals passed short:\s*([0-9,]+)", txt)
    if m:
        row["signals_short"] = _to_int(m.group(1))

    m = re.search(r"Signals rejected:\s*([0-9,]+)", txt)
    if m:
        row["signals_rejected"] = _to_int(m.group(1))

    m = re.search(r"Long\s+trades=\s*[0-9,]+\s+win_rate=\s*([0-9.]+)%", txt)
    if m:
        row["long_win_rate_pct"] = _to_float(m.group(1))

    m = re.search(r"Short\s+trades=\s*[0-9,]+\s+win_rate=\s*([0-9.]+)%", txt)
    if m:
        row["short_win_rate_pct"] = _to_float(m.group(1))

    return row


def parse_job_out(path: Path) -> dict:
    txt = path.read_text(errors="ignore")
    return {
        "file": path.name,
        "completed": "Overnight backtests complete for" in txt,
        "has_traceback": "Traceback" in txt,
        "has_nan_error": "Input y contains NaN" in txt,
        "has_line_error": "ERROR: line" in txt,
    }


def main() -> int:
    result_logs = sorted(RESULT_DIR.glob("*.log"))
    job_logs = sorted(JOB_LOG_DIR.glob(f"overnight_bt_{JOB_ID}_*.out"))

    if not result_logs:
        print(f"No result logs found in {RESULT_DIR}")
        return 1

    rows = [parse_result_log(p) for p in result_logs]
    df = pd.DataFrame(rows)
    df = df.sort_values(["ticker", "target", "file"]).reset_index(drop=True)

    out_checks = pd.DataFrame([parse_job_out(p) for p in job_logs])

    csv_path = ANALYSIS_DIR / f"overnight_{JOB_ID}_metrics.csv"
    df.to_csv(csv_path, index=False)

    by_target = (
        df.groupby("target", dropna=False)
        .agg(
            runs=("file", "count"),
            trades=("total_trades", "sum"),
            cum_pnl_raw=("cum_pnl_raw", "sum"),
            mean_sharpe=("sharpe", "mean"),
            dropped_rows=("dropped_rows", "sum"),
        )
        .reset_index()
    )

    by_ticker = (
        df.groupby("ticker", dropna=False)
        .agg(
            runs=("file", "count"),
            trades=("total_trades", "sum"),
            cum_pnl_raw=("cum_pnl_raw", "sum"),
            mean_sharpe=("sharpe", "mean"),
            dropped_rows=("dropped_rows", "sum"),
        )
        .reset_index()
    )

    md_path = ANALYSIS_DIR / f"overnight_{JOB_ID}_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Overnight Backtest Summary (Job {JOB_ID})\n\n")
        f.write("## Execution Health\n")
        f.write(f"- Job out files: {len(out_checks)}\n")
        f.write(f"- Completed markers: {int(out_checks['completed'].sum())}/{len(out_checks)}\n")
        f.write(f"- Tracebacks: {int(out_checks['has_traceback'].sum())}\n")
        f.write(f"- NaN errors: {int(out_checks['has_nan_error'].sum())}\n")
        f.write(f"- Shell line errors: {int(out_checks['has_line_error'].sum())}\n\n")

        f.write("## Run-Level Metrics\n")
        f.write(df[[
            "ticker", "target", "patch", "dropped_rows", "total_valid_bursts", "total_trades",
            "longs", "shorts", "cum_pnl_raw", "sharpe", "signals_evaluated"
        ]].to_markdown(index=False))
        f.write("\n\n")

        f.write("## Aggregate by Target\n")
        f.write(by_target.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Aggregate by Ticker\n")
        f.write(by_ticker.to_markdown(index=False))
        f.write("\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(df[["ticker", "target", "cum_pnl_raw", "sharpe", "total_trades", "dropped_rows"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
