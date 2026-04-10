#!/usr/bin/env python3
import csv
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUT_DIR = RESULTS_DIR / "research_digest"

TICKERS = ["NVDA", "TSLA", "JPM", "MS"]


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_text_safe(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return ""


def parse_log_filename(name: str):
    # Expected patterns:
    #   <prefix>_<jobid>_<taskid>.out
    #   <prefix>_<jobid>.out
    if not (name.endswith(".out") or name.endswith(".log")):
        return "unknown", "", ""

    stem = name[:-4]
    parts = stem.split("_")
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        return "_".join(parts[:-2]), parts[-2], parts[-1]
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1]), parts[-1], ""
    return stem, "", ""


def classify_log_purpose(prefix: str) -> str:
    mapping = {
        "optuna_physical": "Optuna physical parameter search",
        "sweep": "Static volume sweep",
        "sweep_frac": "Fractional ADV sweep",
        "eval_topcfg": "Top-config model eval (legacy workflow)",
        "post_sweep": "Post-sweep ranking merge (legacy workflow)",
        "backtest": "Backtest run",
        "pnl_sim": "PnL simulation run",
    }
    return mapping.get(prefix, "Unknown/other")


def classify_log_status(text: str) -> str:
    if re.search(r"Traceback|ModuleNotFoundError|CalledProcessError|ImportError|ERROR: line", text, re.IGNORECASE):
        return "error"
    if re.search(r"Completed:|task complete|finished at|Sweep complete|Fractional sweep complete", text, re.IGNORECASE):
        return "completed"
    return "unknown"


def recommended_action(log_location: str, prefix: str, status: str) -> str:
    if status == "error":
        return "keep_for_debug"
    if "logs_hoffman2" in log_location:
        return "archive_or_delete_after_backup"
    if prefix in {"optuna_physical", "sweep", "sweep_frac"}:
        return "archive_after_metrics_snapshot"
    return "archive_or_delete_after_backup"


def extract_text_field(text: str, pattern: str) -> str:
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def build_log_catalog():
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in {".out", ".log"}:
            continue
        if ".git" in path.parts:
            continue

        text = read_text_safe(path)
        prefix, job_id, task_id = parse_log_filename(path.name)
        status = classify_log_status(text)
        stat = path.stat()
        rel_parent = str(path.parent.relative_to(ROOT))
        rows.append(
            {
                "log_dir": rel_parent,
                "file": path.name,
                "prefix": prefix,
                "job_id": job_id,
                "task_id": task_id,
                "purpose": classify_log_purpose(prefix),
                "status": status,
                "size_kb": round(stat.st_size / 1024.0, 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "ticker_hint": extract_text_field(text, r"ticker\s*=\s*([A-Z]{1,5})|Ticker:\s*([A-Z]{1,5})"),
                "target_hint": extract_text_field(text, r"Target:\s*([a-zA-Z0-9_]+)"),
                "contains_nan_error": bool(re.search(r"Input y contains NaN", text)),
                "contains_traceback": bool(re.search(r"Traceback", text)),
                "recommended_action": recommended_action(rel_parent, prefix, status),
            }
        )
    return rows


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def build_hoffman_pull_index():
    rows = []
    metrics_files = sorted(ROOT.glob("hoffman_pull_*/analysis/*_metrics.csv"))
    consolidated_files_dir = RESULTS_DIR / "hoffman_pull_consolidated" / "files"
    if consolidated_files_dir.exists():
        metrics_files.extend(sorted(consolidated_files_dir.glob("*_metrics.csv")))

    for metrics in metrics_files:
        if consolidated_files_dir in metrics.parents:
            # Consolidated format: 02_hoffman_pull_...__analysis__..._metrics.csv
            run_folder = re.sub(r"^\d+_", "", metrics.name.split("__", 1)[0])
        else:
            run_folder = metrics.parents[1].name
        run_label = metrics.stem.replace("_metrics", "")
        with metrics.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = list(reader)

        if not data:
            continue

        total_trades = 0.0
        total_pnl = 0.0
        sharpe_vals = []
        tickers = set()
        targets = set()

        for row in data:
            tickers.add(row.get("ticker", ""))
            targets.add(row.get("target", ""))
            try:
                total_trades += float(row.get("trades") or row.get("total_trades") or 0.0)
            except Exception:
                pass
            try:
                total_pnl += float(row.get("cum_pnl_raw") or 0.0)
            except Exception:
                pass
            try:
                sharpe_vals.append(float(row.get("sharpe") or 0.0))
            except Exception:
                pass

        mean_sharpe = sum(sharpe_vals) / len(sharpe_vals) if sharpe_vals else 0.0
        rows.append(
            {
                "run_folder": run_folder,
                "metrics_file": str(metrics.relative_to(ROOT)),
                "run_label": run_label,
                "rows": len(data),
                "tickers": ",".join(sorted(t for t in tickers if t)),
                "targets": ",".join(sorted(t for t in targets if t)),
                "total_trades": round(total_trades, 2),
                "total_cum_pnl_raw": round(total_pnl, 2),
                "mean_sharpe": round(mean_sharpe, 4),
            }
        )

    return rows


def build_sweep_coverage_matrix():
    rows = []
    for ticker in TICKERS:
        static_root = RESULTS_DIR / f"silence_sweep_{ticker}" / "logreg_l2"
        frac_root = RESULTS_DIR / f"silence_sweep_frac_{ticker}" / "logreg_l2"
        rows.append(
            {
                "ticker": ticker,
                "static_short_exists": (static_root / "short").is_dir(),
                "static_long_exists": (static_root / "long").is_dir(),
                "frac_short_exists": (frac_root / "short").is_dir(),
                "frac_long_exists": (frac_root / "long").is_dir(),
                "static_dir_exists": static_root.is_dir(),
                "frac_dir_exists": frac_root.is_dir(),
            }
        )
    return rows


def parse_leaderboard_top_rows(path: Path, top_n: int = 3):
    text = read_text_safe(path)
    generated = extract_text_field(text, r"Generated:\s*([^\n]+)")

    rows = []
    in_table = False
    for line in text.splitlines():
        if line.startswith("| Rank | Model | Target |"):
            in_table = True
            continue
        if in_table and line.startswith("|------"):
            continue
        if in_table and line.startswith("|"):
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) >= 8 and parts[0].isdigit():
                rows.append(
                    {
                        "rank": parts[0],
                        "model": parts[1],
                        "target": parts[2],
                        "auc": parts[3],
                        "accuracy": parts[4],
                        "f1": parts[5],
                    }
                )
                if len(rows) >= top_n:
                    break
        if in_table and line.strip() == "":
            break

    return generated, rows


def build_stock_params_inventory():
    rows = []
    for ticker in TICKERS:
        folder = RESULTS_DIR / f"{ticker}_params"
        for horizon in ["short", "long"]:
            md = folder / f"{horizon}_leaderboard.md"
            generated, top_rows = parse_leaderboard_top_rows(md)
            if top_rows:
                for tr in top_rows:
                    rows.append(
                        {
                            "ticker": ticker,
                            "horizon": horizon,
                            "source_file": str(md.relative_to(ROOT)),
                            "generated": generated,
                            "rank": tr["rank"],
                            "model": tr["model"],
                            "target": tr["target"],
                            "auc": tr["auc"],
                            "accuracy": tr["accuracy"],
                            "f1": tr["f1"],
                        }
                    )
            else:
                rows.append(
                    {
                        "ticker": ticker,
                        "horizon": horizon,
                        "source_file": str(md.relative_to(ROOT)),
                        "generated": generated,
                        "rank": "",
                        "model": "",
                        "target": "",
                        "auc": "",
                        "accuracy": "",
                        "f1": "",
                    }
                )
    return rows


def write_markdown_summary(log_rows, pull_rows, coverage_rows, params_rows):
    by_dir = {}
    for r in log_rows:
        by_dir.setdefault(r["log_dir"], 0)
        by_dir[r["log_dir"]] += 1

    error_logs = [r for r in log_rows if r["status"] == "error"]
    frac_missing_long = [r["ticker"] for r in coverage_rows if not r["frac_long_exists"]]

    md = []
    md.append("# Research Digest Summary")
    md.append("")
    md.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    md.append("")
    md.append("## What This Digest Contains")
    md.append("- `research_log_catalog.csv`: one row per `.out` log with inferred purpose/status and cleanup recommendation.")
    md.append("- `research_hoffman_pull_index.csv`: rolled-up metrics from `hoffman_pull_*/analysis/*_metrics.csv`.")
    md.append("- `research_sweep_coverage.csv`: matrix of static/fractional short/long folder presence by ticker.")
    md.append("- `research_stock_params_inventory.csv`: top-ranked models from each `results/*_params/*_leaderboard.md`.")
    md.append("")

    md.append("## Log Folder Interpretation")
    md.append("- `logs/`: newer active runs in the current workflow (Optuna + backtests).")
    md.append("- `logs_hoffman2/`: older/legacy run outputs from previous sweep/eval workflow snapshots.")
    md.append("")
    md.append("## Log Counts")
    for k, v in sorted(by_dir.items()):
        md.append(f"- `{k}`: {v} files")
    md.append(f"- Error-status logs: {len(error_logs)}")
    md.append("")

    md.append("## Sweep Coverage Gaps")
    if frac_missing_long:
        md.append(f"- Fractional sweep missing `long/` for: {', '.join(frac_missing_long)}")
    else:
        md.append("- Fractional sweep has `long/` for all tracked tickers.")
    md.append("")

    md.append("## `*_params` Folder Meaning")
    md.append("Each `results/<TICKER>_params/` folder stores model-zoo leaderboard snapshots:")
    md.append("- `short_leaderboard.md`: short-horizon classification targets (`cls_1m/3m/5m/10m`).")
    md.append("- `long_leaderboard.md`: long-horizon targets (`cls_close/clop/clcl`).")
    md.append("These are ranking reports (AUC/accuracy/F1/Brier/time), not trade-PnL backtests.")
    md.append("")

    md.append("## Hoffman Pull Index (High-Level)")
    for r in pull_rows:
        md.append(
            f"- `{r['run_folder']}` / `{r['run_label']}`: trades={r['total_trades']}, "
            f"cum_pnl_raw={r['total_cum_pnl_raw']}, mean_sharpe={r['mean_sharpe']}, "
            f"targets={r['targets']}"
        )
    md.append("")

    md.append("## Recommended Cleanup (No Deletions Performed)")
    md.append("- Safe first-pass archive/delete candidates: most `logs_hoffman2/*.out` after confirming CSV metrics already exist.")
    md.append("- Keep for reproducibility/debug now: recent `logs/optuna_physical_*.out` and any log flagged `error` in `research_log_catalog.csv`.")
    md.append("- `results/silence_sweep_MS/` should only be deleted if you intentionally drop MS from static sweep baselines; otherwise keep for cross-ticker consistency.")

    (OUT_DIR / "RESEARCH_DIGEST_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main():
    ensure_out_dir()

    log_rows = build_log_catalog()
    write_csv(
        OUT_DIR / "research_log_catalog.csv",
        log_rows,
        [
            "log_dir",
            "file",
            "prefix",
            "job_id",
            "task_id",
            "purpose",
            "status",
            "size_kb",
            "modified",
            "ticker_hint",
            "target_hint",
            "contains_nan_error",
            "contains_traceback",
            "recommended_action",
        ],
    )

    pull_rows = build_hoffman_pull_index()
    write_csv(
        OUT_DIR / "research_hoffman_pull_index.csv",
        pull_rows,
        [
            "run_folder",
            "metrics_file",
            "run_label",
            "rows",
            "tickers",
            "targets",
            "total_trades",
            "total_cum_pnl_raw",
            "mean_sharpe",
        ],
    )

    coverage_rows = build_sweep_coverage_matrix()
    write_csv(
        OUT_DIR / "research_sweep_coverage.csv",
        coverage_rows,
        [
            "ticker",
            "static_short_exists",
            "static_long_exists",
            "frac_short_exists",
            "frac_long_exists",
            "static_dir_exists",
            "frac_dir_exists",
        ],
    )

    params_rows = build_stock_params_inventory()
    write_csv(
        OUT_DIR / "research_stock_params_inventory.csv",
        params_rows,
        [
            "ticker",
            "horizon",
            "source_file",
            "generated",
            "rank",
            "model",
            "target",
            "auc",
            "accuracy",
            "f1",
        ],
    )

    write_markdown_summary(log_rows, pull_rows, coverage_rows, params_rows)

    print(f"Wrote digest files to: {OUT_DIR}")


if __name__ == "__main__":
    main()
