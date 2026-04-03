#!/usr/bin/env python3
"""
silence_optimized_sweep.py

Efficient parameter search workflow:
1) Parse message files ONCE per silence threshold (-s) with minimal filtering.
2) Compute permanence ONCE per precomputed burst CSV.
3) Apply post-filters (-v, -d, -r, -k) in Python and train model(s).

This avoids reparsing raw message files for every parameter combination.
"""

import argparse
import csv
import itertools
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run(cmd):
    subprocess.run(cmd, check=True)


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def classify_and_filter(df, min_vol, dir_thresh, vol_ratio, kappa, require_directional):
    out = df.copy()

    # Ensure required columns exist.
    required = [
        "Volume", "BuyCount", "SellCount", "BuyVolume", "SellVolume", "D_b",
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns for post-filtering: {missing}")

    out["TradeCount"] = out["BuyCount"] + out["SellCount"]
    total = out["TradeCount"].clip(lower=1)

    out["BuyRatioPost"] = out["BuyCount"] / total
    out["SellRatioPost"] = out["SellCount"] / total

    major_count = np.maximum(out["BuyRatioPost"], out["SellRatioPost"])
    major_vol = np.maximum(out["BuyVolume"], out["SellVolume"])
    minor_vol = np.minimum(out["BuyVolume"], out["SellVolume"])
    out["MinMaxVolRatioPost"] = np.where(major_vol > 0, minor_vol / major_vol, 1.0)

    buy_wins = out["BuyRatioPost"] >= out["SellRatioPost"]
    directional_ok = (major_count >= dir_thresh) & (out["MinMaxVolRatioPost"] <= vol_ratio)

    out["Direction"] = 0
    out.loc[directional_ok & buy_wins, "Direction"] = 1
    out.loc[directional_ok & (~buy_wins), "Direction"] = -1

    mask = out["Volume"] >= min_vol
    if require_directional:
        mask &= out["Direction"] != 0
    if kappa > 0:
        mask &= out["D_b"].notna() & (out["D_b"] >= kappa)

    return out[mask].copy()


def find_score(result_json):
    with open(result_json) as f:
        res = json.load(f)
    pooled = res.get("pooled", {})
    if "AUC" in pooled:
        return "AUC", pooled["AUC"]
    if "MAE" in pooled:
        return "MAE", pooled["MAE"]
    return "Metric", None


def write_summary(rows, output_path):
    fieldnames = [
        "ticker", "config", "target", "silence", "min_vol",
        "dir_thresh", "vol_ratio", "kappa", "rows", "metric_name", "metric_value",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Efficient sweep: parse once per silence, filter/train many combos.")
    ap.add_argument("--stock-folder", required=True, help="Path to one stock folder with message CSVs")
    ap.add_argument("--ticker", required=True, help="Ticker symbol for output naming")
    ap.add_argument("--open", required=True, help="open_all.csv path")
    ap.add_argument("--close", required=True, help="close_all.csv path")
    ap.add_argument("--data-processor", default="./data_processor", help="Path to C++ binary")
    ap.add_argument("--workers", type=int, default=1, help="Parallel day workers for data_processor (-j)")
    ap.add_argument("--outdir", default="results/silence_sweep", help="Output root")
    ap.add_argument("--precompute-dir", default=None,
                    help="Optional shared cache dir for precompute/permanence files across phases")

    ap.add_argument("--silence-values", required=True, help="Comma list, e.g. 0.5,1,2")
    ap.add_argument("--min-vol-values", required=True, help="Comma list, e.g. 50,100,200")
    ap.add_argument("--dir-thresh-values", required=True, help="Comma list, e.g. 0.8,0.9")
    ap.add_argument("--vol-ratio-values", required=True, help="Comma list, e.g. 0.3,0.5")
    ap.add_argument("--kappa-values", required=True, help="Comma list, e.g. 0,0.1,0.2")

    ap.add_argument("--tau-max", type=float, default=10.0, help="Fixed -t value for C++ precompute")
    ap.add_argument("--rth-start", type=float, default=34200.0, help="Fixed -b")
    ap.add_argument("--rth-end", type=float, default=57600.0, help="Fixed -e")

    ap.add_argument("--model", default="logreg_l2", help="train_model_zoo --model")
    ap.add_argument("--target", default="cls_close",
                    help="Comma-separated target keys for train_model_zoo --target, e.g. cls_1m,cls_5m,cls_close")
    ap.add_argument("--features", default="extended", choices=["base", "extended"])
    ap.add_argument("--min-train-months", type=int, default=3)
    ap.add_argument("--require-directional", action="store_true", help="Drop mixed bursts after post-classification")
    ap.add_argument("--min-rows", type=int, default=500, help="Skip configs with too few bursts")
    ap.add_argument("--skip-existing", action="store_true",
                    help="If all expected result JSONs exist for a config, reuse them and skip retraining")
    ap.add_argument("--write-skipped-candidates", action="store_true",
                    help="Also write candidate CSV for configs below --min-rows (default: do not write)")

    args = ap.parse_args()
    target_list = [t.strip() for t in args.target.split(",") if t.strip()]
    if not target_list:
        raise ValueError("No valid targets provided via --target")

    short_targets = {"cls_1m", "cls_3m", "cls_5m", "cls_10m", "reg_1m", "reg_3m", "reg_5m", "reg_10m"}
    long_targets = {"cls_close", "cls_clop", "cls_clcl", "reg_close", "reg_clop", "reg_clcl"}
    has_short_horizon_target = any(t in short_targets for t in target_list)
    has_long_horizon_target = any(t in long_targets for t in target_list)

    silence_values = parse_float_list(args.silence_values)
    min_vol_values = parse_int_list(args.min_vol_values)
    dir_thresh_values = parse_float_list(args.dir_thresh_values)
    vol_ratio_values = parse_float_list(args.vol_ratio_values)
    kappa_values = parse_float_list(args.kappa_values)

    # kappa filtering is meaningful only for long-horizon targets.
    # If short and long targets are mixed in one run, one shared candidate set
    # cannot be simultaneously correct for both with nonzero kappa.
    if has_short_horizon_target and has_long_horizon_target and any(k > 0 for k in kappa_values):
        raise ValueError(
            "Mixed short+long targets with nonzero --kappa-values are unsupported. "
            "Run two sweeps (short targets with kappa=0, long targets with kappa grid)."
        )

    outdir = Path(args.outdir)
    precompute_dir = Path(args.precompute_dir) if args.precompute_dir else (outdir / "precompute")
    candidates_dir = outdir / "candidates"
    model_dir = outdir / "models"
    for d in [outdir, precompute_dir, candidates_dir, model_dir]:
        d.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    summary_csv = outdir / "sweep_summary.csv"
    progress_counter = 0
    checkpoint_every = 5

    for s in silence_values:
        s_tag = str(s).replace(".", "p")
        raw_csv = precompute_dir / f"bursts_{args.ticker}_s{s_tag}.csv"
        perm_csv = raw_csv.with_name(raw_csv.stem + "_filtered" + raw_csv.suffix)

        # Precompute bursts once per silence threshold with minimal filtering.
        if not raw_csv.exists():
            run([
                args.data_processor,
                args.stock_folder,
                str(raw_csv),
                "-s", str(s),
                "-v", "1",
                "-d", "0.5",
                "-r", "1.0",
                "-k", "0",
                "-t", str(args.tau_max),
                "-j", str(max(1, args.workers)),
                "-b", str(args.rth_start),
                "-e", str(args.rth_end),
            ])
        else:
            print(f"[cache] Reusing existing precompute file: {raw_csv}")

        # Add permanence columns once (kappa disabled here).
        if not perm_csv.exists():
            run([
                "python3", "src_py/compute_permanence.py",
                str(raw_csv),
                args.open,
                args.close,
                "--kappa", "0",
            ])
        else:
            print(f"[cache] Reusing existing permanence file: {perm_csv}")

        base_df = pd.read_csv(perm_csv)

        for min_vol, dth, vr, k in itertools.product(
            min_vol_values, dir_thresh_values, vol_ratio_values, kappa_values
        ):
            effective_kappa = 0.0 if has_short_horizon_target else k
            filtered = classify_and_filter(
                base_df,
                min_vol=min_vol,
                dir_thresh=dth,
                vol_ratio=vr,
                kappa=effective_kappa,
                require_directional=args.require_directional,
            )

            config_tag = (
                f"s{s_tag}_v{min_vol}_d{dth}_r{vr}_k{effective_kappa}".replace(".", "p")
            )
            candidate_csv = candidates_dir / f"{args.ticker}_{config_tag}.csv"
            out_model_dir = model_dir / config_tag
            out_model_dir.mkdir(parents=True, exist_ok=True)

            expected_jsons = [out_model_dir / f"{args.model}__{tgt}.json" for tgt in target_list]

            if args.skip_existing and all(p.exists() for p in expected_jsons):
                rows_count = len(filtered)
                for tgt, result_json in zip(target_list, expected_jsons):
                    metric_name, metric_value = find_score(result_json)
                    summary_rows.append({
                        "ticker": args.ticker,
                        "config": config_tag,
                        "target": tgt,
                        "silence": s,
                        "min_vol": min_vol,
                        "dir_thresh": dth,
                        "vol_ratio": vr,
                        "kappa": k,
                        "rows": rows_count,
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                    })
                    progress_counter += 1
                    if progress_counter % checkpoint_every == 0:
                        write_summary(summary_rows, summary_csv)
                continue

            if len(filtered) < args.min_rows:
                if args.write_skipped_candidates:
                    filtered.to_csv(candidate_csv, index=False)
                summary_rows.append({
                    "ticker": args.ticker,
                    "config": config_tag,
                    "silence": s,
                    "min_vol": min_vol,
                    "dir_thresh": dth,
                    "vol_ratio": vr,
                    "kappa": k,
                    "rows": len(filtered),
                    "metric_name": "SKIP",
                    "metric_value": "",
                })
                progress_counter += 1
                if progress_counter % checkpoint_every == 0:
                    write_summary(summary_rows, summary_csv)
                continue

            # Persist trainable candidate sets; these are needed for top-config replay.
            if not candidate_csv.exists():
                filtered.to_csv(candidate_csv, index=False)

            run([
                "python3", "src_py/train_model_zoo.py",
                str(candidate_csv),
                "--model", args.model,
                "--target", ",".join(target_list),
                "--features", args.features,
                "--outdir", str(out_model_dir),
                "--min-train-months", str(args.min_train_months),
            ])

            for tgt in target_list:
                result_json = out_model_dir / f"{args.model}__{tgt}.json"
                if not result_json.exists():
                    summary_rows.append({
                        "ticker": args.ticker,
                        "config": config_tag,
                        "target": tgt,
                        "silence": s,
                        "min_vol": min_vol,
                        "dir_thresh": dth,
                        "vol_ratio": vr,
                        "kappa": k,
                        "rows": len(filtered),
                        "metric_name": "MISSING",
                        "metric_value": "",
                    })
                    progress_counter += 1
                    if progress_counter % checkpoint_every == 0:
                        write_summary(summary_rows, summary_csv)
                    continue

                metric_name, metric_value = find_score(result_json)
                summary_rows.append({
                    "ticker": args.ticker,
                    "config": config_tag,
                    "target": tgt,
                    "silence": s,
                    "min_vol": min_vol,
                    "dir_thresh": dth,
                    "vol_ratio": vr,
                    "kappa": k,
                    "rows": len(filtered),
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                })
                progress_counter += 1
                if progress_counter % checkpoint_every == 0:
                    write_summary(summary_rows, summary_csv)

    write_summary(summary_rows, summary_csv)

    valid_rows = [r for r in summary_rows if isinstance(r.get("metric_value"), (int, float))]
    if valid_rows:
        metric_name = valid_rows[0]["metric_name"]
        reverse = metric_name.upper() == "AUC"
        ranked = sorted(valid_rows, key=lambda r: r["metric_value"], reverse=reverse)
        best = ranked[0]
        best_json = outdir / "best_config.json"
        with open(best_json, "w") as f:
            json.dump(best, f, indent=2)

        ranked_csv = outdir / "ranked_configs.csv"
        pd.DataFrame(ranked).to_csv(ranked_csv, index=False)

        # Plot top configurations for quick visual comparison.
        # Chart top rows for the first target in the requested list.
        chart_target = target_list[0]
        target_ranked = [r for r in ranked if r.get("target") == chart_target]
        top_n = min(25, len(target_ranked))
        top_df = pd.DataFrame(target_ranked[:top_n]).copy() if top_n > 0 else pd.DataFrame()
        if top_n > 0:
            plt.figure(figsize=(14, 6))
            plt.bar(top_df["config"], top_df["metric_value"])
            plt.xticks(rotation=75, ha="right", fontsize=8)
            plt.ylabel(metric_name)
            plt.title(f"Top {top_n} parameter combinations for {chart_target} ({metric_name})")
            plt.tight_layout()
            chart_path = outdir / f"top_configs_{chart_target}.png"
            plt.savefig(chart_path, dpi=140)
            plt.close()
            print(f"Saved: {chart_path}")

        print(f"Best config ({metric_name}): {best['config']} -> {best['metric_value']}")
        print(f"Saved: {best_json}")
        print(f"Saved: {ranked_csv}")

    print(f"Saved sweep summary: {summary_csv}")


if __name__ == "__main__":
    main()
