#!/usr/bin/env python3
"""
analyze_zoo.py — Post-hoc analysis of Model Zoo results.

Aggregates all per-model JSON results, builds leaderboards, comparison
plots, and a publication-quality summary.

Usage:
    python src_py/analyze_zoo.py results/zoo_NVDA/
"""

import pandas as pd
import numpy as np
import argparse
import os
import json
import glob
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, TwoSlopeNorm


def load_results(zoo_dir):
    """Load all model result JSONs from a zoo directory."""
    results = []
    for path in sorted(glob.glob(os.path.join(zoo_dir, '*__*.json'))):
        with open(path) as f:
            r = json.load(f)
            r['_file'] = os.path.basename(path)
            results.append(r)
    return results


def build_leaderboard(results, task='binary'):
    """Build a DataFrame leaderboard."""
    rows = []
    for r in results:
        if r['task_type'] != task:
            continue
        p = r['pooled']
        monthly = r.get('monthly', {})
        monthly_vals = [v for v in monthly.values() if isinstance(v, (int, float))]

        row = {
            'Model': r['model_name'],
            'Key': r['model_key'],
            'Target': r['target'],
            'Folds': r['n_folds'],
            'Time (s)': r['elapsed_sec'],
        }

        if task == 'binary':
            row.update({
                'AUC (pooled)': p.get('AUC', np.nan),
                'Accuracy':     p.get('Accuracy', np.nan),
                'Precision':    p.get('Precision', np.nan),
                'Recall':       p.get('Recall', np.nan),
                'F1':           p.get('F1', np.nan),
                'Brier':        p.get('Brier', np.nan),
                'AUC mean':     np.mean(monthly_vals) if monthly_vals else np.nan,
                'AUC std':      np.std(monthly_vals) if monthly_vals else np.nan,
                'AUC min':      np.min(monthly_vals) if monthly_vals else np.nan,
                'AUC max':      np.max(monthly_vals) if monthly_vals else np.nan,
            })
        elif task == 'regression':
            row.update({
                'MAE':    p.get('MAE', np.nan),
                'RMSE':   p.get('RMSE', np.nan),
                'R²':     p.get('R2', np.nan),
                'DirAcc': p.get('DirAcc', np.nan),
            })

        rows.append(row)

    df = pd.DataFrame(rows)
    if task == 'binary' and 'AUC (pooled)' in df.columns:
        df = df.sort_values('AUC (pooled)', ascending=False).reset_index(drop=True)
        df.index += 1
        df.index.name = 'Rank'
    elif task == 'regression' and 'MAE' in df.columns:
        df = df.sort_values('MAE', ascending=True).reset_index(drop=True)
        df.index += 1
        df.index.name = 'Rank'

    return df


def plot_auc_leaderboard(df_cls, outdir):
    """Horizontal bar chart of AUC across all models."""
    if df_cls.empty:
        return

    # Only cls_close for the main chart
    df = df_cls[df_cls['Target'] == 'cls_close'].copy()
    if df.empty:
        df = df_cls.copy()

    df = df.sort_values('AUC (pooled)', ascending=True).tail(25)

    fig, ax = plt.subplots(figsize=(12, max(5, len(df) * 0.4)))

    colors = []
    for auc in df['AUC (pooled)']:
        if auc > 0.58:   colors.append('#1B5E20')
        elif auc > 0.55: colors.append('#4CAF50')
        elif auc > 0.52: colors.append('#81C784')
        elif auc > 0.50: colors.append('#C8E6C9')
        else:            colors.append('#FFCDD2')

    bars = ax.barh(range(len(df)), df['AUC (pooled)'], color=colors,
                   edgecolor='white', linewidth=0.5)

    # Error bars from AUC std
    if 'AUC std' in df.columns:
        ax.errorbar(df['AUC (pooled)'], range(len(df)),
                    xerr=df['AUC std'], fmt='none', ecolor='gray',
                    elinewidth=0.8, capsize=2)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Model'], fontsize=9)
    ax.set_xlabel('Pooled AUC (walk-forward CV)', fontsize=11)
    ax.set_title('Model Zoo — Classification AUC Leaderboard\n'
                 '(permanence > 1, walk-forward monthly folds)',
                 fontweight='bold', fontsize=13)
    ax.axvline(0.5, color='red', ls='--', lw=1.2, alpha=0.7, label='Random baseline')
    ax.legend(fontsize=9)

    for bar, auc in zip(bars, df['AUC (pooled)']):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f'{auc:.4f}', va='center', fontsize=8, fontweight='bold')

    ax.set_xlim(0.45, df['AUC (pooled)'].max() + 0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'leaderboard_auc.png'), dpi=200)
    plt.close(fig)


def plot_horizon_comparison(results, outdir):
    """Compare AUC across different prediction horizons for each model."""
    horizons = ['cls_1m', 'cls_3m', 'cls_5m', 'cls_10m', 'cls_close']
    horizon_labels = ['1 min', '3 min', '5 min', '10 min', 'Close']

    # Get models that have results for multiple horizons
    model_data = {}
    for r in results:
        if r['task_type'] != 'binary':
            continue
        mk = r['model_key']
        tk = r['target']
        if tk in horizons:
            if mk not in model_data:
                model_data[mk] = {'name': r['model_name']}
            model_data[mk][tk] = r['pooled'].get('AUC', np.nan)

    # Only keep models with ≥3 horizons
    model_data = {k: v for k, v in model_data.items()
                  if sum(h in v for h in horizons) >= 3}

    if len(model_data) < 2:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(horizons))
    width = 0.8 / len(model_data)

    for i, (mk, data) in enumerate(sorted(model_data.items(),
            key=lambda x: x[1].get('cls_close', 0), reverse=True)[:8]):
        aucs = [data.get(h, np.nan) for h in horizons]
        offset = (i - len(model_data)/2 + 0.5) * width
        ax.bar(x + offset, aucs, width * 0.9, label=data['name'], alpha=0.85)

    ax.axhline(0.5, color='red', ls='--', lw=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(horizon_labels)
    ax.set_ylabel('AUC')
    ax.set_title('AUC by Prediction Horizon — Top Models', fontweight='bold')
    ax.legend(fontsize=7, ncol=2, loc='upper left')
    ax.set_ylim(0.45, ax.get_ylim()[1] + 0.03)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'horizon_comparison.png'), dpi=150)
    plt.close(fig)


def plot_monthly_heatmap(results, outdir):
    """Heatmap of monthly AUC for cls_close models."""
    close_results = [r for r in results
                     if r['task_type'] == 'binary' and r['target'] == 'cls_close']
    if len(close_results) < 2:
        return

    close_results.sort(key=lambda r: r['pooled'].get('AUC', 0), reverse=True)
    top = close_results[:min(15, len(close_results))]

    all_months = sorted(set(m for r in top for m in r.get('monthly', {}).keys()))
    if not all_months:
        return

    data = np.full((len(top), len(all_months)), np.nan)
    for i, r in enumerate(top):
        for j, m in enumerate(all_months):
            val = r.get('monthly', {}).get(m, np.nan)
            if isinstance(val, (int, float)):
                data[i, j] = val

    fig, ax = plt.subplots(figsize=(18, max(4, len(top) * 0.5)))
    cmap = plt.cm.RdYlGn
    norm = TwoSlopeNorm(vmin=0.42, vcenter=0.5, vmax=0.7)
    im = ax.imshow(data, aspect='auto', cmap=cmap, norm=norm)

    ax.set_xticks(range(len(all_months)))
    ax.set_xticklabels(all_months, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([r['model_name'] for r in top], fontsize=8)
    ax.set_title('Monthly AUC Heatmap — Top Models (φ_close > 1)', fontweight='bold')
    fig.colorbar(im, ax=ax, label='AUC', shrink=0.6)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                c = 'white' if (data[i,j] > 0.62 or data[i,j] < 0.46) else 'black'
                ax.text(j, i, f'{data[i,j]:.2f}', ha='center', va='center',
                        fontsize=5, color=c)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'monthly_heatmap.png'), dpi=200)
    plt.close(fig)


def plot_auc_vs_time(df_cls, outdir):
    """Scatter: AUC vs compute time (Pareto frontier)."""
    if df_cls.empty:
        return

    df = df_cls[df_cls['Target'] == 'cls_close'].copy()
    if df.empty:
        df = df_cls.copy()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(df['Time (s)'], df['AUC (pooled)'], s=80, c='steelblue',
               edgecolor='white', linewidth=0.5, zorder=3)

    for _, row in df.iterrows():
        ax.annotate(row['Model'], (row['Time (s)'], row['AUC (pooled)']),
                    fontsize=7, ha='left', va='bottom',
                    xytext=(4, 3), textcoords='offset points')

    ax.axhline(0.5, color='red', ls='--', lw=0.8, alpha=0.5)
    ax.set_xlabel('Total Wall Time (seconds)', fontsize=11)
    ax.set_ylabel('Pooled AUC', fontsize=11)
    ax.set_title('AUC vs Compute Time — Efficiency Frontier', fontweight='bold')
    ax.set_xscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'auc_vs_time.png'), dpi=150)
    plt.close(fig)


def plot_metric_radar(df_cls, outdir):
    """Radar chart for top-5 models showing multiple metrics."""
    if df_cls.empty or len(df_cls) < 3:
        return

    df = df_cls[df_cls['Target'] == 'cls_close'].copy().head(6)
    if len(df) < 3:
        return

    metrics = ['AUC (pooled)', 'Accuracy', 'Precision', 'F1']
    available = [m for m in metrics if m in df.columns]
    if len(available) < 3:
        return

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(available), endpoint=False).tolist()
    angles += angles[:1]

    for _, row in df.iterrows():
        values = [row[m] for m in available]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=1.5, markersize=4,
                label=row['Model'])
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available, fontsize=9)
    ax.set_ylim(0.4, 0.8)
    ax.set_title('Multi-Metric Comparison — Top Models', fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'metric_radar.png'), dpi=150)
    plt.close(fig)


def write_summary_md(df_cls, df_reg, results, outdir):
    """Write comprehensive Markdown summary."""
    md_path = os.path.join(outdir, 'MODEL_ZOO_REPORT.md')
    with open(md_path, 'w') as f:
        f.write("# 🧪 Model Zoo — Permanence Prediction Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"Total models evaluated: **{len(results)}**\n\n")

        # Best model
        if not df_cls.empty:
            best = df_cls.iloc[0]
            f.write(f"## 🏆 Best Classifier: **{best['Model']}**\n\n")
            f.write(f"- **AUC (pooled):** {best['AUC (pooled)']:.4f}\n")
            f.write(f"- **Accuracy:** {best['Accuracy']:.4f}\n")
            f.write(f"- **F1:** {best['F1']:.4f}\n")
            f.write(f"- **Brier Score:** {best['Brier']:.4f}\n")
            f.write(f"- **Monthly AUC:** {best['AUC mean']:.4f} ± {best['AUC std']:.4f} "
                    f"(range: {best['AUC min']:.4f}–{best['AUC max']:.4f})\n\n")

        # Classification leaderboard
        if not df_cls.empty:
            f.write("## Classification Leaderboard\n\n")
            cols = ['Model', 'Target', 'AUC (pooled)', 'Accuracy', 'F1',
                    'Brier', 'AUC mean', 'AUC std', 'Time (s)']
            cols = [c for c in cols if c in df_cls.columns]
            f.write(df_cls[cols].to_markdown(index=True))
            f.write("\n\n")

        # Regression leaderboard
        if not df_reg.empty:
            f.write("## Regression Leaderboard\n\n")
            cols = ['Model', 'Target', 'MAE', 'RMSE', 'R²', 'DirAcc', 'Time (s)']
            cols = [c for c in cols if c in df_reg.columns]
            f.write(df_reg[cols].to_markdown(index=True))
            f.write("\n\n")

        # Key findings
        f.write("## Key Findings\n\n")
        if not df_cls.empty:
            above_55 = df_cls[df_cls['AUC (pooled)'] > 0.55]
            above_60 = df_cls[df_cls['AUC (pooled)'] > 0.60]
            f.write(f"- **{len(above_55)}** models achieved AUC > 0.55\n")
            f.write(f"- **{len(above_60)}** models achieved AUC > 0.60\n")

            # Check if nonlinear models beat linear
            linear_keys = {'logreg_l2', 'logreg_l1', 'logreg_en', 'sgd_hinge', 'ridge_cls'}
            lin = df_cls[df_cls['Key'].isin(linear_keys)]
            nonlin = df_cls[~df_cls['Key'].isin(linear_keys)]
            if not lin.empty and not nonlin.empty:
                lin_best = lin['AUC (pooled)'].max()
                nonlin_best = nonlin['AUC (pooled)'].max()
                if nonlin_best > lin_best + 0.01:
                    f.write(f"- Nonlinear models ({nonlin_best:.4f}) substantially "
                            f"outperform linear ({lin_best:.4f}) → nonlinear signal exists\n")
                else:
                    f.write(f"- Linear ({lin_best:.4f}) ≈ nonlinear ({nonlin_best:.4f}) "
                            f"→ signal may be approximately linear\n")

        f.write("\n## Plots\n\n")
        f.write("- `leaderboard_auc.png` — AUC bar chart\n")
        f.write("- `monthly_heatmap.png` — Monthly AUC heatmap\n")
        f.write("- `horizon_comparison.png` — AUC by prediction horizon\n")
        f.write("- `auc_vs_time.png` — Efficiency frontier\n")
        f.write("- `metric_radar.png` — Multi-metric radar\n")

    print(f"  → {md_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Analyze Model Zoo results.")
    ap.add_argument('zoo_dir',
        help='Directory with per-model JSON results')
    args = ap.parse_args()

    print(f"Loading results from {args.zoo_dir} …")
    results = load_results(args.zoo_dir)
    print(f"  Found {len(results)} result files")

    if not results:
        print("No results found. Run train_model_zoo.py first.")
        return

    df_cls = build_leaderboard(results, task='binary')
    df_reg = build_leaderboard(results, task='regression')

    print(f"\n  Classification models: {len(df_cls)}")
    print(f"  Regression models:     {len(df_reg)}")

    if not df_cls.empty:
        print(f"\n  Top-5 Classifiers (AUC):")
        for _, row in df_cls.head(5).iterrows():
            print(f"    {row['Model']:<30s}  AUC={row['AUC (pooled)']:.4f}  "
                  f"({row['Target']})")

    if not df_reg.empty:
        print(f"\n  Top-5 Regressors (MAE):")
        for _, row in df_reg.head(5).iterrows():
            print(f"    {row['Model']:<30s}  MAE={row['MAE']:.4f}  "
                  f"R²={row['R²']:.4f}")

    # Generate plots
    print("\nGenerating plots …")
    plot_auc_leaderboard(df_cls, args.zoo_dir)
    print("  → leaderboard_auc.png")
    plot_monthly_heatmap(results, args.zoo_dir)
    print("  → monthly_heatmap.png")
    plot_horizon_comparison(results, args.zoo_dir)
    print("  → horizon_comparison.png")
    plot_auc_vs_time(df_cls, args.zoo_dir)
    print("  → auc_vs_time.png")
    plot_metric_radar(df_cls, args.zoo_dir)
    print("  → metric_radar.png")

    # Write report
    write_summary_md(df_cls, df_reg, results, args.zoo_dir)

    # Save CSVs
    if not df_cls.empty:
        csv_path = os.path.join(args.zoo_dir, 'leaderboard_cls.csv')
        df_cls.to_csv(csv_path)
        print(f"  → {csv_path}")
    if not df_reg.empty:
        csv_path = os.path.join(args.zoo_dir, 'leaderboard_reg.csv')
        df_reg.to_csv(csv_path)
        print(f"  → {csv_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
