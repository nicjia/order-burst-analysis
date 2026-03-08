#!/usr/bin/env python3
"""
eda_phase1.py — Phase I: Ex Post Measurement EDA

Exploratory plots to understand how permanence varies across bursts
as a function of observable characteristics (BurstVolume, Duration,
PeakImpact, Direction, Ticker, etc.).

Reads the *_filtered.csv (or _unfiltered.csv) produced by
compute_permanence.py and generates a set of publication-quality
figures saved to an output directory.

Usage:
    python src_py/eda_phase1.py <bursts_csv> [--outdir plots/]

Example:
    python src_py/eda_phase1.py bursts_tsla_filtered.csv --outdir plots/
"""

import pandas as pd
import numpy as np
import matplotlib
from torch import sub
matplotlib.use('Agg')                    # headless – works on Hoffman2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import argparse
import os


# ── Style ────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
    'figure.dpi': 150,
})

PERM_COL = 'Perm_tCLOSE'   # primary permanence horizon for EDA

# ── Helpers ──────────────────────────────────────────────────

def iqr_bounds(vals, k=1.5):
    """Return (lo, hi) bounds based on IQR fence: [Q1 − k·IQR, Q3 + k·IQR]."""
    q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def winsorize(s, lo, hi):
    """Clip a Series to [lo, hi] and return the clipped version."""
    return s.clip(lower=lo, upper=hi)


def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} bursts from {path}")
    print(f"Columns: {list(df.columns)}")

    # ── Derived columns if missing ───────────────────────────
    if 'Duration' not in df.columns and 'StartTime' in df.columns and 'EndTime' in df.columns:
        df['Duration'] = df['EndTime'] - df['StartTime']
    if 'PeakImpact' not in df.columns and 'PeakPrice' in df.columns and 'StartPrice' in df.columns:
        df['PeakImpact'] = (df['PeakPrice'] - df['StartPrice']).abs()

    # Rename Volume → BurstVolume for clarity in plots
    if 'Volume' in df.columns and 'BurstVolume' not in df.columns:
        df['BurstVolume'] = df['Volume']

    return df


# ── 1.  Permanence Histograms ───────────────────────────────

def plot_permanence_histograms(df, outdir):
    """Histograms of every Perm_* column, using IQR fences to focus on the core distribution."""
    perm_cols = sorted([c for c in df.columns if c.startswith('Perm_')])
    n = len(perm_cols)
    if n == 0:
        print("  SKIP: no Perm_* columns")
        return

    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    axes = np.atleast_2d(axes)

    for i, col in enumerate(perm_cols):
        ax = axes[i // ncols, i % ncols]
        vals = df[col].dropna()
        total = len(vals)

        # Use IQR fences (k=2.5) — much better for heavy tails than percentile clip
        lo, hi = iqr_bounds(vals, k=2.5)
        vals_clip = vals[(vals >= lo) & (vals <= hi)]
        pct_shown = 100 * len(vals_clip) / total

        ax.hist(vals_clip, bins=80, color='steelblue', edgecolor='none', alpha=0.8)
        ax.axvline(0, color='black', lw=0.8, ls='--')
        ax.axvline(vals.median(), color='red', lw=1.5, ls='-',
                   label=f'median = {vals.median():.3f}')
        ax.axvline(vals.mean(), color='orange', lw=1.2, ls=':',
                   label=f'mean = {vals.mean():.3f}')

        ax.set_title(col, fontweight='bold')
        ax.set_xlabel('Permanence φ')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8, loc='upper right')
        ax.annotate(f'{pct_shown:.1f}% of data shown\n(IQR fence, k=2.5)',
                    xy=(0.02, 0.95), xycoords='axes fraction', fontsize=7,
                    va='top', color='grey')

    # Turn off empty subplots
    for i in range(n, nrows * ncols):
        axes[i // ncols, i % ncols].set_visible(False)

    fig.suptitle('Permanence Distributions (IQR-fenced)', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(outdir, 'permanence_histograms.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  → {path}")


# ── 2.  Scatter: Perm_tCLOSE vs Observable Features ─────────
def plot_scatter_vs_features(df, outdir):
    """Hexbin density plots of Perm_tCLOSE vs BurstVolume, Duration, PeakImpact.

    Permanence is winsorized to IQR fences before plotting so the
    decile-mean trendline is actually visible.
    """
    features = ['BurstVolume', 'Duration', 'PeakImpact']
    available = [f for f in features if f in df.columns]
    if PERM_COL not in df.columns or not available:
        print(f"  SKIP scatter: need {PERM_COL} and at least one of {features}")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(6.5 * len(available), 5.5))
    if len(available) == 1:
        axes = [axes]

    # Winsorize permanence once — IQR fence k=2.5
    y_lo, y_hi = iqr_bounds(df[PERM_COL].dropna(), k=2.5)

    for ax, feat in zip(axes, available):
        sub = df[[feat, PERM_COL]].dropna().copy()
        sub['y'] = winsorize(sub[PERM_COL], y_lo, y_hi)

        # Clip x-axis to 99th percentile (0 is a natural lower bound)
        x_hi = sub[feat].quantile(0.99)
        sub = sub[sub[feat] <= x_hi]

        # Use log x-axis for BurstVolume (highly right-skewed)
        use_log_x = (feat == 'BurstVolume') and (sub[feat].min() > 0)

        hb = ax.hexbin(
            sub[feat], sub['y'],
            gridsize=50, cmap='Blues', mincnt=1,
            norm=LogNorm(),
            xscale='log' if use_log_x else 'linear',
        )
        ax.axhline(0, color='black', lw=0.8, ls='--')
        ax.axhline(1, color='grey', lw=0.6, ls=':', alpha=0.6)

        # Decile-mean overlay
        try:
            sub['bin'] = pd.qcut(sub[feat], 10, duplicates='drop')
            means   = sub.groupby('bin', observed=False)['y'].mean()
            centers = sub.groupby('bin', observed=False)[feat].mean()
            ax.plot(centers, means, 'o-', color='red', lw=2.5, ms=7,
                    markeredgecolor='white', markeredgewidth=0.8, zorder=5,
                    label='Decile mean')
            ax.legend(fontsize=9, loc='upper right')
        except Exception:
            pass

        ax.set_xlabel(feat + ('  (log scale)' if use_log_x else ''))
        ax.set_ylabel(f'{PERM_COL}  (winsorized)')
        ax.set_title(f'{PERM_COL}  vs  {feat}', fontweight='bold')
        fig.colorbar(hb, ax=ax, label='count (log)', shrink=0.85)

    fig.suptitle('Permanence vs Burst Characteristics', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(outdir, 'perm_vs_features_scatter.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  → {path}")

# ── 3.  Box-plots by Direction ───────────────────────────────

def plot_boxplot_by_direction(df, outdir):
    """Box-plots of permanence split by burst direction (+1 / −1).

    Uses IQR-fenced data and whis=[5, 95] so the whiskers span
    the 5th–95th percentile of the *fenced* data — keeps the boxes
    tight and readable even with heavy tails.
    """
    # Chronological order
    target_order = [
        'Perm_t1m', 'Perm_t3m', 'Perm_t5m', 'Perm_t10m',
        'Perm_tCLOSE', 'Perm_CLOP', 'Perm_CLCL'
    ]
    perm_cols = [c for c in target_order if c in df.columns]

    if not perm_cols or 'Direction' not in df.columns:
        print("  SKIP boxplot by direction")
        return

    fig, axes = plt.subplots(1, len(perm_cols), figsize=(3.5 * len(perm_cols), 5))
    if len(perm_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, perm_cols):
        groups = []
        labels = []
        for d_val, d_lbl in [(1, 'Buy (+1)'), (-1, 'Sell (−1)')]:
            vals = df.loc[df['Direction'] == d_val, col].dropna()
            # IQR-fence to remove extreme tail before boxing
            lo, hi = iqr_bounds(vals, k=2.5)
            groups.append(vals[(vals >= lo) & (vals <= hi)])
            labels.append(d_lbl)

        bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True, widths=0.5,
                        medianprops=dict(color='red', lw=2),
                        showfliers=False, whis=[5, 95])
        colors = ['#5599dd', '#dd7755']
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)

        ax.axhline(0, color='black', lw=0.8, ls='--')
        ax.axhline(1, color='grey', lw=0.6, ls=':', alpha=0.5)
        ax.set_title(col, fontweight='bold', fontsize=10)
        ax.set_ylabel('Permanence φ')

    fig.suptitle('Permanence by Burst Direction (Chronological)', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(outdir, 'perm_by_direction_boxplot.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  → {path}")

# ── 4.  D_b distribution & filter visualisation ─────────────

def plot_decay_filter(df, outdir):
    """Visualise D_b distribution and D_b vs PeakImpact with κ boundary lines."""
    if 'D_b' not in df.columns or 'PeakImpact' not in df.columns:
        print("  SKIP D_b plot: columns missing")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) Histogram of D_b — IQR fence
    ax = axes[0]
    vals = df['D_b'].dropna()
    lo, hi = iqr_bounds(vals, k=2.5)
    vals_clip = vals[(vals >= lo) & (vals <= hi)]
    pct = 100 * len(vals_clip) / len(vals)

    ax.hist(vals_clip, bins=80, color='steelblue', edgecolor='none', alpha=0.8)
    ax.axvline(0, color='black', lw=0.8, ls='--')
    ax.axvline(vals.median(), color='red', lw=1.5,
               label=f'median = {vals.median():.4f}')

    
    ax.set_xlabel('D_b  (avg short-horizon displacement)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of D_b', fontweight='bold')
    ax.legend()
    ax.annotate(f'{pct:.1f}% of data shown', xy=(0.97, 0.95),
                xycoords='axes fraction', ha='right', va='top',
                fontsize=8, color='grey')

    # (b) D_b vs PeakImpact hexbin with κ lines
    ax = axes[1]
    sub = df[['D_b', 'PeakImpact']].dropna()

    x_hi = sub['PeakImpact'].quantile(0.95)
    y_lo = sub['D_b'].quantile(0.01)
    y_hi = sub['D_b'].quantile(0.95)
    sub_clip = sub[(sub['PeakImpact'] <= x_hi) &
                   (sub['D_b'] >= y_lo) & (sub['D_b'] <= y_hi)]

    ax.hexbin(sub_clip['PeakImpact'], sub_clip['D_b'],
              gridsize=50, cmap='Blues', mincnt=1, norm=LogNorm())
    ax.axhline(0, color='black', lw=0.8, ls='--')

    x_range = np.linspace(0, x_hi, 100)
    for kval, c, ls in [(0.10, 'orange', '--'), (0.25, 'red', '-'), (0.50, 'darkred', ':')]:
        ax.plot(x_range, kval * x_range, color=c, ls=ls, lw=2, label=f'κ = {kval}')

    ax.set_xlim(0, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel('PeakImpact')
    ax.set_ylabel('D_b')
    ax.set_title('D_b  vs  PeakImpact  (decay filter)', fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')

    fig.tight_layout()
    path = os.path.join(outdir, 'decay_filter_Db.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  → {path}")


# ── 5.  Heatmap: median permanence by Volume-decile × Duration-decile

def plot_heatmap_vol_dur(df, outdir):
    """Median Perm_tCLOSE heatmap binned by BurstVolume and Duration quintiles.

    Uses *median* (robust to extreme φ values) and annotates each cell.
    """
    needed = [PERM_COL, 'BurstVolume', 'Duration']
    if not all(c in df.columns for c in needed):
        print(f"  SKIP heatmap: need {needed}")
        return

    sub = df[needed].dropna().copy()

    # Winsorize φ before aggregating so a handful of ±1000 values
    # don't dominate the colour scale
    y_lo, y_hi = iqr_bounds(sub[PERM_COL], k=3)
    sub[PERM_COL] = winsorize(sub[PERM_COL], y_lo, y_hi)

    try:
        sub['vol_q'] = pd.qcut(sub['BurstVolume'], 5, duplicates='drop')
        sub['dur_q'] = pd.qcut(sub['Duration'],    5, duplicates='drop')
    except ValueError:
        print("  SKIP heatmap: not enough unique values for qcut")
        return

    pivot = sub.groupby(['dur_q', 'vol_q'], observed=False)[PERM_COL].median().unstack()
    counts = sub.groupby(['dur_q', 'vol_q'], observed=False)[PERM_COL].count().unstack()

    # Symmetric colour range centered on 0
    vabs = max(abs(pivot.values[np.isfinite(pivot.values)].min()),
               abs(pivot.values[np.isfinite(pivot.values)].max()),
               0.01)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(pivot.values, cmap='RdBu_r', aspect='auto',
                   vmin=-vabs, vmax=vabs)

    # Annotate cells with median + count
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            val = pivot.values[r, c]
            cnt = counts.values[r, c]
            if np.isfinite(val):
                txt_col = 'white' if abs(val) > 0.6 * vabs else 'black'
                ax.text(c, r, f'{val:.2f}\n(n={int(cnt)})',
                        ha='center', va='center', fontsize=8, color=txt_col)

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([str(r) for r in pivot.index], fontsize=8)
    ax.set_xlabel('BurstVolume quintile')
    ax.set_ylabel('Duration quintile')
    ax.set_title(f'Median {PERM_COL}  by Volume × Duration  (winsorized)',
                 fontweight='bold')
    fig.colorbar(im, ax=ax, label=f'Median {PERM_COL}', shrink=0.85)
    fig.tight_layout()
    path = os.path.join(outdir, 'heatmap_vol_dur.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  → {path}")


# ── 6.  Bursts per week ──────────────────────────────────────

def plot_bursts_per_week(df, outdir):
    """Bar chart of burst counts aggregated by ISO week.

    Each bar = one calendar week.  The bar height is the *total* number
    of bursts that week, and the annotation inside the bar shows the
    average bursts/day (dividing by the number of trading days that
    week, which handles short weeks correctly).
    """
    if 'Date' not in df.columns:
        print("  SKIP bursts-per-week: no Date column")
        return

    daily = df.groupby('Date').size().reset_index(name='n_bursts')
    daily['Date'] = pd.to_datetime(daily['Date'])

    # ISO-week label  "2023-W01"
    daily['week'] = daily['Date'].dt.isocalendar().year.astype(str) + '-W' + \
                    daily['Date'].dt.isocalendar().week.astype(str).str.zfill(2)
    weekly = daily.groupby('week').agg(
        total  = ('n_bursts', 'sum'),
        avg_day= ('n_bursts', 'mean'),
        n_days = ('n_bursts', 'count'),
    ).reset_index()
    # Sort chronologically by the Monday of each ISO week
    weekly['monday'] = pd.to_datetime(
        weekly['week'].str[:4] + weekly['week'].str[5:], format='%G%V%u',
        errors='coerce'
    )
    # fallback: parse via Monday
    if weekly['monday'].isna().all():
        weekly['monday'] = weekly['week'].apply(
            lambda w: pd.Timestamp.fromisocalendar(int(w[:4]), int(w[6:]), 1))
    weekly = weekly.sort_values('monday').reset_index(drop=True)

    # ── Plot ──
    n = len(weekly)
    fig_w = max(12, n * 0.35)          # scale width to # of weeks
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))

    colors = plt.cm.Blues(np.linspace(0.3, 0.85, n))
    bars = ax.bar(range(n), weekly['total'], color=colors, edgecolor='none',
                  width=0.8)

    # Annotate every bar with avg bursts/day (inside the bar)
    for i, (b, avg) in enumerate(zip(bars, weekly['avg_day'])):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 0.5,
                f'{avg:,.0f}', ha='center', va='center', fontsize=6,
                color='white', fontweight='bold', rotation=90)

    ax.set_xticks(range(n))
    ax.set_xticklabels(weekly['week'], rotation=90, fontsize=7)
    ax.set_ylabel('Total bursts')
    ax.set_xlabel('Week (ISO)')
    ax.set_title('Bursts per Week  (bar height = weekly total, label = avg / trading day)',
                 fontweight='bold')

    # Horizontal mean line
    overall_mean = weekly['total'].mean()
    ax.axhline(overall_mean, color='red', ls='--', lw=1,
               label=f'mean = {overall_mean:,.0f} / wk')
    ax.legend(fontsize=9)

    fig.tight_layout()
    path = os.path.join(outdir, 'bursts_per_week.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  → {path}")


# ── 7.  Summary stats table ─────────────────────────────────

def print_summary_table(df, outdir):
    """Write a Markdown summary table of key statistics."""
    perm_cols = sorted([c for c in df.columns if c.startswith('Perm_')])
    extra = ['D_b', 'PeakImpact', 'BurstVolume', 'Duration']
    cols = [c for c in perm_cols + extra if c in df.columns]

    stats = df[cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    stats['non-null'] = df[cols].notna().sum()

    md_path = os.path.join(outdir, 'summary_stats.md')
    with open(md_path, 'w') as f:
        f.write("# Phase I – Summary Statistics\n\n")
        f.write(f"Total bursts: **{len(df)}**\n\n")
        if 'Ticker' in df.columns:
            f.write(f"Tickers: {', '.join(sorted(df['Ticker'].unique()))}\n\n")
        if 'Date' in df.columns:
            f.write(f"Date range: {df['Date'].min()} → {df['Date'].max()}\n\n")
        f.write(stats.to_markdown())
        f.write("\n")
    print(f"  → {md_path}")


# ── Main ─────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Phase I EDA plots for burst permanence.")
    ap.add_argument('bursts_csv', help='Burst CSV with Perm_* and D_b columns')
    ap.add_argument('--outdir', default='plots', help='Directory for output figures (default: plots/)')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_data(args.bursts_csv)

    print("\nGenerating plots …")
    plot_permanence_histograms(df, args.outdir)
    plot_scatter_vs_features(df, args.outdir)
    plot_boxplot_by_direction(df, args.outdir)
    plot_decay_filter(df, args.outdir)
    plot_heatmap_vol_dur(df, args.outdir)
    plot_bursts_per_week(df, args.outdir)
    print_summary_table(df, args.outdir)
    print(f"\nDone — all figures saved to {args.outdir}/")


if __name__ == '__main__':
    main()
