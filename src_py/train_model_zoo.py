#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_model_zoo.py — Comprehensive Model Zoo for Permanence Prediction

Runs ~20 ML algorithms across multiple targets and feature sets with
walk-forward cross-validation. Designed for Hoffman2 (SLURM job arrays).

MODELS (classification):
  ┌────────────────────────────────────────────────────────────────────┐
  │ Gradient Boosted Trees                                            │
  │   1.  LightGBM (default)                                         │
  │   2.  LightGBM (Optuna-tuned)                                    │
  │   3.  XGBoost                                                     │
  │   4.  XGBoost (Optuna-tuned)                                      │
  │   5.  CatBoost                                                    │
  │   6.  sklearn GradientBoosting (HistGradientBoosting)             │
  │ Tree Ensembles                                                    │
  │   7.  Random Forest                                               │
  │   8.  Extra Trees                                                 │
  │   9.  AdaBoost (decision stumps)                                  │
  │ Linear                                                            │
  │  10.  Logistic Regression (L2)                                    │
  │  11.  Logistic Regression (L1)                                    │
  │  12.  Logistic Regression (ElasticNet)                            │
  │  13.  Linear SVM (SGD w/ hinge)                                   │
  │  14.  Ridge Classifier                                            │
  │ Non-parametric                                                    │
  │  15.  KNN (k=50, subsample if >200k)                              │
  │  16.  SVM-RBF (subsample to 50k)                                  │
  │ Neural                                                            │
  │  17.  MLP (2 hidden layers)                                       │
  │  18.  MLP (3 hidden layers, wider)                                │
  │ Ensemble                                                          │
  │  19.  Stacking (LGB + XGB + RF → LogReg meta)                    │
  │  20.  Voting (LGB + XGB + RF, soft)                               │
  │ Probabilistic                                                     │
  │  21.  Naive Bayes (Gaussian)                                      │
  │  22.  Calibrated LightGBM (isotonic)                              │
  └────────────────────────────────────────────────────────────────────┘

TARGETS:
    - Binary:  φ_tCLOSE > 0  (default)
    - Binary:  φ_t1m > 0
    - Binary:  φ_t3m > 0
    - Binary:  φ_t5m > 0
    - Binary:  φ_t10m > 0
  - 3-class: reversed / neutral / persisted
  - Regression: φ_tCLOSE (winsorized, Huber)

FEATURE SETS:
  - base:      Raw features from C++ engine
  - extended:  + interactions, log-transforms, ratios
  - pca:       PCA(n=10) of extended features
  - poly:      + degree-2 polynomial on top-5 features

Usage:
  # Run one model:
  python src_py/train_model_zoo.py results/bursts_NVDA_filtered.csv \\
      --model lgb --target cls_close --outdir results/zoo/

  # Run all models:
  python src_py/train_model_zoo.py results/bursts_NVDA_filtered.csv \\
      --model all --outdir results/zoo/

  # Run specific model with Optuna tuning:
  python src_py/train_model_zoo.py results/bursts_NVDA_filtered.csv \\
      --model lgb_tuned --target cls_close --outdir results/zoo/

  # Hoffman2 (SLURM array — each model is a separate job):
  sbatch hoffman2_model_zoo.sh results/bursts_NVDA_filtered.csv
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
import warnings
import json
import time
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import (
    LogisticRegression, SGDClassifier, RidgeClassifier,
    Ridge, Lasso, ElasticNet, SGDRegressor,
)
# Enable experimental flag for HistGradientBoosting (needed on sklearn <1.0)
try:
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa
except ImportError:
    pass  # sklearn >= 1.0 doesn't need this

from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,
    HistGradientBoostingClassifier, VotingClassifier, StackingClassifier,
    RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor,
    HistGradientBoostingRegressor, VotingRegressor, StackingRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    StandardScaler, QuantileTransformer, PolynomialFeatures,
)
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, brier_score_loss, log_loss, mean_squared_error,
    mean_absolute_error, r2_score, cohen_kappa_score,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# ─── optional imports ───
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# CatBoost disabled — binary incompatible with Hoffman2's 2020 NumPy
HAS_CB = False

SEED = 42
np.random.seed(SEED)

# Runtime-focused defaults for very large datasets.
OPTUNA_TRIALS = 12


# ═════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ═════════════════════════════════════════════════════════════════

# Maps model key → (human name, requires_lib, task_type)
MODEL_REGISTRY_CLS = {
    'lgb':           ('LightGBM',                  'lgb',    'cls'),
    'lgb_tuned':     ('LightGBM (Optuna)',          'lgb',    'cls'),
    'xgb':           ('XGBoost',                    'xgb',    'cls'),
    'xgb_tuned':     ('XGBoost (Optuna)',            'xgb',    'cls'),

    'histgbt':       ('HistGradientBoosting',        None,    'cls'),
    'rf':            ('Random Forest',               None,    'cls'),
    'et':            ('Extra Trees',                 None,    'cls'),
    'adaboost':      ('AdaBoost',                    None,    'cls'),
    'logreg_l2':     ('LogReg (L2)',                 None,    'cls'),
    'logreg_l1':     ('LogReg (L1)',                 None,    'cls'),
    'logreg_en':     ('LogReg (ElasticNet)',         None,    'cls'),
    'sgd_hinge':     ('Linear SVM (SGD)',            None,    'cls'),
    'ridge_cls':     ('Ridge Classifier',            None,    'cls'),
    'knn':           ('KNN (k=50)',                  None,    'cls'),
    'svm_rbf':       ('SVM-RBF (subsample)',         None,    'cls'),
    'mlp_small':     ('MLP (256-128)',               None,    'cls'),
    'mlp_large':     ('MLP (512-256-128)',           None,    'cls'),
    'stacking':      ('Stacking (LGB+XGB+RF→LR)',   'lgb',   'cls'),
    'voting':        ('Voting (LGB+XGB+RF)',         'lgb',   'cls'),
    'naive_bayes':   ('Gaussian NB',                 None,    'cls'),
    'lgb_calibrated':('LightGBM (Calibrated)',       'lgb',   'cls'),
}

MODEL_REGISTRY_REG = {
    'lgb_reg':       ('LightGBM Reg',               'lgb',   'reg'),
    'lgb_reg_tuned': ('LightGBM Reg (Optuna)',       'lgb',   'reg'),
    'xgb_reg':       ('XGBoost Reg',                 'xgb',   'reg'),

    'histgbt_reg':   ('HistGradientBoosting Reg',    None,    'reg'),
    'rf_reg':        ('Random Forest Reg',            None,    'reg'),
    'et_reg':        ('Extra Trees Reg',              None,    'reg'),
    'ridge_reg':     ('Ridge Reg',                    None,    'reg'),
    'lasso_reg':     ('Lasso Reg',                    None,    'reg'),
    'elasticnet_reg':('ElasticNet Reg',               None,    'reg'),
    'mlp_reg':       ('MLP Reg (256-128)',             None,    'reg'),
    'svr_linear':    ('Linear SVR (subsample)',        None,    'reg'),
    'knn_reg':       ('KNN Reg (k=50)',                None,    'reg'),
}

ALL_MODELS = {**MODEL_REGISTRY_CLS, **MODEL_REGISTRY_REG}


# ═════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════

BASE_FEATURE_COLS = [
    'Direction', 'BurstVolume', 'TradeCount', 'Duration',
    'PeakImpact', 'D_b', 'AvgTradeSize', 'PriceChange',
]

# Features derived from D_b that inherit its look-ahead bias.
# D_b = (1/4) Σ_τ Direction × (Mid_τ − StartPrice) for τ ∈ {1m, 3m, 5m, 10m},
# so it leaks 3m/5m/10m future prices when predicting the 1m horizon, etc.
# These must be DROPPED when the target horizon is shorter than 10m.
DB_TAINTED_FEATURES = {
    'D_b', 'Dir_x_Db', 'Impact_x_Db', 'AvgSize_x_Db', 'DbSquared', 'Db_qrank',
}

# Targets whose horizon is ≤ 10m.  D_b averages displacements at 1/3/5/10m,
# so it leaks future prices for ALL intraday horizons (including 10m itself,
# since Mid_10m is both a D_b input and the basis of Perm_t10m).
DB_LEAKY_TARGETS = {'cls_1m', 'cls_3m', 'cls_5m', 'cls_10m',
                    'reg_1m', 'reg_3m', 'reg_5m', 'reg_10m'}

EXTENDED_FEATURE_COLS = BASE_FEATURE_COLS + [
    'TimeOfDay', 'LogVolume', 'LogPeakImpact', 'ImpactPerShare',
    'RecentBurstCount', 'RecentBurstVol',
    # ── Interactions ──
    'Dir_x_Volume', 'Dir_x_Impact', 'Dir_x_Db',
    'Volume_x_Impact', 'Volume_x_Duration',
    'Impact_x_Db', 'Impact_x_TradeCount',
    'AvgSize_x_Impact', 'AvgSize_x_Db',
    # ── Ratios ──
    'ImpactPerTrade', 'VolumePerSec',
    'DbSquared', 'ImpactSquared',
    # ── Quantile-rank features ──
    'Volume_qrank', 'Impact_qrank', 'Db_qrank',
    # ── Clustered features ──
    'RecentBurstCountOpp',  # opposite direction
    'RecentBurstVolOpp',
    'NetRecentFlow',        # same_vol - opp_vol
    'BurstDensity5m',       # total bursts in 5min window
    # ── Time features ──
    'TimeOfDaySin', 'TimeOfDayCos',  # cyclical encoding
    'IsOpen15', 'IsClose15',          # first/last 15 min flags
    'HourOfDay',
    # ── Rolling price context ──
    'PriceLevel',    # log(StartPrice)
    'VolPerDollar',  # BurstVolume * StartPrice (dollar volume)
    # ── Multi-horizon permanence at shorter horizon (if avail) ──
    # (NO — these are targets / forward-looking. Excluded.)
]


def engineer_features(df):
    """Build Z_b feature vector. STRICTLY no look-ahead."""
    df = df.copy()

    # ── Basic ────────────────────────────────────────────────
    if 'Volume' in df.columns and 'BurstVolume' not in df.columns:
        df['BurstVolume'] = df['Volume']
    if 'BurstVolume' not in df.columns and 'Volume' in df.columns:
        df['BurstVolume'] = df['Volume']
    elif 'BurstVolume' not in df.columns:
        df['BurstVolume'] = 0

    if 'Duration' not in df.columns:
        df['Duration'] = df['EndTime'] - df['StartTime']

    df['AvgTradeSize'] = df['BurstVolume'] / df['TradeCount'].clip(lower=1)
    df['PriceChange']  = df['Direction'] * (df['EndPrice'] - df['StartPrice'])

    # ── Time-of-day ──────────────────────────────────────────
    RTH_START, RTH_END = 34200.0, 57600.0
    df['TimeOfDay'] = (df['StartTime'] - RTH_START) / (RTH_END - RTH_START)
    df['TimeOfDay'] = df['TimeOfDay'].clip(0, 1)

    df['TimeOfDaySin'] = np.sin(2 * np.pi * df['TimeOfDay'])
    df['TimeOfDayCos'] = np.cos(2 * np.pi * df['TimeOfDay'])
    df['IsOpen15']  = (df['StartTime'] < RTH_START + 900).astype(int)
    df['IsClose15'] = (df['StartTime'] > RTH_END - 900).astype(int)
    df['HourOfDay'] = ((df['StartTime'] - RTH_START) / 3600).clip(0, 6.5).astype(int)

    # ── Log transforms ───────────────────────────────────────
    if 'PeakImpact' not in df.columns:
        if 'PeakPrice' in df.columns and 'StartPrice' in df.columns:
            df['PeakImpact'] = (df['PeakPrice'] - df['StartPrice']).abs()
        else:
            df['PeakImpact'] = 0.0

    df['LogVolume']     = np.log1p(df['BurstVolume'])
    df['LogPeakImpact'] = np.log1p(df['PeakImpact'].abs() * 10000)

    # ── Ratios ───────────────────────────────────────────────
    df['ImpactPerShare'] = df['PeakImpact'] / df['BurstVolume'].clip(lower=1)
    df['ImpactPerTrade'] = df['PeakImpact'] / df['TradeCount'].clip(lower=1)
    df['VolumePerSec']   = df['BurstVolume'] / df['Duration'].clip(lower=0.001)

    # ── Interactions ─────────────────────────────────────────
    df['Dir_x_Volume']   = df['Direction'] * df['LogVolume']
    df['Dir_x_Impact']   = df['Direction'] * df['LogPeakImpact']
    df['Dir_x_Db']       = df['Direction'] * df['D_b']
    df['Volume_x_Impact']   = df['LogVolume'] * df['LogPeakImpact']
    df['Volume_x_Duration'] = df['LogVolume'] * np.log1p(df['Duration'])
    df['Impact_x_Db']       = df['LogPeakImpact'] * df['D_b']
    df['Impact_x_TradeCount'] = df['LogPeakImpact'] * np.log1p(df['TradeCount'])
    df['AvgSize_x_Impact'] = np.log1p(df['AvgTradeSize']) * df['LogPeakImpact']
    df['AvgSize_x_Db']     = np.log1p(df['AvgTradeSize']) * df['D_b']

    # ── Squared terms ────────────────────────────────────────
    df['DbSquared']      = df['D_b'] ** 2
    df['ImpactSquared']  = df['LogPeakImpact'] ** 2

    # ── Price-level context ──────────────────────────────────
    df['PriceLevel'] = np.log(df['StartPrice'].clip(lower=0.01))
    df['VolPerDollar'] = df['BurstVolume'] * df['StartPrice']

    # ── Quantile-rank features (per-date, backward-looking) ──
    # We rank each burst relative to all PRIOR bursts on that date.
    # This is safe: only uses past info within the day.
    df['Volume_qrank'] = 0.5
    df['Impact_qrank'] = 0.5
    df['Db_qrank']     = 0.5

    import bisect
    for _, day_df in df.groupby('Date'):
        idx = day_df.index
        n = len(idx)
        if n < 2:
            continue

        # O(N log N) online rank: compare each point against sorted past values.
        for col, qcol in [('BurstVolume', 'Volume_qrank'),
                          ('PeakImpact', 'Impact_qrank'),
                          ('D_b', 'Db_qrank')]:
            vals = day_df[col].values
            ranks = np.zeros(n)
            past_vals = []
            for i in range(n):
                v = vals[i]
                pos = bisect.bisect_left(past_vals, v)
                ranks[i] = (pos / i) if i > 0 else 0.5
                bisect.insort(past_vals, v)
            df.loc[idx, qcol] = ranks

    # ── Cross-burst features (rolling 5-min window) ──────────
    df['RecentBurstCount'] = 0
    df['RecentBurstVol']   = 0
    df['RecentBurstCountOpp'] = 0
    df['RecentBurstVolOpp']   = 0
    df['BurstDensity5m']      = 0

    LOOKBACK = 300.0
    for _, day_df in df.groupby('Date'):
        idx = day_df.index
        starts = day_df['StartTime'].values
        dirs   = day_df['Direction'].values
        vols   = day_df['BurstVolume'].values

        cnt_same = np.zeros(len(idx), dtype=int)
        vol_same = np.zeros(len(idx), dtype=float)
        cnt_opp  = np.zeros(len(idx), dtype=int)
        vol_opp  = np.zeros(len(idx), dtype=float)
        density  = np.zeros(len(idx), dtype=int)

        for i in range(len(idx)):
            t = starts[i]
            d = dirs[i]
            j = i - 1
            while j >= 0 and starts[j] >= t - LOOKBACK:
                density[i] += 1
                if dirs[j] == d:
                    cnt_same[i] += 1
                    vol_same[i] += vols[j]
                else:
                    cnt_opp[i] += 1
                    vol_opp[i] += vols[j]
                j -= 1

        df.loc[idx, 'RecentBurstCount']    = cnt_same
        df.loc[idx, 'RecentBurstVol']      = vol_same
        df.loc[idx, 'RecentBurstCountOpp'] = cnt_opp
        df.loc[idx, 'RecentBurstVolOpp']   = vol_opp
        df.loc[idx, 'BurstDensity5m']      = density

    df['NetRecentFlow'] = df['RecentBurstVol'] - df['RecentBurstVolOpp']

    return df


# ═════════════════════════════════════════════════════════════════
# TARGET CONSTRUCTION
# ═════════════════════════════════════════════════════════════════

TARGET_MAP = {
    # ── Intraday horizons (use UNFILTERED data only) ─────────
    'cls_1m':     ('Perm_t1m',     'binary',  0.0),   # φ_t1m > 0
    'cls_3m':     ('Perm_t3m',     'binary',  0.0),   # φ_t3m > 0
    'cls_5m':     ('Perm_t5m',     'binary',  0.0),   # φ_t5m > 0
    'cls_10m':    ('Perm_t10m',    'binary',  0.0),   # φ_t10m > 0
    # ── End-of-day / next-day (valid on filtered OR unfiltered) ─
    'cls_close':  ('Perm_tCLOSE', 'binary',  0.0),   # φ_tCLOSE > 0  (same-day 4pm)
    'cls_clop':   ('Perm_CLOP',   'binary',  0.0),   # φ_CLOP > 0    (next-day open)
    'cls_clcl':   ('Perm_CLCL',   'binary',  0.0),   # φ_CLCL > 0    (next-day close)
    # ── Regression ───────────────────────────────────────────
    'reg_close':  ('Perm_tCLOSE', 'regression', None),
    'reg_clop':   ('Perm_CLOP',   'regression', None),
    'reg_clcl':   ('Perm_CLCL',   'regression', None),
    'reg_1m':     ('Perm_t1m',     'regression', None),
    'reg_5m':     ('Perm_t5m',     'regression', None),
}


def build_target(df, target_key):
    """Build target vector. Returns (y, task_type, meta)."""
    col, task, threshold = TARGET_MAP[target_key]

    if col not in df.columns:
        raise ValueError(f"Target column '{col}' not in data.")

    vals = df[col].values.copy()

    if task == 'binary':
        y = (vals > threshold).astype(int)
        return y, 'binary', {'pos_rate': y.mean(), 'threshold': threshold}

    elif task == '3class':
        # reversed: φ < -0.5, neutral: -0.5 ≤ φ ≤ 0.5, persisted: φ > 0.5
        y = np.where(vals < -0.5, 0,
            np.where(vals > 0.5, 2, 1))
        return y, 'multiclass', {
            'class_dist': {0: (y==0).mean(), 1: (y==1).mean(), 2: (y==2).mean()}}

    elif task == 'regression':
        lo = np.nanpercentile(vals, 1)
        hi = np.nanpercentile(vals, 99)
        y = np.clip(vals, lo, hi)
        return y, 'regression', {'lo': lo, 'hi': hi}


# ═════════════════════════════════════════════════════════════════
# WALK-FORWARD CV
# ═════════════════════════════════════════════════════════════════

def walk_forward_splits(df, min_train_months=3):
    """Expanding-window walk-forward splits by calendar month."""
    df_ym = pd.to_datetime(df['Date'].astype(str)).dt.to_period('M')
    months = sorted(df_ym.unique())

    splits = []
    for i in range(min_train_months, len(months)):
        train_mask = df_ym.isin(months[:i])
        test_mask  = df_ym == months[i]
        if test_mask.sum() == 0:
            continue
        splits.append((
            df.index[train_mask].values,
            df.index[test_mask].values,
            str(months[i]),
        ))
    return splits


# ═════════════════════════════════════════════════════════════════
# MODEL BUILDERS
# ═════════════════════════════════════════════════════════════════

def _needs_scaling(model_key):
    """Models that need standardized input."""
    return model_key in {
        'logreg_l2', 'logreg_l1', 'logreg_en', 'sgd_hinge',
        'ridge_cls', 'knn', 'svm_rbf', 'mlp_small', 'mlp_large',
        'ridge_reg', 'lasso_reg', 'elasticnet_reg', 'mlp_reg',
        'svr_linear', 'knn_reg', 'naive_bayes',
    }


def _needs_subsample(model_key):
    """Models that are too slow for full data — subsample training set."""
    LIMITS = {
        'svm_rbf': 30_000,
        'knn': 80_000,
        'knn_reg': 200_000,
        'svr_linear': 100_000,
        'mlp_small': 200_000,
        'mlp_large': 120_000,
        'rf': 250_000,
        'et': 250_000,
        'adaboost': 250_000,
        'histgbt': 300_000,
        'lgb': 350_000,
        'xgb': 300_000,
        'lgb_tuned': 200_000,
        'xgb_tuned': 150_000,
        'stacking': 120_000,
        'voting': 150_000,
        'lgb_calibrated': 150_000,
    }
    return LIMITS.get(model_key, None)


def _time_ordered_train_val_split(X, y, val_frac=0.2, min_val=200):
    """Split training data into chronological train/validation blocks."""
    n = len(y)
    val_n = max(min_val, int(n * val_frac))
    if n <= val_n + 1:
        return X, y, None, None
    return X[:-val_n], y[:-val_n], X[-val_n:], y[-val_n:]


def build_classifier(model_key, n_features, n_train):
    """Return an sklearn-compatible classifier."""

    if model_key == 'lgb' and HAS_LGB:
        return _LGBWrapper('cls', {
            'objective': 'binary', 'metric': 'auc',
            'learning_rate': 0.05, 'num_leaves': 63, 'max_depth': 7,
            'min_child_samples': 200, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
            'verbose': -1, 'seed': SEED, 'n_jobs': -1,
        })

    elif model_key == 'lgb_tuned' and HAS_LGB and HAS_OPTUNA:
        return _LGBTunedWrapper('cls')

    elif model_key == 'xgb' and HAS_XGB:
        return xgb.XGBClassifier(
            n_estimators=400, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, eval_metric='auc', use_label_encoder=False,
            early_stopping_rounds=30, verbosity=0, random_state=SEED,
            n_jobs=-1, tree_method='hist',
        )

    elif model_key == 'xgb_tuned' and HAS_XGB and HAS_OPTUNA:
        return _XGBTunedWrapper('cls')

    elif model_key == 'histgbt':
        return HistGradientBoostingClassifier(
            max_iter=250, max_depth=7, learning_rate=0.05,
            min_samples_leaf=200, max_leaf_nodes=63,
            l2_regularization=1.0, random_state=SEED,
            early_stopping=True, n_iter_no_change=25,
            validation_fraction=0.1,
        )

    elif model_key == 'rf':
        return RandomForestClassifier(
            n_estimators=220, max_depth=12, min_samples_leaf=100,
            max_features='sqrt', random_state=SEED, n_jobs=-1,
        )

    elif model_key == 'et':
        return ExtraTreesClassifier(
            n_estimators=220, max_depth=12, min_samples_leaf=100,
            max_features='sqrt', random_state=SEED, n_jobs=-1,
        )

    elif model_key == 'adaboost':
        return AdaBoostClassifier(
            n_estimators=120, learning_rate=0.1, random_state=SEED,
        )

    elif model_key == 'logreg_l2':
        return LogisticRegression(
            C=1.0, penalty='l2', max_iter=1000, solver='lbfgs',
            random_state=SEED, n_jobs=-1,
        )

    elif model_key == 'logreg_l1':
        return LogisticRegression(
            C=1.0, penalty='l1', max_iter=1000, solver='saga',
            random_state=SEED, n_jobs=-1,
        )

    elif model_key == 'logreg_en':
        return LogisticRegression(
            C=1.0, penalty='elasticnet', l1_ratio=0.5,
            max_iter=1000, solver='saga', random_state=SEED, n_jobs=-1,
        )

    elif model_key == 'sgd_hinge':
        return CalibratedClassifierCV(
            SGDClassifier(loss='hinge', alpha=1e-4, max_iter=1000,
                          random_state=SEED, n_jobs=-1),
            cv=3, method='sigmoid',
        )

    elif model_key == 'ridge_cls':
        return CalibratedClassifierCV(
            RidgeClassifier(alpha=1.0, random_state=SEED),
            cv=3, method='sigmoid',
        )

    elif model_key == 'knn':
        return KNeighborsClassifier(
            n_neighbors=50, weights='distance', n_jobs=-1,
        )

    elif model_key == 'svm_rbf':
        return SVC(
            C=1.0, kernel='rbf', gamma='scale', probability=True,
            random_state=SEED, max_iter=2500,
        )

    elif model_key == 'mlp_small':
        return MLPClassifier(
            hidden_layer_sizes=(256, 128), activation='relu',
            solver='adam', alpha=1e-4, batch_size=1024,
            learning_rate='adaptive', learning_rate_init=1e-3,
            max_iter=100, early_stopping=True, n_iter_no_change=10,
            validation_fraction=0.1, random_state=SEED,
        )

    elif model_key == 'mlp_large':
        return MLPClassifier(
            hidden_layer_sizes=(512, 256, 128), activation='relu',
            solver='adam', alpha=1e-4, batch_size=2048,
            learning_rate='adaptive', learning_rate_init=1e-3,
            max_iter=140, early_stopping=True, n_iter_no_change=10,
            validation_fraction=0.1, random_state=SEED,
        )

    elif model_key == 'stacking' and HAS_LGB and HAS_XGB:
        return StackingClassifier(
            estimators=[
                ('lgb', HistGradientBoostingClassifier(
                    max_iter=140, max_depth=6, learning_rate=0.05,
                    random_state=SEED, early_stopping=True, n_iter_no_change=15,
                    validation_fraction=0.1)),
                ('xgb', xgb.XGBClassifier(
                    n_estimators=160, max_depth=6, learning_rate=0.05,
                    eval_metric='auc', use_label_encoder=False,
                    verbosity=0, random_state=SEED, n_jobs=-1,
                    tree_method='hist')),
                ('rf', RandomForestClassifier(
                    n_estimators=120, max_depth=10, min_samples_leaf=100,
                    random_state=SEED, n_jobs=-1)),
            ],
            final_estimator=LogisticRegression(C=1.0, max_iter=300),
            cv=3, n_jobs=-1, passthrough=False,
        )

    elif model_key == 'voting' and HAS_LGB and HAS_XGB:
        return VotingClassifier(
            estimators=[
                ('lgb', HistGradientBoostingClassifier(
                    max_iter=140, max_depth=6, learning_rate=0.05,
                    random_state=SEED, early_stopping=True, n_iter_no_change=15,
                    validation_fraction=0.1)),
                ('xgb', xgb.XGBClassifier(
                    n_estimators=160, max_depth=6, learning_rate=0.05,
                    eval_metric='auc', use_label_encoder=False,
                    verbosity=0, random_state=SEED, n_jobs=-1,
                    tree_method='hist')),
                ('rf', RandomForestClassifier(
                    n_estimators=120, max_depth=10, min_samples_leaf=100,
                    random_state=SEED, n_jobs=-1)),
            ],
            voting='soft', n_jobs=-1,
        )

    elif model_key == 'naive_bayes':
        return GaussianNB()

    elif model_key == 'lgb_calibrated' and HAS_LGB:
        return CalibratedClassifierCV(
            lgb.LGBMClassifier(
                n_estimators=240, max_depth=7, learning_rate=0.05,
                num_leaves=63, min_child_samples=200, subsample=0.8,
                colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                verbose=-1, random_state=SEED, n_jobs=-1,
            ),
            cv=2, method='sigmoid',
        )

    else:
        raise ValueError(f"Model '{model_key}' not available (missing library?)")


def build_regressor(model_key, n_features, n_train):
    """Return an sklearn-compatible regressor."""

    if model_key == 'lgb_reg' and HAS_LGB:
        return _LGBWrapper('reg', {
            'objective': 'huber', 'metric': 'mae',
            'learning_rate': 0.05, 'num_leaves': 63, 'max_depth': 7,
            'min_child_samples': 200, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
            'verbose': -1, 'seed': SEED, 'n_jobs': -1,
        })

    elif model_key == 'lgb_reg_tuned' and HAS_LGB and HAS_OPTUNA:
        return _LGBTunedWrapper('reg')

    elif model_key == 'xgb_reg' and HAS_XGB:
        return xgb.XGBRegressor(
            n_estimators=1000, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, eval_metric='mae',
            early_stopping_rounds=50, verbosity=0, random_state=SEED,
            n_jobs=-1, tree_method='hist',
        )



    elif model_key == 'histgbt_reg':
        return HistGradientBoostingRegressor(
            max_iter=500, max_depth=7, learning_rate=0.05,
            min_samples_leaf=200, max_leaf_nodes=63,
            l2_regularization=1.0, loss='absolute_error',
            random_state=SEED, early_stopping=True,
            n_iter_no_change=50, validation_fraction=0.1,
        )

    elif model_key == 'rf_reg':
        return RandomForestRegressor(
            n_estimators=500, max_depth=12, min_samples_leaf=100,
            max_features='sqrt', random_state=SEED, n_jobs=-1,
        )

    elif model_key == 'et_reg':
        return ExtraTreesRegressor(
            n_estimators=500, max_depth=12, min_samples_leaf=100,
            max_features='sqrt', random_state=SEED, n_jobs=-1,
        )

    elif model_key == 'ridge_reg':
        return Ridge(alpha=1.0, random_state=SEED)

    elif model_key == 'lasso_reg':
        return Lasso(alpha=0.01, max_iter=2000, random_state=SEED)

    elif model_key == 'elasticnet_reg':
        return ElasticNet(
            alpha=0.01, l1_ratio=0.5, max_iter=2000, random_state=SEED)

    elif model_key == 'mlp_reg':
        return MLPRegressor(
            hidden_layer_sizes=(256, 128), activation='relu',
            solver='adam', alpha=1e-4, batch_size=1024,
            learning_rate='adaptive', learning_rate_init=1e-3,
            max_iter=200, early_stopping=True, n_iter_no_change=20,
            validation_fraction=0.1, random_state=SEED,
        )

    elif model_key == 'svr_linear':
        return LinearSVR(
            C=1.0, epsilon=0.1, max_iter=5000, random_state=SEED)

    elif model_key == 'knn_reg':
        return KNeighborsRegressor(
            n_neighbors=50, weights='distance', n_jobs=-1)

    else:
        raise ValueError(f"Regressor '{model_key}' not available.")


# ═════════════════════════════════════════════════════════════════
# CUSTOM WRAPPERS (LightGBM native API with early stopping)
# ═════════════════════════════════════════════════════════════════

class _LGBWrapper:
    """Thin wrapper around lgb.train() for walk-forward with early stopping."""
    def __init__(self, task, params):
        self.task = task
        self.params = params
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        dtrain = lgb.Dataset(X_train, label=y_train)
        callbacks = []
        valid_sets = [dtrain]
        if X_val is not None:
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            valid_sets.append(dval)
            callbacks.append(lgb.early_stopping(50, verbose=False))
        self.model = lgb.train(
            self.params, dtrain, num_boost_round=500,
            valid_sets=valid_sets, callbacks=callbacks)
        return self

    def predict_proba(self, X):
        preds = self.model.predict(X)
        return np.column_stack([1 - preds, preds])

    def predict(self, X):
        return self.model.predict(X)

    def feature_importance(self):
        return self.model.feature_importance(importance_type='gain')

    @property
    def _is_lgb_wrapper(self):
        return True


class _LGBTunedWrapper:
    """Optuna-tuned LightGBM."""
    def __init__(self, task):
        self.task = task
        self.model = None
        self.best_params = None

    def _get_lgb_objective(self, y):
        """Detect binary vs multiclass from labels."""
        n_classes = len(np.unique(y))
        if self.task == 'reg':
            return {'objective': 'huber', 'metric': 'mae'}
        elif n_classes <= 2:
            return {'objective': 'binary', 'metric': 'auc'}
        else:
            return {'objective': 'multiclass', 'metric': 'multi_logloss',
                    'num_class': n_classes}

    def _score(self, y_true, preds, n_classes):
        """Evaluate predictions — handles binary, multiclass, regression."""
        if self.task == 'reg':
            return -mean_absolute_error(y_true, preds)
        elif n_classes <= 2:
            return roc_auc_score(y_true, preds)
        else:
            # preds is (n_samples, n_classes) for multiclass
            return roc_auc_score(y_true, preds, multi_class='ovr', average='macro')

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        obj_params = self._get_lgb_objective(y_train)
        n_classes = len(np.unique(y_train))

        if X_val is None:
            # Can't tune without val set — fall back to defaults
            fallback = {
                'learning_rate': 0.05, 'num_leaves': 63, 'max_depth': 7,
                'verbose': -1, 'seed': SEED, 'n_jobs': -1,
            }
            fallback.update(obj_params)
            wrapper = _LGBWrapper(self.task, fallback)
            wrapper.fit(X_train, y_train, X_val, y_val)
            self.model = wrapper.model
            return self

        def objective(trial):
            params = {
                'verbose': -1, 'seed': SEED, 'n_jobs': -1,
                'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 255),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_samples': trial.suggest_int('min_child_samples', 20, 500),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
            }
            params.update(obj_params)
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            model = lgb.train(
                params, dtrain, num_boost_round=250,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(20, verbose=False)])
            preds = model.predict(X_val)
            return self._score(y_val, preds, n_classes)

        study = optuna.create_study(
            direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
        self.best_params = study.best_params

        # Retrain with best params
        best_p = {
            'verbose': -1, 'seed': SEED, 'n_jobs': -1,
            'learning_rate': self.best_params['lr'],
            'num_leaves': self.best_params['num_leaves'],
            'max_depth': self.best_params['max_depth'],
            'min_child_samples': self.best_params['min_child_samples'],
            'subsample': self.best_params['subsample'],
            'colsample_bytree': self.best_params['colsample_bytree'],
            'reg_alpha': self.best_params['reg_alpha'],
            'reg_lambda': self.best_params['reg_lambda'],
        }
        best_p.update(obj_params)
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        self.model = lgb.train(
            best_p, dtrain, num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(25, verbose=False)])
        return self

    def predict_proba(self, X):
        preds = self.model.predict(X)
        return np.column_stack([1 - preds, preds])

    def predict(self, X):
        return self.model.predict(X)

    @property
    def _is_lgb_wrapper(self):
        return True


class _XGBTunedWrapper:
    """Optuna-tuned XGBoost."""
    def __init__(self, task):
        self.task = task
        self.model = None
        self.best_params = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is None or not HAS_OPTUNA:
            self.model = xgb.XGBClassifier(
                n_estimators=250, max_depth=7, learning_rate=0.05,
                verbosity=0, random_state=SEED, n_jobs=-1, tree_method='hist')
            self.model.fit(X_train, y_train)
            return self

        def objective(trial):
            params = {
                'n_estimators': 250,
                'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
                'gamma': trial.suggest_float('gamma', 0, 5),
            }
            model = xgb.XGBClassifier(
                **params, eval_metric='auc', use_label_encoder=False,
                early_stopping_rounds=20, verbosity=0, random_state=SEED,
                n_jobs=-1, tree_method='hist')
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, preds)

        study = optuna.create_study(
            direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
        self.best_params = study.best_params

        self.model = xgb.XGBClassifier(
            n_estimators=400,
            learning_rate=self.best_params['lr'],
            max_depth=self.best_params['max_depth'],
            subsample=self.best_params['subsample'],
            colsample_bytree=self.best_params['colsample_bytree'],
            reg_alpha=self.best_params['reg_alpha'],
            reg_lambda=self.best_params['reg_lambda'],
            min_child_weight=self.best_params['min_child_weight'],
            gamma=self.best_params['gamma'],
            eval_metric='auc', use_label_encoder=False,
            early_stopping_rounds=25, verbosity=0,
            random_state=SEED, n_jobs=-1, tree_method='hist')
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)

    @property
    def _is_lgb_wrapper(self):
        return False


# ═════════════════════════════════════════════════════════════════
# EVALUATION
# ═════════════════════════════════════════════════════════════════

def eval_binary(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        'AUC':       roc_auc_score(y_true, y_prob),
        'Accuracy':  accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall':    recall_score(y_true, y_pred, zero_division=0),
        'F1':        f1_score(y_true, y_pred, zero_division=0),
        'Brier':     brier_score_loss(y_true, y_prob),
        'LogLoss':   log_loss(y_true, y_prob, labels=[0, 1]),
        'pos_rate':  float(y_true.mean()),
        'n':         len(y_true),
    }
    return metrics


def eval_multiclass(y_true, y_prob):
    """Evaluate 3-class predictions."""
    y_pred = y_prob.argmax(axis=1)
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Kappa':    cohen_kappa_score(y_true, y_pred),
        'n':        len(y_true),
    }
    try:
        metrics['AUC_ovr'] = roc_auc_score(
            y_true, y_prob, multi_class='ovr', average='macro')
    except:
        metrics['AUC_ovr'] = float('nan')
    return metrics


def eval_regression(y_true, y_pred):
    return {
        'RMSE':   np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE':    mean_absolute_error(y_true, y_pred),
        'R2':     r2_score(y_true, y_pred),
        'DirAcc': accuracy_score(
            (y_true > np.median(y_true)).astype(int),
            (y_pred > np.median(y_true)).astype(int)),
        'n':      len(y_true),
    }


# ═════════════════════════════════════════════════════════════════
# CORE RUNNER
# ═════════════════════════════════════════════════════════════════

def run_single_model(df, feat_cols, target_key, model_key, splits, outdir):
    """Train & evaluate one model with walk-forward CV. Save results."""
    target_col, task_type, _ = TARGET_MAP[target_key]
    y, task_type, meta = build_target(df, target_key)

    # Check library availability
    info = ALL_MODELS.get(model_key)
    if info is None:
        print(f"  SKIP: '{model_key}' not in registry")
        return None

    model_name, req_lib, model_task = info

    is_regression = model_task == 'reg'
    if is_regression and task_type != 'regression':
        print(f"  SKIP: regressor '{model_key}' can't do task '{task_type}'")
        return None
    if not is_regression and task_type == 'regression':
        print(f"  SKIP: classifier '{model_key}' can't do regression")
        return None

    needs_scale = _needs_scaling(model_key)
    subsample_limit = _needs_subsample(model_key)

    # Filter to available features
    feat_available = [f for f in feat_cols if f in df.columns]

    # ── D_b look-ahead guard ────────────────────────────────
    # D_b aggregates price displacements at 1m/3m/5m/10m.  For targets
    # whose horizon < 10m, D_b (and its derived features) leak future
    # information and must be excluded.
    if target_key in DB_LEAKY_TARGETS:
        n_before = len(feat_available)
        feat_available = [f for f in feat_available if f not in DB_TAINTED_FEATURES]
        n_dropped = n_before - len(feat_available)
        if n_dropped:
            print(f"  ⚠ D_b look-ahead guard: dropped {n_dropped} tainted feature(s) "
                  f"for target '{target_key}'")

    X_all = df[feat_available].values.astype(np.float32)

    fold_results = []
    all_y_true = []
    all_y_pred = []

    scaler = StandardScaler() if needs_scale else None

    t_start = time.time()

    for fold_i, (train_idx, test_idx, month_label) in enumerate(splits):
        X_train = X_all[train_idx]
        X_test  = X_all[test_idx]
        y_train = y[train_idx]
        y_test  = y[test_idx]

        # Subsample if needed
        if subsample_limit and len(X_train) > subsample_limit:
            # Keep the most recent observations to preserve time order.
            X_train_fit_raw = X_train[-subsample_limit:]
            y_train_fit_raw = y_train[-subsample_limit:]
        else:
            X_train_fit_raw = X_train
            y_train_fit_raw = y_train

        X_train_fit, y_train_fit, X_val_raw, y_val = _time_ordered_train_val_split(
            X_train_fit_raw, y_train_fit_raw)

        if task_type in {'binary', 'multiclass'} and len(np.unique(y_train_fit)) < 2:
            print(f"  SKIP fold {fold_i}: training set collapsed to one class")
            continue

        if X_val_raw is not None and task_type in {'binary', 'multiclass'} and len(np.unique(y_val)) < 2:
            X_val_raw, y_val = None, None

        # Scale
        if needs_scale:
            scaler.fit(X_train_fit)
            X_train_fit = scaler.transform(X_train_fit)
            X_val_s     = scaler.transform(X_val_raw) if X_val_raw is not None else None
            X_test_s    = scaler.transform(X_test)
        else:
            X_val_s = X_val_raw
            X_test_s = X_test

        # Build model
        try:
            if is_regression:
                model = build_regressor(model_key, len(feat_available), len(X_train_fit))
            else:
                model = build_classifier(model_key, len(feat_available), len(X_train_fit))
        except ValueError as e:
            print(f"  SKIP fold {fold_i}: {e}")
            return None

        # Fit
        try:
            if hasattr(model, '_is_lgb_wrapper') and model._is_lgb_wrapper:
                model.fit(X_train_fit, y_train_fit, X_val_s, y_val)
            elif hasattr(model, 'fit'):
                # XGBoost with eval_set
                if isinstance(model, (xgb.XGBClassifier if HAS_XGB else type(None),
                                      xgb.XGBRegressor if HAS_XGB else type(None))):
                    if X_val_s is not None:
                        model.fit(X_train_fit, y_train_fit,
                                  eval_set=[(X_val_s, y_val)], verbose=False)
                    else:
                        model.fit(X_train_fit, y_train_fit, verbose=False)
                else:
                    model.fit(X_train_fit, y_train_fit)
        except Exception as e:
            print(f"  ERROR fold {fold_i} ({model_key}): {e}")
            continue

        # Predict
        try:
            if is_regression:
                y_pred = model.predict(X_test_s)
                metrics = eval_regression(y_test, y_pred)
            elif task_type == 'binary':
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X_test_s)
                    if y_prob.ndim == 2:
                        y_prob = y_prob[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_prob = model.decision_function(X_test_s)
                    # Rescale to [0,1]
                    y_prob = 1 / (1 + np.exp(-y_prob))
                else:
                    y_prob = model.predict(X_test_s).astype(float)
                metrics = eval_binary(y_test, y_prob)
                all_y_pred.extend(y_prob)
            elif task_type == 'multiclass':
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X_test_s)
                else:
                    y_pred_mc = model.predict(X_test_s)
                    y_prob = np.eye(3)[y_pred_mc.astype(int)]
                metrics = eval_multiclass(y_test, y_prob)

            all_y_true.extend(y_test)

        except Exception as e:
            print(f"  ERROR predict fold {fold_i} ({model_key}): {e}")
            continue

        metrics['month'] = month_label
        metrics['n_train'] = len(X_train_fit)
        metrics['n_test'] = len(y_test)
        fold_results.append(metrics)

        # Progress
        primary = metrics.get('AUC', metrics.get('AUC_ovr', metrics.get('MAE', 0)))
        print(f"    [{fold_i+1:2d}/{len(splits)}] {month_label}  "
              f"primary={primary:.4f}  n_test={len(y_test):,}")

    elapsed = time.time() - t_start

    if not fold_results:
        print(f"  No successful folds for {model_key}")
        return None

    # ── Pooled metrics ───────────────────────────────────────
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred) if all_y_pred else None

    if task_type == 'binary' and all_y_pred is not None:
        pooled = eval_binary(all_y_true, all_y_pred)
    elif task_type == 'regression':
        pooled = {
            'RMSE': np.mean([r['RMSE'] for r in fold_results]),
            'MAE':  np.mean([r['MAE'] for r in fold_results]),
            'R2':   np.mean([r['R2'] for r in fold_results]),
            'DirAcc': np.mean([r['DirAcc'] for r in fold_results]),
        }
    else:
        pooled = {
            'Accuracy': np.mean([r['Accuracy'] for r in fold_results]),
        }

    # ── Monthly AUC timeseries ───────────────────────────────
    monthly = {}
    for r in fold_results:
        key = r['month']
        if 'AUC' in r:
            monthly[key] = round(r['AUC'], 4)
        elif 'MAE' in r:
            monthly[key] = round(r['MAE'], 4)

    result = {
        'model_key': model_key,
        'model_name': model_name,
        'target': target_key,
        'task_type': task_type,
        'n_features': len(feat_available),
        'features': feat_available,
        'db_features_dropped': target_key in DB_LEAKY_TARGETS,
        'n_folds': len(fold_results),
        'elapsed_sec': round(elapsed, 1),
        'pooled': {k: float(v) if isinstance(v, (float, np.floating)) else v
                   for k, v in pooled.items()},
        'monthly': monthly,
        'fold_results': fold_results,
        'meta': meta,
    }

    # ── Save ─────────────────────────────────────────────────
    out_path = os.path.join(outdir, f'{model_key}__{target_key}.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  ✓ {model_name} ({target_key}): saved → {out_path}  "
          f"[{elapsed:.0f}s]")

    return result


# ═════════════════════════════════════════════════════════════════
# COMPARISON & REPORTING
# ═════════════════════════════════════════════════════════════════

def aggregate_results(outdir):
    """Load all result JSONs and build comparison table."""
    results = []
    for fname in sorted(os.listdir(outdir)):
        if fname.endswith('.json') and '__' in fname:
            with open(os.path.join(outdir, fname)) as f:
                results.append(json.load(f))

    if not results:
        return

    # ── Classification comparison ────────────────────────────
    cls_results = [r for r in results if r['task_type'] == 'binary']
    reg_results = [r for r in results if r['task_type'] == 'regression']

    if cls_results:
        print("\n" + "=" * 90)
        print("CLASSIFICATION LEADERBOARD")
        print("=" * 90)
        cls_results.sort(key=lambda r: r['pooled'].get('AUC', 0), reverse=True)
        print(f"{'Rank':<5} {'Model':<30} {'Target':<12} {'AUC':>7} "
              f"{'Acc':>7} {'F1':>7} {'Brier':>7} {'Time':>6}")
        print("-" * 90)
        for i, r in enumerate(cls_results):
            p = r['pooled']
            print(f"{i+1:<5} {r['model_name']:<30} {r['target']:<12} "
                  f"{p.get('AUC', 0):>7.4f} {p.get('Accuracy', 0):>7.4f} "
                  f"{p.get('F1', 0):>7.4f} {p.get('Brier', 1):>7.4f} "
                  f"{r['elapsed_sec']:>5.0f}s")

    if reg_results:
        print("\n" + "=" * 90)
        print("REGRESSION LEADERBOARD")
        print("=" * 90)
        reg_results.sort(key=lambda r: r['pooled'].get('MAE', 1e9))
        print(f"{'Rank':<5} {'Model':<30} {'Target':<12} "
              f"{'MAE':>10} {'RMSE':>10} {'R²':>8} {'DirAcc':>7} {'Time':>6}")
        print("-" * 90)
        for i, r in enumerate(reg_results):
            p = r['pooled']
            print(f"{i+1:<5} {r['model_name']:<30} {r['target']:<12} "
                  f"{p.get('MAE', 0):>10.4f} {p.get('RMSE', 0):>10.4f} "
                  f"{p.get('R2', 0):>8.4f} {p.get('DirAcc', 0):>7.4f} "
                  f"{r['elapsed_sec']:>5.0f}s")

    # ── Save Markdown leaderboard ────────────────────────────
    md_path = os.path.join(outdir, 'leaderboard.md')
    with open(md_path, 'w') as f:
        f.write("# Model Zoo — Permanence Prediction Leaderboard\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        if cls_results:
            f.write("## Classification (target: φ > 0)\n\n")
            f.write("| Rank | Model | Target | AUC | Accuracy | F1 | Brier | Time |\n")
            f.write("|------|-------|--------|-----|----------|----|-------|------|\n")
            for i, r in enumerate(cls_results):
                p = r['pooled']
                f.write(f"| {i+1} | {r['model_name']} | {r['target']} | "
                        f"{p.get('AUC',0):.4f} | {p.get('Accuracy',0):.4f} | "
                        f"{p.get('F1',0):.4f} | {p.get('Brier',1):.4f} | "
                        f"{r['elapsed_sec']:.0f}s |\n")

        if reg_results:
            f.write("\n## Regression (target: φ continuous)\n\n")
            f.write("| Rank | Model | Target | MAE | RMSE | R² | DirAcc | Time |\n")
            f.write("|------|-------|--------|-----|------|----|--------|------|\n")
            for i, r in enumerate(reg_results):
                p = r['pooled']
                f.write(f"| {i+1} | {r['model_name']} | {r['target']} | "
                        f"{p.get('MAE',0):.4f} | {p.get('RMSE',0):.4f} | "
                        f"{p.get('R2',0):.4f} | {p.get('DirAcc',0):.4f} | "
                        f"{r['elapsed_sec']:.0f}s |\n")

    print(f"\n  → {md_path}")

    # ── Comparison plots ─────────────────────────────────────
    if cls_results:
        _plot_auc_comparison(cls_results, outdir)
        _plot_monthly_auc_heatmap(cls_results, outdir)

    return results


def _plot_auc_comparison(cls_results, outdir):
    """Bar chart comparing AUC across models."""
    names = [r['model_name'] for r in cls_results]
    aucs  = [r['pooled'].get('AUC', 0) for r in cls_results]
    targets = [r['target'] for r in cls_results]
    labels = [f"{n}\n({t})" for n, t in zip(names, targets)]

    fig, ax = plt.subplots(figsize=(14, max(6, len(names) * 0.35)))
    colors = ['#2196F3' if a > 0.55 else '#90CAF9' if a > 0.52 else '#BBDEFB'
              for a in aucs]
    bars = ax.barh(range(len(names)), aucs, color=colors, edgecolor='white')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Pooled AUC (walk-forward)')
    ax.set_title('Model Zoo — Classification AUC Comparison', fontweight='bold')
    ax.axvline(0.5, color='red', ls='--', lw=1, label='Random')
    ax.legend()
    ax.set_xlim(0.45, max(aucs) + 0.02)

    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{auc:.4f}', va='center', fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'auc_comparison.png'), dpi=150)
    plt.close(fig)
    print(f"  → {outdir}/auc_comparison.png")


def _plot_monthly_auc_heatmap(cls_results, outdir):
    """Heatmap of monthly AUC for top models."""
    # Only cls_close target
    close_results = [r for r in cls_results if r['target'] == 'cls_close']
    if len(close_results) < 2:
        return

    close_results.sort(key=lambda r: r['pooled'].get('AUC', 0), reverse=True)
    top = close_results[:min(12, len(close_results))]

    all_months = sorted(set(m for r in top for m in r['monthly'].keys()))
    data = np.full((len(top), len(all_months)), np.nan)
    for i, r in enumerate(top):
        for j, m in enumerate(all_months):
            data[i, j] = r['monthly'].get(m, np.nan)

    fig, ax = plt.subplots(figsize=(16, max(4, len(top) * 0.5)))
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0.45, vmax=0.65)
    ax.set_xticks(range(len(all_months)))
    ax.set_xticklabels(all_months, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([r['model_name'] for r in top], fontsize=8)
    ax.set_title('Monthly AUC Heatmap — Top Models (cls_close)', fontweight='bold')
    fig.colorbar(im, ax=ax, label='AUC', shrink=0.7)

    # Annotate
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f'{data[i,j]:.2f}', ha='center', va='center',
                        fontsize=5, color='black' if 0.48 < data[i,j] < 0.58 else 'white')

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'monthly_auc_heatmap.png'), dpi=150)
    plt.close(fig)
    print(f"  → {outdir}/monthly_auc_heatmap.png")


# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════

def get_model_list(model_arg):
    """Parse --model argument into list of model keys."""
    if model_arg == 'all':
        return list(ALL_MODELS.keys())
    elif model_arg == 'all_cls':
        return list(MODEL_REGISTRY_CLS.keys())
    elif model_arg == 'all_reg':
        return list(MODEL_REGISTRY_REG.keys())
    elif model_arg == 'fast':
        # Quick subset for testing
        return ['lgb', 'xgb', 'histgbt', 'rf', 'logreg_l2', 'mlp_small',
                'lgb_reg', 'xgb_reg', 'rf_reg', 'ridge_reg']
    elif model_arg == 'boosted':
        return ['lgb', 'lgb_tuned', 'xgb', 'xgb_tuned', 'histgbt',
                'lgb_reg', 'lgb_reg_tuned', 'xgb_reg', 'histgbt_reg']
    elif model_arg == 'aggregate':
        # Just aggregate existing results
        return []
    else:
        return model_arg.split(',')


def get_target_list(target_arg):
    """Parse --target argument into list of target keys."""
    if target_arg == 'all':
        return list(TARGET_MAP.keys())
    elif target_arg == 'all_cls':
        return [k for k, v in TARGET_MAP.items() if v[1] != 'regression']
    elif target_arg == 'all_reg':
        return [k for k, v in TARGET_MAP.items() if v[1] == 'regression']
    elif target_arg == 'all_horizons':
        return ['cls_close', 'cls_1m', 'cls_3m', 'cls_5m', 'cls_10m']
    elif target_arg == 'short':
        # Intraday horizons — run on UNFILTERED data only
        return ['cls_1m', 'cls_3m', 'cls_5m', 'cls_10m']
    elif target_arg == 'long':
        # End-of-day / next-day — valid on filtered data
        return ['cls_close', 'cls_clop', 'cls_clcl']
    else:
        return target_arg.split(',')


def main():
    ap = argparse.ArgumentParser(
        description="Model Zoo: Comprehensive ML for permanence prediction.")
    ap.add_argument('bursts_csv',
        help='Filtered burst CSV with Perm_* columns')
    ap.add_argument('--model', default='all',
        help='Model(s): all, all_cls, all_reg, fast, boosted, aggregate, '
             'or comma-separated keys (e.g. lgb,xgb,rf)')
    ap.add_argument('--target', default='cls_close',
        help='Target(s): all, all_cls, all_reg, all_horizons, '
             'or comma-separated (e.g. cls_close,cls_1m,reg_close)')
    ap.add_argument('--features', default='extended',
        choices=['base', 'extended'],
        help='Feature set to use (default: extended)')
    ap.add_argument('--outdir', default='results/zoo/',
        help='Output directory (default: results/zoo/)')
    ap.add_argument('--min-train-months', type=int, default=3,
        help='Min training months before first test fold')
    ap.add_argument('--slurm-index', type=int, default=None,
        help='SLURM_ARRAY_TASK_ID — run only the Nth (model, target) combo')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    model_keys = get_model_list(args.model)
    target_keys = get_target_list(args.target)

    # Handle aggregate-only mode
    if args.model == 'aggregate':
        aggregate_results(args.outdir)
        return

    feat_cols = EXTENDED_FEATURE_COLS if args.features == 'extended' else BASE_FEATURE_COLS

    # ── Load data ────────────────────────────────────────────
    print(f"Loading {args.bursts_csv} …")
    df = pd.read_csv(args.bursts_csv)
    print(f"  {len(df):,} bursts")

    # Rename Volume → BurstVolume if needed
    if 'Volume' in df.columns and 'BurstVolume' not in df.columns:
        df['BurstVolume'] = df['Volume']

    print("Engineering features …")
    df = engineer_features(df)

    # Drop NaNs only for columns required by the requested targets.
    # Using all target columns can over-drop rows unnecessarily.
    target_cols = sorted({TARGET_MAP[k][0] for k in target_keys if k in TARGET_MAP})
    valid_mask = df[target_cols].notna().all(axis=1)
    df = df[valid_mask].reset_index(drop=True)
    print(f"  {len(df):,} bursts after dropping missing targets")

    # Walk-forward splits
    splits = walk_forward_splits(df, min_train_months=args.min_train_months)
    print(f"  {len(splits)} walk-forward folds ({splits[0][2]} → {splits[-1][2]})")

    # ── Build job list ───────────────────────────────────────
    jobs = []
    for mk in model_keys:
        info = ALL_MODELS.get(mk)
        if info is None:
            print(f"  WARN: unknown model '{mk}', skipping")
            continue
        _, _, model_task = info

        for tk in target_keys:
            _, task_type, _ = TARGET_MAP[tk]
            # Match model task to target task
            if model_task == 'reg' and task_type != 'regression':
                continue
            if model_task == 'cls' and task_type == 'regression':
                continue
            jobs.append((mk, tk))

    print(f"\n  Total jobs: {len(jobs)}")

    # ── SLURM array: run only one job ────────────────────────
    if args.slurm_index is not None:
        if args.slurm_index >= len(jobs):
            print(f"  SLURM index {args.slurm_index} >= {len(jobs)} jobs — nothing to do")
            return
        jobs = [jobs[args.slurm_index]]
        print(f"  SLURM job {args.slurm_index}: {jobs[0]}")

    # ── Run all jobs ─────────────────────────────────────────
    all_results = []
    for i, (mk, tk) in enumerate(jobs):
        print(f"\n{'═'*70}")
        print(f"  [{i+1}/{len(jobs)}] Model={mk}  Target={tk}")
        print(f"{'═'*70}")

        result = run_single_model(df, feat_cols, tk, mk, splits, args.outdir)
        if result:
            all_results.append(result)

    # ── Aggregate ────────────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n{'═'*70}")
        print("AGGREGATING RESULTS")
        print(f"{'═'*70}")
        aggregate_results(args.outdir)

    # ── Print available libraries ────────────────────────────
    print(f"\n  Libraries: LightGBM={HAS_LGB}  XGBoost={HAS_XGB}  "
          f"CatBoost={HAS_CB}  Optuna={HAS_OPTUNA}")
    print("Done.")


if __name__ == '__main__':
    main()
