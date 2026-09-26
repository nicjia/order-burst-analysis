#!/usr/bin/env python3
"""burst279-oracle2: (a) break-even AUC per exit on a fine grid; (b) the real DEV-trained model against a noisy
oracle with the SAME AUC on the same label -- if the real model's AUC is tradable information, the two earn alike."""
import sys as _sys, pathlib as _pl  # shared project modules live in src_py/
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "src_py"))
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import fp_multiday_h1 as H
import burst279_model as B
import burst279_daily as BD
import burst279_oracle as O

ROOT = Path(__file__).resolve().parents[3]
GRID = [0.505, 0.51, 0.515, 0.52, 0.525, 0.53, 0.54, 0.55]
rng = np.random.default_rng(7)


def noisy_book(base, lab, col, auc):
    sigma = 1.0 / (np.sqrt(2) * norm.ppf(auc))
    s = lab.astype(float) + sigma * rng.standard_normal(len(lab))
    q10, q90 = np.quantile(s, [0.1, 0.9])
    f, d = O.book(base[s >= q90], +1, col), O.book(base[s <= q10], -1, col)
    return f["net"], d["net"]


cal = H.calendar()
tr = BD.load("DEV", cal, 400000)
te = BD.load("TEST", cal, 1500000)
te = te[(te.spread_dec > 0) & (te.spread_dec < 500)].copy(); te["cost"] = te.spread_dec / 2 + B.EXIT_COST_BPS
out = {"breakeven": {}, "real_vs_oracle": {}}
print("(a) NOISY ORACLE net bps per trade, TEST, top-decile follow | bottom-decile fade, label CONT (sign after T_dec)")
print("  exit   " + " ".join("AUC%-6.3f" % a for a in GRID))
for ex, col in (("close", "d_close"), ("open", "d_open"), ("cc", "d_cc")):
    base = te[te[col].notna()]; lab = (base[col] > 0).to_numpy()
    vals = [noisy_book(base, lab, col, a) for a in GRID]
    out["breakeven"][ex] = {str(a): v for a, v in zip(GRID, vals)}
    print("  %-6s " % ex + " ".join("%+4.1f|%+4.1f" % v for v in vals))
print("\n(b) REAL MODEL vs NOISY ORACLE AT THE SAME AUC (TEST, exit at close)")
for lab_name, col_lab, thr in (("PERM_close", "phi_close", 0.5), ("CONT_close", "d_close", 0.0)):
    t = tr[tr[col_lab].notna() & tr.d_close.notna()]
    ytr = (t[col_lab] >= thr) if lab_name.startswith("PERM") else (t[col_lab] > 0)
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.05, min_samples_leaf=200,
                                       early_stopping=False, random_state=1).fit(t[BD.FEATS].to_numpy(float), ytr.astype(int))
    base = te[te[col_lab].notna() & te.d_close.notna()]
    y = ((base[col_lab] >= thr) if lab_name.startswith("PERM") else (base[col_lab] > 0)).to_numpy()
    p = m.predict_proba(base[BD.FEATS].to_numpy(float))[:, 1]
    auc = float(roc_auc_score(y, p))
    q10, q90 = np.quantile(m.predict_proba(t[BD.FEATS].to_numpy(float))[:, 1], [0.1, 0.9])
    rf, rd = O.book(base[p >= q90], +1, "d_close"), O.book(base[p <= q10], -1, "d_close")
    of, od = noisy_book(base, y, "d_close", auc)
    out["real_vs_oracle"][lab_name] = dict(auc=auc, real_follow_net=rf["net"], real_fade_net=rd["net"],
                                           real_follow_gross=rf["gross"], real_fade_gross=rd["gross"], oracle_follow_net=of, oracle_fade_net=od)
    print("  %-10s AUC %.3f | real model: follow net %+6.2f (gross %+5.2f)  fade net %+6.2f (gross %+5.2f) | oracle at same AUC: follow %+6.2f  fade %+6.2f"
          % (lab_name, auc, rf["net"], rf["gross"], rd["net"], rd["gross"], of, od))
(ROOT / "results" / "burst_forecasting" / "burst279_v1" / "oracle2.json").write_text(json.dumps(out, indent=1, default=float))
