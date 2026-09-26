#!/usr/bin/env python3
"""P4 revisit v1, POST-HOC robustness check (not pre-registered): does the Q2 institutional association survive a
control for the stock's contemporaneous quarterly return?

Why: a burst is "informative" when price moved its way over ten minutes, so quarterly informative flow is mechanically
aligned with the quarter's own return, and institutional ownership changes co-move with contemporaneous returns
(Sias, Starks and Titman 2006). The pre-registered Q2(b)/(c) regressions control for the *previous* quarter's return
only, so the informative-minus-other contrast could be a return channel. This re-runs (b) 13F, (c) mutual-fund
holdings and (c) flow-induced trading with ret_q, the compounded return over the same calendar quarter, added to
the pre-registered controls. Nothing else changes: same inputs, same code path, same winsorization and clustering.
It can only weaken the pre-registered result, not create one. Output: analysis/{CELL}_q2ext_retctrl.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import p4_analyze as PA
import p4_external_tests as X

_orig_panel = X.quarterly_panel
_orig_fe = X.fe_regression


def panel_with_ret(g, state):
    q = _orig_panel(g, state)
    s = state.set_index(["permno", "quarter"])
    return q.join(s[["ret"]].rename(columns={"ret": "ret_q"}), on=["permno", "quarter"])


def run(cell, d):
    nd_path = Path(d) / ("nameday_%s.csv.gz" % cell)
    PA.check_protocol(cell, [nd_path])
    nd = pd.read_csv(nd_path, dtype={"date": str})
    years = sorted(nd.date.str[:4].astype(int).unique())
    c = X.crsp_daily(sorted(set(years) | {min(years) - 1}))
    state = X.quarter_end_state(c)
    cusip_hist = pd.read_csv(X.DATA / "cusip_hist.csv.gz", dtype={"cusip": str})
    out = dict(cell=cell, post_hoc=True, inputs={nd_path.name: PA.sha(nd_path)})
    for fam in sorted(nd.family.unique()):
        q = panel_with_ret(X.signals(nd, fam), state)
        diag = q[["q_info", "q_other", "ret_q"]].replace([np.inf, -np.inf], np.nan).dropna()
        r = dict(corr_info_ret=float(diag.q_info.corr(diag.ret_q, method="spearman")),
                 corr_other_ret=float(diag.q_other.corr(diag.ret_q, method="spearman")))
        for label, extra in (("base", []), ("ret_ctrl", ["ret_q"])):
            X.fe_regression = (lambda df, y, xs, _e=extra, **kw: _orig_fe(df, y, xs + _e, **kw))
            r[label] = dict(b_13F=X.test_13f(q, cusip_hist), c_MF=X.test_mf(q, years), c_FIT=X.test_fit(q, years))
        X.fe_regression = _orig_fe
        out[fam] = r
    path = PA.ANALYSIS / ("%s_q2ext_retctrl.json" % cell)
    path.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=["VAL", "TEST", "ERA2"])
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    run(a.cell, a.dir)
