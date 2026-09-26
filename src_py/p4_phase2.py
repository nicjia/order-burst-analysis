#!/usr/bin/env python3
"""P4 revisit v1, Q3 (Phase II): predict post-decision persistence from information available at T_dec.

P4_REVISIT_DESIGN.md section 6, Q3. Fit (local, DEV only): per family, ridge (alpha 1 on standardized features
winsorized at DEV 1/99%) and HistGradientBoostingRegressor (max_depth 3, 300 iterations, learning rate 0.05,
min_samples_leaf 200) of d_close (bps, winsorized 1/99% per year) on the design features, using the aggregator's
per-burst DEV sample (eligible large bursts). Models are exported as JSON (ridge coefficients; boosting trees
as node arrays) so the cluster can predict with numpy only. theta for S_pred is the DEV prediction quantile whose
selection rate equals the realized informative (kappa 0.5) rate. Evaluate (VAL, TEST): daily Spearman rank IC
between prediction and d_close (and d_open, d_cc) on the cell's sample, NW(10); protocol lock as p4_analyze.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import p4_analyze as PA

COMMON = ["log_q_adv", "log_n", "log_duration", "peak_bps", "dmean_bps", "ratio", "d_slope_bps", "spread_b",
          "spread_dec", "tod", "own_open_dec_bps", "pre30_bps", "mode_share", "link_back"]
FAMILY_EXTRA = {"T": ["truncated_share", "program_score"], "S": ["exec_dec", "cancel_dec"]}
MODELS_DIR = Path(__file__).resolve().parents[1] / "results" / "p4_revisit_v1" / "phase2"


def features(df, fam):
    X = pd.DataFrame(index=df.index)
    X["log_q_adv"] = df.log_q_adv
    X["log_n"] = np.log1p(df.n)
    X["log_duration"] = np.log1p(df.t_e - df.t_b)
    X["d_slope_bps"] = df.d600_bps - df.d60_bps
    for c in ("peak_bps", "dmean_bps", "ratio", "spread_b", "spread_dec", "tod", "own_open_dec_bps", "pre30_bps",
              "mode_share", "link_back"):
        X[c] = df[c]
    for c in FAMILY_EXTRA[fam]:
        X[c] = df[c]
    return X[COMMON + FAMILY_EXTRA[fam]].replace([np.inf, -np.inf], np.nan)


def export_hgb(model):
    trees = []
    for it in model._predictors:
        nodes = it[0].nodes
        trees.append(dict(value=nodes["value"].tolist(), feature_idx=nodes["feature_idx"].tolist(),
                          threshold=nodes["num_threshold"].tolist(), missing_left=nodes["missing_go_to_left"].tolist(),
                          left=nodes["left"].tolist(), right=nodes["right"].tolist(), leaf=nodes["is_leaf"].tolist()))
    return dict(kind="hgb", baseline=float(np.ravel(model._baseline_prediction)[0]), trees=trees)


def predict(spec, X):
    """numpy-only prediction from an exported spec (X: 2-D float array, NaN allowed)."""
    X = np.asarray(X, float)
    if spec["kind"] == "ridge":
        Z = (np.clip(X, spec["lo"], spec["hi"]) - spec["mu"]) / spec["sd"]
        Z = np.where(np.isfinite(Z), Z, 0.0)
        return spec["intercept"] + Z @ np.asarray(spec["coef"])
    out = np.full(len(X), spec["baseline"])
    rows = np.arange(len(X))
    for t in spec["trees"]:
        value = np.asarray(t["value"]); feat = np.asarray(t["feature_idx"]); thr = np.asarray(t["threshold"])
        mleft = np.asarray(t["missing_left"], bool); left = np.asarray(t["left"]); right = np.asarray(t["right"])
        leaf = np.asarray(t["leaf"], bool)
        node = np.zeros(len(X), dtype=np.int64)
        active = ~leaf[node]
        while active.any():
            idx = rows[active]; nd = node[active]
            x = X[idx, feat[nd]]
            go_left = np.where(np.isnan(x), mleft[nd], x <= thr[nd])
            node[active] = np.where(go_left, left[nd], right[nd])
            active = ~leaf[node]
        out += value[node]
    return out


def winsor_by_year(y, dates):
    y = pd.Series(np.asarray(y, float)); yr = pd.Series(np.asarray(dates)).str[:4]
    lo = y.groupby(yr).transform(lambda s: s.quantile(0.01)); hi = y.groupby(yr).transform(lambda s: s.quantile(0.99))
    return y.clip(lo, hi).to_numpy()


def fit(sample_path, out_dir, max_rows=3_000_000, seed=20260915):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    df = pd.read_csv(sample_path, dtype={"date": str})
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for fam, g in df.groupby("family"):
        g = g[np.isfinite(g.d_close)]
        if len(g) > max_rows:
            g = g.sample(max_rows, random_state=seed)
        X = features(g, fam); y = winsor_by_year(g.d_close, g.date)
        lo, hi = X.quantile(0.01), X.quantile(0.99)
        Xc = X.clip(lo, hi, axis=1)
        mu, sd = Xc.mean(), Xc.std().replace(0, 1.0)
        Z = ((Xc - mu) / sd).fillna(0.0)
        ridge = Ridge(alpha=1.0).fit(Z.to_numpy(), y)
        rspec = dict(kind="ridge", features=list(X.columns), lo=lo.tolist(), hi=hi.tolist(), mu=mu.tolist(),
                     sd=sd.tolist(), coef=ridge.coef_.tolist(), intercept=float(ridge.intercept_))
        hgb = HistGradientBoostingRegressor(max_depth=3, max_iter=300, learning_rate=0.05, min_samples_leaf=200,
                                            early_stopping=False, random_state=seed).fit(X.to_numpy(float), y)
        hspec = export_hgb(hgb); hspec["features"] = list(X.columns)
        # the export must reproduce sklearn exactly
        chk = X.to_numpy(float)[:20000]
        np.testing.assert_allclose(predict(hspec, chk), hgb.predict(chk), rtol=1e-9, atol=1e-9)
        info_rate = float(g.info_k50.mean())
        for name, spec in (("ridge", rspec), ("hgb", hspec)):
            pred = predict(spec, X.to_numpy(float))
            spec["theta"] = float(np.quantile(pred, 1 - info_rate))
            spec["info_rate_dev"] = info_rate
            (out_dir / ("model_%s_%s.json" % (fam, name))).write_text(json.dumps(spec))
        report[fam] = dict(rows=int(len(g)), info_rate=info_rate, ridge_coef=dict(zip(X.columns, np.round(ridge.coef_, 4).tolist())))
    (out_dir / "fit_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def evaluate(cell, sample_path, model_dir):
    PA.check_protocol(cell, [sample_path])
    df = pd.read_csv(sample_path, dtype={"date": str})
    res = dict(cell=cell, inputs={Path(sample_path).name: PA.sha(sample_path)})
    for fam, g in df.groupby("family"):
        X = features(g, fam).to_numpy(float)
        r = {}
        for name in ("ridge", "hgb"):
            spec = json.loads((Path(model_dir) / ("model_%s_%s.json" % (fam, name))).read_text())
            pred = predict(spec, X)
            f = pd.DataFrame(dict(date=g.date.to_numpy(), pred=pred, d_close=g.d_close.to_numpy(),
                                  d_open=g.d_open.to_numpy(), d_cc=g.d_cc.to_numpy()))
            r[name] = {}
            for target in ("d_close", "d_open", "d_cc"):
                ics = []
                for d, h in f[["date", "pred", target]].dropna().groupby("date"):
                    if len(h) >= 30:
                        ics.append(h.pred.rank().corr(h[target].rank()))
                r[name][target] = PA.nw_t(np.array(ics))
        best = max(("ridge", "hgb"), key=lambda m: r[m]["d_close"]["t"] or -np.inf)
        r["best_model_d_close"] = best
        r["gate_val"] = bool(r[best]["d_close"]["t"] is not None and r[best]["d_close"]["mean"] > 0
                             and PA.p_two_sided(r[best]["d_close"]["t"]) * 2 < 0.0027)   # t > 3 after Bonferroni x2
        res[fam] = r
    out = PA.ANALYSIS / ("%s_q3.json" % cell)
    out.write_text(json.dumps(res, indent=1, default=float))
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit"); f.add_argument("--sample", required=True); f.add_argument("--out", default=str(MODELS_DIR))
    e = sub.add_parser("evaluate"); e.add_argument("--cell", required=True, choices=["VAL", "TEST", "ERA2"])
    e.add_argument("--sample", required=True); e.add_argument("--models", default=str(MODELS_DIR))
    args = ap.parse_args()
    if args.cmd == "fit":
        fit(args.sample, Path(args.out))
    else:
        evaluate(args.cell, args.sample, args.models)


if __name__ == "__main__":
    main()
