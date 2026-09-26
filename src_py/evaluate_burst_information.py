#!/usr/bin/env python3
"""Frozen exploratory state/regime/burst comparisons on identical prospective decisions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import burst_information_extract as BI


MODELS = {
    "ridge_state": BI.BASE,
    "gbt_state": BI.BASE,
    "gbt_state_regime": BI.BASE + BI.REGIME,
    "gbt_state_regime_burst": BI.BASE + BI.REGIME + BI.BURST,
    "gbt_state_regime_burst_score": BI.BASE + BI.REGIME + BI.BURST + ["fragment_score"],
}
CONTRASTS = (
    ("nonlinearity", "ridge_state", "gbt_state"),
    ("regime", "gbt_state", "gbt_state_regime"),
    ("burst", "gbt_state_regime", "gbt_state_regime_burst"),
    ("simulation_score", "gbt_state_regime_burst", "gbt_state_regime_burst_score"),
)


def nw(values, lags=10):
    x = np.asarray(values, float)
    if not len(x) or not np.isfinite(x).all():
        return dict(n=len(x), mean=None, se=None, t=None, lo95=None, hi95=None)
    mean = float(x.mean()); u = x - mean; n = len(x)
    if n < 2:
        return dict(n=n, mean=mean, se=None, t=None, lo95=None, hi95=None)
    lag = min(lags, n - 1)
    variance = float(np.dot(u, u) / n)
    for k in range(1, lag + 1):
        variance += 2 * (1 - k / (lag + 1)) * float(np.dot(u[k:], u[:-k]) / n)
    se = float(np.sqrt(max(variance, 0) / n))
    return dict(n=n, mean=mean, se=se, t=mean / se if se > 0 else None,
                lo95=mean - 1.96 * se, hi95=mean + 1.96 * se)


def day_stat(frame, values):
    temp = frame[["ticker", "date"]].copy()
    temp["value"] = np.asarray(values, float)
    daily = temp.groupby(["date", "ticker"]).value.mean().groupby("date").mean().sort_index()
    return nw(daily.to_numpy())


def matrix(frame, columns):
    out = frame[columns].to_numpy(float).copy()
    for j, name in enumerate(columns):
        if (name.startswith(("count", "volume", "ambiguous_count"))
                or name in ("n_packets", "duration", "depth_start", "intensity")):
            out[:, j] = np.sign(out[:, j]) * np.log1p(np.abs(out[:, j]))
    return out


def nonoverlap(frame):
    """Common, score-independent arrival schedule for all execution policies."""
    chosen = []
    for _key, group in frame.groupby(["ticker", "date"]):
        until = -np.inf
        for idx, row in group.sort_values(["decision_time", "row_id"]).iterrows():
            if row.decision_time >= until:
                chosen.append(idx)
                until = row.reference_time + 60
    return chosen


def audit_inputs(root, spec):
    receipts = []; failures = []
    for ticker in spec["seen_names"] + spec["heldout_names"]:
        for date in spec["dates"]:
            status = root / "status" / ticker / (date + ".txt")
            value = status.read_text().strip() if status.exists() else "absent"
            receipts.append(dict(ticker=ticker, date=date, status=value))
            if value not in ("ok", "missing"):
                failures.append(receipts[-1])
            if value == "ok" and not (root / "rows" / ticker / (date + ".csv")).is_file():
                failures.append(dict(ticker=ticker, date=date, status="missing_output"))
    if failures:
        raise ValueError("incomplete/failed input receipts: " + repr(failures[:10]))
    return receipts


def load(root, spec):
    receipts = audit_inputs(root, spec)
    frames = []
    for receipt in receipts:
        if receipt["status"] != "ok":
            continue
        path = root / "rows" / receipt["ticker"] / (receipt["date"] + ".csv")
        metadata = json.loads((root / "status" / receipt["ticker"] /
                               (receipt["date"] + ".json")).read_text())
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            if metadata["rows"] != 0:
                raise ValueError("empty output contradicts row-count receipt " + str(path))
            continue
        if len(frame) != metadata["rows"]:
            raise ValueError("output row-count mismatch " + str(path))
        if not frame.empty:
            if not (frame.ticker == receipt["ticker"]).all() or not (frame.date == int(receipt["date"])).all():
                raise ValueError("incorrect ticker/date " + str(path))
            frames.append(frame)
    if not frames:
        raise ValueError("no valid panels")
    data = pd.concat(frames, ignore_index=True)
    if data.row_id.duplicated().any():
        raise ValueError("duplicate landmarks")
    if not set(data.stage).issubset(set(BI.STAGES)):
        raise ValueError("unexpected landmark stage")
    if not np.allclose(data.reference_time - data.decision_time, 1.0, atol=1e-9, rtol=0):
        raise ValueError("incorrect action latency")
    audit = {"receipts": pd.DataFrame(receipts).status.value_counts().to_dict(),
             "n_rows": len(data), "invalid": int((~data.valid).sum()),
             "extreme_price": int(data.extreme_price.sum()),
             "input_design_sha256": hashlib.sha256((root / "design.json").read_bytes()).hexdigest()}
    usable = data.valid & ~data.extreme_price
    audit["usable_name_days"] = int(data[usable][["ticker", "date"]].drop_duplicates().shape[0])
    return data[usable].reset_index(drop=True), audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root); spec = json.loads((root / "design.json").read_text())
    data, audit = load(root, spec)
    seen = set(spec["seen_names"])
    data["cohort"] = np.where(data.ticker.isin(seen), "seen", "heldout")
    summaries = []; execution = []; daily_rows = []
    prediction_dir = root / "predictions"; prediction_dir.mkdir(exist_ok=True)
    model_dir = root / "models"; model_dir.mkdir(exist_ok=True)
    for stage in BI.STAGES:
        train = data[(data.date // 10000 == spec["training_year"]) & data.ticker.isin(seen)
                     & (data.stage == stage)].copy()
        test = data[(data.date // 10000 == spec["evaluation_year"]) & (data.stage == stage)].copy()
        if len(train) < 100 or test.empty or set(test.cohort) != {"seen", "heldout"}:
            raise ValueError("insufficient training/test cohorts for " + stage)
        weights = 1 / train.groupby(["ticker", "date"]).row_id.transform("count").to_numpy()
        weights *= len(weights) / weights.sum()
        for target in BI.TARGETS:
            panel = test[["ticker", "date", "row_id", "stage", "cohort", "decision_time",
                          "reference_time", "reference_depth", "wait_depth", target]].copy()
            panel = panel.rename(columns={target: "target"})
            ytrain = train[target].to_numpy(float)
            for name, features in MODELS.items():
                xtrain = matrix(train, features); xtest = matrix(test, features)
                if name == "ridge_state":
                    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
                    model.fit(xtrain, ytrain, ridge__sample_weight=weights,
                              standardscaler__sample_weight=weights)
                else:
                    model = HistGradientBoostingRegressor(**spec["gbt"])
                    model.fit(xtrain, ytrain, sample_weight=weights)
                panel[name] = model.predict(xtest)
                joblib.dump(dict(model=model, features=features,
                                 training_year=spec["training_year"], training_names=sorted(seen)),
                            model_dir / (stage + "_" + target + "_" + name + ".joblib"))
            panel.to_csv(prediction_dir / (stage + "_" + target + ".csv.gz"), index=False)
            for cohort, group in panel.groupby("cohort"):
                y = group.target.to_numpy(float)
                for label, base, aug in CONTRASTS:
                    loss_base = (y - group[base].to_numpy()) ** 2
                    loss_aug = (y - group[aug].to_numpy()) ** 2
                    delta = loss_base - loss_aug
                    stat = day_stat(group, delta)
                    base_stat = day_stat(group, loss_base)
                    stat.update(stage=stage, target=target, cohort=cohort, contrast=label,
                                base_mse=base_stat["mean"], n_rows=len(group),
                                n_names=int(group.ticker.nunique()),
                                improvement_fraction=stat["mean"] / base_stat["mean"]
                                if base_stat["mean"] else None)
                    summaries.append(stat)
                    dd = group[["ticker", "date"]].copy(); dd["delta"] = delta
                    dd = dd.groupby(["ticker", "date"]).delta.mean().reset_index()
                    dd = dd.assign(stage=stage, target=target, cohort=cohort, contrast=label)
                    daily_rows.extend(dd.to_dict(orient="records"))
                if target == "wait_cost_60s":
                    # Diagnostic economics only, never a claim of a passed profitability gate.
                    eligible = group[group.reference_depth >= 1]
                    eligible = eligible.loc[nonoverlap(eligible)]
                    if (eligible.wait_depth < 1).any():
                        raise ValueError("invalid future execution book: cannot silently select opportunities using future depth")
                    costs = eligible.target.to_numpy()
                    for name in MODELS:
                        # Predicting that waiting is cheaper => wait. Every opportunity fills.
                        policy = np.where(eligible[name].to_numpy() < 0, costs, 0.0)
                        baseline_policy = np.where(eligible.gbt_state_regime.to_numpy() < 0, costs, 0.0)
                        result = dict(stage=stage, cohort=cohort, model=name, n_orders=len(eligible),
                                      savings_vs_now=day_stat(eligible, -policy),
                                      savings_vs_wait=day_stat(eligible, costs - policy),
                                      savings_vs_state_regime=day_stat(eligible, baseline_policy - policy),
                                      wait_fraction=float((eligible[name] < 0).mean()))
                        execution.append(result)
            print("finished", stage, target, "train", len(train), "test", len(test), flush=True)
    pd.DataFrame(daily_rows).to_csv(root / "daily_comparisons.csv", index=False)
    result = {"experiment": spec["experiment"], "scope": spec["status"], "audit": audit,
              "runtime": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
              "inference_note": "Exploratory; 20 sampled evaluation dates, NW10 over observed dates. No fresh-holdout claim; no significance-based selection.",
              "comparisons": summaries, "execution_diagnostics": execution,
              "execution_note": "Indicative one-share touch costs only. Equal fixed fees cancel. No finite-size impact, routing, or passive queue simulation."}
    (root / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
