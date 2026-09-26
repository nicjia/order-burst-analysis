#!/usr/bin/env python3
"""Paired legacy/session-corrected join fits on new synthetic train and test days."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import fragment_reconstruction as FR
import metaorder_models as MM
import metaorder_join_v2 as V2
import metaorder_simulation as MS


def metrics(y, prob):
    chosen = prob >= 0.5
    tp = int(((y == 1) & chosen).sum()); positives = int(y.sum())
    return dict(n=len(y), positive_rate=float(y.mean()), brier=float(np.mean((y - prob) ** 2)),
                average_precision=float(average_precision_score(y, prob)),
                auc=float(roc_auc_score(y, prob)), selected=int(chosen.sum()),
                precision=tp / int(chosen.sum()) if chosen.any() else None,
                recall=tp / positives if positives else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-days", type=int, default=30)
    ap.add_argument("--test-days", type=int, default=18)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model = MM.RidgeLogit.from_dict(json.loads(Path("config/metaorder_simulation_model.json").read_text())["fragment_model"])
    train = []; test = []
    scenarios = ("splitting", "herding", "overlap", "pauses", "partial", "full")
    for day in range(args.train_days + args.test_days):
        packets = MS.simulate_day(seed=610000 + day, scenario=scenarios[day % len(scenarios)])
        f = MM.add_simulated_fragment_labels(FR.form_fragments(packets), packets)
        f = MM.score_fragments(f, model)
        f["simulation_day"] = day; f["scenario"] = scenarios[day % len(scenarios)]
        (train if day < args.train_days else test).append(f)
    train = pd.concat(train, ignore_index=True); test = pd.concat(test, ignore_index=True)
    legacy_pairs = MM.join_feature_frame(train)
    legacy_pairs = legacy_pairs[legacy_pairs.gap <= 1800].copy()
    a = train.loc[legacy_pairs.left_index]; b = train.loc[legacy_pairs.right_index]
    crosses = a.simulation_day.to_numpy() != b.simulation_day.to_numpy()
    old_label = (a.dominant_parent.to_numpy() >= 0) & (a.dominant_parent.to_numpy() == b.dominant_parent.to_numpy())
    legacy_model = MM.fit_join_model(train)
    corrected_model = V2.fit_join_model(train)
    pairs = V2.join_feature_frame(test)
    pairs = pairs[pairs.gap <= 1800].copy()
    y = V2.pair_labels(test, pairs)
    x = pairs[MM.JOIN_FEATURES].to_numpy(float)
    rows = pairs.copy(); rows["label"] = y
    rows["left_day"] = test.loc[pairs.left_index, "simulation_day"].to_numpy()
    rows["right_day"] = test.loc[pairs.right_index, "simulation_day"].to_numpy()
    rows["left_parent"] = test.loc[pairs.left_index, "dominant_parent"].to_numpy()
    rows["right_parent"] = test.loc[pairs.right_index, "dominant_parent"].to_numpy()
    rows["simulation_day"] = test.loc[pairs.left_index, "simulation_day"].to_numpy()
    rows["scenario"] = test.loc[pairs.left_index, "scenario"].to_numpy()
    rows["legacy_probability"] = legacy_model.predict_proba(x)
    rows["corrected_probability"] = corrected_model.predict_proba(x)
    result = dict(experiment="join-session-diagnostic-v1", train_days=args.train_days,
                  test_days=args.test_days, train_fragments=len(train), test_fragments=len(test),
                  legacy_candidate_pairs=len(legacy_pairs), legacy_cross_day_fraction=float(crosses.mean()),
                  legacy_positive_labels=int(old_label.sum()),
                  cross_day_positive_labels=int((crosses & old_label).sum()),
                  legacy_on_valid_test_pairs=metrics(y, rows.legacy_probability.to_numpy()),
                  corrected_on_valid_test_pairs=metrics(y, rows.corrected_probability.to_numpy()),
                  scope="Synthetic new days only; target is matching dominant parents, not real parent identification. Existing frozen model untouched.")
    daily = []
    for day, group in rows.groupby("simulation_day"):
        delta = (group.label - group.legacy_probability) ** 2 - (group.label - group.corrected_probability) ** 2
        daily.append(dict(day=int(day), scenario=str(group.scenario.iloc[0]), delta_brier=float(delta.mean())))
    result["daily_delta_brier"] = daily
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path.with_suffix(".csv"), index=False)
    training_receipt = legacy_pairs.copy()
    training_receipt["left_day"] = a.simulation_day.to_numpy()
    training_receipt["right_day"] = b.simulation_day.to_numpy()
    training_receipt["left_parent"] = a.dominant_parent.to_numpy()
    training_receipt["right_parent"] = b.dominant_parent.to_numpy()
    training_receipt["legacy_label"] = old_label.astype(int)
    training_receipt.to_csv(path.with_name(path.stem + "_training_pairs.csv"), index=False)
    path.with_name(path.stem + "_models.json").write_text(json.dumps(
        {"legacy": legacy_model.to_dict(MM.JOIN_FEATURES),
         "session_corrected": corrected_model.to_dict(MM.JOIN_FEATURES)}, indent=2) + "\n")
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "daily_delta_brier"}, indent=2))


if __name__ == "__main__":
    main()
