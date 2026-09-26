#!/usr/bin/env python3
"""Stage 3b of program-evidence-v1: the fingerprint-v1 program score for run/60 bursts.

Identical fit and evaluation to program_score.py (imported, unchanged), applied to run/60 burst
rows. Also saves what downstream modules need and program_score.py does not keep: the
standardization, the coefficients, and the 80th-percentile linear score of the 2024 training
bursts that defines "program bursts".
"""
import argparse
import json
from pathlib import Path

import numpy as np

import program_score as PS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(20260913)
    train_all, test_all = PS.load(args.train), PS.load(args.test)
    train, test = PS.usable(train_all), PS.usable(test_all)
    model = PS.fit(train)
    s_train = PS.score(model, train)
    in_table, in_stat = PS.decile_table(train, s_train, rng)
    out_table, out_stat = PS.decile_table(test, PS.score(model, test), rng)
    threshold = float(np.quantile(s_train, 0.8))
    coef = dict(zip(["intercept"] + PS.FEATURES, model["beta"].tolist()))
    result = dict(scope="run/60 bursts; fit on exploration (2024) names; evaluated once on confirmation (2021) names",
                  train_bursts=len(train_all), train_usable=len(train), test_bursts=len(test_all), test_usable=len(test),
                  converged=model["converged"], standardized_coefficients=coef,
                  gates=dict(P1b=bool(out_stat["ci95"] is not None and out_stat["ci95"][0] > 0),
                             P2b=bool(out_stat["spearman_decile_excess"] > 0.7)),
                  in_sample=dict(deciles=in_table.to_dict(orient="records"), **in_stat),
                  out_of_sample=dict(deciles=out_table.to_dict(orient="records"), **out_stat))
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    Path(args.model).write_text(json.dumps(dict(features=PS.FEATURES, mu=model["mu"].tolist(), sd=model["sd"].tolist(),
                                                beta=model["beta"].tolist(), program_threshold_q80=threshold,
                                                rule="run", gap_s=60.0, min_packets=3), indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("in_sample", "out_of_sample")}, indent=1))
    print(out_table.round(3).to_string(index=False)); print(out_stat)


if __name__ == "__main__":
    main()
