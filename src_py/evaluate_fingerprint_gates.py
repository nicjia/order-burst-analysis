#!/usr/bin/env python3
"""Apply the frozen fingerprint-v1 confirmation gates (BURST_FINGERPRINT_DESIGN.md) exactly once.

C1  2021 depth-matched name-median ratio, u_nonround, lower 95% bound > 1 at 0.5-2 s and 2-10 s.
C2  2021 corrected state similarity: upper 95% bound < 1 for spread and log depth at 2-10 s and 10-30 s.
C3  exploration-selected definition ranks top five in 2021 by mean per-name J (u_nonround, min 3,
    depth-matched null); Spearman of mean per-name J across the 33 rule x gap definitions > 0.8.
P1  program score, 2021: top-minus-bottom decile excess per pair, lower 95% bound > 0.
P2  program score, 2021: Spearman(decile, excess per pair) > 0.7.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

NULL = "cross_day_depth_matched"


def j_table(summary):
    t = pd.DataFrame(summary["burst_definitions"])
    t = t[(t.null == NULL) & (t.size_class == "u_nonround") & (t.min_packets == 3)]
    return t.set_index(["rule", "gap_s"]).name_mean_j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explore", required=True)
    ap.add_argument("--confirm", required=True)
    ap.add_argument("--explore-state", required=True)
    ap.add_argument("--confirm-state", required=True)
    ap.add_argument("--score")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ex, co = json.loads(Path(a.explore).read_text()), json.loads(Path(a.confirm).read_text())
    exs, cos = json.loads(Path(a.explore_state).read_text()), json.loads(Path(a.confirm_state).read_text())
    gates = {}

    c1 = {lag: co["e1_existence"]["%s/u_nonround/%s" % (NULL, lag)] for lag in ("0.5-2s", "2-10s")}
    gates["C1"] = dict(passed=all(v["name_median_ratio_ci95"][0] > 1 for v in c1.values()),
                       detail={lag: dict(median=v["name_median_ratio"], ci95=v["name_median_ratio_ci95"],
                                         names=v["names_eligible"]) for lag, v in c1.items()})

    keys = ["2-10s/spread_bps", "2-10s/log_exec_depth", "10-30s/spread_bps", "10-30s/log_exec_depth"]
    gates["C2"] = dict(passed=all(cos["ranges"][k]["name_median_ratio_ci95"][1] < 1 for k in keys),
                       detail={k: dict(confirm_median=cos["ranges"][k]["name_median_ratio"],
                                       confirm_ci95=cos["ranges"][k]["name_median_ratio_ci95"],
                                       explore_median=exs["ranges"][k]["name_median_ratio"],
                                       explore_ci95=exs["ranges"][k]["name_median_ratio_ci95"]) for k in keys})

    sel = ex["selected_definition"]
    rank = next(r["rank"] for r in co["selection"][NULL] if r["rule"] == sel["rule"] and r["gap_s"] == sel["gap_s"])
    je, jc = j_table(ex), j_table(co)
    both = pd.concat([je.rename("explore"), jc.rename("confirm")], axis=1).dropna()
    rho = float(both.explore.corr(both.confirm, method="spearman"))
    gates["C3"] = dict(passed=bool(rank <= 5 and rho > 0.8),
                       detail=dict(selected=dict(rule=sel["rule"], gap_s=sel["gap_s"]), confirm_rank=int(rank),
                                   spearman_j=rho, definitions=int(len(both)),
                                   confirm_top5=[dict(rule=r["rule"], gap_s=r["gap_s"], j=r["name_mean_j"])
                                                 for r in co["selection"][NULL][:5]]))

    if a.score:
        sc = json.loads(Path(a.score).read_text())["out_of_sample"]
        gates["P1"] = dict(passed=bool(sc["ci95"] and sc["ci95"][0] > 0),
                           detail=dict(top_minus_bottom=sc["top_minus_bottom_excess_per_1000_pairs"], ci95=sc["ci95"]))
        gates["P2"] = dict(passed=bool(sc["spearman_decile_excess"] > 0.7), detail=dict(spearman=sc["spearman_decile_excess"]))
    Path(a.out).write_text(json.dumps(gates, indent=1) + "\n")
    for g, v in gates.items():
        print(g, "PASS" if v["passed"] else "FAIL", json.dumps(v["detail"])[:400])


if __name__ == "__main__":
    main()
