#!/usr/bin/env python3
"""Aggregate metaorder-v1 M3 (synthetic parent calibration), 2024 exploration."""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

import metaorder_inject as MI

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results" / "metaorder_v1" / "explore_2024" / "inject"


def main():
    parents = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(D / "*.csv")))], ignore_index=True)
    live = parents[parents.n_children > 0].copy()
    live["recall_children"] = live.children_in_bursts / live.n_children
    live["link_recall"] = np.where(live.majority_bursts > 0, live.majority_bursts_linked_same_parent / live.majority_bursts.clip(lower=1), np.nan)
    out = dict(parents=int(len(parents)), parents_injected=int(len(live)), names=int(parents.ticker.nunique()),
               skipped_participation_cap=int((parents.n_children == 0).sum()))

    def summ(g):
        return dict(parents=int(len(g)), median_children=float(g.n_children.median()),
                    child_recall=float(g.recall_children.mean()), mean_purity=float(g.mean_purity.mean()),
                    share_with_majority_bursts=float((g.majority_bursts > 0).mean()),
                    majority_bursts_per_parent=float(g.majority_bursts.mean()),
                    link_recall=float(g.link_recall.mean()),
                    mean_score_program=float(g.mean_score_program.mean()), mean_score_link=float(g.mean_score_link.mean()))
    out["all"] = summ(live)
    for col in ("fixed", "locked", "interval", "duration"):
        out["by_" + col] = {str(k): summ(g) for k, g in live.groupby(col)}
    s_all, y_all, l_all = [], [], []
    bursts = []
    for p in sorted(glob.glob(str(D / "*.bursts.json"))):
        for day in json.loads(Path(p).read_text()):
            if "_injected_majority" not in day:
                continue
            y = np.array(day["_injected_majority"], bool)
            s_all.append(np.array(day["_scores_program"])); l_all.append(np.array(day["_scores_link"])); y_all.append(y)
            bursts.append({k: v for k, v in day.items() if not k.startswith("_")})
    b = pd.DataFrame(bursts)
    out["pooled_auc_program"] = MI.auc(np.concatenate(s_all), np.concatenate(y_all))
    out["pooled_auc_link"] = MI.auc(np.concatenate(l_all), np.concatenate(y_all))
    out["median_name_day_auc_program"] = float(b.auc_program.median()); out["median_name_day_auc_link"] = float(b.auc_link.median())
    out["injected_majority_bursts"] = int(b.injected_majority.sum()); out["bursts"] = int(b.bursts.sum())
    out["link_ratio_injected"] = float(b.link_matches_injected.sum() / b.link_expected_injected.sum())
    out["link_ratio_background"] = float(b.link_matches_background.sum() / b.link_expected_background.sum())
    (ROOT / "results" / "metaorder_v1" / "m3_explore_2024.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
