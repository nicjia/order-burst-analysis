#!/usr/bin/env python3
"""burst-feature probes (ideas 7 and 10): what the per-burst sample says about hidden-liquidity share.

Idea 7: does the share of a burst's volume executed against hidden liquidity predict how much of its impact
holds to the close (P4 d_close)? Controls: log size/ADV, peak impact, spread at the burst, truncated share,
time of day, program score. Day fixed effects, PERMNO-clustered. Trade bursts only.
Usage: burst_feature_probes.py CELL [SAMPLE_PATH]
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import p4_external_tests as X

ROOT = Path(__file__).resolve().parents[1]
cell = sys.argv[1]
path = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "results" / "p4_revisit_v1" / "agg" / cell / ("sample_%s.csv.gz" % cell))
cols = ["permno", "date", "family", "hidden_share", "truncated_share", "program_score", "log_q_adv", "peak_bps",
        "spread_b", "tod", "d_close", "d_open", "ratio", "mode_share"]
d = pd.read_csv(path, usecols=cols, dtype={"date": str})
d = d[d.family == "T"].drop(columns="family").replace([np.inf, -np.inf], np.nan)
d["q_info"] = d.hidden_share
d["q_other"] = d.mode_share
res = dict(cell=cell, n=int(len(d)), hidden_share_mean=float(d.hidden_share.mean()),
           hidden_share_p90=float(d.hidden_share.quantile(0.9)))
xs = ["q_info", "q_other", "log_q_adv", "peak_bps", "spread_b", "truncated_share", "program_score", "tod"]
for y in ("d_close", "d_open", "ratio"):
    r = X.fe_regression(d, y, xs, fe="date", cluster="permno")
    res[y] = r
    if r:
        print("%-5s %-8s n %7d | hidden_share %+7.3f (t %5.2f) | mode_share %+7.3f (t %5.2f) | peak %+6.3f (t %5.2f)"
              % (cell, y, r["n"], r["q_info"]["b"], r["q_info"]["t"], r["q_other"]["b"], r["q_other"]["t"],
                 r["peak_bps"]["b"], r["peak_bps"]["t"]))
(ROOT / "results" / "burst_probes_v1").mkdir(parents=True, exist_ok=True)
(ROOT / "results" / "burst_probes_v1" / ("%s.json" % cell)).write_text(json.dumps(res, indent=1, default=float))
