#!/usr/bin/env python3
"""Idea 4: does the program score (fit on within-day size repetition, VERIFIED 1.27) predict CROSS-DAY linkage,
a criterion it was never fit on? Keys with >= 3 one-sided bursts are classified linked / mirror / unlinked as in
fingerprint-multiday-v1 amendment A1; their mean program scores are compared with day FE and PERMNO clusters."""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import p4_external_tests as X
import fp_multiday_h1 as H

ROOT = Path(__file__).resolve().parents[1]
cell = sys.argv[1]
cal = H.calendar()
fp = pd.read_csv(H.T / ("%s_fps.csv.gz" % cell), dtype={"date": str})
fp = fp[fp["size"] % 100 != 0].copy()
fp["day"] = fp.date.map(cal).astype(int)
w = fp.pivot_table(index=["permno", "size", "day"], columns="side", values=["nb", "score"], aggfunc={"nb": "sum", "score": "mean"})
w.columns = ["%s_%s" % (a, "B" if b == 1 else "S") for a, b in w.columns]
w = w.reset_index().fillna({"nb_B": 0, "nb_S": 0})
prev = w[["permno", "size", "day", "nb_B", "nb_S"]].rename(columns={"nb_B": "nb_B_p", "nb_S": "nb_S_p"})
prev["day"] = prev.day + 1
w = w.merge(prev, on=["permno", "size", "day"], how="left").fillna({"nb_B_p": 0, "nb_S_p": 0})
rows = []
for side, o in (("B", "S"), ("S", "B")):
    q = w[(w["nb_" + side] >= 3) & (w["nb_" + o] == 0) & w["score_" + side].notna()].copy()
    q["linked"] = ((q["nb_%s_p" % side] >= 3) & (q["nb_%s_p" % o] == 0)).astype(float)
    q["mirror"] = ((q["nb_%s_p" % o] >= 3) & (q["nb_%s_p" % side] == 0)).astype(float)
    rows.append(pd.DataFrame(dict(permno=q.permno, day=q.day, score=q["score_" + side], nb=q["nb_" + side],
                                  size=q["size"], linked=q.linked, mirror=q.mirror)))
d = pd.concat(rows, ignore_index=True)
d["date"] = d.day.astype(str); d["log_nb"] = np.log(d.nb); d["log_size"] = np.log(d["size"])
d = d.rename(columns={"linked": "q_info", "mirror": "q_other"})
r = X.fe_regression(d, "score", ["q_info", "q_other", "log_nb", "log_size"], fe="date", cluster="permno")
res = dict(cell=cell, keys=int(len(d)), linked_share=float(d.q_info.mean()), mirror_share=float(d.q_other.mean()),
           score_linked=float(d.score[d.q_info == 1].mean()), score_mirror=float(d.score[d.q_other == 1].mean()),
           score_unlinked=float(d.score[(d.q_info == 0) & (d.q_other == 0)].mean()), reg=r)
(ROOT / "results" / "burst_probes_v1" / ("score_linkage_%s.json" % cell)).write_text(json.dumps(res, indent=1, default=float))
print("%s keys %d | mean score: linked %.3f  mirror %.3f  unlinked %.3f" % (cell, res["keys"], res["score_linked"], res["score_mirror"], res["score_unlinked"]))
print("  regression: linked %+.4f (t %5.2f)  mirror %+.4f (t %5.2f)  linked-minus-mirror %+.4f (t %5.2f)"
      % (r["q_info"]["b"], r["q_info"]["t"], r["q_other"]["b"], r["q_other"]["t"], r["info_minus_other"]["b"], r["info_minus_other"]["t"]))
