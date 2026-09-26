#!/usr/bin/env python3
"""burst-pseudo-v1 stage 2: is the real-time short-horizon forecast burst information, or would any recent move do?

Same features for real bursts and matched pseudo events (random time, same duration and side):
CTRL (move since the open, 30-min pre-move, time of day, spread), PATH (move over the event window, 60-s pre-move,
duration), BOOK (queue imbalance at the end and start of the window, quote OFI before and during, trade-flow
imbalance before). Test stocks / 2024 only; models trained on the train stocks, 2022-23.
  real->real     model trained on real bursts, scored on real bursts
  pseudo->pseudo model trained on pseudo events, scored on pseudo events (predictability at any moment)
  real->pseudo   the burst model applied to pseudo events
plus the raw rank IC of the window move alone. Horizons +10 s, +60 s, +300 s, +30 min and close (market-excess).
"""
import sys as _sys, pathlib as _pl  # shared project modules live in src_py/
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "src_py"))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import p4_analyze as PA
import burst_defs_raw2_model as M

F = ["since_open", "pre30m", "tod", "spread_dec", "move_during", "pre60", "dur",
     "qimb_e", "qimb_b", "qofi_pre60", "qofi_during", "tfi_pre60"]
H = ["r10", "r60", "r300", "r1800_x", "r_close_x"]


def main():
    train = set(pd.read_csv(M.ROOT / "results" / "p4_revisit_v1" / "jobs" / "DEFS_RAW_train_names.txt", header=None)[0])
    d = pd.read_csv(M.D / "burst_pseudo" / "PSEUDO_events.csv.gz", dtype={"date": str})
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d[(d.spread_dec > 0) & (d.spread_dec < 500)]
    for c in ("r10", "r60", "r300", "r1800", "r_close"):
        d.loc[d[c].abs() > 1000, c] = np.nan
    d = M.add_market(d, "t_dec", None)
    tr = d[d.permno.isin(train) & d.date.str[:4].isin(["2022", "2023"])]
    te = d[~d.permno.isin(train) & (d.date.str[:4] == "2024")]
    rows = []
    for defn in sorted(d.defn.unique()):
        for y in H:
            out = {}
            for src in ("real", "pseudo"):
                t = tr[(tr.defn == defn) & (tr.kind == src)]
                m, lo, hi = M.hgb(t, F, y)
                for tgt in ("real", "pseudo"):
                    e = te[(te.defn == defn) & (te.kind == tgt) & te[y].notna()]
                    s, ic, sp = M.daily_ic(e.date.to_numpy(), m.predict(e[F].to_numpy(float)), e[y].clip(lo, hi).to_numpy())
                    out["%s->%s" % (src, tgt)] = (s, ic, sp)
            raw = {}
            for tgt in ("real", "pseudo"):
                e = te[(te.defn == defn) & (te.kind == tgt) & te[y].notna()]
                _, ic, _ = M.daily_ic(e.date.to_numpy(), e.move_during.to_numpy(), e[y].to_numpy())
                raw[tgt] = ic
            gap = PA.nw_t((out["real->real"][0] - out["pseudo->pseudo"][0]).dropna().to_numpy())
            rows.append(dict(defn=defn, horizon=y,
                             real_real=out["real->real"][1]["mean"], t_rr=out["real->real"][1]["t"],
                             pseudo_pseudo=out["pseudo->pseudo"][1]["mean"], t_pp=out["pseudo->pseudo"][1]["t"],
                             real_on_pseudo=out["real->pseudo"][1]["mean"],
                             real_minus_pseudo=gap["mean"], t_gap=gap["t"],
                             move_ic_real=raw["real"]["mean"], move_ic_pseudo=raw["pseudo"]["mean"],
                             d10_d1_real=out["real->real"][2]["mean"], d10_d1_pseudo=out["pseudo->pseudo"][2]["mean"]))
        print(defn, "done", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(M.D / "burst_pseudo" / "real_vs_pseudo.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
