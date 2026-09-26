#!/usr/bin/env python3
"""Is the short-horizon burst forecast mechanical? (v4 events; models trained on the train stocks 2022-23; 2025 test.)

A burst that takes out the best quote widens the spread and displaces the mid; when the level refills, the mid moves
back. If the forecast were only that refill, it would vanish in bursts that left the spread unchanged and did not
move the mid. The 2025 IC of the full model (and of controls + book, and the gain) is split by what the burst did to
the book: spread at the decision vs just before the burst in ticks (unchanged / widened / narrowed), the mid's move
during the burst (none / with the burst / against it), both clean conditions together, and a one-tick spread at the
decision. Models are fitted once on all training events; only the evaluation is split.
"""
import sys as _sys, pathlib as _pl  # shared project modules live in src_py/
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "src_py"))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import p4_analyze as PA
import burst_defs_raw2_model as M
import burst_defs_raw4_model as V4

DEFS = ("early5", "run0.01", "run0.1", "levelclear")
HOR = ("r1", "r10", "r60", "r300", "r1800_x", "r_close_x")


def main():
    train = set(pd.read_csv(M.ROOT / "results" / "p4_revisit_v1" / "jobs" / "DEFS_RAW_train_names.txt", header=None)[0])
    d = V4.load_events()
    d = d[d.kind == "real"].copy()
    s = d.side.to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        m_d = d.m_close.to_numpy(float) / (1 + s * d.r_close.to_numpy(float) / 1e4)
        m_b = m_d / (1 + s * d.move_during.to_numpy(float) / 1e4)
        d["tick_dec"] = np.rint(d.spread_dec.to_numpy(float) / 1e4 * m_d / 0.01)
        d["tick_b"] = np.rint(d.spread_b.to_numpy(float) / 1e4 * m_b / 0.01)
    d = d[np.isfinite(d.tick_dec) & np.isfinite(d.tick_b)]
    subsets = {
        "all": lambda x: np.ones(len(x), bool),
        "spread unchanged (ticks)": lambda x: (x.tick_dec == x.tick_b).to_numpy(),
        "spread widened": lambda x: (x.tick_dec > x.tick_b).to_numpy(),
        "spread narrowed": lambda x: (x.tick_dec < x.tick_b).to_numpy(),
        "mid did not move during the burst": lambda x: (x.move_during == 0).to_numpy(),
        "mid moved with the burst": lambda x: (x.move_during > 0).to_numpy(),
        "mid moved against the burst": lambda x: (x.move_during < 0).to_numpy(),
        "spread unchanged AND mid did not move": lambda x: ((x.tick_dec == x.tick_b) & (x.move_during == 0)).to_numpy(),
        "one-tick spread at the decision": lambda x: (x.tick_dec == 1).to_numpy(),
        "spread of 2+ ticks at the decision": lambda x: (x.tick_dec >= 2).to_numpy(),
    }
    F = V4.CTRL + V4.BOOK + V4.BURST
    tr_all, tests = V4.split(d, train)
    te_all = tests["2025 all stocks"]
    rows = []
    for dn in DEFS:
        tr, te = tr_all[tr_all.defn == dn], te_all[te_all.defn == dn]
        for y in HOR:
            mf, lo, hi = M.hgb(tr, F, y); mb, _, _ = M.hgb(tr, V4.CTRL + V4.BOOK, y)
            e = te[te[y].notna()]
            pf = mf.predict(e[F].to_numpy(float)); pb = mb.predict(e[V4.CTRL + V4.BOOK].to_numpy(float))
            yy = e[y].clip(lo, hi).to_numpy(float); dates = e.date.to_numpy()
            for nm, fn in subsets.items():
                k = fn(e)
                if k.sum() < 2000:
                    continue
                sf, icf, _ = M.daily_ic(dates[k], pf[k], yy[k]); sb, icb, _ = M.daily_ic(dates[k], pb[k], yy[k])
                _, icm, _ = M.daily_ic(dates[k], e.move_during.to_numpy(float)[k], yy[k])
                common = sf.index.intersection(sb.index)
                g = PA.nw_t((sf[common] - sb[common]).to_numpy()) if len(sf) == len(sb) else PA.nw_t((sf - sb).to_numpy())
                nz = yy[k] != 0
                rows.append(dict(defn=dn, horizon=y, subset=nm, share=float(k.mean()), ic=icf["mean"], t=icf["t"],
                                 ic_ctrl_book=icb["mean"], gain=g["mean"], t_gain=g["t"], move_only_ic=icm["mean"],
                                 zero_share=float(1 - nz.mean())))
        print(dn, "done", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(V4.D4 / "v4_mechanical.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    order = list(subsets)
    for v in ("ic", "gain", "t_gain", "move_only_ic"):
        print("\n### %s (2025 all stocks)" % v)
        print(R.pivot_table(index=["defn", "subset"], columns="horizon", values=v).reindex(columns=list(HOR))
              .reindex(pd.MultiIndex.from_product([DEFS, order])).dropna(how="all").round(3).to_string())
    print("\n### share of events")
    print(R[R.horizon == "r10"].pivot_table(index="subset", columns="defn", values="share").reindex(order).round(3).to_string())


if __name__ == "__main__":
    main()
