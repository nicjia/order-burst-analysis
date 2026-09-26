#!/usr/bin/env python3
"""ClusterLOB replication, local stages.

fit   K-means++ (k = 3) on 2023 sampled events: features log1p-transformed and standardised (means / sds from the
      2023 sample) -> clusterlob_centers.json (mu, sd, centers).
eval  on the 30-minute bucket files from the apply pass (2023 + 2024). Order-flow imbalance per cluster:
      size-based OFI_k / total size in the bucket, count-based OFI_k / events in the bucket; plain OFI likewise.
      Targets as in the paper: CONR (same-bucket mid return), FRNB (next bucket), FREB (bucket end to the close),
      all in excess of the cross-sectional mean of that date-bucket. Clusters are labelled on 2023 only:
      directional = highest correlation with CONR, opportunistic = highest with FREB among the rest, market-making
      = the last. 2024 is the test: pooled correlation with FRNB / FREB, and a sign strategy (sign of OFI x next
      bucket return, equal-weight across stocks per date-bucket, daily P&L, annualised Sharpe), per tick group.
Usage: clusterlob_fit_eval.py fit | eval
"""
import sys as _sys, pathlib as _pl  # shared project modules live in src_py/
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "src_py"))
import json, glob
from pathlib import Path
import numpy as np, pandas as pd
import p4_analyze as PA

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "results" / "burst_forecasting" / "clusterlob"
FEATS = ["vol_level", "t_mid", "t_first", "t_prev", "sbs", "obs"]


def fit():
    from sklearn.cluster import KMeans
    files = glob.glob(str(D / "fit" / "*" / "*.csv.gz"))
    s = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    Z = np.log1p(np.clip(s[FEATS].fillna(0.0).to_numpy(float), 0, None))
    mu, sd = Z.mean(0), Z.std(0); sd[sd == 0] = 1.0
    km = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=20260925).fit((Z - mu) / sd)
    out = dict(mu=mu.tolist(), sd=sd.tolist(), centers=km.cluster_centers_.tolist(), n_fit=int(len(s)), files=len(files))
    (D / "clusterlob_centers.json").write_text(json.dumps(out, indent=1))
    lab = km.labels_
    prof = pd.DataFrame(np.expm1(Z)).assign(k=lab, ty=s.ty.to_numpy()).groupby("k").median()
    prof.columns = FEATS + ["ty_median"]
    print("fitted on %d events from %d stock-days" % (len(s), len(files)))
    print("cluster sizes:", np.bincount(lab).tolist())
    print("median raw features by cluster:\n", prof.round(2).to_string())
    print("event-type mix by cluster:\n", pd.crosstab(lab, s.ty, normalize="index").round(3).to_string())


def evaluate():
    names = pd.read_csv(ROOT / "results" / "p4_revisit_v1" / "jobs" / "CLOB_names.csv")
    grp = dict(zip(names.permno, names.tick_group))
    files = glob.glob(str(D / "apply" / "*" / "*.csv.gz"))
    b = pd.concat([pd.read_csv(f, dtype={"date": str}).assign(permno=int(Path(f).parent.name)) for f in files], ignore_index=True)
    b = b.sort_values(["permno", "date", "bucket"])
    with np.errstate(invalid="ignore", divide="ignore"):
        b["conr"] = np.log(b.m_end / b.m_start) * 1e4
        b["freb"] = np.log(b.m_close / b.m_end) * 1e4
    b["frnb"] = b.groupby(["permno", "date"]).conr.shift(-1)
    for c in ("conr", "frnb", "freb"):
        b[c] = b[c] - b.groupby(["date", "bucket"])[c].transform("mean")       # cross-sectional excess
        b.loc[b[c].abs() > 1000, c] = np.nan
    tot_n = sum(b["n_%d" % k] for k in range(3))
    for k in range(3):
        b["s%d" % k] = b["ofi_size_%d" % k] / b.total_size.where(b.total_size > 0)
        b["c%d" % k] = b["ofi_count_%d" % k] / tot_n.where(tot_n > 0)
    b["s_all"] = b.ofi_size_all / b.total_size.where(b.total_size > 0)
    b["c_all"] = b.ofi_count_all / tot_n.where(tot_n > 0)
    b["tick_group"] = b.permno.map(grp)
    b = b.replace([np.inf, -np.inf], np.nan)
    tr, te = b[b.date.str[:4] == "2023"], b[b.date.str[:4] == "2024"]
    corr = {k: {t: tr[["s%d" % k, t]].corr().iloc[0, 1] for t in ("conr", "frnb", "freb")} for k in range(3)}
    d_ = max(range(3), key=lambda k: corr[k]["conr"])
    rest = [k for k in range(3) if k != d_]
    o_ = max(rest, key=lambda k: corr[k]["freb"])
    m_ = [k for k in rest if k != o_][0]
    names_k = {d_: "directional", o_: "opportunistic", m_: "market-making"}
    print("2023 labelling (corr of size-OFI with CONR / FRNB / FREB):")
    for k in range(3):
        print("  cluster %d -> %-13s %s" % (k, names_k[k], " ".join("%s %+.4f" % (t, v) for t, v in corr[k].items())))
    rows = []
    for g in ["all"] + sorted(te.tick_group.dropna().unique()):
        e = te if g == "all" else te[te.tick_group == g]
        for sig, lab in [("s%d" % k, names_k[k] + " size") for k in range(3)] + [("c%d" % k, names_k[k] + " count") for k in range(3)] + [("s_all", "plain size OFI"), ("c_all", "plain count OFI")]:
            for tgt in ("frnb", "freb"):
                x = e[[sig, tgt, "date", "bucket"]].dropna()
                if len(x) < 500:
                    continue
                pnl = np.sign(x[sig]) * x[tgt]
                day = pnl.groupby(x.date).mean()
                rows.append(dict(tick_group=g, signal=lab, target=tgt.upper(), corr=float(x[sig].corr(x[tgt])),
                                 pnl_bps=float(pnl.mean()), t=PA.nw_t(day.to_numpy())["t"],
                                 sharpe=float(day.mean() / day.std() * np.sqrt(252)) if day.std() > 0 else np.nan,
                                 hit=float((pnl > 0).mean())))
    R = pd.DataFrame(rows)
    R.to_csv(D / "clusterlob_test2024.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 300)
    print("\n2024 TEST (cross-sectional excess returns; sign strategy at the mid, per date-bucket):")
    print(R.round(4).to_string(index=False))


if __name__ == "__main__":
    fit() if _sys.argv[1] == "fit" else evaluate()
