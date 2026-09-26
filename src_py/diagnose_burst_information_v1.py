#!/usr/bin/env python3
"""Post-hoc diagnostics for burst-information-v1 (added 2026-09-13, after its results were read).

The frozen v1 evaluation compares models by relative MSE only. These checks ask whether those
comparisons can carry the interpretation given to them:

1. zero_forecast   R^2 of every saved prediction against a zero forecast (targets are signed,
                   so zero is the no-information benchmark), same aggregation as v1.
2. loss_shares     how much of each cohort's baseline MSE and of each contrast comes from each
                   name (raw packet counts are never normalized in v1).
3. normalized      the burst and score contrasts refit on per-name scale-free targets:
                   flow / (trailing 300s signed-packet rate x horizon), return / half-spread.
4. deciles         2024 within-name deciles of a 2023-fitted ridge return forecast, gross and
                   net of crossing the spread at the one-second reference time.
5. surprise        per-name 2023 persistence model of future same-side flow; 2024 returns
                   regressed on its predicted flow and on the surprise.

Everything here is exploratory: chosen after the v1 outcomes were seen, on years already used.
No v1 file is modified; outputs go to <root>/posthoc/.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import burst_information_extract as BI
import evaluate_burst_information as EV

MODELS = list(EV.MODELS)
SIGNED_BURST = [c if c != "depth_imbalance_start" else "signed_depth_imbalance_start" for c in BI.BURST]


def agg_mean(frame, values):
    t = frame[["ticker", "date"]].assign(v=np.asarray(values, float))
    return t.groupby(["date", "ticker"]).v.mean().groupby("date").mean()


def zero_forecast(root):
    rows = []
    for stage in BI.STAGES:
        for target in BI.TARGETS:
            p = pd.read_csv(root / "predictions" / ("%s_%s.csv.gz" % (stage, target)))
            for cohort, g in p.groupby("cohort"):
                y = g.target.to_numpy()
                mse0 = agg_mean(g, y ** 2).mean()
                rec = dict(stage=stage, target=target, cohort=cohort, mse_zero_forecast=mse0)
                for m in MODELS:
                    rec["r2_vs_zero_" + m] = 1 - agg_mean(g, (y - g[m].to_numpy()) ** 2).mean() / mse0
                beats = [1 - ((h.target - h.ridge_state) ** 2).sum() / (h.target ** 2).sum() > 0
                         for _tk, h in g.groupby("ticker")]
                rec["names_ridge_beats_zero"] = int(sum(beats)); rec["names"] = len(beats)
                rows.append(rec)
    return pd.DataFrame(rows)


def loss_shares(root):
    rows = []
    for stage in BI.STAGES:
        for target in BI.TARGETS:
            p = pd.read_csv(root / "predictions" / ("%s_%s.csv.gz" % (stage, target)))
            for cohort, g in p.groupby("cohort"):
                y = g.target.to_numpy()
                base = (y - g.gbt_state_regime.to_numpy()) ** 2
                delta = base - (y - g.gbt_state_regime_burst.to_numpy()) ** 2
                t = g[["ticker", "date"]].assign(b=base, d=delta)
                nd = t.groupby(["date", "ticker"]).mean().reset_index()
                n = nd.groupby("date").ticker.transform("count")
                share_b = (nd.b / n).groupby(nd.ticker).sum(); share_b /= share_b.sum()
                share_d = (nd.d / n).groupby(nd.ticker).sum()
                top = share_b.sort_values(ascending=False)
                rows.append(dict(stage=stage, target=target, cohort=cohort,
                                 top1=top.index[0], top1_share=top.iloc[0],
                                 top2=top.index[1], top2_share=top.iloc[1],
                                 top2_share_of_burst_delta=float(share_d[top.index[:2]].sum() / share_d.sum())
                                 if share_d.sum() != 0 else None))
    return pd.DataFrame(rows)


def load_rows(root, spec):
    data, _audit = EV.load(root, spec)
    data["cohort"] = np.where(data.ticker.isin(set(spec["seen_names"])), "seen", "heldout")
    data["year"] = data.date // 10000
    rate = data.count_300s / 300.0
    half = (data.spread_bps / 2).clip(lower=1e-6)
    for h in (60, 300):
        data["flow_%ds_n" % h] = data["flow_%ds" % h] / np.maximum(1.0, rate * h)
        data["return_%ds_n" % h] = data["return_%ds" % h] / half
    data["signed_depth_imbalance_start"] = data.sign * data.depth_imbalance_start
    data["entry_cost_bps"] = data.sign * (data.reference_touch - data.reference_mid) / data.reference_mid * 1e4
    return data


def fit(kind, x, y, w, gbt):
    if kind == "ridge":
        m = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        m.fit(x, y, ridge__sample_weight=w, standardscaler__sample_weight=w)
    else:
        m = HistGradientBoostingRegressor(**gbt)
        m.fit(x, y, sample_weight=w)
    return m


def normalized(data, spec):
    blocks = {"state_regime": BI.BASE + BI.REGIME,
              "burst": BI.BASE + BI.REGIME + BI.BURST,
              "burst_score": BI.BASE + BI.REGIME + BI.BURST + ["fragment_score"],
              "burst_signed_imbalance": BI.BASE + BI.REGIME + SIGNED_BURST}
    rows = []
    for stage in ("third", "completion"):
        tr = data[(data.year == 2023) & (data.cohort == "seen") & (data.stage == stage)]
        te = data[(data.year == 2024) & (data.stage == stage)]
        w = 1 / tr.groupby(["ticker", "date"]).row_id.transform("count").to_numpy(); w *= len(w) / w.sum()
        for target in ("flow_60s_n", "flow_300s_n", "return_60s_n", "return_300s_n"):
            for kind in ("ridge", "gbt"):
                pred = {b: fit(kind, EV.matrix(tr, c), tr[target].to_numpy(), w, spec["gbt"]).predict(EV.matrix(te, c))
                        for b, c in blocks.items()}
                for cohort in ("seen", "heldout"):
                    g = te.cohort.to_numpy() == cohort
                    y = te[target].to_numpy()[g]; frame = te[g]
                    base = (y - pred["state_regime"][g]) ** 2
                    r2 = lambda p: np.mean([1 - ((h[target] - p[frame.index.get_indexer(h.index)]) ** 2).sum()
                                            / (h[target] ** 2).sum() for _tk, h in frame.groupby("ticker")])
                    for aug in ("burst", "burst_score", "burst_signed_imbalance"):
                        ref = "burst" if aug == "burst_score" else "state_regime"
                        lb = (y - pred[ref][g]) ** 2; la = (y - pred[aug][g]) ** 2
                        daily = agg_mean(frame, lb - la); daily_base = agg_mean(frame, lb)
                        s = EV.nw(daily.to_numpy())
                        per_name = frame[["ticker"]].assign(d=lb - la, b=lb).groupby("ticker").sum()
                        rows.append(dict(stage=stage, target=target, model=kind, cohort=cohort,
                                         contrast="%s vs %s" % (aug, ref),
                                         improvement_pct=100 * s["mean"] / daily_base.mean(), nw_t=s["t"],
                                         names_improved=int((per_name.d > 0).sum()), names=len(per_name),
                                         mean_name_r2_vs_zero_reference=r2(pred[ref][g]),
                                         mean_name_r2_vs_zero_augmented=r2(pred[aug][g])))
    return pd.DataFrame(rows)


def deciles(data):
    cols = BI.BASE + BI.REGIME + BI.BURST + ["fragment_score"]
    rows = []
    for stage in ("third", "completion"):
        for h in (60, 300):
            target = "return_%ds" % h
            tr = data[(data.year == 2023) & (data.stage == stage)]
            te = data[(data.year == 2024) & (data.stage == stage)].copy()
            w = 1 / tr.groupby(["ticker", "date"]).row_id.transform("count").to_numpy(); w *= len(w) / w.sum()
            m = fit("ridge", EV.matrix(tr, cols), tr[target].to_numpy(), w, None)
            te["pred"] = m.predict(EV.matrix(te, cols))
            te["decile"] = te.groupby("ticker").pred.transform(lambda x: pd.qcut(x.rank(method="first"), 10, labels=False))
            te["gross"] = te[target]; te["net_one_way"] = te[target] - te.entry_cost_bps
            te["net_round_trip"] = te[target] - 2 * te.entry_cost_bps
            byname = te.groupby(["decile", "ticker"])[["gross", "net_one_way", "net_round_trip", "entry_cost_bps"]].mean()
            d = byname.groupby("decile").mean()
            d["names_net_one_way_positive"] = byname.net_one_way.gt(0).groupby("decile").mean()
            for dec, r in d.iterrows():
                rows.append(dict(stage=stage, horizon_s=h, decile=int(dec), **r.to_dict()))
    return pd.DataFrame(rows)


def surprise(data):
    rows = []
    for stage in ("third", "completion"):
        for h in (60, 300):
            f, r = "flow_%ds" % h, "return_%ds" % h
            for tk, g in data[data.stage == stage].groupby("ticker"):
                a, b = g[g.year == 2023], g[g.year == 2024]
                if len(a) < 150 or len(b) < 150:
                    continue
                X = lambda x: np.column_stack([np.ones(len(x)), x.count_imbalance_60s, x.count_imbalance_300s, x.count_300s])
                beta = np.linalg.lstsq(X(a), a[f].to_numpy(), rcond=None)[0]
                pf = X(b) @ beta; yf = b[f].to_numpy(); sur = yf - pf
                Z = np.column_stack([np.ones(len(b)), pf, sur]); y = b[r].to_numpy()
                c = np.linalg.lstsq(Z, y, rcond=None)[0]; res = y - Z @ c
                cov = res @ res / (len(y) - 3) * np.linalg.inv(Z.T @ Z)
                rows.append(dict(stage=stage, horizon_s=h, ticker=tk, n_2024=len(b),
                                 flow_r2_oos=1 - ((yf - pf) ** 2).sum() / ((yf - yf.mean()) ** 2).sum(),
                                 bps_per_predicted_packet=c[1], t_predicted=c[1] / np.sqrt(cov[1, 1]),
                                 bps_per_surprise_packet=c[2], t_surprise=c[2] / np.sqrt(cov[2, 2])))
    per = pd.DataFrame(rows)
    summary = per.groupby(["stage", "horizon_s"]).apply(lambda x: pd.Series(dict(
        names=len(x), median_flow_r2_oos=x.flow_r2_oos.median(),
        median_bps_per_predicted_packet=x.bps_per_predicted_packet.median(),
        median_bps_per_surprise_packet=x.bps_per_surprise_packet.median(),
        names_t_predicted_gt2=int((x.t_predicted > 2).sum()), names_t_surprise_gt2=int((x.t_surprise > 2).sum()),
        names_predicted_below_surprise=int((x.bps_per_predicted_packet < x.bps_per_surprise_packet).sum()))),
        include_groups=False).reset_index()
    return per, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root); spec = json.loads((root / "design.json").read_text())
    out = root / "posthoc"; out.mkdir(exist_ok=True)
    zf = zero_forecast(root); zf.to_csv(out / "zero_forecast.csv", index=False)
    ls = loss_shares(root); ls.to_csv(out / "loss_shares.csv", index=False)
    data = load_rows(root, spec)
    nm = normalized(data, spec); nm.to_csv(out / "normalized_contrasts.csv", index=False)
    dc = deciles(data); dc.to_csv(out / "return_deciles.csv", index=False)
    per, sm = surprise(data); per.to_csv(out / "surprise_pricing_by_name.csv", index=False)
    sm.to_csv(out / "surprise_pricing_summary.csv", index=False)
    here = Path(__file__).resolve()
    ret = zf[zf.target.isin(["return_60s", "return_300s", "wait_cost_60s"])]
    model_cells = ret[["r2_vs_zero_" + m for m in MODELS]].to_numpy()
    result = dict(
        scope="post-hoc exploratory diagnostics of burst-information-v1; not a confirmation test",
        source_sha256={Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                       [str(here), str(here.parent / "evaluate_burst_information.py"),
                        str(here.parent / "burst_information_extract.py")]},
        return_and_wait_model_cells=int(model_cells.size),
        return_and_wait_model_cells_beating_zero=int((model_cells > 0).sum()),
        primary_heldout_top2=ls[(ls.stage == "third") & (ls.target == "flow_300s") & (ls.cohort == "heldout")]
        .to_dict(orient="records"),
        decile_cells=int(len(dc)), decile_cells_net_one_way_negative=int((dc.net_one_way < 0).sum()),
        surprise_pricing=sm.to_dict(orient="records"))
    (out / "diagnostics.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "surprise_pricing"}, indent=1, default=float))


if __name__ == "__main__":
    main()
