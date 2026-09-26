#!/usr/bin/env python3
"""Aggregate program-evidence-v1 modules C (campaigns), F (markouts), G (synchrony), H (passive)."""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

BOOT = 1000
ROOT = Path(__file__).resolve().parents[1]


def ci(b):
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if len(b) else None


def nw_t(x, lags):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    m = x.mean(); e = x - m
    v = e @ e / n
    for l in range(1, min(lags, n - 1) + 1):
        v += 2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / n
    return float(m), float(m / np.sqrt(v / n)) if v > 0 else np.nan


# ----------------------------------------------------------------------------- C
def module_c(pattern, rng):
    frames = []
    for p in sorted(glob.glob(pattern)):
        try:
            f = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            continue
        if len(f):
            frames.append(f)
    d = pd.concat(frames, ignore_index=True)
    d = d[np.isfinite(d.m_pre) & (d.m_pre > 0)].copy()
    d["impact"] = d.side * (d.m_post - d.m_pre) / d.m_pre * 1e4
    for h in (60, 300, 900, 1800, 3600):
        d["impact_%d" % h] = d.side * (d["m_%d" % h] - d.m_pre) / d.m_pre * 1e4
    d["pretrend"] = d.side * (d.m_pre - d.m_before300) / d.m_before300 * 1e4
    d["x_q"] = d.q / d.v_day
    d["x_same"] = d.v_same / d.v_day
    d["duration"] = d.t1 - d.t0
    tot = d.same_after_v + d.same_before_v
    d["flow_drop"] = np.where(tot > 0, (d.same_after_v - d.same_before_v) / tot, np.nan)
    camp = d[d.kind == "campaign"]; plac = d[d.kind == "placebo"]
    names = camp.ticker.unique()
    out = dict(campaigns=int(len(camp)), placebos=int(len(plac)), names=int(len(names)),
               median_children=float(camp.n.median()), median_duration_s=float(camp.duration.median()),
               median_q_over_daily_volume=float(camp.x_q.median()),
               share_buy=float((camp.side > 0).mean()))

    def exponent(frame, xcol):
        f = frame[np.isfinite(frame.impact) & np.isfinite(frame.sigma_day_bps) & (frame.sigma_day_bps > 0) & (frame[xcol] > 0)]
        if len(f) < 200:
            return np.nan, None
        y = f.impact / f.sigma_day_bps
        bins = pd.qcut(np.log(f[xcol]), 10, labels=False, duplicates="drop")
        g = pd.DataFrame(dict(x=f[xcol], y=y, b=bins)).groupby("b").mean()
        g = g[g.y > 0]
        if len(g) < 4:
            return np.nan, g
        slope = np.polyfit(np.log(g.x), np.log(g.y), 1)[0]
        return float(slope), g

    def by_name_boot(frame, fn):
        groups = {k: v for k, v in frame.groupby("ticker")}
        keys = list(groups)
        vals = []
        for _ in range(BOOT):
            pick = rng.integers(0, len(keys), len(keys))
            vals.append(fn(pd.concat([groups[keys[i]] for i in pick], ignore_index=True)))
        return ci(vals)

    for xcol in ("x_q", "x_same"):
        slope, g = exponent(camp, xcol)
        out["C1_exponent_" + xcol] = dict(value=slope, ci95=by_name_boot(camp, lambda f: exponent(f, xcol)[0]),
                                           bins=None if g is None else g.reset_index().round(6).to_dict(orient="records"),
                                           in_pre_declared_range=bool(0.3 <= slope <= 0.7) if np.isfinite(slope) else None)

    def ratio(frame, h):
        f = frame[np.isfinite(frame.impact) & np.isfinite(frame["impact_%d" % h])]
        return float(f["impact_%d" % h].mean() / f.impact.mean()) if len(f) and f.impact.mean() != 0 else np.nan
    out["C2_reversion"] = {}
    for h in (60, 300, 900, 1800, 3600):
        f = camp[np.isfinite(camp.impact) & np.isfinite(camp["impact_%d" % h])]
        out["C2_reversion"]["%ds" % h] = dict(
            mean_impact_end=float(f.impact.mean()), mean_impact_h=float(f["impact_%d" % h].mean()),
            ratio=ratio(camp, h), ratio_ci95=by_name_boot(camp, lambda fr, h=h: ratio(fr, h)), n=int(len(f)))
    r30 = out["C2_reversion"]["1800s"]["ratio"]
    out["C2_in_pre_declared_range"] = bool(0.4 <= r30 <= 0.9) if np.isfinite(r30) else None
    out["mean_pretrend_bps"] = float(camp.pretrend.mean())
    # C3: campaigns vs placebos in the same duration x same-side-share cells (quintiles on pooled rows)
    both = pd.concat([camp, plac])
    both = both[np.isfinite(both.impact) & np.isfinite(both.impact_1800)]
    both["cell"] = (pd.qcut(both.duration.rank(method="first"), 5, labels=False) * 5
                    + pd.qcut(both.x_same.rank(method="first"), 5, labels=False))
    rows = []
    for cell, g in both.groupby("cell"):
        c, p = g[g.kind == "campaign"], g[g.kind == "placebo"]
        if len(c) >= 20 and len(p) >= 20:
            rows.append(dict(cell=int(cell), n_c=len(c), n_p=len(p), impact_c=c.impact.mean(), impact_p=p.impact.mean(),
                             i1800_c=c.impact_1800.mean(), i1800_p=p.impact_1800.mean(),
                             flow_drop_c=c.flow_drop.mean(), flow_drop_p=p.flow_drop.mean(),
                             pretrend_c=c.pretrend.mean(), pretrend_p=p.pretrend.mean()))
    cells = pd.DataFrame(rows)
    if len(cells):
        w = cells.n_c / cells.n_c.sum()
        out["C3_matched_cells"] = dict(
            cells=int(len(cells)),
            impact_end_campaign_minus_placebo=float((w * (cells.impact_c - cells.impact_p)).sum()),
            impact_1800_campaign_minus_placebo=float((w * (cells.i1800_c - cells.i1800_p)).sum()),
            flow_drop_campaign_minus_placebo=float((w * (cells.flow_drop_c - cells.flow_drop_p)).sum()),
            pretrend_campaign_minus_placebo=float((w * (cells.pretrend_c - cells.pretrend_p)).sum()),
            table=cells.round(4).to_dict(orient="records"))
    summaries = []
    for p in sorted(glob.glob(pattern.replace(".csv.gz", ".csv.summary.json"))):
        summaries += json.loads(Path(p).read_text())
    s = pd.DataFrame(summaries)
    if len(s):
        exp = s.filter(like="expected_chance").sum().sum(); obs = s.filter(like="campaigns_side").sum().sum()
        out["chance_campaigns_expected_vs_observed"] = dict(expected=float(exp), observed=int(obs))
    return out


# ----------------------------------------------------------------------------- F
def module_f(pattern, rng, h_primary=60):
    names = []
    for p in sorted(glob.glob(pattern)):
        with np.load(p) as z:
            if len(z["dates"]):
                names.append(dict(ticker=str(z["ticker"]), dates=z["dates"], m=z["markouts"], vol=z["group_volume"],
                                  horizons=z["horizons"]))
    hz = list(names[0]["horizons"])
    labels = ["bottom", "middle", "program", "unscored", "no_burst"]
    out = dict(names=len(names), horizons=[int(h) for h in hz])
    vol = np.sum([n["vol"].sum(0) for n in names], axis=0)
    out["volume_share_by_group"] = {labels[i]: float(vol[i] / vol.sum()) for i in range(5)}

    def name_day_contrast(m, g1, g0, hi):
        s, c = m[0, :, :, hi], m[2, :, :, hi]
        ok = (c[g1] > 0) & (c[g0] > 0)
        if not ok.any():
            return np.nan
        return float(np.mean(s[g1][ok] / c[g1][ok] - s[g0][ok] / c[g0][ok]))

    for (g1, g0, lab) in ((2, 0, "program_minus_bottom"), (2, 4, "program_minus_no_burst"), (0, 4, "bottom_minus_no_burst")):
        res = {}
        for hi, h in enumerate(hz):
            per_date = {}
            per_name = []
            for n in names:
                vals = [name_day_contrast(n["m"][i], g1, g0, hi) for i in range(len(n["dates"]))]
                for d, v in zip(n["dates"], vals):
                    if np.isfinite(v):
                        per_date.setdefault(str(d), []).append(v)
                v = np.array(vals, float); v = v[np.isfinite(v)]
                if len(v):
                    per_name.append(v.mean())
            daily = [np.mean(per_date[d]) for d in sorted(per_date)]
            mean, t = nw_t(daily, 2)
            per_name = np.array(per_name)
            boots = [per_name[rng.integers(0, len(per_name), len(per_name))].mean() for _ in range(BOOT)]
            res["%ds" % h] = dict(mean_bps=mean, nw_t=t, days=len(daily), name_mean_bps=float(per_name.mean()),
                                  name_boot_ci95=ci(boots), names=int(len(per_name)))
        out[lab] = res
    # level of markouts by group (pooled, equal weight per name)
    lev = {}
    for hi, h in enumerate(hz):
        lev["%ds" % h] = {}
        for gi, g in enumerate(labels):
            per = []
            for n in names:
                s = n["m"][:, 0, gi, :, hi].sum(); c = n["m"][:, 2, gi, :, hi].sum()
                if c > 0:
                    per.append(s / c)
            lev["%ds" % h][g] = float(np.mean(per)) if per else None
    out["mean_markout_bps_by_group"] = lev
    t = out["program_minus_bottom"]["%ds" % h_primary]["nw_t"]
    out["F1_exploration_gate_abs_t_gt_3"] = bool(np.isfinite(t) and abs(t) > 3)
    return out


# ----------------------------------------------------------------------------- G
def etf_overlap(names, year):
    w = pd.read_csv(ROOT / "data" / "wrds" / ("etf_weights_%d.csv.gz" % year))
    piv = w.pivot_table(index="constituent_ticker", columns="composite_ticker", values="weight", aggfunc="sum").fillna(0)
    M = np.zeros((len(names), len(names)))
    idx = [piv.index.get_loc(n) if n in piv.index else -1 for n in names]
    W = piv.to_numpy()
    for a in range(len(names)):
        if idx[a] < 0:
            continue
        for b in range(a + 1, len(names)):
            if idx[b] < 0:
                continue
            M[a, b] = M[b, a] = np.minimum(W[idx[a]], W[idx[b]]).sum()
    return M


def sic2(names, year):
    n = pd.read_csv(ROOT / "data" / "wrds" / "crsp_names_fingerprint.csv.gz")
    n["namedt"] = pd.to_datetime(n.namedt); n["nameendt"] = pd.to_datetime(n.nameendt)
    ref = pd.Timestamp("%d-06-30" % year)
    n = n[(n.namedt <= ref) & (n.nameendt >= ref)]
    m = n.groupby("ticker").siccd.first()
    return np.array([int(m[x]) // 100 if x in m.index and np.isfinite(m[x]) and m[x] > 0 else -1 for x in names])


def module_g(pattern, year, rng):
    files = sorted(glob.glob(pattern))
    pooled = None; pair = None; names = None; npk = None
    for p in files:
        with np.load(p) as z:
            nm = list(z["names"])
            if names is None:
                names = sorted(set(nm)); pos = {k: i for i, k in enumerate(names)}
            # dates can differ in available names: re-index into the union seen on the first file
            for x in nm:
                if x not in pos:
                    pos[x] = len(names); names.append(x)
    N = len(names); pos = {k: i for i, k in enumerate(names)}
    pair = np.zeros((7, 2, 2, N, N)); npk = np.zeros(N); nprog = np.zeros(N)
    for p in files:
        with np.load(p) as z:
            ix = np.array([pos[x] for x in z["names"]])
            pooled = z["pooled"].astype(float) if pooled is None else pooled + z["pooled"]
            pair[np.ix_(range(7), range(2), range(2), ix, ix)] += z["pair_1ms"]
            npk[ix] += z["n_packets"]; nprog[ix] += z["n_program"]
    deltas = [1e-4, 1e-3, 1e-2]
    out = dict(dates=len(files), names=N, program_packet_share=float(nprog.sum() / npk.sum()))
    for di, dlt in enumerate(deltas):
        c = pooled[di]                                   # [offset, rel, prog_i, prog_j]
        tot = c.sum((2, 3))
        out["sync_ratio_%gms" % (dlt * 1e3)] = dict(
            same_side=float(tot[0, 0] / tot[1:, 0].mean()), opposite_side=float(tot[0, 1] / tot[1:, 1].mean()),
            same_side_program_i=float(c[0, 0, 1].sum() / c[1:, 0, 1].sum(-1).mean()),
            same_side_other_i=float(c[0, 0, 0].sum() / c[1:, 0, 0].sum(-1).mean()),
            same_side_both_program=float(c[0, 0, 1, 1] / c[1:, 0, 1, 1].mean()),
            same_side_neither_program=float(c[0, 0, 0, 0] / c[1:, 0, 0, 0].mean()))

    def g1_stat(weights):
        # weights: per-name multiplicity; pair weight = w_i * w_j
        W = np.outer(weights, weights)
        res = []
        for prog in (1, 0):
            obs = (pair[0, 0, prog] * W).sum()
            null = np.mean([(pair[o, 0, prog] * W).sum() for o in range(1, 7)])
            res.append(obs / null if null > 0 else np.nan)
        return res[0] - res[1], res
    diff, (sr_p, sr_o) = g1_stat(np.ones(N))
    boots = []
    for _ in range(BOOT):
        wts = np.bincount(rng.integers(0, N, N), minlength=N).astype(float)
        boots.append(g1_stat(wts)[0])
    out["G1"] = dict(sync_ratio_program=float(sr_p), sync_ratio_other=float(sr_o), difference=float(diff), ci95=ci(boots),
                     gate=bool(ci(boots)[0] > 0))
    # G2: pair-level regression
    ov = etf_overlap(names, year); sc = sic2(names, year)
    rows = []
    both = pair[:, 0].sum(1)                               # [offset, i, j], same side, all packets i
    for a in range(N):
        for b in range(a + 1, N):
            obs = both[0, a, b] + both[0, b, a]
            null = np.mean([both[o, a, b] + both[o, b, a] for o in range(1, 7)])
            if null >= 5:
                rows.append((a, b, np.log((obs + 0.5) / (null + 0.5)), ov[a, b], float(sc[a] == sc[b] and sc[a] >= 0),
                             np.log(npk[a] * npk[b])))
    R = np.array(rows)
    if len(R) > 50:
        X = np.column_stack([np.ones(len(R)), R[:, 3], R[:, 4], R[:, 5]]); y = R[:, 2]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        bb = []
        for _ in range(BOOT):
            wts = np.bincount(rng.integers(0, N, N), minlength=N).astype(float)
            w = wts[R[:, 0].astype(int)] * wts[R[:, 1].astype(int)]
            if w.sum() == 0:
                continue
            sw = np.sqrt(w)
            bb.append(np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0])
        bb = np.array(bb)
        se = bb.std(0)
        out["G2"] = dict(pairs=int(len(R)), coef=dict(zip(["const", "etf_overlap", "same_sic2", "log_activity"], beta.tolist())),
                         boot_se=dict(zip(["const", "etf_overlap", "same_sic2", "log_activity"], se.tolist())),
                         t_etf_overlap=float(beta[1] / se[1]), t_same_sic2=float(beta[2] / se[2]),
                         mean_overlap=float(R[:, 3].mean()), share_same_sic2=float(R[:, 4].mean()))
    for etf in ("SPY", "QQQ", "TQQQ", "SQQQ", "SMH"):
        if etf in pos:
            e = pos[etf]
            obs = pair[0, 0, :, e, :].sum() + pair[0, 0, :, :, e].sum()
            null = np.mean([pair[o, 0, :, e, :].sum() + pair[o, 0, :, :, e].sum() for o in range(1, 7)])
            out["sync_with_%s_same_side_1ms" % etf] = float(obs / null) if null else None
    return out


# ----------------------------------------------------------------------------- H
def module_h(pattern, rng):
    per = []
    for p in sorted(glob.glob(pattern)):
        with np.load(p) as z:
            if len(z["dates"]) and len(z["pairs"]):
                per.append({k: z[k] for k in z.files})
    edges = per[0]["edges"]
    lagmask = (edges[:-1] >= 0.5) & (edges[1:] <= 10)

    def ratio(w_pairs, w_match, c_pairs, c_match):
        rate = np.where(c_pairs > 0, c_match / np.maximum(c_pairs, 1), np.nan)
        exp = np.nansum((w_pairs * rate)[lagmask]); obs = w_match[lagmask].sum()
        return (obs / exp if exp > 0 else np.nan), exp

    h1, h2s, h2o, h2c = [], [], [], []
    for z in per:
        w = z["h1_within"].sum(0); c = z["h1_cross"].sum(0)
        r, e = ratio(w[0], w[1], c[0], c[1])
        if e >= 20:
            h1.append(r)
        w2 = z["h2_within"].sum(0); c2 = z["h2_cross"].sum(0)          # [rel, pairs/matches, bin]
        rs, es = ratio(w2[0, 0], w2[0, 1], c2[0, 0], c2[0, 1]); ro, eo = ratio(w2[1, 0], w2[1, 1], c2[1, 0], c2[1, 1])
        if es >= 20:
            h2s.append(rs)
        if es >= 20 and eo >= 20:
            h2o.append(ro); h2c.append(rs / ro)

    def med(v):
        v = np.asarray(v, float)
        b = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(BOOT)]
        return dict(median=float(np.median(v)), ci95=ci(b), names=int(len(v)), share_above_1=float(np.mean(v > 1)))
    out = dict(names=len(per), H1=med(h1), H2_same_economic_side=med(h2s), H2_opposite_side_control=med(h2o),
               H2_same_over_opposite=med(h2c))
    out["gates"] = dict(H1=bool(out["H1"]["ci95"][0] > 1), H2=bool(out["H2_same_economic_side"]["ci95"][0] > 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=["explore_2024", "confirm_2021"])
    ap.add_argument("--modules", default="CFGH")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base = ROOT / "results" / "program_evidence_v1" / args.group
    year = int(args.group[-4:])
    rng = np.random.default_rng(20260914)
    res = dict(group=args.group)
    if "C" in args.modules:
        res["C"] = module_c(str(base / "campaigns" / "*.csv.gz"), rng)
    if "F" in args.modules:
        res["F"] = module_f(str(base / "markouts" / "*.npz"), rng)
    if "G" in args.modules:
        res["G"] = module_g(str(base / "sync" / "*.npz"), year, rng)
    if "H" in args.modules:
        res["H"] = module_h(str(base / "passive" / "stats" / "*.npz"), rng)
    Path(args.out).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps({k: (v.get("gates") if isinstance(v, dict) else v) for k, v in res.items()}, indent=1, default=str))


if __name__ == "__main__":
    main()
