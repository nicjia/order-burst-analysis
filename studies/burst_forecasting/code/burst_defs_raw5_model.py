#!/usr/bin/env python3
"""burst-defs-raw-v5 stage 2: FORECASTING ONLY on the larger universe (676 stocks, 144 sampled dates 2022-25).

Train on the train names in 2022-23 (the 60 v2-v4 train names plus a fixed hash-half of the new names); test on the
other names in 2024 and in 2025, and on the train names in 2025 (out of time only). Gradient boosting (depth 3, 200
iterations) as in v2-v4; daily rank IC, Newey-West over dates; paired gains.

Parts (python3 burst_defs_raw5_model.py <part> ...):
  placebo   Every event type (bursts, pairs, isolated / large isolated / any single orders, random times): IC of a
            burst-blind model (GENERIC + DEPTH) and of the same plus the event's own features (EVENT), for the mid, the
            far-side quote, the near-side quote and the microprice at +1 s ... close, and the direction of the first
            change of each. Mean signed moves by type (continuation).                  [A29, A11, A30]
  book      Nested baselines on the main types: GENERIC; + DEPTH (multi-level book rebuilt from the messages); +
            EVENT; + BHIST (bursts in the last 1 / 5 / 30 min); and the v4 feature set.  [deeper-book check]
  dose      Dose-response by the number of orders (1, 2, 3, 4, 5-9, 10+) and size-matched comparisons of bursts vs
            large isolated orders.
  robust    early5 and run0.1: IC by year, spread in ticks (1 vs 2+), tick-to-price, time of day, per stock, top-10
            stocks dropped, stock-block bootstrap CIs; the v4-leak check (whole-day OFI scale), 2025 by train / test
            names; pre-burst drift (A10); share of the move before the burst (A1); raw vs market-excess (A35).
            [A1, A10, A31, A32, A33, A35, A36, A37]
  cross     Market-wide vs idiosyncratic bursts: other stocks' and SPY / QQQ bursts in the last 1 / 5 / 30 s. [A35]
  panel     The v4 5-minute cross-sectional panel on the larger universe.
  check     The rebuilt-book level-1 checks from the extraction.                     [A39]
"""
import sys as _sys, pathlib as _pl  # shared project modules live in src_py/
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "src_py"))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import glob, json, zlib
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import p4_analyze as PA
import burst_defs_raw2_model as M
import burst_defs_raw4_model as V4

D5 = M.D / "burst_defs_raw5"
WIN = (1, 5, 10, 30, 60, 300)
KT = (1, 2, 3, 5, 10)
TYPES = [("early5", "real"), ("run0.01", "real"), ("run0.1", "real"), ("run0.5", "real"), ("levelclear", "real"),
         ("cancel", "real"), ("hidden", "real"), ("run60", "real"), ("pair", "real"), ("iso", "single"),
         ("iso_large", "single"), ("any", "single"), ("run0.1", "pseudo")]
MAIN = [("early5", "real"), ("run0.1", "real"), ("run0.5", "real"), ("levelclear", "real"), ("pair", "real"),
        ("iso", "single"), ("iso_large", "single"), ("any", "single"), ("run0.1", "pseudo")]
FAM = {"mid": "r", "far quote": "f", "near quote": "n", "microprice": "u"}
HOR = ["1", "10", "60", "300", "1800", "_close"]
HX = {"1800", "_close"}                                           # market-excess horizons
GENERIC = (["since_open", "tod", "spread_dec", "spread_ticks", "qimb_dec", "micro_gap"]
           + ["g_%s%d" % (k, W) for W in WIN for k in ("ret", "tfi", "nt", "ofi")]
           + ["g_volr10", "g_volr60", "g_volr300", "g_lrv60", "g_lrv300"])
DEPTH = ["d_imb%d_dec" % k for k in KT] + ["d_lfar1_dec", "d_lnear1_dec", "d_lfar5_dec", "d_lnear5_dec"]
DEPTH_PRE = ["d_imb%d_pre" % k for k in KT] + ["d_lfar1_pre", "d_lnear1_pre", "d_lfar5_pre", "d_lnear5_pre"]
EVENT = (["move_during", "pre60", "pre30m", "opp_consumed", "spread_change", "spread_ticks_b", "log_n", "log_dur",
          "log_q_adv", "mode_share", "nonround", "size_cv", "size_to_depth", "iat_cv", "iat_med", "phase_R", "walk_share",
          "hid_share", "imb_first", "imb_last", "qofi_pre60", "qofi_during", "tfi_pre60", "canc_opp", "canc_same"] + DEPTH_PRE)
BHIST = ["bh_%s%d" % (k, W) for W in (60, 300, 1800) for k in ("same", "opp", "flow")]
V4SET = V4.CTRL + V4.BOOK + V4.BURST
V4SET_LEAK = [c + "_v4" if c in ("qofi_pre60", "qofi_during") else c for c in V4SET]
TESTS = ("2024 test names", "2025 test names", "2025 train names")
ETF = (84398, 86755)                                             # SPY, QQQ: used only for the cross-stock features
CAP_TRAIN = 150000                                              # training rows per fit (random subsample)


def target_cols(fam, h):
    return "%s%s%s" % (FAM[fam], h, "_x" if h in HX else "")


def train_names():
    old = set(pd.read_csv(M.ROOT / "results" / "p4_revisit_v1" / "jobs" / "DEFS_RAW_train_names.txt", header=None)[0])
    raw = pd.read_csv(M.ROOT / "results" / "p4_revisit_v1" / "jobs" / "DEFS_RAW.txt", sep=" ", header=None, names=["date", "tk", "permno"])
    old_test = set(raw.permno) - old
    v5 = pd.read_csv(M.ROOT / "results" / "p4_revisit_v1" / "jobs" / "DEFS_V5.txt", sep=" ", header=None, names=["date", "tk", "permno"])
    new = sorted(set(v5.permno) - old - old_test)
    half = {p for p in new if zlib.crc32(str(p).encode()) % 2 == 0}
    return old | half


def load(dn, kind, cols=None):
    fs = sorted(glob.glob(str(D5 / ("V5_events_%s_%s_part*.csv.gz" % (dn, kind)))))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f, dtype={"date": str, "defn": "category", "kind": "category", "ticker": "category"},
                               usecols=cols) for f in fs], ignore_index=True)
    num = d.select_dtypes("float64").columns
    d[num] = d[num].astype(np.float32)
    nd = pd.read_csv(M.ROOT / "results" / "p4_revisit_v1" / "agg" / "TEST" / "nameday_TEST.csv.gz", dtype={"date": str},
                     usecols=["permno", "date", "family", "adv20"])
    d = d.merge(nd[nd.family == "T"][["permno", "date", "adv20"]], on=["permno", "date"], how="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        d["log_n"] = np.log(d.n_used.where(d.n_used > 0)); d["log_dur"] = np.log1p(d.dur)
        d["log_q_adv"] = np.log(d.vol_used.where(d.vol_used > 0) / d.adv20)
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d[(d.spread_dec > 0) & (d.spread_dec < 500) & ~d.permno.isin(ETF)]
    for w in ("m", "f"):
        if "wait_%s" % w in d:
            d["lwait_%s" % w] = np.log(d["wait_%s" % w].clip(lower=0.001))
    dix, I = V4.market()
    k0 = V4.minute_mark(d.t_dec.to_numpy(float)); dates = d.date.to_numpy(); s = d.side.to_numpy(float)
    m0 = V4.mk_at(dates, k0, dix, I)
    m30 = (V4.mk_at(dates, V4.minute_mark(d.t_dec.to_numpy(float) + 1800), dix, I) - m0) * 1e4
    mcl = (V4.mk_at(dates, 389, dix, I) - m0) * 1e4
    for f in FAM.values():
        for h in HOR:
            c = "%s%s" % (f, h)
            if c in d:
                d.loc[d[c].abs() > 1000, c] = np.nan
        if "%s1800" % f in d:
            d["%s1800_x" % f] = d["%s1800" % f] - s * m30
        if "%s_close" % f in d:
            d["%s_close_x" % f] = d["%s_close" % f] - s * mcl
    return d


def split(d, tr_names):
    y = d.date.str[:4]; tr = d.permno.isin(tr_names)
    return d[tr & y.isin(["2022", "2023"])], {TESTS[0]: d[~tr & (y == "2024")], TESTS[1]: d[~tr & (y == "2025")],
                                              TESTS[2]: d[tr & (y == "2025")]}


def fit_eval(tr, tests, feats, y, base_feats=None, auc=False):
    """IC (and gain over base_feats) of gradient boosting on every test set; returns rows."""
    t = tr[tr[y].notna()]
    if len(t) < 3000:
        return []
    if len(t) > CAP_TRAIN:
        t = t.sample(CAP_TRAIN, random_state=0)
    feats = [c for c in feats if c in t]
    mf, lo, hi = M.hgb(t, feats, y)
    mb = M.hgb(t, [c for c in base_feats if c in t], y)[0] if base_feats is not None else None
    out = []
    for tn, e in tests.items():
        e = e[e[y].notna()]
        if len(e) < 1000:
            continue
        yy = e[y].clip(lo, hi).to_numpy(float); dates = e.date.to_numpy()
        pf = mf.predict(e[feats].to_numpy(float))
        sf, icf, sp = M.daily_ic(dates, pf, yy)
        row = dict(test=tn, n=int(len(e)), ic=icf["mean"], t=icf["t"], decile_spread=sp["mean"],
                   hit=float(np.mean(np.sign(pf[yy != 0]) == np.sign(yy[yy != 0]))) if (yy != 0).any() else np.nan)
        if auc and (yy != 0).any():
            row["auc"] = float(roc_auc_score(yy[yy != 0] > 0, pf[yy != 0]))
        if mb is not None:
            bf = [c for c in base_feats if c in e]
            pb = mb.predict(e[bf].to_numpy(float))
            sb, icb, _ = M.daily_ic(dates, pb, yy)
            g = PA.nw_t((sf - sb).to_numpy())
            row.update(ic_base=icb["mean"], gain=g["mean"], t_gain=g["t"])
            if auc and (yy != 0).any():
                row["auc_base"] = float(roc_auc_score(yy[yy != 0] > 0, pb[yy != 0]))
        out.append(row)
    return out


def part_placebo(trn):
    rows, cont = [], []
    for dn, kind in TYPES:
        d = load(dn, kind)
        if d is None:
            continue
        tr, tests = split(d, trn)
        fams = list(FAM) if (dn, kind) in MAIN else ["mid", "far quote"]
        for fam in fams:
            hs = HOR if fam != "near quote" else ["1", "10", "60"]
            for h in hs:
                y = target_cols(fam, h)
                for r in fit_eval(tr, tests, GENERIC + DEPTH + EVENT, y, GENERIC + DEPTH):
                    rows.append(dict(defn=dn, kind=kind, target=fam, horizon=h, **r))
        for fc in (("first_m", "mid"), ("first_f", "far quote"), ("first_n", "near quote")):
            for r in fit_eval(tr, tests, GENERIC + DEPTH + EVENT, fc[0], GENERIC + DEPTH, auc=True):
                rows.append(dict(defn=dn, kind=kind, target=fc[1], horizon="first change", **r))
        if (dn, kind) in MAIN:
            for wc in (("lwait_m", "mid"), ("lwait_f", "far quote")):          # WHEN the price next changes
                for r in fit_eval(tr, tests, GENERIC + DEPTH + EVENT, wc[0], GENERIC + DEPTH):
                    rows.append(dict(defn=dn, kind=kind, target=wc[1], horizon="log time to next change", **r))
        allt = pd.concat(tests.values())
        for fam, f in FAM.items():
            cont.append(dict(defn=dn, kind=kind, target=fam, n=int(len(allt)), move_during=float(allt.move_during.mean()),
                             **{"+%s" % h: float(allt["%s%s" % (f, h)].mean()) for h in ("1", "10", "60", "300")}))
        print("placebo", dn, kind, "done", flush=True)
        pd.DataFrame(rows).to_csv(D5 / "v5_placebo.csv", index=False); pd.DataFrame(cont).to_csv(D5 / "v5_continuation.csv", index=False)
    return pd.DataFrame(rows), pd.DataFrame(cont)


def part_book(trn):
    sets = [("generic", GENERIC), ("generic + depth", GENERIC + DEPTH), ("+ event", GENERIC + DEPTH + EVENT),
            ("+ event + history", GENERIC + DEPTH + EVENT + BHIST), ("v4 set", V4SET), ("v4 set, whole-day OFI scale", V4SET_LEAK)]
    rows = []
    for dn, kind in [("early5", "real"), ("run0.1", "real"), ("levelclear", "real"), ("iso", "single"), ("run0.1", "pseudo")]:
        d = load(dn, kind)
        if d is None:
            continue
        tr, tests = split(d, trn)
        for fam in ("mid", "far quote"):
            for h in ("1", "10", "60", "300", "1800", "_close"):
                y = target_cols(fam, h)
                prev = None
                for nm, cols in sets:
                    for r in fit_eval(tr, tests, cols, y, prev if nm not in ("v4 set", "v4 set, whole-day OFI scale") else None):
                        rows.append(dict(defn=dn, kind=kind, target=fam, horizon=h, model=nm, **r))
                    if nm in ("generic", "generic + depth", "+ event"):
                        prev = cols
        print("book", dn, kind, "done", flush=True)
        pd.DataFrame(rows).to_csv(D5 / "v5_book.csv", index=False)
    return pd.DataFrame(rows)


def part_dose(trn):
    """Dose-response in the number of orders at the same confirmation lag (+0.6 s): isolated (1), pairs (2), run0.5
    bursts by n; and bursts vs large isolated orders matched on volume / ADV."""
    parts = []
    for dn, kind in (("iso", "single"), ("pair", "real"), ("run0.5", "real"), ("iso_large", "single")):
        d = load(dn, kind)
        if d is not None:
            parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["norders"] = pd.cut(d.n_used, [0, 1, 2, 3, 4, 9, 1e9], labels=["1", "2", "3", "4", "5-9", "10+"])
    tr, tests = split(d, trn)
    rows = []
    for y in ("r10", "r60", "r300", "f10", "f60", "f300", "r1800_x"):
        t = tr[tr[y].notna()]
        feats = [c for c in GENERIC + DEPTH + EVENT if c in t]
        mf, lo, hi = M.hgb(t, feats, y)                                # one model for all sizes
        for tn, e in tests.items():
            e = e[e[y].notna()].copy()
            e["p"] = mf.predict(e[feats].to_numpy(float)); e["yy"] = e[y].clip(lo, hi)
            for (dn, k), g in e.groupby(["defn", "norders"], observed=True):
                if len(g) < 2000:
                    continue
                _, ic, _ = M.daily_ic(g.date.to_numpy(), g.p.to_numpy(), g.yy.to_numpy(), minn=10)
                rows.append(dict(target=y, test=tn, defn=dn, norders=str(k), n=len(g), ic=ic["mean"], t=ic["t"],
                                 mean_move=float(g[y].mean()), mean_push=float(g.move_during.mean())))
            # size-matched: run0.5 bursts vs large isolated orders in the same log(volume / ADV) quintile
            m = e[e.defn.isin(["run0.5", "iso_large"])].dropna(subset=["log_q_adv"]).copy()
            if len(m) < 5000:
                continue
            m["qbin"] = pd.qcut(m.log_q_adv.rank(method="first"), 5, labels=False)      # quintiles of the pooled sizes
            for (dn, qb), g in m.groupby(["defn", "qbin"]):
                if len(g) < 1000:
                    continue
                _, ic, _ = M.daily_ic(g.date.to_numpy(), g.p.to_numpy(), g.yy.to_numpy(), minn=10)
                rows.append(dict(target=y, test=tn, defn=dn, norders="size quintile %d" % (qb + 1), n=len(g), ic=ic["mean"],
                                 t=ic["t"], mean_move=float(g[y].mean()), mean_push=float(g.move_during.mean())))
    R = pd.DataFrame(rows); R.to_csv(D5 / "v5_dose.csv", index=False)
    return R


def stock_boot(dates, permno, p, y, B=200, seed=0):
    """Pooled IC with a stock-block bootstrap CI (resampling stocks with replacement)."""
    df = pd.DataFrame(dict(d=dates, s=permno, p=p, y=y)).dropna()
    df["rp"] = df.groupby("d").p.rank(pct=True); df["ry"] = df.groupby("d").y.rank(pct=True)
    g = df.groupby("s")
    stat = g.apply(lambda x: pd.Series(dict(a=((x.rp - .5) * (x.ry - .5)).sum(), b=((x.rp - .5) ** 2).sum(),
                                            c=((x.ry - .5) ** 2).sum())), include_groups=False)
    rng = np.random.default_rng(seed); A = stat.to_numpy()
    est = A[:, 0].sum() / np.sqrt(A[:, 1].sum() * A[:, 2].sum())
    bs = []
    for _ in range(B):
        k = rng.integers(len(A), size=len(A)); S = A[k].sum(0)
        bs.append(S[0] / np.sqrt(S[1] * S[2]))
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def part_robust(trn):
    rows = []
    for dn, kind in (("early5", "real"), ("run0.1", "real")):
        d = load(dn, kind)
        tr, tests = split(d, trn)
        feats = [c for c in GENERIC + DEPTH + EVENT if c in d]
        for y in ("r10", "r60", "r300", "f60", "r1800_x", "r_close_x", "r1800", "r_close"):
            t = tr[tr[y].notna()]; mf, lo, hi = M.hgb(t, feats, y)
            no_pre = [c for c in feats if c not in ("pre60", "pre30m")]; mnp = M.hgb(t, no_pre, y)[0]
            ev = pd.concat([e.assign(test=tn) for tn, e in tests.items()])
            ev = ev[ev[y].notna()].copy()
            ev["p"] = mf.predict(ev[feats].to_numpy(float)); ev["p_nopre"] = mnp.predict(ev[no_pre].to_numpy(float))
            ev["yy"] = ev[y].clip(lo, hi)
            ev["year"] = ev.date.str[:4]
            ev["ticks"] = np.where(ev.spread_ticks <= 1, "1 tick", "2+ ticks")
            ev["tod_b"] = pd.cut(ev.tod, [-1, 1 / 13, 4 / 13, 9 / 13, 12 / 13, 2], labels=["9:30-10:00", "10:00-11:30", "11:30-14:00", "14:00-15:30", "15:30-16:00"])
            ev["tickpx"] = pd.qcut((ev.spread_dec / ev.spread_ticks.where(ev.spread_ticks > 0)).rank(method="first"), 3,
                                   labels=["tick small vs price", "middle", "tick large vs price"])
            base = dict(defn=dn, target=y)
            for tn, g in ev.groupby("test"):
                _, ic, _ = M.daily_ic(g.date.to_numpy(), g.p.to_numpy(), g.yy.to_numpy())
                rows.append(dict(base, split="all", bucket=tn, ic=ic["mean"], t=ic["t"], n=len(g)))
                _, ic2, _ = M.daily_ic(g.date.to_numpy(), g.p_nopre.to_numpy(), g.yy.to_numpy())
                rows.append(dict(base, split="A10: without the pre-burst drift features", bucket=tn, ic=ic2["mean"], t=ic2["t"], n=len(g)))
                est, lo_, hi_ = stock_boot(g.date.to_numpy(), g.permno.to_numpy(), g.p.to_numpy(), g.yy.to_numpy())
                rows.append(dict(base, split="A31: pooled IC, stock-block bootstrap 95% CI", bucket=tn, ic=est, ci_lo=lo_, ci_hi=hi_, n=len(g)))
                per = g.groupby("permno")[["p", "yy"]].apply(lambda x: x.p.rank().corr(x.yy.rank()) if len(x) > 100 else np.nan).dropna()
                rows.append(dict(base, split="A32: share of stocks with IC > 0", bucket=tn, ic=float((per > 0).mean()), n=len(per)))
                top = g.permno.value_counts().index[:10]
                gg = g[~g.permno.isin(top)]
                _, ic3, _ = M.daily_ic(gg.date.to_numpy(), gg.p.to_numpy(), gg.yy.to_numpy())
                rows.append(dict(base, split="A32: without the 10 most active stocks", bucket=tn, ic=ic3["mean"], t=ic3["t"], n=len(gg)))
                for col, lab in (("year", "A33: year"), ("ticks", "A37: spread at the decision"), ("tickpx", "A37: tick size relative to price"),
                                 ("tod_b", "A36: time of day")):
                    for k, h in g.groupby(col, observed=True):
                        if len(h) < 2000:
                            continue
                        _, ic4, _ = M.daily_ic(h.date.to_numpy(), h.p.to_numpy(), h.yy.to_numpy(), minn=10)
                        rows.append(dict(base, split=lab, bucket="%s | %s" % (tn, k), ic=ic4["mean"], t=ic4["t"], n=len(h)))
            print("robust", dn, y, "done", flush=True)
        # A1: signed mid move before vs after the burst began (event window between), vs random times
        pseudo = load("run0.1", "pseudo")
        for nm, x in (("%s bursts" % dn, d), ("random times", pseudo)):
            x = x[x.date.str[:4].isin(["2024", "2025"])]
            pre, during, post = x.pre60.mean(), x.move_during.mean(), x.r60.mean()
            tot = pre + during + post
            rows.append(dict(defn=dn, target="signed mid", split="A1: share of the t_b-60s -> t_dec+60s move before t_b", bucket=nm,
                             ic=float(pre / tot) if tot else np.nan, pre=float(pre), during=float(during), post=float(post), n=len(x)))
        pd.DataFrame(rows).to_csv(D5 / "v5_robust.csv", index=False)
    return pd.DataFrame(rows)


def cross_features(d):
    """For each event: other stocks' bursts of 5+ orders decided in [td - W, td) (same and opposite side, per 100 stocks
    trading that day) and SPY / QQQ bursts in the same windows. Uses only bursts decided before the decision."""
    fs = sorted(glob.glob(str(D5 / "V5_blist_part*.csv.gz")))
    bl = pd.concat([pd.read_csv(f, dtype={"date": str}) for f in fs], ignore_index=True)
    stk, etf = bl[~bl.permno.isin(ETF)], bl[bl.permno.isin(ETF)]
    per100 = 100.0 / stk.groupby("date").permno.nunique()
    out = {}
    for W in (1, 5, 30):
        for k in ("same", "opp"):
            out["x_%s%d" % (k, W)] = np.zeros(len(d)); out["etf_%s%d" % (k, W)] = np.zeros(len(d))
    d = d.reset_index(drop=True)
    td = d.t_dec.to_numpy(float); side = d.side.to_numpy(float)
    by_date = {"x": dict(tuple(stk.groupby("date"))), "etf": dict(tuple(etf.groupby("date")))}
    empty = stk.iloc[:0]
    for date, idx in d.groupby("date").indices.items():
        for prefix in ("x", "etf"):
            b = by_date[prefix].get(date, empty)
            for sd in (1, -1):
                tb = np.sort(b.t_dec[b.side == sd].to_numpy(float))
                for W in (1, 5, 30):
                    cnt = np.searchsorted(tb, td[idx], "left") - np.searchsorted(tb, td[idx] - W, "left")
                    k = np.where(side[idx] == sd, "same", "opp")
                    for kk in ("same", "opp"):
                        sel = k == kk
                        out["%s_%s%d" % (prefix, kk, W)][idx[sel]] += cnt[sel]
        # remove the event's own stock from the cross-stock counts
        own = by_date["x"].get(date, empty)
        for pm, jdx in d.iloc[idx].groupby("permno").indices.items():
            jj = idx[jdx]; ob = own[own.permno == pm]
            for sd in (1, -1):
                tb = np.sort(ob.t_dec[ob.side == sd].to_numpy(float))
                for W in (1, 5, 30):
                    cnt = np.searchsorted(tb, td[jj], "left") - np.searchsorted(tb, td[jj] - W, "left")
                    same = side[jj] == sd
                    out["x_same%d" % W][jj[same]] -= cnt[same]; out["x_opp%d" % W][jj[~same]] -= cnt[~same]
    f = pd.DataFrame(out)
    scale = d.date.map(per100).to_numpy(float)
    for c in [c for c in f if c.startswith("x_")]:
        f[c] = f[c] * scale
    return pd.concat([d, f], axis=1)


def part_cross(trn):
    """A35 at the event level: do bursts that coincide with bursts in many other stocks (or in SPY / QQQ) behave
    differently? IC and mean signed moves by coincidence, and the gain from adding the cross-stock features."""
    CROSS = ["x_%s%d" % (k, W) for W in (1, 5, 30) for k in ("same", "opp")] + ["etf_%s%d" % (k, W) for W in (1, 5, 30) for k in ("same", "opp")]
    rows = []
    for dn, kind in (("early5", "real"), ("run0.1", "real"), ("iso", "single"), ("run0.1", "pseudo")):
        d = load(dn, kind)
        if d is None:
            continue
        d = cross_features(d)
        tr, tests = split(d, trn)
        pos = d.x_same5[d.x_same5 > 0]
        cuts = [-np.inf, 0, pos.quantile(1 / 3), pos.quantile(2 / 3), np.inf] if len(pos) > 100 else [-np.inf, 0, np.inf]
        for y in ("r10", "r60", "r300", "f60", "r1800_x", "r_close_x"):
            for r in fit_eval(tr, tests, GENERIC + DEPTH + EVENT + CROSS, y, GENERIC + DEPTH + EVENT):
                rows.append(dict(defn=dn, kind=kind, target=y, split="gain from cross-stock features", bucket="all", **r))
            t = tr[tr[y].notna()]; t = t.sample(min(len(t), CAP_TRAIN), random_state=0)
            feats = [c for c in GENERIC + DEPTH + EVENT if c in t]; mf, lo, hi = M.hgb(t, feats, y)
            for tn, e in tests.items():
                e = e[e[y].notna()].copy(); e["p"] = mf.predict(e[feats].to_numpy(float)); e["yy"] = e[y].clip(lo, hi)
                e["xb"] = pd.cut(e.x_same5, cuts, labels=False)
                e["eb"] = np.where(e.etf_same5 > 0, "SPY/QQQ same-side burst in the last 5 s", "no SPY/QQQ burst")
                for col, lab in (("xb", "other stocks' same-side bursts in the last 5 s (bucket 0 = none)"), ("eb", "ETF")):
                    for k, g in e.groupby(col):
                        if len(g) < 2000:
                            continue
                        _, ic, _ = M.daily_ic(g.date.to_numpy(), g.p.to_numpy(), g.yy.to_numpy(), minn=10)
                        rows.append(dict(defn=dn, kind=kind, target=y, test=tn, split=lab, bucket=str(k), n=len(g),
                                         ic=ic["mean"], t=ic["t"], mean_move=float(g[y].mean())))
        print("cross", dn, kind, "done", flush=True)
        pd.DataFrame(rows).to_csv(D5 / "v5_cross.csv", index=False)
    return pd.DataFrame(rows)


def part_check():
    rows = [json.loads(l) for l in open(D5 / "V5_check.jsonl") if l.strip().startswith("{")]
    c = pd.DataFrame(rows)
    c = c[c.n.fillna(0) > 0]
    out = dict(stock_days=len(c), events=int(c.n.sum()),
               bid_exact_mean=float(np.average(c.bid_exact, weights=c.n)), ask_exact_mean=float(np.average(c.ask_exact, weights=c.n)),
               bid_ratio_median=float(c.bid_ratio_med.median()), ask_ratio_median=float(c.ask_ratio_med.median()),
               share_days_all_exact=float(((c.bid_exact == 1) & (c.ask_exact == 1)).mean()))
    json.dump(out, open(D5 / "v5_check.json", "w"), indent=1)
    return out


def main():
    what = _sys.argv[1:] or ["check", "placebo", "book", "dose", "robust", "cross", "panel"]
    trn = train_names()
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 800); pd.set_option("display.max_columns", 40)
    if "check" in what:
        print("\n### A39 rebuilt book vs quote path (level 1)"); print(part_check())
    if "placebo" in what:
        P, C = part_placebo(trn)
        for tn in TESTS:
            x = P[(P.test == tn) & (P.target.isin(["mid", "far quote"]))]
            print("\n### IC (GENERIC+DEPTH+EVENT), %s" % tn)
            print(x.pivot_table(index=["defn", "kind"], columns=["target", "horizon"], values="ic").round(3).to_string())
            print("\n### gain of EVENT over GENERIC+DEPTH, %s" % tn)
            print(x.pivot_table(index=["defn", "kind"], columns=["target", "horizon"], values="gain").round(3).to_string())
        print("\n### mean signed moves (bps)"); print(C.round(2).to_string(index=False))
    if "book" in what:
        B = part_book(trn)
        print(B.pivot_table(index=["defn", "kind", "target", "horizon"], columns=["test", "model"], values="ic").round(3).to_string())
    if "dose" in what:
        R = part_dose(trn); print(R.round(3).to_string(index=False))
    if "robust" in what:
        R = part_robust(trn); print(R.round(4).to_string(index=False))
    if "cross" in what:
        R = part_cross(trn); print(R.round(4).to_string(index=False))
    if "panel" in what:
        bins = sorted(glob.glob(str(D5 / "V5_bins_part*.csv.gz")))
        if bins:
            pd.concat([pd.read_csv(f, dtype={"date": str}) for f in bins]).to_csv(D5 / "V5_bins_all.csv.gz", index=False)
            U, C, T, V = V4.part_c(trn, D5 / "V5_bins_all.csv.gz")
            for df, nm in ((U, "panel_univariate"), (C, "panel_models"), (T, "panel_close_by_tod"), (V, "panel_volatility")):
                df.to_csv(D5 / ("v5_%s.csv" % nm), index=False)
            print(C.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
