#!/usr/bin/env python3
"""burst-defs-raw-v4 stage 2: FORECASTING ONLY (no fills, no costs, no strategies).

(A) Term structure (events). For each definition, the IC of gradient-boosting forecasts of the signed mid move from the
    decision time to +1 s ... +3600 s, to the close, to the next open and to the next close (30 min onward
    market-excess; the overnight market leg is the panel's equal-weight mean): the full model (controls + book +
    burst) and the controls + book model, the paired gain (daily NW t), out-of-sample R^2 vs a zero forecast, sign hit
    rate and direction AUC on non-zero moves (and the share of zero moves), the top-minus-bottom decile of the
    realized mid move, and the IC of the burst's own move alone. Also the direction of the FIRST mid change after the
    decision (within 5 min; no zero outcomes). Real bursts vs matched pseudo events (random time, same duration and
    side) in the same 10:00-15:30 window.
(B) Volatility (events). log(1 + realized variance) over the next 60 / 300 / 1800 s: pre-event realized variance
    (60 / 300 / 1800 s) + controls; + book; + burst features. IC, R^2 and the paired gain of the burst features.
    Real vs pseudo: mean log RV after minus before.
(C) 5-minute panel over ALL bursts (cross-sectional, per date x bin; per-bin ICs averaged within the day, NW over days).
    Direction: next bin, next three bins, bin end to close (all market-excess). BASE = past excess returns (1 / 3 / 6
    bins), since-open excess return, all-trade flow imbalance and quote OFI (1 bin, 3 bins, since open), relative
    volume, bin. BURST = signed burst volume / volume (1 bin, 3 bins, since open) and log burst counts (1 / 3 bins) for
    every definition. Univariate ICs of the key signals; model IC base vs base + burst; close-target IC by time of day.
    Volatility: log(1 + RV) of the next bin from HAR-style past RV + packet counts, vs + burst counts.
(E) Market timing: cross-stock means of the flow and burst-imbalance signals per date x bin -> the next 5 / 15 minutes
    and rest of day of the equal-weight market index (time series; 2022-23 in sample, 2024 and 2025 out of sample).
(D) Daily: the day's burst imbalance per definition (whole day and last hour) -> overnight, next open-to-close, next
    close-to-close and next-5-day close-to-close mid returns; cross-sectional ICs, base (day and last-hour return,
    realized variance, all-trade flow imbalance, quote OFI, relative volume) vs base + bursts, linear in ranks.
Train: train stocks, 2022-23. Tests: the 2024 test stocks; every 2025 stock. Gradient boosting depth 3, fixed.
Usage: burst_defs_raw4_model.py [events|panel|market|daily|all]
"""
import sys as _sys, pathlib as _pl  # shared project modules live in src_py/
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "src_py"))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import p4_analyze as PA
import burst_defs_raw2_model as M

D4 = M.D / "burst_defs_raw4"
CTRL = ["since_open", "pre30m", "tod", "spread_dec"]
BOOK = ["imb_first", "imb_last", "qimb_dec", "qofi_pre60", "qofi_during", "tfi_pre60", "micro_gap", "canc_opp", "canc_same"]
BURST = ["move_during", "pre60", "opp_consumed", "spread_change", "log_n", "log_dur", "log_q_adv", "mode_share", "nonround",
         "size_cv", "size_to_depth", "iat_cv", "iat_med", "phase_R"]
RVPRE = ["lrvpre60", "lrvpre300", "lrvpre1800"]
HS = ["r1", "r2", "r5", "r10", "r30", "r60", "r120", "r300", "r600", "r1800_x", "r3600_x", "r_close_x", "r_next_open_x",
      "r_next_close_x"]
DEFS = ("run0.01", "run0.1", "early5", "levelclear", "cancel", "hidden", "run60")
TESTS = ("2024 test stocks", "2025 all stocks")


def market():
    z = np.load(M.D / "market_index" / "TEST_mkt.npz")
    return {x: i for i, x in enumerate(z["dates"])}, z["idx"]


def mk_at(dates, k, dix, I):
    di = pd.Series(dates).map(dix).fillna(-1).astype(int).to_numpy()
    k = np.broadcast_to(k, di.shape) if np.ndim(k) == 0 else k
    return np.where(di >= 0, I[np.maximum(di, 0), k], np.nan)


def minute_mark(t):
    """Index of the first market-index grid mark (9:31 + k minutes) at or after time t."""
    return np.clip(np.ceil((np.minimum(t, 57600.0) - 34200.0) / 60.0).astype(int) - 1, 0, 389)


def r2(y, p):
    ok = np.isfinite(y) & np.isfinite(p)
    return 1 - np.sum((y[ok] - p[ok]) ** 2) / np.sum(y[ok] ** 2)


def r2_mean(y, p, m):
    ok = np.isfinite(y) & np.isfinite(p)
    return 1 - np.sum((y[ok] - p[ok]) ** 2) / np.sum((y[ok] - m) ** 2)


def split(d, train):
    tr = d[d.permno.isin(train) & d.date.str[:4].isin(["2022", "2023"])]
    return tr, {TESTS[0]: d[~d.permno.isin(train) & (d.date.str[:4] == "2024")], TESTS[1]: d[d.date.str[:4] == "2025"]}


def next_day_mids(dix):
    """Per stock-day: the 16:00 mid, the next trading day's 9:31 and 16:00 mids, and the equal-weight panel means of the
    overnight and next-day open-to-close log returns (bps) used as the market leg of the next-day targets."""
    dd = pd.read_csv(D4 / "V4_bins.csv.gz", usecols=["permno", "date", "bin", "m_open", "m_close"], dtype={"date": str})
    dd = dd[dd.bin == 0].drop(columns="bin")
    cal = sorted(dix); pos = {x: i for i, x in enumerate(cal)}
    dd["date_next"] = dd.date.map(lambda x: cal[pos[x] + 1] if x in pos and pos[x] + 1 < len(cal) else None)
    nx = dd[["permno", "date", "m_open", "m_close"]].rename(columns={"date": "date_next", "m_open": "mo1", "m_close": "mc1"})
    dd = dd.merge(nx, on=["permno", "date_next"], how="left")
    with np.errstate(invalid="ignore", divide="ignore"):
        dd["on"] = np.log(dd.mo1 / dd.m_close) * 1e4; dd["oc1"] = np.log(dd.mc1 / dd.mo1) * 1e4
    for c in ("on", "oc1"):
        dd.loc[dd[c].abs() > 3000, c] = np.nan
    dd = dd.join(dd.groupby("date")[["on", "oc1"]].mean().rename(columns={"on": "mkt_on", "oc1": "mkt_oc1"}), on="date")
    return dd[["permno", "date", "m_close", "mo1", "mc1", "mkt_on", "mkt_oc1"]]


# ----------------------------------------------------------------------------------------------------------- events
def load_events():
    d = pd.read_csv(D4 / "V4_events.csv.gz", dtype={"date": str, "defn": "category", "kind": "category", "ticker": "category"})
    num = d.select_dtypes("float64").columns
    d[num] = d[num].astype(np.float32)
    nd = pd.read_csv(M.ROOT / "results" / "p4_revisit_v1" / "agg" / "TEST" / "nameday_TEST.csv.gz", dtype={"date": str},
                     usecols=["permno", "date", "family", "adv20"])
    d = d.merge(nd[nd.family == "T"][["permno", "date", "adv20"]], on=["permno", "date"], how="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        d["log_n"] = np.log(d.n_used.where(d.n_used > 0)); d["log_dur"] = np.log1p(d.dur)
        d["log_q_adv"] = np.log(d.vol_used.where(d.vol_used > 0) / d.adv20)
    d = d.replace([np.inf, -np.inf], np.nan); d = d[(d.spread_dec > 0) & (d.spread_dec < 500)]
    for c in ["r%d" % h for h in (1, 2, 5, 10, 30, 60, 120, 300, 600, 1800, 3600)] + ["r_close"]:
        d.loc[d[c].abs() > 1000, c] = np.nan
    dix, I = market()
    nx = next_day_mids(dix)
    d = d.merge(nx, on=["permno", "date"], how="left")
    k0 = minute_mark(d.t_dec.to_numpy()); dates = d.date.to_numpy(); s = d.side.to_numpy()
    m0 = mk_at(dates, k0, dix, I)
    for h in (1800, 3600):
        d["r%d_x" % h] = d["r%d" % h] - s * (mk_at(dates, minute_mark(d.t_dec.to_numpy() + h), dix, I) - m0) * 1e4
    rest = (mk_at(dates, 389, dix, I) - m0) * 1e4
    d["r_close_x"] = d.r_close - s * rest
    with np.errstate(invalid="ignore", divide="ignore"):
        m_d = d.m_close / (1 + s * d.r_close / 1e4)          # r_close = side * (m_close - m_d) / m_d * 1e4
        d["r_next_open_x"] = s * np.log(d.mo1 / m_d) * 1e4 - s * (rest + d.mkt_on)
        d["r_next_close_x"] = s * np.log(d.mc1 / m_d) * 1e4 - s * (rest + d.mkt_on + d.mkt_oc1)
    for c in ("r_next_open_x", "r_next_close_x"):
        d.loc[d[c].abs() > 3000, c] = np.nan
    for H in (60, 300, 1800):
        d["lrv%d" % H] = np.log1p(d["rv%d" % H]); d["lrvpre%d" % H] = np.log1p(d["rvpre%d" % H])
    d["window"] = np.where((d.t_b >= 36000.0) & (d.t_dec <= 55800.0), "10:00-15:30", "outside")
    R = d[["r1", "r2", "r5", "r10", "r30", "r60", "r120", "r300"]].to_numpy(float)
    nz = np.isfinite(R) & (R != 0); first = np.argmax(nz, axis=1)
    d["first_move"] = np.where(nz.any(axis=1), np.sign(R[np.arange(len(R)), first]), np.nan)   # direction of the first mid change
    return d


def part_a(d, train):
    tr_all, tests = split(d, train)
    F = CTRL + BOOK + BURST
    groups = [(dn, "real", None) for dn in DEFS] + [("run0.1", "real", "10:00-15:30"), ("run0.1", "pseudo", "10:00-15:30")]
    rows = []
    for dn, kind, win in groups:
        sel = lambda x: x[(x.defn == dn) & (x.kind == kind) & ((x.window == win) if win else True)]
        tr = sel(tr_all)
        if len(tr) < 3000:
            continue
        label = "%s %s%s" % (dn, kind, " " + win if win else "")
        for y in HS + ["first_move"]:
            mf, lo, hi = M.hgb(tr, F, y); mb, _, _ = M.hgb(tr, CTRL + BOOK, y)
            ym = float(tr[y].clip(lo, hi).mean())
            for tn, te in tests.items():
                e = sel(te); e = e[e[y].notna()]
                if len(e) < 1000:
                    continue
                yy = e[y].clip(lo, hi).to_numpy(float); dates = e.date.to_numpy()
                pf = mf.predict(e[F].to_numpy(float)); pb = mb.predict(e[CTRL + BOOK].to_numpy(float))
                sf, icf, sp = M.daily_ic(dates, pf, yy); sb, icb, _ = M.daily_ic(dates, pb, yy)
                _, icm, _ = M.daily_ic(dates, e.move_during.to_numpy(float), yy)
                g = PA.nw_t((sf - sb).to_numpy())
                rows.append(dict(group=label, horizon=y, test=tn, n=int(len(e)), ic=icf["mean"], t=icf["t"], ic_ctrl_book=icb["mean"],
                                 burst_gain=g["mean"], t_gain=g["t"], move_only_ic=icm["mean"], r2_vs_zero=r2(yy, pf),
                                 r2_vs_train_mean=r2_mean(yy, pf, ym), zero_share=float(np.mean(yy == 0)),
                                 hit_nonzero=float(np.mean(np.sign(pf[yy != 0]) == np.sign(yy[yy != 0]))),
                                 auc_up=float(roc_auc_score(yy[yy != 0] > 0, pf[yy != 0])),
                                 auc_up_ctrl_book=float(roc_auc_score(yy[yy != 0] > 0, pb[yy != 0])),
                                 decile_spread_bps=sp["mean"]))
        print("  A %s done" % label, flush=True)
    return pd.DataFrame(rows)


def part_b(d, train):
    tr_all, tests = split(d, train)
    base = RVPRE + ["tod", "spread_dec"]
    sets = {"pre-RV + controls": base, "+ book": base + BOOK, "+ book + burst": base + BOOK + BURST + ["since_open", "pre30m"]}
    groups = [(dn, "real", None) for dn in DEFS] + [("run0.1", "real", "10:00-15:30"), ("run0.1", "pseudo", "10:00-15:30")]
    rows, lift = [], []
    for dn, kind, win in groups:
        sel = lambda x: x[(x.defn == dn) & (x.kind == kind) & ((x.window == win) if win else True)]
        tr = sel(tr_all)
        if len(tr) < 3000:
            continue
        label = "%s %s%s" % (dn, kind, " " + win if win else "")
        for H in (60, 300, 1800):
            y = "lrv%d" % H
            ms = {k: M.hgb(tr, v, y)[0] for k, v in sets.items()}
            for tn, te in tests.items():
                e = sel(te); e = e[e[y].notna()]
                if len(e) < 1000:
                    continue
                yy = e[y].to_numpy(float); dates = e.date.to_numpy()
                out = dict(group=label, target="log(1+RV) next %d s" % H, test=tn, n=int(len(e)))
                ser = {}
                for k, v in sets.items():
                    p = ms[k].predict(e[v].to_numpy(float))
                    ser[k], ic, _ = M.daily_ic(dates, p, yy)
                    out["ic " + k] = ic["mean"]; out["r2 " + k] = r2_mean(yy, p, float(tr[y].mean()))
                g = PA.nw_t((ser["+ book + burst"] - ser["+ book"]).to_numpy())
                out["burst gain"] = g["mean"]; out["t gain"] = g["t"]
                rows.append(out)
                if H == 300:
                    lift.append(dict(group=label, test=tn, mean_lrv_before=float(e.lrvpre300.mean()), mean_lrv_after=float(e.lrv300.mean()),
                                     after_minus_before=float((e.lrv300 - e.lrvpre300).mean())))
        print("  B %s done" % label, flush=True)
    return pd.DataFrame(rows), pd.DataFrame(lift)


# ------------------------------------------------------------------------------------------------------------ panel
def group_ic(gid, p, y, minn=20):
    """Spearman correlation of p and y within each group id (groups with >= minn valid rows)."""
    ok = np.isfinite(p) & np.isfinite(y)
    df = pd.DataFrame(dict(g=gid[ok], p=p[ok], y=y[ok]))
    df["rp"] = df.groupby("g").p.rank(); df["ry"] = df.groupby("g").y.rank()
    g = df.groupby("g")
    df["rp"] -= g.rp.transform("mean"); df["ry"] -= g.ry.transform("mean")
    df["a"], df["b"], df["c"] = df.rp * df.ry, df.rp ** 2, df.ry ** 2
    s = df.groupby("g")[["a", "b", "c"]].sum(); n = df.groupby("g").size()
    with np.errstate(invalid="ignore", divide="ignore"):
        ic = s.a / np.sqrt(s.b * s.c)
    return ic[(n >= minn) & np.isfinite(ic)]


def group_spread(gid, p, y, q=0.1):
    ok = np.isfinite(p) & np.isfinite(y)
    df = pd.DataFrame(dict(g=gid[ok], p=p[ok], y=y[ok]))
    df["pr"] = df.groupby("g").p.rank(pct=True)
    return (df[df.pr > 1 - q].groupby("g").y.mean() - df[df.pr <= q].groupby("g").y.mean()).dropna()


def by_day(s):
    """Per-(date x bin) values -> daily means -> NW(10) over days."""
    dly = s.groupby(s.index // 100).mean()
    return dly, PA.nw_t(dly.to_numpy())


def build_panel(path=None):
    b = pd.read_csv(path or (D4 / "V4_bins.csv.gz"), dtype={"date": str, "ticker": "category"})
    b = b.sort_values(["permno", "date", "bin"]).reset_index(drop=True)
    n = len(b) // 78
    assert len(b) == 78 * n and (b.bin.to_numpy().reshape(n, 78) == np.arange(78)).all()
    key = b[["permno", "date"]].iloc[::78].reset_index(drop=True)
    A = lambda c: b[c].to_numpy(float).reshape(n, 78)
    dix, I = market()
    di = key.date.map(dix).fillna(-1).astype(int).to_numpy()
    Im = np.where(di[:, None] >= 0, I[np.maximum(di, 0)], np.nan) * 1e4          # market index, bps, at 9:31 + k min
    kend = 5 * (np.arange(78) + 1) - 1; kst = np.r_[0, kend[:-1]]
    ms, me = A("m_start"), A("m_end"); mo = b.m_open.to_numpy(float)[::78][:, None]; mc = b.m_close.to_numpy(float)[::78][:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        ret_x = np.log(me / ms) * 1e4 - (Im[:, kend] - Im[:, kst])
        since_x = np.log(me / mo) * 1e4 - Im[:, kend]
        y_close = np.log(mc / me) * 1e4 - (Im[:, [389]] - Im[:, kend])
    y_next = np.full_like(ret_x, np.nan); y_next[:, :-1] = ret_x[:, 1:]
    y_next3 = np.full_like(ret_x, np.nan); y_next3[:, :-3] = ret_x[:, 1:-2] + ret_x[:, 2:-1] + ret_x[:, 3:]
    mkt_past = Im[:, kend] - Im[:, kst]
    mkt_next = np.full_like(mkt_past, np.nan); mkt_next[:, :-1] = mkt_past[:, 1:]
    mkt_next3 = np.full_like(mkt_past, np.nan); mkt_next3[:, :-3] = mkt_past[:, 1:-2] + mkt_past[:, 2:-1] + mkt_past[:, 3:]
    mkt_close = Im[:, [389]] - Im[:, kend]
    rv = A("rv"); lrv = np.log1p(rv)
    y_rv = np.full_like(rv, np.nan); y_rv[:, :-1] = lrv[:, 1:]

    def roll(x, L):
        c = np.nancumsum(x, axis=1); o = c.copy(); o[:, L:] = c[:, L:] - c[:, :-L]; return o

    def ratio(a, v):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(v > 0, a / v, np.nan)
    flow, vol, qofi, npk = A("flow"), A("volume"), A("qofi"), A("npk")
    cols = dict(past_ret1=ret_x, past_ret3=roll(ret_x, 3), past_ret6=roll(ret_x, 6), since_open_x=since_x,
                tfi1=ratio(flow, vol), tfi3=ratio(roll(flow, 3), roll(vol, 3)), tfi_cum=ratio(np.nancumsum(flow, 1), np.nancumsum(vol, 1)),
                qofi1=qofi, qofi3=roll(qofi, 3), qofi_cum=np.nancumsum(qofi, 1),
                vol_rel=ratio(vol * np.arange(1, 79), np.nancumsum(vol, 1)),
                lrv1=lrv, lrv3=np.log1p(roll(rv, 3) / 3), lrv12=np.log1p(roll(rv, 12) / np.minimum(np.arange(1, 79), 12)),
                lnpk1=np.log1p(npk), lnpk3=np.log1p(roll(npk, 3)),
                y_next=y_next, y_next3=y_next3, y_close=y_close, y_rv=y_rv,
                mkt_past=mkt_past, mkt_next=mkt_next, mkt_next3=mkt_next3, mkt_close=mkt_close)
    burst = []
    for c in [c for c in b.columns if c.startswith("sv_")]:
        dn = c[3:]; sv, nb = A(c), A("nb_" + dn)
        cols["bf_%s_1" % dn] = ratio(sv, vol); cols["bf_%s_3" % dn] = ratio(roll(sv, 3), roll(vol, 3))
        cols["bf_%s_cum" % dn] = ratio(np.nancumsum(sv, 1), np.nancumsum(vol, 1))
        cols["bn_%s_1" % dn] = np.log1p(nb); cols["bn_%s_3" % dn] = np.log1p(roll(nb, 3))
        burst += ["bf_%s_1" % dn, "bf_%s_3" % dn, "bf_%s_cum" % dn, "bn_%s_1" % dn, "bn_%s_3" % dn]
    P = pd.DataFrame({k: v.ravel().astype(np.float32) for k, v in cols.items()})
    P["permno"] = np.repeat(key.permno.to_numpy(), 78); P["date"] = np.repeat(key.date.to_numpy(), 78)
    P["bin"] = np.tile(np.arange(78), n)
    P["gid"] = P.date.astype(np.int64) * 100 + P.bin
    P = P[(P.bin >= 6) & (P.bin <= 76)].replace([np.inf, -np.inf], np.nan)   # 10:00 onward; the last bin has no next bin
    for c in ("y_next", "y_next3", "y_close"):
        P.loc[P[c].abs() > 1000, c] = np.nan
    return P, burst


def part_c(train, path=None):
    P, burst = build_panel(path)
    BASE = ["past_ret1", "past_ret3", "past_ret6", "since_open_x", "tfi1", "tfi3", "tfi_cum", "qofi1", "qofi3", "qofi_cum",
            "vol_rel", "bin"]
    tr, tests = split(P, train)
    print("panel: train rows %d, tests %s" % (len(tr), {k: len(v) for k, v in tests.items()}), flush=True)
    uni, mod, tod, vol = [], [], [], []
    sig = ["past_ret1", "since_open_x", "tfi1", "tfi_cum", "qofi1", "qofi_cum"] + [c for c in burst if c.startswith("bf_")]
    for tn, e in tests.items():
        gid = e.gid.to_numpy()
        for y in ("y_next", "y_next3", "y_close"):
            yy = e[y].to_numpy(float)
            for c in sig:
                _, s = by_day(group_ic(gid, e[c].to_numpy(float), yy))
                uni.append(dict(test=tn, target=y, signal=c, ic=s["mean"], t=s["t"]))
    for y in ("y_next", "y_next3", "y_close"):
        mb, lo, hi = M.hgb(tr, BASE, y); mf, _, _ = M.hgb(tr, BASE + burst, y)
        for tn, e in tests.items():
            e = e[e[y].notna()]; gid = e.gid.to_numpy(); yy = e[y].clip(lo, hi).to_numpy(float)
            pb, pf = mb.predict(e[BASE].to_numpy(float)), mf.predict(e[BASE + burst].to_numpy(float))
            db, sb = by_day(group_ic(gid, pb, yy)); df_, sf = by_day(group_ic(gid, pf, yy))
            common = db.index.intersection(df_.index)
            g = PA.nw_t((df_[common] - db[common]).to_numpy())
            _, sp = by_day(group_spread(gid, pf, yy))
            mod.append(dict(target=y, test=tn, ic_base=sb["mean"], t_base=sb["t"], ic_with_bursts=sf["mean"], t_with=sf["t"],
                            burst_gain=g["mean"], t_gain=g["t"], decile_spread_bps=sp["mean"]))
            if y == "y_close":
                bins = e.bin.to_numpy()
                for lab, lo_b, hi_b in (("10:00-11:30", 6, 23), ("11:30-13:30", 24, 47), ("13:30-15:00", 48, 65), ("15:00-15:55", 66, 76)):
                    k = (bins >= lo_b) & (bins <= hi_b)
                    _, s1 = by_day(group_ic(gid[k], pb[k], yy[k])); _, s2 = by_day(group_ic(gid[k], pf[k], yy[k]))
                    tod.append(dict(test=tn, window=lab, ic_base=s1["mean"], t_base=s1["t"], ic_with_bursts=s2["mean"], t_with=s2["t"]))
        print("  C %s done" % y, flush=True)
    VB = ["lrv1", "lrv3", "lrv12", "lnpk1", "lnpk3", "vol_rel", "bin"]
    VBURST = [c for c in burst if c.startswith("bn_")]
    mb, _, _ = M.hgb(tr, VB, "y_rv"); mf, _, _ = M.hgb(tr, VB + VBURST, "y_rv")
    for tn, e in tests.items():
        e = e[e.y_rv.notna()]; gid = e.gid.to_numpy(); yy = e.y_rv.to_numpy(float)
        pb, pf = mb.predict(e[VB].to_numpy(float)), mf.predict(e[VB + VBURST].to_numpy(float))
        db, sb = by_day(group_ic(gid, pb, yy)); df_, sf = by_day(group_ic(gid, pf, yy))
        common = db.index.intersection(df_.index); g = PA.nw_t((df_[common] - db[common]).to_numpy())
        ym = float(tr.y_rv.mean())
        vol.append(dict(test=tn, target="log(1+RV) next 5 min", ic_har_counts=sb["mean"], ic_with_burst_counts=sf["mean"],
                        burst_gain=g["mean"], t_gain=g["t"], r2_har_counts=r2_mean(yy, pb, ym), r2_with_bursts=r2_mean(yy, pf, ym)))
    return pd.DataFrame(uni), pd.DataFrame(mod), pd.DataFrame(tod), pd.DataFrame(vol)


def part_e():
    """Market timing: equal-weight means across ALL panel stocks per date x bin of the flow and burst-imbalance signals ->
    the next bin, next three bins and rest-of-day return of the equal-weight minute market index. Time series:
    correlation with NW t (lags cover the target overlap), by period; OLS fitted on 2022-23, out-of-sample R^2 in 2024
    and 2025 with and without the burst aggregates."""
    P, burst = build_panel()
    sig = ["tfi1", "tfi3", "qofi1", "qofi3", "past_ret1"] + [c for c in burst if c.startswith("bf_") and not c.endswith("_cum")]
    agg = P.groupby("gid")[sig + ["mkt_past", "mkt_next", "mkt_next3", "mkt_close"]].mean().sort_index()
    agg["n"] = P.groupby("gid").size()
    del P
    agg = agg[agg.n >= 20]
    yr = (agg.index // 1000000).astype(str)
    agg["period"] = np.where(np.isin(yr, ["2022", "2023"]), "2022-23", yr)
    lags = {"mkt_next": 5, "mkt_next3": 10, "mkt_close": 80}
    rows = []
    for per, g in agg.groupby("period"):
        for y, L in lags.items():
            for c in sig + ["mkt_past"]:
                x = g[[c, y]].dropna()
                if len(x) < 500:
                    continue
                z = ((x[c] - x[c].mean()) / x[c].std()) * ((x[y] - x[y].mean()) / x[y].std())
                w = PA.nw_t(z.to_numpy(), lags=L)
                rows.append(dict(period=per, target=y, signal=c, corr=w["mean"], t=w["t"], spearman=x[c].rank().corr(x[y].rank()), n=len(x)))
    base = ["mkt_past", "tfi1", "qofi1"]
    full = base + [c for c in burst if c.startswith("bf_") and c.endswith("_1")]
    oos = []
    for y in lags:
        tr = agg[agg.period == "2022-23"].dropna(subset=full + [y])
        for cols, lab in ((base, "market past + all-trade flow + OFI"), (full, "+ burst imbalances")):
            X = np.c_[np.ones(len(tr)), tr[cols].to_numpy(float)]
            beta = np.linalg.lstsq(X, tr[y].to_numpy(float), rcond=None)[0]
            for per in ("2024", "2025"):
                te = agg[agg.period == per].dropna(subset=full + [y])
                p = np.c_[np.ones(len(te)), te[cols].to_numpy(float)] @ beta
                yy = te[y].to_numpy(float)
                oos.append(dict(target=y, model=lab, period=per, n=len(te), r2_vs_zero=r2(yy, p), corr=float(np.corrcoef(p, yy)[0, 1])))
    return pd.DataFrame(rows), pd.DataFrame(oos)


def build_daily():
    """One row per stock-day: day and last-hour returns, realized variance, flow, OFI, relative volume, the day's and
    last hour's signed burst imbalance and burst share per definition (all known at 16:00), and the overnight, next
    open-to-close, next close-to-close and next-5-day mid returns (with date-demeaned copies)."""
    b = pd.read_csv(D4 / "V4_bins.csv.gz", dtype={"date": str, "ticker": "category"})
    b = b.sort_values(["permno", "date", "bin"]).reset_index(drop=True)
    n = len(b) // 78
    assert len(b) == 78 * n
    A = lambda c: b[c].to_numpy(float).reshape(n, 78)
    day = b[["permno", "date"]].iloc[::78].reset_index(drop=True)
    ms, me, vol, flow, qofi, rv, npk = A("m_start"), A("m_end"), A("volume"), A("flow"), A("qofi"), A("rv"), A("npk")
    day["mo"], day["mc"] = b.m_open.to_numpy(float)[::78], b.m_close.to_numpy(float)[::78]
    with np.errstate(invalid="ignore", divide="ignore"):
        day["day_ret"] = np.log(day.mc / day.mo) * 1e4
        day["last_hour_ret"] = np.log(me[:, 77] / ms[:, 66]) * 1e4
        day["lrv_day"] = np.log1p(np.nansum(rv, 1))
        day["tfi_day"] = np.nansum(flow, 1) / np.nansum(vol, 1); day["tfi_lh"] = np.nansum(flow[:, 66:], 1) / np.nansum(vol[:, 66:], 1)
        day["qofi_day"] = np.nansum(qofi, 1); day["qofi_lh"] = np.nansum(qofi[:, 66:], 1)
        day["lvol"] = np.log(np.nansum(vol, 1))
        burst = []
        for c in [c for c in b.columns if c.startswith("sv_")]:
            dn = c[3:]; sv, nb = A(c), A("nb_" + dn)
            day["bf_%s_day" % dn] = np.nansum(sv, 1) / np.nansum(vol, 1)
            day["bf_%s_lh" % dn] = np.nansum(sv[:, 66:], 1) / np.nansum(vol[:, 66:], 1)
            day["bshare_%s" % dn] = np.nansum(nb, 1) / np.nansum(npk, 1)
            burst += ["bf_%s_day" % dn, "bf_%s_lh" % dn, "bshare_%s" % dn]
    del b
    day = day.replace([np.inf, -np.inf], np.nan).sort_values(["permno", "date"]).reset_index(drop=True)
    day["vol_rel"] = day.lvol - day.groupby("permno").lvol.transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    dix, _ = market(); cal = sorted(dix); pos = {x: i for i, x in enumerate(cal)}
    ahead = lambda k: day.date.map(lambda x: cal[pos[x] + k] if x in pos and pos[x] + k < len(cal) else None)
    look = day.set_index(["permno", "date"])[["mo", "mc"]]
    def at(k, col):
        idx = pd.MultiIndex.from_arrays([day.permno, ahead(k)])
        return look[col].reindex(idx).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        mo1, mc1, mc5 = at(1, "mo"), at(1, "mc"), at(5, "mc")
        day["y_overnight"] = np.log(mo1 / day.mc) * 1e4; day["y_next_oc"] = np.log(mc1 / mo1) * 1e4
        day["y_next_cc"] = np.log(mc1 / day.mc) * 1e4; day["y_next5_cc"] = np.log(mc5 / day.mc) * 1e4
    Y = ["y_overnight", "y_next_oc", "y_next_cc", "y_next5_cc"]
    for y in Y:
        day.loc[day[y].abs() > 3000, y] = np.nan
        day[y + "_dm"] = day[y] - day.groupby("date")[y].transform("mean")
    day["gid"] = day.date.astype(np.int64)
    return day, burst, Y


def rank_x(e, cols):
    """Per-date percentile ranks centred at 0; missing values sit at the median."""
    return np.nan_to_num(e.groupby("date")[cols].rank(pct=True).to_numpy(float) - 0.5)


def rank_ols(tr, cols, y):
    t = tr[tr[y].notna()]
    yy = t[y].clip(*t[y].quantile([0.01, 0.99])).to_numpy(float)
    return np.linalg.lstsq(np.c_[np.ones(len(t)), rank_x(t, cols)], yy, rcond=None)[0]


def rank_predict(e, cols, beta):
    return np.c_[np.ones(len(e)), rank_x(e, cols)] @ beta


def part_d(train):
    """Daily horizon: the day's burst imbalance (known at 16:00) -> overnight, next open-to-close, next close-to-close and
    next-5-day close-to-close mid returns. Cross-sectional (per date) rank ICs, NW over dates. Models are linear in
    per-date ranks (gradient boosting overfits the ~22k training stock-days), fitted to date-demeaned targets."""
    day, burst, Y = build_daily()
    BASE = ["day_ret", "last_hour_ret", "lrv_day", "tfi_day", "tfi_lh", "qofi_day", "qofi_lh", "vol_rel"]
    tr, tests = split(day, train)
    print("daily: train %d stock-days, tests %s" % (len(tr), {k: len(v) for k, v in tests.items()}), flush=True)
    uni, mod = [], []
    sig = ["day_ret", "last_hour_ret", "tfi_day", "tfi_lh", "qofi_day"] + [c for c in burst if c.startswith("bf_")]
    for tn, e in tests.items():
        for y in Y:
            for c in sig:
                s = group_ic(e.gid.to_numpy(), e[c].to_numpy(float), e[y].to_numpy(float))
                w = PA.nw_t(s.to_numpy())
                uni.append(dict(test=tn, target=y, signal=c, ic=w["mean"], t=w["t"], days=len(s)))
    for y in Y:
        yd = y + "_dm"
        bb, bf = rank_ols(tr, BASE, yd), rank_ols(tr, BASE + burst, yd)
        for tn, e in tests.items():
            e = e[e[y].notna()]; gid = e.gid.to_numpy(); yy = e[yd].to_numpy(float)
            sb = group_ic(gid, rank_predict(e, BASE, bb), yy); sf = group_ic(gid, rank_predict(e, BASE + burst, bf), yy)
            common = sb.index.intersection(sf.index); g = PA.nw_t((sf[common] - sb[common]).to_numpy())
            wb, wf = PA.nw_t(sb.to_numpy()), PA.nw_t(sf.to_numpy())
            mod.append(dict(target=y, test=tn, days=len(common), ic_base=wb["mean"], t_base=wb["t"], ic_with_bursts=wf["mean"],
                            t_with=wf["t"], burst_gain=g["mean"], t_gain=g["t"]))
    return pd.DataFrame(uni), pd.DataFrame(mod)


def main():
    what = _sys.argv[1] if len(_sys.argv) > 1 else "all"
    train = set(pd.read_csv(M.ROOT / "results" / "p4_revisit_v1" / "jobs" / "DEFS_RAW_train_names.txt", header=None)[0])
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 30)
    if what in ("events", "all"):
        d = load_events()
        print("events %d (%s)" % (len(d), d.groupby(["defn", "kind"], observed=True).size().to_dict()), flush=True)
        TS = part_a(d, train); TS.to_csv(D4 / "v4_term_structure.csv", index=False)
        VO, LI = part_b(d, train); VO.to_csv(D4 / "v4_volatility.csv", index=False); LI.to_csv(D4 / "v4_vol_lift.csv", index=False)
        del d
        for tn in TESTS:
            x = TS[TS.test == tn]
            print("\n### TERM STRUCTURE: IC of the full model, %s" % tn)
            print(x.pivot_table(index="group", columns="horizon", values="ic").reindex(columns=HS).round(3).to_string())
            print("\n### burst gain over controls + book (IC), %s" % tn)
            print(x.pivot_table(index="group", columns="horizon", values="burst_gain").reindex(columns=HS).round(3).to_string())
            print("\n### t of the gain, %s" % tn)
            print(x.pivot_table(index="group", columns="horizon", values="t_gain").reindex(columns=HS).round(1).to_string())
            print("\n### out-of-sample R^2 vs zero (%%), %s" % tn)
            print((100 * x.pivot_table(index="group", columns="horizon", values="r2_vs_zero")).reindex(columns=HS).round(2).to_string())
            print("\n### sign hit rate on non-zero moves, %s" % tn)
            print(x.pivot_table(index="group", columns="horizon", values="hit_nonzero").reindex(columns=HS).round(3).to_string())
            print("\n### direction AUC on non-zero moves (full model / controls + book), %s" % tn)
            print(x.pivot_table(index="group", columns="horizon", values="auc_up").reindex(columns=HS + ["first_move"]).round(3).to_string())
            print(x.pivot_table(index="group", columns="horizon", values="auc_up_ctrl_book").reindex(columns=HS + ["first_move"]).round(3).to_string())
            print("\n### IC of the burst's own move alone, %s" % tn)
            print(x.pivot_table(index="group", columns="horizon", values="move_only_ic").reindex(columns=HS).round(3).to_string())
        print("\n### VOLATILITY (events)")
        print(VO.round(4).to_string(index=False))
        print(LI.round(3).to_string(index=False))
    if what in ("panel", "all"):
        U, C, T, V = part_c(train)
        for df, nm in ((U, "panel_univariate"), (C, "panel_models"), (T, "panel_close_by_tod"), (V, "panel_volatility")):
            df.to_csv(D4 / ("v4_%s.csv" % nm), index=False)
        print("\n### 5-MINUTE PANEL: univariate cross-sectional ICs (daily-averaged, NW t)")
        print(U.pivot_table(index="signal", columns=["test", "target"], values="ic").round(4).to_string())
        print(U.pivot_table(index="signal", columns=["test", "target"], values="t").round(1).to_string())
        print("\n### 5-MINUTE PANEL: base vs base + bursts")
        print(C.round(4).to_string(index=False))
        print("\n### bin end -> close, by time of day")
        print(T.round(4).to_string(index=False))
        print("\n### next-bin volatility")
        print(V.round(4).to_string(index=False))
    if what in ("market", "all"):
        ME, MO = part_e()
        ME.to_csv(D4 / "v4_market_timing.csv", index=False); MO.to_csv(D4 / "v4_market_timing_oos.csv", index=False)
        print("\n### MARKET TIMING: correlation of cross-stock aggregates with the next market return (NW t)")
        print(ME.pivot_table(index="signal", columns=["target", "period"], values="corr").round(3).to_string())
        print(ME.pivot_table(index="signal", columns=["target", "period"], values="t").round(1).to_string())
        print(MO.round(4).to_string(index=False))
    if what in ("daily", "all"):
        DU, DM = part_d(train)
        DU.to_csv(D4 / "v4_daily_univariate.csv", index=False); DM.to_csv(D4 / "v4_daily_models.csv", index=False)
        print("\n### DAILY: univariate cross-sectional ICs (NW t over dates)")
        print(DU.pivot_table(index="signal", columns=["test", "target"], values="ic").round(4).to_string())
        print(DU.pivot_table(index="signal", columns=["test", "target"], values="t").round(1).to_string())
        print("\n### DAILY: base vs base + bursts")
        print(DM.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
