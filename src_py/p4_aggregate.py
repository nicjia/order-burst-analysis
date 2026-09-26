#!/usr/bin/env python3
"""P4 revisit v1, stage 2 pass 1: name-day statistics and a per-burst sample from the stage-1 npz files.

P4_REVISIT_DESIGN.md sections 4-6. Runs on the cluster per cell. For each PERMNO (dates in order):
  large     |Q_b| >= 80th percentile of |Q_b| over the name's previous 20 available name-days of the same
            family (>= 5 required; otherwise the name-day has no large classification)
  peak      max(peak_raw, 0.01); informative(k): dmean >= k * peak for k in 0.25, 0.5 (primary), 0.75
  t_dec     max(t_b + 600, t_e + 10); a burst is used at clock c only if t_dec <= c (15:50, or 15:30 for tCLOSE)
  pseudo    the burst's pseudo-burst, informative(k) on its own path; used at clock c only if the real burst is used
            at c and the pseudo window starts after the real burst ends and decides by c (amendment A4)
  d_close   side * (mid_close - m_dec) / m_dec in bps; d_open and d_cc use CRSP next open / close on day t's
            split basis (dlycumfacpr); the pseudo analogue replaces m_dec by the pseudo m_dec
  linkage   among bursts of one family with a non-round modal size (mode_count >= 2): same-side and
            opposite-side bursts with the same modal size starting within 30 minutes (either direction)
Outputs (licensed-data derivatives, cluster or gitignored local):
  nameday_<cell>.csv.gz  one row per (permno, date, family): signals, class sums/counts, CRSP controls, targets
  strata_<cell>.csv.gz   Q2(a) sums per (permno, date, family, hour, size quintile)
  sample_<cell>.csv.gz   eligible large bursts, at most --per-nameday per name-day and family (deterministic)
"""
import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

RTH0 = 34200.0
CLOCKS = {"1550": 57000.0, "1530": 55800.0}
# Amendment A3: US equity early-close sessions (13:00). After the early close the NASDAQ book holds stub quotes,
# so every clock mid is invalid; these dates are excluded entirely (not counted as available days either).
EARLY_CLOSE = {
    "20120703", "20121123", "20121224", "20130703", "20131129", "20131224", "20140703", "20141128", "20141224",
    "20151127", "20151224", "20161125", "20170703", "20171124", "20180703", "20181123", "20181224", "20190703",
    "20191129", "20191224", "20201127", "20201224", "20211126", "20221125", "20230703", "20231124", "20240703",
    "20241129", "20241224", "20250703", "20251128", "20251224"}
# Amendment A3: a mid is valid only if the quoted spread at that time is at most 500 bps (stub quotes otherwise).
MAX_SPREAD_BPS = 500.0
KAPPAS = (0.25, 0.5, 0.75)
TICK = 0.01
LINK_WINDOW = 1800.0


def sample_keep(permno, date, family, n, cap):
    """Deterministic uniform subset of at most `cap` of the name-day's n candidates (design Q3: sampled by name-day)."""
    seed = int(hashlib.sha256(("p4-revisit-v1|sample|%s|%s|%s" % (permno, date, family)).encode()).hexdigest()[:16], 16)
    if n <= cap:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).permutation(n)[:cap])


# ---------------------------------------------------------------------------------------------
# CRSP

def crsp_frame(df):
    """Per-date controls and next-day prices for one PERMNO (rows sorted by date)."""
    d = df.sort_values("date").reset_index(drop=True).copy()
    for c in ("dlyopen", "dlyclose", "dlyprc", "dlyret", "dlyvol", "shrout", "dlycap", "dlycumfacpr"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    vol = d.dlyvol
    d["adv20"] = vol.shift(1).rolling(20, min_periods=10).mean()
    d["dvol20"] = (vol * d.dlyprc.abs()).shift(1).rolling(20, min_periods=10).mean()
    d["sigma20"] = d.dlyret.shift(1).rolling(20, min_periods=10).std()
    d["turn20"] = (vol / (d.shrout * 1000.0)).shift(1).rolling(20, min_periods=10).mean()
    d["ret_lag1"] = d.dlyret.shift(1)
    d["ret_lag5"] = (1 + d.dlyret).shift(1).rolling(5, min_periods=5).apply(np.prod, raw=True) - 1
    d["cap_lag1"] = d.dlycap.shift(1)
    # CRSP cumulative price factor (10 before a 10:1 split, 1 after): a day t+1 price on day t's basis is
    # price_{t+1} * cfacpr_t / cfacpr_{t+1} (amendment A5; the reversed ratio made splits look like -99% gaps).
    fac = d.dlycumfacpr
    d["open_next_adj"] = d.dlyopen.shift(-1) * fac / fac.shift(-1)
    d["close_next_adj"] = d.dlyclose.shift(-1) * fac / fac.shift(-1)
    d["ret_next"] = d.dlyret.shift(-1)
    d["clop"] = d.open_next_adj / d.dlyclose - 1
    return d.set_index("date")


# ---------------------------------------------------------------------------------------------
# Bursts

def family_arrays(z, fam):
    keys = [k for k in z.files if k.startswith(fam + "_")]
    if not keys:
        return None
    b = {k[len(fam) + 1:]: z[k].astype(np.float64) if z[k].dtype.kind == "f" else z[k] for k in keys}
    b["side"] = b["side"].astype(np.int64)
    b["t_dec"] = np.maximum(b["t_b"] + 600.0, b["t_e"] + 10.0)
    dur = b["t_e"] - b["t_b"]
    b["ps_t_dec"] = np.maximum(b["ps_t"] + 600.0, b["ps_t"] + dur + 10.0)
    b["peak"] = np.maximum(np.nan_to_num(b["peak_raw"], nan=-np.inf), TICK)
    b["ps_peak"] = np.maximum(np.nan_to_num(b["ps_peak_raw"], nan=-np.inf), TICK)
    return b


def links(b):
    """Same- and opposite-side links within 30 minutes and backward same-side links, by modal non-round size."""
    n = len(b["t_b"])
    same = np.zeros(n, bool); opp = np.zeros(n, bool); back = np.zeros(n, np.int64)
    size = b["mode_size"]; cnt = b["mode_count"]
    valid = np.isfinite(size) & (cnt >= 2) & (np.nan_to_num(size) % 100 != 0)
    idx = np.flatnonzero(valid)
    if len(idx) < 2:
        return valid, same, opp, back
    order = idx[np.lexsort((b["t_b"][idx], size[idx]))]
    s_sorted = size[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s_sorted)) + 1, len(order)]
    for a, e in zip(starts[:-1], starts[1:]):
        if e - a < 2:
            continue
        grp = order[a:e]
        t = b["t_b"][grp]; sd = b["side"][grp]
        for sgn in (1, -1):
            mine = grp[sd == sgn]; tm = b["t_b"][mine]
            other = grp[sd == sgn]; to = b["t_b"][other]
            lo = np.searchsorted(to, tm - LINK_WINDOW, "left"); hi = np.searchsorted(to, tm + LINK_WINDOW, "right")
            same[mine] = (hi - lo) > 1                      # itself is in the window
            back[mine] = np.searchsorted(to, tm, "left") - lo
            opp_t = b["t_b"][grp[sd == -sgn]]
            lo2 = np.searchsorted(opp_t, tm - LINK_WINDOW, "left"); hi2 = np.searchsorted(opp_t, tm + LINK_WINDOW, "right")
            opp[mine] = (hi2 - lo2) > 0
    return valid, same, opp, back


def burst_frame(b, keep, ctx, crsp_row, permno, date, fam):
    """Features and outcomes for bursts `keep` (the sample schema; also the Phase II feature source)."""
    m_ref = b["m_ref"][keep]; sk = ctx["s"][keep]; vol = ctx["vol"]
    adv = crsp_row.get("adv20", np.nan); o_t = crsp_row.get("dlyopen", np.nan)
    mid_close, on, cn = ctx["mid_close"], ctx["on"], ctx["cn"]
    with np.errstate(invalid="ignore", divide="ignore"):
        bps = lambda x: x / m_ref * 1e4  # noqa: E731
        f = pd.DataFrame(dict(
            permno=permno, date=date, family=fam, side=b["side"][keep], t_b=b["t_b"][keep], t_e=b["t_e"][keep],
            n=b["n"][keep], log_q_adv=np.log(vol[keep] / adv), mode_share=b["mode_count"][keep] / b["n"][keep],
            mode_nonround=ctx["valid"][keep].astype(int), peak_bps=bps(b["peak"][keep]), dmean_bps=bps(b["dmean"][keep]),
            d60_bps=bps(b["d60"][keep]), d600_bps=bps(b["d600"][keep]), ratio=b["dmean"][keep] / b["peak"][keep],
            spread_b=b["spread_b"][keep], spread_dec=b["spread_dec"][keep],
            pre30_bps=sk * (m_ref - b["m_pre30"][keep]) / b["m_pre30"][keep] * 1e4,
            own_open_dec_bps=sk * (b["m_dec"][keep] - o_t) / o_t * 1e4, tod=(b["t_b"][keep] - RTH0) / 23400.0,
            link_same=ctx["same"][keep].astype(int), link_opp=ctx["opp"][keep].astype(int), link_back=ctx["back"][keep],
            info_k25=ctx["info"][0.25][keep].astype(int), info_k50=ctx["info"][0.5][keep].astype(int),
            info_k75=ctx["info"][0.75][keep].astype(int),
            ps_info_k50=(ctx["ps_info"][0.5] & ctx["ps_elig"])[keep].astype(int),
            d_close=ctx["d_close"][keep], d_open=ctx["d_open"][keep], d_cc=ctx["d_cc"][keep],
            ps_d_close=ctx["ps_d_close"][keep],
            phi_close=sk * (mid_close - m_ref) / b["peak"][keep],
            phi_open=sk * (on - m_ref) / b["peak"][keep], phi_cc=sk * (cn - m_ref) / b["peak"][keep]))
        for extra in ("truncated_share", "hidden_share", "program_score", "exec_dec", "cancel_dec"):
            if extra in b:
                v = b[extra][keep]
                f[extra] = v / vol[keep] if extra in ("exec_dec", "cancel_dec") else v
    return f


def predict_spec(spec, X):
    import p4_phase2
    return p4_phase2.predict(spec, X)


def design_features(frame, fam):
    import p4_phase2
    return p4_phase2.features(frame, fam)


def name_day(b, day, crsp_row, threshold, fam, permno, date, cap, models=None):
    """One name-day, one family: nameday row dict, strata rows, sample frame."""
    s = b["side"].astype(float); vol = b["vol"]; q = s * vol
    large = vol >= threshold if np.isfinite(threshold) else np.zeros(len(vol), bool)
    close_ok = day.get("spread_close", np.nan) <= MAX_SPREAD_BPS
    mid_close = day.get("mid_close", np.nan) if close_ok else np.nan
    dec_ok = np.nan_to_num(b["spread_dec"], nan=np.inf) <= MAX_SPREAD_BPS
    with np.errstate(invalid="ignore", divide="ignore"):
        d_close = s * (mid_close - b["m_dec"]) / b["m_dec"] * 1e4
        ps_d_close = s * (mid_close - b["ps_m_dec"]) / b["ps_m_dec"] * 1e4
        on, cn = crsp_row.get("open_next_adj", np.nan), crsp_row.get("close_next_adj", np.nan)
        d_open = s * (on - b["m_dec"]) / b["m_dec"] * 1e4
        d_cc = s * (cn - b["m_dec"]) / b["m_dec"] * 1e4
    info = {k: b["dmean"] >= k * b["peak"] for k in KAPPAS}
    ps_info = {k: b["ps_dmean"] >= k * b["ps_peak"] for k in KAPPAS}
    row = dict(permno=permno, date=date, family=fam, n_bursts=len(vol), has_large=bool(np.isfinite(threshold)),
               threshold=threshold, n_large=int(large.sum()))
    after_real = np.nan_to_num(b["ps_t"], nan=-np.inf) >= b["t_e"]   # A4: pseudo window starts after the real burst
    for cname, clock in CLOCKS.items():
        elig = (b["t_dec"] <= clock) & dec_ok
        ps_elig = elig & after_real & (b["ps_t_dec"] <= clock)          # A4: the real burst must also be known
        row["S_all_" + cname] = float(q[elig].sum())
        row["S_large_" + cname] = float(q[elig & large].sum())
        row["n_elig_" + cname] = int(elig.sum())
        for k in KAPPAS:
            tag = "%s_k%02d" % (cname, int(k * 100))
            row["S_info_" + tag] = float(q[elig & large & info[k]].sum())
            row["S_pseudo_" + tag] = float(q[large & ps_elig & ps_info[k]].sum())
            row["n_info_" + tag] = int((elig & large & info[k]).sum())
            row["n_pseudo_" + tag] = int((large & ps_elig & ps_info[k]).sum())
    elig = (b["t_dec"] <= CLOCKS["1550"]) & dec_ok
    ps_elig = elig & after_real & (b["ps_t_dec"] <= CLOCKS["1550"])
    classes = {"all": elig, "large": elig & large}
    for k in KAPPAS:
        kk = "k%02d" % int(k * 100)
        classes["info_" + kk] = elig & large & info[k]
        classes["non_" + kk] = elig & large & ~info[k]
    for name, m in classes.items():
        for oname, o in (("dclose", d_close), ("dopen", d_open), ("dcc", d_cc)):
            ok = m & np.isfinite(o)
            row["sum_%s_%s" % (oname, name)] = float(o[ok].sum()); row["n_%s_%s" % (oname, name)] = int(ok.sum())
    for k in KAPPAS:
        kk = "k%02d" % int(k * 100)
        m = large & ps_elig & ps_info[k] & np.isfinite(ps_d_close)
        row["sum_dclose_pseudo_" + kk] = float(ps_d_close[m].sum()); row["n_dclose_pseudo_" + kk] = int(m.sum())

    # Q2(a) strata over large bursts with a known decision before the close
    valid, same, opp, back = links(b)
    strata = []
    use = large & (b["t_dec"] < 57600.0) & valid & dec_ok
    if use.any():
        hour = np.clip(((b["t_b"] - RTH0) // 3600).astype(int), 0, 6)
        rank = pd.Series(vol[use]).rank(method="first").to_numpy()
        quint = np.zeros(len(vol), int); quint[use] = np.minimum((5 * (rank - 1) // use.sum()).astype(int), 4)
        inf = info[0.5]
        frame = pd.DataFrame(dict(hour=hour[use], quint=quint[use], info=inf[use].astype(int),
                                  same=same[use].astype(int), opp=opp[use].astype(int)))
        g = frame.groupby(["hour", "quint", "info"]).agg(n=("same", "size"), same=("same", "sum"), opp=("opp", "sum")).reset_index()
        for r in g.itertuples(index=False):
            strata.append(dict(permno=permno, date=date, family=fam, hour=r.hour, quint=r.quint, info=r.info,
                               n=int(r.n), same=int(r.same), opp=int(r.opp)))

    # per-burst sample: eligible (15:50) large bursts
    samp = None
    cand = np.flatnonzero(elig & large)
    ctx = dict(s=s, vol=vol, valid=valid, same=same, opp=opp, back=back, info=info, ps_info=ps_info, ps_elig=ps_elig,
               d_close=d_close, d_open=d_open, d_cc=d_cc, ps_d_close=ps_d_close, mid_close=mid_close, on=on, cn=cn)
    if len(cand):
        keep = cand[sample_keep(permno, date, fam, len(cand), cap)]
        if len(keep):
            samp = burst_frame(b, keep, ctx, crsp_row, permno, date, fam)
        if models:
            frame = burst_frame(b, cand, ctx, crsp_row, permno, date, fam)
            for (mfam, mname), spec in models.items():
                if mfam != fam:
                    continue
                X = design_features(frame, fam)[spec["features"]].to_numpy(float)
                sel = predict_spec(spec, X) > spec["theta"]
                for cname, clock in CLOCKS.items():
                    ok = sel & (b["t_dec"][cand] <= clock)
                    row["S_pred_%s_%s" % (mname, cname)] = float(q[cand][ok].sum())
                    row["n_pred_%s_%s" % (mname, cname)] = int(ok.sum())
    return row, strata, samp


def day_controls(day, crsp_row):
    out = {}
    for k in ("buy_1530", "sell_1530", "buy_1550", "sell_1550", "unsigned_1550", "mid_open", "mid_1530", "mid_1550",
              "mid_close", "spread_1530", "spread_1550", "spread_close", "n_packets"):
        out[k] = day.get(k, np.nan)
    for k in ("1530", "1550", "close"):
        ok = out["spread_" + k] <= MAX_SPREAD_BPS
        out["valid_" + k] = bool(ok)
        if not ok:
            out["mid_" + k] = np.nan
    for k in ("dlyopen", "dlyclose", "dlyret", "adv20", "dvol20", "sigma20", "turn20", "ret_lag1", "ret_lag5",
              "cap_lag1", "clop", "ret_next", "primaryexch", "dlyvol"):
        out[k] = crsp_row.get(k, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        o = out["dlyopen"]
        out["own_open_1550"] = out["mid_1550"] / o - 1 if o and np.isfinite(o) else np.nan
        out["own_open_1530"] = out["mid_1530"] / o - 1 if o and np.isfinite(o) else np.nan
        out["tclose"] = out["mid_close"] / out["mid_1530"] - 1
    return out


def load_models(model_dir):
    if not model_dir:
        return None
    models = {}
    for f in sorted(Path(model_dir).glob("model_*_*.json")):
        _, fam, name = f.stem.split("_", 2)
        models[(fam, name)] = json.loads(f.read_text())
    return models or None


def process_permno(args):
    permno, dates, npz_dir, crsp_rows, out_dir, cap, model_dir = args
    models = load_models(model_dir)
    out_dir = Path(out_dir)
    done = out_dir / ("nameday_%d.csv.gz" % permno)
    if done.exists():
        return permno, "skip"
    crsp = crsp_frame(crsp_rows) if crsp_rows is not None and len(crsp_rows) else pd.DataFrame()
    buffers = {"T": [], "S": []}
    rows, strata, samples = [], [], []
    for date in sorted(dates):
        if date in EARLY_CLOSE:
            continue
        f = Path(npz_dir) / str(permno) / ("%s.npz" % date)
        if not f.exists():
            continue
        z = np.load(f)
        day = json.loads(str(z["day_json"]))
        crow = crsp.loc[date].to_dict() if len(crsp) and date in crsp.index else {}
        ctrl = day_controls(day, crow)
        for fam in ("T", "S"):
            b = family_arrays(z, fam)
            prev = buffers[fam]
            threshold = float(np.quantile(np.concatenate(prev), 0.8)) if len(prev) >= 5 and sum(map(len, prev)) else np.nan
            if b is None:
                buffers[fam] = (prev + [np.zeros(0)])[-20:]
                rows.append(dict(dict(permno=permno, date=date, family=fam, n_bursts=0, has_large=bool(np.isfinite(threshold)),
                                      threshold=threshold, n_large=0), **ctrl))
                continue
            row, st, sm = name_day(b, day, crow, threshold, fam, permno, date, cap, models)
            row.update(ctrl)
            rows.append(row); strata.extend(st)
            if sm is not None:
                samples.append(sm)
            buffers[fam] = (prev + [b["vol"]])[-20:]
    pd.DataFrame(strata).to_csv(out_dir / ("strata_%d.csv.gz" % permno), index=False, compression="gzip")
    (pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()).to_csv(
        out_dir / ("sample_%d.csv.gz" % permno), index=False, compression="gzip")
    tmp = out_dir / ("nameday_%d.part.csv.gz" % permno)
    pd.DataFrame(rows).to_csv(tmp, index=False, compression="gzip")
    tmp.rename(done)
    return permno, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--jobs", required=True, help="job file: date ticker permno")
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--crsp-dir", required=True, help="directory with crsp_<year>.csv.gz from p4_universe.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-nameday", type=int, default=10)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--permnos", default=None, help="comma-separated subset (testing)")
    ap.add_argument("--models", default=None, help="Phase II model directory (adds S_pred columns)")
    args = ap.parse_args()
    jobs = pd.read_csv(args.jobs, sep=" ", header=None, names=["date", "ticker", "permno"], dtype={"date": str})
    if args.permnos:
        jobs = jobs[jobs.permno.isin([int(p) for p in args.permnos.split(",")])]
    parts = Path(args.out) / "parts"; parts.mkdir(parents=True, exist_ok=True)
    years = sorted(jobs.date.str[:4].astype(int).unique())
    crsp = []
    for y in years:
        c = pd.read_csv(Path(args.crsp_dir) / ("crsp_%d.csv.gz" % y), dtype={"date": str})
        crsp.append(c[c.permno.isin(jobs.permno.unique())])
    crsp = pd.concat(crsp, ignore_index=True).drop_duplicates(["permno", "date"])
    by_permno = {int(p): g for p, g in crsp.groupby("permno")}
    tasks = [(int(p), g.date.tolist(), args.npz_dir, by_permno.get(int(p)), str(parts), args.per_nameday, args.models)
             for p, g in jobs.groupby("permno")]
    with Pool(args.procs) as pool:
        for i, (permno, n) in enumerate(pool.imap_unordered(process_permno, tasks), 1):
            if i % 50 == 0:
                print("done", i, "of", len(tasks), flush=True)
    for kind in ("nameday", "strata", "sample"):
        files = sorted(parts.glob(kind + "_*.csv.gz"))
        frames = []
        for f in files:
            try:
                frames.append(pd.read_csv(f, dtype={"date": str}))
            except pd.errors.EmptyDataError:
                pass
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        out.to_csv(Path(args.out) / ("%s_%s.csv.gz" % (kind, args.cell)), index=False, compression="gzip")
        print(kind, len(out))


if __name__ == "__main__":
    main()
