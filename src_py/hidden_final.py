#!/usr/bin/env python3
"""
hidden_final.py — three remaining referee tests in ONE pass over the message stream, so
each ticker-day is streamed from the archive once rather than three times.

TEST A — FULL-UNIVERSE TERM STRUCTURE TO THE CLOSE.
  The published term-structure table rests on 48 names over 2023 alone, which forces a
  subsample qualifier onto every "to the close" claim. Same methodology as hidden_term.py
  (quote-rule-signed aggressive hidden bursts; markouts at 3/15/30/60/120 min and to the
  close; each netted against a TIME-OF-DAY-STRATIFIED placebo drawing each burst's
  counterfactual time within its own 30-minute bucket), run on the full 474-name panel.

TEST B — VAR SPECIFICATION ISOLATION.
  The markout and the Hasbrouck VAR disagree, and the pre-drift decomposition ruled out the
  obvious explanation. What remains is specification. The original VAR(12) on a 10-second
  clock reported a permanent response; the current VAR(30) on a 60-second clock with ridge
  shrinkage turns negative by 10 minutes. We fit the 2x2 grid so the two candidate causes
  are separated rather than confounded:
      S1  10s clock, 12 lags, no ridge   (the original specification)
      S2  10s clock, 12 lags, ridge
      S3  60s clock, 30 lags, no ridge
      S4  60s clock, 30 lags, ridge      (the current specification)
  Comparing S1 vs S3 isolates the clock/memory horizon; S1 vs S2 isolates shrinkage.

TEST C — SEQUENTIAL-SWEEP DEPLETION.
  A type-5 print executes against non-displayed liquidity and so does not itself consume the
  displayed queue -- which is why the pre-trade depletion ratio is a size normalization
  rather than a literal consumed fraction. The mechanical channel that DOES apply to hidden
  prints is the same aggressor's marketable remainder sweeping displayed levels milliseconds
  later. We therefore measure the touch on the side being taken immediately before the print
  and again at +100ms and +1s, call the level "swept" if its price moved away or its size
  more than halved at an unchanged price, and compare markouts for swept and unswept prints.
  If the footprint lives only in swept prints it is displacement; if it survives in unswept
  ones it is not.

Output: ticker,date,
        n_b,mk3,mk15,mk30,mk60,mk120,mkclose,pmk3,pmk15,pmk30,pmk60,pmk120,pmkclose,
        s1_stat,s1_3,s1_10,s1_30, s2_stat,s2_3,s2_10,s2_30,
        s3_stat,s3_3,s3_10,s3_30, s4_stat,s4_3,s4_10,s4_30,
        n_sw,fsw100,fsw1s,dsz100,dsz1s,mk3_sw,mk3_unsw
"""
import argparse, os, re, sys
import numpy as np, pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import burst_alt as BA

RTH0, RTH1 = 34200.0, 57600.0
NCOL = 38
NA = "{t},{d}" + ",nan" * (NCOL - 2)
SPECS = [(10.0, 12, 0.0), (10.0, 12, 1e-2), (60.0, 30, 0.0), (60.0, 30, 1e-2)]


def wm(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; x = x[np.abs(x) <= 1000]
    return float(np.mean(x)) if len(x) else np.nan


def bursts_from(ht, sign, minrun=3, gap=1.0):
    nz = sign != 0
    ht, sign = ht[nz], sign[nz]
    ends, dirs = [], []
    i, n = 0, len(ht)
    while i < n:
        j = i
        while j + 1 < n and sign[j + 1] == sign[i] and (ht[j + 1] - ht[j]) < gap:
            j += 1
        if j - i + 1 >= minrun:
            ends.append(ht[j]); dirs.append(int(sign[i]))
        i = j + 1
    return np.array(ends, float), np.array(dirs)


def mk(bt, bm, t0, dirs, base, dt=None, target_t=None):
    tt = (t0 + dt) if target_t is None else np.full(len(t0), target_t)
    e = BA.mid_at(bt, bm, tt)
    return wm(dirs * (e - base) / base * 1e4)


def fit_var(X, p, ridge):
    T = len(X)
    if T < 4 * p + 30:
        return None
    Y = X[p:]
    Z = np.column_stack([X[p - i - 1:T - i - 1] for i in range(p)])
    Z = np.column_stack([np.ones(len(Z)), Z])
    G = Z.T @ Z
    if ridge > 0:
        G = G + ridge * len(Z) * np.eye(Z.shape[1])
    else:
        G = G + 1e-8 * np.eye(Z.shape[1])
    try:
        B = np.linalg.solve(G, Z.T @ Y)
    except np.linalg.LinAlgError:
        return None
    return [B[1 + 2 * i:3 + 2 * i].T for i in range(p)]


def irf_multi(A, p, want):
    """Cumulative response of dmid to a unit flow shock at each horizon in `want`.
    Iterates the MA representation once to max(want) rather than once per horizon."""
    smax = max(want)
    psi = [np.eye(2)]
    cum = 1.0 * psi[0][1, 0]
    out = {}
    if 0 in want:
        out[0] = cum
    for s in range(1, smax + 1):
        acc = np.zeros((2, 2))
        for i in range(min(s, p)):
            acc += A[i] @ psi[s - i - 1]
        psi.append(acc)
        cum += acc[1, 0]
        if s in want:
            out[s] = float(cum)
    return [out[s] for s in want]


def run_var(bt, bm, ht, q, hsz, sel, delta, plag, ridge):
    """Returns (stationary_flag, irf_3min, irf_10min, irf_30min) or NaNs."""
    edges = np.arange(RTH0, RTH1 + delta, delta); nb = len(edges) - 1
    if nb < 4 * plag + 30:
        return (np.nan,) * 4
    flow = np.zeros(nb)
    bi = np.clip(((ht[sel] - RTH0) / delta).astype(int), 0, nb - 1)
    np.add.at(flow, bi, q[sel] * hsz[sel])
    me = BA.mid_at(bt, bm, edges)
    me = pd.Series(me).ffill().bfill().to_numpy()
    if not np.all(np.isfinite(me[:nb])) or np.any(me[:nb] <= 0):
        return (np.nan,) * 4
    dmid = np.clip(1e4 * np.diff(me) / me[:nb], -500, 500)
    ok = np.isfinite(flow) & np.isfinite(dmid)
    if ok.sum() < 4 * plag + 30 or np.std(flow[ok]) < 1e-9 or np.std(dmid[ok]) < 1e-9:
        return (np.nan,) * 4
    f = (flow[ok] - flow[ok].mean()) / flow[ok].std()
    X = np.column_stack([f, dmid[ok]])
    A = fit_var(X, plag, ridge)
    if A is None:
        return (np.nan,) * 4
    comp = np.zeros((2 * plag, 2 * plag))
    for i in range(plag):
        comp[0:2, 2 * i:2 * i + 2] = A[i]
    if plag > 1:
        comp[2:2 * plag, 0:2 * (plag - 1)] = np.eye(2 * (plag - 1))
    try:
        stat = int(np.max(np.abs(np.linalg.eigvals(comp))) < 1.0)
    except np.linalg.LinAlgError:
        return (np.nan,) * 4
    steps = [int(round(s / delta)) for s in (180.0, 600.0, 1800.0)]
    r = irf_multi(A, plag, steps)
    if not all(np.isfinite(v) and abs(v) < 1e4 for v in r):
        return (np.nan,) * 4
    return (stat, r[0], r[1], r[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True); ap.add_argument("--ticker", required=True)
    a = ap.parse_args()
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", os.path.basename(a.msg))
    date = int(m.group(1) + m.group(2) + m.group(3)) if m else 0
    try:
        bt, bm, bb, ba, bbsz, basz, ofi, trades = BA.reconstruct(a.msg)
        df = pd.read_csv(a.msg, header=None, usecols=[0, 1, 3, 4], names=["t", "ty", "sz", "px"])
        h = df[df.ty == 5]
        if len(h) < 3 or len(bt) < 50:
            print(NA.format(t=a.ticker, d=date)); return
        ht = h.t.to_numpy(float)
        hpx = h.px.to_numpy(float) / BA.SCALE
        hsz = h.sz.to_numpy(float)
        midh = BA.mid_at(bt, bm, ht)
        ok = np.isfinite(midh) & (midh > 0)
        ht, hpx, hsz, midh = ht[ok], hpx[ok], hsz[ok], midh[ok]
        if len(ht) < 3:
            print(NA.format(t=a.ticker, d=date)); return
        q = np.where(hpx > midh, 1, np.where(hpx < midh, -1, 0))

        # ---------- TEST A: term structure + TOD placebo ----------
        vA = [0] + [np.nan] * 12
        ends, dirs = bursts_from(ht, q)
        if len(ends) > 0:
            e0 = BA.mid_at(bt, bm, ends)
            H = [180., 900., 1800., 3600., 7200.]
            mks = [mk(bt, bm, ends, dirs, e0, dt=dt) for dt in H]
            mkclose = mk(bt, bm, ends, dirs, e0, target_t=RTH1)
            rng = np.random.default_rng((date + abs(hash(a.ticker))) % (2 ** 32))
            buckets = RTH0 + np.floor(np.clip(ends - RTH0, 0, None) / 1800.0) * 1800.0
            rt = np.clip(buckets + rng.uniform(0, 1800.0, len(ends)), RTH0, RTH1 - 1.0)
            p0 = BA.mid_at(bt, bm, rt)
            pmks = [mk(bt, bm, rt, dirs, p0, dt=dt) for dt in H]
            pmkclose = mk(bt, bm, rt, dirs, p0, target_t=RTH1)
            vA = [len(ends)] + mks + [mkclose] + pmks + [pmkclose]

        # ---------- TEST B: VAR specification grid ----------
        sel = (q != 0) & (ht >= RTH0) & (ht < RTH1)
        vB = []
        if sel.sum() >= 20:
            for (delta, plag, ridge) in SPECS:
                vB.extend(run_var(bt, bm, ht, q, hsz, sel, delta, plag, ridge))
        else:
            vB = [np.nan] * 16

        # ---------- TEST C: sequential sweep ----------
        vC = [0] + [np.nan] * 6
        agg = (q != 0) & (ht >= RTH0 + 180.0) & (ht < RTH1 - 1800.0)
        if agg.sum() >= 10:
            ta, qa = ht[agg], q[agg]
            pb0, pa0 = BA.bbo_at(bt, bb, ba, ta - 1e-3)
            bs0, as0 = BA.bbo_at(bt, bbsz, basz, ta - 1e-3)
            out = {}
            for tag, dt in (("100", 0.1), ("1s", 1.0)):
                pb1, pa1 = BA.bbo_at(bt, bb, ba, ta + dt)
                bs1, as1 = BA.bbo_at(bt, bbsz, basz, ta + dt)
                # side being taken: a buy lifts the ask, a sell hits the bid
                p_pre = np.where(qa > 0, pa0, pb0)
                p_post = np.where(qa > 0, pa1, pb1)
                s_pre = np.where(qa > 0, as0, bs0)
                s_post = np.where(qa > 0, as1, bs1)
                moved = np.where(qa > 0, p_post > p_pre, p_post < p_pre)
                shrank = (p_post == p_pre) & (s_post < 0.5 * s_pre)
                good = np.isfinite(p_pre) & np.isfinite(p_post) & np.isfinite(s_pre) \
                    & np.isfinite(s_post) & (s_pre > 0)
                out[tag] = (moved | shrank) & good
                out["d" + tag] = wm(np.where(good, (s_post - s_pre) / np.maximum(s_pre, 1.0) * 100.0, np.nan))
            sw = out["100"] | out["1s"]
            f3 = BA.mid_at(bt, bm, ta + 180.0)
            m0 = BA.mid_at(bt, bm, ta)
            with np.errstate(invalid="ignore", divide="ignore"):
                r3 = qa * (f3 - m0) / m0 * 1e4
            vC = [int(sw.sum()),
                  float(out["100"].mean()), float(out["1s"].mean()),
                  out["d100"], out["d1s"],
                  wm(r3[sw]), wm(r3[~sw])]

        vals = vA + vB + vC
        out = [a.ticker, str(date)]
        for v in vals:
            out.append(str(int(v)) if isinstance(v, (int, np.integer)) else "%.5f" % v)
        print(",".join(out))
    except Exception as e:
        print(f"{a.ticker},{date},ERR,{e}", file=sys.stderr)
        print(NA.format(t=a.ticker, d=date))


if __name__ == "__main__":
    main()
