#!/usr/bin/env python3
"""burst-defs-raw-v1 (cluster, per name-day): seven burst definitions from one LOBSTER download, with order-book
features, all measured on the same economic packets (native signs) and the same mid path as p4-revisit-v1.

Definitions: run1 / run5 / run60 / run300 (same-side runs, a sign change or unsigned packet ends a run, gap cap
1/5/60/300 s, >= 3 packets); stream2 / stream5 (each side's own sequence, gap cap 2/5 s, >= 3 packets);
hawkes (the original C++ detector: intensity clustering beta 1, trigger 0.3, direction by count share >= 0.763 and
minority volume <= 0.28 x majority, volume >= 0.00197 x the day's volume) on correctly signed packets.

Per burst, known at the decision time T_dec = max(t_b + 600 s, t_e + 10 s): size, children, duration, fingerprint
(modal untruncated clip count), spreads, peak impact and displacement path (p4 definitions), move since the open,
30-min pre-move, time of day, and ORDER-BOOK FEATURES: touch queue imbalance at the first packet and averaged
over the burst (signed by side), trade-flow imbalance of all packets in the 60 s before t_b, and quote order-flow
imbalance (Cont-Kukanov-Stoikov) over the 60 s before t_b and during [t_b, t_e + 10 s), normalized by mean touch
depth, signed by side. Outcomes (NOT features): d30 = mid move 30 min after T_dec, d_close = to the last mid
before 16:00, both signed, bps. Up to CAP bursts per definition per name-day, sampled at random (seeded).
Usage: burst_defs_raw.py --msg FILE --ticker TK --out OUT.csv.gz [--helper p4_bbo]
"""
import sys as _sys, pathlib as _pl  # shared project modules live in src_py/
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "src_py"))
import argparse, re, zlib
from pathlib import Path
import numpy as np
import pandas as pd
import p4_extract as X
import p4_packets as PK
import fingerprint_packets as FP
import fingerprint_stats as FS
import p4_q0_legacy as LG

CAP = 12
try:
    import p4_aggregate as AG
    EARLY = set(AG.EARLY_CLOSE)
except Exception:
    EARLY = set()


def groups_from_ids(ids, t, sign, vol, minsize=3):
    counts = np.bincount(ids)
    first = np.full(len(counts), -1); last = np.full(len(counts), -1)
    order = np.arange(len(ids))
    first_idx = pd.Series(order).groupby(ids).min().to_numpy(); last_idx = pd.Series(order).groupby(ids).max().to_numpy()
    uniq = np.unique(ids)
    first[uniq] = first_idx; last[uniq] = last_idx
    s = sign[np.maximum(first, 0)]
    keep = np.flatnonzero((counts >= minsize) & (s != 0))
    member = np.full(len(ids), -1); remap = np.full(len(counts), -1); remap[keep] = np.arange(len(keep))
    member = remap[ids]
    v = np.bincount(member[member >= 0], weights=vol[member >= 0], minlength=len(keep))
    return dict(t_b=t[first[keep]], t_e=t[last[keep]], side=s[keep].astype(float), n=counts[keep].astype(float), vol=v), member


def ofi_cumulative(bt, bb, ba, qb, qa):
    """Cont-Kukanov-Stoikov order-flow imbalance, cumulative over the BBO path."""
    e = np.zeros(len(bt))
    if len(bt) > 1:
        b0, b1, a0, a1 = bb[:-1], bb[1:], ba[:-1], ba[1:]
        q0b, q1b, q0a, q1a = qb[:-1], qb[1:], qa[:-1], qa[1:]
        e[1:] = (np.where(b1 >= b0, q1b, 0) - np.where(b1 <= b0, q0b, 0)
                 - np.where(a1 <= a0, q1a, 0) + np.where(a1 >= a0, q0a, 0))
    e = np.nan_to_num(e)
    return np.cumsum(e)


def run(msg_path, ticker, date, helper):
    if date in EARLY:
        return pd.DataFrame()
    msg = X.read_messages(msg_path)
    context, bid_i, ask_i, _ = X.bbo_context(msg_path, msg, helper)
    mid = X.MidPath(context[0], context[1], context[2], context[3])
    bt, bb, ba, qb, qa = context[0], context[2], context[3], context[4], context[5]
    cum_ofi = ofi_cumulative(bt, bb, ba, qb, qa)
    depth = np.nanmean(np.where(qb + qa > 0, (qb + qa) / 2, np.nan)) if len(qb) else np.nan
    packets = PK.fast_packets(msg, context)
    if not len(packets):
        return pd.DataFrame()
    day = FP.packet_arrays(packets)
    t, sign, vol = day["time"], day["sign"].astype(int), day["volume"]
    rth = (t >= X.RTH0) & (t < X.RTH1)
    t, sign, vol = t[rth], sign[rth], vol[rth]
    imb = day["imbalance"][rth]; unt = day["untruncated"][rth]
    if len(t) < 10:
        return pd.DataFrame()
    csv = np.cumsum(sign * vol); cvol = np.cumsum(vol)

    def window_sum(c, a, b):
        ia = np.searchsorted(t, a, "left"); ib = np.searchsorted(t, b, "left")
        ca = np.where(ia > 0, c[np.maximum(ia - 1, 0)], 0.0); cb = np.where(ib > 0, c[np.maximum(ib - 1, 0)], 0.0)
        return cb - ca

    def ofi_window(a, b):
        ia = np.searchsorted(bt, a, "left") - 1; ib = np.searchsorted(bt, b, "left") - 1
        ca = np.where(ia >= 0, cum_ofi[np.maximum(ia, 0)], 0.0); cb = np.where(ib >= 0, cum_ofi[np.maximum(ib, 0)], 0.0)
        return (cb - ca) / depth if depth and np.isfinite(depth) else np.full(len(a), np.nan)

    defs = {}
    for g in (1.0, 5.0, 60.0, 300.0):
        ids, _ = FS.burst_ids(t, sign, g, "run")
        defs["run%d" % g] = groups_from_ids(ids, t, sign, vol)
    for g in (2.0, 5.0):
        ids, _ = FS.burst_ids(t, sign, g, "stream")
        defs["stream%d" % g] = groups_from_ids(ids, t, sign, vol)
    sg = sign != 0
    if sg.sum() > 10:
        hb = LG.legacy_bursts(t[sg], sign[sg] > 0, vol[sg], LG.VOL_FRAC * vol.sum())
        if hb and len(hb["start"]):
            k = hb["direction"] != 0
            hid = LG.hawkes_ids(t[sg])
            defs["hawkes"] = (dict(t_b=hb["start"][k], t_e=hb["end"][k], side=hb["direction"][k].astype(float),
                                   n=np.full(k.sum(), np.nan), vol=hb["vol"][k]), None)
    m_open = mid.at(np.array([X.RTH0 + 60.0]))[0]
    m_close = mid.at(np.array([X.RTH1]))[0]
    rng = np.random.default_rng(zlib.crc32(("%s|%s" % (ticker, date)).encode()))
    out = []
    for dn, (b, member) in defs.items():
        nb = len(b["t_b"])
        if nb == 0:
            continue
        b = {k: np.asarray(v, float) for k, v in b.items()}
        b["t_dec"] = np.maximum(b["t_b"] + X.DECISION_LAG, b["t_e"] + X.PEAK_TAIL)
        ok = np.flatnonzero(b["t_dec"] <= X.RTH1 - 60.0)
        if not len(ok):
            continue
        pick = np.sort(rng.choice(ok, size=min(CAP, len(ok)), replace=False))
        s = {k: v[pick] for k, v in b.items()}
        X.measure(s, mid, (ticker, date, dn))
        side, mref, mdec = s["side"], s["m_ref"], s["m_dec"]
        if member is not None:
            mode, mcount = X.modal_size(np.where(unt, member, -1), vol, nb)
            mc = mcount[pick]
            bi = pick
            imb_mean = np.array([np.nanmean(imb[member == j]) if (member == j).any() else np.nan for j in bi])
            first_imb = np.array([imb[np.flatnonzero(member == j)[0]] if (member == j).any() else np.nan for j in bi])
        else:
            mc = np.full(len(pick), np.nan); imb_mean = np.full(len(pick), np.nan)
            i0 = np.searchsorted(t, s["t_b"], "left"); first_imb = np.where(i0 < len(t), side * np.sign(sign[np.minimum(i0, len(t) - 1)]) * imb[np.minimum(i0, len(t) - 1)], np.nan)
        tv = window_sum(cvol, s["t_b"] - 60, s["t_b"])
        with np.errstate(invalid="ignore", divide="ignore"):
            frame = pd.DataFrame(dict(
                defn=dn, date=date, ticker=ticker, side=side, t_b=s["t_b"], t_e=s["t_e"], n=s["n"], vol=s["vol"], mode_count=mc,
                spread_b=s["spread_b"], spread_dec=s["spread_dec"], peak_bps=s["peak_raw"] / mref * 1e4,
                d60_bps=s["d60"] / mref * 1e4, d600_bps=s["d600"] / mref * 1e4, dmean_bps=s["dmean"] / mref * 1e4,
                ratio=s["dmean"] / s["peak_raw"], imb_first=first_imb, imb_mean=imb_mean,
                tfi60=side * window_sum(csv, s["t_b"] - 60, s["t_b"]) / np.where(tv > 0, tv, np.nan),
                qofi60=side * ofi_window(s["t_b"] - 60, s["t_b"]), qofi_burst=side * ofi_window(s["t_b"], s["t_e"] + X.PEAK_TAIL),
                own_open_bps=side * (mdec - m_open) / m_open * 1e4, pre30_bps=side * (mref - s["m_pre30"]) / s["m_pre30"] * 1e4,
                tod=(s["t_b"] - X.RTH0) / 23400.0, t_dec=s["t_dec"],
                d30=side * (mid.at(np.minimum(s["t_dec"] + 1800, X.RTH1)) - mdec) / mdec * 1e4,
                d_close=side * (m_close - mdec) / mdec * 1e4))
        out.append(frame)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", required=True); ap.add_argument("--ticker", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--helper", default=None)
    a = ap.parse_args()
    date = "".join(re.search(r"(\d{4})-(\d{2})-(\d{2})", Path(a.msg).name).groups())
    df = run(a.msg, a.ticker, date, a.helper)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(len(df))


if __name__ == "__main__":
    main()
