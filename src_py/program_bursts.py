#!/usr/bin/env python3
"""Run/60 bursts and the stage-3b program score on one day of cached packets (vectorized).

Features reproduce fingerprint_burst_rows.burst_rows for the run rule exactly (tested), transformed
as in program_score.load, and scored with results/program_evidence_v1/program_model_run60.json.
Run bursts contain no opposite-side or unsigned packets, so those two features are zero.
"""
import json
from pathlib import Path

import numpy as np

import fingerprint_stats as FS

MODEL_PATH = Path(__file__).resolve().parents[1] / "results" / "program_evidence_v1" / "program_model_run60.json"


def load_model(path=MODEL_PATH):
    m = json.loads(Path(path).read_text())
    m["mu"] = np.asarray(m["mu"]); m["sd"] = np.asarray(m["sd"]); m["beta"] = np.asarray(m["beta"])
    return m


def run_bursts(day, gap=60.0, min_packets=3):
    """Per-packet burst index (-1 if none) and a dict of per-burst arrays for run bursts."""
    t = np.asarray(day["time"], float); sign = np.asarray(day["sign"], int); n = len(t)
    ids, _ = FS.burst_ids(t, sign, gap, "run")
    if n == 0:
        return np.full(0, -1), {}
    counts = np.bincount(ids)
    first = np.r_[0, np.cumsum(counts)[:-1]]                 # run ids are contiguous in time
    burst_sign = sign[first]
    keep = (counts >= min_packets) & (burst_sign != 0)
    kept = np.flatnonzero(keep)
    remap = np.full(len(counts), -1); remap[kept] = np.arange(len(kept))
    member = remap[ids]
    if not len(kept):
        return member, {}
    f = first[kept]; c = counts[kept]; last = f + c - 1
    tt0, tt1 = t[f], t[last]
    dur = tt1 - tt0
    # inter-arrival statistics per burst
    gaps = np.diff(t)
    same = ids[1:] == ids[:-1]
    gid = ids[1:][same]; gval = gaps[same]
    k_of = remap[gid]; ok = k_of >= 0; k_of = k_of[ok]; gval = gval[ok]
    ng = np.bincount(k_of, minlength=len(kept)).astype(float)
    gsum = np.bincount(k_of, weights=gval, minlength=len(kept))
    gsq = np.bincount(k_of, weights=gval ** 2, minlength=len(kept))
    gmean = np.where(ng > 0, gsum / np.maximum(ng, 1), 0.0)
    gstd = np.sqrt(np.maximum(np.where(ng > 0, gsq / np.maximum(ng, 1) - gmean ** 2, 0.0), 0.0))
    iat_cv = np.where((ng > 1) & (gmean > 0), gstd / np.where(gmean > 0, gmean, 1.0), 0.0)
    # median gap per burst
    order = np.lexsort((gval, k_of))
    iat_median = np.zeros(len(kept))
    if len(order):
        ks = k_of[order]; vs = gval[order]
        starts = np.searchsorted(ks, np.arange(len(kept)), side="left")
        ends = np.searchsorted(ks, np.arange(len(kept)), side="right")
        m = ends > starts
        lo = starts + (ends - starts - 1) // 2; hi = starts + (ends - starts) // 2
        iat_median[m] = 0.5 * (vs[lo[m]] + vs[hi[m]])
    unt = np.asarray(day["untruncated"], bool)
    k_packet = member[member >= 0]
    trunc_share = np.bincount(k_packet, weights=(~unt[member >= 0]).astype(float), minlength=len(kept)) / c
    hid = np.asarray(day["hidden_share"], float)[member >= 0]
    hid_ok = np.isfinite(hid)
    hid_share = np.bincount(k_packet[hid_ok], weights=hid[hid_ok], minlength=len(kept)) / np.maximum(
        np.bincount(k_packet[hid_ok], minlength=len(kept)), 1)
    hid_share = np.where(np.bincount(k_packet[hid_ok], minlength=len(kept)) > 0, hid_share, np.nan)
    act = np.searchsorted(t, t, side="left") - np.searchsorted(t, t - 300.0, side="left")
    depth0 = np.asarray(day["exec_depth"], float)[f]
    vol = np.asarray(day["volume"], float)
    table = dict(
        first=f, last=last, side=burst_sign[kept], start=tt0, end=tt1, n_packets=c.astype(float),
        volume=np.bincount(k_packet, weights=vol[member >= 0], minlength=len(kept)),
        duration=dur, intensity=c / np.maximum(dur, 1e-3), iat_cv=iat_cv, iat_median=iat_median,
        truncated_share=trunc_share, hidden_share=hid_share,
        spread_bps=np.asarray(day["spread_bps"], float)[f],
        log_exec_depth=np.where(np.isfinite(depth0), np.log(np.maximum(depth0, 1)), np.nan),
        imbalance=np.asarray(day["imbalance"], float)[f], tod=(tt0 - FS.RTH0) / 23400.0,
        trailing_activity=act[f].astype(float),
    )
    return member, table


def features(table):
    tb = table
    X = {
        "log_packets": np.log1p(tb["n_packets"]), "log_duration": np.log1p(tb["duration"]),
        "log_intensity": np.log(np.clip(tb["intensity"], 1e-6, None)), "iat_cv": tb["iat_cv"],
        "log_iat_median": np.log1p(tb["iat_median"]), "truncated_share": tb["truncated_share"],
        "hidden_share": tb["hidden_share"], "log_spread": np.log(np.clip(tb["spread_bps"], 1e-3, None)),
        "log_exec_depth": tb["log_exec_depth"], "imbalance": tb["imbalance"], "tod": tb["tod"],
        "tod2": tb["tod"] ** 2, "log_activity": np.log1p(tb["trailing_activity"]),
        "opposite_share": np.zeros(len(tb["tod"])), "unsigned_share": np.zeros(len(tb["tod"])),
    }
    return X


def score(table, model):
    if not table:
        return np.zeros(0)
    X = features(table)
    M = np.column_stack([X[k] for k in model["features"]])
    Z = np.column_stack([np.ones(len(M)), (M - model["mu"]) / model["sd"]])
    return Z @ model["beta"]
