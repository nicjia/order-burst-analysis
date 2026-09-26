#!/usr/bin/env python3
"""Statistics of program-evidence-v1 modules A, B and I, shared by the aggregator and the tests."""
import numpy as np

import evidence_stats as ES

PHI_LO, PHI_HI = ES.PHI_MS[:-1], ES.PHI_MS[1:]


def locked_mask(delta_ms):
    return (PHI_LO >= -delta_ms) & (PHI_HI <= delta_ms)


def concentration(counts, delta_ms=10):
    """counts [k, phi] summed over days: share of pairs with |phi| < delta, over the uniform share."""
    c = np.asarray(counts, float)
    tot = c.sum()
    if tot <= 0:
        return np.nan
    return (c[..., locked_mask(delta_ms)].sum() / tot) / (2 * delta_ms / 1000.0)


def phase_ratios(z, delta_ms=10):
    """A1 (within vs cross-day), A2 (identical vs different size), all-signed A1, window pair counts."""
    w_all = z["a_w_all"].sum(0); w_id = z["a_w_id"].sum(0); c_all = z["a_c_all"].sum(0)
    w_sig = z["a_w_sig"].sum(0); c_sig = z["a_c_sig"].sum(0)
    return dict(
        a1=concentration(w_all, delta_ms) / concentration(c_all, delta_ms),
        a2=concentration(w_id, delta_ms) / concentration(w_all - w_id, delta_ms),
        a1_signed=concentration(w_sig, delta_ms) / concentration(c_sig, delta_ms),
        a1_opposite=concentration(z["a_w_opp"].sum(0), delta_ms) / concentration(c_all, delta_ms),
        a1_crossday_identical=concentration(z["a_c_id"].sum(0), delta_ms) / concentration(c_all, delta_ms),
        window_pairs=float(w_all.sum()), identical_window_pairs=float(w_id.sum()),
    )


def burst_phase_j(z, delta_ms=10):
    """Per definition [rule, gap]: phase TPR, FPR, expected and excess locked pairs (primary delta only)."""
    if delta_ms != int(ES.DELTA * 1000):
        raise ValueError("within-burst counts exist only for the primary delta")
    c_cross = concentration(z["a_c_all"].sum(0), delta_ms)
    base = 2 * delta_ms / 1000.0 * c_cross
    w_all = z["a_w_all"].sum(0)
    locked_all = w_all[:, locked_mask(delta_ms)].sum()
    window_all = w_all.sum()
    x_all = locked_all - window_all * base
    b = z["a_b"][:, :, :, 0].sum(0)                      # [rule, gap, k, 3]
    locked_in = b[..., 1].sum(-1); window_in = b.sum((-1, -2))
    x_in = locked_in - window_in * base
    tpr = np.where(x_all > 0, np.clip(x_in, 0, None) / x_all, np.nan)
    fpr = window_in / window_all if window_all else np.full(window_in.shape, np.nan)
    return dict(tpr=tpr, fpr=fpr, j=tpr - fpr, expected_locked=window_all * base, excess_locked=x_all)


def burst_size_j(z):
    """Size-fingerprint TPR/FPR/J over lags [0.5, 60.5) s with the depth-matched cross-day null."""
    cp, cm = z["a_c_dm"].sum(0)
    rate = cm / cp if cp else np.nan
    wp, wm = z["a_w_dm"].sum(0)
    x_all = wm - wp * rate
    bp = z["a_b_dm"][..., 0].sum(0); bm = z["a_b_dm"][..., 1].sum(0)
    x_in = bm - bp * rate
    tpr = np.where(x_all > 0, np.clip(x_in, 0, None) / x_all, np.nan)
    fpr = bp / wp if wp else np.full(bp.shape, np.nan)
    return dict(tpr=tpr, fpr=fpr, j=tpr - fpr, expected=wp * rate, excess=x_all)


def side_observed_expected(z, cls=0, matched=True):
    """Observed and expected identical-size matches [relation(same, opp), lag bin], lag-specific rates."""
    suf = "_dm" if matched else ""
    wp = z["b_w_pairs" + suf][:, cls].sum(0).astype(float)
    wm = z["b_w_match" + suf][:, cls].sum(0).astype(float)
    cp = z["b_c_pairs" + suf][:, cls].sum(0).astype(float)
    cm = z["b_c_match" + suf][:, cls].sum(0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(cp > 0, cm / cp, np.nan)
    return wm, wp * rate


def lag_range_mask(lo, hi):
    return (ES.B_EDGES[:-1] >= lo) & (ES.B_EDGES[1:] <= hi)


def side_ratio(obs, exp, lo, hi):
    m = lag_range_mask(lo, hi)
    o = obs[..., m].sum(-1); e = np.nansum(exp[..., m], -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(e > 0, o / e, np.nan)


def multiday_did(z, lo=0.0, hi=3600.0):
    """(adjacent/distant same-side rate) / (adjacent/distant opposite-side rate), u_nonround, matched."""
    m = lag_range_mask(lo, hi)
    ap = z["b_c_pairs_dm"][:, 0][..., m].sum((0, -1)).astype(float)   # [relation]
    am = z["b_c_match_dm"][:, 0][..., m].sum((0, -1)).astype(float)
    dp = z["b_d_pairs_dm"][..., m].sum((0, -1)).astype(float)
    dm = z["b_d_match_dm"][..., m].sum((0, -1)).astype(float)
    if np.any(ap == 0) or np.any(dp == 0) or np.any(am == 0) or np.any(dm == 0):
        return np.nan, dict(adj_match=am, adj_pairs=ap, dist_match=dm, dist_pairs=dp)
    adj = am / ap; dist = dm / dp
    return float((adj[0] / dist[0]) / (adj[1] / dist[1])), dict(adj_match=am, adj_pairs=ap, dist_match=dm, dist_pairs=dp)


def dollar_d(counts):
    """counts [.., e-bin, dm(up, down, zero), ds(neg, pos)] -> D per e-bin."""
    c = np.asarray(counts, float)
    up = c[..., 0, :]; down = c[..., 1, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        p_up = up[..., 0] / up.sum(-1)
        p_down = down[..., 0] / down.sum(-1)
    return p_up - p_down
