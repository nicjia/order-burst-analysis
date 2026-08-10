#!/usr/bin/env python3
"""agg_tickplacebo.py — aggregate results/hid_tp into the two referee tables."""
import glob, sys
import numpy as np, pandas as pd

COLS = ["ticker","date","n_agg","n_mid","n_far","mk3_agg_qd","mk3_agg_tick",
        "n_mb","mk3_mid","mk15_mid","mk30_mid","n_pb","mk3_plc","mk15_plc","mk30_plc",
        "dpre","m3","m10","m30","a3","a10","a30","e3","e10","e30",
        "dpreA","m3A","m10A","m30A","a3A","a10A","a30A","e3A","e10A","e30A"]


def nw_t(x, L=10):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        return np.nan
    e = x - x.mean()
    v = (e * e).sum() / n
    for l in range(1, L + 1):
        v += 2.0 * (1 - l / (L + 1.0)) * (e[l:] * e[:-l]).sum() / n
    se = np.sqrt(v / n)
    return float(x.mean() / se) if se > 0 else np.nan


def stat(df, col):
    """Equal-weighted daily mean across names, then NW(10) t on the daily series."""
    d = df[["date", col]].copy()
    d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d[np.isfinite(d[col]) & (d[col].abs() <= 1000)]
    if len(d) < 100:
        return np.nan, np.nan, 0
    daily = d.groupby("date")[col].mean()
    return float(daily.mean()), nw_t(daily.to_numpy()), int(len(d))


def main():
    fs = sorted(glob.glob("results/hid_tp/out/*.csv"))
    fr = []
    for f in fs:
        try:
            fr.append(pd.read_csv(f, on_bad_lines="skip"))
        except Exception:
            pass
    df = pd.concat(fr, ignore_index=True)
    df = df[df.get("date").notna()]
    for c in COLS[2:]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].astype(int)
    df.to_csv("results/hid_tp/all.csv", index=False)
    print("panel: %d name-days, %d names, %d dates\n" % (len(df), df.ticker.nunique(), df.date.nunique()))

    def row(label, col):
        m, t, n = stat(df, col)
        print("  %-42s %8.3f  (t=%7.2f)  n=%d" % (label, m, t, n))

    print("=" * 78)
    print("ITEM 1a — AT-MIDPOINT LEG vs MATCHED TICK-RULE PLACEBO")
    print("=" * 78)
    for h, mc, pc in [("3 min", "mk3_mid", "mk3_plc"),
                      ("15 min", "mk15_mid", "mk15_plc"),
                      ("30 min", "mk30_mid", "mk30_plc")]:
        mm, mt, mn = stat(df, mc); pm, pt, pn = stat(df, pc)
        share = (pm / mm * 100) if (np.isfinite(mm) and abs(mm) > 1e-9) else np.nan
        print("  %-7s  at-mid %7.3f (t=%7.2f)   placebo %7.3f (t=%7.2f)   "
              "placebo/at-mid = %6.1f%%" % (h, mm, mt, pm, pt, share))

    print("\n" + "=" * 78)
    print("ITEM 1b — AGGRESSIVE LEG: QUOTE SIGN vs PURE TICK SIGN (3 min)")
    print("=" * 78)
    row("aggressive, quote-rule signed", "mk3_agg_qd")
    row("aggressive, PURE tick-rule signed", "mk3_agg_tick")

    print("\n" + "=" * 78)
    print("ITEM 4 — MARKOUT DECOMPOSITION   m_h = a_h + b_h*mean(d)")
    print("=" * 78)
    for tag, suf in [("OUTSIDE-THE-QUOTE prints", ""), ("ALL AGGRESSIVE prints", "A")]:
        print("\n  %s" % tag)
        dm, dt_, dn = stat(df, "dpre" + suf)
        print("    mean pre-print drift (-30s)            %8.3f  (t=%7.2f)  n=%d" % (dm, dt_, dn))
        print("    %-10s %10s %10s %10s" % ("horizon", "markout m", "orthog. a", "continu. e"))
        for h, s in [("3 min", "3"), ("10 min", "10"), ("30 min", "30")]:
            mm, mt, _ = stat(df, "m" + s + suf)
            am, at, _ = stat(df, "a" + s + suf)
            em, et, _ = stat(df, "e" + s + suf)
            print("    %-10s %8.3f   %8.3f   %8.3f      (t: m=%.1f a=%.1f e=%.1f)  "
                  "a+e=%.3f" % (h, mm, am, em, mt, at, et, am + em))

    print("\n" + "=" * 78)
    print("COUNTS")
    print("=" * 78)
    for c in ["n_agg", "n_mid", "n_far", "n_mb", "n_pb"]:
        print("  %-8s median/day %8.0f   total %14.0f" % (c, df[c].median(), df[c].sum()))


if __name__ == "__main__":
    main()
