#!/usr/bin/env python3
"""agg_final.py — aggregate results/hid_fin into the three remaining referee tables."""
import glob
import numpy as np, pandas as pd


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


def stat(df, col, cap=1000.0):
    d = df[["date", col]].copy()
    d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d[np.isfinite(d[col]) & (d[col].abs() <= cap)]
    if len(d) < 100:
        return np.nan, np.nan, 0
    daily = d.groupby("date")[col].mean()
    return float(daily.mean()), nw_t(daily.to_numpy()), int(len(d))


def stat_series(df, s, cap=1000.0):
    d = pd.DataFrame({"date": df["date"], "v": pd.to_numeric(s, errors="coerce")})
    d = d[np.isfinite(d.v) & (d.v.abs() <= cap)]
    if len(d) < 100:
        return np.nan, np.nan, 0
    daily = d.groupby("date")["v"].mean()
    return float(daily.mean()), nw_t(daily.to_numpy()), int(len(d))


def main():
    fr = []
    for f in sorted(glob.glob("results/hid_fin/out/*.csv")):
        try:
            fr.append(pd.read_csv(f, on_bad_lines="skip"))
        except Exception:
            pass
    df = pd.concat(fr, ignore_index=True)
    df = df[df["date"].notna()]
    for c in df.columns[2:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = df["date"].astype(int)
    df.to_csv("results/hid_fin/all.csv", index=False)
    print("panel: %d name-days, %d names, %d dates\n" % (len(df), df.ticker.nunique(), df.date.nunique()))

    print("=" * 84)
    print("TEST A — FULL-UNIVERSE TERM STRUCTURE, TOD-STRATIFIED PLACEBO")
    print("=" * 84)
    print("  %-9s %10s %10s %10s %9s %9s" % ("horizon", "burst", "placebo", "net", "net t", "n"))
    for lab, b, p in [("3 min", "mk3", "pmk3"), ("15 min", "mk15", "pmk15"),
                      ("30 min", "mk30", "pmk30"), ("1 hour", "mk60", "pmk60"),
                      ("2 hour", "mk120", "pmk120"), ("to close", "mkclose", "pmkclose")]:
        bm, _, _ = stat(df, b); pm, _, _ = stat(df, p)
        nm, nt, nn = stat_series(df, df[b] - df[p])
        print("  %-9s %10.3f %10.3f %10.3f %9.2f %9d" % (lab, bm, pm, nm, nt, nn))

    print("\n" + "=" * 84)
    print("TEST B — VAR SPECIFICATION GRID (cumulative IRF to a 1-SD flow innovation, bps)")
    print("=" * 84)
    names = {"s1": "10s clock, 12 lags, no ridge  (original spec)",
             "s2": "10s clock, 12 lags, ridge",
             "s3": "60s clock, 30 lags, no ridge",
             "s4": "60s clock, 30 lags, ridge     (current spec)"}
    for cap in (1e4, 50.0):
        print("\n  [outlier cap |IRF| <= %g]" % cap)
        print("  %-46s %9s %9s %9s %7s" % ("specification", "3 min", "10 min", "30 min", "%stat"))
        for s in ("s1", "s2", "s3", "s4"):
            r = []
            for h in ("3", "10", "30"):
                m, t, _ = stat(df, "%s_%s" % (s, h), cap=cap)
                r.append((m, t))
            ps = df["%s_stat" % s].mean() * 100
            print("  %-46s %9.3f %9.3f %9.3f %6.0f%%" % (names[s], r[0][0], r[1][0], r[2][0], ps))
            print("  %-46s %9s %9s %9s" % ("", "(t=%.1f)" % r[0][1], "(t=%.1f)" % r[1][1], "(t=%.1f)" % r[2][1]))

    print("\n" + "=" * 84)
    print("TEST C — SEQUENTIAL SWEEP: DOES DISPLAYED DEPTH FALL AFTER THE HIDDEN PRINT?")
    print("=" * 84)
    for lab, c in [("frac. swept within 100ms", "fsw100"), ("frac. swept within 1s", "fsw1s"),
                   ("mean depth change @100ms (%)", "dsz100"), ("mean depth change @1s (%)", "dsz1s")]:
        m, t, n = stat(df, c, cap=1e6)
        print("  %-32s %9.3f  (t=%6.1f)  n=%d" % (lab, m, t, n))
    print()
    for lab, c in [("3-min markout, SWEPT prints", "mk3_sw"),
                   ("3-min markout, UNSWEPT prints", "mk3_unsw")]:
        m, t, n = stat(df, c)
        print("  %-32s %9.3f  (t=%6.1f)  n=%d" % (lab, m, t, n))


if __name__ == "__main__":
    main()
