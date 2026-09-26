import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src_py"))
sys.path.insert(0, os.path.dirname(__file__))
import fingerprint_burst_rows as FBR  # noqa: E402
import fingerprint_stats as FS  # noqa: E402
import metaorder_features as MF  # noqa: E402
from test_evidence import make_day  # noqa: E402


def noisy_day(seed=0, n=3000, hours=2.0):
    rng = np.random.default_rng(seed)
    day = make_day(rng, n=n, hours=hours, max_size=300,
                   program=dict(starts=[300, 2500], children=80, period=3.0, size=137, side=1))
    m = len(day["time"])
    day["untruncated"] = rng.random(m) < 0.7
    day["hidden_share"] = np.where(rng.random(m) < 0.2, rng.random(m), 0.0)
    day["spread_bps"] = rng.uniform(1, 5, m); day["exec_depth"] = rng.uniform(50, 5000, m)
    day["imbalance"] = rng.uniform(-1, 1, m)
    day["sign"] = np.where(rng.random(m) < 0.05, 0, day["sign"]).astype(np.int8)
    return day


class BurstTableTests(unittest.TestCase):
    def test_run_features_match_fingerprint_burst_rows(self):
        day = noisy_day(1)
        n = len(day["time"])
        rows = FBR.burst_rows(day, "20240101", "SYN", "run", 60.0, np.zeros(n, np.int64), {}, None)
        member, tab = MF.burst_table(day, "run", 60.0)
        self.assertEqual(len(rows), len(tab["start"]))
        order = np.argsort(tab["start"], kind="stable")
        rows = sorted(rows, key=lambda r: r["start"])
        for k in ("n_packets", "duration", "intensity", "iat_cv", "iat_median", "truncated_share", "hidden_share",
                  "spread_bps", "log_exec_depth", "imbalance", "tod", "trailing_activity"):
            np.testing.assert_allclose(tab[k][order], [r[k] for r in rows], rtol=1e-9, atol=1e-12, err_msg=k)

    def test_stream_features_match_fingerprint_burst_rows(self):
        day = noisy_day(2)
        n = len(day["time"])
        rows = FBR.burst_rows(day, "20240101", "SYN", "stream", 5.0, np.zeros(n, np.int64), {}, None)
        member, tab = MF.burst_table(day, "stream", 5.0)
        self.assertEqual(len(rows), len(tab["start"]))
        order = np.argsort(tab["start"] + 1e-6 * tab["side"], kind="stable")
        rows = sorted(rows, key=lambda r: r["start"] + 1e-6 * r["side"])
        for k in ("n_packets", "duration", "iat_cv", "iat_median", "truncated_share", "spread_bps", "tod"):
            np.testing.assert_allclose(tab[k][order], [r[k] for r in rows], rtol=1e-9, atol=1e-12, err_msg=k)

    def test_prefix_and_context(self):
        t0 = FS.RTH0
        day = dict(time=t0 + np.array([0, 10, 11, 13, 20, 21, 22.5, 100.0]), sign=np.array([1, -1, -1, -1, 1, 1, 1, -1], np.int8),
                   volume=np.array([5, 10, 10, 10, 7, 7, 7, 3.0]), untruncated=np.array([1, 1, 0, 1, 1, 1, 1, 1], bool),
                   mid=np.full(8, 10.0), spread_bps=np.full(8, 2.0), exec_depth=np.full(8, 100.0), imbalance=np.zeros(8),
                   hidden_share=np.zeros(8), n_messages=np.ones(8, np.int32))
        member, tab = MF.burst_table(day, "run", 60.0)
        self.assertEqual(list(tab["side"]), [-1, 1])
        np.testing.assert_allclose(tab["pre_duration"], [3.0, 2.5])
        np.testing.assert_allclose(tab["pre_truncated_share"], [1 / 3, 0.0])
        # second burst (buys starting at t=20): trailing 300 s signed volume = 5 + 30 = 35; opposite in-burst = 30
        np.testing.assert_allclose(tab["ctx_opp_share_300"][1], 30 / 35)
        np.testing.assert_allclose(tab["ctx_same_share_300"][1], 0.0)


class EvidenceTests(unittest.TestCase):
    def brute(self, day, member, tab, dbin, rates):
        t = day["time"]; sign = day["sign"]; size, masks = FS.size_class_masks(day, np.array([], np.int64))
        u = masks["u_nonround"] & (member >= 0)
        K = len(tab["side"])
        within = np.zeros((K, 3)); link = np.zeros((K, 2, 3))
        idx = np.flatnonzero(u)
        for i in idx:
            for j in idx:
                if i == j:
                    continue
                k = member[i]
                if member[j] == k and t[j] > t[i] or (member[j] == k and t[j] == t[i] and j > i):
                    if dbin[i] == dbin[j] and t[j] - t[i] < 300:
                        within[k, 0] += 1; within[k, 1] += size[i] == size[j]
                        within[k, 2] += rates[(0, int(tab["side"][k]), int(dbin[i]))]
                if member[j] != k and tab["end"][k] < t[j] <= tab["end"][k] + 1800 and dbin[i] == dbin[j]:
                    s = int(sign[i])
                    rel = 0 if sign[j] == sign[i] else 1
                    link[k, rel, 0] += 1; link[k, rel, 1] += size[i] == size[j]; link[k, rel, 2] += rates[(rel, s, int(dbin[i]))]
        return within, link

    def test_within_and_link_evidence_equal_brute_force(self):
        rng = np.random.default_rng(5)
        day = noisy_day(3, n=500, hours=1.0)
        day["volume"] = rng.integers(1, 8, len(day["time"])).astype(float)
        thr = np.quantile(day["exec_depth"], FS.DEPTH_QUANTILES)
        dbin = FS.depth_bins(day["exec_depth"], thr)
        rates = {(rel, s, q): 0.01 * (1 + rel) * (1 + q) * (1.5 if s > 0 else 1.0) for rel in (0, 1) for s in (1, -1) for q in range(4)}
        for rule, gap in (("run", 60.0), ("stream", 5.0)):
            member, tab = MF.burst_table(day, rule, gap)
            wp, wr, we = MF.within_evidence(day, member, tab, dbin, rates)
            link = MF.link_evidence(day, member, tab, dbin, rates)
            bw, bl = self.brute(day, member, tab, dbin, rates)
            np.testing.assert_allclose(np.column_stack([wp, wr, we]), bw, err_msg=rule)
            np.testing.assert_allclose(link, bl, err_msg=rule)


if __name__ == "__main__":
    unittest.main()
