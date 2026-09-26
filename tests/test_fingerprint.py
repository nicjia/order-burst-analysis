import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src_py"))
import fingerprint_stats as FS  # noqa: E402


def brute_pairs(t, edges, key=None, group=None, excluded=None):
    n = len(t)
    order = sorted(range(n), key=lambda i: (t[i], i))
    out = np.zeros(len(edges) - 1, int)
    for a in range(n):
        for b in range(a + 1, n):
            i, j = order[a], order[b]
            if key is not None and key[i] != key[j]:
                continue
            if group is not None and group[i] != group[j]:
                continue
            if excluded is not None and (excluded[i] or excluded[j]):
                continue
            lag = t[j] - t[i]
            k = np.searchsorted(edges, lag, side="right") - 1
            if 0 <= k < len(out) and lag < edges[-1]:
                out[k] += 1
    return out


class PairCountTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.t = np.sort(34200 + rng.uniform(0, 400, 160))
        self.t[10] = self.t[9]  # a timestamp tie
        self.key = rng.choice([7, 13, 250, 999], len(self.t))
        self.sign = rng.choice([-1, 0, 1], len(self.t), p=[.45, .1, .45])
        self.edges = np.array([0, .1, .5, 2, 10, 30, 120, 300.0])

    def test_totals_and_matches_equal_brute_force(self):
        np.testing.assert_array_equal(FS.pair_counts(self.t, self.edges),
                                      brute_pairs(self.t, self.edges))
        np.testing.assert_array_equal(FS.pair_counts(self.t, self.edges, key=self.key),
                                      brute_pairs(self.t, self.edges, key=self.key))

    def test_group_and_exclusion_equal_brute_force(self):
        for rule in FS.RULES:
            ids, size = FS.burst_ids(self.t, self.sign, 5.0, rule)
            for s in (1, -1):
                m = self.sign == s
                ex = size[m] < 3
                got = FS.pair_counts(self.t[m], self.edges, key=self.key[m], group=ids[m], excluded=ex)
                want = brute_pairs(self.t[m], self.edges, key=self.key[m], group=ids[m], excluded=ex)
                np.testing.assert_array_equal(got, want, err_msg=rule)

    def test_cross_counts_equal_brute_force(self):
        rng = np.random.default_rng(5)
        tb = np.sort(34200 + rng.uniform(0, 400, 90)); kb = rng.choice([7, 13, 250], 90)
        want = np.zeros(len(self.edges) - 1, int); want_m = want.copy()
        for i in range(len(self.t)):
            for j in range(len(tb)):
                lag = tb[j] - self.t[i]
                k = np.searchsorted(self.edges, lag, side="right") - 1
                if 0 <= k < len(want) and lag < self.edges[-1]:
                    want[k] += 1
                    want_m[k] += int(self.key[i] == kb[j])
        np.testing.assert_array_equal(FS.cross_counts(self.t, tb, self.edges), want)
        np.testing.assert_array_equal(FS.cross_counts(self.t, tb, self.edges, self.key, kb), want_m)

    def test_burst_rules(self):
        t = np.array([0, .2, .4, 3, 3.1, 3.2, 3.3]) + 34200
        sign = np.array([1, 1, -1, 1, 1, 0, 1])
        run, _ = FS.burst_ids(t, sign, 1.0, "run")
        self.assertEqual(list(run), [0, 0, 1, 2, 2, 3, 4])
        stream, _ = FS.burst_ids(t, sign, 1.0, "stream")
        self.assertEqual(len({stream[0], stream[1]}), 1)          # buys at 0, .2 together
        self.assertEqual(len({stream[3], stream[4], stream[6]}), 1)  # sell/unsigned ignored
        self.assertNotEqual(stream[1], stream[3])                  # 2.6s gap splits
        timing, size = FS.burst_ids(t, sign, 1.0, "timing")
        self.assertEqual(list(timing), [0, 0, 0, 1, 1, 1, 1])
        self.assertEqual(list(size), [3, 3, 3, 4, 4, 4, 4])


class FingerprintSignalTests(unittest.TestCase):
    def test_planted_program_creates_short_lag_excess_and_state_similarity(self):
        rng = np.random.default_rng(11)

        def day(with_program):
            n = 4000
            t = np.sort(34200 + rng.uniform(0, 23000, n))
            size = rng.choice(np.arange(1, 400), n)
            sign = rng.choice([-1, 1], n)
            spread = rng.uniform(1, 20, n)
            if with_program:
                start = 40000.0
                pt = start + np.cumsum(rng.uniform(1, 4, 120))
                t = np.r_[t, pt]; size = np.r_[size, np.full(120, 137)]
                sign = np.r_[sign, np.ones(120, int)]; spread = np.r_[spread, np.full(120, 5.0)]
            o = np.argsort(t, kind="stable")
            return dict(time=t[o], sign=sign[o].astype(np.int8), volume=size[o].astype(float),
                        untruncated=np.ones(len(t), bool), spread_bps=spread[o],
                        exec_depth=np.full(len(t), 1000.0), imbalance=np.zeros(len(t)))
        days = {"20240102": day(True), "20240103": day(False),
                "20240104": day(False), "20240105": day(False)}
        pairs = [("20240102", "20240103"), ("20240104", "20240105")]
        res = FS.analyze(days, pairs)
        ci = FS.CLASSES.index("u_nonround"); d0 = list(res["dates"]).index("20240102")
        short = slice(0, np.searchsorted(FS.FINE_EDGES, 10.0))
        within = res["within_matches"][d0, ci, 0, short].sum() / res["within_pairs"][d0, ci, 0, short].sum()
        null = res["cross_matches"][0, ci].sum() / res["cross_pairs"][0, ci].sum()
        self.assertGreater(within, 3 * null)
        # program children share spread 5.0; matched recurrences are more similar than controls
        cnt = res["recur_count"][d0, 0]; sm = res["recur_sum"][d0, 0]
        b = 1  # [2, 10) seconds, where the planted program recurs
        self.assertLess(sm[0, b, 0] / cnt[0, b], sm[1, b, 0] / cnt[1, b])
        # a 5s stream burst keeps most planted same-size pairs together; a 0.1s one does not
        gi5 = FS.GAPS.index(5.0); gi01 = FS.GAPS.index(0.1); ri = FS.RULES.index("stream")
        bci = FS.BURST_CLASSES.index("u_nonround")
        m5 = res["burst_matches"][d0, ri, gi5, 0, bci].sum(); m01 = res["burst_matches"][d0, ri, gi01, 0, bci].sum()
        self.assertGreater(m5, 10 * max(m01, 1))


class DepthCensoringTests(unittest.TestCase):
    """Local size distributions tied to book depth mimic same-origin repeats at short lags.

    The depth-matched cross-day null (pairs share an executed-side depth quartile) must remove that,
    and, being cross-day, it cannot absorb a same-day program.
    """

    def make_day(self, rng, censoring, program=False):
        n = 20000
        t = np.sort(34200 + rng.uniform(0, 23000, n))
        sign = rng.choice([-1, 1], n)
        if censoring:
            thin = (rng.random(int(23400 / 120) + 2) < 0.5)[((t - 34200) // 120).astype(int)]
            depth = np.where(thin, 6.0, 5000.0)
            size = np.where(thin, rng.integers(1, 6, n), rng.integers(50, 300, n))
        else:
            depth = np.full(n, 5000.0)
            size = rng.integers(1, 300, n)
        if program:
            pt = 38000 + np.cumsum(rng.uniform(3, 8, 300))  # one program, ~27 minutes of children
            t = np.r_[t, pt]; size = np.r_[size, np.full(300, 137)]
            sign = np.r_[sign, np.ones(300, int)]; depth = np.r_[depth, np.full(300, 5000.0)]
        o = np.argsort(t, kind="stable")
        t, size, sign, depth = t[o], size[o], sign[o], depth[o]
        return dict(time=t, sign=sign.astype(np.int8), volume=size.astype(float), untruncated=size < depth,
                    spread_bps=np.where(depth < 10, 8.0, 1.0), exec_depth=depth, imbalance=np.zeros(len(t)))

    def ratios(self, censoring, program, seed):
        rng = np.random.default_rng(seed)
        days = {d: self.make_day(rng, censoring, program and d == "20240102")
                for d in ("20240102", "20240103", "20240104", "20240105")}
        res = FS.analyze(days, [("20240102", "20240103"), ("20240104", "20240105")], "SYN")
        ci = FS.CLASSES.index("u_nonround"); d0 = list(res["dates"]).index("20240102")
        short = slice(0, int(np.searchsorted(FS.FINE_EDGES, 30.0)))
        cross = res["cross_matches"][0, ci].sum() / res["cross_pairs"][0, ci].sum()
        vs_cross = res["within_matches"][d0, ci, 0, short].sum() / (res["within_pairs"][d0, ci, 0, short].sum() * cross)
        cross_dm = res["cross_matches_dm"][0, ci].sum() / res["cross_pairs_dm"][0, ci].sum()
        vs_dm = res["within_matches_dm"][d0, ci, short].sum() / (res["within_pairs_dm"][d0, ci, short].sum() * cross_dm)
        return vs_cross, vs_dm

    def test_censoring_fools_cross_day_null_but_not_depth_matched_null(self):
        vs_cross, vs_dm = self.ratios(censoring=True, program=False, seed=22)
        self.assertGreater(vs_cross, 1.5)        # the artifact
        self.assertLess(abs(vs_dm - 1.0), 0.1)   # removed by depth matching

    def test_program_survives_depth_matched_null(self):
        vs_cross, vs_dm = self.ratios(censoring=False, program=True, seed=21)
        self.assertGreater(vs_dm, 1.5)
        self.assertLess(abs(vs_dm - vs_cross), 0.15)  # without censoring both nulls agree


class StateSimilarityJitterTests(unittest.TestCase):
    """Book state drifts with time and sizes are independent of state: no program, no similarity.

    The v2 control (packet nearest in time to t_j) inherits a lag jitter that makes controls look
    less similar at short lags. The corrected statistic compares within narrow lag bins and must
    give a ratio near 1.
    """

    def test_lag_binned_state_similarity_is_unbiased_without_programs(self):
        import fingerprint_state as FST
        rng = np.random.default_rng(8)
        n = 40000
        t = np.sort(34200 + rng.uniform(0, 23000, n))
        walk = np.cumsum(rng.normal(0, 0.05, n))            # spread drifts with time
        size = rng.integers(1, 60, n)                       # small alphabet: many recurrences
        day = dict(time=t, sign=rng.choice([-1, 1], n).astype(np.int8), volume=size.astype(float),
                   untruncated=np.ones(n, bool), spread_bps=5 + walk, exec_depth=np.full(n, 5000.0),
                   imbalance=np.zeros(n), hidden_share=np.zeros(n))
        count, sums = FST.pairs_for_day(day)
        edges = FST.LAG_EDGES
        sel = (edges[:-1] >= 2) & (edges[1:] <= 10)
        w = count[0, sel]
        matched = (sums[0, sel, 0] / np.maximum(count[0, sel], 1) * w).sum() / w.sum()
        control = (sums[1, sel, 0] / np.maximum(count[1, sel], 1) * w).sum() / w.sum()
        self.assertLess(abs(matched / control - 1.0), 0.05)


if __name__ == "__main__":
    unittest.main()
