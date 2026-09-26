import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src_py"))
import evidence_formulas as EF  # noqa: E402
import evidence_stats as ES  # noqa: E402
import fingerprint_stats as FS  # noqa: E402

T0 = 34200.0


def brute_within(t, edges, key=None, group=None, excluded=None):
    n = len(t)
    order = sorted(range(n), key=lambda i: (t[i], i))
    out = np.zeros(len(edges) - 1, int)
    for a in range(n):
        for b in range(a + 1, n):
            i, j = order[a], order[b]
            if key is not None and key[i] != key[j]:
                continue
            if group is not None and (group[i] != group[j] or excluded[i]):
                continue
            k = np.searchsorted(edges, t[j] - t[i], side="right") - 1
            if 0 <= k < len(out):
                out[k] += 1
    return out


class CountingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.t = np.sort(T0 + rng.uniform(0, 200, 180))
        self.t[20] = self.t[19]
        self.key = rng.choice([101, 133, 257], len(self.t))
        self.edges = ES.PHASE_EDGES[:200]

    def test_phase_within_and_cross_equal_brute_force(self):
        np.testing.assert_array_equal(ES.phase_within(self.t, edges=self.edges), brute_within(self.t, self.edges))
        np.testing.assert_array_equal(ES.phase_within(self.t, self.key, edges=self.edges),
                                      brute_within(self.t, self.edges, key=self.key))
        rng = np.random.default_rng(12)
        tb = np.sort(T0 + rng.uniform(0, 200, 90)); kb = rng.choice([101, 133], 90)
        want = np.zeros(len(self.edges) - 1, int); want_k = want.copy()
        for i in range(len(self.t)):
            for j in range(len(tb)):
                k = np.searchsorted(self.edges, tb[j] - self.t[i], side="right") - 1
                if 0 <= k < len(want):
                    want[k] += 1; want_k[k] += int(self.key[i] == kb[j])
        np.testing.assert_array_equal(ES.phase_cross(self.t, tb, edges=self.edges), want)
        np.testing.assert_array_equal(ES.phase_cross(self.t, tb, self.key, kb, edges=self.edges), want_k)

    def test_within_group_counts_equal_brute_force(self):
        sign = np.ones(len(self.t), int)
        for gap in (0.5, 2.0, 5.0):
            ids, bsize = FS.burst_ids(self.t, sign, gap, "run")
            ex = bsize < 3
            for key in (None, self.key):
                got = ES.within_group_counts(self.t, ids, ex, ES.LOCK_EDGES[:40], key)
                want = brute_within(self.t, ES.LOCK_EDGES[:40], key=key, group=ids, excluded=ex)
                np.testing.assert_array_equal(got, want)

    def test_chains_split_on_size_and_gap(self):
        t = T0 + np.array([0, 10, 20, 400, 410, 5, 15])
        size = np.array([7, 7, 7, 7, 7, 9, 9])
        sign = np.array([1, 1, -1, 1, 1, -1, -1])
        ch = ES.chains(t, sign, size, gap=300)
        rows = sorted(map(tuple, ch.tolist()))
        self.assertEqual(rows, sorted([(3, 2, 1, 20), (2, 2, 0, 10), (2, 0, 2, 10)]))


def make_day(rng, n=6000, hours=4.0, program=None, two_sided=None, size_shift=0, dollar=None, vol_bps=2.0,
             max_size=400):
    """Synthetic packet day. Background: Poisson times, random non-round sizes and sides."""
    T = hours * 3600
    t = np.sort(T0 + rng.uniform(0, T, n))
    size = rng.integers(1, max_size, n) + size_shift
    size = np.where(size % 100 == 0, size + 1, size)
    sign = rng.choice([-1, 1], n)
    parts = [(t, size, sign)]
    if program:
        for start in program["starts"]:
            k = np.arange(program["children"])
            phase = 0.0 if program.get("anchored") else rng.uniform(0, 1)
            tt = T0 + start + phase + k * program["period"] + rng.normal(0, 0.001, len(k))
            parts.append((tt, np.full(len(k), program["size"]), np.full(len(k), program["side"])))
    if two_sided:
        for start in two_sided["starts"]:
            k = np.arange(two_sided["children"])
            tt = T0 + start + np.cumsum(rng.exponential(two_sided["mean_gap"], len(k)))
            parts.append((tt, np.full(len(k), two_sided["size"]), np.where(k % 2 == 0, 1, -1)))
    t = np.concatenate([p[0] for p in parts]); size = np.concatenate([p[1] for p in parts])
    sign = np.concatenate([p[2] for p in parts])
    order = np.argsort(t, kind="stable"); t, size, sign = t[order], size[order], sign[order]
    steps = rng.normal(0, vol_bps * 1e-4, len(t)) * np.sqrt(np.r_[1.0, np.diff(t)])
    mid = 40.0 * np.exp(np.cumsum(steps))
    if dollar:
        # Dollar-targeted children inserted at a fixed cadence; their share count follows the mid.
        tt = T0 + np.arange(dollar["start"], dollar["start"] + dollar["duration"], dollar["period"])
        mid_at = np.interp(tt, t, mid)
        extra_size = np.rint(dollar["notional"] / mid_at).astype(np.int64)
        for side in (1, -1):
            t = np.r_[t, tt + (0.3 if side < 0 else 0.0)]; size = np.r_[size, extra_size]
            sign = np.r_[sign, np.full(len(tt), side)]; mid = np.r_[mid, mid_at]
        order = np.argsort(t, kind="stable"); t, size, sign, mid = t[order], size[order], sign[order], mid[order]
    m = len(t)
    return dict(time=t, sign=sign.astype(np.int8), volume=size.astype(float), untruncated=np.ones(m, bool),
                mid=mid, spread_bps=np.full(m, 2.0), exec_depth=np.full(m, 10000.0), imbalance=np.zeros(m),
                hidden_share=np.zeros(m), n_messages=np.ones(m, np.int32))


def run_days(builder, n_pairs=3, seed=0, modules="ABI"):
    rng = np.random.default_rng(seed)
    dates = ["2024%04d" % (101 + i) for i in range(2 * n_pairs)]
    days = {d: builder(rng, i) for i, d in enumerate(dates)}
    pairs = [(dates[2 * i], dates[2 * i + 1]) for i in range(n_pairs)]
    return ES.analyze(days, pairs, "SYN", modules=modules)


class TimingFingerprintTests(unittest.TestCase):
    def test_no_program_gives_unit_phase_ratios(self):
        z = run_days(lambda rng, i: make_day(rng), seed=1, modules="A")
        r = EF.phase_ratios(z)
        self.assertAlmostEqual(r["a1"], 1.0, delta=0.12)
        self.assertAlmostEqual(r["a1_signed"], 1.0, delta=0.12)

    def test_timer_program_is_phase_locked_and_identical_sizes_more_so(self):
        z = run_days(lambda rng, i: make_day(rng, program=dict(starts=[600, 4000, 9000], children=120, period=1.0,
                                                               size=137 + 2 * i, side=1)), seed=2, modules="A")
        r = EF.phase_ratios(z)
        self.assertGreater(r["a1"], 1.3)
        self.assertGreater(r["a2"], 2.0)
        j = EF.burst_phase_j(z)
        self.assertTrue(np.isfinite(j["j"]).any())

    def test_wall_clock_anchored_unrelated_traders_do_not_fake_agreement(self):
        # Many unrelated traders firing on whole seconds with random sizes, every day: timing alone
        # is locked within and across days (A1 ~ 1), and size identity adds nothing (A2 ~ 1).
        def anchored(rng, i):
            d = make_day(rng)
            n = 6000
            tt = T0 + np.floor(rng.uniform(0, 4 * 3600, n)) + np.abs(rng.normal(0, 0.002, n))
            sz = rng.integers(1, 400, n); sz = np.where(sz % 100 == 0, sz + 1, sz)
            t = np.r_[d["time"], tt]; order = np.argsort(t, kind="stable")
            for key in d:
                extra = {"time": tt, "sign": rng.choice([-1, 1], n), "volume": sz}.get(key)
                if extra is None:
                    extra = np.repeat(d[key][:1], n)
                d[key] = np.r_[d[key], extra][order]
            return d
        r = EF.phase_ratios(run_days(anchored, seed=9, modules="A"))
        self.assertAlmostEqual(r["a1"], 1.0, delta=0.12)
        self.assertAlmostEqual(r["a2"], 1.0, delta=0.25)


class SideStructureTests(unittest.TestCase):
    def ratios(self, z):
        obs, exp = EF.side_observed_expected(z)
        return EF.side_ratio(obs, exp, 2, 60)

    def test_one_sided_program_has_no_opposite_side_excess(self):
        z = run_days(lambda rng, i: make_day(rng, program=dict(starts=[600, 5000], children=150, period=3.0,
                                                               size=233 + 2 * i, side=-1)), seed=3, modules="B")
        same, opp = self.ratios(z)
        self.assertGreater(same, 1.5)
        self.assertAlmostEqual(opp, 1.0, delta=0.2)

    def test_two_sided_algorithm_creates_opposite_side_excess(self):
        z = run_days(lambda rng, i: make_day(rng, two_sided=dict(starts=[600, 5000], children=300, mean_gap=3.0,
                                                                 size=233 + 2 * i)), seed=4, modules="B")
        same, opp = self.ratios(z)
        self.assertGreater(opp, 1.5)

    def test_multiday_did(self):
        # Adjacent days share a buy program's size; distant days do not. Size distributions drift
        # between date pairs on both sides.
        def with_campaign(rng, i):
            prog = dict(starts=[1000, 6000], children=200, period=4.0, size=[119, 211, 307][i // 2], side=1)
            return make_day(rng, program=prog, size_shift=[0, 40, 80][i // 2])

        def drift_only(rng, i):
            return make_day(rng, size_shift=[0, 40, 80][i // 2])
        did_prog, _ = EF.multiday_did(run_days(with_campaign, seed=5, modules="B"))
        did_null, _ = EF.multiday_did(run_days(drift_only, seed=6, modules="B"))
        self.assertGreater(did_prog, 1.3)
        self.assertAlmostEqual(did_null, 1.0, delta=0.15)


class DollarSizeTests(unittest.TestCase):
    def d_stat(self, z):
        c = z["i_counts"].sum(0)                      # [side, near/far, e, dm, ds]
        d = EF.dollar_d(c)                            # [side, near/far, e]
        return d[:, 0, 2] - d[:, 1, 2]

    def test_dollar_targeting_detected_on_both_sides(self):
        dollar = dict(start=300, duration=12000, period=2.0, notional=60000)
        z = run_days(lambda rng, i: make_day(rng, dollar=dollar, vol_bps=4.0, max_size=4000), n_pairs=2, seed=7,
                     modules="I")
        diff = self.d_stat(z)
        self.assertGreater(diff[0], 0.2)
        self.assertGreater(diff[1], 0.2)

    def test_no_targeting_gives_zero(self):
        z = run_days(lambda rng, i: make_day(rng, n=40000, vol_bps=4.0, max_size=4000), n_pairs=2, seed=8,
                     modules="I")
        diff = self.d_stat(z)
        self.assertTrue(np.all(np.abs(diff) < 0.05), diff)


if __name__ == "__main__":
    unittest.main()


class ProgramBurstFeatureTests(unittest.TestCase):
    def test_vectorized_features_match_burst_rows(self):
        import fingerprint_burst_rows as FBR
        import program_bursts as PB
        rng = np.random.default_rng(21)
        day = make_day(rng, n=4000, hours=2.0, program=dict(starts=[100, 3000], children=60, period=0.7, size=233, side=1))
        n = len(day["time"])
        day["untruncated"] = rng.random(n) < 0.7
        day["hidden_share"] = np.where(rng.random(n) < 0.2, rng.random(n), 0.0)
        day["spread_bps"] = rng.uniform(1, 5, n); day["exec_depth"] = rng.uniform(50, 5000, n)
        day["imbalance"] = rng.uniform(-1, 1, n)
        day["sign"] = np.where(rng.random(n) < 0.05, 0, day["sign"]).astype(np.int8)
        rows = FBR.burst_rows(day, "20240101", "SYN", "run", 60.0, np.zeros(n, np.int64), {}, None)
        member, table = PB.run_bursts(day)
        self.assertEqual(len(rows), len(table["start"]))
        keys = ["n_packets", "duration", "intensity", "iat_cv", "iat_median", "truncated_share", "hidden_share",
                "spread_bps", "log_exec_depth", "imbalance", "tod", "trailing_activity"]
        for k in keys:
            np.testing.assert_allclose(table[k], [r[k] for r in rows], rtol=1e-9, atol=1e-12, err_msg=k)
        np.testing.assert_array_equal(table["side"], [r["side"] for r in rows])
        self.assertTrue(np.all(member[member >= 0] < len(rows)))


class SyncCountTests(unittest.TestCase):
    def test_sync_counts_equal_brute_force(self):
        import evidence_sync as SY
        rng = np.random.default_rng(31)
        n = 400
        T = np.sort(T0 + rng.uniform(0, 30, n)); T[5] = T[4] + 0.0005
        NAME = rng.integers(0, 4, n); SIGN = rng.choice([-1, 1], n); PROG = rng.random(n) < 0.3
        pooled, pair = SY.sync_counts(T, NAME, SIGN, PROG, 4)
        want = np.zeros_like(pooled); want_pair = np.zeros_like(pair)
        for di, delta in enumerate(SY.DELTAS):
            for oi, o in enumerate(SY.OFFSETS):
                for i in range(n):
                    for j in range(n):
                        if NAME[i] == NAME[j]:
                            continue
                        x = T[j] - T[i] - o
                        if -delta <= x < delta:
                            rel = int(SIGN[i] != SIGN[j])
                            want[di, oi, rel, int(PROG[i]), int(PROG[j])] += 1
                            if di == 1:
                                want_pair[oi, rel, int(PROG[i]), NAME[i], NAME[j]] += 1
        np.testing.assert_array_equal(pooled, want)
        np.testing.assert_array_equal(pair, want_pair)


class CampaignTests(unittest.TestCase):
    def test_campaigns_link_identical_rare_sizes_and_placebos_do_not_overlap(self):
        import evidence_campaigns as EC
        rng = np.random.default_rng(41)
        prog = dict(starts=[1000], children=30, period=20.0, size=4567, side=1)
        days = {("2024%04d" % (101 + i)): make_day(rng, n=3000, hours=6.0, max_size=3000,
                                                    program=prog if i == 0 else None) for i in range(4)}
        pairs = [("20240101", "20240102"), ("20240103", "20240104")]
        pair_of = {d: p for p in pairs for d in p}
        rates, n_other = EC.side_base_rates(days, set(pair_of["20240101"]))
        rows, summary = EC.day_rows(days["20240101"], "20240101", "SYN", rates, n_other, seed=1)
        camp = [r for r in rows if r["kind"] == "campaign" and r["size"] == 4567]
        self.assertEqual(len(camp), 1)
        self.assertEqual(camp[0]["n"], 30)
        self.assertAlmostEqual(camp[0]["t1"] - camp[0]["t0"], 29 * 20.0, delta=1.0)
        windows = [(r["t0"], r["t1"]) for r in rows if r["kind"] == "campaign" and r["side"] == 1]
        for r in rows:
            if r["kind"] == "placebo" and r["side"] == 1:
                self.assertTrue(all(r["t1"] < a or r["t0"] > b for a, b in windows))


class PassiveTests(unittest.TestCase):
    def test_classify_replace_and_remainder(self):
        import passive_extract as PE
        import tempfile
        rows = [
            (34200.0, 1, 1, 100, 1000000, 1), (34200.0, 1, 2, 100, 1010000, -1),     # quote 100.00 / 101.00
            (34201.0, 1, 3, 37, 1000000, 1),        # at touch, bid
            (34202.0, 1, 4, 41, 1005000, 1),        # inside, bid
            (34203.0, 1, 5, 43, 990000, -1),        # ask below bid? marketable-looking -> "better" than ask
            (34204.0, 3, 3, 37, 1000000, 1), (34204.0, 1, 6, 37, 999900, 1),  # replace: delete + add same t, side
            (34205.0, 4, 2, 10, 1010000, -1), (34205.0, 1, 7, 53, 1010000, 1),  # remainder posted with an execution
        ]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "X_2024-01-01_34200000_57600000_message_10.csv")
            with open(path, "w") as f:
                for r in rows:
                    f.write("%.9f,%d,%d,%d,%d,%d\n" % r)
            arrays, counts = PE.extract(path)
        self.assertEqual(list(arrays["size"]), [37, 41, 43, 37, 53])
        # the last add is priced through the prevailing bid (at the ask), so it is "better than touch"
        self.assertEqual(list(arrays["pos"]), [1, 0, 0, 2, 0])
        self.assertEqual(list(arrays["replace"]), [False, False, False, True, False])
        self.assertEqual(list(arrays["remainder"]), [False, False, False, False, True])

    def test_h2_counts_equal_brute_force(self):
        import passive_stats as PS2
        rng = np.random.default_rng(51)
        p = make_day(rng, n=300, hours=0.1)
        na = 250
        a = dict(time=np.sort(T0 + rng.uniform(0, 360, na)), side=rng.choice([-1, 1], na).astype(np.int8),
                 size=rng.integers(1, 30, na), pos=rng.integers(0, 3, na).astype(np.int8),
                 replace=rng.random(na) < 0.1, remainder=rng.random(na) < 0.1)
        p["volume"] = rng.integers(1, 30, len(p["time"])).astype(float)
        got = PS2.h2_counts(p, a)
        size = np.rint(p["volume"]).astype(int); u = (size % 100 != 0)
        want = np.zeros_like(got)
        plain = ~a["replace"] & ~a["remainder"]
        for i in np.flatnonzero(u):
            for j in np.flatnonzero(plain):
                lag = abs(a["time"][j] - p["time"][i])
                k = np.searchsorted(PS2.EDGES, lag, side="right") - 1
                if not (0 <= k < len(PS2.EDGES) - 1):
                    continue
                rel = 0 if a["side"][j] == p["sign"][i] else 1
                want[rel, 0, k] += 1
                want[rel, 1, k] += int(a["size"][j] == size[i])
        np.testing.assert_array_equal(got, want)


class PriceMatchedNullTests(unittest.TestCase):
    def test_dollar_sized_unrelated_traders_are_removed_by_price_matching(self):
        # Unrelated traders on both sides size orders in dollars ($2,000-$20,000 in $1,000 steps);
        # the price level differs across days, so a depth-matched cross-day null sees an excess on
        # both sides; within a day, a 10 bps price-matched long-lag null does not.
        import evidence_price_null as PN

        def build(rng, i):
            n = 8000
            t = np.sort(T0 + rng.uniform(0, 4 * 3600, n))
            price = 40.0 * (1 + 0.04 * i) * np.exp(np.cumsum(rng.normal(0, 2e-4, n)))
            notional = rng.integers(2, 21, n) * 1000.0
            size = np.rint(notional / price).astype(np.int64)
            size = np.where(size % 100 == 0, size + 1, size)
            m = len(t)
            return dict(time=t, sign=rng.choice([-1, 1], n).astype(np.int8), volume=size.astype(float),
                        untruncated=np.ones(m, bool), mid=price, spread_bps=np.full(m, 2.0),
                        exec_depth=np.full(m, 10000.0), imbalance=np.zeros(m), hidden_share=np.zeros(m),
                        n_messages=np.ones(m, np.int32))
        rng = np.random.default_rng(61)
        dates = ["2024%04d" % (101 + i) for i in range(4)]
        days = {d: build(rng, i) for i, d in enumerate(dates)}
        pairs = [(dates[0], dates[1]), (dates[2], dates[3])]
        dm = ES.analyze(days, pairs, "SYN", modules="B")
        obs, exp = EF.side_observed_expected(dm)
        same_dm, opp_dm = EF.side_ratio(obs, exp, 2, 60)
        self.assertGreater(opp_dm, 1.3)
        z = PN.analyze(days, pairs)
        w = z["within_10"].sum(0).astype(float)                      # [relation, pairs/matches, lag]
        short = (PN.EDGES[:-1] >= 2) & (PN.EDGES[1:] <= 60)
        long_ = PN.EDGES[:-1] >= 3600
        for rel in (0, 1):
            r_short = w[rel, 1][short].sum() / w[rel, 0][short].sum()
            r_long = w[rel, 1][long_].sum() / w[rel, 0][long_].sum()
            self.assertAlmostEqual(r_short / r_long, 1.0, delta=0.15)

    def test_planted_program_survives_within_day_price_matched_null(self):
        import evidence_price_null as PN
        rng = np.random.default_rng(62)
        day = make_day(rng, n=20000, hours=6.0, max_size=600)
        for k, size in enumerate((173, 219, 347, 411, 457, 523)):
            prog = make_day(rng, n=0, hours=6.0, program=dict(starts=[1000 + 3000 * k], children=100, period=2.0, size=size, side=1))
            order = np.argsort(np.r_[day["time"], prog["time"]], kind="stable")
            for key in day:
                day[key] = np.r_[day[key], prog[key]][order]
        day["mid"] = 40.0 * np.exp(np.cumsum(rng.normal(0, 2e-4, len(day["time"])) * np.sqrt(np.r_[1.0, np.diff(day["time"])])))
        z = PN.analyze({"20240101": day, "20240102": make_day(rng, n=8000, hours=6.0, max_size=600)},
                       [("20240101", "20240102")])
        w = z["within_50"][0].astype(float)
        short = (PN.EDGES[:-1] >= 2) & (PN.EDGES[1:] <= 60); long_ = PN.EDGES[:-1] >= 3600
        r = [(w[rel, 1][short].sum() / w[rel, 0][short].sum()) / (w[rel, 1][long_].sum() / w[rel, 0][long_].sum())
             for rel in (0, 1)]
        self.assertGreater(r[0], 1.5)
        self.assertAlmostEqual(r[1], 1.0, delta=0.25)
