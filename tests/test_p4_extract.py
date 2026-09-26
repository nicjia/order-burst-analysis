import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src_py"))

import burst_alt as BA  # noqa: E402
import p4_extract as PX  # noqa: E402

T0 = 34200.0
P = 10000  # LOBSTER price units per dollar


def write_tape(path, rows):
    with open(path, "w") as fh:
        for t, ty, oid, sz, px, dr in rows:
            fh.write("%.9f,%d,%d,%d,%d,%d\n" % (t, ty, oid, sz, px, dr))


def random_tape(seed, n=6000):
    """Adds, partial cancels, deletes, visible and hidden executions, unknown-order cancels, crosses."""
    rng = np.random.default_rng(seed)
    live = {}
    rows = []
    t = T0 - 50.0
    oid = 1
    for _ in range(n):
        t += float(rng.exponential(0.05))
        u = rng.random()
        if u < 0.45 or not live:
            side = 1 if rng.random() < 0.5 else -1
            depth = int(rng.integers(0, 6)) * 100
            px = 1000000 - depth if side == 1 else 1000500 + depth
            sz = int(rng.integers(1, 5)) * 100 + int(rng.integers(0, 3)) * 37
            if rng.random() < 0.01:                      # occasional crossing add, deleted 1 ms later
                px = 1000600 if side == 1 else 999900
                rows.append((t, 1, oid, sz, px, side))
                t += 0.001
                rows.append((t, 3, oid, sz, px, side))
                oid += 1
                continue
            live[oid] = [side, px, sz]
            rows.append((t, 1, oid, sz, px, side))
            oid += 1
        elif u < 0.60:
            k = list(live)[int(rng.integers(len(live)))]
            side, px, sz = live[k]
            cut = int(rng.integers(1, sz + 1))
            if cut >= sz:
                rows.append((t, 3, k, sz, px, side)); del live[k]
            else:
                rows.append((t, 2, k, cut, px, side)); live[k][2] -= cut
        elif u < 0.75:
            k = list(live)[int(rng.integers(len(live)))]
            side, px, sz = live[k]
            rows.append((t, 3, k, sz, px, side)); del live[k]
        elif u < 0.88:
            k = list(live)[int(rng.integers(len(live)))]
            side, px, sz = live[k]
            q = int(rng.integers(1, sz + 1))
            rows.append((t, 4, k, q, px, side))
            live[k][2] -= q
            if live[k][2] <= 0:
                del live[k]
        elif u < 0.95:
            rows.append((t, 5, 0, int(rng.integers(1, 300)), 1000250, 1))
        else:                                             # order that predates the file
            side = 1 if rng.random() < 0.5 else -1
            px = (1000000 if side == 1 else 1000500) + int(rng.integers(-3, 4)) * 100
            rows.append((t, int(rng.choice([2, 3, 4])), 10 ** 9 + oid, int(rng.integers(1, 200)), px, side))
            oid += 1
    return rows


def compile_helper(tmp):
    cxx = shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        return None
    exe = os.path.join(tmp, "p4_bbo")
    subprocess.run([cxx, "-O2", "-std=c++11", "-o", exe, os.path.join(REPO, "src_cpp", "p4_bbo.cpp")], check=True)
    return exe


class BBOHelperTest(unittest.TestCase):
    def test_cpp_bbo_matches_python_reconstruct_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = compile_helper(tmp)
            if exe is None:
                self.skipTest("no C++ compiler")
            for seed in (1, 2, 3):
                path = os.path.join(tmp, "X_2024-01-02_message.csv")
                write_tape(path, random_tape(seed))
                msg = PX.read_messages(path)
                ctx_cpp, bid_i, ask_i, engine = PX.bbo_context(path, msg, exe)
                self.assertEqual(engine, "cpp")
                ref = BA.reconstruct(path)
                self.assertGreater(len(ref[0]), 100)
                for i in range(6):
                    np.testing.assert_array_equal(ctx_cpp[i], ref[i])
                np.testing.assert_array_equal(bid_i.astype(float) / BA.SCALE, ref[2])


def packet_tape(seed):
    """random_tape plus same-timestamp sweeps, sign conflicts, hidden/visible mixes, invalid directions."""
    rng = np.random.default_rng(seed)
    rows = random_tape(seed, n=3000)
    extra = []
    for k in range(400):
        t = T0 + 5 + float(rng.random()) * 200
        kind = rng.integers(0, 5)
        oid = 7_000_000 + k * 10
        if kind == 0:      # sweep: several visible rows, one sign, plus a hidden fill
            for j in range(int(rng.integers(2, 5))):
                extra.append((t, 4, oid + j, int(rng.integers(1, 300)), 1000500 + 100 * j, -1))
            extra.append((t, 5, 0, 50, 1000550, 1))
        elif kind == 1:    # conflicting native signs at one timestamp, with hidden rows in and outside the quote
            extra.append((t, 4, oid, 100, 1000500, -1)); extra.append((t, 4, oid + 1, 37, 1000000, 1))
            extra.append((t, 5, 0, 20, 1000250, 1)); extra.append((t, 5, 0, 20, 1009999, 1))
            extra.append((t, 5, 0, 20, 990001, -1))
        elif kind == 2:    # hidden only
            extra.append((t, 5, 0, int(rng.integers(1, 200)), int(rng.choice([1000250, 1009999, 990001])), 1))
        elif kind == 3:    # invalid direction on a visible row, alone and with a hidden row
            extra.append((t, 4, oid, 10, 1000500, 0)); extra.append((t, 5, 0, 10, 1009999, 1))
        else:              # visible one sign plus an invalid-direction row (kept in the packet)
            extra.append((t, 4, oid, 10, 1000500, -1)); extra.append((t, 4, oid + 1, 11, 1000500, 0))
    return sorted(rows + extra, key=lambda r: r[0])


class FastPacketTest(unittest.TestCase):
    def test_fast_packets_equal_canonical(self):
        import execution_packets as EP
        import p4_packets as PK
        with tempfile.TemporaryDirectory() as tmp:
            for seed in (11, 12):
                path = os.path.join(tmp, "X_2024-01-02_message.csv")
                write_tape(path, packet_tape(seed))
                msg = PX.read_messages(path)
                ctx = BA.reconstruct(path)
                _c, ref = EP.reconstruct_packets(path, context=ctx)
                got = PK.fast_packets(msg, ctx)
                self.assertGreater(len(ref), 500)
                self.assertTrue({"native-conflict-split", "invalid-native-direction", "outside-prequote",
                                 "ambiguous-hidden", "native+timestamp"} <= set(ref.sign_source))
                pd.testing.assert_frame_equal(got.drop(columns="vwap"), ref.drop(columns="vwap"), check_exact=True)
                np.testing.assert_allclose(got.vwap.to_numpy(float), ref.vwap.to_numpy(float), rtol=1e-12)


class QualifyingAddTest(unittest.TestCase):
    def test_rules(self):
        rows = [
            (T0 - 10, 1, 1, 300, 100 * P, 1),           # best bid 100.00
            (T0 - 10, 1, 2, 300, 100 * P + 1000, -1),    # best ask 100.10
            (T0 + 1.0, 1, 10, 137, 100 * P, 1),          # A at touch: qualifies
            (T0 + 2.0, 1, 11, 137, 100 * P - 100, 1),    # B behind touch
            (T0 + 3.0, 1, 12, 137, 100 * P + 500, 1),    # C inside spread: qualifies
            (T0 + 4.0, 3, 10, 137, 100 * P, 1),          # delete of A ...
            (T0 + 4.0, 1, 13, 137, 100 * P, 1),          # D ... replaced at the same time: replace half
            (T0 + 5.0, 4, 2, 100, 100 * P + 1000, -1),   # execution ...
            (T0 + 5.0, 1, 14, 50, 100 * P + 500, 1),     # E remainder at the execution timestamp
            (T0 + 6.0, 1, 15, 137, 100 * P + 500, 1),    # F fleeting ...
            (T0 + 6.5, 3, 15, 137, 100 * P + 500, 1),
            (T0 + 7.0, 1, 16, 137, 100 * P + 500, 1),    # G executed before its quick delete
            (T0 + 7.2, 4, 16, 37, 100 * P + 500, 1),
            (T0 + 7.5, 3, 16, 100, 100 * P + 500, 1),
            (T0 + 8.0, 1, 17, 137, 100 * P + 500, 1),    # H deleted after 2 s: not fleeting
            (T0 + 10.0, 3, 17, 137, 100 * P + 500, 1),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "X_2024-01-02_message.csv")
            write_tape(path, rows)
            msg = PX.read_messages(path)
            ctx, bid_i, ask_i, _ = PX.bbo_context(path, msg, None)
            mid = PX.MidPath(ctx[0], ctx[1], ctx[2], ctx[3])
            adds, m = PX.qualifying_adds(msg, bid_i, ask_i, mid)
        oids = list(adds["oid"])
        self.assertEqual(oids, [10, 11, 12, 13, 14, 15, 16, 17])
        q = dict(zip(oids, m["qualifying"]))
        self.assertTrue(q[10]); self.assertFalse(q[11]); self.assertTrue(q[12])
        self.assertTrue(dict(zip(oids, m["replace"]))[13]); self.assertFalse(q[13])
        self.assertTrue(dict(zip(oids, m["remainder"]))[14]); self.assertFalse(q[14])
        self.assertTrue(dict(zip(oids, m["fleeting"]))[15]); self.assertFalse(q[15])
        self.assertFalse(dict(zip(oids, m["fleeting"]))[16])
        self.assertFalse(dict(zip(oids, m["fleeting"]))[17]); self.assertTrue(q[17])


class SubmissionBurstTest(unittest.TestCase):
    def test_run_rule_and_order_outcomes(self):
        n = 9
        adds = dict(t=T0 + np.array([1, 2, 3, 4, 5, 6, 80, 81, 82.0]),
                    side=np.array([1, 1, 1, -1, -1, 1, 1, 1, 1]),
                    size=np.array([137, 137, 200, 100, 100, 50, 30, 30, 31]),
                    oid=np.arange(100, 100 + n))
        S = PX.submission_bursts(None, adds, np.ones(n, bool))
        self.assertEqual(len(S["t_b"]), 2)                 # bids 1-3; the 70 s gap splits 6 from 80-82
        np.testing.assert_array_equal(S["t_b"], T0 + np.array([1, 80.0]))
        np.testing.assert_array_equal(S["t_e"], T0 + np.array([3, 82.0]))
        np.testing.assert_array_equal(S["vol"], [474, 91])
        np.testing.assert_array_equal(S["mode_size"], [137, 30])
        np.testing.assert_array_equal(S["mode_count"], [2, 2])
        S["t_dec"] = np.array([T0 + 10.0, T0 + 90.0])
        msg = pd.DataFrame(dict(
            t=T0 + np.array([5.0, 12.0, 85.0, 95.0, 86.0]), ty=[4, 4, 4, 4, 3],
            oid=[100, 101, 106, 107, 108], sz=[40, 60, 10, 30, 31], px=0, dr=1))
        PX.order_outcomes_by_decision(msg, S)
        np.testing.assert_array_equal(S["exec_dec"], [40, 10])     # 101 executes after t_dec
        np.testing.assert_array_equal(S["cancel_dec"], [0, 31])


class MeasureTest(unittest.TestCase):
    def setUp(self):
        bt = T0 + np.array([-5.0, 10.0, 20.0, 70.0, 200.0, 650.0, 20000.0])
        bm = np.array([100.00, 100.02, 100.05, 100.03, 100.01, 99.99, 100.10])
        self.mid = PX.MidPath(bt, bm, bm - 0.01, bm + 0.01)

    def test_p4_quantities(self):
        B = dict(t_b=np.array([T0 + 5.0, T0 + 5.0]), t_e=np.array([T0 + 15.0, T0 + 15.0]),
                 side=np.array([1, -1], np.int8))
        PX.measure(B, self.mid, ("X", "20240102", "T"))
        np.testing.assert_allclose(B["m_ref"], [100.00, 100.00])
        # window [5, 25): mids 100.00, 100.02, 100.05 -> buy peak +0.05, sell peak 0.00
        np.testing.assert_allclose(B["peak_raw"], [0.05, 0.0], atol=1e-12)
        np.testing.assert_allclose(B["d60"], [0.05, -0.05], atol=1e-12)      # m(65) = 100.05
        np.testing.assert_allclose(B["d180"], [0.03, -0.03], atol=1e-12)     # m(185) = 100.03
        np.testing.assert_allclose(B["d300"], [0.01, -0.01], atol=1e-12)     # m(305) = 100.01
        np.testing.assert_allclose(B["d600"], [0.01, -0.01], atol=1e-12)     # m(605) = 100.01
        np.testing.assert_allclose(B["t_dec"], T0 + 605.0)
        np.testing.assert_allclose(B["m_dec"], [100.01, 100.01])

    def test_pseudo_bursts_are_deterministic_and_in_range(self):
        B1 = dict(t_b=T0 + np.arange(50.0), t_e=T0 + np.arange(50.0) + 30, side=np.ones(50, np.int8))
        B2 = {k: v.copy() for k, v in B1.items()}
        B3 = {k: v.copy() for k, v in B1.items()}
        PX.measure(B1, self.mid, ("X", "20240102", "T"))
        PX.measure(B2, self.mid, ("X", "20240102", "T"))
        PX.measure(B3, self.mid, ("X", "20240102", "S"))
        np.testing.assert_array_equal(B1["ps_t"], B2["ps_t"])
        self.assertFalse(np.allclose(B1["ps_t"], B3["ps_t"]))
        self.assertTrue(np.all(B1["ps_t"] >= T0))
        self.assertTrue(np.all(B1["ps_t"] + 30 + 600 <= PX.RTH1))


class EndToEndTest(unittest.TestCase):
    def test_engines_agree_and_bursts_form(self):
        rows = random_tape(7, n=4000)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "X_2024-01-02_message.csv")
            write_tape(path, rows)
            ref = BA.reconstruct(path)
            bid, ask = int(round(ref[2][-1] * P)), int(round(ref[3][-1] * P))
            # plant a run of buy packets against a new ask order and a run of bid adds at the touch
            base = max(r[0] for r in rows) + 5
            rows += [(base, 1, 5_000_001, 1000, ask, -1)]
            for k in range(5):
                rows.append((base + 1 + k, 4, 5_000_001, 17, ask, -1))
                rows.append((base + 1.5 + k, 1, 6_000_000 + k, 211, bid, 1))
            write_tape(path, rows)
            exe = compile_helper(tmp)
            a_py, s_py = PX.extract(path, "X", "20240102", helper=None, model=None)
            if exe is None:
                self.skipTest("no C++ compiler")
            a_cpp, s_cpp = PX.extract(path, "X", "20240102", helper=exe, model=None)
        self.assertEqual(s_py["engine"], "python"); self.assertEqual(s_cpp["engine"], "cpp")
        self.assertGreater(s_py["n_T"], 0); self.assertGreater(s_py["n_S"], 0)
        self.assertEqual(sorted(a_py), sorted(a_cpp))
        for k in a_py:
            np.testing.assert_array_equal(a_py[k], a_cpp[k], err_msg=k)
        self.assertEqual(json.dumps(s_py["day"], sort_keys=True), json.dumps(s_cpp["day"], sort_keys=True))
        self.assertTrue(np.isfinite(s_py["day"]["mid_open"]))


if __name__ == "__main__":
    unittest.main()
