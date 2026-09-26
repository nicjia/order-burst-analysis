#!/usr/bin/env python3
"""Checks of the v5 extractor on a synthetic LOBSTER day (no licensed data).

1. The rebuilt depth ladder equals a plain order-by-order replay of the message file at random times (10 levels).
2. The mid, far-quote, near-quote and microprice targets and the first-change targets equal values recomputed from
   the quote path.
3. No look-ahead: decision-time features recomputed from the message file TRUNCATED at the decision time are equal.
Usage: python3 test_burst_defs_raw5.py [p4_bbo helper path]
"""
import sys, tempfile, unittest
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src_py")); sys.path.insert(0, str(HERE.parent / "code")); sys.path.insert(0, str(HERE))
import p4_extract as X
import burst_defs_raw5 as V5
import synthetic_lobster as S

HELPER = sys.argv[1] if len(sys.argv) > 1 and Path(sys.argv[1]).exists() else None
TMP = Path(tempfile.mkdtemp())
MSG = TMP / "SYN_2024-06-04_34200000_57600000_message_0.csv"
import os
S.synth_day(MSG, seed=int(os.environ.get("SYN_SEED", "7")))


def replay(msg, q):
    """Visible volume per (side, price) from all messages strictly before q, order by order."""
    book = {1: {}, -1: {}}; live = {}
    for tt, ty, oid, sz, px, dr in msg.itertuples(index=False):
        if tt >= q:
            break
        if ty == 1:
            live[oid] = [dr, px, sz]; book[dr][px] = book[dr].get(px, 0) + sz
        elif ty in (2, 3, 4) and oid in live:
            d, p, _ = live[oid]
            book[d][p] = book[d].get(p, 0) - sz
            live[oid][2] -= sz
            if live[oid][2] <= 0:
                del live[oid]
    return book


class V5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.msg = X.read_messages(MSG)
        cls.ev, cls.bins, cls.chk, cls.blist = V5.run(str(MSG), "SYN", "20240604", HELPER)
        cls.ctx, _, _, _ = X.bbo_context(str(MSG), cls.msg, HELPER)

    def test_depth_ladder_equals_replay(self):
        D = V5.Depth(self.msg); mid = X.MidPath(*self.ctx[:4])
        rng = np.random.default_rng(0)
        for q in rng.uniform(34300, 57500, 25):
            b, a = mid.at(np.array([q]), "b")[0], mid.at(np.array([q]), "a")[0]
            vb, va = D.ladder(np.array([q]), np.array([round(b * 1e4)]), np.array([round(a * 1e4)]))
            bk = replay(self.msg, q)
            for k in range(10):
                self.assertEqual(vb[0, k], max(bk[1].get(round(b * 1e4) - k * V5.TICK, 0), 0))
                self.assertEqual(va[0, k], max(bk[-1].get(round(a * 1e4) + k * V5.TICK, 0), 0))

    def test_l1_check_is_exact(self):
        self.assertEqual(self.chk["bid_exact"], 1.0); self.assertEqual(self.chk["ask_exact"], 1.0)

    def test_targets_from_quote_path(self):
        bt, bm, bb, ba, qb, qa = self.ctx[:6]
        def at(arr, q):
            i = np.searchsorted(bt, q, "left") - 1
            return arr[i]
        for _, e in self.ev.sample(40, random_state=1).iterrows():
            td, s = e.t_dec, e.side
            md = at(bm, td); b0, a0 = at(bb, td), at(ba, td)
            q = min(td + 10, X.RTH1)
            self.assertAlmostEqual(e.r10, s * (at(bm, q) - md) / md * 1e4, places=6)
            far0, far1 = (b0, at(bb, q)) if s > 0 else (a0, at(ba, q))
            near0, near1 = (a0, at(ba, q)) if s > 0 else (b0, at(bb, q))
            self.assertAlmostEqual(e.f10, s * (far1 - far0) / md * 1e4, places=6)
            self.assertAlmostEqual(e.n10, s * (near1 - near0) / md * 1e4, places=6)
            mu = lambda tt: (at(ba, tt) * at(qb, tt) + at(bb, tt) * at(qa, tt)) / (at(qb, tt) + at(qa, tt))
            self.assertAlmostEqual(e.u10, s * (mu(q) - mu(td)) / mu(td) * 1e4, places=6)
            i0 = np.searchsorted(bt, td, "left") - 1
            later = np.flatnonzero((np.arange(len(bm)) > i0) & (bm != bm[i0]))
            if len(later) and bt[later[0]] - td <= 300:
                self.assertEqual(e.first_m, s * np.sign(bm[later[0]] - bm[i0]))
                self.assertAlmostEqual(e.wait_m, bt[later[0]] - td, places=6)

    def test_features_use_only_the_past(self):
        """Every decision-time feature of every event decided before a cut equals the value recomputed from the
        message file truncated at that cut. Positive control: the v4 OFI features scaled by the whole day's mean
        depth (a known leak) must differ."""
        import re
        cap = V5.CAP; V5.CAP = 10 ** 6                      # take every event, so both runs contain the same ones
        try:
            full_ev, _, _, _ = V5.run(str(MSG), "SYN", "20240604", HELPER)
            raw = pd.read_csv(MSG, header=None)
            target = re.compile(r"^(r|f|n|u)(\d+|_close)$|^first_|^wait_")
            feats = [c for c in full_ev.columns if not target.match(c) and c not in ("defn", "kind", "date", "ticker")
                     and not c.endswith("_v4")]
            compared, leak_seen = 0, False
            for cut in (40000.0, 47000.0, 53000.0):
                path = TMP / ("SYN_2024-06-04_cut%d_message_0.csv" % int(cut))
                raw[raw[0] < cut].to_csv(path, header=False, index=False)
                ev2, _, _, _ = V5.run(str(path), "SYN", "20240604", HELPER)
                a = full_ev[(full_ev.kind != "pseudo") & (full_ev.t_dec <= cut)]
                m = a.merge(ev2[ev2.kind != "pseudo"], on=["defn", "kind", "side", "t_b", "t_dec"], suffixes=("", "_cut"))
                self.assertGreater(len(m), 20)
                # the event SET is decided without look-ahead too: every event known by the cut exists in the cut run
                missing = a.merge(ev2, on=["defn", "kind", "side", "t_b", "t_dec"], how="left", indicator=True)
                self.assertEqual(len(m), len(a), "events selected with future information: %s"
                                 % missing[missing._merge == "left_only"].defn.value_counts().to_dict())
                for c in feats:
                    if c in ("side", "t_b", "t_dec"):
                        continue
                    x, y = m[c].to_numpy(float), m[c + "_cut"].to_numpy(float)
                    both = np.isfinite(x) | np.isfinite(y)
                    np.testing.assert_allclose(x[both], y[both], rtol=1e-9, atol=1e-9, err_msg="%s at cut %d" % (c, cut))
                compared += len(m)
                leak_seen |= not np.allclose(m["qofi_during_v4"], m["qofi_during_v4_cut"], equal_nan=True)
            self.assertGreater(compared, 100)
            self.assertTrue(leak_seen, "the positive control (whole-day OFI scale) should differ")
        finally:
            V5.CAP = cap

if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
