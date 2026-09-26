import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src_py"))
import queue_sim as QS  # noqa: E402

T0 = 34200.0


def msgs(rows):
    return pd.DataFrame(rows, columns=["t", "ty", "oid", "sz", "px", "dr"])


def base_book():
    # bid 100.00 (id 1, 300), ask 101.00: A=10 (100), B=11 (100); deeper ask 101.01 (id 12, 200)
    return [(T0 + 0.0, 1, 1, 300, 1000000, 1), (T0 + 0.1, 1, 10, 100, 1010000, -1),
            (T0 + 0.2, 1, 11, 100, 1010000, -1), (T0 + 0.3, 1, 12, 200, 1010100, -1)]


def run(rows, tau, side=-1):
    trig = pd.DataFrame(dict(tau=[tau], side=[side], label=["x"]))
    out, bbo = QS.simulate(msgs(rows), trig)
    return out.iloc[0], bbo


class QueueSimTests(unittest.TestCase):
    def test_fill_after_queue_ahead_consumed(self):
        rows = base_book() + [(T0 + 2, 4, 10, 100, 1010000, -1), (T0 + 3, 4, 11, 100, 1010000, -1),
                              (T0 + 4, 1, 13, 100, 1010000, -1), (T0 + 5, 4, 13, 50, 1010000, -1)]
        r, _ = run(rows, T0 + 1.0)
        self.assertTrue(r.posted); self.assertEqual(r.ahead0, 200)
        self.assertTrue(r.filled); self.assertAlmostEqual(r.fill_time, T0 + 5)

    def test_cancellations_ahead_advance_the_queue(self):
        rows = base_book() + [(T0 + 2, 3, 10, 100, 1010000, -1), (T0 + 3, 2, 11, 60, 1010000, -1),
                              (T0 + 4, 1, 14, 100, 1010000, -1), (T0 + 5, 4, 11, 40, 1010000, -1),
                              (T0 + 6, 4, 14, 10, 1010000, -1)]
        r, _ = run(rows, T0 + 1.0)
        self.assertTrue(r.filled); self.assertAlmostEqual(r.fill_time, T0 + 6)

    def test_not_filled_while_orders_ahead_remain(self):
        rows = base_book() + [(T0 + 2, 4, 10, 100, 1010000, -1), (T0 + 3, 4, 11, 50, 1010000, -1)]
        r, _ = run(rows, T0 + 1.0)
        self.assertFalse(r.filled); self.assertEqual(r.exit_reason, "timeout")

    def test_better_price_on_own_side_cancels(self):
        rows = base_book() + [(T0 + 2, 1, 15, 100, 1009900, -1), (T0 + 3, 4, 15, 100, 1009900, -1)]
        r, _ = run(rows, T0 + 1.0)
        self.assertFalse(r.filled); self.assertEqual(r.exit_reason, "improved_away")

    def test_trade_through_fills(self):
        rows = base_book() + [(T0 + 2, 4, 10, 100, 1010000, -1), (T0 + 2, 4, 11, 100, 1010000, -1),
                              (T0 + 2, 4, 12, 50, 1010100, -1)]
        r, _ = run(rows, T0 + 1.0)
        self.assertTrue(r.filled); self.assertEqual(r.exit_reason, "filled_through")

    def test_level_emptied_by_cancellations_exits(self):
        rows = base_book() + [(T0 + 2, 3, 10, 100, 1010000, -1), (T0 + 3, 3, 11, 100, 1010000, -1)]
        r, _ = run(rows, T0 + 1.0)
        self.assertFalse(r.filled); self.assertEqual(r.exit_reason, "level_emptied")

    def test_bid_side_and_markout_sign(self):
        rows = base_book() + [(T0 + 2, 1, 20, 100, 1000000, 1), (T0 + 3, 4, 1, 300, 1000000, 1),
                              (T0 + 4, 4, 20, 100, 1000000, 1), (T0 + 5, 1, 21, 100, 999900, 1)]
        r, bbo = run(rows, T0 + 1.0, side=1)
        self.assertTrue(r.filled); self.assertAlmostEqual(r.fill_time, T0 + 4)
        out = QS.markouts(pd.DataFrame([r]), bbo)
        # after the fill the bid falls to 99.99 while the ask stays 101.00: mid 100.495 below the fill price 100.00? no:
        # bought at 100.00; later mid = (99.99 + 101.00) / 2 = 100.495 -> provider gains
        self.assertGreater(out["markout_10s_bps"].iloc[0], 0)

    def test_post_waits_for_messages_at_same_timestamp(self):
        rows = base_book() + [(T0 + 1.0, 1, 30, 100, 1010000, -1), (T0 + 2, 4, 10, 100, 1010000, -1),
                              (T0 + 3, 4, 11, 100, 1010000, -1), (T0 + 4, 4, 30, 100, 1010000, -1),
                              (T0 + 5, 1, 31, 100, 1010000, -1), (T0 + 6, 4, 31, 100, 1010000, -1)]
        r, _ = run(rows, T0 + 1.0)
        self.assertEqual(r.ahead0, 300)                     # order 30 at the same timestamp is ahead
        self.assertAlmostEqual(r.fill_time, T0 + 6)


if __name__ == "__main__":
    unittest.main()
