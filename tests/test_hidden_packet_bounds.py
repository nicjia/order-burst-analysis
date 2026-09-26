import os
import sys
import unittest

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src_py"))

import hidden_packet_bounds as HP
import aggregate_hidden_packet_bounds as AG


class HiddenPacketBoundsTest(unittest.TestCase):
    def test_packet_categories_ignore_type5_direction_and_bound_unsigned(self):
        packets = pd.DataFrame({
            "packet_id": [0, 1, 2, 3],
            "time": [34300.0, 34310.0, 34320.0, 34330.0],
            "sign": [1, -1, 0, 0],
            "n_hidden": [1, 1, 1, 1], "n_visible": [1, 0, 0, 0],
            "pre_bid": [99.0] * 4, "pre_ask": [101.0] * 4,
            "vwap": [101.0, 98.0, 100.5, 100.0],
            "min_price": [101.0, 98.0, 100.5, 100.0],
            "max_price": [101.0, 98.0, 100.5, 100.0],
            "hidden_volume": [10.0, 20.0, 30.0, 40.0],
        })
        bt = np.array([34200.0, 37000.0])
        mid = np.array([100.0, 100.0]); bid = np.array([99.0, 99.0])
        ask = np.array([101.0, 101.0]); size = np.array([100.0, 100.0])
        context = (bt, mid, bid, ask, size, size, np.zeros(2), (np.array([]),) * 4)
        row = HP.summarize(packets, context, "TEST", 20250102).iloc[0]
        self.assertEqual(int(row.n_hidden_packets), 4)
        self.assertEqual(int(row.n_mixed_packets), 1)
        self.assertEqual(int(row.n_outside_packets), 1)
        self.assertEqual(int(row.n_unsigned_packets), 2)
        self.assertEqual(int(row.n_unsigned_away_packets), 1)
        self.assertEqual(int(row.n_midpoint_packets), 1)
        self.assertEqual(int(row.n_known_3m), 2)
        self.assertLessEqual(float(row.bound_all_lo_3m), float(row.bound_all_hi_3m))

    def test_aggregator_refuses_a_missing_period(self):
        only_2024 = pd.DataFrame({"date": [20240102, 20240103], "ticker": ["A", "B"]})
        with self.assertRaises(ValueError):
            AG.split_periods(only_2024)
        both = pd.DataFrame({"date": [20240102, 20250102], "ticker": ["A", "B"]})
        self.assertEqual(set(AG.split_periods(both)), set(AG.PERIODS))


if __name__ == "__main__":
    unittest.main()
