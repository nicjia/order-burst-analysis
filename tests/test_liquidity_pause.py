import os
import sys
import unittest

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src_py"))

import liquidity_pause_common as LP
import strict_continuation_common as SC


class LiquidityPauseTest(unittest.TestCase):
    def _fragment(self, fragment_id, end, score=0.9):
        row = {name: 1.0 for name in SC.CONTROL_FEATURES}
        row.update({"fragment_id": fragment_id, "end_time": end, "sign": 1,
                    "fragment_score": score})
        return row

    def test_nonoverlap_risk_rows_and_directional_liquidity(self):
        fragments = pd.DataFrame([
            self._fragment(0, 34210.0),
            self._fragment(1, 34220.0),  # skipped: overlaps fixed window from fragment 0
            self._fragment(2, 34520.0, score=0.2),
        ])
        packets = pd.DataFrame({
            "packet_id": [0, 1, 2], "time": [34211.5, 34212.0, 34530.0],
            "sign": [1, -1, -1], "volume": [10.0, 20.0, 30.0],
        })
        bt = np.array([34200.0, 34210.0, 34210.5, 34520.0, 34521.0])
        bid = np.array([99.0, 99.0, 98.0, 99.0, 99.0])
        ask = np.array([101.0, 101.0, 102.0, 101.0, 101.0])
        bsz = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        asz = np.array([100.0, 100.0, 50.0, 100.0, 100.0])
        mid = (bid + ask) / 2.0
        context = (bt, mid, bid, ask, bsz, asz, np.zeros(len(bt)),
                   (np.array([]),) * 4)
        risk = LP.build_risk_rows(fragments, packets, context, score_threshold=0.8)
        self.assertEqual(set(risk.fragment_id), {0, 2})
        first = risk[risk.fragment_id == 0]
        self.assertEqual(len(first), 1)
        self.assertEqual(float(first.event.iloc[0]), 1.0)
        self.assertGreater(float(first.risk_spread_change.iloc[0]), 0.0)
        self.assertLess(float(first.risk_contra_depth_change.iloc[0]), 0.0)
        self.assertEqual(float(first.selected_x_spread_change.iloc[0]),
                         float(first.risk_spread_change.iloc[0]))
        censored = risk[risk.fragment_id == 2]
        self.assertEqual(len(censored), len(LP.INTERVALS))
        self.assertEqual(float(censored.event.sum()), 0.0)

    def test_feature_sets_have_no_future_labels(self):
        self.assertTrue(set(SC.CONTROL_FEATURES).issubset(LP.BASE_FEATURES))
        self.assertFalse(any(name.startswith("future_") for name in LP.JOINT_FEATURES))
        self.assertEqual(LP.SPREAD_FEATURES[-1], "selected_x_spread_change")
        self.assertEqual(LP.DEPTH_FEATURES[-1], "selected_x_contra_depth_change")


if __name__ == "__main__":
    unittest.main()
