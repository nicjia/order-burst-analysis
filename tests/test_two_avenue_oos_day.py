import os
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src_py"))

import two_avenue_oos_day as OOS


class CampaignReversalTest(unittest.TestCase):
    def test_excludes_singletons_and_orders_decisions_chronologically(self):
        frame = pd.DataFrame({
            "fragment_id": [0, 1, 2, 3, 4],
            "campaign_id": [10, 10, 5, 1, 1],
            "end_time": [90.0, 100.0, 150.0, 190.0, 200.0],
            "sign": [1, 1, -1, 1, 1],
        })
        calls = []

        def fake_bbo(_bt, _bb, _ba, query):
            q = np.asarray(query, float)
            calls.append(q.copy())
            return np.full(q.shape, 100.0), np.full(q.shape, 101.0)

        context = tuple(np.array([]) for _ in range(8))
        with mock.patch.object(OOS.BA, "bbo_at", side_effect=fake_bbo):
            result = OOS._campaign_reversal(frame, context, wait_seconds=0.0)

        np.testing.assert_array_equal(calls[0], np.array([100.0, 200.0]))
        # The singleton at 150 is excluded, and the two real campaigns overlap at 5m.
        self.assertEqual(result["reversal_trades_5m"], 1)


if __name__ == "__main__":
    unittest.main()
