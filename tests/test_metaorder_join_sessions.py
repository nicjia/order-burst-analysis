import os
import sys
import unittest

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src_py"))
import metaorder_models as legacy
import metaorder_join_v2 as corrected


class JoinSessionTests(unittest.TestCase):
    def test_reused_parent_ids_on_different_days_are_never_join_candidates(self):
        frame = pd.DataFrame({"simulation_day": [0, 0, 1, 1], "sign": [1] * 4,
                              "start_time": [1., 3., 2., 4.], "end_time": [1.1, 3.1, 2.1, 4.1],
                              "volume": [100.] * 4, "duration": [.1] * 4,
                              "spread_start": [.01] * 4, "depth_start": [100.] * 4,
                              "depth_imbalance_start": [0.] * 4, "dominant_parent": [7] * 4})
        old = legacy.join_feature_frame(frame)
        self.assertEqual(len(old), 3)
        with self.assertRaises(ValueError):
            corrected.pair_labels(frame, old)
        new = corrected.join_feature_frame(frame)
        self.assertEqual(list(zip(new.left_index, new.right_index)), [(0, 1), (2, 3)])
        self.assertEqual(corrected.pair_labels(frame, new).tolist(), [1., 1.])

    def test_campaign_ids_never_cross_sessions_even_when_every_pair_is_accepted(self):
        class AlwaysJoin:
            def predict_proba(self, x):
                return np.ones(len(x))
        frame = pd.DataFrame({"date": [20230103, 20230103, 20230104, 20230104],
                              "ticker": ["A"] * 4, "sign": [1] * 4,
                              "start_time": [1., 3., 2., 4.], "end_time": [1.1, 3.1, 2.1, 4.1],
                              "volume": [100.] * 4, "duration": [.1] * 4,
                              "spread_start": [.01] * 4, "depth_start": [100.] * 4,
                              "depth_imbalance_start": [0.] * 4})
        result = corrected.stitch_campaigns(frame, AlwaysJoin())
        self.assertEqual(result.campaign_id.iloc[0], result.campaign_id.iloc[1])
        self.assertEqual(result.campaign_id.iloc[2], result.campaign_id.iloc[3])
        self.assertNotEqual(result.campaign_id.iloc[0], result.campaign_id.iloc[2])


if __name__ == "__main__":
    unittest.main()
