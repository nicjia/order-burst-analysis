import os
import sys
import unittest

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src_py"))

import fragment_reconstruction as FR


class PriorFlowStateTest(unittest.TestCase):
    def test_uses_only_packets_strictly_before_fragment_start(self):
        packets = pd.DataFrame({
            "time": [10.0, 20.0, 30.0, 40.0, 40.0],
            "sign": [1, -1, 1, -1, 1],
            "volume": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        fragments = pd.DataFrame({"start_time": [40.0], "sign": [1]})
        out = FR.attach_prior_flow_state(fragments, packets, horizons=(25.0,))
        # The 20/30 packets are in the lookback. Both packets at t=40 are excluded.
        self.assertEqual(out.loc[0, "prior_total_count_25s"], 2)
        self.assertEqual(out.loc[0, "prior_count_imbalance_25s"], 0)
        self.assertEqual(out.loc[0, "prior_total_volume_25s"], 50.0)
        self.assertEqual(out.loc[0, "prior_volume_imbalance_25s"], 10.0)


if __name__ == "__main__":
    unittest.main()
