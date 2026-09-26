import os
import sys
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src_py"))

import burst_quality_v3 as BQ


class SignBlindTimingBlocksTest(unittest.TestCase):
    def test_boundaries_depend_only_on_times(self):
        times = np.cumsum(np.tile([0.001, 0.002, 0.010, 0.003, 0.040], 200))
        blocks_a, cutoff_a = BQ.timing_blocks(times, 0.30)
        # Signs are deliberately not an argument: even an adversarial sign permutation
        # cannot alter the selected packet geometry.
        signs = np.resize(np.array([-1, 0, 1, 1]), len(times))
        np.random.default_rng(7).shuffle(signs)
        blocks_b, cutoff_b = BQ.timing_blocks(times, 0.30)
        self.assertEqual(blocks_a, blocks_b)
        self.assertEqual(cutoff_a, cutoff_b)

    def test_requires_predeclared_minimum_size(self):
        times = np.array([0.0, 0.1, 10.0, 10.1])
        blocks, _ = BQ.timing_blocks(times, 0.50, min_packets=3)
        self.assertEqual(blocks, [])


if __name__ == "__main__":
    unittest.main()
