import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src_py"))
from metaorder_participation import participation_labels


class ParticipationTests(unittest.TestCase):
    def label(self, ids, volume=None, source="simulation"):
        p = pd.DataFrame({"parent_id": ids, "volume": volume or [1] * len(ids)})
        f = pd.DataFrame({"packet_first": [0], "packet_last": [len(ids) - 1]})
        return participation_labels(f, p, truth_source=source).iloc[0]

    def test_multiple_parents_are_participation_not_noise(self):
        r = self.label([0] * 4 + [1] * 4 + [-1] * 2)
        self.assertEqual(r.parent_packet_fraction, .8)
        self.assertEqual(r.dominant_parent_packet_fraction, .4)
        self.assertEqual(r.distinct_parents, 2)
        self.assertEqual(r.participation_class, "program_heavy")

    def test_packet_share_and_volume_share_are_distinct(self):
        r = self.label([0, -1, -1], [80, 10, 10])
        self.assertEqual(r.parent_packet_fraction, 1 / 3)
        self.assertEqual(r.parent_volume_fraction, .8)
        self.assertEqual(r.participation_class, "program_heavy")

    def test_missing_truth_cannot_become_noise(self):
        for ids in ([0, np.nan], [-2, 0], [0.5, 1]):
            with self.assertRaises(ValueError):
                self.label(ids)
        with self.assertRaises(ValueError):
            self.label([-1, -1], source="anonymous_lobster")

    def test_noise_and_mixed_are_separate(self):
        self.assertEqual(self.label([-1, -1]).participation_class, "noise_heavy")
        self.assertEqual(self.label([0, -1]).participation_class, "mixed")

    def test_fragment_bounds_cannot_silently_truncate(self):
        p = pd.DataFrame({"parent_id": [0, -1], "volume": [1, 1]})
        for a, b in [(-1, 1), (0, 2), (.5, 1), (1, 0)]:
            f = pd.DataFrame({"packet_first": [a], "packet_last": [b]})
            with self.assertRaises(ValueError):
                participation_labels(f, p, truth_source="simulation")


if __name__ == "__main__":
    unittest.main()
