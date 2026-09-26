import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src_py"))

import fragment_reconstruction as FR
import metaorder_models as MM
import metaorder_simulation as MS


class MetaorderSimulationTest(unittest.TestCase):
    def test_herding_has_no_parent_answer_key(self):
        packets = MS.simulate_day(11, scenario="herding")
        self.assertTrue((packets.parent_id == -1).all())
        fragments = FR.form_fragments(packets)
        labeled = MM.add_simulated_fragment_labels(fragments, packets)
        self.assertTrue((labeled.true_fragment == 0).all())

    def test_full_scenario_contains_splits_and_partial_observation(self):
        packets = MS.simulate_day(12, scenario="full")
        self.assertTrue((packets.parent_id >= 0).any())
        self.assertTrue((packets.parent_id == -1).any())
        fragments = FR.form_fragments(packets)
        labeled = MM.add_simulated_fragment_labels(fragments, packets)
        metrics = MM.recovery_metrics(labeled, packets)
        self.assertGreater(metrics["n_fragments"], 0)
        self.assertGreaterEqual(metrics["mean_purity"], 0.0)
        self.assertLessEqual(metrics["mean_purity"], 1.0)


if __name__ == "__main__":
    unittest.main()

