import os
import sys
import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src_py"))
import burst_information_extract as BI
import burst_recovery_diagnostic as RD
import metaorder_models as MM
import metaorder_simulation as MS


class BurstInformationTests(unittest.TestCase):
    def test_recognition_requires_observed_trigger_or_timeout(self):
        p = pd.DataFrame({"time": [34300, 34300.1, 34300.2, 34301.0],
                          "sign": [1, 1, 1, -1]})
        rows = BI.landmarks(p)
        self.assertEqual(rows[0], (0, 2, 34300.2, "third"))
        self.assertEqual(rows[1], (0, 2, 34301.0, "completion"))
        self.assertEqual(BI.landmarks(p.iloc[:3]), [rows[0]])
        p.loc[3, "time"] = 34302
        self.assertEqual(BI.landmarks(p)[1][2], 34301.2)

    def test_regime_features_do_not_read_unfinished_second(self):
        p = pd.DataFrame({"time": [34200.2, 34200.7, 34201.2], "sign": [1, 1, 1]})
        a = BI.regime_filter(p)
        p.loc[2, "sign"] = -1
        b = BI.regime_filter(p)
        np.testing.assert_equal(a[:2], b[:2])
        self.assertNotEqual(a[2], b[2])

    def test_noise_is_not_a_shared_parent(self):
        p = pd.DataFrame({"parent_id": [-1, -1, -1, 2, 2, 2]})
        m = RD.recovery(p, [(0, 3), (3, 6)], 3)
        self.assertEqual(m["pair_precision"], 0.5)
        self.assertEqual(m["pair_recall_observed"], 1.0)
        self.assertEqual(m["noise_only_fraction"], 0.5)

    def test_observational_equivalence_is_explicit(self):
        split, _ = RD.tape(19, "isolated")
        herd, _ = RD.tape(19, "herding_only")
        pd.testing.assert_frame_equal(split.drop(columns="parent_id"), herd.drop(columns="parent_id"))
        self.assertEqual(RD.blocks(split, "run"), RD.blocks(herd, "run"))
        self.assertEqual(RD.recovery(herd, RD.blocks(herd, "run"))["pair_precision"], 0)

    def test_future_tape_does_not_change_features_at_an_existing_landmark(self):
        packets = MS.simulate_day(401, "splitting", background_rate=0, n_parents=8)
        bt = np.arange(34200., 57601.)
        mid = np.full(len(bt), 100.)
        context = (bt, mid, mid - .01, mid + .01, mid * 10, mid * 10, {}, ())
        spec = json.loads((Path(__file__).resolve().parents[1] /
                           "config/metaorder_simulation_model.json").read_text())
        model = MM.RidgeLogit.from_dict(spec["fragment_model"])
        full = BI.extract_packets(packets, context, "TEST", 20230103, model, 1)
        self.assertGreater(len(full), 0)
        landmark = full[full.stage == "third"].iloc[0]
        prefix = packets[packets.time <= landmark.decision_time].copy()
        partial = BI.extract_packets(prefix, context, "TEST", 20230103, model, 1)
        matched = partial[partial.row_id == landmark.row_id].iloc[0]
        columns = list(dict.fromkeys(BI.BASE + BI.REGIME + BI.BURST + ["fragment_score"]))
        np.testing.assert_allclose(matched[columns].to_numpy(float),
                                   landmark[columns].to_numpy(float), atol=0, rtol=0)
        self.assertEqual(landmark.reference_time, landmark.decision_time + 1)


if __name__ == "__main__":
    unittest.main()
