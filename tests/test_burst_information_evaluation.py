import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src_py"))
from evaluate_burst_information import audit_inputs, day_stat, nonoverlap


class EvaluationTests(unittest.TestCase):
    def test_day_unit_does_not_overweight_active_names(self):
        frame = pd.DataFrame({"ticker": ["A", "A", "B", "A", "B"],
                              "date": [1, 1, 1, 2, 2]})
        # Name-day means are (2,10), then (4,8); both day means equal 6.
        result = day_stat(frame, [1, 3, 10, 4, 8])
        self.assertEqual(result["mean"], 6)
        self.assertEqual(result["n"], 2)

    def test_nonoverlap_orders_by_time_and_is_policy_independent(self):
        frame = pd.DataFrame({"ticker": ["A"] * 4, "date": [1] * 4,
                              "decision_time": [100., 0., 50., 61.],
                              "reference_time": [101., 1., 51., 62.],
                              "row_id": ["a", "b", "c", "d"],
                              "prediction": [2., -3., -9., 1.]})
        self.assertEqual(nonoverlap(frame), [1, 3])
        frame["prediction"] *= -100
        self.assertEqual(nonoverlap(frame), [1, 3])

    def test_missing_year_and_failed_receipts_are_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = dict(seen_names=["A"], heldout_names=["B"], dates=["20230103", "20240102"])
            for name in ("A", "B"):
                (root / "status" / name).mkdir(parents=True)
                (root / "status" / name / "20230103.txt").write_text("missing\n")
            with self.assertRaises(ValueError):
                audit_inputs(root, spec)
            for name in ("A", "B"):
                (root / "status" / name / "20240102.txt").write_text("download_failure\n")
            with self.assertRaises(ValueError):
                audit_inputs(root, spec)


if __name__ == "__main__":
    unittest.main()
