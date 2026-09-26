import os
import sys
import unittest

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src_py"))

import strict_continuation_common as SC
import strict_continuation_oos_day as OOS
import two_avenue_evaluate as EV


class StrictContinuationTest(unittest.TestCase):
    def test_controls_are_price_free_complete_and_lagged(self):
        self.assertTrue(set(EV.BASE_FEATURES).issubset(SC.CONTROL_FEATURES))
        self.assertIn("prior_count_imbalance_60s", SC.CONTROL_FEATURES)
        self.assertIn("prior_volume_imbalance_300s", SC.CONTROL_FEATURES)
        self.assertEqual(SC.AUGMENTED_FEATURES[-1], "fragment_score")
        self.assertFalse(any(x.startswith("future_") for x in SC.CONTROL_FEATURES))

    def test_signed_log_volume_target(self):
        frame = pd.DataFrame({"future_volume_imbalance_60s": [-9.0, 0.0, 9.0]})
        values = SC.target_values(frame, "volume_60s")
        np.testing.assert_allclose(values, [-np.log(10.0), 0.0, np.log(10.0)])

    def test_nested_oos_summary_emits_incremental_loss(self):
        rng = np.random.default_rng(19)
        n = 120
        frame = pd.DataFrame(index=np.arange(n))
        for name in SC.CONTROL_FEATURES:
            if name in ("depth_imbalance_start",):
                frame[name] = rng.uniform(-0.8, 0.8, n)
            elif name.startswith("prior_") and "imbalance" in name:
                frame[name] = rng.normal(0.0, 20.0, n)
            else:
                frame[name] = rng.lognormal(1.0, 0.4, n)
        frame["fragment_score"] = rng.uniform(0.0, 1.0, n)
        signal = 4.0 * frame.fragment_score.to_numpy() + rng.normal(0.0, 0.1, n)
        frame["future_count_imbalance_60s"] = signal
        frame["future_count_imbalance_300s"] = signal
        frame["future_volume_imbalance_60s"] = np.expm1(np.abs(signal)) * np.sign(signal)
        frame["future_volume_imbalance_300s"] = frame["future_volume_imbalance_60s"]
        frozen = {"score_top_decile": 0.8, "targets": {}}
        for target in SC.TARGETS:
            y = SC.target_values(frame, target)
            base = EV.Ridge(1.0).fit(EV._matrix(frame, SC.CONTROL_FEATURES), y)
            aug = EV.Ridge(1.0).fit(EV._matrix(frame, SC.AUGMENTED_FEATURES), y)
            frozen["targets"][target] = {
                "base": base.to_dict(SC.CONTROL_FEATURES),
                "augmented": aug.to_dict(SC.AUGMENTED_FEATURES),
            }
        result = OOS.summarize(frame, "TEST", 20250102, frozen).iloc[0]
        self.assertGreater(result["n_selected"], 0)
        self.assertGreater(result["count_60s_delta_mse"], 0.0)


if __name__ == "__main__":
    unittest.main()
