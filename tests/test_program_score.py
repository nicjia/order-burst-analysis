import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src_py"))
import program_score as PS  # noqa: E402


def synthetic(rng, names, n=400):
    rows = []
    for tk in names:
        for _ in range(n):
            iat_cv = rng.uniform(0, 2)
            pairs = int(rng.integers(3, 40))
            q = rng.uniform(0.002, 0.02)
            program = rng.random() < 1 / (1 + np.exp(4 * (iat_cv - 0.5)))  # regular timing => program
            p = q + (1 - q) * (0.3 if program else 0.0)
            rows.append(dict(ticker=tk, n_packets=pairs, duration=rng.uniform(1, 60), intensity=rng.uniform(0.1, 5),
                             iat_cv=iat_cv, iat_median=rng.uniform(0.1, 5), truncated_share=rng.uniform(0, 1),
                             hidden_share=rng.uniform(0, .3), spread_bps=rng.uniform(1, 10), log_exec_depth=rng.uniform(3, 8),
                             imbalance=rng.uniform(-1, 1), tod=rng.uniform(0, 1), trailing_activity=rng.integers(1, 500),
                             n_opposite=int(rng.integers(0, 5)), n_unsigned=int(rng.integers(0, 3)),
                             dm_pairs=pairs, dm_repeats=int(rng.binomial(pairs, p)), dm_expected=pairs * q))
    return pd.DataFrame(rows)


class ProgramScoreTests(unittest.TestCase):
    def test_recovers_planted_feature_and_lift_out_of_sample(self):
        rng = np.random.default_rng(4)
        train = synthetic(rng, ["A%d" % i for i in range(12)])
        test = synthetic(rng, ["B%d" % i for i in range(12)])
        for frame in (train, test):
            frame["log_packets"] = np.log1p(frame.n_packets); frame["log_duration"] = np.log1p(frame.duration)
            frame["log_intensity"] = np.log(frame.intensity); frame["log_iat_median"] = np.log1p(frame.iat_median)
            frame["log_spread"] = np.log(frame.spread_bps); frame["tod2"] = frame.tod ** 2
            frame["log_activity"] = np.log1p(frame.trailing_activity)
            total = frame.n_packets + frame.n_opposite + frame.n_unsigned
            frame["opposite_share"] = frame.n_opposite / total; frame["unsigned_share"] = frame.n_unsigned / total
        model = PS.fit(PS.usable(train))
        coef = dict(zip(["intercept"] + PS.FEATURES, model["beta"]))
        self.assertLess(coef["iat_cv"], 0)
        self.assertEqual(max(PS.FEATURES, key=lambda f: abs(coef[f])), "iat_cv")
        PS.BOOT = 50
        table, stat = PS.decile_table(PS.usable(test), PS.score(model, PS.usable(test)), np.random.default_rng(1))
        self.assertGreater(stat["top_minus_bottom_excess_per_1000_pairs"], 100)
        self.assertGreater(stat["ci95"][0], 0)
        self.assertGreater(stat["spearman_decile_excess"], 0.7)


if __name__ == "__main__":
    unittest.main()
