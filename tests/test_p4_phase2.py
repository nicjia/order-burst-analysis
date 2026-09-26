import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src_py"))
import p4_phase2 as P2  # noqa: E402


class Phase2ExportTest(unittest.TestCase):
    def test_hgb_export_matches_sklearn_with_missing(self):
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
        except ImportError:
            self.skipTest("needs scikit-learn")
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20000, 6))
        X[rng.random(X.shape) < 0.05] = np.nan
        y = np.nan_to_num(X[:, 0]) * 2 - np.nan_to_num(X[:, 1]) ** 2 + np.isnan(X[:, 2]) * 3 + rng.normal(size=len(X))
        m = HistGradientBoostingRegressor(max_depth=3, max_iter=60, learning_rate=0.1, min_samples_leaf=50,
                                          early_stopping=False, random_state=1).fit(X, y)
        spec = P2.export_hgb(m)
        Xt = rng.normal(size=(5000, 6)); Xt[rng.random(Xt.shape) < 0.1] = np.nan
        np.testing.assert_allclose(P2.predict(spec, Xt), m.predict(Xt), rtol=1e-9, atol=1e-9)

    def test_ridge_spec_prediction(self):
        spec = dict(kind="ridge", lo=[-1, -1], hi=[1, 1], mu=[0, 0], sd=[1, 2], coef=[2.0, 4.0], intercept=0.5)
        X = np.array([[0.5, 2.0], [np.nan, -0.5]])
        np.testing.assert_allclose(P2.predict(spec, X), [0.5 + 1.0 + 2.0, 0.5 + 0.0 - 1.0])


if __name__ == "__main__":
    unittest.main()
