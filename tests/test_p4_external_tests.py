import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src_py"))
import p4_external_tests as X  # noqa: E402


class ExternalTestsTest(unittest.TestCase):
    def test_cluster_ols_matches_reference(self):
        try:
            import statsmodels.api as sm
        except ImportError:
            self.skipTest("needs statsmodels")
        rng = np.random.default_rng(0)
        n, G = 2000, 80
        g = rng.integers(0, G, n)
        x = rng.normal(size=(n, 2)) + rng.normal(size=(G, 2))[g]
        y = x @ [1.5, -0.5] + rng.normal(size=G)[g] + rng.normal(size=n)
        b, V = X.cluster_ols(y, x, g)
        ref = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds=dict(groups=g))
        np.testing.assert_allclose(b, ref.params, rtol=1e-10)
        np.testing.assert_allclose(np.sqrt(np.diag(V)), ref.bse, rtol=1e-6)

    def test_fe_regression_recovers_difference(self):
        rng = np.random.default_rng(1)
        rows = []
        for p in range(300):
            for q in range(12):
                qi, qo = rng.normal(), rng.normal()
                rows.append(dict(permno=p, quarter=q, q_info=qi, q_other=qo, ret_prev=rng.normal(), log_cap_prev=rng.normal(),
                                 turn=rng.random(), y=0.3 * qi + 0.1 * qo + q * 0.5 + rng.normal()))
        r = X.fe_regression(pd.DataFrame(rows), "y", ["q_info", "q_other", "ret_prev", "log_cap_prev", "turn"])
        self.assertAlmostEqual(r["q_info"]["b"], 0.3, delta=0.05)
        self.assertAlmostEqual(r["info_minus_other"]["b"], 0.2, delta=0.07)
        self.assertTrue(r["pass"])

    def test_index_event_z(self):
        dates = pd.bdate_range("2020-01-01", periods=120)
        rows = []
        for p, kind in ((1, "add"), (2, "add"), (3, "delete"), (4, "delete")):
            for i, d in enumerate(dates):
                bump = (1.0 if kind == "add" else -1.0) if 95 <= i < 100 else 0.0
                rows.append(dict(permno=p, dt=d, x_info=np.sin(i) * 0.1 + bump, x_other=np.cos(i) * 0.1))
        g = pd.DataFrame(rows)
        ev = pd.DataFrame(dict(index="SP500", kind=["add", "add", "delete", "delete"], permno=[1, 2, 3, 4],
                               date=[dates[100].strftime("%Y%m%d")] * 4))
        r = X.test_index(g, ev)
        self.assertEqual(r["events"], 4)
        self.assertGreater(r["z_info"]["add_minus_delete"], 10)
        self.assertLess(abs(r["z_other"]["add_minus_delete"]), 1e-9)


if __name__ == "__main__":
    unittest.main()
