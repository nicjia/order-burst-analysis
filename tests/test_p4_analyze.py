import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src_py"))
import p4_analyze as A  # noqa: E402


def panel(n_names=80, n_days=120, seed=0, beta=2000.0):
    rng = np.random.default_rng(seed)
    dates = ["2020%04d" % (101 + d) for d in range(n_days)]
    rows = []
    for p in range(n_names):
        for d in dates:
            r = dict(permno=p, date=d, family="T", has_large=True, adv20=1e6, cap_lag1=1e6 * (1 + rng.random()),
                     dvol20=1e8 * (1 + rng.random()), sigma20=0.02 + 0.01 * rng.random(), turn20=0.01 * (1 + rng.random()),
                     ret_lag1=rng.normal(0, 0.02), ret_lag5=rng.normal(0, 0.04))
            for clock in ("1550", "1530"):
                r["S_info_%s_k50" % clock] = rng.normal(0, 1e4)
                for k in ("k25", "k75"):
                    r["S_info_%s_%s" % (clock, k)] = rng.normal(0, 1e4)
                    r["S_pseudo_%s_%s" % (clock, k)] = rng.normal(0, 1e4)
                r["S_pseudo_%s_k50" % clock] = rng.normal(0, 1e4)
                r["S_large_" + clock] = rng.normal(0, 2e4); r["S_all_" + clock] = rng.normal(0, 5e4)
                r["buy_" + clock] = 1e5 * rng.random(); r["sell_" + clock] = 1e5 * rng.random()
                r["own_open_" + clock] = rng.normal(0, 0.01); r["spread_" + clock] = 5 + rng.random()
            sig = r["S_info_1550_k50"] / r["adv20"]
            r["clop"] = (beta * sig + rng.normal(0, 20)) / 1e4
            r["ret_next"] = rng.normal(0, 100) / 1e4
            r["tclose"] = rng.normal(0, 30) / 1e4
            for k in A.KAPPAS:
                n = 5
                info = rng.normal(2.0, 5.0); pseudo = rng.normal(0.0, 5.0); non = rng.normal(0.5, 5.0)
                for o in ("dclose", "dopen", "dcc"):
                    r["sum_%s_info_%s" % (o, k)] = info * n; r["n_%s_info_%s" % (o, k)] = n
                    r["sum_%s_non_%s" % (o, k)] = non * n; r["n_%s_non_%s" % (o, k)] = n
                r["sum_dclose_pseudo_" + k] = pseudo * n; r["n_dclose_pseudo_" + k] = n
            for base in ("large", "all"):
                r["sum_dclose_" + base] = 1.0; r["n_dclose_" + base] = 1
            rows.append(r)
    return pd.DataFrame(rows)


class AnalyzeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nd = panel()

    def test_q1_recovers_difference(self):
        r = A.q1(self.nd, np.random.default_rng(1), n_boot=200)["T"]["k50"]
        self.assertAlmostEqual(r["d_pseudo"]["mean"], 2.0, delta=0.3)
        self.assertGreater(r["d_pseudo"]["t"], 5)
        lo, hi = r["d_pseudo_ci95"]
        self.assertLess(lo, 2.0); self.assertGreater(hi, 2.0)
        self.assertAlmostEqual(r["d_non"]["mean"], 1.5, delta=0.3)

    def test_q4_recovers_slope_and_holm(self):
        r = A.fm_cell(self.nd[self.nd.family == "T"], "CLOP")
        self.assertAlmostEqual(r["S_sig"]["mean"], 2000.0, delta=150.0)
        self.assertGreater(r["S_sig"]["t"], 3)
        self.assertLess(abs(r["S_large"]["t"]), 3.5)
        self.assertEqual(A.holm([0.001, 0.02, 0.5]), [True, True, False])   # 0.02 <= 0.05/2
        self.assertEqual(A.holm([0.001, 0.03, 0.5]), [True, False, False])
        self.assertEqual(A.holm([0.001, 0.02, 0.03]), [True, True, True])

    def test_portfolio_costs(self):
        g = self.nd[self.nd.family == "T"]
        p2 = A.portfolio(g, "CLOP", cost_side_bps=2.0)
        self.assertAlmostEqual(p2["gross"]["mean_bps"] - p2["net"]["mean_bps"], 8.0, places=6)
        self.assertGreater(p2["gross"]["mean_bps"], 0)

    def test_q2a_stratified(self):
        rows = []
        for p in range(40):
            for d in range(10):
                for h in range(3):
                    rows.append(dict(permno=p, date=str(d), family="T", hour=h, quint=0, info=1, n=10, same=3, opp=1))
                    rows.append(dict(permno=p, date=str(d), family="T", hour=h, quint=0, info=0, n=30, same=3, opp=3))
        rows.append(dict(permno=99, date="0", family="T", hour=0, quint=1, info=1, n=5, same=5, opp=0))  # no pair
        r = A.q2a(pd.DataFrame(rows), np.random.default_rng(2), n_boot=100)["T"]
        self.assertAlmostEqual(r["same"], 0.2); self.assertAlmostEqual(r["opp"], 0.0)
        self.assertAlmostEqual(r["directional"], 0.2); self.assertTrue(r["pass_directional"])
        self.assertEqual(r["names"], 40)

    def test_deflated_sharpe_and_protocol(self):
        hi = A.deflated_sharpe_prob(0.3, 750, 0.0, 3.0, 300)
        lo = A.deflated_sharpe_prob(0.02, 750, 0.0, 3.0, 300)
        self.assertGreater(hi, 0.99); self.assertLess(lo, 0.5)
        with tempfile.TemporaryDirectory() as tmp:
            A.ANALYSIS = Path(tmp)
            f = Path(tmp) / "x.csv"; f.write_text("a\n1\n")
            with self.assertRaises(SystemExit):
                A.check_protocol("TEST", [f])
            A.check_protocol("VAL", [f])
            (Path(tmp) / "VAL_primary.json").write_text("{}")
            A.check_protocol("TEST", [f])
            self.assertEqual(len((Path(tmp) / "access_log.txt").read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
