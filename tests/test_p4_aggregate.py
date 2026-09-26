import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src_py"))
import p4_aggregate as AG  # noqa: E402

T0 = 34200.0


def bursts(**kw):
    n = len(kw["t_b"])
    b = dict(t_b=np.asarray(kw["t_b"], float), t_e=np.asarray(kw.get("t_e", np.asarray(kw["t_b"]) + 30), float),
             side=np.asarray(kw["side"], np.int64), vol=np.asarray(kw.get("vol", [100.0] * n), float),
             n=np.full(n, 5.0), mode_size=np.asarray(kw.get("mode_size", [np.nan] * n), float),
             mode_count=np.asarray(kw.get("mode_count", [0.0] * n), float),
             m_ref=np.full(n, 100.0), peak_raw=np.asarray(kw.get("peak_raw", [0.05] * n), float),
             dmean=np.asarray(kw.get("dmean", [0.0] * n), float), d60=np.zeros(n), d600=np.zeros(n),
             m_dec=np.asarray(kw.get("m_dec", [100.0] * n), float), spread_b=np.ones(n), spread_dec=np.ones(n),
             m_pre30=np.full(n, 100.0), ps_t=np.asarray(kw.get("ps_t", np.asarray(kw["t_b"])), float),
             ps_mref=np.full(n, 100.0), ps_peak_raw=np.asarray(kw.get("ps_peak_raw", [0.05] * n), float),
             ps_dmean=np.asarray(kw.get("ps_dmean", [0.0] * n), float),
             ps_m_dec=np.asarray(kw.get("ps_m_dec", [100.0] * n), float))
    b["t_dec"] = np.maximum(b["t_b"] + 600, b["t_e"] + 10)
    b["ps_t_dec"] = np.maximum(b["ps_t"] + 600, b["ps_t"] + (b["t_e"] - b["t_b"]) + 10)
    b["peak"] = np.maximum(b["peak_raw"], AG.TICK)
    b["ps_peak"] = np.maximum(b["ps_peak_raw"], AG.TICK)
    return b


class LinkTest(unittest.TestCase):
    def test_links(self):
        b = bursts(t_b=T0 + np.array([0, 100, 1000, 5000, 200, 300]), side=[1, 1, -1, 1, 1, 1],
                   mode_size=[137, 137, 137, 137, 200, 55], mode_count=[2, 3, 2, 2, 2, 1])
        valid, same, opp, back = AG.links(b)
        np.testing.assert_array_equal(valid, [True, True, True, True, False, False])   # 200 is round; count 1
        np.testing.assert_array_equal(same, [True, True, False, False, False, False])  # 5000 s is outside 30 min
        np.testing.assert_array_equal(opp, [True, True, True, False, False, False])
        np.testing.assert_array_equal(back, [0, 1, 0, 0, 0, 0])


class NameDayTest(unittest.TestCase):
    def test_classes_signals_and_pseudo(self):
        # bursts: 0 large informative early; 1 large not informative; 2 small; 3 large informative but t_dec after 15:50
        b = bursts(t_b=[T0 + 100, T0 + 200, T0 + 300, 56800.0], side=[1, -1, 1, 1], vol=[1000, 900, 10, 1000],
                   peak_raw=[0.10, 0.10, 0.10, 0.10], dmean=[0.06, 0.01, 0.06, 0.06],
                   m_dec=[100.0, 100.0, 100.0, 100.0], ps_dmean=[0.0, 0.08, 0.0, 0.0], ps_peak_raw=[0.1, 0.1, 0.1, 0.1],
                   ps_t=[T0 + 5000, T0 + 6000, T0 + 7000, T0 + 8000])
        day = dict(mid_close=100.5, spread_close=2.0)
        crow = dict(open_next_adj=101.0, close_next_adj=99.0, adv20=1e5, dlyopen=99.0)
        row, strata, samp = AG.name_day(b, day, crow, threshold=500.0, fam="T", permno=1, date="20200102", cap=1000)
        self.assertEqual(row["n_large"], 3)
        self.assertEqual(row["S_all_1550"], 1000 - 900 + 10)          # burst 3 decides after 15:50
        self.assertEqual(row["S_large_1550"], 100)
        self.assertEqual(row["S_info_1550_k50"], 1000)
        self.assertEqual(row["S_pseudo_1550_k50"], -900)              # burst 1's pseudo is informative
        self.assertEqual(row["n_dclose_info_k50"], 1)
        self.assertAlmostEqual(row["sum_dclose_info_k50"], 50.0)      # +0.5 on 100 = 50 bps
        self.assertAlmostEqual(row["sum_dclose_non_k50"], -50.0)      # sell burst, price up
        self.assertAlmostEqual(row["sum_dopen_info_k50"], 100.0)
        self.assertAlmostEqual(row["sum_dcc_non_k50"], 100.0)
        self.assertAlmostEqual(row["sum_dclose_pseudo_k50"], -50.0)
        self.assertEqual(len(samp), 2)
        np.testing.assert_allclose(samp.phi_close, [5.0, -5.0])


class EndToEndTest(unittest.TestCase):
    def test_trailing_threshold_and_crsp_join(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz = os.path.join(tmp, "npz", "7"); os.makedirs(npz)
            dates = ["202001%02d" % d for d in (2, 3, 6, 7, 8, 9, 10)]
            for i, d in enumerate(dates):
                vol = np.array([10.0, 20.0, 30.0, 40.0, 50.0]) * (1 + i)
                arrays = {"T_" + k: v for k, v in dict(
                    t_b=T0 + np.arange(5) * 10.0, t_e=T0 + np.arange(5) * 10.0 + 5, side=np.ones(5, np.int8),
                    n=np.full(5, 3.0), vol=vol, mode_size=np.full(5, np.nan), mode_count=np.zeros(5),
                    truncated_share=np.zeros(5), hidden_share=np.zeros(5), program_score=np.zeros(5),
                    m_ref=np.full(5, 50.0), peak_raw=np.full(5, 0.02), d60=np.zeros(5), dmean=np.full(5, 0.02),
                    d600=np.zeros(5), m_dec=np.full(5, 50.0), spread_b=np.ones(5), spread_dec=np.ones(5),
                    m_pre30=np.full(5, np.nan), ps_t=T0 + 1000 + np.arange(5), ps_mref=np.full(5, 50.0),
                    ps_peak_raw=np.full(5, 0.02), ps_dmean=np.zeros(5), ps_m_dec=np.full(5, 50.0)).items()}
                day = dict(mid_close=50.5, mid_1530=50.0, mid_1550=50.2, mid_open=49.0, buy_1550=1.0, sell_1550=2.0,
                           spread_close=2.0, spread_1530=2.0, spread_1550=2.0)
                np.savez(os.path.join(npz, d + ".npz"), day_json=np.array(json.dumps(day)), grid_mid=np.zeros(390, np.float32), **arrays)
            crsp = pd.DataFrame(dict(permno=7, date=dates, dlyopen=50.0, dlyclose=50.0, dlyprc=50.0, dlyret=0.0,
                                     dlyvol=1e5, shrout=1000, dlycap=5e4, dlycumfacpr=2.0, primaryexch="Q"))
            # 2:1 split on the last day: CRSP's cumulative factor is 2 before it and 1 from it on
            crsp.loc[crsp.date == "20200110", ["dlyopen", "dlyclose", "dlycumfacpr"]] = [25.0, 25.0, 1.0]
            out = os.path.join(tmp, "parts"); os.makedirs(out)
            AG.process_permno((7, dates, os.path.join(tmp, "npz"), crsp, out, 1000, None))
            nd = pd.read_csv(os.path.join(out, "nameday_7.csv.gz"), dtype={"date": str})
        t = nd[nd.family == "T"].set_index("date")
        self.assertFalse(t.loc["20200108", "has_large"])            # 4 prior days
        self.assertTrue(t.loc["20200109", "has_large"])             # 5 prior days
        prior = np.concatenate([np.array([10.0, 20, 30, 40, 50]) * (1 + i) for i in range(5)])
        self.assertAlmostEqual(t.loc["20200109", "threshold"], np.quantile(prior, 0.8))
        # the split-day open of 25 is worth 25 * 2 / 1 = 50 on day-t basis: d_open = (50 - 50) / 50 = 0
        self.assertAlmostEqual(t.loc["20200109", "sum_dopen_all"], 0.0)
        self.assertEqual(t.loc["20200109", "n_dopen_all"], 5)
        self.assertTrue(np.isnan(t.loc["20200110", "clop"]))        # no next day in the frame


class PredictionSignalTest(unittest.TestCase):
    def test_s_pred_uses_threshold_and_clock(self):
        import p4_phase2 as P2
        b = bursts(t_b=[T0 + 100, T0 + 200, 55500.0], side=[1, -1, 1], vol=[1000, 900, 800],
                   peak_raw=[0.1, 0.1, 0.1], dmean=[0.06, -0.02, 0.05])
        b["truncated_share"] = np.zeros(3); b["program_score"] = np.zeros(3)
        feats = P2.COMMON + P2.FAMILY_EXTRA["T"]
        k = len(feats)
        coef = [0.0] * k; coef[feats.index("dmean_bps")] = 1.0
        spec = dict(kind="ridge", features=feats, lo=[-1e9] * k, hi=[1e9] * k, mu=[0.0] * k, sd=[1.0] * k,
                    coef=coef, intercept=0.0, theta=0.0)
        crow = dict(open_next_adj=100.0, close_next_adj=100.0, adv20=1e5, dlyopen=100.0)
        row, _, _ = AG.name_day(b, dict(mid_close=100.0, spread_close=2.0), crow, threshold=500.0, fam="T", permno=1, date="20200102",
                                cap=0, models={("T", "ridge"): spec})
        self.assertEqual(row["S_pred_ridge_1550"], 1000 + 800)     # the sell burst's dmean is negative
        self.assertEqual(row["S_pred_ridge_1530"], 1000)           # burst 3 decides at 15:35


class ValidityTest(unittest.TestCase):
    def test_stub_quotes_and_early_close(self):
        b = bursts(t_b=[T0 + 100, T0 + 200], side=[1, 1], vol=[1000, 1000], dmean=[0.06, 0.06])
        b["spread_dec"] = np.array([2.0, 900.0])                      # the second decision mid is a stub quote
        crow = dict(open_next_adj=101.0, close_next_adj=99.0, adv20=1e5, dlyopen=99.0)
        row, _, _ = AG.name_day(b, dict(mid_close=100.5, spread_close=2.0), crow, 500.0, "T", 1, "20200102", 0)
        self.assertEqual(row["S_info_1550_k50"], 1000)
        self.assertEqual(row["n_dclose_info_k50"], 1)
        row, _, _ = AG.name_day(b, dict(mid_close=100.5, spread_close=900.0), crow, 500.0, "T", 1, "20200102", 0)
        self.assertEqual(row["n_dclose_info_k50"], 0)                 # the close mid is invalid
        self.assertEqual(row["n_dopen_info_k50"], 1)                  # CRSP next open still usable
        ctl = AG.day_controls(dict(mid_1550=50.0, spread_1550=900.0, mid_1530=50.0, spread_1530=3.0,
                                   mid_close=50.0, spread_close=3.0), dict(dlyopen=49.0))
        self.assertFalse(ctl["valid_1550"]); self.assertTrue(np.isnan(ctl["own_open_1550"]))
        self.assertTrue(ctl["valid_1530"])
        self.assertIn("20191224", AG.EARLY_CLOSE)


class PseudoPlaceboTest(unittest.TestCase):
    def test_pseudo_requires_known_real_burst_and_later_window(self):
        # 0: late real burst (decides at 15:50) with an early pseudo window -> excluded: the window precedes the burst
        #    (a window starting after the burst and deciding by the clock implies the real burst decided by the clock)
        # 1: pseudo window before the real burst -> excluded (its displacement contains the real burst's impact)
        # 2: pseudo after the real burst, both decided before 15:50 -> counted
        b = bursts(t_b=[56700.0 - 600 + 300, T0 + 5000, T0 + 100], side=[1, 1, -1], vol=[1000, 1000, 1000],
                   ps_t=[T0 + 10, T0 + 100, T0 + 3000], ps_dmean=[0.08, 0.08, 0.08], ps_peak_raw=[0.1, 0.1, 0.1])
        crow = dict(open_next_adj=100.0, close_next_adj=100.0, adv20=1e5, dlyopen=100.0)
        row, _, _ = AG.name_day(b, dict(mid_close=100.2, spread_close=2.0), crow, 500.0, "T", 1, "20200102", 0)
        self.assertEqual(row["S_pseudo_1550_k50"], -1000)
        self.assertEqual(row["n_dclose_pseudo_k50"], 1)
        self.assertAlmostEqual(row["sum_dclose_pseudo_k50"], -20.0)    # sell side, close 0.2 above the pseudo decision mid


if __name__ == "__main__":
    unittest.main()
