import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src_py"))
from burst_alt import reconstruct, mid_at, bbo_at

class BookTests(unittest.TestCase):
    def reconstruct(self, messages):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "messages.csv"
            path.write_text(messages)
            return reconstruct(path)

    def test_execution_removes_best_ask_and_hidden_print_preserves_book(self):
        ctx = self.reconstruct("34200,1,1,100,1000000,1\n34201,1,2,100,1000200,-1\n34202,1,3,70,1000300,-1\n34203,4,2,100,1000200,-1\n34204,5,4,25,1000150,1\n")
        times, mids, bids, asks, bid_sizes, ask_sizes, ofi, trades = ctx
        np.testing.assert_allclose(times, [34201, 34203])
        np.testing.assert_allclose(asks, [100.02, 100.03])
        np.testing.assert_allclose(ask_sizes, [100, 70])
        np.testing.assert_array_equal(trades[3], [False, True])
        self.assertEqual(trades[1][0], 1)

    def test_cancellation_updates_depth_without_changing_price(self):
        ctx = self.reconstruct("34200,1,1,100,1000000,1\n34201,1,2,100,1000200,-1\n34202,2,1,40,1000000,1\n")
        np.testing.assert_allclose(ctx[4], [100, 60])
        np.testing.assert_allclose(ctx[1], [100.01, 100.01])

    def test_lookup_never_uses_a_future_quote(self):
        t = np.array([1., 3.])
        q = np.array([0., 1., 2., 3.])
        np.testing.assert_allclose(mid_at(t, np.array([10., 20.]), q), [np.nan, 10., 10., 20.])
        bid, ask = bbo_at(t, np.array([9., 19.]), np.array([11., 21.]), q)
        np.testing.assert_allclose(bid, [np.nan, 9., 9., 19.])
        np.testing.assert_allclose(ask, [np.nan, 11., 11., 21.])

if __name__ == "__main__":
    unittest.main()
