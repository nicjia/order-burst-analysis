import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src_py'))
from ranked_burst_pilot import pnl


class QuoteArithmeticTests(unittest.TestCase):
    def test_flat_price_pays_both_half_spreads(self):
        for side in (-1, 1):
            self.assertAlmostEqual(pnl(100, 2, 100, 4, side), -3)

    def test_follow_and_fade_cannot_both_gain_from_same_move(self):
        # 100 -> 101, 2 bp entry spread, 4 bp exit spread.
        self.assertAlmostEqual(pnl(100, 2, 101, 4, 1), 96.98)
        self.assertAlmostEqual(pnl(100, 2, 101, 4, -1), -103.02)
        self.assertAlmostEqual(pnl(100, 2, 101, 4, 1)+pnl(100, 2, 101, 4, -1), -6.04)


if __name__ == '__main__':
    unittest.main()
