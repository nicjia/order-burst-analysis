import os
import sys
import tempfile
import unittest

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src_py"))

import execution_packets as EP
import fragment_reconstruction as FR


class ExecutionPacketTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.NamedTemporaryFile(mode="w", suffix="_2023-01-03_message.csv",
                                                delete=False)
        rows = [
            # time, type, order id, size, price, direction
            (34200.0, 1, 1, 500, 100000, 1),
            (34200.0, 1, 2, 500, 100200, -1),
            # One economic buy packet split across visible and hidden executions.  The bogus
            # type-5 direction must be ignored and the native type-4 sign inherited.
            (34201.0, 4, 2, 40, 100200, -1),
            (34201.0, 5, 0, 60, 100150, 1),
            # Hidden-only, outside the prequote: conservatively signable buy.
            (34201.5, 5, 0, 25, 100300, 1),
            # Hidden-only inside the spread: ambiguous even though Direction=1.
            (34201.8, 5, 0, 30, 100100, 1),
        ]
        for row in rows:
            self.temp.write(",".join(str(x) for x in row) + "\n")
        self.temp.close()

    def tearDown(self):
        os.unlink(self.temp.name)

    def test_message_rows_collapse_and_hidden_direction_is_ignored(self):
        _context, packets = EP.reconstruct_packets(self.temp.name)
        self.assertEqual(len(packets), 3)
        first = packets.iloc[0]
        self.assertEqual(first.sign, 1)
        self.assertEqual(first.n_messages, 2)
        self.assertEqual(first.n_visible, 1)
        self.assertEqual(first.n_hidden, 1)
        self.assertEqual(first.volume, 100)
        self.assertEqual(first.sign_source, "native+timestamp")
        self.assertEqual(packets.iloc[1].sign, 1)
        self.assertEqual(packets.iloc[1].sign_source, "outside-prequote")
        self.assertEqual(packets.iloc[2].sign, 0)
        self.assertEqual(packets.iloc[2].sign_source, "ambiguous-hidden")

    def test_ambiguous_packet_breaks_directional_fragment(self):
        _context, packets = EP.reconstruct_packets(self.temp.name)
        fragments = FR.form_fragments(packets, gap=1.0, min_packets=2)
        self.assertEqual(len(fragments), 1)
        self.assertEqual(int(fragments.iloc[0].n_packets), 2)
        self.assertEqual(int(fragments.iloc[0].n_messages), 3)


if __name__ == "__main__":
    unittest.main()
