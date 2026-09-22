import subprocess
import unittest
import tempfile
from pathlib import Path


class DetectorCliTests(unittest.TestCase):
    def test_future_return_gate_is_rejected_before_reading_data(self):
        binary = Path(__file__).resolve().parents[1] / "data_processor"
        self.assertTrue(binary.exists(), "Build the detector with make first")
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([str(binary), "/nonexistent", str(Path(folder) / "out.csv"), "-k", "0.5"],
                                    text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("nonzero -k is disabled", result.stderr)
