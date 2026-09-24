import sys
import tempfile
import unittest
from pathlib import Path

# Allow running from the repository root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from milan_forecast.data import extract_areas, memory_probe, rank_areas, raw_files
from milan_forecast.experiments import TEST_START, TRAIN_END, metrics, split_windows


class DataTests(unittest.TestCase):
    def test_incomplete_daily_collection_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "sms-call-internet-mi-2013-11-01.txt").write_text("")
            with self.assertRaisesRegex(ValueError, "61 files"):
                raw_files(Path(temporary))

    def test_country_aggregation_timezone_and_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            file = Path(temporary) / "sms-call-internet-mi-2013-11-01.txt"
            # Nov 1 at 00:00 in Rome is Oct 31 at 23:00 UTC.
            stamp = int(pd.Timestamp("2013-11-01", tz="Europe/Rome").timestamp() * 1000)
            file.write_text(f"4159\t{stamp}\t39\t0\t0\t0\t0\t1.5\n"
                            f"4159\t{stamp}\t33\t0\t0\t0\t0\t2.5\n"
                            f"4556\t{stamp}\t39\t0\t0\t0\t0\t3\n")
            totals, _ = rank_areas([file], chunk_size=2)
            frame, coverage = extract_areas([file], [4159, 4556], chunk_size=2)
            self.assertEqual(totals.loc[4159], 4)
            self.assertEqual(frame.iloc[0][4159], 4)
            self.assertTrue(np.isnan(frame.iloc[1][4159]))
            self.assertEqual(coverage["4159"]["observed_intervals"], 1)
            self.assertGreater(memory_probe(file)["reduction_percent"], 0)

    def test_chronology_and_no_target_leakage(self):
        index = pd.date_range("2013-11-01", "2014-01-01", freq="10min", inclusive="left", tz="Europe/Rome")
        series = pd.Series(np.arange(len(index), dtype=float), index=index)
        X, y, times, split = split_windows(series)
        self.assertEqual(X[0, -1], y[0] - 1)
        self.assertTrue(all(times[split["train"]] < TRAIN_END))
        self.assertTrue(all(times[split["test"]] >= TEST_START))
        self.assertEqual(int(split["test"].sum()), 7 * 144)

    def test_mape_excludes_zero_only(self):
        result = metrics(np.array([0., 2.]), np.array([1., 1.]))
        self.assertEqual(result["mae"], 1)
        self.assertEqual(result["mape_percent"], 50)
        self.assertEqual(result["mape_nonzero_count"], 1)


if __name__ == "__main__":
    unittest.main()
