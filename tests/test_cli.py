"""End-to-end tests for the command-line interface."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from test_analysis import sample_customers
from flo_customer_segmentation.cli import main


class CliTests(unittest.TestCase):
    def test_cli_writes_campaign_lists_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "customers.csv"
            output_dir = root / "output"
            sample_customers().to_csv(input_path, index=False)

            result = main(
                [
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                    "--analysis-date",
                    "2021-06-03",
                ]
            )

            self.assertEqual(result, 0)
            expected_files = {
                "new_brand_target_customer_ids.csv",
                "discount_target_customer_ids.csv",
                "segment_summary.csv",
            }
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                expected_files,
            )
            target_ids = pd.read_csv(
                output_dir / "new_brand_target_customer_ids.csv"
            )
            self.assertEqual(target_ids.columns.tolist(), ["master_id"])

    def test_cli_reports_missing_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_path = Path(temporary_directory) / "missing.csv"
            result = main(["--input", str(missing_path)])

        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
