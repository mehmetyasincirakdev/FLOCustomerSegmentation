"""Tests for RFM analysis and campaign targeting."""

import unittest
from datetime import date, timedelta

import pandas as pd

from flo_customer_segmentation.analysis import (
    create_rfm,
    create_segment_summary,
    create_target_customers,
    prepare_data,
)


def sample_customers() -> pd.DataFrame:
    analysis_date = date(2021, 6, 1)
    return pd.DataFrame(
        {
            "master_id": [f"customer-{index}" for index in range(10)],
            "last_order_date": [
                analysis_date - timedelta(days=index + 1) for index in range(10)
            ],
            "order_num_total_ever_online": list(range(1, 11)),
            "order_num_total_ever_offline": [0] * 10,
            "customer_value_total_ever_online": [100.0 * (index + 1) for index in range(10)],
            "customer_value_total_ever_offline": [0.0] * 10,
            "interested_in_categories_12": [
                "KADIN" if index % 2 == 0 else "ERKEK"
                for index in range(10)
            ],
        }
    )


class AnalysisTests(unittest.TestCase):
    def test_prepare_data_returns_copy_and_calculates_totals(self) -> None:
        source = sample_customers()
        prepared = prepare_data(source)

        self.assertNotIn("order_num_total", source.columns)
        self.assertEqual(prepared.loc[0, "order_num_total"], 1)
        self.assertEqual(prepared.loc[0, "customer_value_total"], 100.0)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(prepared["last_order_date"]))

    def test_rfm_scores_are_complete_and_use_dynamic_analysis_date(self) -> None:
        source = sample_customers()
        rfm = create_rfm(source)

        self.assertEqual(len(rfm), len(source))
        self.assertEqual(rfm["recency"].min(), 2)
        self.assertEqual(rfm["recency"].max(), 11)
        self.assertTrue(rfm["segment"].notna().all())
        self.assertTrue(rfm["RF_SCORE"].str.fullmatch(r"[1-5]{2}").all())
        self.assertTrue(rfm["RFM_SCORE"].str.fullmatch(r"[1-5]{3}").all())

    def test_rfm_handles_tied_values(self) -> None:
        source = sample_customers()
        source["order_num_total_ever_online"] = 1
        source["customer_value_total_ever_online"] = 100

        rfm = create_rfm(source, analysis_date=date(2021, 6, 1))

        self.assertEqual(rfm["frequency_score"].nunique(), 5)
        self.assertEqual(rfm["monetary_score"].nunique(), 5)

    def test_missing_columns_and_too_few_customers_are_reported(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            prepare_data(pd.DataFrame({"master_id": ["customer-1"]}))
        with self.assertRaisesRegex(ValueError, "At least five customers"):
            create_rfm(sample_customers().head(4))

    def test_campaign_rules_and_summary(self) -> None:
        prepared = prepare_data(sample_customers())
        rfm = create_rfm(prepared)
        rfm["segment"] = [
            "champions",
            "loyal_customers",
            "loyal_customers",
            "cant_loose",
            "hibernating",
            "new_customers",
            "at_Risk",
            "champions",
            "cant_loose",
            "hibernating",
        ]
        rfm["monetary"] = [300, 250, 500, 300, 100, 200, 400, 900, 600, 700]
        prepared["interested_in_categories_12"] = [
            "KADIN",
            "KADIN",
            "KADIN",
            "ERKEK",
            "ÇOCUK",
            "COCUK",
            "ERKEK",
            None,
            "KADIN",
            "ÇOCUK",
        ]

        targets = create_target_customers(prepared, rfm)

        self.assertEqual(
            targets["new_brand_target_customer_ids"].tolist(),
            ["customer-0", "customer-2"],
        )
        self.assertEqual(
            targets["discount_target_customer_ids"].tolist(),
            ["customer-3", "customer-4", "customer-5", "customer-9"],
        )
        summary = create_segment_summary(rfm)
        self.assertEqual(summary["customer_count"].sum(), len(rfm))
        self.assertIn("monetary_mean", summary.columns)


if __name__ == "__main__":
    unittest.main()
