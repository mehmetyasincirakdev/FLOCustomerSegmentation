"""Command-line entry point for the FLO customer segmentation project."""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path
from typing import Sequence

import pandas as pd

from flo_customer_segmentation.analysis import (
    create_rfm,
    create_segment_summary,
    create_target_customers,
    prepare_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "flo_data_20k.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
LOGGER = logging.getLogger("flo_customer_segmentation")


def _parse_analysis_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Analysis date must use YYYY-MM-DD format."
        ) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate RFM segments and create FLO campaign target lists."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Source CSV path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated CSV files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--analysis-date",
        type=_parse_analysis_date,
        help="RFM analysis date in YYYY-MM-DD format; defaults to last order + 2 days.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    if not args.input.is_file():
        LOGGER.error(
            "Input file not found: %s. Place flo_data_20k.csv in data/raw/ or "
            "provide --input.",
            args.input,
        )
        return 1

    try:
        source_data = pd.read_csv(args.input)
        prepared_data = prepare_data(source_data)
        rfm = create_rfm(prepared_data, analysis_date=args.analysis_date)
        targets = create_target_customers(prepared_data, rfm)
        summary = create_segment_summary(rfm)
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        LOGGER.error("Analysis failed: %s", error)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, customer_ids in targets.items():
        output_path = args.output_dir / f"{filename}.csv"
        customer_ids.to_frame(name="master_id").to_csv(output_path, index=False)
        LOGGER.info("Saved %s customer IDs to %s", len(customer_ids), output_path)

    summary_path = args.output_dir / "segment_summary.csv"
    summary.to_csv(summary_path, index=False)
    LOGGER.info("Saved segment summary to %s", summary_path)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
