"""Data preparation, RFM scoring, and campaign targeting."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

CUSTOMER_ID = "master_id"
REQUIRED_COLUMNS = {
    CUSTOMER_ID,
    "last_order_date",
    "order_num_total_ever_online",
    "order_num_total_ever_offline",
    "customer_value_total_ever_online",
    "customer_value_total_ever_offline",
    "interested_in_categories_12",
}

SEGMENTS = {
    "11": "hibernating",
    "12": "hibernating",
    "21": "hibernating",
    "22": "hibernating",
    "13": "at_Risk",
    "14": "at_Risk",
    "23": "at_Risk",
    "24": "at_Risk",
    "15": "cant_loose",
    "25": "cant_loose",
    "31": "about_to_sleep",
    "32": "about_to_sleep",
    "33": "need_attention",
    "34": "loyal_customers",
    "35": "loyal_customers",
    "44": "loyal_customers",
    "45": "loyal_customers",
    "41": "promising",
    "51": "new_customers",
    "42": "potential_loyalists",
    "43": "potential_loyalists",
    "52": "potential_loyalists",
    "53": "potential_loyalists",
    "54": "champions",
    "55": "champions",
}


def prepare_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Validate the source columns and return a prepared copy of the data."""
    missing_columns = sorted(REQUIRED_COLUMNS.difference(dataframe.columns))
    if missing_columns:
        raise ValueError(
            "Source data is missing required columns: "
            + ", ".join(missing_columns)
        )

    prepared = dataframe.copy()
    if prepared.empty:
        raise ValueError("Source data contains no customers.")
    if prepared[CUSTOMER_ID].isna().any():
        raise ValueError("Customer IDs must not be empty.")
    if prepared[CUSTOMER_ID].duplicated().any():
        raise ValueError("Customer IDs must be unique.")

    numeric_columns = [
        "order_num_total_ever_online",
        "order_num_total_ever_offline",
        "customer_value_total_ever_online",
        "customer_value_total_ever_offline",
    ]
    for column in numeric_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="raise")
        if prepared[column].isna().any():
            raise ValueError(f"Column '{column}' must not contain empty values.")
        if (prepared[column] < 0).any():
            raise ValueError(f"Column '{column}' must not contain negative values.")

    date_columns = [
        column for column in prepared.columns if column.endswith("_date")
    ]
    for column in date_columns:
        prepared[column] = pd.to_datetime(prepared[column], errors="raise")
    if prepared["last_order_date"].isna().any():
        raise ValueError("Column 'last_order_date' must not contain empty values.")

    prepared["order_num_total"] = (
        prepared["order_num_total_ever_online"]
        + prepared["order_num_total_ever_offline"]
    )
    prepared["customer_value_total"] = (
        prepared["customer_value_total_ever_online"]
        + prepared["customer_value_total_ever_offline"]
    )
    return prepared


def _score_by_quantile(values: pd.Series, labels: list[int]) -> pd.Series:
    """Assign five quantile scores while handling tied source values."""
    ranked_values = values.rank(method="first")
    return pd.qcut(ranked_values, q=5, labels=labels).astype(int)


def create_rfm(
    dataframe: pd.DataFrame,
    analysis_date: date | datetime | None = None,
) -> pd.DataFrame:
    """Calculate customer RFM metrics, scores, and segments without mutating input."""
    prepared = prepare_data(dataframe)
    latest_order_date = prepared["last_order_date"].max()
    if analysis_date is None:
        effective_analysis_date = latest_order_date + timedelta(days=2)
    else:
        effective_analysis_date = pd.Timestamp(analysis_date)

    if effective_analysis_date < latest_order_date:
        raise ValueError("Analysis date must not precede the latest order date.")
    if len(prepared) < 5:
        raise ValueError("At least five customers are required for RFM scoring.")

    rfm = pd.DataFrame({CUSTOMER_ID: prepared[CUSTOMER_ID]})
    rfm["recency"] = (
        effective_analysis_date - prepared["last_order_date"]
    ).dt.days
    rfm["frequency"] = prepared["order_num_total"]
    rfm["monetary"] = prepared["customer_value_total"]
    rfm["recency_score"] = _score_by_quantile(rfm["recency"], [5, 4, 3, 2, 1])
    rfm["frequency_score"] = _score_by_quantile(rfm["frequency"], [1, 2, 3, 4, 5])
    rfm["monetary_score"] = _score_by_quantile(rfm["monetary"], [1, 2, 3, 4, 5])
    rfm["RF_SCORE"] = (
        rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)
    )
    rfm["RFM_SCORE"] = (
        rfm["RF_SCORE"] + rfm["monetary_score"].astype(str)
    )
    rfm["segment"] = rfm["RF_SCORE"].map(SEGMENTS)
    if rfm["segment"].isna().any():
        raise ValueError("One or more RFM scores do not map to a known segment.")
    return rfm


def create_target_customers(
    prepared_data: pd.DataFrame,
    rfm: pd.DataFrame,
) -> dict[str, pd.Series]:
    """Return customer IDs for the new-brand and discount campaigns."""
    customer_segments = rfm[[CUSTOMER_ID, "segment", "monetary"]]
    customers = prepared_data.merge(
        customer_segments,
        on=CUSTOMER_ID,
        how="inner",
        validate="one_to_one",
    )
    categories = customers["interested_in_categories_12"].fillna("").astype(str)
    women_interested = categories.str.contains("KADIN", case=False, regex=False)
    men_or_children_interested = categories.str.contains(
        r"(?:ERKEK|COCUK|ÇOCUK)", case=False, regex=True
    )

    new_brand = customers.loc[
        customers["segment"].isin(["champions", "loyal_customers"])
        & (customers["monetary"] > 250)
        & women_interested,
        CUSTOMER_ID,
    ]
    discount = customers.loc[
        customers["segment"].isin(["cant_loose", "hibernating", "new_customers"])
        & men_or_children_interested,
        CUSTOMER_ID,
    ]
    return {
        "new_brand_target_customer_ids": new_brand.reset_index(drop=True),
        "discount_target_customer_ids": discount.reset_index(drop=True),
    }


def create_segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """Summarize customer counts and average RFM metrics by segment."""
    return (
        rfm.groupby("segment", as_index=False, observed=True)
        .agg(
            customer_count=(CUSTOMER_ID, "count"),
            recency_mean=("recency", "mean"),
            frequency_mean=("frequency", "mean"),
            monetary_mean=("monetary", "mean"),
        )
        .sort_values("segment")
        .reset_index(drop=True)
    )
