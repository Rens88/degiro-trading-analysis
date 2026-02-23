from __future__ import annotations

import numpy as np
import pandas as pd

from src.reconciliation import TotalsResult
from src.tables import build_four_tables, build_monthly_starting_portfolio_value_table


def test_build_monthly_starting_portfolio_value_table_two_datasets() -> None:
    index = pd.date_range("2026-01-01", "2026-03-05", freq="D")
    a = pd.DataFrame({"portfolio_value": np.arange(len(index), dtype=float) + 100.0}, index=index)
    b = pd.DataFrame({"portfolio_value": np.full(len(index), 10.0)}, index=index)

    out = build_monthly_starting_portfolio_value_table(
        per_dataset_metrics={
            "Dataset A": a,
            "Dataset B": b,
        }
    )

    assert list(out["month_start"]) == ["2026-02-01", "2026-03-01"]
    assert list(out["valuation_date"]) == ["2026-01-31", "2026-02-28"]

    a_feb = float(a.loc["2026-01-31", "portfolio_value"])
    a_mar = float(a.loc["2026-02-28", "portfolio_value"])
    assert out.loc[0, "Dataset A_portfolio_value_eur"] == a_feb
    assert out.loc[1, "Dataset A_portfolio_value_eur"] == a_mar
    assert out.loc[0, "Dataset B_portfolio_value_eur"] == 10.0
    assert out.loc[1, "Dataset B_portfolio_value_eur"] == 10.0
    assert out.loc[0, "total_portfolio_value_eur"] == a_feb + 10.0
    assert out.loc[1, "total_portfolio_value_eur"] == a_mar + 10.0


def test_build_monthly_starting_portfolio_value_table_returns_empty_without_prior_day() -> None:
    index = pd.date_range("2026-01-01", "2026-01-31", freq="D")
    only_one_month = pd.DataFrame({"portfolio_value": np.arange(len(index), dtype=float)}, index=index)

    out = build_monthly_starting_portfolio_value_table(
        per_dataset_metrics={"Dataset A": only_one_month}
    )

    assert out.empty


def test_build_monthly_starting_portfolio_value_table_with_manual_values_and_delta() -> None:
    index = pd.date_range("2023-01-01", "2023-03-05", freq="D")
    a = pd.DataFrame({"portfolio_value": np.arange(len(index), dtype=float) + 100.0}, index=index)

    manual = pd.DataFrame(
        {
            "tracked_date": [
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-01-16"),  # latest in Jan should be used for Jan month bucket
                pd.Timestamp("2023-02-01"),
            ],
            "manual_tracked_total_eur": [1000.0, 1200.0, 1300.0],
        }
    )

    out = build_monthly_starting_portfolio_value_table(
        per_dataset_metrics={"Dataset A": a},
        manual_tracked_values=manual,
    )

    jan_row = out.loc[out["month_start"] == "2023-01-01"].iloc[0]
    feb_row = out.loc[out["month_start"] == "2023-02-01"].iloc[0]
    mar_row = out.loc[out["month_start"] == "2023-03-01"].iloc[0]

    assert jan_row["manual_source_date"] == "2023-01-16"
    assert jan_row["manual_tracked_total_eur"] == 1200.0
    assert np.isnan(jan_row["total_portfolio_value_eur"])
    assert np.isnan(jan_row["delta_total_vs_manual_eur"])

    expected_feb_total = float(a.loc["2023-01-31", "portfolio_value"])
    assert feb_row["manual_source_date"] == "2023-02-01"
    assert feb_row["manual_tracked_total_eur"] == 1300.0
    assert feb_row["total_portfolio_value_eur"] == expected_feb_total
    assert feb_row["delta_total_vs_manual_eur"] == expected_feb_total - 1300.0

    assert np.isnan(mar_row["manual_tracked_total_eur"])
    assert np.isnan(mar_row["delta_total_vs_manual_eur"])


def test_build_four_tables_uses_desired_holding_counts_for_targets() -> None:
    holdings = pd.DataFrame(
        [
            {
                "account_label": "Dataset A",
                "instrument_id": "ETF1",
                "product": "ETF One",
                "isin": "IE00TEST00001",
                "ticker": "ETF1",
                "is_etf": True,
                "quantity": 1.0,
                "value_eur": 200.0,
            },
            {
                "account_label": "Dataset A",
                "instrument_id": "STK1",
                "product": "Stock One",
                "isin": "US0000000001",
                "ticker": "STK1",
                "is_etf": False,
                "quantity": 1.0,
                "value_eur": 100.0,
            },
        ]
    )
    totals = TotalsResult(
        account_label="Combined",
        positions_value_eur=300.0,
        cash_value_eur=700.0,
        cash_source="account balance",
        total_value_eur=1000.0,
    )

    out = build_four_tables(
        holdings=holdings,
        totals=totals,
        target_etf_fraction=0.5,
        desired_etf_holdings=4,
        desired_non_etf_holdings=12,
        min_over_value_eur=0.0,
    )

    etf_row = out["etf"].iloc[0]
    non_etf_row = out["non_etf"].iloc[0]
    assert etf_row["target_per_holding_pct"] == 12.5
    assert non_etf_row["target_per_holding_pct"] == (0.5 / 12.0 * 100.0)
    assert etf_row["target_value_eur"] == 125.0
    assert round(float(non_etf_row["target_value_eur"]), 6) == round(1000.0 * (0.5 / 12.0), 6)

    over = out["over_target"]
    assert set(over["ticker"]) == {"ETF1", "STK1"}
