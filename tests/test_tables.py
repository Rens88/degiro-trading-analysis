from __future__ import annotations

import numpy as np
import pandas as pd

from src.reconciliation import TotalsResult
from src.tables import (
    build_four_tables,
    build_latest_valued_holdings,
    build_monthly_starting_portfolio_value_table,
    unify_holding_product_names,
)


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
        target_etf_pct=50.0,
        target_non_etf_pct=40.0,
        target_cash_pct=10.0,
        desired_etf_holdings=4,
        desired_non_etf_holdings=12,
        min_over_value_eur=0.0,
    )

    etf_row = out["etf"].iloc[0]
    non_etf_row = out["non_etf"].iloc[0]
    assert etf_row["target_per_holding_pct"] == 12.5
    assert non_etf_row["target_per_holding_pct"] == (40.0 / 12.0)
    assert etf_row["target_value_eur"] == 125.0
    assert round(float(non_etf_row["target_value_eur"]), 6) == round(1000.0 * (40.0 / 12.0 / 100.0), 6)
    summary = out["summary"]
    assert float(summary.loc[summary["metric"] == "target cash position", "value_eur"].iloc[0]) == 100.0

    over = out["over_target"]
    assert set(over["ticker"]) == {"ETF1", "STK1"}


def test_build_four_tables_zeroes_target_for_holdings_beyond_desired_count() -> None:
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
                "value_eur": 400.0,
            },
            {
                "account_label": "Dataset A",
                "instrument_id": "ETF2",
                "product": "ETF Two",
                "isin": "IE00TEST00002",
                "ticker": "ETF2",
                "is_etf": True,
                "quantity": 1.0,
                "value_eur": 300.0,
            },
            {
                "account_label": "Dataset A",
                "instrument_id": "ETF3",
                "product": "ETF Three",
                "isin": "IE00TEST00003",
                "ticker": "ETF3",
                "is_etf": True,
                "quantity": 1.0,
                "value_eur": 200.0,
            },
            {
                "account_label": "Dataset A",
                "instrument_id": "ETF4",
                "product": "ETF Four",
                "isin": "IE00TEST00004",
                "ticker": "ETF4",
                "is_etf": True,
                "quantity": 1.0,
                "value_eur": 100.0,
            },
        ]
    )
    totals = TotalsResult(
        account_label="Combined",
        positions_value_eur=1000.0,
        cash_value_eur=0.0,
        cash_source="account balance",
        total_value_eur=1000.0,
    )

    out = build_four_tables(
        holdings=holdings,
        totals=totals,
        target_etf_pct=40.0,
        target_non_etf_pct=0.0,
        target_cash_pct=60.0,
        desired_etf_holdings=2,
        desired_non_etf_holdings=0,
        min_over_value_eur=0.0,
    )

    etf = out["etf"].reset_index(drop=True)
    assert list(etf["ticker"]) == ["ETF1", "ETF2", "ETF3", "ETF4"]
    assert list(etf["target_per_holding_pct"]) == [20.0, 20.0, 0.0, 0.0]
    assert list(etf["target_value_eur"]) == [200.0, 200.0, 0.0, 0.0]
    assert list(etf["over_target_eur"]) == [200.0, 100.0, 200.0, 100.0]


def test_unify_holding_product_names_reports_and_applies_first_product() -> None:
    holdings = pd.DataFrame(
        [
            {
                "account_label": "Dataset B",
                "instrument_id": "ETF1",
                "product": "ETF One Short",
                "isin": "IE00TEST00001",
                "ticker": "ETF1",
                "is_etf": True,
                "quantity": 1.0,
                "value_eur": 100.0,
            },
            {
                "account_label": "Dataset A",
                "instrument_id": "ETF1",
                "product": "ETF One Long Name",
                "isin": "IE00TEST00001",
                "ticker": "ETF1",
                "is_etf": True,
                "quantity": 2.0,
                "value_eur": 200.0,
            },
        ]
    )

    normalized, inconsistencies = unify_holding_product_names(holdings)

    assert list(normalized["product"]) == ["ETF One Short", "ETF One Short"]
    assert len(inconsistencies) == 1

    inconsistency = inconsistencies[0]
    group_df = inconsistency["group"]
    products_df = inconsistency["products"]
    assert group_df.loc[0, "instrument_id"] == "ETF1"
    assert group_df.loc[0, "isin"] == "IE00TEST00001"
    assert group_df.loc[0, "ticker"] == "ETF1"
    assert bool(group_df.loc[0, "is_etf"]) is True
    assert list(products_df["product"]) == ["ETF One Short", "ETF One Long Name"]
    assert inconsistency["chosen_product"] == "ETF One Short"


def test_build_four_tables_aggregates_after_product_name_unification() -> None:
    holdings = pd.DataFrame(
        [
            {
                "account_label": "Dataset B",
                "instrument_id": "ETF1",
                "product": "ETF One Short",
                "isin": "IE00TEST00001",
                "ticker": "ETF1",
                "is_etf": True,
                "quantity": 1.0,
                "value_eur": 100.0,
            },
            {
                "account_label": "Dataset A",
                "instrument_id": "ETF1",
                "product": "ETF One Long Name",
                "isin": "IE00TEST00001",
                "ticker": "ETF1",
                "is_etf": True,
                "quantity": 2.0,
                "value_eur": 200.0,
            },
        ]
    )
    totals = TotalsResult(
        account_label="Combined",
        positions_value_eur=300.0,
        cash_value_eur=200.0,
        cash_source="account balance",
        total_value_eur=500.0,
    )

    out = build_four_tables(
        holdings=holdings,
        totals=totals,
        target_etf_pct=50.0,
        target_non_etf_pct=40.0,
        target_cash_pct=10.0,
        desired_etf_holdings=4,
        desired_non_etf_holdings=12,
        min_over_value_eur=0.0,
    )

    assert len(out["etf"]) == 1
    etf_row = out["etf"].iloc[0]
    assert etf_row["product"] == "ETF One Short"
    assert etf_row["quantity"] == 3.0
    assert etf_row["value_eur"] == 300.0
    assert etf_row["account_label"] == "Dataset A, Dataset B"


def test_build_latest_valued_holdings_uses_latest_price_times_quantity() -> None:
    holdings = pd.DataFrame(
        [
            {
                "account_label": "Dataset A",
                "instrument_id": "ETF1",
                "product": "ETF One",
                "isin": "IE00TEST00001",
                "ticker": "ETF1",
                "is_etf": True,
                "quantity": 2.0,
                "value_eur": -150.0,
            }
        ]
    )
    positions = pd.DataFrame({"ETF1": [3.0]}, index=[pd.Timestamp("2026-01-01")])
    prices = pd.DataFrame({"ETF1": [125.0]}, index=[pd.Timestamp("2026-01-01")])

    out = build_latest_valued_holdings(holdings, positions=positions, prices_eur=prices)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["quantity"] == 3.0
    assert row["last_px_eur"] == 125.0
    assert row["value_eur"] == 375.0
