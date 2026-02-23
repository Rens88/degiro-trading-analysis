from __future__ import annotations

import numpy as np
import pandas as pd

from src.plots import (
    build_benchmark_comparison_figure,
    build_cash_allocation_figure,
    build_degiro_costs_quarterly_figure,
    build_drawdown_figure,
    build_holdings_over_time_figure,
    build_normalized_median_figure,
    build_performance_over_time_figure,
    build_period_decomposition_figure,
)


def test_normalized_plot_uses_ticker_label_and_product_hover() -> None:
    index = pd.date_range("2026-01-01", periods=10, freq="D")
    prices = pd.DataFrame({"US0000000001": [100.0 + i for i in range(10)]}, index=index)
    instruments = pd.DataFrame(
        [
            {
                "instrument_id": "US0000000001",
                "ticker": "TEST",
                "product": "Test Product",
            }
        ]
    )

    fig = build_normalized_median_figure(
        prices_eur=prices,
        instruments=instruments,
        lookback_months=12,
        median_window_months=1,
    )

    assert len(fig.data) == 1
    trace = fig.data[0]
    assert trace.name == "TEST"
    assert "Product: %{customdata}" in str(trace.hovertemplate)
    assert trace.customdata[0] == "Test Product"


def test_period_decomposition_figure_has_expected_traces() -> None:
    period_df = pd.DataFrame(
        {
            "period": ["30D", "90D"],
            "net_flow_eur": [100.0, 200.0],
            "investment_pnl_eur": [50.0, -25.0],
            "total_change_eur": [150.0, 175.0],
        }
    )
    fig = build_period_decomposition_figure(period_df)
    assert len(fig.data) == 3
    assert fig.data[0].name == "Net external flow"
    assert fig.data[1].name == "Investment P/L (ex-flows)"
    assert fig.data[2].name == "Total value change"


def test_drawdown_and_cash_allocation_figures_have_data() -> None:
    index = pd.date_range("2026-01-01", periods=5, freq="D")
    metrics = pd.DataFrame(
        {
            "portfolio_value": [1000.0, 980.0, 990.0, 970.0, 1010.0],
            "cash": [200.0, 180.0, 190.0, 170.0, 210.0],
            "total_deposits": [900.0, 900.0, 900.0, 900.0, 900.0],
        },
        index=index,
    )
    drawdown_fig = build_drawdown_figure(metrics)
    cash_fig = build_cash_allocation_figure(metrics, target_cash_pct=10.0)

    assert len(drawdown_fig.data) == 2
    assert len(cash_fig.data) == 2
    assert np.isfinite(float(cash_fig.data[0].y[0]))


def test_performance_and_benchmark_figures_have_data() -> None:
    index = pd.date_range("2026-01-01", periods=5, freq="D")
    metrics = pd.DataFrame(
        {
            "portfolio_value": [1000.0, 1020.0, 1015.0, 1030.0, 1040.0],
            "total_deposits": [900.0, 905.0, 905.0, 910.0, 915.0],
            "simple_return": [0.11, 0.12, 0.12, 0.13, 0.14],
        },
        index=index,
    )
    levels = pd.DataFrame(
        {
            "MSCI World": [100.0, 101.0, 102.0, 101.5, 103.0],
            "AEX": [100.0, 99.0, 100.0, 101.0, 102.0],
        },
        index=index,
    )
    perf_fig = build_performance_over_time_figure(metrics)
    bench_fig = build_benchmark_comparison_figure(levels)
    assert len(perf_fig.data) == 3
    assert len(bench_fig.data) == 2


def test_holdings_over_time_hover_contains_cost_basis() -> None:
    index = pd.date_range("2026-01-01", periods=3, freq="D")
    positions = pd.DataFrame({"ID1": [1.0, 1.0, 0.0]}, index=index)
    prices = pd.DataFrame({"ID1": [10.0, 11.0, 12.0]}, index=index)
    cost_basis = pd.DataFrame({"ID1": [10.0, 10.0, 0.0]}, index=index)
    instruments = pd.DataFrame(
        [{"instrument_id": "ID1", "product": "Holding One", "ticker": "H1"}]
    )
    fig = build_holdings_over_time_figure(
        positions=positions,
        prices_eur=prices,
        instruments=instruments,
        cash_series=pd.Series([100.0, 105.0, 110.0], index=index),
        cost_basis_eur=cost_basis,
    )
    assert len(fig.data) >= 2
    assert "Net spent for current shares" in str(fig.data[0].hovertemplate)


def test_degiro_quarterly_costs_figure_has_stacked_dataset_traces() -> None:
    costs = pd.DataFrame(
        {
            "quarter": ["2025Q1", "2025Q1", "2025Q2", "2025Q2"],
            "dataset": ["Dataset A", "Dataset B", "Dataset A", "Dataset B"],
            "total_costs_eur": [10.0, 20.0, 15.0, 25.0],
        }
    )
    fig = build_degiro_costs_quarterly_figure(costs)
    assert len(fig.data) == 2
    assert fig.layout.barmode == "stack"


def test_degiro_quarterly_costs_figure_adds_trade_and_market_lines_on_secondary_axis() -> None:
    costs = pd.DataFrame(
        {
            "quarter": ["2025Q1", "2025Q1", "2025Q2", "2025Q2"],
            "dataset": ["Dataset A", "Dataset B", "Dataset A", "Dataset B"],
            "total_costs_eur": [10.0, 20.0, 15.0, 25.0],
            "trade_count": [2, 3, 1, 4],
            "market_count": [1, 2, 1, 2],
        }
    )
    fig = build_degiro_costs_quarterly_figure(costs)
    names = [str(trace.name) for trace in fig.data]
    assert "Trades (#)" in names
    assert "Markets (#)" in names
    assert len(fig.data) == 4
    assert fig.layout.barmode == "stack"
    assert fig.layout.yaxis2.title.text == "Count"
