from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BASE_RED
from src.plots import (
    DISTINCT_NORMALIZED_COLORS,
    ETF_NORMALIZED_COLORS,
    NON_ETF_NORMALIZED_COLORS,
    build_benchmark_comparison_figure,
    build_cash_allocation_figure,
    build_degiro_costs_quarterly_figure,
    build_drawdown_figure,
    build_holdings_segment_pie_figure,
    build_holdings_over_time_figure,
    build_normalized_median_figure,
    build_normalized_median_window_switcher_figure,
    build_performance_over_time_figure,
    build_period_decomposition_figure,
)


def test_normalized_plot_uses_ticker_label_and_product_hover() -> None:
    index = pd.date_range("2026-01-01", periods=10, freq="D")
    prices = pd.DataFrame({"US0000000001": [100.0 + i for i in range(10)]}, index=index)
    holdings_catalog = pd.DataFrame(
        [
            {
                "instrument_id": "US0000000001",
                "ticker": "TEST",
                "product": "Test Product",
                "is_etf": True,
                "over_target_eur": 150.0,
                "is_over_target_threshold": True,
                "target_status": "Over target",
            }
        ]
    )

    fig = build_normalized_median_figure(
        prices_eur=prices,
        holdings_catalog=holdings_catalog,
        lookback_months=12,
        median_window_months=1,
    )

    assert len(fig.data) == 1
    trace = fig.data[0]
    assert trace.name == "TEST"
    assert trace.uid == "US0000000001"
    assert "Price (EUR): %{customdata[1]:,.2f}" in str(trace.hovertemplate)
    assert "Holding type: %{customdata[5]}" in str(trace.hovertemplate)
    assert "Threshold status: %{customdata[7]}" in str(trace.hovertemplate)
    assert trace.customdata[0][0] == "Test Product"
    assert float(trace.customdata[0][1]) == 100.0
    assert float(trace.customdata[0][2]) == 109.0
    assert trace.customdata[0][3] == "Jan 10, 2026"
    assert trace.customdata[0][5] == "ETF"
    assert trace.customdata[0][6] == "Over target"
    assert trace.customdata[0][7] == "Above threshold"
    assert trace.line.dash == "solid"
    assert trace.line.color in DISTINCT_NORMALIZED_COLORS
    assert float(trace.line.width) == 3.0


def test_normalized_plot_hover_includes_local_price_for_foreign_currency() -> None:
    index = pd.date_range("2026-01-01", periods=10, freq="D")
    prices_eur = pd.DataFrame({"US0000000001": [100.0 + i for i in range(10)]}, index=index)
    prices_local = pd.DataFrame({"US0000000001": [90.0 + i for i in range(10)]}, index=index)
    holdings_catalog = pd.DataFrame(
        [
            {
                "instrument_id": "US0000000001",
                "ticker": "TEST",
                "product": "Test Product",
                "currency": "USD",
                "is_etf": False,
            }
        ]
    )

    fig = build_normalized_median_figure(
        prices_eur=prices_eur,
        prices_local=prices_local,
        holdings_catalog=holdings_catalog,
        lookback_months=12,
        median_window_months=1,
    )

    trace = fig.data[0]
    assert "Price (USD): %{customdata[8]:,.2f}" in str(trace.hovertemplate)
    assert "Most recent price (USD): %{customdata[9]:,.2f}" in str(trace.hovertemplate)
    assert float(trace.customdata[0][8]) == 90.0
    assert float(trace.customdata[0][9]) == 99.0


def test_normalized_plot_styles_holdings_without_legend_groups() -> None:
    index = pd.date_range("2026-01-01", periods=10, freq="D")
    prices = pd.DataFrame(
        {
            "ETF1": np.linspace(100.0, 110.0, len(index)),
            "STOCK1": np.linspace(50.0, 55.0, len(index)),
        },
        index=index,
    )
    holdings_catalog = pd.DataFrame(
        [
            {"instrument_id": "ETF1", "ticker": "ETF1", "product": "ETF One", "is_etf": True},
            {
                "instrument_id": "STOCK1",
                "ticker": "STK1",
                "product": "Stock One",
                "is_etf": False,
                "over_target_eur": -40.0,
                "is_over_target_threshold": False,
                "target_status": "Under target",
            },
        ]
    )

    fig = build_normalized_median_figure(
        prices_eur=prices,
        holdings_catalog=holdings_catalog,
        lookback_months=12,
        median_window_months=1,
    )

    assert len(fig.data) == 2
    assert fig.data[0].name == "ETF1"
    assert fig.data[0].line.dash == "solid"
    assert fig.data[0].line.color in ETF_NORMALIZED_COLORS
    assert fig.data[1].name == "STK1"
    assert fig.data[1].line.dash == "solid"
    assert fig.data[1].line.color in NON_ETF_NORMALIZED_COLORS
    assert fig.layout.hovermode == "closest"
    assert fig.layout.legend.itemclick == "toggle"
    assert fig.layout.legend.itemdoubleclick == "toggleothers"
    assert fig.layout.legend.orientation == "v"
    assert float(fig.layout.legend.x) > 1.0
    assert fig.layout.uirevision == "normalized-median-chart"


def test_normalized_plot_uses_distinct_palette_for_homogeneous_subset() -> None:
    index = pd.date_range("2026-01-01", periods=20, freq="D")
    prices = pd.DataFrame(
        {
            "ETF1": np.linspace(100.0, 120.0, len(index)),
            "ETF2": np.linspace(90.0, 110.0, len(index)),
        },
        index=index,
    )
    holdings_catalog = pd.DataFrame(
        [
            {"instrument_id": "ETF1", "ticker": "ETF1", "product": "ETF One", "is_etf": True},
            {"instrument_id": "ETF2", "ticker": "ETF2", "product": "ETF Two", "is_etf": True},
        ]
    )

    fig = build_normalized_median_figure(
        prices_eur=prices,
        holdings_catalog=holdings_catalog,
        lookback_months=12,
        median_window_months=1,
    )

    colors = [trace.line.color for trace in fig.data]
    assert colors[0] in DISTINCT_NORMALIZED_COLORS
    assert colors[1] in DISTINCT_NORMALIZED_COLORS
    assert colors[1] not in ETF_NORMALIZED_COLORS


def test_normalized_multi_window_switcher_uses_client_side_buttons() -> None:
    index = pd.date_range("2026-01-01", periods=60, freq="D")
    prices = pd.DataFrame({"ID1": np.linspace(100.0, 140.0, len(index))}, index=index)

    figures = {
        3: build_normalized_median_figure(
            prices_eur=prices,
            instruments=None,
            lookback_months=12,
            median_window_months=3,
        ),
        6: build_normalized_median_figure(
            prices_eur=prices,
            instruments=None,
            lookback_months=12,
            median_window_months=6,
        ),
    }
    combined = build_normalized_median_window_switcher_figure(
        figures,
        default_window_months=6,
    )

    assert len(combined.data) == 2
    assert combined.data[0].visible is False
    assert combined.data[1].visible is True
    assert combined.layout.title.text == figures[6].layout.title.text
    assert float(combined.layout.title.x) == 0.01
    assert len(combined.layout.updatemenus) == 1
    buttons = list(combined.layout.updatemenus[0].buttons)
    assert [button.label for button in buttons] == ["3 months", "6 months"]
    assert float(combined.layout.updatemenus[0].x) == 1.0
    assert int(combined.layout.margin.t) >= 120


def test_normalized_plot_supports_all_time_lookback() -> None:
    index = pd.date_range("2025-01-01", periods=40, freq="D")
    prices = pd.DataFrame({"ID1": np.linspace(100.0, 140.0, len(index))}, index=index)

    fig = build_normalized_median_figure(
        prices_eur=prices,
        instruments=None,
        lookback_months=0,
        median_window_months=3,
    )

    assert len(fig.data) == 1
    trace = fig.data[0]
    assert len(trace.x) == len(index)
    assert "all-time lookback" in str(fig.layout.title.text)


def test_normalized_plot_accepts_lowpass_filter_flag_without_crashing() -> None:
    index = pd.date_range("2025-01-01", periods=60, freq="D")
    prices = pd.DataFrame({"ID1": np.linspace(100.0, 140.0, len(index))}, index=index)

    fig = build_normalized_median_figure(
        prices_eur=prices,
        instruments=None,
        lookback_months=12,
        median_window_months=3,
        apply_lowpass_filter=True,
    )

    assert len(fig.data) == 1
    assert "Butterworth" in str(fig.layout.title.text)


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


def test_holdings_segment_pie_highlights_over_target_slices() -> None:
    holdings = pd.DataFrame(
        [
            {
                "ticker": "ETF1",
                "product": "ETF One",
                "value_eur": 600.0,
                "over_target_eur": 150.0,
                "is_over_target_threshold": True,
                "target_per_holding_pct": 12.5,
            },
            {
                "ticker": "ETF2",
                "product": "ETF Two",
                "value_eur": 400.0,
                "over_target_eur": -20.0,
                "is_over_target_threshold": False,
                "target_per_holding_pct": 12.5,
            },
        ]
    )
    fig = build_holdings_segment_pie_figure(
        holdings_df=holdings,
        title="ETF holdings (% of ETF sleeve)",
        total_portfolio_value_eur=2_000.0,
        target_per_holding_pct=12.5,
    )

    assert len(fig.data) == 1
    trace = fig.data[0]
    colors = list(trace.marker.colors)
    pull_values = list(trace.pull)
    assert colors[0] == BASE_RED
    assert colors[1] != BASE_RED
    assert pull_values[0] > 0.0
    assert pull_values[1] == 0.0
    assert trace.textinfo == "label+text"
    assert "Portfolio weight:" in str(trace.hovertext[0])
    assert "ETF/non-ETF sleeve weight:" in str(trace.hovertext[0])
    assert "Over target: EUR 150.00" in str(trace.hovertext[0])
    assert fig.layout.annotations[0].text == "Target:<br>12.5%"


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
