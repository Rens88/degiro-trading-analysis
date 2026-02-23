from __future__ import annotations

import numpy as np
import pandas as pd

from src.plots import (
    build_cash_allocation_figure,
    build_drawdown_figure,
    build_normalized_median_figure,
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
