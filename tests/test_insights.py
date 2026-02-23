from __future__ import annotations

import pandas as pd

from src.insights import (
    build_action_plan_table,
    build_ai_generated_insights,
    build_drawdown_summary_table,
    build_monthly_performance_table,
    build_period_performance_table,
)


def _sample_metrics() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=400, freq="D")
    portfolio = pd.Series(10000.0, index=index, dtype="float64")
    deposits = pd.Series(9000.0, index=index, dtype="float64")
    cash = pd.Series(2500.0, index=index, dtype="float64")

    # Add flow and performance dynamics.
    deposits.iloc[100:] += 1000.0
    portfolio.iloc[100:] += 1200.0
    portfolio.iloc[250:] -= 800.0
    portfolio.iloc[320:] += 1400.0

    return pd.DataFrame(
        {
            "portfolio_value": portfolio,
            "total_deposits": deposits,
            "cash": cash,
        },
        index=index,
    )


def test_period_and_monthly_performance_tables_not_empty() -> None:
    metrics = _sample_metrics()
    period_df = build_period_performance_table(metrics)
    monthly_df = build_monthly_performance_table(metrics)

    assert not period_df.empty
    assert "Since start" in set(period_df["period"])
    assert not monthly_df.empty
    assert "investment_pnl_eur" in monthly_df.columns


def test_drawdown_summary_has_expected_metrics() -> None:
    metrics = _sample_metrics()
    drawdown_df = build_drawdown_summary_table(metrics)
    assert not drawdown_df.empty
    assert "Current drawdown (%)" in set(drawdown_df["metric"])
    assert "Max drawdown (%)" in set(drawdown_df["metric"])


def test_action_plan_and_bundle_generation() -> None:
    metrics = _sample_metrics()
    period_df = build_period_performance_table(metrics)
    over_target_df = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "product": ["A", "B"],
            "over_target_eur": [600.0, 200.0],
        }
    )
    action_df = build_action_plan_table(
        metrics_df=metrics,
        period_performance_df=period_df,
        over_target_df=over_target_df,
    )
    bundle = build_ai_generated_insights(metrics_df=metrics, over_target_df=over_target_df)

    assert not action_df.empty
    assert "future_action" in action_df.columns
    assert isinstance(bundle.get("summary_lines"), list)
    assert "period_performance_df" in bundle
