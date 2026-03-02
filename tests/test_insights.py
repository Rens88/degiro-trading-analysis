from __future__ import annotations

import pandas as pd

from src.insights import (
    _compute_irr_regular,
    _compute_xirr,
    build_action_plan_table,
    build_ai_generated_insights,
    build_ai_spread_analysis,
    build_drawdown_summary_table,
    build_monthly_performance_table,
    build_performance_dashboard,
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


def test_performance_dashboard_has_all_time_yearly_quarterly() -> None:
    metrics = _sample_metrics()
    tx = pd.DataFrame(
        {
            "datetime": [metrics.index[20], metrics.index[40], metrics.index[200]],
            "instrument_id": ["A", "B", "A"],
            "product": ["Alpha", "Beta", "Alpha"],
            "ticker": ["ALP", "BET", "ALP"],
            "quantity": [1.0, -1.0, 2.0],
            "total_eur": [-100.0, 120.0, -210.0],
            "is_cash_like": [False, False, False],
        }
    )
    account = pd.DataFrame(
        {
            "datetime": [metrics.index[5], metrics.index[150], metrics.index[250]],
            "type": ["external_deposit", "transaction_fee", "external_withdrawal"],
            "is_external_flow": [True, False, True],
            "is_cost": [False, True, False],
            "change_eur": [500.0, -2.0, -300.0],
            "raw_change": [500.0, -2.0, -300.0],
            "currency": ["EUR", "EUR", "EUR"],
        }
    )
    benchmark_returns = pd.DataFrame(
        {
            "MSCI World": pd.Series(0.0004, index=metrics.index[1:]),
            "AEX": pd.Series(0.0002, index=metrics.index[1:]),
        }
    )
    out = build_performance_dashboard(
        metrics_df=metrics,
        transactions_df=tx,
        account_df=account,
        benchmark_returns_df=benchmark_returns,
    )
    assert not out["all_time_df"].empty
    assert not out["yearly_df"].empty
    assert not out["quarterly_df"].empty
    assert "xirr_pct" in out["all_time_df"].columns


def test_spread_analysis_outputs_style_and_next_steps() -> None:
    holdings = pd.DataFrame(
        {
            "instrument_id": ["ETF1", "STK1", "CLOSED1"],
            "product": ["MSCI World ETF", "Dividend Stock", "Closed Position"],
            "ticker": ["IWDA", "DIV1", "OLD1"],
            "currency": ["EUR", "USD", "EUR"],
            "is_etf": [True, False, False],
            "quantity": [5.0, 10.0, 0.0],
            "value_eur": [800.0, 700.0, 0.0],
        }
    )
    idx = pd.date_range("2025-01-01", periods=200, freq="D")
    prices = pd.DataFrame(
        {
            "ETF1": pd.Series(range(200), index=idx, dtype="float64") + 100.0,
            "STK1": pd.Series(range(200), index=idx, dtype="float64") + 100.0,
        }
    )
    out = build_ai_spread_analysis(
        holdings_df=holdings,
        total_value_eur=1700.0,
        cash_value_eur=200.0,
        cash_detail_df=pd.DataFrame({"currency": ["EUR"], "balance_eur": [200.0]}),
        strategy={
            "target_etf_fraction": 0.5,
            "desired_etf_holdings": 2,
            "desired_non_etf_holdings": 3,
            "target_cash_pct": 5.0,
            "max_single_holding_pct": 60.0,
            "max_top5_holdings_pct": 95.0,
            "max_single_currency_pct": 90.0,
            "max_single_industry_pct": 90.0,
            "max_pair_correlation": 0.90,
            "min_total_holdings": 2,
            "min_over_value_eur": 10.0,
            "target_currency_pct": {"EUR": 60.0, "USD": 40.0},
            "target_industry_pct": {"Broad-market ETF": 50.0, "Unclassified": 50.0},
            "target_style_pct": {"Blend": 50.0, "Dividend": 50.0},
        },
        prices_eur=prices,
        auto_append_ticker_characteristics=False,
    )
    assert "style_allocation_df" in out
    assert "what_to_do_next_df" in out
    assert isinstance(out["what_to_do_next_df"], pd.DataFrame)

    checks = out["strategy_checks_df"]
    holdings_row = checks.loc[checks["metric"] == "Total non-cash holdings (count)"].iloc[0]
    assert float(holdings_row["actual"]) == 2.0


def test_spread_analysis_uses_csv_characteristics_only(tmp_path) -> None:
    csv_path = tmp_path / "ticker_classification_complete.csv"
    csv_path.write_text(
        "\n".join(
            [
                (
                    "instrument_id,ticker,product,currency,asset_class,primary_style,secondary_factor,"
                    "gics_sector,gics_industry_group,gics_industry,gics_sub_industry"
                ),
                "ID_MATCH,IDTKR,Instrument by id,EUR,Equity,Growth,Momentum,IT,Software,CSV Industry ID,CSV Sub ID",
                "ANOTHER_ID,TKR_FALLBACK,Instrument by ticker,USD,Equity,Value,Defensive,Health,Pharma,CSV Industry Ticker,CSV Sub Ticker",
            ]
        ),
        encoding="utf-8",
    )

    holdings = pd.DataFrame(
        {
            "instrument_id": ["ID_MATCH", "UNLISTED_ID"],
            "product": ["First product", "Second product"],
            "ticker": ["DIFFERENT_TICKER", "TKR_FALLBACK"],
            "currency": ["EUR", "USD"],
            "is_etf": [False, False],
            "quantity": [5.0, 3.0],
            "value_eur": [1000.0, 500.0],
        }
    )
    out = build_ai_spread_analysis(
        holdings_df=holdings,
        total_value_eur=1500.0,
        cash_value_eur=0.0,
        cash_detail_df=pd.DataFrame(),
        strategy={
            "target_etf_fraction": 0.5,
            "desired_etf_holdings": 2,
            "desired_non_etf_holdings": 3,
            "target_cash_pct": 5.0,
            "max_single_holding_pct": 80.0,
            "max_top5_holdings_pct": 95.0,
            "max_single_currency_pct": 90.0,
            "max_single_industry_pct": 90.0,
            "max_pair_correlation": 0.90,
            "min_total_holdings": 2,
            "min_over_value_eur": 10.0,
            "target_currency_pct": {},
            "target_industry_pct": {},
            "target_style_pct": {},
        },
        prices_eur=pd.DataFrame(),
        ticker_classifications_path=csv_path,
        auto_append_ticker_characteristics=False,
    )

    concentration = out["concentration_df"]
    row_id = concentration.loc[concentration["product"] == "First product"].iloc[0]
    assert str(row_id["style"]) == "Growth"
    assert str(row_id["industry"]) == "CSV Industry ID"

    row_ticker = concentration.loc[concentration["product"] == "Second product"].iloc[0]
    assert str(row_ticker["style"]) == "Value"
    assert str(row_ticker["industry"]) == "CSV Industry Ticker"


def test_spread_analysis_handles_duplicate_instrument_ids_for_correlation() -> None:
    holdings = pd.DataFrame(
        {
            "instrument_id": ["DUP1", "DUP1", "OTHER1"],
            "product": ["Duplicate Name A", "Duplicate Name B", "Other Holding"],
            "ticker": ["DUPA", "DUPB", "OTHR"],
            "currency": ["EUR", "EUR", "USD"],
            "is_etf": [False, False, True],
            "quantity": [5.0, 3.0, 4.0],
            "value_eur": [600.0, 400.0, 500.0],
        }
    )
    idx = pd.date_range("2025-01-01", periods=220, freq="D")
    dup_px = pd.Series(range(220), index=idx, dtype="float64") + 100.0
    prices = pd.DataFrame(
        {
            "DUP1": dup_px,
            "OTHER1": dup_px * 1.01 + 2.0,
        }
    )

    out = build_ai_spread_analysis(
        holdings_df=holdings,
        total_value_eur=1700.0,
        cash_value_eur=200.0,
        cash_detail_df=pd.DataFrame({"currency": ["EUR"], "balance_eur": [200.0]}),
        strategy={
            "target_etf_fraction": 0.5,
            "desired_etf_holdings": 2,
            "desired_non_etf_holdings": 3,
            "target_cash_pct": 5.0,
            "max_single_holding_pct": 70.0,
            "max_top5_holdings_pct": 95.0,
            "max_single_currency_pct": 90.0,
            "max_single_industry_pct": 90.0,
            "max_pair_correlation": 0.85,
            "min_total_holdings": 2,
            "min_over_value_eur": 10.0,
            "target_currency_pct": {},
            "target_industry_pct": {},
            "target_style_pct": {},
        },
        prices_eur=prices,
        auto_append_ticker_characteristics=False,
    )

    assert isinstance(out.get("correlation_warnings_df"), pd.DataFrame)


def test_spread_analysis_reports_instrument_metadata_conflicts() -> None:
    holdings = pd.DataFrame(
        {
            "instrument_id": ["DUP1", "DUP1", "OTHER1"],
            "product": ["Duplicate Name A", "Duplicate Name B", "Other Holding"],
            "ticker": ["DUPA", "DUPB", "OTHR"],
            "currency": ["EUR", "USD", "USD"],
            "is_etf": [False, False, True],
            "quantity": [5.0, 3.0, 4.0],
            "value_eur": [600.0, 400.0, 500.0],
        }
    )
    idx = pd.date_range("2025-01-01", periods=220, freq="D")
    prices = pd.DataFrame(
        {
            "DUP1": pd.Series(range(220), index=idx, dtype="float64") + 100.0,
            "OTHER1": pd.Series(range(220), index=idx, dtype="float64") + 150.0,
        }
    )

    out = build_ai_spread_analysis(
        holdings_df=holdings,
        total_value_eur=1700.0,
        cash_value_eur=200.0,
        cash_detail_df=pd.DataFrame({"currency": ["EUR"], "balance_eur": [200.0]}),
        strategy={
            "target_etf_fraction": 0.5,
            "desired_etf_holdings": 2,
            "desired_non_etf_holdings": 3,
            "target_cash_pct": 5.0,
            "max_single_holding_pct": 70.0,
            "max_top5_holdings_pct": 95.0,
            "max_single_currency_pct": 90.0,
            "max_single_industry_pct": 90.0,
            "max_pair_correlation": 0.85,
            "min_total_holdings": 2,
            "min_over_value_eur": 10.0,
            "target_currency_pct": {},
            "target_industry_pct": {},
            "target_style_pct": {},
        },
        prices_eur=prices,
        auto_append_ticker_characteristics=False,
    )

    conflicts = out.get("instrument_metadata_conflicts_df", pd.DataFrame())
    assert not conflicts.empty
    assert "DUP1" in set(conflicts["instrument_id"].astype(str))
    assert any("Data quality warning:" in str(line) for line in out.get("summary_lines", []))


def test_irr_helpers_do_not_emit_runtime_warnings_for_extreme_brackets() -> None:
    import warnings

    regular_flows = pd.Series([-1000.0] + [0.0] * 800 + [100.0], dtype="float64").to_numpy()
    dated_flows = [
        (pd.Timestamp("2000-01-01"), -1000.0),
        (pd.Timestamp("2026-01-01"), 100.0),
    ]

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", RuntimeWarning)
        _ = _compute_irr_regular(regular_flows)
        _ = _compute_xirr(dated_flows)

    runtime_warnings = [w for w in captured if issubclass(w.category, RuntimeWarning)]
    assert runtime_warnings == []
