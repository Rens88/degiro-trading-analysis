"""
AGENT_NOTE: Domain analytics for performance and spread diagnostics.

Interdependencies:
- Consumes normalized and merged frames from `src/app.py`.
- Outputs dictionaries with stable keys consumed by:
  - Streamlit rendering in `src/app.py`
  - CLI action detection in `src/strategy_check.py`
- Uses optional `prices_eur` from `src/portfolio_timeseries.py` for
  high-correlation warning pairs.

When editing:
- Treat return dict keys from `build_performance_dashboard()` and
  `build_ai_spread_analysis()` as shared interfaces.
- Keep strategy field names aligned with `src/config.py`, `src/app.py`,
  and `src/strategy_check.py`.
- See `src/INTERDEPENDENCIES.md` for the cross-module contract map.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ticker_characteristics import resolve_ticker_characteristics


def build_ai_generated_insights(
    *,
    metrics_df: pd.DataFrame,
    over_target_df: pd.DataFrame,
) -> dict[str, Any]:
    period_performance_df = build_period_performance_table(metrics_df)
    monthly_performance_df = build_monthly_performance_table(metrics_df)
    drawdown_summary_df = build_drawdown_summary_table(metrics_df)
    action_plan_df = build_action_plan_table(
        metrics_df=metrics_df,
        period_performance_df=period_performance_df,
        over_target_df=over_target_df,
    )
    summary_lines = build_summary_lines(
        metrics_df=metrics_df,
        period_performance_df=period_performance_df,
        drawdown_summary_df=drawdown_summary_df,
    )
    return {
        "period_performance_df": period_performance_df,
        "monthly_performance_df": monthly_performance_df,
        "drawdown_summary_df": drawdown_summary_df,
        "action_plan_df": action_plan_df,
        "summary_lines": summary_lines,
    }


def build_performance_dashboard(
    *,
    metrics_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    account_df: pd.DataFrame,
    benchmark_returns_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build period performance tables and summary KPIs used in Section 4."""
    metrics = _normalized_metrics_df(metrics_df)
    if metrics.empty:
        return {
            "all_time_df": pd.DataFrame(),
            "yearly_df": pd.DataFrame(),
            "quarterly_df": pd.DataFrame(),
            "benchmark_stats_df": pd.DataFrame(),
            "summary": {},
        }

    all_time_df = _build_period_table(
        metrics=metrics,
        transactions_df=transactions_df,
        account_df=account_df,
        period_mode="all_time",
    )
    yearly_df = _build_period_table(
        metrics=metrics,
        transactions_df=transactions_df,
        account_df=account_df,
        period_mode="yearly",
    )
    quarterly_df = _build_period_table(
        metrics=metrics,
        transactions_df=transactions_df,
        account_df=account_df,
        period_mode="quarterly",
    )
    benchmark_stats_df = _build_benchmark_stats_table(
        metrics=metrics,
        benchmark_returns_df=benchmark_returns_df,
    )

    summary_row = all_time_df.iloc[0] if not all_time_df.empty else pd.Series(dtype="float64")
    summary = {
        "all_time_twr_pct": _to_float(summary_row.get("twr_ex_flows_pct", np.nan), np.nan),
        "all_time_irr_pct": _to_float(summary_row.get("irr_annualized_pct", np.nan), np.nan),
        "all_time_xirr_pct": _to_float(summary_row.get("xirr_pct", np.nan), np.nan),
        "all_time_start_value_eur": _to_float(summary_row.get("start_value_eur", np.nan), np.nan),
        "all_time_end_value_eur": _to_float(summary_row.get("end_value_eur", np.nan), np.nan),
        "all_time_net_deposit_eur": _to_float(summary_row.get("net_deposited_eur", np.nan), np.nan),
        "all_time_investment_pnl_eur": _to_float(summary_row.get("investment_pnl_eur", np.nan), np.nan),
    }
    return {
        "all_time_df": all_time_df,
        "yearly_df": yearly_df,
        "quarterly_df": quarterly_df,
        "benchmark_stats_df": benchmark_stats_df,
        "summary": summary,
    }


def build_ai_spread_analysis(
    *,
    holdings_df: pd.DataFrame,
    total_value_eur: float,
    cash_value_eur: float,
    cash_detail_df: pd.DataFrame,
    strategy: dict[str, Any],
    prices_eur: pd.DataFrame | None = None,
    ticker_classifications_path: Path | str | None = None,
    auto_append_ticker_characteristics: bool = False,
) -> dict[str, Any]:
    """
    Build spread/concentration diagnostics and action plans.

    Shared outputs are consumed by Streamlit Section 5 and by startup CLI
    gating in `src/strategy_check.py`.
    """
    holdings = _normalized_holdings_for_spread(holdings_df)
    if holdings.empty and (not np.isfinite(total_value_eur) or total_value_eur <= 0.0):
        return {}

    total_value = float(total_value_eur) if np.isfinite(total_value_eur) else np.nan
    invested_value = float(pd.to_numeric(holdings["value_eur"], errors="coerce").sum()) if not holdings.empty else 0.0
    cash_value = float(cash_value_eur) if np.isfinite(cash_value_eur) else np.nan
    if not np.isfinite(total_value) or total_value <= 0.0:
        total_value = invested_value + (cash_value if np.isfinite(cash_value) else 0.0)
    if not np.isfinite(cash_value):
        cash_value = max(total_value - invested_value, 0.0)
    if total_value <= 0.0:
        return {}

    target_etf_fraction = _to_float(strategy.get("target_etf_fraction"), 0.5)
    desired_etf_holdings = max(_to_int(strategy.get("desired_etf_holdings"), 4), 1)
    desired_non_etf_holdings = max(_to_int(strategy.get("desired_non_etf_holdings"), 12), 1)
    target_cash_pct = _to_float(strategy.get("target_cash_pct"), 10.0)
    max_single_holding_pct = _to_float(strategy.get("max_single_holding_pct"), 12.0)
    max_top5_holdings_pct = _to_float(strategy.get("max_top5_holdings_pct"), 55.0)
    max_single_currency_pct = _to_float(strategy.get("max_single_currency_pct"), 65.0)
    max_single_industry_pct = _to_float(strategy.get("max_single_industry_pct"), 35.0)
    max_pair_correlation = _to_float(strategy.get("max_pair_correlation"), 0.90)
    min_total_holdings = max(_to_int(strategy.get("min_total_holdings"), 12), 1)
    min_over_value_eur = max(_to_float(strategy.get("min_over_value_eur"), 0.0), 0.0)

    target_currency_pct = {
        str(k).upper(): float(v)
        for k, v in _to_target_pct_map(strategy.get("target_currency_pct")).items()
    }
    target_industry_pct = _to_target_pct_map(strategy.get("target_industry_pct"))
    target_style_pct = _to_target_pct_map(strategy.get("target_style_pct"))
    holding_category_overrides = _normalize_holding_category_overrides(
        strategy.get("holding_category_overrides")
    )

    resolved_characteristics, _ = resolve_ticker_characteristics(
        instruments_df=holdings,
        ticker_classifications_path=ticker_classifications_path,
        auto_append_missing=bool(auto_append_ticker_characteristics),
    )
    style_series = pd.Series("", index=holdings.index, dtype="object")
    industry_series = pd.Series("", index=holdings.index, dtype="object")
    if isinstance(resolved_characteristics, pd.DataFrame) and not resolved_characteristics.empty:
        if "style" in resolved_characteristics.columns:
            style_series = resolved_characteristics["style"]
        if "industry" in resolved_characteristics.columns:
            industry_series = resolved_characteristics["industry"]
    holdings["style"] = style_series.fillna("").astype(str).str.strip()
    holdings["industry"] = industry_series.fillna("").astype(str).str.strip()
    holdings["style"] = holdings["style"].replace("", "Unclassified")
    holdings["industry"] = holdings["industry"].replace("", "Unclassified")
    holdings = _apply_holding_category_overrides(
        holdings=holdings,
        overrides=holding_category_overrides,
    )

    etf_mask = holdings["is_etf"].fillna(False).astype(bool)
    etf_value = float(holdings.loc[etf_mask, "value_eur"].sum())
    non_etf_value = float(holdings.loc[~etf_mask, "value_eur"].sum())
    etf_count = int(holdings.loc[etf_mask, "instrument_id"].nunique())
    non_etf_count = int(holdings.loc[~etf_mask, "instrument_id"].nunique())
    total_holdings = int(holdings["instrument_id"].nunique())
    holdings["value_pct_total"] = np.where(total_value > 0.0, holdings["value_eur"] / total_value * 100.0, np.nan)

    etf_pct = _pct(etf_value, total_value)
    non_etf_pct = _pct(non_etf_value, total_value)
    cash_pct = _pct(cash_value, total_value)
    etf_non_etf_ratio = np.nan
    if non_etf_value > 0.0:
        etf_non_etf_ratio = etf_value / non_etf_value

    concentration_df = holdings.sort_values("value_eur", ascending=False).copy()
    concentration_df = concentration_df[
        [
            "product",
            "ticker",
            "currency",
            "industry",
            "style",
            "is_etf",
            "value_eur",
            "value_pct_total",
        ]
    ].head(10)
    concentration_df.insert(0, "rank", range(1, len(concentration_df) + 1))
    largest_holding_pct = float(concentration_df["value_pct_total"].iloc[0]) if not concentration_df.empty else 0.0
    top5_pct = float(concentration_df["value_pct_total"].head(5).sum()) if not concentration_df.empty else 0.0

    currency_allocation_df = _build_currency_allocation_df(
        holdings=holdings,
        cash_detail_df=cash_detail_df,
        fallback_cash_eur=cash_value,
        total_value=total_value,
    )
    largest_currency = currency_allocation_df.head(1)
    largest_currency_name = str(largest_currency.iloc[0]["currency"]) if not largest_currency.empty else "N/A"
    largest_currency_pct = float(largest_currency.iloc[0]["pct_total"]) if not largest_currency.empty else 0.0

    industry_allocation_df = _build_industry_allocation_df(holdings=holdings, total_value=total_value)
    largest_industry = industry_allocation_df.head(1)
    largest_industry_name = str(largest_industry.iloc[0]["industry"]) if not largest_industry.empty else "N/A"
    largest_industry_pct = float(largest_industry.iloc[0]["pct_total"]) if not largest_industry.empty else 0.0
    style_allocation_df = _build_style_allocation_df(holdings=holdings, total_value=total_value)

    currency_allocation_df = _attach_target_pct_columns(
        allocation_df=currency_allocation_df,
        key_col="currency",
        value_col="total_eur",
        total_col="pct_total",
        target_pct_map=target_currency_pct,
        total_value=total_value,
    )
    industry_allocation_df = _attach_target_pct_columns(
        allocation_df=industry_allocation_df,
        key_col="industry",
        value_col="value_eur",
        total_col="pct_total",
        target_pct_map=target_industry_pct,
        total_value=total_value,
    )
    style_allocation_df = _attach_target_pct_columns(
        allocation_df=style_allocation_df,
        key_col="style",
        value_col="value_eur",
        total_col="pct_total",
        target_pct_map=target_style_pct,
        total_value=total_value,
    )

    correlation_warnings_df = _compute_high_correlation_pairs(
        holdings=holdings,
        prices_eur=prices_eur,
        min_correlation=max_pair_correlation,
    )
    highest_pair_correlation_pct = (
        float(correlation_warnings_df["correlation"].max()) * 100.0
        if not correlation_warnings_df.empty
        else np.nan
    )

    etf_non_etf_df = pd.DataFrame(
        [
            {
                "segment": "ETF",
                "value_eur": etf_value,
                "share_pct_total": etf_pct,
                "target_share_pct_total": target_etf_fraction * 100.0,
                "delta_vs_target_pct": etf_pct - target_etf_fraction * 100.0,
                "holdings_count": etf_count,
                "target_holdings_count": desired_etf_holdings,
            },
            {
                "segment": "Non-ETF",
                "value_eur": non_etf_value,
                "share_pct_total": non_etf_pct,
                "target_share_pct_total": (1.0 - target_etf_fraction) * 100.0,
                "delta_vs_target_pct": non_etf_pct - (1.0 - target_etf_fraction) * 100.0,
                "holdings_count": non_etf_count,
                "target_holdings_count": desired_non_etf_holdings,
            },
            {
                "segment": "Cash",
                "value_eur": cash_value,
                "share_pct_total": cash_pct,
                "target_share_pct_total": target_cash_pct,
                "delta_vs_target_pct": cash_pct - target_cash_pct,
                "holdings_count": np.nan,
                "target_holdings_count": np.nan,
            },
        ]
    )

    strategy_rows = [
        {
            "metric": "ETF share (%)",
            "actual": etf_pct,
            "target_or_limit": f"target {target_etf_fraction * 100.0:,.1f}%",
            "delta": etf_pct - target_etf_fraction * 100.0,
            "status": _status_target(etf_pct, target_etf_fraction * 100.0, tolerance=3.0),
        },
        {
            "metric": "Cash share (%)",
            "actual": cash_pct,
            "target_or_limit": f"target {target_cash_pct:,.1f}%",
            "delta": cash_pct - target_cash_pct,
            "status": _status_target(cash_pct, target_cash_pct, tolerance=2.0),
        },
        {
            "metric": "Largest holding (%)",
            "actual": largest_holding_pct,
            "target_or_limit": f"<= {max_single_holding_pct:,.1f}%",
            "delta": largest_holding_pct - max_single_holding_pct,
            "status": _status_max(largest_holding_pct, max_single_holding_pct),
        },
        {
            "metric": "Top-5 concentration (%)",
            "actual": top5_pct,
            "target_or_limit": f"<= {max_top5_holdings_pct:,.1f}%",
            "delta": top5_pct - max_top5_holdings_pct,
            "status": _status_max(top5_pct, max_top5_holdings_pct),
        },
        {
            "metric": "Largest currency (%)",
            "actual": largest_currency_pct,
            "target_or_limit": f"<= {max_single_currency_pct:,.1f}%",
            "delta": largest_currency_pct - max_single_currency_pct,
            "status": _status_max(largest_currency_pct, max_single_currency_pct),
        },
        {
            "metric": "Largest industry (%)",
            "actual": largest_industry_pct,
            "target_or_limit": f"<= {max_single_industry_pct:,.1f}%",
            "delta": largest_industry_pct - max_single_industry_pct,
            "status": _status_max(largest_industry_pct, max_single_industry_pct),
        },
        {
            "metric": "Highest pair correlation (%)",
            "actual": highest_pair_correlation_pct,
            "target_or_limit": f"<= {max_pair_correlation * 100.0:,.1f}%",
            "delta": (
                highest_pair_correlation_pct - max_pair_correlation * 100.0
                if np.isfinite(highest_pair_correlation_pct)
                else np.nan
            ),
            "status": (
                _status_max(highest_pair_correlation_pct, max_pair_correlation * 100.0)
                if np.isfinite(highest_pair_correlation_pct)
                else "N/A"
            ),
        },
        {
            "metric": "Total non-cash holdings (count)",
            "actual": float(total_holdings),
            "target_or_limit": f">= {min_total_holdings:d}",
            "delta": float(total_holdings - min_total_holdings),
            "status": _status_min(float(total_holdings), float(min_total_holdings)),
        },
    ]
    strategy_checks_df = pd.DataFrame(strategy_rows)

    action_rows: list[dict[str, Any]] = []
    what_to_do_rows: list[dict[str, Any]] = []

    n_etf = max(int(desired_etf_holdings), 1)
    n_non_etf = max(int(desired_non_etf_holdings), 1)
    holdings["target_per_holding_pct"] = np.where(
        holdings["is_etf"],
        target_etf_fraction / n_etf * 100.0,
        (1.0 - target_etf_fraction) / n_non_etf * 100.0,
    )
    holdings["target_value_eur"] = holdings["target_per_holding_pct"] / 100.0 * total_value
    holdings["over_target_eur"] = holdings["value_eur"] - holdings["target_value_eur"]
    over_target_holdings = holdings.loc[
        pd.to_numeric(holdings["over_target_eur"], errors="coerce") > float(min_over_value_eur)
    ].copy()
    over_target_holdings = over_target_holdings.sort_values("over_target_eur", ascending=False)
    for row in over_target_holdings.itertuples(index=False):
        what_to_do_rows.append(
            {
                "priority": "High",
                "requires_action": True,
                "action_type": "Trim over-target holding",
                "instrument_id": getattr(row, "instrument_id", ""),
                "product": getattr(row, "product", ""),
                "ticker": getattr(row, "ticker", ""),
                "segment": "ETF" if bool(getattr(row, "is_etf", False)) else "Non-ETF",
                "current_pct": float(getattr(row, "value_pct_total", np.nan)),
                "target_pct": float(getattr(row, "target_per_holding_pct", np.nan)),
                "suggested_amount_eur": float(getattr(row, "over_target_eur", np.nan)),
                "note": "Sell this amount to move back to target-per-holding.",
            }
        )
    if not over_target_holdings.empty:
        trim_total = float(pd.to_numeric(over_target_holdings["over_target_eur"], errors="coerce").sum())
        action_rows.append(
            {
                "priority": "High",
                "requires_action": True,
                "theme": "Rebalancing",
                "what_we_see": (
                    f"{len(over_target_holdings)} holding(s) are above threshold by more than "
                    f"EUR {min_over_value_eur:,.2f}."
                ),
                "future_action": (
                    f"Trim approximately EUR {trim_total:,.2f} in total to return near per-holding targets."
                ),
            }
        )

    etf_low = holdings.loc[holdings["is_etf"]].sort_values("value_pct_total", ascending=True).head(2)
    non_etf_low = holdings.loc[~holdings["is_etf"]].sort_values("value_pct_total", ascending=True).head(5)
    for row in etf_low.itertuples(index=False):
        what_to_do_rows.append(
            {
                "priority": "Low",
                "requires_action": False,
                "action_type": "Buy focus (ETF)",
                "instrument_id": getattr(row, "instrument_id", ""),
                "product": getattr(row, "product", ""),
                "ticker": getattr(row, "ticker", ""),
                "segment": "ETF",
                "current_pct": float(getattr(row, "value_pct_total", np.nan)),
                "target_pct": float(getattr(row, "target_per_holding_pct", np.nan)),
                "suggested_amount_eur": np.nan,
                "note": "One of the two lowest ETF weights; prioritize on next buys.",
            }
        )
    for row in non_etf_low.itertuples(index=False):
        what_to_do_rows.append(
            {
                "priority": "Low",
                "requires_action": False,
                "action_type": "Buy focus (Non-ETF)",
                "instrument_id": getattr(row, "instrument_id", ""),
                "product": getattr(row, "product", ""),
                "ticker": getattr(row, "ticker", ""),
                "segment": "Non-ETF",
                "current_pct": float(getattr(row, "value_pct_total", np.nan)),
                "target_pct": float(getattr(row, "target_per_holding_pct", np.nan)),
                "suggested_amount_eur": np.nan,
                "note": "One of the five lowest non-ETF weights; prioritize on next buys.",
            }
        )

    target_cash_value = target_cash_pct / 100.0 * total_value
    deployable_cash = max(cash_value - target_cash_value, 0.0)
    etf_gap_eur = max(target_etf_fraction * total_value - etf_value, 0.0)
    non_etf_gap_eur = max((1.0 - target_etf_fraction) * total_value - non_etf_value, 0.0)
    gap_total = etf_gap_eur + non_etf_gap_eur
    if deployable_cash > 0.0:
        if gap_total > 0.0:
            cash_to_etf = deployable_cash * etf_gap_eur / gap_total
            cash_to_non_etf = deployable_cash * non_etf_gap_eur / gap_total
        else:
            cash_to_etf = deployable_cash * target_etf_fraction
            cash_to_non_etf = deployable_cash * (1.0 - target_etf_fraction)
        what_to_do_rows.append(
            {
                "priority": "Medium",
                "requires_action": True,
                "action_type": "Deploy available cash",
                "instrument_id": "",
                "product": "",
                "ticker": "",
                "segment": "Portfolio",
                "current_pct": cash_pct,
                "target_pct": target_cash_pct,
                "suggested_amount_eur": deployable_cash,
                "note": (
                    f"Allocate about EUR {cash_to_etf:,.2f} to ETF and EUR {cash_to_non_etf:,.2f} to non-ETF."
                ),
            }
        )
        action_rows.append(
            {
                "priority": "Medium",
                "requires_action": True,
                "theme": "Cash deployment",
                "what_we_see": f"Cash above target by approximately EUR {deployable_cash:,.2f}.",
                "future_action": (
                    f"Deploy roughly EUR {cash_to_etf:,.2f} to ETF and EUR {cash_to_non_etf:,.2f} to non-ETF."
                ),
            }
        )

    if deployable_cash > 0.0 and target_currency_pct:
        currency_plan_df = _build_allocation_plan_from_targets(
            allocation_df=currency_allocation_df,
            key_col="currency",
            value_col="total_eur",
            target_pct_map=target_currency_pct,
            deployable_cash=deployable_cash,
            total_value=total_value,
        )
        for row in currency_plan_df.itertuples(index=False):
            what_to_do_rows.append(
                {
                    "priority": "Low",
                    "requires_action": False,
                    "action_type": "Currency buy allocation",
                    "instrument_id": "",
                    "product": "",
                    "ticker": "",
                    "segment": str(getattr(row, "bucket", "")),
                    "current_pct": float(getattr(row, "current_pct", np.nan)),
                    "target_pct": float(getattr(row, "target_pct", np.nan)),
                    "suggested_amount_eur": float(getattr(row, "allocation_eur", np.nan)),
                    "note": "Suggested split of deployable cash by currency target.",
                }
            )

    if deployable_cash > 0.0 and target_industry_pct:
        industry_plan_df = _build_allocation_plan_from_targets(
            allocation_df=industry_allocation_df,
            key_col="industry",
            value_col="value_eur",
            target_pct_map=target_industry_pct,
            deployable_cash=deployable_cash,
            total_value=total_value,
        )
        for row in industry_plan_df.itertuples(index=False):
            what_to_do_rows.append(
                {
                    "priority": "Low",
                    "requires_action": False,
                    "action_type": "Industry buy allocation",
                    "instrument_id": "",
                    "product": "",
                    "ticker": "",
                    "segment": str(getattr(row, "bucket", "")),
                    "current_pct": float(getattr(row, "current_pct", np.nan)),
                    "target_pct": float(getattr(row, "target_pct", np.nan)),
                    "suggested_amount_eur": float(getattr(row, "allocation_eur", np.nan)),
                    "note": "Suggested split of deployable cash by industry target.",
                }
            )

    if deployable_cash > 0.0 and target_style_pct:
        style_plan_df = _build_allocation_plan_from_targets(
            allocation_df=style_allocation_df,
            key_col="style",
            value_col="value_eur",
            target_pct_map=target_style_pct,
            deployable_cash=deployable_cash,
            total_value=total_value,
        )
        for row in style_plan_df.itertuples(index=False):
            what_to_do_rows.append(
                {
                    "priority": "Low",
                    "requires_action": False,
                    "action_type": "Style buy allocation",
                    "instrument_id": "",
                    "product": "",
                    "ticker": "",
                    "segment": str(getattr(row, "bucket", "")),
                    "current_pct": float(getattr(row, "current_pct", np.nan)),
                    "target_pct": float(getattr(row, "target_pct", np.nan)),
                    "suggested_amount_eur": float(getattr(row, "allocation_eur", np.nan)),
                    "note": "Suggested split of deployable cash by style target.",
                }
            )

    if cash_pct > target_cash_pct + 2.0:
        deploy_eur = (cash_pct - target_cash_pct) / 100.0 * total_value
        action_rows.append(
            {
                "priority": "High",
                "requires_action": True,
                "theme": "Cash allocation",
                "what_we_see": f"Cash is {cash_pct:,.1f}% vs target {target_cash_pct:,.1f}%.",
                "future_action": f"Deploy about EUR {deploy_eur:,.2f} into target allocations over time.",
            }
        )
    elif cash_pct < max(target_cash_pct - 2.0, 0.0):
        replenish_eur = (target_cash_pct - cash_pct) / 100.0 * total_value
        action_rows.append(
            {
                "priority": "Medium",
                "requires_action": True,
                "theme": "Liquidity buffer",
                "what_we_see": f"Cash is {cash_pct:,.1f}% vs target {target_cash_pct:,.1f}%.",
                "future_action": f"Build about EUR {replenish_eur:,.2f} cash buffer from new flows.",
            }
        )

    if etf_pct > target_etf_fraction * 100.0 + 3.0:
        shift = (etf_pct - target_etf_fraction * 100.0) / 100.0 * total_value
        action_rows.append(
            {
                "priority": "Medium",
                "requires_action": True,
                "theme": "ETF / non-ETF spread",
                "what_we_see": f"ETF share is {etf_pct:,.1f}% vs target {target_etf_fraction * 100.0:,.1f}%.",
                "future_action": f"Favor non-ETF adds by about EUR {shift:,.2f} over next contributions.",
            }
        )
    elif etf_pct < target_etf_fraction * 100.0 - 3.0:
        shift = (target_etf_fraction * 100.0 - etf_pct) / 100.0 * total_value
        action_rows.append(
            {
                "priority": "Medium",
                "requires_action": True,
                "theme": "ETF / non-ETF spread",
                "what_we_see": f"ETF share is {etf_pct:,.1f}% vs target {target_etf_fraction * 100.0:,.1f}%.",
                "future_action": f"Favor ETF adds by about EUR {shift:,.2f} over next contributions.",
            }
        )

    if largest_holding_pct > max_single_holding_pct:
        action_rows.append(
            {
                "priority": "High",
                "requires_action": True,
                "theme": "Single-name concentration",
                "what_we_see": f"Largest position is {largest_holding_pct:,.1f}% of portfolio.",
                "future_action": "Pause adds to the largest holding and diversify incremental capital.",
            }
        )
    if top5_pct > max_top5_holdings_pct:
        action_rows.append(
            {
                "priority": "Medium",
                "requires_action": True,
                "theme": "Top-5 concentration",
                "what_we_see": f"Top 5 holdings represent {top5_pct:,.1f}% of portfolio.",
                "future_action": "Direct new buys toward smaller positions until concentration declines.",
            }
        )
    if largest_currency_pct > max_single_currency_pct:
        action_rows.append(
            {
                "priority": "Medium",
                "requires_action": True,
                "theme": "Currency concentration",
                "what_we_see": f"Largest currency exposure is {largest_currency_name} at {largest_currency_pct:,.1f}%.",
                "future_action": "Add positions/cash in other currencies to lower single-currency dependency.",
            }
        )
    if largest_industry_pct > max_single_industry_pct:
        action_rows.append(
            {
                "priority": "Medium",
                "requires_action": True,
                "theme": "Industry concentration",
                "what_we_see": f"Largest industry bucket is {largest_industry_name} at {largest_industry_pct:,.1f}%.",
                "future_action": "Rebalance future additions toward underrepresented industries.",
            }
        )
    if total_holdings < min_total_holdings:
        action_rows.append(
            {
                "priority": "Low",
                "requires_action": False,
                "theme": "Breadth",
                "what_we_see": f"Portfolio has {total_holdings} non-cash holdings vs target minimum {min_total_holdings}.",
                "future_action": "Increase breadth gradually while preserving ETF/non-ETF strategy.",
            }
        )
    if not correlation_warnings_df.empty:
        top_pair = correlation_warnings_df.iloc[0]
        action_rows.append(
            {
                "priority": "High",
                "requires_action": True,
                "theme": "Correlation concentration",
                "what_we_see": (
                    f"Highest pair correlation is {float(top_pair['correlation']) * 100.0:,.1f}% between "
                    f"{top_pair['ticker_a']} and {top_pair['ticker_b']}."
                ),
                "future_action": (
                    "Treat highly correlated pairs as one risk bucket and reduce overlap in future buys."
                ),
            }
        )

    summary_lines = [
        (
            f"ETF/non-ETF split: {etf_pct:,.1f}% / {non_etf_pct:,.1f}% "
            f"(target {target_etf_fraction * 100.0:,.1f}% / {(1.0 - target_etf_fraction) * 100.0:,.1f}%)."
        ),
        f"Cash allocation: {cash_pct:,.1f}% (target {target_cash_pct:,.1f}%).",
        f"Concentration: largest holding {largest_holding_pct:,.1f}%, top-5 {top5_pct:,.1f}%.",
        (
            f"Broadest exposures: currency {largest_currency_name} {largest_currency_pct:,.1f}%, "
            f"industry {largest_industry_name} {largest_industry_pct:,.1f}%."
        ),
    ]
    if np.isfinite(etf_non_etf_ratio):
        summary_lines.append(f"Current ETF:non-ETF value ratio = {etf_non_etf_ratio:,.2f}x.")
    if not correlation_warnings_df.empty:
        summary_lines.append(
            f"Correlation risk: {len(correlation_warnings_df)} pair(s) above {max_pair_correlation * 100.0:,.1f}%."
        )
    if target_currency_pct:
        summary_lines.append("Currency target map is configured and used in buy-allocation suggestions.")
    if target_industry_pct:
        summary_lines.append("Industry target map is configured and used in buy-allocation suggestions.")
    if target_style_pct:
        summary_lines.append("Style target map is configured and used in buy-allocation suggestions.")

    action_plan_df = pd.DataFrame(action_rows).reset_index(drop=True)
    what_to_do_next_df = pd.DataFrame(what_to_do_rows).reset_index(drop=True)
    return {
        "summary_lines": summary_lines,
        "strategy_checks_df": strategy_checks_df,
        "etf_non_etf_df": etf_non_etf_df,
        "currency_allocation_df": currency_allocation_df,
        "industry_allocation_df": industry_allocation_df,
        "style_allocation_df": style_allocation_df,
        "concentration_df": concentration_df.reset_index(drop=True),
        "correlation_warnings_df": correlation_warnings_df.reset_index(drop=True),
        "action_plan_df": action_plan_df,
        "what_to_do_next_df": what_to_do_next_df,
        "target_currency_pct": target_currency_pct,
        "target_industry_pct": target_industry_pct,
        "target_style_pct": target_style_pct,
    }


def build_period_performance_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = _normalized_metrics_df(metrics_df)
    required = {"portfolio_value", "total_deposits"}
    if df.empty or not required.issubset(set(df.columns)):
        return pd.DataFrame()

    periods: list[tuple[str, int | None]] = [
        ("30D", 30),
        ("90D", 90),
        ("180D", 180),
        ("365D", 365),
        ("Since start", None),
    ]
    end_date = df.index.max()
    rows: list[dict[str, Any]] = []
    for label, days in periods:
        if days is None:
            sub = df.copy()
        else:
            cutoff = end_date - pd.Timedelta(days=int(days))
            sub = df.loc[df.index >= cutoff].copy()
        if sub.shape[0] < 2:
            continue

        start_date = sub.index.min()
        start_val = float(sub.iloc[0]["portfolio_value"])
        end_val = float(sub.iloc[-1]["portfolio_value"])
        start_dep = float(sub.iloc[0]["total_deposits"])
        end_dep = float(sub.iloc[-1]["total_deposits"])

        total_change = end_val - start_val
        net_flow = end_dep - start_dep
        investment_pnl = total_change - net_flow
        twr = _compute_twr(sub)

        day_count = max(int((sub.index.max() - sub.index.min()).days), 1)
        annualized_twr = np.nan
        if np.isfinite(twr) and twr > -0.999 and day_count >= 30:
            annualized_twr = (1.0 + twr) ** (365.0 / day_count) - 1.0

        rows.append(
            {
                "period": label,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": sub.index.max().strftime("%Y-%m-%d"),
                "start_value_eur": start_val,
                "end_value_eur": end_val,
                "total_change_eur": total_change,
                "net_flow_eur": net_flow,
                "investment_pnl_eur": investment_pnl,
                "twr_ex_flows_pct": twr * 100.0 if np.isfinite(twr) else np.nan,
                "annualized_twr_pct": annualized_twr * 100.0 if np.isfinite(annualized_twr) else np.nan,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_monthly_performance_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = _normalized_metrics_df(metrics_df)
    required = {"portfolio_value", "total_deposits"}
    if df.empty or not required.issubset(set(df.columns)):
        return pd.DataFrame()

    month_end = df[["portfolio_value", "total_deposits"]].resample("ME").last().dropna(how="all")
    if month_end.shape[0] < 2:
        return pd.DataFrame()

    out = month_end.copy()
    out["start_value_eur"] = out["portfolio_value"].shift(1)
    out["end_value_eur"] = out["portfolio_value"]
    out["net_flow_eur"] = out["total_deposits"].diff()
    out["total_change_eur"] = out["end_value_eur"] - out["start_value_eur"]
    out["investment_pnl_eur"] = out["total_change_eur"] - out["net_flow_eur"]
    out["return_ex_flows_pct"] = np.where(
        out["start_value_eur"] > 0.0,
        out["investment_pnl_eur"] / out["start_value_eur"] * 100.0,
        np.nan,
    )
    out = out.iloc[1:].copy()
    out.insert(0, "month", out.index.strftime("%Y-%m"))
    keep = [
        "month",
        "start_value_eur",
        "end_value_eur",
        "total_change_eur",
        "net_flow_eur",
        "investment_pnl_eur",
        "return_ex_flows_pct",
    ]
    return out[keep].reset_index(drop=True)


def _build_period_table(
    *,
    metrics: pd.DataFrame,
    transactions_df: pd.DataFrame,
    account_df: pd.DataFrame,
    period_mode: str,
) -> pd.DataFrame:
    # AGENT_NOTE: Column names from this table are rendered directly in
    # `app.py` performance tabs. Keep names stable unless UI/tests are updated.
    required_cols = {"portfolio_value", "total_deposits"}
    if metrics.empty or not required_cols.issubset(set(metrics.columns)):
        return pd.DataFrame()

    tx = transactions_df.copy() if isinstance(transactions_df, pd.DataFrame) else pd.DataFrame()
    if not tx.empty and "datetime" in tx.columns:
        tx["datetime"] = pd.to_datetime(tx["datetime"], errors="coerce")
    acct = account_df.copy() if isinstance(account_df, pd.DataFrame) else pd.DataFrame()
    if not acct.empty and "datetime" in acct.columns:
        acct["datetime"] = pd.to_datetime(acct["datetime"], errors="coerce")
    if not acct.empty:
        acct["change_eur_effective"] = _account_change_eur_effective(acct)

    rows: list[dict[str, Any]] = []
    if period_mode == "all_time":
        row = _build_single_period_row(
            label="All time",
            sub=metrics,
            tx=tx,
            acct=acct,
        )
        if row:
            rows.append(row)
    elif period_mode == "yearly":
        for period, sub in metrics.groupby(metrics.index.to_period("Y")):
            row = _build_single_period_row(
                label=str(period),
                sub=sub,
                tx=tx,
                acct=acct,
            )
            if row:
                rows.append(row)
    elif period_mode == "quarterly":
        for period, sub in metrics.groupby(metrics.index.to_period("Q")):
            row = _build_single_period_row(
                label=str(period),
                sub=sub,
                tx=tx,
                acct=acct,
            )
            if row:
                rows.append(row)
    else:
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)


def _build_single_period_row(
    *,
    label: str,
    sub: pd.DataFrame,
    tx: pd.DataFrame,
    acct: pd.DataFrame,
) -> dict[str, Any] | None:
    if sub is None or sub.empty or sub.shape[0] < 2:
        return None
    sub = sub.sort_index()
    start = pd.to_datetime(sub.index.min(), errors="coerce")
    end = pd.to_datetime(sub.index.max(), errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return None

    start_val = float(pd.to_numeric(sub.iloc[0].get("portfolio_value", np.nan), errors="coerce"))
    end_val = float(pd.to_numeric(sub.iloc[-1].get("portfolio_value", np.nan), errors="coerce"))
    start_dep = float(pd.to_numeric(sub.iloc[0].get("total_deposits", np.nan), errors="coerce"))
    end_dep = float(pd.to_numeric(sub.iloc[-1].get("total_deposits", np.nan), errors="coerce"))
    if not np.isfinite(start_val) or not np.isfinite(end_val):
        return None

    net_deposited = end_dep - start_dep if np.isfinite(start_dep) and np.isfinite(end_dep) else np.nan
    total_change = end_val - start_val
    investment_pnl = total_change - net_deposited if np.isfinite(net_deposited) else np.nan
    twr = _compute_twr(sub)

    tx_sub = pd.DataFrame()
    if isinstance(tx, pd.DataFrame) and not tx.empty and "datetime" in tx.columns:
        tx_sub = tx.loc[(tx["datetime"] >= start) & (tx["datetime"] <= end)].copy()
        if "is_cash_like" in tx_sub.columns:
            tx_sub = tx_sub.loc[~tx_sub["is_cash_like"].fillna(False)].copy()
    acct_sub = pd.DataFrame()
    if isinstance(acct, pd.DataFrame) and not acct.empty and "datetime" in acct.columns:
        acct_sub = acct.loc[(acct["datetime"] >= start) & (acct["datetime"] <= end)].copy()

    flow_series = pd.to_numeric(sub["total_deposits"], errors="coerce").diff().fillna(
        pd.to_numeric(sub["total_deposits"], errors="coerce")
    )
    flow_series = flow_series.astype(float)

    xirr_cashflows: list[tuple[pd.Timestamp, float]] = [(start, -start_val)]
    if not acct_sub.empty:
        if "is_external_flow" in acct_sub.columns:
            external_mask = acct_sub["is_external_flow"].fillna(False).astype(bool)
        elif "type" in acct_sub.columns:
            external_mask = acct_sub["type"].astype(str).isin({"external_deposit", "external_withdrawal"})
        else:
            external_mask = pd.Series(False, index=acct_sub.index, dtype=bool)
        external = acct_sub.loc[external_mask].copy().sort_values("datetime")
        if not external.empty:
            external_flow = pd.to_numeric(external.get("change_eur_effective"), errors="coerce")
            for dt, flow in zip(external["datetime"], external_flow):
                if pd.isna(dt) or not np.isfinite(flow) or abs(flow) <= 1e-9:
                    continue
                # Investor perspective: deposit is outflow, withdrawal is inflow.
                xirr_cashflows.append((pd.Timestamp(dt), -float(flow)))
    if len(xirr_cashflows) == 1:
        for dt, value in flow_series.iloc[1:].items():
            flow = float(value) if np.isfinite(value) else 0.0
            if abs(flow) > 1e-9:
                xirr_cashflows.append((pd.Timestamp(dt), -flow))
    xirr_cashflows.append((end, end_val))
    xirr = _compute_xirr(xirr_cashflows)

    regular_cashflows = np.zeros(len(sub), dtype="float64")
    regular_cashflows[0] = -start_val
    if len(sub) > 1:
        regular_cashflows[1:] = -flow_series.iloc[1:].to_numpy(dtype="float64")
    regular_cashflows[-1] += end_val
    irr_period = _compute_irr_regular(regular_cashflows)
    irr_annualized = (1.0 + irr_period) ** 365.0 - 1.0 if np.isfinite(irr_period) and irr_period > -0.999 else np.nan

    number_of_trades = 0
    buy_trades = 0
    sell_trades = 0
    most_traded_holding = ""
    most_traded_count = 0
    if not tx_sub.empty:
        qty = pd.to_numeric(tx_sub.get("quantity"), errors="coerce")
        trade_mask = qty.notna() & qty.ne(0.0)
        tx_trades = tx_sub.loc[trade_mask].copy()
        number_of_trades = int(len(tx_trades))
        buy_trades = int((pd.to_numeric(tx_trades.get("quantity"), errors="coerce") > 0.0).sum())
        sell_trades = int((pd.to_numeric(tx_trades.get("quantity"), errors="coerce") < 0.0).sum())
        if not tx_trades.empty:
            tx_trades["label"] = tx_trades.apply(
                lambda row: _holding_display_label(
                    product=str(row.get("product", "")),
                    ticker=str(row.get("ticker", "")),
                    instrument_id=str(row.get("instrument_id", "")),
                ),
                axis=1,
            )
            counts = tx_trades["label"].value_counts()
            if not counts.empty:
                most_traded_holding = str(counts.index[0])
                most_traded_count = int(counts.iloc[0])

    transaction_fee_eur = 0.0
    account_fee_eur = 0.0
    dividend_tax_eur = 0.0
    platform_costs_eur = 0.0
    if not acct_sub.empty:
        cost_mask = acct_sub.get("is_cost", pd.Series(False, index=acct_sub.index)).fillna(False).astype(bool)
        if "is_cost" not in acct_sub.columns and "type" in acct_sub.columns:
            cost_mask = acct_sub["type"].astype(str).isin({"transaction_fee", "account_fee", "dividend_tax"})
        costs = acct_sub.loc[cost_mask].copy()
        if not costs.empty:
            cost_val = pd.to_numeric(costs.get("change_eur_effective"), errors="coerce")
            costs["cost_outflow_eur"] = np.where(cost_val < 0.0, -cost_val, 0.0)
            cost_type = costs["type"].astype(str) if "type" in costs.columns else pd.Series("", index=costs.index)
            transaction_fee_eur = float(
                costs.loc[cost_type.eq("transaction_fee"), "cost_outflow_eur"].sum()
            )
            account_fee_eur = float(
                costs.loc[cost_type.eq("account_fee"), "cost_outflow_eur"].sum()
            )
            dividend_tax_eur = float(
                costs.loc[cost_type.eq("dividend_tax"), "cost_outflow_eur"].sum()
            )
            platform_costs_eur = transaction_fee_eur + account_fee_eur

    row = {
        "period": label,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "start_value_eur": start_val,
        "net_deposited_eur": net_deposited,
        "end_value_eur": end_val,
        "total_change_eur": total_change,
        "investment_pnl_eur": investment_pnl,
        "twr_ex_flows_pct": twr * 100.0 if np.isfinite(twr) else np.nan,
        "irr_annualized_pct": irr_annualized * 100.0 if np.isfinite(irr_annualized) else np.nan,
        "xirr_pct": xirr * 100.0 if np.isfinite(xirr) else np.nan,
        "transaction_fee_eur": transaction_fee_eur,
        "account_fee_eur": account_fee_eur,
        "dividend_tax_eur": dividend_tax_eur,
        "platform_costs_eur": platform_costs_eur,
        "number_of_trades": number_of_trades,
        "buy_trades": buy_trades,
        "sell_trades": sell_trades,
        "most_traded_holding": most_traded_holding,
        "most_traded_count": most_traded_count,
    }
    return row


def _account_change_eur_effective(account_df: pd.DataFrame) -> pd.Series:
    if account_df is None or account_df.empty:
        return pd.Series(dtype="float64")
    change = pd.to_numeric(account_df.get("change_eur"), errors="coerce")
    raw_change = pd.to_numeric(account_df.get("raw_change"), errors="coerce")
    currency = account_df.get("currency", pd.Series("", index=account_df.index)).fillna("").astype(str).str.upper()
    return np.where(change.notna(), change, np.where(currency.eq("EUR"), raw_change, np.nan))


def _compute_xirr(cashflows: list[tuple[pd.Timestamp, float]]) -> float:
    if not cashflows:
        return np.nan
    cleaned = [(pd.Timestamp(d), float(v)) for d, v in cashflows if pd.notna(d) and np.isfinite(v)]
    if len(cleaned) < 2:
        return np.nan
    values = np.array([v for _, v in cleaned], dtype="float64")
    if not ((values > 0.0).any() and (values < 0.0).any()):
        return np.nan
    base = cleaned[0][0]
    years = np.array([(d - base).days / 365.0 for d, _ in cleaned], dtype="float64")

    def xnpv(rate: float) -> float:
        if not np.isfinite(rate) or rate <= -0.999999:
            return np.nan
        return _safe_discounted_npv(values=values, exponents=years, rate=rate)

    lo = -0.999
    hi = 1.0
    f_lo = xnpv(lo)
    f_hi = xnpv(hi)
    attempts = 0
    while np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo * f_hi > 0.0 and attempts < 25:
        hi *= 2.0
        f_hi = xnpv(hi)
        attempts += 1
    if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0.0:
        return np.nan
    for _ in range(120):
        mid = (lo + hi) / 2.0
        f_mid = xnpv(mid)
        if not np.isfinite(f_mid):
            return np.nan
        if abs(f_mid) < 1e-9:
            return float(mid)
        if f_lo * f_mid <= 0.0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return float((lo + hi) / 2.0)


def _compute_irr_regular(cashflows: np.ndarray) -> float:
    if cashflows is None or len(cashflows) < 2:
        return np.nan
    arr = np.asarray(cashflows, dtype="float64")
    if not ((arr > 0.0).any() and (arr < 0.0).any()):
        return np.nan
    t = np.arange(len(arr), dtype="float64")

    def npv(rate: float) -> float:
        if not np.isfinite(rate) or rate <= -0.999999:
            return np.nan
        return _safe_discounted_npv(values=arr, exponents=t, rate=rate)

    lo = -0.999
    hi = 1.0
    f_lo = npv(lo)
    f_hi = npv(hi)
    attempts = 0
    while np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo * f_hi > 0.0 and attempts < 25:
        hi *= 2.0
        f_hi = npv(hi)
        attempts += 1
    if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0.0:
        return np.nan
    for _ in range(120):
        mid = (lo + hi) / 2.0
        f_mid = npv(mid)
        if not np.isfinite(f_mid):
            return np.nan
        if abs(f_mid) < 1e-10:
            return float(mid)
        if f_lo * f_mid <= 0.0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return float((lo + hi) / 2.0)


def _safe_discounted_npv(*, values: np.ndarray, exponents: np.ndarray, rate: float) -> float:
    """
    Numerically safer discounted sum used by IRR/XIRR root-finding.

    Avoids noisy RuntimeWarnings from divide-by-zero/overflow in extreme
    bracket attempts while preserving sign behavior for bisection.
    """
    base = 1.0 + float(rate)
    if not np.isfinite(base) or base <= 0.0:
        return np.nan
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        denom = np.power(base, exponents)
        discounted = values / denom

    finite_mask = np.isfinite(discounted)
    has_pos_inf = bool(np.isposinf(discounted).any())
    has_neg_inf = bool(np.isneginf(discounted).any())

    if has_pos_inf and has_neg_inf:
        return np.nan
    if has_pos_inf:
        return float("inf")
    if has_neg_inf:
        return float("-inf")
    if not finite_mask.any():
        return np.nan
    return float(np.sum(discounted[finite_mask]))


def _holding_display_label(*, product: str, ticker: str, instrument_id: str) -> str:
    p = str(product).strip()
    t = str(ticker).strip()
    i = str(instrument_id).strip()
    if p and t:
        return f"{p} ({t})"
    if p:
        return p
    if t:
        return t
    return i


def _portfolio_daily_twr_returns(metrics: pd.DataFrame) -> pd.Series:
    df = _normalized_metrics_df(metrics)
    required = {"portfolio_value", "total_deposits"}
    if df.empty or not required.issubset(set(df.columns)) or len(df) < 2:
        return pd.Series(dtype="float64")
    values = pd.to_numeric(df["portfolio_value"], errors="coerce")
    deposits = pd.to_numeric(df["total_deposits"], errors="coerce")
    ret = pd.Series(np.nan, index=df.index, dtype="float64")
    for i in range(1, len(df)):
        prev = float(values.iloc[i - 1])
        cur = float(values.iloc[i])
        if not np.isfinite(prev) or prev <= 0.0 or not np.isfinite(cur):
            continue
        d0 = float(deposits.iloc[i - 1]) if np.isfinite(deposits.iloc[i - 1]) else np.nan
        d1 = float(deposits.iloc[i]) if np.isfinite(deposits.iloc[i]) else np.nan
        flow = (d1 - d0) if np.isfinite(d0) and np.isfinite(d1) else 0.0
        ret.iloc[i] = (cur - flow) / prev - 1.0
    return ret.dropna()


def _build_benchmark_stats_table(
    *,
    metrics: pd.DataFrame,
    benchmark_returns_df: pd.DataFrame | None,
) -> pd.DataFrame:
    # AGENT_NOTE: Produces alpha/beta stats displayed in Section 4 benchmark tab.
    if benchmark_returns_df is None or benchmark_returns_df.empty:
        return pd.DataFrame()
    pr = _portfolio_daily_twr_returns(metrics)
    if pr.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for col in benchmark_returns_df.columns:
        rb = pd.to_numeric(benchmark_returns_df[col], errors="coerce").dropna()
        if rb.empty:
            continue
        aligned = pd.concat([pr.rename("portfolio"), rb.rename("benchmark")], axis=1).dropna()
        if aligned.shape[0] < 40:
            continue
        cov = float(np.cov(aligned["portfolio"], aligned["benchmark"])[0, 1])
        var_b = float(np.var(aligned["benchmark"]))
        beta = cov / var_b if var_b > 1e-12 else np.nan
        mean_p = float(aligned["portfolio"].mean())
        mean_b = float(aligned["benchmark"].mean())
        alpha_daily = mean_p - beta * mean_b if np.isfinite(beta) else np.nan
        alpha_ann = alpha_daily * 252.0 if np.isfinite(alpha_daily) else np.nan
        corr = float(aligned["portfolio"].corr(aligned["benchmark"]))
        tracking_err = float((aligned["portfolio"] - aligned["benchmark"]).std(ddof=0) * np.sqrt(252.0))
        port_cum = float((1.0 + aligned["portfolio"]).prod() - 1.0)
        bench_cum = float((1.0 + aligned["benchmark"]).prod() - 1.0)
        rows.append(
            {
                "benchmark": str(col),
                "beta": beta,
                "alpha_annualized_pct": alpha_ann * 100.0 if np.isfinite(alpha_ann) else np.nan,
                "correlation": corr,
                "tracking_error_annualized_pct": tracking_err * 100.0 if np.isfinite(tracking_err) else np.nan,
                "portfolio_return_aligned_pct": port_cum * 100.0 if np.isfinite(port_cum) else np.nan,
                "benchmark_return_aligned_pct": bench_cum * 100.0 if np.isfinite(bench_cum) else np.nan,
                "sample_days": int(aligned.shape[0]),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("sample_days", ascending=False).reset_index(drop=True)


def build_drawdown_summary_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = _normalized_metrics_df(metrics_df)
    if df.empty or "portfolio_value" not in df.columns:
        return pd.DataFrame()

    equity = pd.to_numeric(df["portfolio_value"], errors="coerce").dropna()
    if equity.empty:
        return pd.DataFrame()

    running_peak = equity.cummax()
    drawdown = (equity / running_peak) - 1.0

    worst_date = drawdown.idxmin()
    worst_dd = float(drawdown.min())
    current_dd = float(drawdown.iloc[-1])

    peak_mask = equity.eq(running_peak)
    last_peak_date = equity.index[peak_mask][-1] if peak_mask.any() else equity.index[-1]
    underwater_days = int((equity.index[-1] - last_peak_date).days) if current_dd < 0.0 else 0
    recovery_needed_pct = np.nan
    if current_dd < 0.0 and current_dd > -0.999:
        recovery_needed_pct = ((1.0 / (1.0 + current_dd)) - 1.0) * 100.0

    rows = [
        {"metric": "Current drawdown (%)", "value": current_dd * 100.0},
        {
            "metric": "Max drawdown (%)",
            "value": worst_dd * 100.0,
            "context": worst_date.strftime("%Y-%m-%d"),
        },
        {"metric": "Days since last peak", "value": underwater_days},
        {"metric": "Recovery needed to new high (%)", "value": recovery_needed_pct},
    ]
    out = pd.DataFrame(rows)
    if "context" not in out.columns:
        out["context"] = ""
    out["context"] = out["context"].fillna("")
    return out[["metric", "value", "context"]]


def build_action_plan_table(
    *,
    metrics_df: pd.DataFrame,
    period_performance_df: pd.DataFrame,
    over_target_df: pd.DataFrame,
) -> pd.DataFrame:
    df = _normalized_metrics_df(metrics_df)
    if df.empty or "portfolio_value" not in df.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    latest = df.iloc[-1]
    portfolio_value = float(latest.get("portfolio_value", np.nan))
    cash_value = float(latest.get("cash", np.nan))
    cash_pct = (cash_value / portfolio_value * 100.0) if portfolio_value > 0 else np.nan

    if np.isfinite(cash_pct) and cash_pct > 15.0:
        deploy_eur = max((cash_pct - 10.0) / 100.0 * portfolio_value, 0.0)
        rows.append(
            {
                "priority": "High",
                "theme": "Cash deployment",
                "what_we_see": f"Cash is {cash_pct:,.1f}% of portfolio.",
                "future_action": f"Deploy about EUR {deploy_eur:,.2f} gradually toward under-target allocations.",
            }
        )
    elif np.isfinite(cash_pct) and cash_pct < 3.0:
        rows.append(
            {
                "priority": "Medium",
                "theme": "Liquidity buffer",
                "what_we_see": f"Cash is only {cash_pct:,.1f}% of portfolio.",
                "future_action": "Pause part of new buys until a 5-10% buffer is restored.",
            }
        )

    if over_target_df is not None and not over_target_df.empty and "over_target_eur" in over_target_df.columns:
        over = over_target_df.copy()
        over = over[pd.to_numeric(over["over_target_eur"], errors="coerce") > 0.0]
        if not over.empty:
            over = over.sort_values("over_target_eur", ascending=False).head(3)
            trim_total = float(pd.to_numeric(over["over_target_eur"], errors="coerce").sum())
            if "ticker" in over.columns:
                name_series = over["ticker"].fillna(over.get("product", ""))
            else:
                name_series = over.get("product", pd.Series([], dtype="object"))
            top_names = ", ".join(str(v) for v in name_series.astype(str).tolist())
            rows.append(
                {
                    "priority": "Medium",
                    "theme": "Rebalancing",
                    "what_we_see": f"Top over-target holdings: {top_names}",
                    "future_action": f"Trim or pause adds; rebalance about EUR {trim_total:,.2f} into under-target names.",
                }
            )

    if period_performance_df is not None and not period_performance_df.empty:
        p90 = period_performance_df.loc[period_performance_df["period"] == "90D"]
        if not p90.empty:
            pnl90 = float(p90.iloc[0]["investment_pnl_eur"])
            if np.isfinite(pnl90) and pnl90 < 0.0:
                rows.append(
                    {
                        "priority": "High",
                        "theme": "Risk control",
                        "what_we_see": f"Last 90D investment P/L is EUR {pnl90:,.2f}.",
                        "future_action": "Reduce concentration risk and favor gradual ETF accumulation for new capital.",
                    }
                )
            elif np.isfinite(pnl90):
                rows.append(
                    {
                        "priority": "Low",
                        "theme": "Momentum",
                        "what_we_see": f"Last 90D investment P/L is EUR {pnl90:,.2f}.",
                        "future_action": "Keep current approach, but rebalance if any name drifts materially above target.",
                    }
                )

    if "total_deposits" in df.columns:
        flow = pd.to_numeric(df["total_deposits"], errors="coerce").diff().fillna(
            pd.to_numeric(df["total_deposits"], errors="coerce")
        )
        monthly = flow.resample("ME").sum()
        if not monthly.empty:
            recent = monthly.tail(6)
            avg_recent = float(recent.mean())
            rows.append(
                {
                    "priority": "Low",
                    "theme": "Contribution pace",
                    "what_we_see": f"Average net flow over last {len(recent)} month(s): EUR {avg_recent:,.2f}/month.",
                    "future_action": "Set a fixed monthly transfer close to this level to stabilize progress.",
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_summary_lines(
    *,
    metrics_df: pd.DataFrame,
    period_performance_df: pd.DataFrame,
    drawdown_summary_df: pd.DataFrame,
) -> list[str]:
    lines: list[str] = []
    df = _normalized_metrics_df(metrics_df)
    if df.empty or "portfolio_value" not in df.columns:
        return lines

    if period_performance_df is not None and not period_performance_df.empty:
        since = period_performance_df.loc[period_performance_df["period"] == "Since start"]
        if not since.empty:
            row = since.iloc[0]
            lines.append(
                "Since start: total change EUR "
                f"{float(row['total_change_eur']):,.2f} = net flows EUR {float(row['net_flow_eur']):,.2f} "
                f"+ investment P/L EUR {float(row['investment_pnl_eur']):,.2f}."
            )

    latest = df.iloc[-1]
    portfolio_value = float(latest.get("portfolio_value", np.nan))
    cash_value = float(latest.get("cash", np.nan))
    if np.isfinite(portfolio_value) and portfolio_value > 0 and np.isfinite(cash_value):
        cash_pct = cash_value / portfolio_value * 100.0
        lines.append(f"Current cash allocation: {cash_pct:,.1f}% (EUR {cash_value:,.2f}).")

    if drawdown_summary_df is not None and not drawdown_summary_df.empty:
        cur = drawdown_summary_df.loc[
            drawdown_summary_df["metric"] == "Current drawdown (%)", "value"
        ]
        max_dd = drawdown_summary_df.loc[
            drawdown_summary_df["metric"] == "Max drawdown (%)", "value"
        ]
        if not cur.empty and not max_dd.empty:
            lines.append(
                f"Drawdown status: current {float(cur.iloc[0]):,.2f}% vs max historical {float(max_dd.iloc[0]):,.2f}%."
            )
    return lines


def _normalized_metrics_df(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame()
    df = metrics_df.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].sort_index()
    return df


def _compute_twr(sub: pd.DataFrame) -> float:
    values = pd.to_numeric(sub["portfolio_value"], errors="coerce")
    deposits = pd.to_numeric(sub["total_deposits"], errors="coerce")
    if len(values) < 2:
        return np.nan

    factors: list[float] = []
    for i in range(1, len(values)):
        prev = float(values.iloc[i - 1])
        cur = float(values.iloc[i])
        if not np.isfinite(prev) or prev <= 0.0 or not np.isfinite(cur):
            continue
        d0 = float(deposits.iloc[i - 1]) if np.isfinite(deposits.iloc[i - 1]) else np.nan
        d1 = float(deposits.iloc[i]) if np.isfinite(deposits.iloc[i]) else np.nan
        flow = (d1 - d0) if np.isfinite(d0) and np.isfinite(d1) else 0.0
        factor = (cur - flow) / prev
        if np.isfinite(factor) and factor > 0.0:
            factors.append(float(factor))

    if not factors:
        return np.nan
    return float(np.prod(factors) - 1.0)


def _normalized_holdings_for_spread(holdings_df: pd.DataFrame) -> pd.DataFrame:
    if holdings_df is None or holdings_df.empty:
        return pd.DataFrame(
            columns=["instrument_id", "product", "ticker", "currency", "is_etf", "quantity", "value_eur"]
        )
    df = holdings_df.copy()
    for col in ["instrument_id", "product", "ticker", "currency"]:
        if col not in df.columns:
            df[col] = ""
    if "is_etf" not in df.columns:
        df["is_etf"] = False
    if "quantity" not in df.columns:
        df["quantity"] = np.nan
    if "value_eur" not in df.columns:
        df["value_eur"] = 0.0

    df["instrument_id"] = df["instrument_id"].fillna("").astype(str)
    df["product"] = df["product"].fillna("").astype(str).str.strip()
    df["ticker"] = df["ticker"].fillna("").astype(str).str.strip()
    df["currency"] = df["currency"].fillna("UNKNOWN").astype(str).str.upper().str.strip()
    df["currency"] = df["currency"].replace("", "UNKNOWN")
    df["is_etf"] = df["is_etf"].fillna(False).astype(bool)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["value_eur"] = pd.to_numeric(df["value_eur"], errors="coerce").fillna(0.0)

    grouped = (
        df.groupby(["instrument_id", "product", "ticker", "currency", "is_etf"], dropna=False, as_index=False)
        .agg(
            quantity=("quantity", "sum"),
            value_eur=("value_eur", "sum"),
        )
        .sort_values("value_eur", ascending=False)
    )
    # Strategy checks only apply to open holdings. Remove fully closed rows.
    open_mask = (
        pd.to_numeric(grouped["quantity"], errors="coerce").abs() > 1e-9
    ) | (pd.to_numeric(grouped["value_eur"], errors="coerce").abs() > 1e-6)
    grouped = grouped.loc[open_mask].copy()
    return grouped.reset_index(drop=True)


def _build_currency_allocation_df(
    *,
    holdings: pd.DataFrame,
    cash_detail_df: pd.DataFrame,
    fallback_cash_eur: float,
    total_value: float,
) -> pd.DataFrame:
    inv = holdings.groupby("currency", dropna=False)["value_eur"].sum().rename("investments_eur")
    inv.index = inv.index.map(lambda x: str(x).upper() if pd.notna(x) else "UNKNOWN")

    cash = pd.Series(dtype="float64")
    if isinstance(cash_detail_df, pd.DataFrame) and not cash_detail_df.empty and "currency" in cash_detail_df.columns:
        d = cash_detail_df.copy()
        d["currency"] = d["currency"].fillna("").astype(str).str.upper().str.strip()
        d = d[~d["currency"].isin({"", "NAN", "TOTAL", "<NONE>"})]
        value_col = "balance_eur_computed" if "balance_eur_computed" in d.columns else "balance_eur"
        if value_col in d.columns:
            d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
            cash = d.groupby("currency")[value_col].sum(min_count=1)
            cash = cash.fillna(0.0)
    if cash.empty and np.isfinite(fallback_cash_eur):
        cash = pd.Series({"EUR": float(fallback_cash_eur)}, dtype="float64")

    idx = sorted(set(inv.index).union(set(cash.index)))
    out = pd.DataFrame(index=idx)
    out["investments_eur"] = inv.reindex(idx).fillna(0.0).astype(float)
    out["cash_eur"] = cash.reindex(idx).fillna(0.0).astype(float)
    out["total_eur"] = out["investments_eur"] + out["cash_eur"]
    out["pct_total"] = np.where(total_value > 0.0, out["total_eur"] / total_value * 100.0, np.nan)
    out.index.name = "currency"
    out = out.reset_index().sort_values("total_eur", ascending=False).reset_index(drop=True)
    return out


def _build_industry_allocation_df(*, holdings: pd.DataFrame, total_value: float) -> pd.DataFrame:
    if holdings.empty:
        return pd.DataFrame(columns=["industry", "value_eur", "pct_total", "holding_count"])
    out = holdings.copy()
    if "industry" not in out.columns:
        out["industry"] = "Unclassified"
    out["industry"] = out["industry"].fillna("Unclassified").astype(str).str.strip().replace("", "Unclassified")
    grouped = (
        out.groupby("industry", dropna=False)
        .agg(
            value_eur=("value_eur", "sum"),
            holding_count=("instrument_id", "nunique"),
        )
        .reset_index()
    )
    grouped["pct_total"] = np.where(total_value > 0.0, grouped["value_eur"] / total_value * 100.0, np.nan)
    return grouped.sort_values("value_eur", ascending=False).reset_index(drop=True)


def _build_style_allocation_df(*, holdings: pd.DataFrame, total_value: float) -> pd.DataFrame:
    if holdings.empty:
        return pd.DataFrame(columns=["style", "value_eur", "pct_total", "holding_count"])
    out = holdings.copy()
    if "style" not in out.columns:
        out["style"] = "Unclassified"
    out["style"] = out["style"].fillna("Unclassified").astype(str).str.strip().replace("", "Unclassified")
    grouped = (
        out.groupby("style", dropna=False)
        .agg(
            value_eur=("value_eur", "sum"),
            holding_count=("instrument_id", "nunique"),
        )
        .reset_index()
    )
    grouped["pct_total"] = np.where(total_value > 0.0, grouped["value_eur"] / total_value * 100.0, np.nan)
    return grouped.sort_values("value_eur", ascending=False).reset_index(drop=True)


def _to_target_pct_map(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        label = str(key).strip()
        if label == "":
            continue
        pct = _to_float(value, np.nan)
        if np.isfinite(pct):
            out[label] = max(float(pct), 0.0)
    return out


def _normalize_holding_category_overrides(raw: Any) -> dict[str, dict[str, str]]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, str]] = {}
    for key, value in raw.items():
        instrument_key = str(key).strip()
        if instrument_key == "" or not isinstance(value, dict):
            continue
        style = str(value.get("style", "")).strip()
        industry = str(value.get("industry", "")).strip()
        normalized: dict[str, str] = {}
        if style:
            normalized["style"] = style
        if industry:
            normalized["industry"] = industry
        if normalized:
            out[instrument_key] = normalized
    return out


def _apply_holding_category_overrides(
    *,
    holdings: pd.DataFrame,
    overrides: dict[str, dict[str, str]],
) -> pd.DataFrame:
    if holdings.empty or not overrides:
        return holdings
    out = holdings.copy()
    for idx in out.index:
        key = str(out.at[idx, "instrument_id"]).strip()
        override = overrides.get(key)
        if not isinstance(override, dict):
            continue
        if "style" in override:
            out.at[idx, "style"] = str(override["style"]).strip()
        if "industry" in override:
            out.at[idx, "industry"] = str(override["industry"]).strip()
    return out


def _attach_target_pct_columns(
    *,
    allocation_df: pd.DataFrame,
    key_col: str,
    value_col: str,
    total_col: str,
    target_pct_map: dict[str, float],
    total_value: float,
) -> pd.DataFrame:
    # AGENT_NOTE: Standardizes target/gap columns for currency/industry/style.
    # Streamlit tables assume these names exist when target maps are provided.
    if allocation_df is None or allocation_df.empty:
        return pd.DataFrame()
    out = allocation_df.copy()
    if key_col not in out.columns:
        return out
    if not target_pct_map:
        out["target_pct"] = np.nan
        out["delta_vs_target_pct"] = np.nan
        out["target_value_eur"] = np.nan
        out["gap_to_target_eur"] = np.nan
        return out

    target_lookup = {str(k).strip(): float(v) for k, v in target_pct_map.items()}
    out["target_pct"] = out[key_col].astype(str).map(target_lookup)
    out["target_value_eur"] = np.where(
        out["target_pct"].notna(),
        out["target_pct"] / 100.0 * total_value,
        np.nan,
    )
    out["delta_vs_target_pct"] = np.where(
        out["target_pct"].notna(),
        pd.to_numeric(out[total_col], errors="coerce") - out["target_pct"],
        np.nan,
    )
    out["gap_to_target_eur"] = np.where(
        out["target_pct"].notna(),
        out["target_value_eur"] - pd.to_numeric(out[value_col], errors="coerce"),
        np.nan,
    )
    return out


def _build_allocation_plan_from_targets(
    *,
    allocation_df: pd.DataFrame,
    key_col: str,
    value_col: str,
    target_pct_map: dict[str, float],
    deployable_cash: float,
    total_value: float,
) -> pd.DataFrame:
    if not target_pct_map or deployable_cash <= 0.0:
        return pd.DataFrame(columns=["bucket", "target_pct", "current_pct", "allocation_eur"])
    current_map: dict[str, float] = {}
    current_pct_map: dict[str, float] = {}
    if isinstance(allocation_df, pd.DataFrame) and not allocation_df.empty:
        for row in allocation_df.itertuples(index=False):
            key = str(getattr(row, key_col, "")).strip()
            if key == "":
                continue
            current_map[key] = float(_to_float(getattr(row, value_col, np.nan), 0.0))
            current_pct_map[key] = float(_to_float(getattr(row, "pct_total", np.nan), np.nan))
    rows: list[dict[str, Any]] = []
    positive_gap_total = 0.0
    for bucket, target_pct in target_pct_map.items():
        target_value = max(float(target_pct), 0.0) / 100.0 * total_value
        current_value = current_map.get(bucket, 0.0)
        gap = max(target_value - current_value, 0.0)
        positive_gap_total += gap
        rows.append(
            {
                "bucket": bucket,
                "target_pct": float(target_pct),
                "current_pct": current_pct_map.get(bucket, np.nan),
                "gap_eur": gap,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if positive_gap_total > 0.0:
        out["allocation_eur"] = deployable_cash * out["gap_eur"] / positive_gap_total
    else:
        target_sum = float(np.nansum(out["target_pct"]))
        if target_sum > 0.0:
            out["allocation_eur"] = deployable_cash * out["target_pct"] / target_sum
        else:
            out["allocation_eur"] = 0.0
    out = out.drop(columns=["gap_eur"])
    return out.sort_values("allocation_eur", ascending=False).reset_index(drop=True)


def _compute_high_correlation_pairs(
    *,
    holdings: pd.DataFrame,
    prices_eur: pd.DataFrame | None,
    min_correlation: float,
    min_history_days: int = 60,
    lookback_days: int = 365,
) -> pd.DataFrame:
    # AGENT_NOTE: Flags potentially redundant holdings for diversification checks.
    # Used by spread analysis UI and startup recommendations.
    if holdings is None or holdings.empty:
        return pd.DataFrame()
    if prices_eur is None or prices_eur.empty:
        return pd.DataFrame()
    series = prices_eur.copy()
    series.columns = series.columns.astype(str)
    ids = [str(v) for v in holdings["instrument_id"].astype(str).tolist() if str(v) in series.columns]
    if len(ids) < 2:
        return pd.DataFrame()
    px = series[ids].copy()
    px.index = pd.to_datetime(px.index, errors="coerce")
    px = px[px.index.notna()].sort_index()
    if px.empty:
        return pd.DataFrame()
    cutoff = px.index.max() - pd.Timedelta(days=int(lookback_days))
    px = px.loc[px.index >= cutoff]
    ret = px.pct_change()
    valid_cols = [col for col in ret.columns if int(ret[col].notna().sum()) >= int(min_history_days)]
    if len(valid_cols) < 2:
        return pd.DataFrame()
    ret = ret[valid_cols]
    corr = ret.corr(min_periods=int(min_history_days))
    if corr.empty:
        return pd.DataFrame()

    meta = holdings.drop_duplicates(subset=["instrument_id"], keep="first").set_index("instrument_id")
    rows: list[dict[str, Any]] = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a = cols[i]
            b = cols[j]
            value = corr.at[a, b]
            if not np.isfinite(value) or float(value) < float(min_correlation):
                continue
            pa = meta.loc[a] if a in meta.index else pd.Series(dtype="object")
            pb = meta.loc[b] if b in meta.index else pd.Series(dtype="object")
            rows.append(
                {
                    "instrument_id_a": a,
                    "ticker_a": str(pa.get("ticker", a)),
                    "product_a": str(pa.get("product", a)),
                    "weight_a_pct": _to_float(pa.get("value_pct_total", np.nan), np.nan),
                    "instrument_id_b": b,
                    "ticker_b": str(pb.get("ticker", b)),
                    "product_b": str(pb.get("product", b)),
                    "weight_b_pct": _to_float(pb.get("value_pct_total", np.nan), np.nan),
                    "combined_weight_pct": _to_float(pa.get("value_pct_total", np.nan), 0.0)
                    + _to_float(pb.get("value_pct_total", np.nan), 0.0),
                    "correlation": float(value),
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["correlation", "combined_weight_pct"], ascending=False)
    return out.reset_index(drop=True)


def _pct(value: float, total: float) -> float:
    if not np.isfinite(value) or not np.isfinite(total) or total == 0.0:
        return np.nan
    return value / total * 100.0


def _to_float(value: Any, default: float) -> float:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
    except Exception:
        pass
    return float(default)


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _status_max(actual: float, limit: float) -> str:
    if not np.isfinite(actual):
        return "N/A"
    return "OK" if actual <= limit else "Above limit"


def _status_min(actual: float, minimum: float) -> str:
    if not np.isfinite(actual):
        return "N/A"
    return "OK" if actual >= minimum else "Below minimum"


def _status_target(actual: float, target: float, tolerance: float) -> str:
    if not np.isfinite(actual):
        return "N/A"
    if abs(actual - target) <= tolerance:
        return "On target"
    if actual > target:
        return "Above target"
    return "Below target"
