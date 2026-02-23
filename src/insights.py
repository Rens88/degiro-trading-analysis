from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


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


def build_ai_spread_analysis(
    *,
    holdings_df: pd.DataFrame,
    total_value_eur: float,
    cash_value_eur: float,
    cash_detail_df: pd.DataFrame,
    strategy: dict[str, Any],
) -> dict[str, Any]:
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
    min_total_holdings = max(_to_int(strategy.get("min_total_holdings"), 12), 1)

    etf_mask = holdings["is_etf"].fillna(False).astype(bool)
    etf_value = float(holdings.loc[etf_mask, "value_eur"].sum())
    non_etf_value = float(holdings.loc[~etf_mask, "value_eur"].sum())
    etf_count = int(holdings.loc[etf_mask, "instrument_id"].nunique())
    non_etf_count = int(holdings.loc[~etf_mask, "instrument_id"].nunique())
    total_holdings = int(holdings["instrument_id"].nunique())

    etf_pct = _pct(etf_value, total_value)
    non_etf_pct = _pct(non_etf_value, total_value)
    cash_pct = _pct(cash_value, total_value)
    etf_non_etf_ratio = np.nan
    if non_etf_value > 0.0:
        etf_non_etf_ratio = etf_value / non_etf_value

    concentration_df = holdings.sort_values("value_eur", ascending=False).copy()
    concentration_df["value_pct_total"] = concentration_df["value_eur"] / total_value * 100.0
    concentration_df = concentration_df[
        ["product", "ticker", "currency", "is_etf", "value_eur", "value_pct_total"]
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
            "metric": "Total non-cash holdings (count)",
            "actual": float(total_holdings),
            "target_or_limit": f">= {min_total_holdings:d}",
            "delta": float(total_holdings - min_total_holdings),
            "status": _status_min(float(total_holdings), float(min_total_holdings)),
        },
    ]
    strategy_checks_df = pd.DataFrame(strategy_rows)

    action_rows: list[dict[str, Any]] = []
    if cash_pct > target_cash_pct + 2.0:
        deploy_eur = (cash_pct - target_cash_pct) / 100.0 * total_value
        action_rows.append(
            {
                "priority": "High",
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
                "theme": "ETF / non-ETF spread",
                "what_we_see": f"ETF share is {etf_pct:,.1f}% vs target {target_etf_fraction * 100.0:,.1f}%.",
                "future_action": f"Favor ETF adds by about EUR {shift:,.2f} over next contributions.",
            }
        )

    if largest_holding_pct > max_single_holding_pct:
        action_rows.append(
            {
                "priority": "High",
                "theme": "Single-name concentration",
                "what_we_see": f"Largest position is {largest_holding_pct:,.1f}% of portfolio.",
                "future_action": "Pause adds to the largest holding and diversify incremental capital.",
            }
        )
    if top5_pct > max_top5_holdings_pct:
        action_rows.append(
            {
                "priority": "Medium",
                "theme": "Top-5 concentration",
                "what_we_see": f"Top 5 holdings represent {top5_pct:,.1f}% of portfolio.",
                "future_action": "Direct new buys toward smaller positions until concentration declines.",
            }
        )
    if largest_currency_pct > max_single_currency_pct:
        action_rows.append(
            {
                "priority": "Medium",
                "theme": "Currency concentration",
                "what_we_see": f"Largest currency exposure is {largest_currency_name} at {largest_currency_pct:,.1f}%.",
                "future_action": "Add positions/cash in other currencies to lower single-currency dependency.",
            }
        )
    if largest_industry_pct > max_single_industry_pct:
        action_rows.append(
            {
                "priority": "Medium",
                "theme": "Industry concentration",
                "what_we_see": f"Largest industry bucket is {largest_industry_name} at {largest_industry_pct:,.1f}%.",
                "future_action": "Rebalance future additions toward underrepresented industries.",
            }
        )
    if total_holdings < min_total_holdings:
        action_rows.append(
            {
                "priority": "Low",
                "theme": "Breadth",
                "what_we_see": f"Portfolio has {total_holdings} non-cash holdings vs target minimum {min_total_holdings}.",
                "future_action": "Increase breadth gradually while preserving ETF/non-ETF strategy.",
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

    action_plan_df = pd.DataFrame(action_rows)
    return {
        "summary_lines": summary_lines,
        "strategy_checks_df": strategy_checks_df,
        "etf_non_etf_df": etf_non_etf_df,
        "currency_allocation_df": currency_allocation_df,
        "industry_allocation_df": industry_allocation_df,
        "concentration_df": concentration_df.reset_index(drop=True),
        "action_plan_df": action_plan_df.reset_index(drop=True),
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
    return grouped


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
    out["industry"] = out.apply(
        lambda row: _infer_industry_bucket(
            product=str(row.get("product", "")),
            ticker=str(row.get("ticker", "")),
            is_etf=bool(row.get("is_etf", False)),
        ),
        axis=1,
    )
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


def _infer_industry_bucket(*, product: str, ticker: str, is_etf: bool) -> str:
    txt = f"{product} {ticker}".upper()
    rules = [
        ("Technology", ["TECH", "SOFTWARE", "SEMICON", "CHIP", "CLOUD", "CYBER", "NASDAQ"]),
        ("Financials", ["BANK", "FINAN", "INSUR", "PAYMENT", "CAPITAL"]),
        ("Healthcare", ["HEALTH", "PHARMA", "BIOTECH", "MEDIC", "THERAP"]),
        ("Consumer", ["CONSUM", "RETAIL", "FOOD", "BEVERAGE", "LUXURY", "APPAREL"]),
        ("Industrials", ["INDUSTR", "AEROSPACE", "DEFEN", "TRANSPORT", "LOGISTICS"]),
        ("Energy", ["ENERGY", "OIL", "GAS", "SOLAR", "WIND", "UTILIT"]),
        ("Materials", ["MINING", "METAL", "STEEL", "CHEMICAL", "MATERIAL"]),
        ("Real Estate", ["REAL ESTATE", "REIT", "PROPERTY"]),
        ("Communication", ["TELECOM", "MEDIA", "COMMUNICATION", "INTERNET"]),
    ]
    for name, keywords in rules:
        if any(k in txt for k in keywords):
            return name
    if is_etf:
        if any(k in txt for k in ["MSCI", "S&P", "STOXX", "WORLD", "ALL-WORLD", "ACWI"]):
            return "Broad-market ETF"
        return "ETF (other)"
    return "Unclassified"


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
