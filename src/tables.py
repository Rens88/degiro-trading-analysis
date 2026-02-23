"""
AGENT_NOTE: Holdings aggregation and strategy threshold tables.

Interdependencies:
- Consumes normalized holdings and `TotalsResult`.
- `over_target` output is rendered in `app.py` and used by
  `strategy_check.py` for action-required signaling.

When editing:
- Keep output column names stable for UI rendering and CLI formatting.
- Keep target-per-holding math aligned with spread strategy assumptions.
- See `src/INTERDEPENDENCIES.md` for contract details.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .reconciliation import TotalsResult


def build_four_tables(
    *,
    holdings: pd.DataFrame,
    totals: TotalsResult,
    target_etf_fraction: float,
    desired_etf_holdings: int,
    desired_non_etf_holdings: int,
    min_over_value_eur: float,
) -> dict[str, pd.DataFrame]:
    aggregated = aggregate_holdings(holdings)
    if aggregated.empty:
        empty = pd.DataFrame()
        return {"etf": empty, "non_etf": empty, "summary": empty, "over_target": empty}

    total_value = float(totals.total_value_eur)
    aggregated["value_pct"] = np.where(total_value != 0, aggregated["value_eur"] / total_value * 100.0, np.nan)

    n_etf = max(int(desired_etf_holdings), 1)
    n_non_etf = max(int(desired_non_etf_holdings), 1)
    aggregated["target_per_holding_pct"] = np.where(
        aggregated["is_etf"],
        target_etf_fraction / n_etf * 100.0,
        (1.0 - target_etf_fraction) / n_non_etf * 100.0,
    )
    aggregated["target_value_eur"] = aggregated["target_per_holding_pct"] / 100.0 * total_value
    aggregated["over_target_eur"] = aggregated["value_eur"] - aggregated["target_value_eur"]

    etf = aggregated[aggregated["is_etf"]].copy().sort_values("value_eur", ascending=False)
    non_etf = aggregated[~aggregated["is_etf"]].copy().sort_values("value_eur", ascending=False)
    over_target = aggregated[aggregated["over_target_eur"] > min_over_value_eur].copy().sort_values(
        "over_target_eur",
        ascending=False,
    )

    etf_value = float(etf["value_eur"].sum())
    non_etf_value = float(non_etf["value_eur"].sum())
    cash_value = float(totals.cash_value_eur)
    combined_value = float(total_value)
    eur_to_etf = target_etf_fraction * combined_value - etf_value
    eur_to_non_etf = (1.0 - target_etf_fraction) * combined_value - non_etf_value
    identity_total = etf_value + non_etf_value + cash_value

    summary = pd.DataFrame(
        [
            {"metric": "combined value", "value_eur": combined_value},
            {"metric": "non-ETF value", "value_eur": non_etf_value},
            {"metric": "ETF value", "value_eur": etf_value},
            {"metric": "cash position", "value_eur": cash_value},
            {"metric": "suggested EUR to ETF for next purchases", "value_eur": eur_to_etf},
            {
                "metric": "suggested EUR to non-ETF for next purchases",
                "value_eur": eur_to_non_etf,
            },
            {"metric": "check ETF + non-ETF + cash", "value_eur": identity_total},
            {"metric": "identity delta (should be 0)", "value_eur": identity_total - combined_value},
        ]
    )

    display_cols = [
        "product",
        "ticker",
        "quantity",
        "value_eur",
        "value_pct",
        "target_per_holding_pct",
        "target_value_eur",
        "over_target_eur",
        "isin",
        "account_label",
    ]
    for df in (etf, non_etf, over_target):
        for col in display_cols:
            if col not in df.columns:
                df[col] = np.nan

    return {
        "etf": etf[display_cols].reset_index(drop=True),
        "non_etf": non_etf[display_cols].reset_index(drop=True),
        "summary": summary,
        "over_target": over_target[display_cols].reset_index(drop=True),
    }


def aggregate_holdings(holdings: pd.DataFrame) -> pd.DataFrame:
    if holdings.empty:
        return pd.DataFrame()
    df = holdings.copy()
    group_cols = ["instrument_id", "product", "isin", "ticker", "is_etf"]

    def joined_accounts(values: pd.Series) -> str:
        unique = sorted({str(v) for v in values.dropna().unique()})
        return ", ".join(unique)

    grouped = df.groupby(group_cols, dropna=False, as_index=False).agg(
        account_label=("account_label", joined_accounts),
        quantity=("quantity", "sum"),
        value_eur=("value_eur", "sum"),
    )
    grouped = grouped.sort_values("value_eur", ascending=False)
    return grouped


def build_monthly_starting_portfolio_value_table(
    *,
    per_dataset_metrics: dict[str, pd.DataFrame],
    manual_tracked_values: pd.DataFrame | None = None,
) -> pd.DataFrame:
    series_by_label: dict[str, pd.Series] = {}
    for label, metrics in per_dataset_metrics.items():
        if metrics is None or metrics.empty or "portfolio_value" not in metrics.columns:
            continue
        index = pd.to_datetime(metrics.index, errors="coerce")
        values = pd.to_numeric(metrics["portfolio_value"], errors="coerce")
        series = pd.Series(values.values, index=index)
        series = series[series.index.notna() & series.notna()]
        if series.empty:
            continue
        series.index = series.index.normalize()
        series = series.groupby(level=0).last().sort_index()
        if not series.empty:
            series_by_label[label] = series.astype(float)

    if not series_by_label:
        base = pd.DataFrame(columns=["month_start", "valuation_date", "total_portfolio_value_eur"])
    else:
        labels = sorted(series_by_label.keys())
        min_date = min(s.index.min() for s in series_by_label.values())
        max_date = max(s.index.max() for s in series_by_label.values())
        month_starts = pd.date_range(start=min_date, end=max_date, freq="MS")

        rows: list[dict[str, float | pd.Timestamp]] = []
        for month_start in month_starts:
            valuation_day = (month_start - pd.Timedelta(days=1)).normalize()
            row: dict[str, float | pd.Timestamp] = {
                "month_start": month_start,
                "valuation_date": valuation_day,
            }
            values: list[float] = []
            for label in labels:
                value = float(series_by_label[label].get(valuation_day, np.nan))
                row[f"{label}_portfolio_value_eur"] = value
                values.append(value)

            if not np.isfinite(values).any():
                continue

            row["total_portfolio_value_eur"] = float(np.nansum(values))
            rows.append(row)

        base = pd.DataFrame(rows)

    if manual_tracked_values is not None and not manual_tracked_values.empty:
        manual = manual_tracked_values.copy()
        manual["tracked_date"] = pd.to_datetime(manual["tracked_date"], errors="coerce")
        manual["manual_tracked_total_eur"] = pd.to_numeric(
            manual["manual_tracked_total_eur"], errors="coerce"
        )
        manual = manual.dropna(subset=["tracked_date", "manual_tracked_total_eur"])
        if not manual.empty:
            manual["month_start"] = manual["tracked_date"].dt.to_period("M").dt.to_timestamp(how="start")
            manual = manual.sort_values(["month_start", "tracked_date"])
            manual = manual.groupby("month_start", as_index=False).tail(1)
            manual = manual.rename(columns={"tracked_date": "manual_source_date"})
            manual = manual.loc[:, ["month_start", "manual_source_date", "manual_tracked_total_eur"]]

            if base.empty:
                base = pd.DataFrame({"month_start": manual["month_start"].copy()})
            base = base.merge(manual, on="month_start", how="outer")
            base["valuation_date"] = base["valuation_date"].where(
                base["valuation_date"].notna(),
                base["month_start"] - pd.Timedelta(days=1),
            )
            base["delta_total_vs_manual_eur"] = (
                base["total_portfolio_value_eur"] - base["manual_tracked_total_eur"]
            )

    if base.empty:
        return base

    base = base.sort_values("month_start").reset_index(drop=True)
    for col in ["month_start", "valuation_date", "manual_source_date"]:
        if col in base.columns:
            base[col] = pd.to_datetime(base[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return base
