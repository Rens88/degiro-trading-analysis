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

from typing import Any

import numpy as np
import pandas as pd

from .reconciliation import TotalsResult

PRODUCT_NAME_GROUP_COLS = ["instrument_id", "isin", "ticker", "is_etf"]


def _normalize_text_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _join_unique_non_empty(values: pd.Series) -> str:
    unique = sorted({_normalize_text_value(v) for v in values if _normalize_text_value(v) != ""})
    return ", ".join(unique)


def unify_holding_product_names(
    holdings: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if holdings is None:
        return pd.DataFrame(), []
    if holdings.empty:
        return holdings.copy(), []

    df = holdings.copy()
    for col in PRODUCT_NAME_GROUP_COLS:
        if col not in df.columns:
            df[col] = np.nan
    if "product" not in df.columns:
        return df, []
    if "account_label" not in df.columns:
        df["account_label"] = np.nan

    inconsistencies: list[dict[str, Any]] = []
    for _, block in df.groupby(PRODUCT_NAME_GROUP_COLS, dropna=False, sort=False):
        product_values = block["product"].map(_normalize_text_value)
        ordered_products: list[str] = []
        seen_products: set[str] = set()
        for product_name in product_values:
            if product_name not in seen_products:
                seen_products.add(product_name)
                ordered_products.append(product_name)
        if len(ordered_products) <= 1:
            continue

        chosen_product = ordered_products[0]
        df.loc[block.index, "product"] = chosen_product

        group_df = block.loc[:, PRODUCT_NAME_GROUP_COLS].head(1).reset_index(drop=True).copy()
        product_rows: list[dict[str, Any]] = []
        for product_name in ordered_products:
            mask = product_values.eq(product_name)
            row: dict[str, Any] = {
                "product": product_name,
                "occurrences": int(mask.sum()),
            }
            account_label = _join_unique_non_empty(block.loc[mask, "account_label"])
            if account_label != "":
                row["account_label"] = account_label
            product_rows.append(row)
        products_df = pd.DataFrame(product_rows)
        product_cols = [col for col in ["product", "account_label", "occurrences"] if col in products_df.columns]

        inconsistencies.append(
            {
                "group": group_df,
                "products": products_df.loc[:, product_cols].reset_index(drop=True),
                "chosen_product": chosen_product,
            }
        )
    return df, inconsistencies


def build_latest_valued_holdings(
    holdings: pd.DataFrame,
    *,
    positions: pd.DataFrame | None = None,
    prices_eur: pd.DataFrame | None = None,
    instruments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if holdings is None or holdings.empty:
        aggregated = pd.DataFrame(
            columns=[
                "instrument_id",
                "product",
                "isin",
                "ticker",
                "is_etf",
                "account_label",
                "currency",
                "quantity",
                "value_eur",
            ]
        )
    else:
        aggregated = aggregate_holdings(holdings)

    out = aggregated.copy()
    if "instrument_id" not in out.columns:
        out["instrument_id"] = pd.Series(dtype="object")
    raw_qty = pd.to_numeric(out.get("quantity"), errors="coerce").abs()
    raw_value_abs = pd.to_numeric(out.get("value_eur"), errors="coerce").abs()
    fallback_px = np.where(raw_qty > 1e-12, raw_value_abs / raw_qty, np.nan)

    latest_qty_map: dict[str, float] = {}
    if isinstance(positions, pd.DataFrame) and not positions.empty:
        latest_positions = positions.iloc[-1]
        latest_qty_map = {
            str(idx): float(val)
            for idx, val in latest_positions.items()
            if pd.notna(val) and np.isfinite(float(val))
        }

    latest_price_map: dict[str, float] = {}
    if isinstance(prices_eur, pd.DataFrame) and not prices_eur.empty:
        latest_prices = prices_eur.iloc[-1]
        latest_price_map = {
            str(idx): float(val)
            for idx, val in latest_prices.items()
            if pd.notna(val) and np.isfinite(float(val))
        }

    existing_ids = set(out["instrument_id"].astype(str))
    missing_ids = [
        iid for iid, qty in latest_qty_map.items() if abs(float(qty)) > 1e-12 and iid not in existing_ids
    ]
    if missing_ids:
        instrument_lookup = pd.DataFrame()
        if isinstance(instruments, pd.DataFrame) and not instruments.empty and "instrument_id" in instruments.columns:
            instrument_lookup = instruments.drop_duplicates(subset=["instrument_id"], keep="first").copy()

        rows_to_add: list[dict[str, Any]] = []
        for instrument_id in missing_ids:
            base_row: dict[str, Any] = {
                "instrument_id": instrument_id,
                "product": instrument_id,
                "isin": instrument_id,
                "ticker": "",
                "is_etf": False,
                "account_label": "",
                "currency": "",
                "quantity": np.nan,
                "value_eur": np.nan,
            }
            if not instrument_lookup.empty:
                match = instrument_lookup.loc[instrument_lookup["instrument_id"].astype(str).eq(str(instrument_id))]
                if not match.empty:
                    row = match.iloc[0]
                    base_row["product"] = row.get("product", instrument_id)
                    base_row["isin"] = row.get("isin", instrument_id)
                    base_row["ticker"] = row.get("ticker", "")
                    base_row["currency"] = row.get("currency", "")
                    base_row["account_label"] = row.get("account_label", "")
                    is_etf = row.get("is_etf", False)
                    base_row["is_etf"] = bool(False if pd.isna(is_etf) else is_etf)
            rows_to_add.append(base_row)

        out = pd.concat([out, pd.DataFrame(rows_to_add)], ignore_index=True)
        raw_qty = pd.to_numeric(out.get("quantity"), errors="coerce").abs()
        raw_value_abs = pd.to_numeric(out.get("value_eur"), errors="coerce").abs()
        fallback_px = np.where(raw_qty > 1e-12, raw_value_abs / raw_qty, np.nan)

    if out.empty:
        return out

    out["quantity"] = out["instrument_id"].map(lambda iid: latest_qty_map.get(str(iid), np.nan))
    out["quantity"] = out["quantity"].where(out["quantity"].notna(), raw_qty).abs()
    out["last_px_eur"] = out["instrument_id"].map(lambda iid: latest_price_map.get(str(iid), np.nan))
    out["last_px_eur"] = out["last_px_eur"].where(pd.to_numeric(out["last_px_eur"], errors="coerce").notna(), fallback_px)
    recomputed_value = pd.to_numeric(out["quantity"], errors="coerce") * pd.to_numeric(
        out["last_px_eur"],
        errors="coerce",
    )
    out["value_eur"] = np.where(recomputed_value.notna(), recomputed_value, raw_value_abs)
    out["value_eur"] = pd.to_numeric(out["value_eur"], errors="coerce").fillna(0.0).abs()
    return out


def apply_ranked_target_per_holding_pct(
    holdings: pd.DataFrame,
    *,
    target_etf_pct: float,
    target_non_etf_pct: float,
    desired_etf_holdings: int,
    desired_non_etf_holdings: int,
) -> pd.DataFrame:
    if holdings is None or holdings.empty:
        return pd.DataFrame() if holdings is None else holdings.copy()

    out = holdings.copy()
    out["value_eur"] = pd.to_numeric(out.get("value_eur"), errors="coerce")
    out["is_etf"] = out.get("is_etf", False)
    out["is_etf"] = out["is_etf"].fillna(False).astype(bool)
    out["target_per_holding_pct"] = 0.0
    out["target_rank_in_segment"] = np.nan

    segment_specs = [
        (True, float(target_etf_pct), max(int(desired_etf_holdings), 0)),
        (False, float(target_non_etf_pct), max(int(desired_non_etf_holdings), 0)),
    ]
    for is_etf_value, segment_target_pct, desired_count in segment_specs:
        segment_idx = (
            out.loc[out["is_etf"].eq(is_etf_value)]
            .sort_values("value_eur", ascending=False, na_position="last")
            .index
        )
        if len(segment_idx) == 0:
            continue
        out.loc[segment_idx, "target_rank_in_segment"] = np.arange(1, len(segment_idx) + 1, dtype=float)
        if desired_count <= 0:
            continue
        kept_idx = list(segment_idx[:desired_count])
        if kept_idx:
            out.loc[kept_idx, "target_per_holding_pct"] = segment_target_pct / desired_count
    return out


def build_four_tables(
    *,
    holdings: pd.DataFrame,
    totals: TotalsResult,
    target_etf_pct: float,
    target_non_etf_pct: float,
    target_cash_pct: float,
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

    aggregated = apply_ranked_target_per_holding_pct(
        aggregated,
        target_etf_pct=float(target_etf_pct),
        target_non_etf_pct=float(target_non_etf_pct),
        desired_etf_holdings=int(desired_etf_holdings),
        desired_non_etf_holdings=int(desired_non_etf_holdings),
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
    target_etf_value = float(target_etf_pct) / 100.0 * combined_value
    target_non_etf_value = float(target_non_etf_pct) / 100.0 * combined_value
    target_cash_value = float(target_cash_pct) / 100.0 * combined_value
    deployable_cash = max(cash_value - target_cash_value, 0.0)
    etf_gap = max(target_etf_value - etf_value, 0.0)
    non_etf_gap = max(target_non_etf_value - non_etf_value, 0.0)
    gap_total = etf_gap + non_etf_gap
    if deployable_cash > 0.0:
        if gap_total > 0.0:
            eur_to_etf = deployable_cash * etf_gap / gap_total
            eur_to_non_etf = deployable_cash * non_etf_gap / gap_total
        else:
            invested_target_total = max(float(target_etf_pct) + float(target_non_etf_pct), 0.0)
            if invested_target_total > 0.0:
                eur_to_etf = deployable_cash * float(target_etf_pct) / invested_target_total
                eur_to_non_etf = deployable_cash * float(target_non_etf_pct) / invested_target_total
            else:
                eur_to_etf = 0.0
                eur_to_non_etf = 0.0
    else:
        eur_to_etf = 0.0
        eur_to_non_etf = 0.0

    cash_shortfall = max(target_cash_value - cash_value, 0.0)
    etf_excess = max(etf_value - target_etf_value, 0.0)
    non_etf_excess = max(non_etf_value - target_non_etf_value, 0.0)
    excess_total = etf_excess + non_etf_excess
    if cash_shortfall > 0.0:
        if excess_total > 0.0:
            etf_to_cash = cash_shortfall * etf_excess / excess_total
            non_etf_to_cash = cash_shortfall * non_etf_excess / excess_total
        else:
            invested_current_total = max(etf_value + non_etf_value, 0.0)
            if invested_current_total > 0.0:
                etf_to_cash = cash_shortfall * etf_value / invested_current_total
                non_etf_to_cash = cash_shortfall * non_etf_value / invested_current_total
            else:
                etf_to_cash = 0.0
                non_etf_to_cash = 0.0
    else:
        etf_to_cash = 0.0
        non_etf_to_cash = 0.0

    identity_total = etf_value + non_etf_value + cash_value

    summary = pd.DataFrame(
        [
            {"metric": "combined value", "value_eur": combined_value},
            {"metric": "non-ETF value", "value_eur": non_etf_value},
            {"metric": "ETF value", "value_eur": etf_value},
            {"metric": "cash position", "value_eur": cash_value},
            {"metric": "target non-ETF value", "value_eur": target_non_etf_value},
            {"metric": "target ETF value", "value_eur": target_etf_value},
            {"metric": "target cash position", "value_eur": target_cash_value},
            {"metric": "suggested EUR to ETF for next purchases", "value_eur": eur_to_etf},
            {
                "metric": "suggested EUR to non-ETF for next purchases",
                "value_eur": eur_to_non_etf,
            },
            {"metric": "suggested EUR from ETF to cash", "value_eur": etf_to_cash},
            {"metric": "suggested EUR from non-ETF to cash", "value_eur": non_etf_to_cash},
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
    df, _ = unify_holding_product_names(holdings)
    group_cols = ["instrument_id", "product", "isin", "ticker", "is_etf"]

    grouped = df.groupby(group_cols, dropna=False, as_index=False).agg(
        account_label=("account_label", _join_unique_non_empty),
        currency=("currency", _join_unique_non_empty),
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
